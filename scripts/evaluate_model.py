import json
import subprocess
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sentencepiece as spm
import torch
from model.attention import MultiHeadAttention
from scripts.data_prep import encode_source, load_split
from scripts.decode import RESULTS, decode, load_model
from scripts.tokenizer import BOS_ID, EOS_ID, PAD_ID, TOKENIZER


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / RESULTS
FIGURES = RESULTS / "figures"
WIKISQL = ROOT / "WikiSQL"
OFFICIAL_EVALUATOR = WIKISQL / "evaluate.py"


def official_metrics(source_file, prediction_file, database_file):
    """run WikiSQL's official evaluator and return its JSON metrics
    """
    command = [
        sys.executable,
        str(OFFICIAL_EVALUATOR),
        str(source_file),
        str(database_file),
        str(prediction_file),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=WIKISQL,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(
            "WikiSQL evaluation failed. Install the project requirements "
            "before running this script:\n"
            "  python -m pip install -r requirements.txt\n\n"
            f"Command: {' '.join(command)}\n"
            f"Evaluator output:\n{details}"
        ) from error
    return json.loads(result.stdout)


def component_accuracy(source_file, prediction_file):
    """calculate select, aggregate, and unordered WHERE accuracy
    """
    correct = {"sel": 0, "agg": 0, "where": 0}
    total = 0

    with source_file.open(encoding="utf-8") as source, \
            prediction_file.open(encoding="utf-8") as prediction:
        for gold_line, predicted_line in zip(source, prediction):
            gold = json.loads(gold_line)["sql"]
            predicted = json.loads(predicted_line).get("query")
            total += 1

            if predicted is None:
                continue

            correct["sel"] += predicted.get("sel") == gold.get("sel")
            correct["agg"] += predicted.get("agg") == gold.get("agg")

            gold_conds = {
                tuple(condition) for condition in gold.get("conds", [])
            }
            predicted_conds = {
                tuple(condition) for condition in predicted.get("conds", [])
            }
            correct["where"] += predicted_conds == gold_conds

    return {
        "sel": 100 * correct["sel"] / total if total else 0.0,
        "agg": 100 * correct["agg"] / total if total else 0.0,
        "where": 100 * correct["where"] / total if total else 0.0,
    }


def parse_failure_rate(prediction_file):
    """return the percentage of prediction rows marked as parse errors
    """
    total = failures = 0
    with prediction_file.open(encoding="utf-8") as prediction:
        for line in prediction:
            total += 1
            failures += "error" in json.loads(line)
    return 100 * failures / total if total else 0.0


@torch.no_grad()
def attention_for_example(model, source_ids, generated_ids):
    """return last-layer decoder cross-attention averaged over heads
    """
    device = next(model.parameters()).device
    src = torch.tensor([source_ids], dtype=torch.long, device=device)
    tgt = torch.tensor([generated_ids], dtype=torch.long, device=device)
    src_mask = (src != PAD_ID).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_mask = torch.tril(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device)
    ).view(1, 1, tgt_len, tgt_len)

    memory = model.encode(src, src_mask)
    x = model.input_layer(tgt)

    for layer_index, layer in enumerate(model.decoder.layers):
        x = layer.add_norms[0](
            x,
            lambda value: layer.mmha(value, value, value, tgt_mask),
        )

        q = layer.cmha.w_q(x)
        k = layer.cmha.w_k(memory)
        v = layer.cmha.w_v(memory)
        q = q.view(1, tgt_len, layer.cmha.h, layer.cmha.d_k).transpose(1, 2)
        k = k.view(1, len(source_ids), layer.cmha.h, layer.cmha.d_k).transpose(1, 2)
        v = v.view(1, len(source_ids), layer.cmha.h, layer.cmha.d_k).transpose(1, 2)

        if layer_index == len(model.decoder.layers) - 1:
            _, weights = MultiHeadAttention.attention(
                q, k, v, src_mask, dropout=None
            )
            return weights.mean(dim=1)[0].cpu()

        x = layer.add_norms[1](
            x,
            lambda value: layer.cmha(value, memory, memory, src_mask),
        )
        x = layer.add_norms[2](x, layer.ffn)

    raise RuntimeError("decoder has no layers")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True, parents=True)
    FIGURES.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = spm.SentencePieceProcessor(
        model_file=f"{TOKENIZER}/sql_sp.model"
    )
    model = load_model(device, tokenizer)

    files = {
        "dev_greedy": RESULTS / "dev_greedy.jsonl",
        "dev_beam": RESULTS / "dev_beam.jsonl",
        "test_greedy": RESULTS / "test_greedy.jsonl",
        "test_beam": RESULTS / "test_beam.jsonl",
    }
    required_files = [*files.values(), OFFICIAL_EVALUATOR]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required evaluation files are missing:\n" + "\n".join(missing)
        )

    metrics = {}
    for name, split in (
        ("dev_greedy", "dev"),
        ("dev_beam", "dev"),
        ("test_greedy", "test"),
        ("test_beam", "test"),
    ):
        metrics[name] = official_metrics(
            WIKISQL / "data" / f"{split}.jsonl",
            files[name],
            WIKISQL / "data" / f"{split}.db",
        )

    final_name = (
        "dev_beam"
        if metrics["dev_beam"]["lf_accuracy"] >= metrics["dev_greedy"]["lf_accuracy"]
        else "dev_greedy"
    )
    final_test_name = "test_beam" if final_name == "dev_beam" else "test_greedy"

    with (RESULTS / "official_metrics.md").open("w", encoding="utf-8") as report:
        report.write("| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |\n")
        report.write("|---|---|---:|---:|---:|\n")
        for name, split, decoding in (
            ("dev_greedy", "Dev", "greedy"),
            ("dev_beam", "Dev", "beam (4)"),
            ("test_greedy", "Test", "greedy"),
            ("test_beam", "Test", "beam (4)"),
        ):
            values = metrics[name]
            parse_failures = parse_failure_rate(files[name])
            report.write(
                f"| {split} | {decoding} | "
                f"{values['lf_accuracy'] * 100:.2f} | "
                f"{values['ex_accuracy'] * 100:.2f} | "
                f"{parse_failures:.2f} |\n"
            )

    dev_components = component_accuracy(
        WIKISQL / "data" / "dev.jsonl",
        files["dev_beam"],
    )
    with (RESULTS / "component_accuracy.md").open("w", encoding="utf-8") as report:
        report.write("| Component | Accuracy (%) |\n")
        report.write("|---|---:|\n")
        report.write(f"| sel column correct | {dev_components['sel']:.2f} |\n")
        report.write(f"| agg correct | {dev_components['agg']:.2f} |\n")
        report.write(f"| WHERE clause correct | {dev_components['where']:.2f} |\n")

    examples, tables = load_split("dev")
    example = examples[0]
    table = tables[example["table_id"]]
    source_text = encode_source(example["question"], table["header"])
    source_ids = tokenizer.encode(source_text) + [EOS_ID]
    generated_ids = decode(model, torch.tensor([source_ids], device=device), beam_size=4)
    generated_ids = [
        token for token in generated_ids if token not in {BOS_ID, PAD_ID}
    ]
    attention = attention_for_example(model, source_ids, generated_ids)

    figure, axis = plt.subplots(
        figsize=(max(10, len(source_ids) / 2), max(5, len(generated_ids) / 2))
    )
    image = axis.imshow(attention.numpy(), aspect="auto", cmap="viridis")
    axis.set_title("Cross-attention map for one dev example")
    axis.set_xlabel("Source tokens")
    axis.set_ylabel("Generated tokens")
    axis.set_xticks(range(len(source_ids)))
    axis.set_xticklabels(
        [tokenizer.id_to_piece(token) for token in source_ids],
        rotation=90,
        fontsize=7,
    )
    axis.set_yticks(range(len(generated_ids)))
    axis.set_yticklabels(
        [tokenizer.id_to_piece(token) for token in generated_ids],
        fontsize=7,
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(FIGURES / "cross_attention_map.png", dpi=150)
    plt.close(figure)

    print(f"Final test decoding: {final_test_name}")
    print("Wrote results/official_metrics.md")
    print("Wrote results/component_accuracy.md")
    print("Wrote results/figures/cross_attention_map.png")
