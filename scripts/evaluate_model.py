import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from scripts.data_prep import DATASET
from scripts.dataset import make_loader
from scripts.decode import RESULTS, beam_search, greedy_decode, load_model, parse_prediction
from scripts.tokenizer import BOS_ID, EOS_ID, PAD_ID
from model.attention import MultiHeadAttention


def run_evaluator(source_file: Path, pred_file: Path, db_file: Path, wikisql_dir: Path):
    evaluator = wikisql_dir / "evaluate.py"
    command = [
        sys.executable,
        str(evaluator),
        str(source_file),
        str(db_file),
        str(pred_file),
    ]
    result = subprocess.run(command, cwd=str(wikisql_dir), capture_output=True, text=True, check=True)
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())


def write_predictions(split: str, mode: str, model, sp, device: str, output_dir: Path):
    dataset_path = DATASET / f"{split}_pairs.jsonl"
    loader = make_loader(str(dataset_path), sp, train=False, batch_size=1)
    predictions = []

    for batch in loader:
        src, _ = batch
        src = src.to(device)
        src_mask = (src != PAD_ID).unsqueeze(1).unsqueeze(1)

        if mode == "beam":
            ids = beam_search(model, src, src_mask, beam_size=4)
        else:
            ids = greedy_decode(model, src, src_mask)

        clean = [idx for idx in ids if idx not in (BOS_ID, EOS_ID)]
        decoded = sp.decode(clean)
        parsed = parse_prediction(decoded)
        predictions.append({"query": parsed} if parsed is not None else {"error": "parse"})

    output_path = output_dir / f"{split}_{mode}_predictions.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item) + "\n")

    return output_path


def normalize_conds(conds):
    normalized = []
    for cond in conds or []:
        if len(cond) == 3:
            normalized.append((str(cond[0]), str(cond[1]), str(cond[2])))
    return set(normalized)


def compute_component_accuracy(source_file: Path, pred_file: Path):
    sel_correct = 0
    agg_correct = 0
    where_correct = 0
    total = 0

    with source_file.open("r", encoding="utf-8") as src_f, pred_file.open("r", encoding="utf-8") as pred_f:
        for src_line, pred_line in zip(src_f, pred_f):
            gold = json.loads(src_line)
            pred = json.loads(pred_line)
            total += 1

            gold_sql = gold["sql"]
            pred_query = pred.get("query")

            if pred_query is None:
                continue

            sel_correct += int(pred_query.get("sel") == gold_sql.get("sel"))
            agg_correct += int(pred_query.get("agg") == gold_sql.get("agg"))
            where_correct += int(normalize_conds(pred_query.get("conds", [])) == normalize_conds(gold_sql.get("conds", [])))

    return {
        "total": total,
        "sel_accuracy": sel_correct / total if total else 0.0,
        "agg_accuracy": agg_correct / total if total else 0.0,
        "where_accuracy": where_correct / total if total else 0.0,
    }


def compute_cross_attention(model, src, src_mask, generated_ids):
    model.eval()
    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)
        tgt = torch.tensor([generated_ids], device=src.device)
        tgt_mask = (tgt != PAD_ID).unsqueeze(1).unsqueeze(1) & torch.tril(
            torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device)
        ).view(1, 1, tgt.size(1), tgt.size(1))

        x = model.input_layer(tgt)
        for layer in model.decoder.layers:
            x = layer.add_norms[0](x, lambda z: layer.mmha(z, z, z, tgt_mask))
            if layer is model.decoder.layers[-1]:
                q = layer.cmha.w_q(x)
                k = layer.cmha.w_k(enc_output)
                v = layer.cmha.w_v(enc_output)

                q = q.view(q.size(0), q.size(1), layer.cmha.h, layer.cmha.d_k)
                k = k.view(k.size(0), k.size(1), layer.cmha.h, layer.cmha.d_k)
                v = v.view(v.size(0), v.size(1), layer.cmha.h, layer.cmha.d_k)

                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                _, weights = MultiHeadAttention.attention(q, k, v, src_mask, None)
                return weights.mean(dim=1)[0].detach().cpu().numpy()

            x = layer.add_norms[1](x, lambda z: layer.cmha(z, enc_output, enc_output, src_mask))
            x = layer.add_norms[2](x, layer.ffn)

    return None


def save_attention_map(model, sp, device, split: str, example_index: int = 0, output_dir: Path = RESULTS):
    dataset_path = DATASET / f"{split}_pairs.jsonl"
    pairs = [json.loads(line) for line in dataset_path.open("r", encoding="utf-8")]
    example = pairs[example_index]
    src_ids = sp.encode(example["src"])
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src != PAD_ID).unsqueeze(1).unsqueeze(1)

    generated = greedy_decode(model, src, src_mask)
    clean = [idx for idx in generated if idx not in (BOS_ID, EOS_ID)]
    prediction = sp.decode(clean)
    attention = compute_cross_attention(model, src, src_mask, clean)

    fig, ax = plt.subplots(figsize=(max(8, len(src_ids) / 3), max(4, len(clean) / 2)))
    image = ax.imshow(attention, aspect="auto", cmap="viridis")
    ax.set_title("Last decoder cross-attention")
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Generated tokens")
    ax.set_xticks(range(min(20, len(src_ids))))
    ax.set_yticks(range(min(20, len(clean))))
    ax.set_xticklabels(
        [sp.id_to_piece(token) for token in src_ids[:20]],
        rotation=90,
        fontsize=8,
    )
    ax.set_yticklabels(
        [sp.id_to_piece(token) for token in clean[:20]],
        fontsize=8,
    )
    fig.colorbar(image, ax=ax, pad=0.02)
    fig.tight_layout()
    output_path = output_dir / "attention_map.png"
    fig.savefig(str(output_path))
    plt.close(fig)

    return {
        "source": example["src"],
        "prediction": prediction,
        "attention_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the WikiSQL text-to-SQL model")
    parser.add_argument("--example-index", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = RESULTS
    results_dir.mkdir(exist_ok=True, parents=True)
    wikisql_dir = repo_root / "WikiSQL"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sp = load_model(device)

    dev_greedy_pred = write_predictions("dev", "greedy", model, sp, device, results_dir)
    dev_beam_pred = write_predictions("dev", "beam", model, sp, device, results_dir)

    print("\nOfficial evaluator on dev (greedy):")
    run_evaluator(
        wikisql_dir / "data" / "dev.jsonl",
        dev_greedy_pred,
        wikisql_dir / "data" / "dev.db",
        wikisql_dir,
    )

    print("\nOfficial evaluator on dev (beam):")
    run_evaluator(
        wikisql_dir / "data" / "dev.jsonl",
        dev_beam_pred,
        wikisql_dir / "data" / "dev.db",
        wikisql_dir,
    )

    dev_component = compute_component_accuracy(wikisql_dir / "data" / "dev.jsonl", dev_greedy_pred)
    dev_component_beam = compute_component_accuracy(wikisql_dir / "data" / "dev.jsonl", dev_beam_pred)
    print("\nComponent accuracy on dev (greedy):", dev_component)
    print("Component accuracy on dev (beam):", dev_component_beam)

    best_mode = "beam" if dev_component_beam["where_accuracy"] >= dev_component["where_accuracy"] else "greedy"
    print(f"\nSelected final decoding choice for test: {best_mode}")

    test_pred = write_predictions("test", best_mode, model, sp, device, results_dir)
    print("\nOfficial evaluator on test:")
    run_evaluator(
        wikisql_dir / "data" / "test.jsonl",
        test_pred,
        wikisql_dir / "data" / "test.db",
        wikisql_dir,
    )

    attention_summary = save_attention_map(model, sp, device, "dev", args.example_index, results_dir)
    print("\nAttention map saved at:", attention_summary["attention_path"])
    print("Example text:", attention_summary["source"])
    print("Prediction:", attention_summary["prediction"])


if __name__ == "__main__":
    main()
