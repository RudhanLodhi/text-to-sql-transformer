import json
from pathlib import Path

import torch
import sentencepiece as spm

from model.transformer import build_transformer
from scripts.data_prep import AGG_OPS, COND_OPS, DATASET
from scripts.dataset import make_loader
from scripts.tokenizer import BOS_ID, EOS_ID, TOKENIZER

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True, parents=True)

def generate_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask


def greedy_decode(model, src, src_mask, max_len=64):
    model.eval()
    with torch.no_grad():
        memory = model.encoder(src, src_mask)
        tgt = torch.tensor([[BOS_ID]], device=src.device)

        for _ in range(max_len):
            tgt_mask = generate_mask(tgt.size(1)).to(src.device)
            out = model.decoder(tgt, memory, src_mask, tgt_mask)
            logits = model.projection(out)[:, -1, :]

            next_token = torch.argmax(logits, dim=-1)
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == EOS_ID:
                break

        return tgt.squeeze(0).tolist()


def beam_search(model, src, src_mask, beam_size=4, max_len=64):
    model.eval()
    with torch.no_grad():
        memory = model.encoder(src, src_mask)
        beams = [(0.0, [BOS_ID])]

        for _ in range(max_len):
            all_candidates = []

            for score, seq in beams:
                if seq[-1] == EOS_ID:
                    all_candidates.append((score, seq))
                    continue

                tgt_tensor = torch.tensor([seq], device=src.device)
                tgt_mask = generate_mask(tgt_tensor.size(1)).to(src.device)

                out = model.decoder(tgt_tensor, memory, src_mask, tgt_mask)
                logits = model.projection(out)[:, -1, :]
                probs = logits.squeeze()

                top_probs, top_idx = torch.topk(probs, beam_size)

                for i in range(beam_size):
                    new_score = score + top_probs[i].item()
                    new_seq = seq + [top_idx[i].item()]
                    all_candidates.append((new_score, new_seq))

            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            beams = ordered[:beam_size]

            if all(seq[-1] == EOS_ID for _, seq in beams):
                break

        return beams[0][1]


def parse_prediction(decoded_str):
    try:
        decoded_str = decoded_str.replace("select ", "").strip()
        agg_idx = 0

        for i, agg in enumerate(AGG_OPS):
            if agg != "" and decoded_str.startswith(agg.lower() + " "):
                agg_idx = i
                decoded_str = decoded_str[len(agg):].strip()
                break

        sel_str = decoded_str.split(" ")[0]
        sel_idx = int(sel_str.replace("<c", "").replace(">", ""))

        conds = []
        if " where " in decoded_str:
            where_part = decoded_str.split(" where ")[1]
            conditions = where_part.split(" and ")

            for cond in conditions:
                parts = cond.split(" ", 2)
                col_idx = int(parts[0].replace("<c", "").replace(">", ""))
                op_idx = COND_OPS.index(parts[1])
                val = parts[2]
                conds.append([col_idx, op_idx, val])

        return {"sel": sel_idx, "agg": agg_idx, "conds": conds}

    except Exception:
        return None


def export_predictions(model, dataloader, sp, output_path, use_beam=False):
    """Write one JSON line per example."""
    with open(output_path, "w", encoding="utf-8") as f:
        for batch in dataloader:
            src, src_mask = batch

            if use_beam:
                predicted_ids = beam_search(model, src, src_mask)
            else:
                predicted_ids = greedy_decode(model, src, src_mask)

            clean_ids = [idx for idx in predicted_ids if idx not in (BOS_ID, EOS_ID)]
            decoded_str = sp.decode(clean_ids)
            parsed_dict = parse_prediction(decoded_str)

            if parsed_dict is None:
                json_line = {"error": "parse"}
            else:
                json_line = {"query": parsed_dict}

            f.write(json.dumps(json_line) + "\n")


def to_readable_sql(parsed_query, column_names):
    if "error" in parsed_query:
        return "ERROR: Could not parse model output into SQL."

    q = parsed_query["query"]
    sel_col_name = column_names[q["sel"]]
    agg_op = AGG_OPS[q["agg"]]

    if agg_op == "":
        sql = f"SELECT {sel_col_name}"
    else:
        sql = f"SELECT {agg_op}({sel_col_name})"

    if len(q["conds"]) > 0:
        sql += " WHERE "
        cond_strings = []
        for col_idx, op_idx, val in q["conds"]:
            col_name = column_names[col_idx]
            op_symbol = COND_OPS[op_idx]
            cond_strings.append(f"{col_name} {op_symbol} '{val}'")

        sql += " AND ".join(cond_strings)

    return sql


def load_model(device):
    model_path = Path("artifacts/checkpoints/best.pt")
    sp = spm.SentencePieceProcessor(model_file=f"{TOKENIZER}/sql_sp.model")
    model = build_transformer(
        vocab_size=sp.get_piece_size(),
        max_len=512,
        d_model=256,
        h=4,
        N=3,
        d_ff=1024,
        dropout=0.1,
        device=device,
    )

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint)

    model.eval()
    return model, sp


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = RESULTS

    tokenizer_path = Path(f"{TOKENIZER}/sql_sp.model")
    if not tokenizer_path.exists():
        status = {
            "status": "missing tokenizer",
            "message": "Run scripts/tokenizer.py to generate artifacts/tokenizer/sql_sp.model before decoding.",
            "output_dir": str(results_dir),
        }
        output_path = results_dir / "decode_status.json"
        output_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"Tokenizer not found. Wrote status to {output_path}")
        return

    test_path = DATASET / "test_pairs.jsonl"
    model, sp = load_model(device)

    if test_path.exists():
        dataloader = make_loader(str(test_path), sp, train=False, batch_size=1)
        output_path = results_dir / "decode_predictions.jsonl"
        export_predictions(model, dataloader, sp, str(output_path))
        print(f"Saved predictions to {output_path}")
        return

    sample_question = "what is the max age"
    sample_source = f"{sample_question.strip()} <sep> <c0> name <c1> age <c2> city".lower()
    sample_ids = sp.encode(sample_source)
    src = torch.tensor([sample_ids], dtype=torch.long, device=device)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(1)

    prediction_ids = greedy_decode(model, src, src_mask)
    clean_ids = [idx for idx in prediction_ids if idx not in (BOS_ID, EOS_ID)]
    prediction = sp.decode(clean_ids)

    output = {
        "source": sample_source,
        "prediction": prediction,
    }
    output_path = results_dir / "decode_sample.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved sample decode output to {output_path}")


if __name__ == "__main__":
    main()
