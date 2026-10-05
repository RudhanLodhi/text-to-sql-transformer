import json
import re
from pathlib import Path
import sentencepiece as spm
import torch
from model.transformer import build_transformer
from scripts.data_prep import AGG_OPS, COND_OPS, DATASET, encode_source, load_split
from scripts.tokenizer import BOS_ID, EOS_ID, PAD_ID, TOKENIZER


RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True, parents=True)
CHECKPOINT = Path("artifacts/checkpoints/best.pt")
MAX_DECODE_LEN = 64
COLUMN_TOKEN = re.compile(r"^<c(\d+)>$")


def _mask(size, device):
    return torch.tril(
        torch.ones(size, size, dtype=torch.bool, device=device)
    ).unsqueeze(0).unsqueeze(0)


def _length_norm_score(item, alpha=0.7):
    tokens, score = item[0], item[1]
    lp = ((5 + len(tokens)) / 6) ** alpha
    return score / lp


@torch.no_grad()
def decode(model, src, beam_size=1, max_len=MAX_DECODE_LEN):
    """src (1, s_seq_len)
    """
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")

    model.eval()
    device = src.device
    src_mask = (src != PAD_ID).unsqueeze(1).unsqueeze(1) # (1, 1, 1, s_seq_len)
    memory = model.encode(src, src_mask) # (1, s_seq_len, d_model)
    beams = [(torch.tensor([[BOS_ID]], device=device), 0.0)] # (1, t_seq_len=1)

    for _ in range(max_len - 1):
        candidates = []

        for sequence, score in beams:
            if sequence[0, -1].item() == EOS_ID:
                candidates.append((sequence, score))
                continue

            tgt_mask = _mask(sequence.size(1), device) # (1, 1, t_seq_len, t_seq_len)
            decoder_output = model.decode(
                sequence,
                memory,
                src_mask,
                tgt_mask,
            ) # (1, t_seq_len, d_model)
            log_probs = model.projection(
                decoder_output[:, -1:, :] # (1, 1, d_model)
            ).squeeze(1) # (1, vocab_size)
            token_scores, token_ids = torch.topk(log_probs, beam_size, dim=-1)

            for token_score, token_id in zip(token_scores[0], token_ids[0]):
                candidates.append((
                    torch.cat((sequence, token_id.view(1, 1)), dim=1),
                    score + token_score.item(),
                ))

        beams = sorted(
            candidates,
            key=_length_norm_score,
            reverse=True,
        )[:beam_size]

        if all(sequence[0, -1].item() == EOS_ID for sequence, _ in beams):
            break

    return max(
        beams,
        key=_length_norm_score,
    )[0][0].tolist()


def greedy_decode(model, src, max_len=MAX_DECODE_LEN):
    return decode(model, src, beam_size=1, max_len=max_len)


def beam_search(model, src, beam_size=4, max_len=MAX_DECODE_LEN):
    return decode(model, src, beam_size=beam_size, max_len=max_len)


def parse_prediction(text):
    tokens = text.lower().split()
    if not tokens or tokens[0] != "select":
        raise ValueError("prediction must start with select")

    index = 1
    agg = 0
    if index < len(tokens) and tokens[index].upper() in AGG_OPS[1:]:
        agg = AGG_OPS.index(tokens[index].upper())
        index += 1

    if index >= len(tokens):
        raise ValueError("missing selected column")
    selected = COLUMN_TOKEN.fullmatch(tokens[index])
    if selected is None:
        raise ValueError("invalid selected column")
    query = {"sel": int(selected.group(1)), "agg": agg, "conds": []}
    index += 1

    while index < len(tokens):
        if tokens[index] not in {"where", "and"}:
            raise ValueError("expected where or and")
        index += 1

        if index + 2 >= len(tokens):
            raise ValueError("incomplete condition")
        column = COLUMN_TOKEN.fullmatch(tokens[index])
        if column is None:
            raise ValueError("invalid condition column")
        index += 1

        if tokens[index] not in COND_OPS:
            raise ValueError("invalid condition operator")
        operator = COND_OPS.index(tokens[index])
        index += 1

        value_start = index
        while index < len(tokens) and tokens[index] not in {"where", "and"}:
            index += 1
        if value_start == index:
            raise ValueError("empty condition value")

        query["conds"].append([
            int(column.group(1)),
            operator,
            " ".join(tokens[value_start:index]),
        ])

    return query


def to_readable_sql(parsed_query, column_names):
    query = parsed_query.get("query", parsed_query)
    selected = column_names[query["sel"]]
    aggregate = AGG_OPS[query["agg"]]
    select_clause = f"{aggregate}({selected})" if aggregate else selected
    sql = f"SELECT {select_clause} FROM table"

    if query["conds"]:
        conditions = [
            f"{column_names[column]} {COND_OPS[operator]} "
            f"'{str(value).replace(chr(39), chr(39) * 2)}'"
            for column, operator, value in query["conds"]
        ]
        sql += " WHERE " + " AND ".join(conditions)

    return sql


def load_model(device, tokenizer):
    model = build_transformer(
        vocab_size=tokenizer.get_piece_size(),
        max_len=512,
        d_model=256,
        h=4,
        N=3,
        d_ff=1024,
        dropout=0.1,
        device=device,
    )
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = spm.SentencePieceProcessor(
        model_file=f"{TOKENIZER}/sql_sp.model"
    )
    model = load_model(device, tokenizer)

    for split in ("dev", "test"):
        examples, tables = load_split(split)

        for name, beam_size in (
            ("greedy", 1),
            ("beam", 4),
        ):
            output_path = RESULTS / f"{split}_{name}.jsonl"

            with output_path.open("w", encoding="utf-8") as output_file:
                for example in examples:
                    header = tables[example["table_id"]]["header"]
                    source = encode_source(example["question"], header)
                    source_ids = tokenizer.encode(source) + [EOS_ID]
                    src = torch.tensor(
                        [source_ids],
                        dtype=torch.long,
                        device=device,
                    )

                    try:
                        predicted_ids = decode(
                            model,
                            src,
                            beam_size=beam_size,
                        )
                        predicted_ids = [
                            token for token in predicted_ids
                            if token not in {BOS_ID, EOS_ID, PAD_ID}
                        ]
                        prediction = tokenizer.decode(predicted_ids)
                        output = {"query": parse_prediction(prediction)}
                    except (IndexError, KeyError, ValueError, RuntimeError):
                        output = {"error": "parse"}

                    output_file.write(
                        json.dumps(output, ensure_ascii=False) + "\n"
                    )

            print(f"Saved predictions to {output_path}")