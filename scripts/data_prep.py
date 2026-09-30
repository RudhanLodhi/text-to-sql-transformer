import json
from pathlib import Path

DATA_DIR = Path("WikiSQL/data")
DATASET = Path("artifacts/dataset"); DATASET.mkdir(exist_ok=True, parents=True)
AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<"]
MAX_COLS = 64  # column tokens <c0> ... <c63>


def load_split(split):
    """return (examples, tables) for 'train', 'dev' or 'test'
    
    Args:
        split (str): one of 'train', 'dev' or 'test'

    Returns:
        tuple: (examples, tables) where examples is a list of dicts and tables is a dict mapping table_id to table dict
    """
    tables = {}

    with open(DATA_DIR / f"{split}.tables.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            tables[t["id"]] = t

    with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    return examples, tables


def encode_source(question, header):
    """question <sep> <c0> col name <c1> col name ...
    
    Args:
        question (str): the natural language question
        header (list): list of column names

    Returns:
        str: the encoded source
    """
    cols = " ".join(
        f"<c{i}> {name}" for i, name in enumerate(header)
    )
    return f"{question.strip()} <sep> {cols}".lower()


def encode_target(sql):
    """{'sel','agg','conds'} -> 'select count <c3> where <c1> = kim manners'
    
    Args:
        sql (dict): a dict with keys 'sel', 'agg', and 'conds'

    Returns:
        str: the encoded target
    """
    out = ["select"]

    if sql["agg"]:
        out.append(AGG_OPS[sql["agg"]].lower())

    out.append(f"<c{sql['sel']}>")

    for i, (col, op, val) in enumerate(sql["conds"]):
        out += [
            "where" if i == 0 else "and",
            f"<c{col}>",
            COND_OPS[op],
            str(val)
        ]

    return " ".join(out).lower()


def build_pairs(split):
    """build pairs of (source, target) for the given split

    Args:
        split (str): one of 'train', 'dev' or 'test'

    Returns:
        list: a list of dicts with keys 'table_id', 'src', and 'tgt'
    """
    examples, tables = load_split(split)
    pairs = []

    for ex in examples:
        header = tables[ex["table_id"]]["header"]

        pairs.append({
            "table_id": ex["table_id"],
            "src": encode_source(ex["question"], header),
            "tgt": encode_target(ex["sql"]),
        })

    return pairs


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        pairs = build_pairs(split)

        with open(
            DATASET / f"{split}_pairs.jsonl", "w", encoding="utf-8"
        ) as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        print(f"{split}: {len(pairs)} pairs")
        print(pairs[0]["src"])
        print(pairs[0]["tgt"])