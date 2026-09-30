import json
from pathlib import Path
import sentencepiece as spm
from scripts.data_prep import MAX_COLS, DATASET

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL = ["<sep>"] + [f"<c{i}>" for i in range(MAX_COLS)]
TOKENIZER = Path("artifacts/tokenizer"); TOKENIZER.mkdir(exist_ok=True, parents=True)


def read_pairs(path):
    """read pairs of (source, target) from a jsonl file
    
    Args:
        path (str): path to the jsonl file

    Returns:
        list: a list of dicts with keys 'table_id', 'src', and 'tgt'
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def train_tokenizer(train_path="train_pairs.jsonl", vocab_size=8000):
    """train a sentencepiece tokenizer on the source and target of the training pairs combined
    
    Args:
        train_path (str, optional): path to the training pairs jsonl file. Defaults to "train_pairs.jsonl".
        vocab_size (int, optional): size of the vocabulary. Defaults to 8000.

    Returns:
        spm.SentencePieceProcessor: the trained tokenizer
    """
    pairs = read_pairs(train_path)

    with open(TOKENIZER / "spm_corpus.txt", "w", encoding="utf-8") as f:
        for p in pairs:  # one shared vocabulary for source and target, as in the original transformer
            f.write(p["src"] + "\n")
            f.write(p["tgt"] + "\n")

    spm.SentencePieceTrainer.train(
        input=str(TOKENIZER / "spm_corpus.txt"),
        model_prefix=f"{TOKENIZER}/sql_sp",
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        user_defined_symbols=SPECIAL, # never split <sep>, <c0>, ...
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )

    return spm.SentencePieceProcessor(model_file=f"{TOKENIZER}/sql_sp.model")


if __name__ == "__main__":
    sp = train_tokenizer(f"{DATASET}/train_pairs.jsonl")

    p = read_pairs(f"{DATASET}/dev_pairs.jsonl")[0]

    print(sp.encode(p["src"], out_type=str))
    print(sp.encode(p["tgt"], out_type=str))
    print(sp.decode(sp.encode(p["tgt"])) == p["tgt"])