import sentencepiece as spm
from model.embeddings import TokenEmbedding, InputLayer
from scripts.dataset import make_loader
from scripts.data_prep import DATASET
from scripts.tokenizer import PAD_ID, TOKENIZER


sp = spm.SentencePieceProcessor(model_file=f"{TOKENIZER}/sql_sp.model")

train_dl = make_loader(f"{DATASET}/train_pairs.jsonl", sp, train=True)
dev_dl = make_loader(f"{DATASET}/dev_pairs.jsonl", sp, train=False)


src, tgt = next(iter(train_dl))

print("src", tuple(src.shape), "tgt", tuple(tgt.shape))


d_model = 256

shared = TokenEmbedding(
    sp.get_piece_size(),
    d_model,
    PAD_ID
)

enc_in = InputLayer(shared, d_model)  # encoder input
dec_in = InputLayer(shared, d_model)  # decoder input (same weights)


x = enc_in(src)  # (batch, s_seq_len, d_model) -> goes into YOUR encoder
y = dec_in(tgt[:, :-1])  # (batch, t_seq_len - 1, d_model) -> goes into YOUR decoder

print(
    "encoder input",
    tuple(x.shape),
    "decoder input",
    tuple(y.shape)
)