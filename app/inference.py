from pathlib import Path
import sentencepiece as spm
import torch
from model.transformer import build_transformer
from scripts.data_prep import encode_source
from scripts.decode import (
    decode,
    parse_prediction,
    to_readable_sql,
)
from scripts.tokenizer import BOS_ID, EOS_ID, TOKENIZER

class TextToSQL:
    def __init__(self, device=None):
        self.root = Path(__file__).resolve().parents[1]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_path = self.root / TOKENIZER / "sql_sp.model"
        checkpoint_path = self.root / "artifacts" / "checkpoints" / "best.pt"

        self.sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self.model = build_transformer(
            vocab_size=self.sp.get_piece_size(),
            max_len=512,
            d_model=256,
            h=4,
            N=3,
            d_ff=1024,
            dropout=0.1,
            device=self.device,
        )
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.eval()

    def _source(self, question, columns):
        return encode_source(question, columns)

    def _predict_mode(self, question, columns, beam_size):
        source = self._source(question, columns)
        source_ids = self.sp.encode(source) + [EOS_ID]
        src = torch.tensor([source_ids], dtype=torch.long, device=self.device)
        output_ids = decode(self.model, src, beam_size=beam_size)

        clean_ids = [
            token for token in output_ids if token not in (BOS_ID, EOS_ID)
        ]
        decoded = self.sp.decode(clean_ids)
        try:
            query = parse_prediction(decoded)
        except ValueError:
            query = None

        return {
            "decoded": decoded,
            "query": query,
            "sql": to_readable_sql({"query": query}, columns) if query else None,
        }

    def predict(self, question, columns, mode="greedy", beam_size=4):
        """Return both greedy and fixed beam-4 predictions for the web UI."""
        del mode, beam_size
        return {
            "question": question,
            "columns": columns,
            "greedy": self._predict_mode(question, columns, beam_size=1),
            "beam": self._predict_mode(question, columns, beam_size=4),
        }