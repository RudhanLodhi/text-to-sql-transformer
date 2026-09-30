import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from scripts.tokenizer import read_pairs, PAD_ID, BOS_ID, EOS_ID

class SQLDataset(Dataset):
    """each item: (source ids, target ids)
    target = <s> ... </s>
    train=True drops over-long pairs
    train=False keeps every row in file order (needed for evaluation)
    """

    def __init__(self, path: str, sp: spm.SentencePieceProcessor, train: bool=True, max_src: int=160, max_tgt: int=64):
        """initializer for SQLDataset that reads pairs of (source, target) from a jsonl file and encodes them using the provided tokenizer
        
        Args:
            path (str): path to the jsonl file
            sp (spm.SentencePieceProcessor): the trained tokenizer
            train (bool, optional): whether to train the model. Defaults to True.
            max_src (int, optional): maximum length of the source sequence. Defaults to 160.
            max_tgt (int, optional): maximum length of the target sequence. Defaults to 64.
        """
        self.items, skipped = [], 0

        for p in read_pairs(path):
            src = sp.encode(p["src"]) + [EOS_ID]
            tgt = [BOS_ID] + sp.encode(p["tgt"]) + [EOS_ID]

            if train and (len(src) > max_src or len(tgt) > max_tgt):
                skipped += 1
                continue

            self.items.append((src, tgt))

        print(f"{path}: kept {len(self.items)}, skipped {skipped}")

    def __len__(self):
        """return the number of items in the dataset
        """
        return len(self.items)

    def __getitem__(self, i):
        """return the i-th item in the dataset

        Args:
            i (int): index of the item to return

        Returns:
            tuple: a tuple of (source ids, target ids)
        """
        return self.items[i]


def collate(batch):
    """collate function to pad the source and target sequences in a batch to the same length

    Args:
        batch (list): a list of tuples of (source ids, target ids)

    Returns:
        tuple: a tuple of (padded source ids, padded target ids)
    """
    srcs, tgts = zip(*batch)

    S, T = max(map(len, srcs)), max(map(len, tgts))

    src = torch.full(
        (len(batch), S),
        PAD_ID,
        dtype=torch.long
    )

    tgt = torch.full(
        (len(batch), T),
        PAD_ID,
        dtype=torch.long
    )

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, :len(s)] = torch.tensor(s)
        tgt[i, :len(t)] = torch.tensor(t)

    return src, tgt # (batch, s_seq_len) and (batch, t_seq_len)


def make_loader(path: str, sp: spm.SentencePieceProcessor, train: bool=True, batch_size: int=64):
    """make a DataLoader for the given path and tokenizer

    Args:
        path (str): path to the jsonl file
        sp (spm.SentencePieceProcessor): the trained tokenizer
        train (bool, optional): whether to train the model. Defaults to True.
        batch_size (int, optional): batch size. Defaults to 64.
    
    Returns:
        DataLoader: a DataLoader for the given path and tokenizer
    """
    return DataLoader(
        SQLDataset(path, sp, train=train),
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collate
    )