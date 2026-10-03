import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    embeddings are multiplied by sqrt(d_model)
    one instance is shared by the encoder, the decoder and (optionally) the final output projection
    """

    def __init__(self, vocab_size: int, d_model: int, pad_id: int=0):
        """initializer for the learnable embeddings in the transformer

        Args:
            vocab_size (int): size of our vocabulary
            d_model (int): model embedding dimention
            pad_id (int, optional): padding index in our vocabulary. Defaults to 0.
        """
        super().__init__()

        self.emb = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id
        )
        self.scale = math.sqrt(d_model)

    def forward(self, ids):
        """forward pass of to get embeddings from token ids

        Args:
            ids (tensor): input ids of shape (batch, seq_len)

        Returns:
            tensor: output embeddings of shape (batch, seq_len, d_model)
        """
        return self.emb(ids) * self.scale


class PositionalEncoding(nn.Module):
    """
    fixed sinusoidal encodings
    pe(pos, 2i) = sin(pos / 10000^(2i/d_model))
    pe(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int=512, dropout: float=0.1):
        """initializer of the fixed sinosoidal positional encoding

        Args:
            d_model (int): model embedding dimention
            max_len (int, optional): maximum length of the sequence allowed. Defaults to 512.
            dropout (float, optional): dropout to be applied on summation of emb + pos. Defaults to 0.1.
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pos = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model) # (seq_len, d_model)

        # pos * div (max_len, d_model/2)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )  # (1, seq_len, d_model)

    def forward(self, x):
        """forward pass that adds token embedding with sinosoidal encoding and then apply dropout

        Args:
            x (tensor): input embeddings of shape (batch, seq_len, d_model)

        Returns:
            tensor: final output of shape (batch, seq_len, d_model)
        """
        pos_encoding = self.pe[:, :x.size(1), :]
        if pos_encoding.device != x.device:
            pos_encoding = pos_encoding.to(x.device)
        return self.dropout(x + pos_encoding)


class InputLayer(nn.Module):
    """
    token ids -> scaled embedding + positional encoding -> dropout
    """

    def __init__(self, token_emb: TokenEmbedding, d_model: int, max_len: int=512, dropout: float=0.1):
        """initializer for input layer that combines both embedding and position

        Args:
            token_emb (TokenEmbedding): trained embedding to be applied on ids
            d_model (int): model embedding dimention
            max_len (int, optional): maximum sequence length allowed. Defaults to 512.
            dropout (float, optional): dropout to be applied to sum of emb + pos. Defaults to 0.1.
        """
        super().__init__()

        self.tok = token_emb
        self.pos = PositionalEncoding(
            d_model,
            max_len,
            dropout
        )

    def forward(self, ids):
        """forward pass of input layer

        Args:
            ids (tensor): input of tokens of shape (batch, seq_len)

        Returns:
            tensor: final output of shape (batch, seq_len, d_model)
        """
        return self.pos(self.tok(ids))