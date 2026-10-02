import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """multi-head attention mechanism
    """

    def __init__(self, d_model: int, h: int, dropout: float=0.1):
        """initializer of multi-head attention mechanism

        Args:
            d_model (int): model hidden dimension
            h (int): number of heads
            dropout (float, optional): dropout to be applied on attention scores. Defaults to 0.1.
        """
        pass

    def forward(self, query, key, value, mask):
        """forward pass for the multi-head attention mechanism

        Args:
            query (tensor): query tensor of shape (batch, seq_len, d_model)
            key (tensor): key tensor of shape (batch, seq_len, d_model)
            value (tensor): value tensor of shape (batch, seq_len, d_model)
            mask (tensor): mask tensor of shape (___)

        Returns:
            tensor: output tensor of shape (batch, seq_len, d_model)
        """
        pass

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout=None):
        """applies multi-head attention to the input tensors.

        Args:
            query (tensor): query tensor of shape (batch, h, seq_len, d_k)
            key (tensor): key tensor of shape (batch, h, seq_len, d_k)
            value (tensor): value tensor of shape (batch, h, seq_len, d_k)
            mask (tensor): mask tensor of shape (___)
            dropout (tensor, optional): dropout layer to be applied on attention scores. Defaults to None.

        Returns:
            tensor: output tensor of shape (batch, h, seq_len, d_k) and attention scores of shape (batch, h, seq_len, seq_len)
        """
        pass