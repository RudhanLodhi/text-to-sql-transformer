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
        super().__init__()
        
        self.d_model = d_model # embedding dim
        self.h = h # no. heads
        self.d_k = d_model // h # dim of each head
        
        self.dropout = nn.Dropout(dropout)
        
        self.w_q = nn.Linear(d_model, d_model) # query
        self.w_k = nn.Linear(d_model, d_model) # key
        self.w_v = nn.Linear(d_model, d_model) # value
        
        self.w_o = nn.Linear(d_model, d_model) # output

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
        Q = self.w_q(query) # (batch, seq_len, d_model) @ (d_model, d_model) -> (batch, seq_len, d_model)
        K = self.w_k(key) # //
        V = self.w_v(value) # //
        
        q = Q.view(Q.size(0), Q.size(1), self.h, self.d_k) # (batch , seq_len, h, d_k)
        k = K.view(K.size(0), K.size(1), self.h, self.d_k) # //
        v = V.view(V.size(0), V.size(1), self.h, self.d_k) # //
        
        q = q.transpose(1, 2) # (batch, h, seq_len, d_k)
        k = k.transpose(1, 2) # //
        v = v.transpose(1, 2) # //
        
        score, _ = MultiHeadAttention.attention(q, k, v, mask, self.dropout) # (batch, h, seq_len, d_k)
        score = score.transpose(1, 2).contiguous().view(score.size(0), -1, self.d_model) # (batch, seq_len, d_model)
        
        return self.w_o(score) # (batch, seq_len, d_model) @ (d_model, d_model) -> (batch, seq_len, d_model)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        """applies multi-head attention to the input tensors.
        
        Args:
            q (tensor): query tensor of shape (batch, h, seq_len, d_k)
            k (tensor): key tensor of shape (batch, h, seq_len, d_k)
            v (tensor): value tensor of shape (batch, h, seq_len, d_k)
            mask (tensor): mask tensor of shape (___)
            dropout (tensor, optional): dropout layer to be applied on attention scores. Defaults to None.

        Returns:
            tensor: output tensor of shape (batch, h, seq_len, d_k) and attention scores of shape (batch, h, seq_len, seq_len)
        """
        d_k = q.size(-1)

        att = q @ k.transpose(-2, -1) / math.sqrt(d_k) # (batch, h ,seq_len, d_k) @ (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)

        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        
        score = torch.softmax(att, dim=-1)

        if dropout is not None:
            score = dropout(score)

        # score @ v (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        return score @ v, score