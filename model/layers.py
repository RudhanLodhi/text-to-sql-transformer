import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class AddNorm(nn.Module):
    """Add & Norm layer acts as a residual connection followed by layer normalization"""
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, sublayer_x):
        added = x + sublayer_x # (batch, seq_len, d_model) + (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        output = self.norm(added)
        
        return output # (batch, seq_len, d_model)

class FeedForward(nn.Module):
    """2 layer point wise feed forward network with ReLU activation and dropout"""
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        
        x = self.linear1(x) # (Batch, seq_len, d_model) @ (d_model, d_ff) -> (Batch, Seq_Len, d_ff)
        x = self.relu(x)   # (Batch, Seq_Len, d_ff)
        x = self.dropout(x)       
        output = self.linear2(x) # (Batch, Seq_Len, d_ff) @ (d_ff, d_model) -> (Batch, Seq_Len, d_model)
        
        return output

class EncoderLayer(nn.Module):
    """encoder layer module
    x -> multi-head attention -> residual connection -> feed forward network -> residual connection
    """

    def __init__(self, multi_head_attention: MultiHeadAttention, feed_forward_network: FeedForward, add_norm: AddNorm, dropout: float=0.1):
        """initializer for encoder layer module

        Args:
            multi_head_attention (MultiHeadAttention): multi-head attention module
            feed_forward_network (FeedForwardNetwork): feed forward network module
            add_norm (AddNorm): add & norm module
            d_model (int): model hidden dimention
            dropout (float, optional): dropout to be applied on encoder layer. Defaults to 0.1.
        """
        super().__init__()
        self.mha = multi_head_attention
        self.ffn = feed_forward_network
        self.add_norm = add_norm
        self.add_norm2 = AddNorm(multi_head_attention.d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask):
        """forward pass for encoder layer module

        Args:
            x (tensor): input tensor of shape (batch, s_seq_len, d_model)
            seq_mask (tensor): sequence mask tensor of shape (___)

        Returns:
            tensor: output tensor of shape (batch, s_seq_len, d_model)
        """
        a = self.mha(x, x, x, src_mask) # (batch, s_seq_len, d_model)
        a = self.add_norm(x, a) # (batch, s_seq_len, d_model)
        
        f = self.ffn(a) # (batch, s_seq_len, d_model)
        output = self.add_norm2(a, f) # (batch, s_seq_len, d_model)
        
        return output

class DecoderLayer(nn.Module):
    """ Decoder layer module
    y -> masked multi-head attention -> add norm -> multi-head cross attention -> add norm -> FFN -> residual connection
    """
    def __init__(self, multi_head_attention: MultiHeadAttention, feed_forward_network: FeedForward, add_norm: AddNorm, dropout: float=0.1):
        """initializer for decoder layer module

        Args:
            multi_head_attention (MultiHeadAttention): multi-head attention module
            feed_forward_network (FeedForwardNetwork): feed forward network module
            add_norm (AddNorm): add & norm module
            dropout (float): Defaults to 0.1.
        """
        super().__init__()
        self.mha1 = multi_head_attention
        self.mha2 = MultiHeadAttention(
            multi_head_attention.d_model,
            multi_head_attention.h,
            dropout
        )
        self.ffn = feed_forward_network
        self.add_norm1 = add_norm
        self.add_norm2 = AddNorm(multi_head_attention.d_model)
        self.add_norm3 = AddNorm(multi_head_attention.d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        """forward pass for decoder layer module

        Args:
            x (tensor): input tensor of shape (batch, t_seq_len, d_model)
            enc_output (tensor): encoder output tensor of shape (batch, s_seq_len, d_model)
            src_mask (tensor): source mask tensor of shape (batch, 1, 1, s_seq_len)
            tgt_mask (tensor): target mask tensor of shape (batch, 1, t_seq_len, t_seq_len)
        """
        a = self.mha1(x, x, x, tgt_mask) # (batch, t_seq_len, d_model)
        b = self.add_norm1(x, a) 
        
        c = self.mha2(b, enc_output, enc_output, src_mask)
        d = self.add_norm2(b, c) 

        e = self.ffn(d) 
        out = self.add_norm3(d, e) # (batch, t_seq_len, d_model)
        return out


class Encoder(nn.Module):
    def __init__(self, d_model: int, N: int , dropout: float = 0.1):
        """Stackes N layers of EncoderLayer to form the complete encoder module"""
        
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(MultiHeadAttention(d_model, h=8, dropout=dropout)
                                                  , FeedForward(d_model, d_ff=1024, dropout=dropout)
                                                  , AddNorm(d_model),
                                                  dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        
        for layer in self.layers:
            x = layer(x, src_mask)

        output = self.norm(x)
        return output
class Decoder(nn.Module):
    """Stack N decoder layers and apply a final layer normalization."""

    def __init__(self, d_model: int, N: int, h: int = 8,
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
                MultiHeadAttention(d_model, h=h, dropout=dropout),
                FeedForward(d_model, d_ff=d_ff, dropout=dropout),
                AddNorm(d_model),
                dropout
            )
            for _ in range(N)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return self.norm(x)
