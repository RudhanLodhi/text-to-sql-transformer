import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class AddNorm(nn.Module):
    """x -> sublayer(x) -> dropout -> add(x) -> norm
    """

    def __init__(self, d_model: int, dropout: float= 0.1):
        """initializer of add and norm component of transformer

        Args:
            d_model (int): model hidden dimention
            dropout (float, optional): dropout to be applied on sublayer output. Defaults to 0.1.
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, sublayer):
        """forward pass of the add & norm layer

        Args:
            x (tensor): input tensor of shape (batch, seq_len, d_model)
            sublayer (function): function to be applied on x

        Returns:
            tensor: output normalized tensor of shape (batch, seq_len, d_model)
        """
        added = x + self.dropout(sublayer(x)) # (batch, seq_len, d_model) + (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        output = self.norm(added)
        
        return output # (batch, seq_len, d_model)


class FeedForward(nn.Module):
    """x -> linear(d_ff) -> relu -> dropout(optional) -> linear(d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """initialzier of the feed forward module

        Args:
            d_model (int): model hidden dimention
            d_ff (int, optional): feed forward intermediate hidden dimention
            dropout (float, optional): dropout to be applied after activation. Defaults to 0.1.
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        """forward pass of the feed forward network

        Args:
            x (tensor): input tensor of shape (batch, seq_len, d_model)

        Returns:
            tensor: output tensor of shape (batch, seq_len, d_model)
        """
        x = self.linear1(x) # (batch, seq_len, d_model) @ (d_model, d_ff) -> (batch, seq_len, d_ff)
        x = self.relu(x) # (batch, seq_len, d_ff)
        x = self.dropout(x)
        output = self.linear2(x) # (batch, seq_len, d_ff) @ (d_ff, d_model) -> (batch, seq_len, d_model)
        
        return output


class EncoderLayer(nn.Module):
    """x -> multi-head attention -> add & norm -> feed forward -> add & norm
    """

    def __init__(self, multi_head_attention: MultiHeadAttention, feed_forward_network: FeedForward, d_model: int, dropout: float=0.1):
        """initializer for encoder layer module

        Args:
            multi_head_attention (MultiHeadAttention): multi-head attention module
            feed_forward_network (FeedForward): feed forward network module
            d_model (int): model hidden dimention
            dropout (float, optional): dropout to be applied on encoder layer. Defaults to 0.1.
        """
        super().__init__()

        self.mha = multi_head_attention
        self.ffn = feed_forward_network
        self.add_norms = nn.ModuleList([AddNorm(d_model, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask=None):
        """forward pass for encoder layer module

        Args:
            x (tensor): input tensor of shape (batch, s_seq_len, d_model)
            seq_mask (tensor): sequence mask tensor of shape (batch, 1, 1, s_seq_len)

        Returns:
            tensor: output tensor of shape (batch, s_seq_len, d_model)
        """
        if src_mask is not None and src_mask.device != x.device:
            src_mask = src_mask.to(x.device)

        x = self.add_norms[0](x, lambda x: self.mha(x, x, x, src_mask)) # (batch, s_seq_len, d_model)
        output = self.add_norms[1](x, self.ffn) # (batch, s_seq_len, d_model)
        
        return output


class DecoderLayer(nn.Module):
    """y -> masked multi-head attention -> add & norm -> cross multi-head attention -> add & norm -> feed forward -> add & norm
    """

    def __init__(self, masked_multi_head_attention: MultiHeadAttention, cross_multi_head_attention: MultiHeadAttention, feed_forward_network: FeedForward, d_model: int, dropout: float=0.1):
        """initializer for decoder layer module

        Args:
            masked_multi_head_attention (MultiHeadAttention): masked multi-head attention module
            cross_multi_head_attention (MultiHeadAttention): cross multi-head attention module
            feed_forward_network (FeedForward): feed forward network module
            d_model (int): model hidden dimention
            dropout (float, optional): dropout to be applied on decoder layer. Defaults to 0.1.
        """
        super().__init__()

        self.mmha = masked_multi_head_attention
        self.cmha = cross_multi_head_attention
        self.ffn = feed_forward_network
        self.add_norms = nn.ModuleList([AddNorm(d_model, dropout) for _ in range(3)])
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """forward pass for decoder layer module

        Args:
            x (tensor): input tensor of shape (batch, t_seq_len, d_model)
            enc_output (tensor): encoder output tensor of shape (batch, s_seq_len, d_model)
            src_mask (tensor): source mask tensor of shape (batch, 1, 1, s_seq_len)
            tgt_mask (tensor): target mask tensor of shape (batch, 1, t_seq_len, t_seq_len)

        Returns:
            tensor: output tensor of shape (batch, t_seq_len, d_model)
        """
        if src_mask is not None and src_mask.device != x.device:
            src_mask = src_mask.to(x.device)
        if tgt_mask is not None and tgt_mask.device != x.device:
            tgt_mask = tgt_mask.to(x.device)

        x = self.add_norms[0](x, lambda x: self.mmha(x, x, x, tgt_mask)) # (batch, t_seq_len, d_model)
        x = self.add_norms[1](x, lambda x: self.cmha(x, enc_output, enc_output, src_mask)) # (batch, t_seq_len, d_model)
        output = self.add_norms[2](x, self.ffn) # (batch, t_seq_len, d_model)

        return output


class Encoder(nn.Module):
    """x -> N * encoder layer -> layer norm
    """

    def __init__(self, d_model: int, layers: nn.ModuleList):
        """initializer for encoder module

        Args:
            d_model (int): model hidden dimention
            layers (nn.ModuleList): list of encoder layers
        """
        super().__init__()

        self.layers = layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        """forward pass for encoder module

        Args:
            x (tensor): input tensor of shape (batch, s_seq_len, d_model)
            src_mask (tensor): sequence mask tensor of shape (batch, 1, 1, s_seq_len)

        Returns:
            tensor: output tensor of shape (batch, s_seq_len, d_model)
        """
        if src_mask is not None and src_mask.device != x.device:
            src_mask = src_mask.to(x.device)

        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x) # (batch, s_seq_len, d_model)


class Decoder(nn.Module):
    """y -> N * decoder layer -> layer norm
    """

    def __init__(self, d_model: int, layers: nn.ModuleList):
        """initializer for decoder module

        Args:
            d_model (int): model hidden dimention
            layers (nn.ModuleList): list of decoder layers
        """
        super().__init__()

        self.layers = layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """forward pass for decoder module

        Args:
            x (tensor): input tensor of shape (batch, t_seq_len, d_model)
            enc_output (tensor): encoder output tensor of shape (batch, s_seq_len, d_model)
            src_mask (tensor): source sequence mask tensor of shape (batch, 1, 1, s_seq_len)
            tgt_mask (tensor): target sequence mask tensor of shape (batch, 1, t_seq_len, t_seq_len)

        Returns:
            tensor: output tensor of shape (batch, t_seq_len, d_model)
        """
        if src_mask is not None and src_mask.device != x.device:
            src_mask = src_mask.to(x.device)
        if tgt_mask is not None and tgt_mask.device != x.device:
            tgt_mask = tgt_mask.to(x.device)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)    
        return self.norm(x) # (batch, t_seq_len, d_model)