import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """layer normalization module
    gemma * (x - mean) / (std + eps) + beta
    gemma and beta are learnable parameters each having d_model dimention (for each hidden feature)
    """

    def __init__(self, d_model: int, eps: float=1e-6):
        """initializer for layer normalization module

        Args:
            d_model (int): model hidden dimention
            eps (float, optional): small value to avoid division by zero. Defaults to 1e-6.
        """
        pass

    def forward(self, x):
        """forward pass for layer normalization module

        Args:
            x (tensor): unnormalized input tensor of shape (batch, seq_len, d_model)

        Returns:
            tensor: normalized output tensor of shape (batch, seq_len, d_model)
        """
        pass


class FeedForwardNetwork(nn.Module):
    """feed forward network module
    x -> linear(d_ff) -> relu -> dropout(optional) -> linear(d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float=0.1):
        """initializer for feed forward network module

        Args:
            d_model (int): model hidden dimension
            d_ff (int): feed forward hidden dimension
            dropout (float, optional): dropout to be applied on feed forward network. Defaults to 0.1.
        """
        pass

    def forward(self, x):
        """forward pass for feed forward network module

        Args:
            x (tensor): input tensor of shape (batch, seq_len, d_model)

        Returns:
            tensor: output tensor of shape (batch, seq_len, d_model)
        """
        pass


class ResidualConnection(nn.Module):
    """residual connection module
    x -> sublayer(x) -> dropout(optional) -> add(x) -> layer_norm
    """

    def __init__(self, d_model: int, dropout: float=0.1):
        """initializer for residual connection module

        Args:
            d_model (int): model hidden dimention
            dropout (float, optional): dropout to be applied on residual connection. Defaults to 0.1.
        """
        pass

    def forward(self, x, sublayer):
        """forward pass for residual connection module

        Args:
            x (tensor): input tensor of shape (batch, seq_len, d_model)
            sublayer (function): function to be applied on x

        Returns:
            tensor: output tensor of shape (batch, seq_len, d_model)
        """
        pass


class EncoderLayer(nn.Module):
    """encoder layer module
    x -> multi-head attention -> residual connection -> feed forward network -> residual connection
    """

    def __init__(self, multi_head_attention: MultiHeadAttention, feed_forward_network: FeedForwardNetwork, d_model: int, dropout: float=0.1):
        """initializer for encoder layer module

        Args:
            multi_head_attention (MultiHeadAttention): multi-head attention module
            feed_forward_network (FeedForwardNetwork): feed forward network module
            d_model (int): model hidden dimention
            dropout (float, optional): dropout to be applied on encoder layer. Defaults to 0.1.
        """
        pass

    def forward(self, x, src_mask):
        """forward pass for encoder layer module

        Args:
            x (tensor): input tensor of shape (batch, s_seq_len, d_model)
            seq_mask (tensor): sequence mask tensor of shape (___)

        Returns:
            tensor: output tensor of shape (batch, s_seq_len, d_model)
        """
        pass


class DecoderLayer(nn.Module):
    """decoder layer module
    x -> masked multi-head attention -> residual connection -> multi-head attention -> residual connection -> feed forward network -> residual connection
    """

    def __init__(self, masked_multi_head_attention: MultiHeadAttention, cross_multi_head_attention: MultiHeadAttention, feed_forward_network: FeedForwardNetwork, d_model: int, dropout: float=0.1):
        """initializer for decoder layer module

        Args:
            masked_multi_head_attention (MultiHeadAttention): masked multi-head attention module
            cross_multi_head_attention (MultiHeadAttention): cross multi-head attention module
            feed_forward_network (FeedForwardNetwork): feed forward network module
            d_model (int): model hidden dimention
            dropout (float, optional): dropout to be applied on decoder layer. Defaults to 0.1.
        """
        pass

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """forward pass for decoder layer module

        Args:
            x (tensor): input tensor of shape (batch, t_seq_len, d_model)
            enc_output (tensor): encoder output tensor of shape (batch, s_seq_len, d_model)
            src_mask (tensor): source sequence mask tensor of shape (___)
            tgt_mask (tensor): target sequence mask tensor of shape (___)

        Returns:
            tensor: output tensor of shape (batch, t_seq_len, d_model)
        """
        pass


class Encoder(nn.Module):
    """encoder module
    x -> N * encoder layer -> layer norm
    """

    def __init__(self, d_model: int, layers: nn.ModuleList):
        """initializer for encoder module

        Args:
            d_model (int): model hidden dimention
            layers (nn.ModuleList): list of encoder layers
        """
        pass

    def forward(self, x, src_mask):
        """forward pass for encoder module

        Args:
            x (tensor): input tensor of shape (batch, s_seq_len, d_model)
            src_mask (tensor): sequence mask tensor of shape (___)

        Returns:
            tensor: output tensor of shape (batch, s_seq_len, d_model)
        """
        pass


class Decoder(nn.Module):
    """decoder module
    x -> N * decoder layer -> layer norm
    """

    def __init__(self, d_mdoel: int, layers: nn.ModuleList):
        """initializer for decoder module

        Args:
            d_model (int): model hidden dimention
            layers (nn.ModuleList): list of decoder layers
        """
        pass

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """forward pass for decoder module

        Args:
            x (tensor): input tensor of shape (batch, t_seq_len, d_model)
            enc_output (tensor): encoder output tensor of shape (batch, s_seq_len, d_model)
            src)_mask (tensor): source sequence mask tensor of shape (___)
            tgt_mask (tensor): target sequence mask tensor of shape (___)

        Returns:
            tensor: output tensor of shape (batch, t_seq_len, d_model)
        """
        pass