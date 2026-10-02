import torch
import torch.nn as nn
from model.embeddings import TokenEmbedding, InputLayer
from model.attention import MultiHeadAttention
from model.layers import Encoder, Decoder, EncoderLayer, DecoderLayer, FeedForwardNetwork


class Transformer(nn.Module):
    """transformer model
    x -> encoder -> decoder -> projection:using transposed embedding matrix -> softmax/log_softmax
    """

    def __init__(self, encoder: Encoder, Decoder: Decoder, embedding: TokenEmbedding, input_layer: InputLayer):
        """initializer for transformer model

        Args:
            encoder (Encoder): encoder module
            Decoder (Decoder): decoder module
            embedding (TokenEmbedding): shared token embedding module to get transposed matrix for projection
            input_layer (InputLayer): input layer module
        """
        pass

    def encode(self, src, src_mask):
        """forward pass for encoder module

        Args:
            src_ids (tensor): source token ids of shape (batch, s_seq_len)
            src_mask (tensor): source sequence mask tensor of shape (___)
        
        Returns:
            tensor: output tensor of shape (batch, s_seq_len, d_model)
        """
        pass

    def decoder(self, tgt, encoder_output, src_mask, tgt_mask):
        """forward pass for decoder module

        Args:
            src_ids (tensor): source token ids of shape (batch, s_seq_len)
            tgt_ids (tensor): target token ids of shape (batch, t_seq_len)
            src_mask (tensor): source sequence mask tensor of shape (___)
            tgt_mask (tensor): target sequence mask tensor of shape (___)

        Returns:
            tensor: output tensor of shape (batch, t_seq_len, d_model)
        """
        pass

    def projection(self, x):
        """forward pass for projection layer module

        Args:
            x (tensor): input tensor of shape (batch, t_seq_len, d_model)

        Returns:
            tensor: output tensor of shape (batch, t_seq_len, vocab_size)
        """
        pass


def build_transformer(vocab_size: int=8000, max_len: int=512, d_model: int=256, h: int=4, N: int=3, d_ff: int=1024, dropout: float=0.1):
    """build the end to end transformer with all components

    Args:
        vocab_size (int, optional): size of our shared vocabulary. Defaults to 8000.
        max_len (int, optional): maximum length of the sequence allowed. Defaults to 512.
        d_model (int, optional): model hidden dimention. Defaults to 256.
        h (int, optional): number of attention heads. Defaults to 4.
        N (int, optional): number of stacked layers on encoder and decoder side. Defaults to 3.
        d_ff (int, optional): feed forward network intermediate hidden dimention. Defaults to 1024.
        dropout (float, optional): dropout to be applied at certain levels within transformer. Defaults to 0.1.

    Returns:
        Transformer: end to end builded transformer model
    """
    embedding = TokenEmbedding(vocab_size, d_model)
    input_layer = InputLayer(embedding, d_model, max_len, dropout)

    encoders = []
    for _ in range(N):
        multi_head_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward_network = FeedForwardNetwork(d_model, d_ff, dropout)
        encoder_layer = EncoderLayer(multi_head_attention, feed_forward_network, d_model, dropout)
        encoders.append(encoder_layer)

    decoders = []
    for _ in range(N):
        masked_multi_head_attention = MultiHeadAttention(d_model, h, dropout)
        cross_multi_head_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward_network = FeedForwardNetwork(d_model, d_ff, dropout)
        decoder_layer = DecoderLayer(masked_multi_head_attention, cross_multi_head_attention, feed_forward_network, d_model, dropout)
        decoders.append(decoder_layer)

    encoder = Encoder(d_model, nn.ModuleList(encoders))
    decoder = Decoder(d_model, nn.ModuleList(decoders))

    transformer = Transformer(encoder, decoder, embedding, input_layer)

    for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return transformer