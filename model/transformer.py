import torch
import torch.nn as nn
from model.embeddings import TokenEmbedding, InputLayer
from model.attention import MultiHeadAttention
from model.layers import Encoder, Decoder, EncoderLayer, DecoderLayer, FeedForward


class Transformer(nn.Module):
    """transformer model
    x -> encoder -> decoder -> projection:using transposed embedding matrix -> softmax/log_softmax
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, embedding: TokenEmbedding, input_layer: InputLayer):
        """initializer for transformer model

        Args:
            encoder (Encoder): encoder module
            decoder (Decoder): decoder module
            embedding (TokenEmbedding): shared token embedding module to get transposed matrix for projection
            input_layer (InputLayer): input layer module
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.input_layer = input_layer

    def forward(self, src, tgt, tf: float=0.5, pad_id: int=0):
        """end-to-end transformer pass

        Args:
            src (tensor): source token ids of shape (batch, s_seq_len)
            tgt (tensor): target token ids including BOS/EOS of shape (batch, t_seq_len)
            tf (float, optional): probability of using the ground-truth previous token. Defaults to 0.5.
            pad_id (int, optional): padding token id. Defaults to 0.

        Returns:
            tensor: logits of shape (batch, t_seq_len, vocab_size)
        """
        src_mask = (src != pad_id).unsqueeze(1).unsqueeze(1) # (batch, 1, 1, s_seq_len)
        enc_output = self.encode(src, src_mask)

        tgt_length = tgt.size(1) - 1 # tokens to be predicted by model after the first input BOS
        dec_input = tgt[:, :1] # first BOS token as decoder input, shape becomes (batch, 1)
        logits = []

        for i in range(tgt_length):
            seq_len = dec_input.shape[1] # current_t_seq_len
            tgt_mask = (dec_input != pad_id).unsqueeze(1).unsqueeze(1) & \
                torch.tril(
                    torch.ones(
                        seq_len, seq_len,
                        dtype=torch.bool, device=dec_input.device,
                    ) # (current_t_seq_len, current_t_seq_len)
                ).view(1, 1, seq_len, seq_len) # (1, 1, current_t_seq_len, current_t_seq_len)
            # (batch, 1, 1, current_t_seq_len) & (1, 1, current_t_seq_len, current_t_seq_len) -> (batch, 1, current_t_seq_len, current_t_seq_len)
            
            dec_output = self.decode(dec_input, enc_output, src_mask, tgt_mask) # (batch, current_t_seq_len, d_model)
            step_logits = self.projection(dec_output[:, -1:, :]) # get logits for the last token (batch, 1, vocab_size)
            logits.append(step_logits)
            
            predicted_token = step_logits.argmax(dim=-1)

            if torch.rand(1).item() < tf:
                next_token = tgt[:, i + 1:i + 2]
            else:
                next_token = predicted_token

            dec_input = torch.cat([dec_input, next_token], dim=1)

        logits = torch.cat(logits, dim=1) # (batch, t_seq_len, d_model)
        return logits


    def encode(self, src, src_mask):
        """forward pass for encoder module

        Args:
            src (tensor): source token ids of shape (batch, s_seq_len)
            src_mask (tensor): source sequence mask tensor of shape (___)
        
        Returns:
            tensor: output tensor of shape (batch, s_seq_len, d_model)
        """
        x = self.input_layer(src)
        return self.encoder(x, src_mask) # (batch, s_seq_len, d_model)

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        """forward pass for decoder module

        Args:
            tgt (tensor): source token ids of shape (batch, t_seq_len)
            enc_output (tensor): target token ids of shape (batch, s_seq_len, d_model)
            src_mask (tensor): source sequence mask tensor of shape (___)
            tgt_mask (tensor): target sequence mask tensor of shape (___)

        Returns:
            tensor: output tensor of shape (batch, t_seq_len, d_model)
        """
        x = self.input_layer(tgt)
        return self.decoder(x, enc_output, src_mask, tgt_mask) # (batch, t_seq_len, d_model)

    def projection(self, x):
        """forward pass for projection layer module

        Args:
            x (tensor): input tensor of shape (batch, t_seq_len, d_model)

        Returns:
            tensor: output tensor of shape (batch, t_seq_len, vocab_size)
        """
        proj_weights = self.embedding.emb.weight.t() # (vocab_size, d_model) -> (d_model, vocab_size)
        logits = x @ proj_weights # (batch, t_seq_len, d_model) @ (d_model, vocab_size) -> (batch, t_seq_len, vocab_size)
        return logits


def build_transformer(vocab_size: int=8000, max_len: int=512, d_model: int=256, h: int=4, N: int=3, d_ff: int=1024, dropout: float=0.1, device=None):
    """build the end to end transformer with all components

    Args:
        vocab_size (int, optional): size of our shared vocabulary. Defaults to 8000.
        max_len (int, optional): maximum length of the sequence allowed. Defaults to 512.
        d_model (int, optional): model hidden dimention. Defaults to 256.
        h (int, optional): number of attention heads. Defaults to 4.
        N (int, optional): number of stacked layers on encoder and decoder side. Defaults to 3.
        d_ff (int, optional): feed forward network intermediate hidden dimention. Defaults to 1024.
        dropout (float, optional): dropout to be applied at certain levels within transformer. Defaults to 0.1.
        device (torch.device | str | None, optional): optional device to place the model on. Defaults to None.

    Returns:
        Transformer: end to end builded transformer model
    """
    embedding = TokenEmbedding(vocab_size, d_model)
    input_layer = InputLayer(embedding, d_model, max_len, dropout)

    encoders = []
    for _ in range(N):
        multi_head_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward_network = FeedForward(d_model, d_ff, dropout)
        encoder_layer = EncoderLayer(multi_head_attention, feed_forward_network, d_model, dropout)
        encoders.append(encoder_layer)

    decoders = []
    for _ in range(N):
        masked_multi_head_attention = MultiHeadAttention(d_model, h, dropout)
        cross_multi_head_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward_network = FeedForward(d_model, d_ff, dropout)
        decoder_layer = DecoderLayer(masked_multi_head_attention, cross_multi_head_attention, feed_forward_network, d_model, dropout)
        decoders.append(decoder_layer)

    encoder = Encoder(d_model, nn.ModuleList(encoders))
    decoder = Decoder(d_model, nn.ModuleList(decoders))

    transformer = Transformer(encoder, decoder, embedding, input_layer)

    for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    if device is not None:
        transformer = transformer.to(device)

    return transformer