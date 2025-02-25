import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.3):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Define linear layers for query, key, and value
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # Output linear layer
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)  # (batch_size, seq_length, embed_dim)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_length, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        )  # (batch_size, num_heads, seq_length, seq_length)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(
            scores, dim=-1
        )  # (batch_size, num_heads, seq_length, seq_length)
        attn = self.dropout(attn)

        # Weighted sum of values
        context = torch.matmul(attn, V)  # (batch_size, num_heads, seq_length, head_dim)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_length, embed_dim)

        # Final linear layer
        out = self.out_linear(context)  # (batch_size, seq_length, embed_dim)

        return out, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.3):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention sublayer
        attn_output, _ = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # Feedforward sublayer
        ffn_output = self.ffn(src)
        src = src + self.dropout(ffn_output)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.3):
        super(TransformerEncoder, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask, src_key_padding_mask)
        src = self.norm(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.3):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Self-attention sublayer
        attn_output, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)

        # Cross-attention sublayer
        attn_output, _ = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm2(tgt)

        # Feedforward sublayer
        ffn_output = self.ffn(tgt)
        tgt = tgt + self.dropout(ffn_output)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.3):
        super(TransformerDecoder, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        for layer in self.layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )
        tgt = self.norm(tgt)
        return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_length, embed_dim)
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dropout=0.3,
        max_len=5000,
    ):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        self.encoder = TransformerEncoder(
            embed_dim, num_heads, ff_dim, num_encoder_layers, dropout
        )
        self.pos_decoder = PositionalEncoding(embed_dim, max_len)
        self.decoder = TransformerDecoder(
            embed_dim, num_heads, ff_dim, num_decoder_layers, dropout
        )

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        src: (batch_size, src_seq_length, embed_dim)
        tgt: (batch_size, tgt_seq_length, embed_dim)
        """
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        memory = self.encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(
            tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask
        )
        return output  # (batch_size, tgt_seq_length, embed_dim)
