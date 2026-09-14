import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from ts_benchmark.baselines.probenet.layers.Embed import PatchEmbedding
from ts_benchmark.baselines.probenet.layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
)


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class VariablePatchEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.n_vars = n_vars

        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )
        self.variable_embedding = nn.Embedding(n_vars, d_model)

    def forward(self, x):
        B, C, T = x.shape
        x_patches, _ = self.patch_embedding(x)
        assert _ == self.n_vars and _ == C

        x_patches = rearrange(x_patches, "(b c) n d -> b c n d", b=B, c=C)

        variable_ids = torch.arange(C, device=x_patches.device)

        x_patches = x_patches + rearrange(
            self.variable_embedding(variable_ids), "c d -> 1 c 1 d"
        )

        x_patches = rearrange(x_patches, "b c n d -> b (c n) d")

        return x_patches, _


class Decoder(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        )

        y = self.norm2(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class HistoricalCrossForecaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.exo_vars = config.enc_in - config.series_dim
        self.series_dim = config.series_dim
        self.d_model = config.d_model
        self.patch_len = config.patch_len
        self.stride = config.patch_len
        self.padding = (self.patch_len - self.seq_len % self.patch_len) % self.patch_len
        self.dropout = config.dropout
        self.target_window = config.pred_len

        self.exo_patch_embedding = VariablePatchEmbedding(
            n_vars=self.exo_vars,
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            padding=self.padding,
            dropout=self.dropout,
        )

        self.endo_patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            padding=self.padding,
            dropout=self.dropout,
        )

        self.decoder = self.__build_decoder(
            d_model=self.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            dropout=self.dropout,
            activation=config.activation,
            output_attention=False,
            factor=config.factor,
        )

        self.head_nf = self.d_model * int(
            (self.seq_len + self.stride - 1) // self.stride
        )

        self.flatten_head = FlattenHead(
            nf=self.head_nf, target_window=self.target_window, head_dropout=self.dropout
        )

    def __build_decoder(
        self,
        d_model,
        d_ff,
        n_heads,
        dropout,
        activation,
        output_attention,
        factor,
    ):
        self_attention = AttentionLayer(
            FullAttention(
                False,
                factor,
                attention_dropout=dropout,
                output_attention=output_attention,
            ),
            d_model,
            n_heads,
        )
        cross_attention = AttentionLayer(
            FullAttention(
                False,
                factor,
                attention_dropout=dropout,
                output_attention=output_attention,
            ),
            d_model,
            n_heads,
        )
        encoder = Decoder(
            self_attention=self_attention,
            cross_attention=cross_attention,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )
        return encoder

    @staticmethod
    def sample_norm(x, mean, stdev):
        return (x - mean) / stdev

    @staticmethod
    def sample_denorm(x, mean, stdev):
        seq_len = x.shape[1]
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        x = x + (mean[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        return x

    def forward(self, history):
        x_endo = history[:, :, : self.series_dim]
        x_exo = history[:, :, self.series_dim :]

        B, L, C = x_exo.shape

        endo_mean = x_endo.mean(dim=1, keepdim=True).detach()
        endo_stdev = torch.sqrt(
            torch.var(x_endo, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()

        exo_mean = x_exo.mean(dim=1, keepdim=True).detach()
        exo_stdev = torch.sqrt(
            torch.var(x_exo, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()

        x_exo_norm = self.sample_norm(x_exo, exo_mean, exo_stdev)
        x_endo_norm = self.sample_norm(x_endo, endo_mean, endo_stdev)

        y_exo_patches, _ = self.exo_patch_embedding(x_exo_norm.transpose(1, 2))
        assert _ == self.exo_vars and _ == C
        y_endo_patches, _ = self.endo_patch_embedding(x_endo_norm.transpose(1, 2))
        assert _ == self.series_dim and _ == self.series_dim

        y_out = self.decoder(y_endo_patches, y_exo_patches)
        y_out = torch.reshape(
            y_out, (-1, self.series_dim, y_out.shape[-2], y_out.shape[-1])
        ).permute(0, 1, 3, 2)

        y_out = self.flatten_head(y_out).permute(0, 2, 1)

        y_out = self.sample_denorm(y_out, endo_mean, endo_stdev)

        return y_out
