import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from ts_benchmark.baselines.probenet.layers.Embed import PatchEmbedding_wo_pos
from ts_benchmark.baselines.probenet.utils.orth_loss import OrthogonalityLoss
from ts_benchmark.baselines.probenet.layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
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


class VariablePatchEmbedding_wo_pos(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.n_vars = n_vars

        self.patch_embedding = PatchEmbedding_wo_pos(
            d_model, patch_len, stride, padding, dropout
        )
        self.variable_embedding = nn.Embedding(n_vars, d_model)

    def forward(self, x):
        B, C, T = x.shape
        x_patches, _ = self.patch_embedding(x)
        assert _ == self.n_vars and _ == C

        N = x_patches.shape[1]

        x_patches = rearrange(x_patches, "(b c) n d -> (b n) c d", b=B, c=C)

        variable_ids = torch.arange(C, device=x_patches.device)

        x_patches = x_patches + rearrange(
            self.variable_embedding(variable_ids), "c d -> 1 c d"
        )

        return x_patches, _, N


class Encoder(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_k=None, x_mask=None, cross_mask=None):
        if x_k is not None:
            out, attn = self.attention(x, x_k, cross, attn_mask=cross_mask)
        else:
            out, attn = self.attention(x, cross, cross, attn_mask=cross_mask)

        x = x + self.dropout(out)

        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), x, attn


class Probe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.exo_vars = config.enc_in - config.series_dim
        self.series_dim = config.series_dim
        self.d_model = config.d_model
        self.patch_len = config.patch_len
        self.stride = config.patch_len
        self.padding = 0
        self.dropout = config.dropout
        self.target_window = config.patch_len
        self.n_templates = config.n_templates
        # Runtime-only intervention. Training and validation keep this at zero;
        # Probe wrappers enable it immediately before final test inference.
        self.patch_order_shift = 0

        self.variable_patch_embedding = VariablePatchEmbedding_wo_pos(
            n_vars=self.exo_vars,
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            padding=self.padding,
            dropout=self.dropout,
        )

        self.template_queries = nn.Embedding(self.n_templates, self.d_model)
        # self.template_keys = nn.Embedding(self.exo_vars, self.d_model)

        self.head_nf = self.d_model * self.n_templates
        # self.head_nf = self.d_model * self.exo_vars  # TODO

        self.flatten_head = FlattenHead(
            nf=self.head_nf, target_window=self.target_window, head_dropout=self.dropout
        )

        self.orth_loss = OrthogonalityLoss()

        self.encoder = self.__build_encoder(
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            dropout=config.dropout,
            activation=config.activation,
            output_attention=False,
            factor=config.factor,
        )

    def __build_encoder(
        self,
        d_model,
        d_ff,
        n_heads,
        dropout,
        activation,
        output_attention,
        factor,
    ):
        attention = AttentionLayer(
            FullAttention(
                False,
                factor,
                attention_dropout=dropout,
                output_attention=output_attention,
            ),
            d_model,
            n_heads,
        )
        encoder = Encoder(
            attention=attention,
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

    def set_patch_order_shift(self, shift=0):
        """Set the cyclic patch shift used for test-time misalignment."""
        self.patch_order_shift = int(shift)

    def _shift_patch_order(self, y_exo):
        """Cyclically shift complete future-exogenous patches.

        The operation preserves the temporal order inside every patch and keeps
        all exogenous variables together. Only the correspondence between a
        patch and its target time interval is changed.
        """
        if self.patch_order_shift == 0:
            return y_exo

        batch_size, horizon, n_vars = y_exo.shape
        if horizon % self.patch_len != 0:
            raise ValueError(
                f"horizon {horizon} must be divisible by patch_len "
                f"{self.patch_len} for patch-order shifting"
            )

        n_patches = horizon // self.patch_len
        if n_patches <= 1:
            raise ValueError(
                "patch-order shifting requires at least two future patches"
            )

        shift = self.patch_order_shift % n_patches
        if shift == 0:
            raise ValueError(
                f"patch_order_shift={self.patch_order_shift} leaves all "
                f"{n_patches} patches aligned"
            )

        patches = y_exo.reshape(batch_size, n_patches, self.patch_len, n_vars)
        patches = torch.roll(patches, shifts=shift, dims=1)
        return patches.reshape(batch_size, horizon, n_vars)

    def forward(self, input, exog_future):
        x_endo = input[:, :, : self.series_dim]
        x_exo = input[:, :, self.series_dim :]
        y_exo = exog_future

        B, L, C = y_exo.shape
        n_exo_vars = C

        endo_mean = x_endo.mean(dim=1, keepdim=True).detach()
        endo_stdev = torch.sqrt(
            torch.var(x_endo, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()

        exo_mean = x_exo.mean(dim=1, keepdim=True).detach()
        exo_stdev = torch.sqrt(
            torch.var(x_exo, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()

        y_exo = self.sample_norm(y_exo, exo_mean, exo_stdev)
        y_exo = self.sample_denorm(y_exo, endo_mean, endo_stdev)

        y_exo = self._shift_patch_order(y_exo)

        y_exo_patches, _, N = self.variable_patch_embedding(y_exo.transpose(1, 2))
        assert _ == n_exo_vars and _ == C

        template_queries = repeat(
            self.template_queries.weight, "k d -> (b n) k d", b=B, n=N
        )

        # template_keys = repeat(
        #     self.template_keys.weight, "k d -> (b n) k d", b=B, n=N
        # )  # TODO
        # latent_output, cross_latents, attn = self.encoder(
        #     template_queries, y_exo_patches, template_keys
        # )  # TODO

        latent_output, cross_latents, attn = self.encoder(
            template_queries, y_exo_patches
        )

        orth_loss = self.orth_loss(cross_latents)

        # latent_output, cross_latents, attn = self.encoder(
        #     y_exo_patches, y_exo_patches
        # )  # TODO
        # orth_loss = 0  # TODO

        y_out = rearrange(
            latent_output,
            "(b n) k d -> b n d k",
            b=B,
            n=N,
        )

        y_out = self.flatten_head(y_out)

        y_out = rearrange(
            y_out,
            "b n p -> b (n p) 1",
        )

        # y_out = self.sample_denorm(y_out, endo_mean, endo_stdev)

        return y_out, orth_loss
