import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, latent_tokens):
        _, K, D = latent_tokens.shape
        if K <= 1:
            return latent_tokens.new_zeros(())

        z = F.normalize(latent_tokens, p=2, dim=-1, eps=self.eps)

        gram = torch.matmul(
            z,
            z.transpose(-1, -2),
        )
        # [BN, K, K]

        off_diagonal_mask = ~torch.eye(
            K,
            device=z.device,
            dtype=torch.bool,
        )

        off_diagonal = gram[:, off_diagonal_mask]
        # [BN, K * (K - 1)]

        return off_diagonal.square().mean()
