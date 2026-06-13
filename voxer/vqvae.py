"""
Vector Quantized VAE with EMA Codebook for 3D Voxels.

Uses the MAE encoder/decoder architecture with a VQ bottleneck.
Codebook is updated via Exponential Moving Average (EMA) for stability.

Reference:
- van den Oord et al., "Neural Discrete Representation Learning" (NeurIPS 2017)
- Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2" (NeurIPS 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with EMA codebook updates.

    Args:
        num_embeddings: Codebook size (number of discrete codes)
        embedding_dim: Dimension of each code vector
        commitment_cost: Weight for commitment loss (beta)
        decay: EMA decay rate for codebook updates (default 0.99)
        epsilon: Small constant for numerical stability
    """

    def __init__(
        self,
        num_embeddings: int = 8192,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.register_buffer("embedding", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", torch.zeros(num_embeddings, embedding_dim))

        nn.init.kaiming_uniform_(self.embedding)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = z.shape
        z_flat = z.reshape(-1, C)

        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * z_flat @ self.embedding.t()
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = encoding_indices.reshape(B, N)

        z_q_flat = self.embedding[encoding_indices.reshape(-1)]
        z_q = z_q_flat.reshape(B, N, C)

        if self.training:
            self._update_ema(z_flat, encoding_indices.reshape(-1))

        loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)

        z_q_st = z + (z_q - z).detach()

        return z_q_st, loss, encoding_indices

    def _update_ema(self, z_flat: torch.Tensor, indices: torch.Tensor):
        with torch.no_grad():
            num_samples = z_flat.shape[0]

            for i in range(self.num_embeddings):
                mask = (indices == i)
                n_i = mask.sum().float()

                self.ema_count[i] = self.ema_count[i] * self.decay + n_i * (1 - self.decay)

                if n_i > 0:
                    z_i = z_flat[mask].mean(dim=0)
                    self.ema_weight[i] = (
                        self.ema_weight[i] * self.decay + z_i * (1 - self.decay)
                    )

                    n = torch.sum(self.ema_count)
                    self.ema_count = self.ema_count / n * num_samples

                    self.embedding[i] = self.ema_weight[i] / (self.ema_count[i] + self.epsilon)

    def get_quantized(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding[indices]

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embedding)


class VoxelVQVAE(nn.Module):
    """
    VQ-VAE for 3D voxels, built on top of a pretrained MAE encoder/decoder.

    Pipeline:
        64^3 voxels -> MAE Encoder -> [B, 512, 768]
        -> Linear 768->256 -> VQ -> Linear 256->768
        -> MAE Decoder -> 64^3 voxels
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_embeddings: int = 8192,
        embedding_dim: int = 256,
        encoder_dim: int = 768,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim

        self.pre_vq = nn.Sequential(
            nn.Linear(encoder_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.post_vq = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
        )

        self.quantizer = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder.encode(x)
        z = self.pre_vq(z)
        _, _, indices = self.quantizer(z)
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() > 2:
            N = 1
            for d in indices.shape[1:]:
                N *= d
            indices = indices.reshape(indices.shape[0], N)
        z_q = self.quantizer.lookup(indices)
        z_q = self.post_vq(z_q)
        return self.decoder.decode(z_q)

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder.decode(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder.encode(x)

        z_pre = self.pre_vq(z)
        z_q, vq_loss, indices = self.quantizer(z_pre)
        z_post = self.post_vq(z_q)

        x_recon = self.decoder.decode(z_post)

        recon_loss = F.mse_loss(x_recon, x)

        total_loss = recon_loss + vq_loss

        return x_recon, total_loss, indices, z

    def get_codebook_usage(self) -> torch.Tensor:
        counts = self.quantizer.ema_count
        return (counts > 0).float().mean()


def create_vqvae_from_mae(
    mae_model: nn.Module,
    num_embeddings: int = 8192,
    embedding_dim: int = 256,
    commitment_cost: float = 0.25,
    decay: float = 0.99,
) -> VoxelVQVAE:
    return VoxelVQVAE(
        encoder=mae_model,
        decoder=mae_model,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        encoder_dim=mae_model.patch_embed.proj.out_features,
        commitment_cost=commitment_cost,
        decay=decay,
    )
