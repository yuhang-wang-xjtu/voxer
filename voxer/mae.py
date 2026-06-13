"""
3D Masked Autoencoder (MAE) for Voxels.

Architecture:
- Patchify: 64x64x64 voxels -> 8x8x8 patches of 8x8x8 each (512 patches)
- Encoder: ViT (processes only visible patches, ~25%)
- Decoder: ViT (processes all patches, reconstructs masked regions)
- Loss: MSE on masked patches only

Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
Adapted for 3D voxels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def patchify_3d(x: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """
    Split 3D tensor into patches.
    x: [B, C, D, H, W] -> [B, N, C * patch_size^3]
    where N = (D/patch_size) * (H/patch_size) * (W/patch_size)
    """
    B, C, D, H, W = x.shape
    assert D % patch_size == 0 and H % patch_size == 0 and W % patch_size == 0, \
        f"Dims ({D},{H},{W}) must be divisible by patch_size {patch_size}"

    n_depth = D // patch_size
    n_height = H // patch_size
    n_width = W // patch_size

    x = x.reshape(B, C, n_depth, patch_size, n_height, patch_size, n_width, patch_size)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    x = x.reshape(B, n_depth * n_height * n_width, C * patch_size * patch_size * patch_size)

    return x, (n_depth, n_height, n_width)


def unpatchify_3d(
    x: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    patch_size: int = 8,
    channels: int = 4
) -> torch.Tensor:
    """
    Inverse of patchify_3d.
    x: [B, N, C * patch_size^3] -> [B, C, D, H, W]
    """
    B = x.shape[0]
    n_depth, n_height, n_width = grid_shape
    C = channels

    x = x.reshape(B, n_depth, n_height, n_width, patch_size, patch_size, patch_size, C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    x = x.reshape(B, C, n_depth * patch_size, n_height * patch_size, n_width * patch_size)

    return x


def sinusoidal_3d_position_encoding(
    num_patches: int,
    embed_dim: int,
    grid_depth: int,
    grid_height: int,
    grid_width: int
) -> torch.Tensor:
    pe = torch.zeros(num_patches, embed_dim)

    z_pos = torch.arange(grid_depth).repeat_interleave(grid_height * grid_width)
    y_pos = torch.arange(grid_height).repeat(grid_width).repeat(grid_depth)
    x_pos = torch.arange(grid_width).repeat(grid_depth * grid_height)

    dim_per_axis = embed_dim // 3
    div_term = torch.exp(
        torch.arange(0, dim_per_axis, 2).float() * (-math.log(10000.0) / dim_per_axis)
    )

    for i, pos in enumerate([z_pos, y_pos, x_pos]):
        start = i * dim_per_axis
        end = start + dim_per_axis
        pos = pos.float().unsqueeze(1)
        pe[:, start + 0: end: 2] = torch.sin(pos * div_term)
        pe[:, start + 1: end: 2] = torch.cos(pos * div_term)

    return pe


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size: int = 8,
        in_channels: int = 4,
        embed_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size ** 3, embed_dim)

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        patches, grid_shape = patchify_3d(x, self.patch_size)
        patches = self.proj(patches)
        return patches, grid_shape


class Attention3D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VoxelMAE(nn.Module):
    """
    3D Masked Autoencoder for voxels.
    """

    def __init__(
        self,
        voxel_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 4,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_depth: int = 4,
        decoder_heads: int = 12,
        decoder_embed_dim: int = 384,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.grid_depth = voxel_size // patch_size
        self.grid_height = voxel_size // patch_size
        self.grid_width = voxel_size // patch_size
        self.num_patches = self.grid_depth * self.grid_height * self.grid_width

        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, encoder_heads, mlp_ratio, dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.enc_to_dec = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, in_channels * patch_size ** 3
        )

    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        patches, grid_shape = self.patch_embed(x)
        patches = patches + self.pos_embed

        x, mask, ids_restore = self.random_masking(patches)

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        x = self.enc_to_dec(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )

        x_ = x_ + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)

        x_ = self.decoder_pred(x_)
        grid_shape = (self.grid_depth, self.grid_height, self.grid_width)
        x_ = unpatchify_3d(x_, grid_shape, self.patch_size, 4)

        return x_

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        patches, grid_shape = self.patch_embed(x)
        patches = patches + self.pos_embed

        for blk in self.encoder_blocks:
            patches = blk(patches)
        patches = self.encoder_norm(patches)

        return patches

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.enc_to_dec(latent)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        grid_shape = (self.grid_depth, self.grid_height, self.grid_width)
        x = unpatchify_3d(x, grid_shape, self.patch_size, 4)

        return x


def mae_vit_small(**kwargs) -> VoxelMAE:
    return VoxelMAE(
        embed_dim=384,
        encoder_depth=12,
        encoder_heads=6,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_heads=6,
        **kwargs
    )


def mae_vit_base(**kwargs) -> VoxelMAE:
    return VoxelMAE(
        embed_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_heads=12,
        **kwargs
    )
