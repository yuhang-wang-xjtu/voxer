"""
Text-Conditioned Autoregressive Transformer for 3D Voxel Token Generation.

Given a text description embedding from a pretrained LLM,
generates a sequence of VQ-VAE codebook indices that reconstruct to a 3D voxel model.

Architecture: GPT-style causal transformer decoder with cross-attention to text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CausalSelfAttention(dim, num_heads, dropout)

        if use_cross_attention:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = CrossAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), causal_mask)

        if self.use_cross_attention:
            x = x + self.cross_attn(self.norm_cross(x), context)

        x = x + self.mlp(self.norm2(x))
        return x


class VoxelGPT(nn.Module):
    """
    Autoregressive transformer that generates voxel token sequences
    conditioned on text embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        num_tokens: int = 512,
        text_embed_dim: int = 384,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.start_token, std=0.02)

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_cross_attention=True,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask

    def forward(
        self,
        token_ids: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Autoregressive training forward pass.

        Input: [start_token, token_0, token_1, ..., token_{T-2}]
        Target: [token_0, token_1, ..., token_{T-1}]

        token_ids: [B, T] - full target token sequence
        text_emb: [B, text_embed_dim] - text embedding from pretrained LLM

        Returns: [B, T, vocab_size] - logits predicting token_ids from shifted inputs
        """
        B, T = token_ids.shape
        device = token_ids.device

        text_feat = self.text_proj(text_emb).unsqueeze(1)

        token_emb = self.token_embed(token_ids)

        start = self.start_token.expand(B, -1, -1)
        x = torch.cat([start, token_emb[:, :-1, :]], dim=1)

        pos = torch.arange(0, T, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos)

        causal_mask = self._get_causal_mask(T, device)

        for layer in self.layers:
            x = layer(x, text_feat, causal_mask)

        x = self.norm(x)
        logits = self.head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        text_emb: torch.Tensor,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 100,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Autoregressively generate a token sequence from text embedding.

        Uses start_token embedding (from training) as the first input,
        then greedily/top-k/top-p samples subsequent tokens.
        """
        self.eval()
        B = text_emb.shape[0]
        device = text_emb.device

        text_feat = self.text_proj(text_emb).unsqueeze(1)

        x = self.start_token.expand(B, 1, -1)
        x = x + self.pos_embed(torch.tensor([0], device=device))

        generated = []

        for step in range(max_tokens):
            T = x.shape[1]
            causal_mask = self._get_causal_mask(T, device)

            h = x
            for layer in self.layers:
                h = layer(h, text_feat, causal_mask)

            h = self.norm(h)
            logits = self.head(h[:, -1:, :]).squeeze(1) / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token)

            next_emb = self.token_embed(next_token)
            next_emb = next_emb + self.pos_embed(torch.tensor([step + 1], device=device))
            x = torch.cat([x, next_emb], dim=1)

        generated = torch.cat(generated, dim=1)
        return generated
