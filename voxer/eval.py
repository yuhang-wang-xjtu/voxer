"""
Evaluation and visualization utilities for Voxer pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os


def evaluate_reconstruction(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    max_batches: int = 10,
) -> dict:
    """
    Evaluate VQ-VAE reconstruction quality.
    Returns IoU, MSE, and per-sample metrics.
    """
    model.eval()
    model = model.to(device)

    ious = []
    mses = []
    codebook_usage = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break

            if isinstance(batch, (tuple, list)):
                voxels = batch[0].to(device)
            else:
                voxels = batch.to(device)

            x_recon, _, indices, _ = model(voxels)

            mse = F.mse_loss(x_recon, voxels).item()
            mses.append(mse)

            gt_occ = (voxels[:, 3] > 0.5).float().reshape(voxels.shape[0], -1)
            recon_occ = (x_recon[:, 3] > 0.5).float().reshape(voxels.shape[0], -1)

            intersection = (gt_occ * recon_occ).sum(dim=1)
            union = (gt_occ + recon_occ).clamp(0, 1).sum(dim=1)
            iou = (intersection / (union + 1e-8)).mean().item()
            ious.append(iou)

            if hasattr(model, 'get_codebook_usage'):
                codebook_usage.append(model.get_codebook_usage().item())

    results = {
        "mse": np.mean(mses),
        "iou": np.mean(ious),
    }
    if codebook_usage:
        results["codebook_usage"] = np.mean(codebook_usage)

    return results


def visualize_voxel(
    voxel: torch.Tensor,
    title: str = "Voxel",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    threshold: float = 0.3,
):
    """
    Visualize a single 3D voxel grid.
    voxel: [4, D, H, W] or [3, D, H, W] or [1, D, H, W]
    """
    if isinstance(voxel, torch.Tensor):
        voxel = voxel.detach().cpu().numpy()

    if voxel.shape[0] == 4:
        rgb = voxel[:3]
        occ = voxel[3]
        rgb = rgb.clip(0, 1)
    elif voxel.shape[0] == 3:
        rgb = voxel
        occ = np.ones_like(voxel[0])
    elif voxel.shape[0] == 1:
        rgb = np.stack([voxel[0]] * 3, axis=0)
        occ = voxel[0]
    else:
        raise ValueError(f"Unexpected voxel shape: {voxel.shape}")

    occ_bool = occ > threshold

    rgb_flat = rgb.transpose(1, 2, 3, 0)
    colors = np.zeros(list(rgb_flat.shape[:-1]) + [4])
    colors[..., :3] = rgb_flat
    colors[..., 3] = occ

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    ax.voxels(occ_bool, facecolors=colors, edgecolor=None)
    ax.set_title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_reconstruction_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    title: str = "Original vs Reconstructed",
    save_path: Optional[str] = None,
):
    """Side-by-side comparison of original and reconstructed voxels."""
    fig = plt.figure(figsize=(16, 8))

    for i, (vol, label) in enumerate([(original, "Original"), (reconstructed, "Reconstructed")]):
        if isinstance(vol, torch.Tensor):
            vol = vol.detach().cpu().numpy()

        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        if vol.shape[0] == 4:
            rgb = vol[:3].clip(0, 1)
            occ = vol[3] > 0.3
        else:
            rgb = vol.clip(0, 1)
            occ = np.ones_like(vol[0]) > 0.5

        rgb_flat = rgb.transpose(1, 2, 3, 0)
        colors = np.zeros(list(rgb_flat.shape[:-1]) + [4])
        colors[..., :3] = rgb_flat
        colors[..., 3] = occ

        ax.voxels(occ, facecolors=colors, edgecolor=None)
        ax.set_title(label)

    fig.suptitle(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU for occupancy prediction."""
    pred_occ = (pred > threshold).float()
    target_occ = (target > threshold).float()

    intersection = (pred_occ * target_occ).sum()
    union = (pred_occ + target_occ).clamp(0, 1).sum()

    return (intersection / (union + 1e-8)).item()


def generate_and_visualize(
    gpt_model: torch.nn.Module,
    vqvae: torch.nn.Module,
    text_emb: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 100,
    top_p: float = 0.95,
    device: str = "cuda",
    save_path: Optional[str] = None,
    num_samples: int = 1,
) -> list:
    """
    Generate multiple voxel samples from a text embedding and visualize them.
    """
    gpt_model.eval()
    vqvae.eval()

    if text_emb.dim() == 1:
        text_emb = text_emb.unsqueeze(0)

    text_emb = text_emb.to(device).float()

    results = []
    for i in range(num_samples):
        with torch.no_grad():
            token_ids = gpt_model.generate(
                text_emb=text_emb,
                max_tokens=512,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        if token_ids.shape[1] > 512:
            token_ids = token_ids[:, :512]

        with torch.no_grad():
            voxel = vqvae.decode(token_ids)

        results.append({
            "tokens": token_ids.cpu(),
            "voxel": voxel.cpu(),
        })

        if num_samples <= 5:
            visualize_voxel(
                voxel[0],
                title=f"Generated Sample {i + 1}",
                save_path=None if save_path is None else f"{save_path}_sample_{i}.png",
            )

    return results
