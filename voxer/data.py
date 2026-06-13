"""
Data loading and preprocessing for voxel 3D models.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple, Optional, List
import os
import json


class VoxelDataset(Dataset):
    """
    Dataset for 3D voxel models with optional text descriptions.

    Each sample is a 64x64x64x4 (RGBA) voxel grid.
    Text descriptions are stored as precomputed embeddings.
    """

    def __init__(
        self,
        voxel_path: str,
        text_embed_path: Optional[str] = None,
        transform: bool = True,
    ):
        data = np.load(voxel_path)
        self.voxels = torch.from_numpy(data).float()

        if self.voxels.dim() == 5:
            pass
        else:
            self.voxels = self.voxels.unsqueeze(0)

        self.text_embeds = None
        if text_embed_path and os.path.exists(text_embed_path):
            self.text_embeds = torch.load(text_embed_path)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.voxels)

    def __getitem__(self, idx: int):
        voxel = self.voxels[idx].clone()

        if voxel.shape[0] == 4 and voxel.shape[1] == 64:
            pass
        elif voxel.shape[-1] == 4:
            voxel = voxel.permute(3, 0, 1, 2)
        else:
            pass

        if self.transform:
            voxel = self._augment(voxel)

        voxel[:3] = voxel[:3] / 255.0
        voxel[3] = voxel[3] / 255.0
        voxel[3] = voxel[3].clamp(0, 1)

        if self.text_embeds is not None and idx < len(self.text_embeds):
            text_emb = self.text_embeds[idx]
            return voxel, text_emb
        else:
            return voxel

    def _augment(self, voxel: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) > 0.5:
            k = torch.randint(0, 4, (1,)).item()
            voxel[:3] = torch.rot90(voxel[:3], k, dims=[1, 2])

        if torch.rand(1) > 0.5:
            voxel = torch.flip(voxel, dims=[1])

        if torch.rand(1) > 0.5:
            voxel = torch.flip(voxel, dims=[2])

        if torch.rand(1) > 0.5:
            voxel = torch.flip(voxel, dims=[3])

        if torch.rand(1) > 0.5:
            scale = 0.9 + torch.rand(1).item() * 0.2
            old_size = voxel.shape[-1]
            new_size = int(old_size * scale)
            if new_size % 2 != old_size % 2:
                new_size += 1

            temp = torch.nn.functional.interpolate(
                voxel.unsqueeze(0), size=new_size, mode="trilinear", align_corners=False
            ).squeeze(0)

            if new_size > old_size:
                start = (new_size - old_size) // 2
                voxel_large = torch.zeros(voxel.shape[0], new_size, new_size, new_size)
                voxel_large[:, start:start + old_size, start:start + old_size, start:start + old_size] = voxel
                voxel = voxel_large
                voxel = torch.nn.functional.interpolate(
                    voxel.unsqueeze(0), size=old_size, mode="trilinear", align_corners=False
                ).squeeze(0)
            else:
                pad_total = old_size - new_size
                pad_start = pad_total // 2
                voxel_padded = torch.zeros(voxel.shape[0], old_size, old_size, old_size)
                voxel_padded[:, pad_start:pad_start + new_size, pad_start:pad_start + new_size, pad_start:pad_start + new_size] = temp
                voxel = voxel_padded

        return voxel


class PreprocessedVoxelDataset(Dataset):
    """
    Dataset that loads preprocessed numpy arrays directly.
    """

    def __init__(self, data: np.ndarray, text_embeds: Optional[np.ndarray] = None):
        self.data = torch.from_numpy(data).float()
        self.text_embeds = (
            torch.from_numpy(text_embeds).float()
            if text_embeds is not None
            else None
        )

        if self.data.dim() == 4:
            self.data = self.data.unsqueeze(0)
        if self.data.shape[1] != 4 and self.data.shape[-1] == 4:
            self.data = self.data.permute(0, 4, 1, 2, 3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        voxel = self.data[idx]
        voxel[:3] = voxel[:3] / 255.0
        voxel[3] = voxel[3] / 255.0

        if self.text_embeds is not None:
            return voxel, self.text_embeds[idx]
        return voxel


def create_dataloaders(
    data: np.ndarray,
    text_embeds: Optional[np.ndarray] = None,
    batch_size: int = 16,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
):
    dataset = PreprocessedVoxelDataset(data, text_embeds)
    total = len(dataset)

    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def save_voxel_data(voxels: np.ndarray, path: str):
    np.save(path, voxels)


def load_voxel_data(path: str) -> np.ndarray:
    return np.load(path)
