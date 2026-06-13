import torch
import torch.nn as nn
import math
import random
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_3d_sinusoidal_positional_encoding(num_patches: int, embed_dim: int):
    pe = torch.zeros(num_patches, embed_dim)
    position = torch.arange(num_patches).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def get_3d_positional_encoding_grid(depth: int, height: int, width: int, embed_dim: int):
    num_patches = depth * height * width
    pe = torch.zeros(num_patches, embed_dim)

    z_pos = torch.arange(depth).repeat_interleave(height * width)
    y_pos = torch.arange(height).repeat(width).repeat(depth)
    x_pos = torch.arange(width).repeat(depth * height)

    div_term = torch.exp(
        torch.arange(0, embed_dim // 3, 2).float()
        * (-math.log(10000.0) / (embed_dim // 3))
    )

    offset = embed_dim // 3
    for i, pos in enumerate([z_pos, y_pos, x_pos]):
        pe[:, offset * i: offset * i + offset] = torch.stack([
            torch.sin(pos.float().unsqueeze(1) * div_term),
            torch.cos(pos.float().unsqueeze(1) * div_term),
        ], dim=-1).reshape(-1, offset)

    return pe


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    else:
        return f"{n / 1e3:.2f}K"


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data
