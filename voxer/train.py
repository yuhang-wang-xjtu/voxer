"""
Training loops for all stages of the Voxer pipeline.

Stage 1: MAE pretraining - mask and reconstruct voxels
Stage 2: VQ-VAE training - discrete tokenization
Stage 3: GPT training - text-conditioned token generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import time
import os
import gc
from typing import Optional, Dict, Tuple, Callable

from voxer.utils import AverageMeter, set_seed


def train_mae(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 200,
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 10,
    device: str = "cuda",
    save_dir: str = "./checkpoints",
    log_interval: int = 20,
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
):
    """
    Train the 3D MAE model.

    Masked patches are reconstructed; loss is computed only on masked regions.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler() if mixed_precision else None
    best_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Stage 1: MAE Pretraining")
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Epochs: {epochs}, LR: {lr}, Warmup: {warmup_epochs}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        model.train()
        loss_meter = AverageMeter()
        step = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (tuple, list)):
                voxels = batch[0].to(device)
            else:
                voxels = batch.to(device)

            if mixed_precision:
                with autocast():
                    pred, mask, _ = model(voxels)

                    loss = (pred - voxels) ** 2
                    loss = loss.mean(dim=1)

                    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    mask_expanded = mask_expanded.expand_as(pred)
                    mask_3d = mask_expanded.mean(dim=1)

                    loss = (loss * mask_3d).sum() / (mask_3d.sum() + 1e-8)
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                pred, mask, _ = model(voxels)

                loss = (pred - voxels) ** 2
                loss = loss.mean(dim=1)

                mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                mask_expanded = mask_expanded.expand_as(pred)
                mask_3d = mask_expanded.mean(dim=1)

                loss = (loss * mask_3d).sum() / (mask_3d.sum() + 1e-8)
                loss = loss / gradient_accumulation_steps
                loss.backward()

            step += 1

            if step % gradient_accumulation_steps == 0:
                if mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            loss_meter.update(loss.item() * gradient_accumulation_steps)

            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if batch_idx % log_interval == 0 and batch_idx > 0:
                tqdm.write(
                    f"Epoch {epoch + 1} Batch {batch_idx}: "
                    f"loss={loss_meter.avg:.4f}"
                )

        avg_loss = loss_meter.avg
        print(f"Epoch {epoch + 1} finished. Avg loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(save_dir, f"mae_best.pt")
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "loss": avg_loss},
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")

        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(save_dir, f"mae_epoch_{epoch + 1}.pt")
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "loss": avg_loss},
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

    print(f"MAE training complete. Best loss: {best_loss:.4f}")
    return model


def train_vqvae(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    device: str = "cuda",
    save_dir: str = "./checkpoints",
    log_interval: int = 20,
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    recon_weight: float = 1.0,
    vq_weight: float = 1.0,
):
    """
    Train the VQ-VAE model.

    Loss = recon_weight * MSE(recon, original) + vq_weight * VQ_loss
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = GradScaler() if mixed_precision else None
    best_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Stage 2: VQ-VAE Training")
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Codebook size: {model.quantizer.num_embeddings}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        model.train()
        total_loss_meter = AverageMeter()
        recon_loss_meter = AverageMeter()
        vq_loss_meter = AverageMeter()
        step = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (tuple, list)):
                voxels = batch[0].to(device)
            else:
                voxels = batch.to(device)

            if mixed_precision:
                with autocast():
                    x_recon, vq_total, _, _ = model(voxels)

                    recon_loss = F.mse_loss(x_recon, voxels)
                    loss = recon_weight * recon_loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                x_recon, vq_total, _, _ = model(voxels)

                recon_loss = F.mse_loss(x_recon, voxels)
                loss = recon_weight * recon_loss / gradient_accumulation_steps
                loss.backward()

            step += 1

            if step % gradient_accumulation_steps == 0:
                if mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            recon_loss_meter.update(recon_loss.item())
            vq_loss_meter.update(vq_total.item())
            total_loss_meter.update(recon_loss.item())

            codebook_usage = model.get_codebook_usage().item()
            pbar.set_postfix({
                "recon": f"{recon_loss_meter.avg:.4f}",
                "vq_code_use": f"{codebook_usage:.2%}",
            })

            if batch_idx % log_interval == 0 and batch_idx > 0:
                tqdm.write(
                    f"Epoch {epoch + 1} Batch {batch_idx}: "
                    f"recon={recon_loss_meter.avg:.4f}, "
                    f"codebook_usage={codebook_usage:.2%}"
                )

        avg_loss = total_loss_meter.avg
        print(
            f"Epoch {epoch + 1} finished. "
            f"Avg recon loss: {avg_loss:.4f}, "
            f"Codebook usage: {model.get_codebook_usage().item():.2%}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(save_dir, f"vqvae_best.pt")
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "loss": avg_loss},
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"vqvae_epoch_{epoch + 1}.pt")
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "loss": avg_loss},
                checkpoint_path,
            )

    print(f"VQ-VAE training complete. Best recon loss: {best_loss:.4f}")
    return model


def train_generator(
    model: nn.Module,
    vqvae: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 300,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_epochs: int = 10,
    device: str = "cuda",
    save_dir: str = "./checkpoints",
    log_interval: int = 20,
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
):
    """
    Train the autoregressive transformer for text-conditioned token generation.

    The VQ-VAE is frozen and used to encode voxels to tokens.
    The GPT predicts tokens autoregressively given text embeddings.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    vqvae = vqvae.to(device)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.9, 0.95))

    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler() if mixed_precision else None
    best_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Stage 3: Text-Conditioned Token Generator Training")
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Vocab size: {model.vocab_size}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        step = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                voxels = batch[0].to(device)
                text_emb = batch[1].to(device)
            else:
                continue

            with torch.no_grad():
                token_ids = vqvae.encode(voxels)

            tokens = token_ids.reshape(token_ids.shape[0], -1)

            if mixed_precision:
                with autocast():
                    logits = model(tokens, text_emb)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        tokens.reshape(-1),
                        ignore_index=-1,
                    )
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                logits = model(tokens, text_emb)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tokens.reshape(-1),
                    ignore_index=-1,
                )
                loss = loss / gradient_accumulation_steps
                loss.backward()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == tokens).float().mean()

            step += 1

            if step % gradient_accumulation_steps == 0:
                if mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            loss_meter.update(loss.item() * gradient_accumulation_steps)
            acc_meter.update(acc.item())

            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.3f}",
            })

            if batch_idx % log_interval == 0 and batch_idx > 0:
                tqdm.write(
                    f"Epoch {epoch + 1} Batch {batch_idx}: "
                    f"loss={loss_meter.avg:.4f}, acc={acc_meter.avg:.3f}"
                )

        avg_loss = loss_meter.avg
        print(f"Epoch {epoch + 1} finished. Loss: {avg_loss:.4f}, Acc: {acc_meter.avg:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(save_dir, f"generator_best.pt")
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "loss": avg_loss},
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")

        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(save_dir, f"generator_epoch_{epoch + 1}.pt")
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "loss": avg_loss},
                checkpoint_path,
            )

    print(f"Generator training complete. Best loss: {best_loss:.4f}")
    return model
