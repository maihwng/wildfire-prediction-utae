"""
train.py
--------
Training script for Wildfire Prediction with U-TAE.
Tracks train/validation loss, IoU, F1, Precision, and Recall.
Saves best checkpoint to checkpoints/ and metrics history to outputs/.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src modules are imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.losses import CombinedLoss, MetricTracker
from src.model import WildfireUTAE
from src.wildfire_dataset import WildfireDataset, get_dataloader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_curves(history: Dict[str, List[float]], output_path: str):
    """Generates and saves clean multi-panel training plots."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss Curve
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    axes[0].set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Combined Loss (Focal + Dice)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # 2. IoU and F1-Score
    axes[1].plot(epochs, history["val_iou"], "g-o", label="Val IoU")
    axes[1].plot(epochs, history["val_f1"], "m--^", label="Val F1-Score")
    axes[1].set_title("Segmentation Overlap (IoU & F1)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # 3. Precision & Recall
    axes[2].plot(epochs, history["val_precision"], "c-o", label="Val Precision")
    axes[2].plot(epochs, history["val_recall"], "y--d", label="Val Recall")
    axes[2].set_title("Detection Balance (Precision & Recall)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Rate")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {output_path}")


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)
    for x, y, _, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision support on CUDA
        use_amp = (device.type == "cuda" and scaler is not None)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss, _ = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    tracker = MetricTracker(threshold=threshold)

    pbar = tqdm(dataloader, desc="Validation", leave=False, dynamic_ncols=True)
    for x, y, _, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss, _ = criterion(logits, y)
        tracker.update(logits, y, loss_val=loss.item())

    return tracker.compute()


def run_training(args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"=" * 70)
    print(f"WILDFIRE U-TAE TRAINING (Device: {device})")
    print(f"=" * 70)

    # Directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data loaders
    print(f"Loading datasets from {args.data_dir}...")
    train_loader = get_dataloader(
        root_dir=args.data_dir,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_samples=args.max_train_samples,
    )
    val_loader = get_dataloader(
        root_dir=args.data_dir,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_samples=args.max_val_samples,
    )
    print(f"Train samples: {len(train_loader.dataset):,} (Batches: {len(train_loader)})")
    print(f"Val samples  : {len(val_loader.dataset):,} (Batches: {len(val_loader)})")

    # Model
    model = WildfireUTAE(input_dim=17).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized WildfireUTAE model: {total_params:,} parameters")

    # Loss and Optimizer
    criterion = CombinedLoss(
        w_focal=args.focal_weight,
        w_dice=args.dice_weight,
        focal_alpha=0.75,
        focal_gamma=2.0,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Tracking metrics
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "lr": [],
        "epoch_time": [],
    }

    best_val_f1 = -1.0
    best_val_iou = -1.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_model.pth")

    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device, threshold=args.threshold)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_iou"].append(float(val_metrics["iou"]))
        history["val_f1"].append(float(val_metrics["f1"]))
        history["val_precision"].append(float(val_metrics["precision"]))
        history["val_recall"].append(float(val_metrics["recall"]))
        history["lr"].append(float(current_lr))
        history["epoch_time"].append(float(epoch_time))

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Rec: {val_metrics['recall']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save checkpoint if best F1 achieved
        if val_metrics["f1"] > best_val_f1 or (val_metrics["f1"] == best_val_f1 and val_metrics["iou"] > best_val_iou):
            best_val_f1 = val_metrics["f1"]
            best_val_iou = val_metrics["iou"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_val_f1,
                    "val_iou": best_val_iou,
                    "val_loss": val_metrics["loss"],
                    "config": vars(args),
                },
                best_checkpoint_path,
            )
            print(f"  ★ New best model saved to {best_checkpoint_path} (F1: {best_val_f1:.4f}, IoU: {best_val_iou:.4f})")

        # Always save latest
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_metrics["f1"],
                "val_iou": val_metrics["iou"],
                "config": vars(args),
            },
            latest_checkpoint_path,
        )

    total_time = time.time() - start_total_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation F1-Score: {best_val_f1:.4f} (IoU: {best_val_iou:.4f})")

    # Save history and curves
    history_json_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_json_path}")

    curves_png_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(history, curves_png_path)

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Wildfire U-TAE Model")
    parser.add_argument("--data_dir", type=str, default=os.path.join(parent_dir, "data", "processed_v1"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--focal_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples (e.g. for quick run)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit val samples")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(parent_dir, "checkpoints"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(parent_dir, "outputs"))

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

