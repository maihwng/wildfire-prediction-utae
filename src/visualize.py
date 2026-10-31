"""
visualize.py
------------
Visualization utilities for spatio-temporal wildfire spread prediction.
Plots multi-day fire progression, weather rasters, probability heatmaps,
error analysis (TP, FP, FN), and temporal attention weights.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import torch

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

from src.model import WildfireUTAE, get_wildfire_utae
from src.wildfire_dataset import WildfireDataset, CHANNEL_NAMES


def plot_prediction_comparison(
    x: torch.Tensor,
    y_true: torch.Tensor,
    y_pred_prob: torch.Tensor,
    persistence: torch.Tensor,
    attn_weights: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    threshold: float = 0.5,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Creates a comprehensive 2x4 panel figure:
      Row 1: 5-Day Historical Fire Progression + Day 5 Ground Truth + Model Prob Map + Model Binary Pred
      Row 2: Temperature raster, Wind raster, Persistence baseline, Error Confusion Map (TP, FP, FN)
    """
    # Convert tensors to 2D numpy arrays
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.detach().cpu().numpy()
    if isinstance(persistence, torch.Tensor):
        persistence = persistence.detach().cpu().numpy()

    # Squeeze extra channel dims
    if y_true.ndim == 3:
        y_true = y_true[0]
    if y_pred_prob.ndim == 3:
        y_pred_prob = y_pred_prob[0]
    if persistence.ndim == 3:
        persistence = persistence[0]

    y_pred_binary = (y_pred_prob >= threshold).astype(np.float32)

    # Compute Confusion Map:
    # 0: TN (Background), 1: TP (Green), 2: FP (Blue), 3: FN (Red)
    confusion = np.zeros_like(y_true, dtype=np.uint8)
    tp_mask = (y_pred_binary == 1) & (y_true == 1)
    fp_mask = (y_pred_binary == 1) & (y_true == 0)
    fn_mask = (y_pred_binary == 0) & (y_true == 1)

    confusion[tp_mask] = 1
    confusion[fp_mask] = 2
    confusion[fn_mask] = 3

    cmap_conf = ListedColormap(["#f0f0f0", "#2ca02c", "#1f77b4", "#d62728"])

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    # 1. Day t-4 Fire
    axes[0, 0].imshow(x[0, 13], cmap="YlOrRd", vmin=0, vmax=1)
    axes[0, 0].set_title("Input Day 1 (t-4) Fire", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Day t-2 Fire
    axes[0, 1].imshow(x[2, 13], cmap="YlOrRd", vmin=0, vmax=1)
    axes[0, 1].set_title("Input Day 3 (t-2) Fire", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    # 3. Day t (Persistence) Fire
    axes[0, 2].imshow(persistence, cmap="YlOrRd", vmin=0, vmax=1)
    fire_px_last = int(np.sum(persistence))
    axes[0, 2].set_title(f"Input Day 5 (t) [Persist]\n({fire_px_last} fire px)", fontsize=10, fontweight="bold")
    axes[0, 2].axis("off")

    # 4. Ground Truth Target (Day t+1)
    axes[0, 3].imshow(y_true, cmap="YlOrRd", vmin=0, vmax=1)
    fire_px_gt = int(np.sum(y_true))
    axes[0, 3].set_title(f"Ground Truth (Day t+1)\n({fire_px_gt} fire px)", fontsize=10, fontweight="bold")
    axes[0, 3].axis("off")

    # 5. Temperature Raster (Channel 1, day t)
    im_temp = axes[1, 0].imshow(x[-1, 1], cmap="coolwarm")
    axes[1, 0].set_title("Day t: Max Temp (tmmx)", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")
    fig.colorbar(im_temp, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 6. Model Probability Heatmap
    im_prob = axes[1, 1].imshow(y_pred_prob, cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title("U-TAE Fire Probability", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")
    fig.colorbar(im_prob, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 7. Model Binary Prediction
    axes[1, 2].imshow(y_pred_binary, cmap="YlOrRd", vmin=0, vmax=1)
    fire_px_pred = int(np.sum(y_pred_binary))
    axes[1, 2].set_title(f"U-TAE Binary Mask (th={threshold})\n({fire_px_pred} fire px)", fontsize=10, fontweight="bold")
    axes[1, 2].axis("off")

    # 8. Error / Confusion Map
    axes[1, 3].imshow(confusion, cmap=cmap_conf, vmin=0, vmax=3)
    axes[1, 3].set_title("Prediction Error Map", fontsize=10, fontweight="bold")
    axes[1, 3].axis("off")

    # Custom legend for confusion map
    patches = [
        mpatches.Patch(color="#2ca02c", label=f"TP: {np.sum(tp_mask)}"),
        mpatches.Patch(color="#1f77b4", label=f"FP: {np.sum(fp_mask)}"),
        mpatches.Patch(color="#d62728", label=f"FN: {np.sum(fn_mask)}"),
    ]
    axes[1, 3].legend(handles=patches, loc="lower right", fontsize=8)

    suptitle = f"Wildfire Prediction Analysis: Spatio-Temporal U-TAE {title_suffix}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    return fig


def plot_temporal_attention(
    attn_weights: np.ndarray,
    save_path: Optional[str] = None,
    sample_info: str = "",
) -> plt.Figure:
    """
    Plots temporal attention distribution across the 5 input days.
    attn_weights: 1D array of length 5 summing to 1.0.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    days = ["Day 1 (t-4)", "Day 2 (t-3)", "Day 3 (t-2)", "Day 4 (t-1)", "Day 5 (t)"]
    colors = ["#4a90e2", "#50e3c2", "#b8e986", "#f5a623", "#d0021b"]

    bars = ax.bar(days, attn_weights, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Attention Weight", fontsize=11, fontweight="bold")
    ax.set_title(f"L-TAE Temporal Attention Distribution {sample_info}", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(0.4, np.max(attn_weights) * 1.25))
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h*100:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved attention plot to {save_path}")

    return fig


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing visualize.py on {device}...")

    data_dir = os.path.join(parent_dir, "data", "processed_v1")
    dataset = WildfireDataset(root_dir=data_dir, split="test", max_samples=50, filter_active_fire=True)
    print(f"Test samples with active fire: {len(dataset)}")

    x, y, p, meta = dataset[0]
    model = WildfireUTAE(input_dim=17).to(device)

    # Checkpoint if available
    ckpt = os.path.join(parent_dir, "checkpoints", "best_model.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        print(f"Loaded weights from {ckpt}")

    x_tensor = x.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits, att = model(x_tensor, return_att=True)
        probs = torch.sigmoid(logits)

    attn_summary = WildfireUTAE.get_attention_weights(att)[0].detach().cpu().numpy()

    out_viz = os.path.join(parent_dir, "outputs", "sample_prediction_viz.png")
    out_attn = os.path.join(parent_dir, "outputs", "sample_temporal_attention.png")

    plot_prediction_comparison(x, y, probs[0], p, save_path=out_viz, threshold=0.5)
    plot_temporal_attention(attn_summary, save_path=out_attn, sample_info=f"({meta['file']})")
    print("visualize.py verification successful!")

