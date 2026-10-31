"""
evaluate.py
-----------
Evaluation script for Next-Day Wildfire Prediction on the 2023 Test Set.
Compares U-TAE against the standard Persistence Baseline (predicting fire[t+1] = fire[t]).

Metrics:
  - Intersection over Union (IoU / Jaccard Index)
  - F1-Score (Dice Coefficient)
  - Precision & Recall
  - Confusion Matrix (TP, FP, FN, TN)
  - Optimal threshold sweep
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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

from src.losses import CombinedLoss, MetricTracker, compute_metrics
from src.model import WildfireUTAE, get_wildfire_utae
from src.wildfire_dataset import get_dataloader


def plot_comparison_chart(
    model_metrics: Dict[str, float],
    persist_metrics: Dict[str, float],
    threshold_results: List[Dict[str, float]],
    save_path: str,
):
    """Generates comparison bar plot and threshold sweep curve."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Panel 1: Bar chart comparison at optimal / default threshold
    metrics_to_plot = ["iou", "f1", "precision", "recall"]
    labels = ["IoU", "F1-Score", "Precision", "Recall"]
    model_vals = [model_metrics[m] for m in metrics_to_plot]
    persist_vals = [persist_metrics[m] for m in metrics_to_plot]

    x = np.arange(len(labels))
    width = 0.35

    axes[0].bar(x - width / 2, model_vals, width, label="U-TAE Model", color="#4a90e2")
    axes[0].bar(x + width / 2, persist_vals, width, label="Persistence Baseline", color="#f5a623")
    axes[0].set_ylabel("Score", fontsize=11, fontweight="bold")
    axes[0].set_title("Test Set 2023: Model vs. Persistence Baseline", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    for i in range(len(labels)):
        axes[0].annotate(f"{model_vals[i]:.3f}", (x[i] - width / 2, model_vals[i] + 0.02), ha="center", fontsize=9, fontweight="bold")
        axes[0].annotate(f"{persist_vals[i]:.3f}", (x[i] + width / 2, persist_vals[i] + 0.02), ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Threshold sweep
    th_vals = [r["threshold"] for r in threshold_results]
    iou_vals = [r["iou"] for r in threshold_results]
    f1_vals = [r["f1"] for r in threshold_results]
    prec_vals = [r["precision"] for r in threshold_results]
    rec_vals = [r["recall"] for r in threshold_results]

    axes[1].plot(th_vals, iou_vals, "g-o", label="IoU")
    axes[1].plot(th_vals, f1_vals, "m--s", label="F1-Score")
    axes[1].plot(th_vals, prec_vals, "b-.", label="Precision")
    axes[1].plot(th_vals, rec_vals, "r:", label="Recall")
    axes[1].axhline(y=persist_metrics["f1"], color="#f5a623", linestyle="--", label=f"Persist F1 ({persist_metrics['f1']:.3f})")
    axes[1].set_xlabel("Decision Threshold", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Score", fontsize=11, fontweight="bold")
    axes[1].set_title("Metrics vs. Probability Threshold (2023 Test)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline comparison chart to {save_path}")


def evaluate_test_set(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"=" * 70)
    print(f"EVALUATING ON 2023 TEST SET (Device: {device})")
    print(f"=" * 70)

    # 1. Load Test DataLoader
    test_loader = get_dataloader(
        root_dir=args.data_dir,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    print(f"Test samples: {len(test_loader.dataset):,} (Batches: {len(test_loader)})")

    # 2. Load Model
    model = WildfireUTAE(input_dim=17).to(device)
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded trained weights from: {args.checkpoint}")
    else:
        print(f"Note: No checkpoint found at '{args.checkpoint}'. Running evaluation with current model weights.")

    model.eval()
    criterion = CombinedLoss(w_focal=1.0, w_dice=1.0)

    # Threshold sweep candidates
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    trackers = {th: MetricTracker(threshold=th) for th in thresholds}
    persist_tracker = MetricTracker(threshold=0.5)

    print("Running inference across test batches...")
    with torch.no_grad():
        for x, y, p, meta in tqdm(test_loader, desc="Testing", dynamic_ncols=True):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)

            logits = model(x)
            loss, _ = criterion(logits, y)

            # Update persistence baseline (p is already binary)
            persist_tracker.update(p, y, loss_val=0.0, is_logits=False)

            # Update model trackers across all thresholds
            for th in thresholds:
                trackers[th].update(logits, y, loss_val=loss.item(), is_logits=True)

    # Compute Persistence Baseline metrics
    persist_metrics = persist_tracker.compute()

    # Compute Model metrics for all thresholds
    sweep_results = []
    best_th = 0.5
    best_f1 = -1.0

    for th in thresholds:
        m = trackers[th].compute()
        sweep_results.append({
            "threshold": th,
            "iou": m["iou"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "loss": m["loss"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"],
        })
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th

    best_model_metrics = trackers[best_th].compute()
    def_model_metrics = trackers[0.5].compute()

    # Summary Output
    print(f"\n" + "=" * 70)
    print(f"EVALUATION RESULTS (2023 Test Set)")
    print(f"=" * 70)
    print(f"{'Method / Model':<30} | {'IoU':<8} | {'F1-Score':<8} | {'Precision':<10} | {'Recall':<8}")
    print(f"-" * 70)
    print(f"{'Persistence Baseline':<30} | {persist_metrics['iou']:<8.4f} | {persist_metrics['f1']:<8.4f} | {persist_metrics['precision']:<10.4f} | {persist_metrics['recall']:<8.4f}")
    print(f"{f'U-TAE (Threshold = 0.5)':<30} | {def_model_metrics['iou']:<8.4f} | {def_model_metrics['f1']:<8.4f} | {def_model_metrics['precision']:<10.4f} | {def_model_metrics['recall']:<8.4f}")
    print(f"{f'U-TAE (Optimal th = {best_th:.1f})':<30} | {best_model_metrics['iou']:<8.4f} | {best_model_metrics['f1']:<8.4f} | {best_model_metrics['precision']:<10.4f} | {best_model_metrics['recall']:<8.4f}")
    print(f"=" * 70)

    # Relative improvement
    eps = 1e-7
    rel_iou = ((best_model_metrics["iou"] - persist_metrics["iou"]) / (persist_metrics["iou"] + eps)) * 100.0
    rel_f1 = ((best_model_metrics["f1"] - persist_metrics["f1"]) / (persist_metrics["f1"] + eps)) * 100.0
    print(f"Performance Delta (Optimal U-TAE vs Persistence):")
    print(f"  Δ IoU     : {best_model_metrics['iou'] - persist_metrics['iou']:+.4f} ({rel_iou:+.1f}%)")
    print(f"  Δ F1-Score: {best_model_metrics['f1'] - persist_metrics['f1']:+.4f} ({rel_f1:+.1f}%)")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. baseline_comparison.csv
    comparison_df = pd.DataFrame([
        {"Method": "Persistence Baseline", "Threshold": "-", **persist_metrics},
        {"Method": "U-TAE (th=0.5)", "Threshold": 0.5, **def_model_metrics},
        {"Method": f"U-TAE (Optimal th={best_th})", "Threshold": best_th, **best_model_metrics},
    ])
    csv_path = os.path.join(args.output_dir, "baseline_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV comparison to {csv_path}")

    # 2. evaluation_report.json
    report = {
        "test_year": 2023,
        "num_samples": len(test_loader.dataset),
        "persistence_baseline": persist_metrics,
        "utae_default_th05": def_model_metrics,
        "utae_optimal": {
            "optimal_threshold": best_th,
            **best_model_metrics,
        },
        "threshold_sweep": sweep_results,
    }
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {report_path}")

    # 3. model_vs_baseline.png
    chart_path = os.path.join(args.output_dir, "model_vs_baseline.png")
    plot_comparison_chart(best_model_metrics, persist_metrics, sweep_results, chart_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wildfire U-TAE on 2023 Test Set")
    parser.add_argument("--data_dir", type=str, default=os.path.join(parent_dir, "data", "processed_v1"))
    parser.add_argument("--checkpoint", type=str, default=os.path.join(parent_dir, "checkpoints", "best_model.pth"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit test samples (e.g. for quick run)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=os.path.join(parent_dir, "outputs"))

    args = parser.parse_args()
    evaluate_test_set(args)


if __name__ == "__main__":
    main()

