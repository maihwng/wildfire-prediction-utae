"""
losses.py
---------
Loss functions and evaluation metrics for Wildfire Spread Prediction.
Designed for extreme class imbalance (sparse active fire pixels).

Includes:
  - FocalLoss: Focuses learning on hard fire pixels and suppresses easy background.
  - DiceLoss: Directly maximizes spatial overlap / F1-score of fire perimeters.
  - CombinedLoss: Weighted combination of Focal Loss and Dice Loss.
  - MetricTracker: Accumulates exact confusion matrix elements for IoU, F1, Precision, Recall.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss with numerical stability via binary cross entropy with logits.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logits of shape (B, 1, H, W)
            targets: Binary ground truth masks of shape (B, 1, H, W)
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Dice = (2 * |P cap T| + smooth) / (|P|^2 + |T|^2 + smooth)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0, square: bool = True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.square = square

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        if self.square:
            cardinality = (probs_flat * probs_flat).sum() + (targets_flat * targets_flat).sum()
        else:
            cardinality = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Weighted combination of Focal Loss and Soft Dice Loss.
    Loss = w_focal * FocalLoss + w_dice * DiceLoss
    """

    def __init__(
        self,
        w_focal: float = 1.0,
        w_dice: float = 1.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
    ):
        super(CombinedLoss, self).__init__()
        self.w_focal = w_focal
        self.w_dice = w_dice
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss(smooth=dice_smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_focal = self.focal(logits, targets)
        l_dice = self.dice(logits, targets)
        total_loss = self.w_focal * l_focal + self.w_dice * l_dice

        loss_dict = {
            "loss": total_loss.item(),
            "focal_loss": l_focal.item(),
            "dice_loss": l_dice.item(),
        }
        return total_loss, loss_dict


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    is_logits: bool = True,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Computes IoU, F1, Precision, and Recall.
    Args:
        pred: Predicted tensor (logits or probabilities) of shape (B, 1, H, W)
        target: Ground truth mask (0.0 or 1.0) of shape (B, 1, H, W)
        threshold: Binarization cutoff
        is_logits: Whether pred contains unnormalized logits
    """
    if is_logits:
        probs = torch.sigmoid(pred)
    else:
        probs = pred

    binary_pred = (probs >= threshold).float()
    target = (target >= 0.5).float()

    tp = (binary_pred * target).sum().item()
    fp = (binary_pred * (1.0 - target)).sum().item()
    fn = ((1.0 - binary_pred) * target).sum().item()
    tn = ((1.0 - binary_pred) * (1.0 - target)).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


class MetricTracker:
    """
    Accumulates exact True Positive, False Positive, and False Negative counts
    across batches to compute true dataset-level IoU, F1, Precision, and Recall.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.total_pixels = 0
        self.total_loss = 0.0
        self.num_batches = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_val: Optional[float] = None,
        is_logits: bool = True,
    ):
        with torch.no_grad():
            if is_logits:
                probs = torch.sigmoid(pred)
            else:
                probs = pred

            binary_pred = (probs >= self.threshold).float()
            target_bin = (target >= 0.5).float()

            self.tp += (binary_pred * target_bin).sum().item()
            self.fp += (binary_pred * (1.0 - target_bin)).sum().item()
            self.fn += ((1.0 - binary_pred) * target_bin).sum().item()
            self.tn += ((1.0 - binary_pred) * (1.0 - target_bin)).sum().item()
            self.total_pixels += target.numel()

            if loss_val is not None:
                self.total_loss += loss_val
                self.num_batches += 1

    def compute(self, eps: float = 1e-7) -> Dict[str, float]:
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        avg_loss = (self.total_loss / self.num_batches) if self.num_batches > 0 else 0.0

        return {
            "loss": float(avg_loss),
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "tn": int(self.tn),
        }


if __name__ == "__main__":
    print("Testing losses.py...")
    criterion = CombinedLoss()

    logits = torch.randn(4, 1, 128, 128, requires_grad=True)
    targets = (torch.rand(4, 1, 128, 128) > 0.95).float()

    loss, details = criterion(logits, targets)
    print(f"Computed loss: {loss.item():.4f}, details: {details}")

    loss.backward()
    print("Backward pass through loss successful! Gradient computed.")

    tracker = MetricTracker(threshold=0.5)
    tracker.update(logits, targets, loss_val=loss.item())
    metrics = tracker.compute()
    print("Metrics computed successfully:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

