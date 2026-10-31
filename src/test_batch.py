"""
test_batch.py
-------------
Sanity check script to verify one complete training batch:
  1. Real data loading from data/processed_v1/train
  2. Tensor shapes, dtypes, and value ranges
  3. NaN / Inf anomaly detection
  4. Forward pass through WildfireUTAE
  5. Loss computation with CombinedLoss (Focal + Dice)
  6. Backward pass and gradient validity check
  7. Optimizer step verification
"""

import os
import sys

import torch
import torch.optim as optim

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.losses import CombinedLoss, compute_metrics
from src.model import WildfireUTAE
from src.wildfire_dataset import WildfireDataset, get_dataloader


def run_batch_sanity_check(batch_size: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=" * 70)
    print(f"BATCH SANITY CHECK (Device: {device})")
    print(f"=" * 70)

    data_dir = os.path.join(parent_dir, "data", "processed_v1")
    print(f"[1/6] Loading real batch from: {data_dir} (split: train)...")

    loader = get_dataloader(
        root_dir=data_dir,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        cache_size=8,
    )

    batch_iter = iter(loader)
    x, y, p, meta = next(batch_iter)

    print(f"  ✓ Batch loaded successfully.")
    print(f"    - Input x shape : {x.shape} (expected: [{batch_size}, 5, 17, 128, 128])")
    print(f"    - Target y shape: {y.shape} (expected: [{batch_size}, 1, 128, 128])")
    print(f"    - Persist p shape: {p.shape} (expected: [{batch_size}, 1, 128, 128])")
    print(f"    - Target fire px : {y.sum().item()} pixels across batch")

    # [2/6] Check dimensions & dtypes
    print(f"\n[2/6] Verifying tensor dimensions and dtypes...")
    assert x.shape == (batch_size, 5, 17, 128, 128), f"Incorrect x shape: {x.shape}"
    assert y.shape == (batch_size, 1, 128, 128), f"Incorrect y shape: {y.shape}"
    assert x.dtype == torch.float32, f"Expected float32 for x, got {x.dtype}"
    assert y.dtype == torch.float32, f"Expected float32 for y, got {y.dtype}"
    print(f"  ✓ Dimensions and dtypes are verified.")

    # [3/6] Check for NaN and Inf
    print(f"\n[3/6] Checking for NaN and Inf values in inputs and targets...")
    has_nan_x = torch.isnan(x).any().item()
    has_inf_x = torch.isinf(x).any().item()
    has_nan_y = torch.isnan(y).any().item()
    has_inf_y = torch.isinf(y).any().item()

    assert not has_nan_x, "NaN detected in input tensor x!"
    assert not has_inf_x, "Inf detected in input tensor x!"
    assert not has_nan_y, "NaN detected in target tensor y!"
    assert not has_inf_y, "Inf detected in target tensor y!"
    print(f"  ✓ No NaN or Inf detected in input or target.")

    # [4/6] Forward pass
    print(f"\n[4/6] Initializing WildfireUTAE and testing forward pass...")
    model = WildfireUTAE(input_dim=17).to(device)
    x = x.to(device)
    y = y.to(device)

    logits, att = model(x, return_att=True)
    assert logits.shape == (batch_size, 1, 128, 128), f"Incorrect logits shape: {logits.shape}"
    assert not torch.isnan(logits).any().item(), "NaN detected in forward logits!"
    assert not torch.isinf(logits).any().item(), "Inf detected in forward logits!"

    probs = torch.sigmoid(logits)
    print(f"  ✓ Forward pass succeeded.")
    print(f"    - Output logits shape: {logits.shape}")
    print(f"    - Output probs range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"    - Attention maps shape: {att.shape}")

    # [5/6] Loss calculation
    print(f"\n[5/6] Computing CombinedLoss (Focal Loss + Soft Dice Loss)...")
    criterion = CombinedLoss(w_focal=1.0, w_dice=1.0)
    loss, loss_details = criterion(logits, y)

    assert not torch.isnan(loss).item(), "Computed loss is NaN!"
    assert not torch.isinf(loss).item(), "Computed loss is Inf!"
    assert loss.item() > 0.0, f"Loss should be positive, got {loss.item()}"
    print(f"  ✓ Loss computed successfully:")
    print(f"    - Total Loss: {loss.item():.4f}")
    print(f"    - Focal Loss: {loss_details['focal_loss']:.4f}")
    print(f"    - Dice Loss : {loss_details['dice_loss']:.4f}")

    # [6/6] Backward pass & optimizer step
    print(f"\n[6/6] Verifying backward pass and gradient backpropagation...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_ok = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"    ! Warning: Param {name} has no gradient.")
                grad_ok = False
            elif torch.isnan(param.grad).any().item() or torch.isinf(param.grad).any().item():
                print(f"    ! Error: Param {name} gradient contains NaN/Inf!")
                grad_ok = False

    assert grad_ok, "Gradient anomaly detected during backward pass!"
    optimizer.step()
    print(f"  ✓ Gradients valid and optimizer step executed successfully.")

    # Calculate metrics on this initial batch
    metrics = compute_metrics(logits, y, threshold=0.5)
    print(f"\nBatch Metrics Summary:")
    print(f"  - IoU      : {metrics['iou']:.4f}")
    print(f"  - F1-Score : {metrics['f1']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall   : {metrics['recall']:.4f}")
    print(f"=" * 70)
    print(f"ALL SANITY CHECKS PASSED SUCCESSFULLY!")
    print(f"=" * 70)


if __name__ == "__main__":
    run_batch_sanity_check(batch_size=4)

