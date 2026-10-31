"""
test_model.py
-------------
Kiểm tra mô hình Spatio-Temporal U-TAE:
1. Khởi tạo mô hình WildfireUTAE với 17 kênh đầu vào.
2. Kiểm tra forward pass: nhận tensor (B, T=5, C=17, H=128, W=128), trả về logits (B, 1, 128, 128).
3. Kiểm tra tính hợp lệ của xác suất Sigmoid (nằm trong khoảng [0, 1]).
4. Kiểm tra các hàm mất mát (Focal Loss, Dice Loss, Combined Loss).
5. Kiểm tra khả năng lan truyền ngược backward pass (tính gradient).
6. Kiểm tra tải trọng số checkpoint best_model.pth nếu có.
"""

import os
import sys

import pytest
import torch

# Đảm bảo đường dẫn gốc của dự án nằm trong sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.losses import CombinedLoss, DiceLoss, FocalLoss
from src.model import WildfireUTAE

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")


@pytest.fixture(scope="module")
def model_fixture() -> WildfireUTAE:
    """Fixture khởi tạo mô hình WildfireUTAE dùng chung cho module test."""
    model = WildfireUTAE(input_dim=17)
    model.eval()
    return model


class TestWildfireUTAEModel:
    """Tập hợp các bài kiểm thử cho kiến trúc U-TAE và hàm Loss."""

    def test_model_instantiation(self, model_fixture):
        """Kiểm tra khởi tạo thành công kiến trúc mạng nơ-ron."""
        assert isinstance(model_fixture, torch.nn.Module)
        total_params = sum(p.numel() for p in model_fixture.parameters())
        assert total_params > 100_000, f"Số lượng tham số quá nhỏ: {total_params}"

    def test_forward_pass_output_shape(self, model_fixture):
        """
        Kiểm tra kích thước đầu ra:
        Đầu vào: (B=2, T=5, C=17, H=128, W=128)
        Đầu ra: (B=2, 1, H=128, W=128)
        """
        x = torch.randn(2, 5, 17, 128, 128)
        with torch.no_grad():
            logits = model_fixture(x)

        assert logits.shape == (2, 1, 128, 128), f"Kích thước logits không đúng: {logits.shape}"

    def test_forward_pass_no_nan_or_inf(self, model_fixture):
        """Đảm bảo forward pass không sinh ra giá trị NaN hoặc Inf."""
        x = torch.randn(2, 5, 17, 128, 128)
        with torch.no_grad():
            logits = model_fixture(x)

        assert not torch.isnan(logits).any(), "Logits chứa giá trị NaN"
        assert not torch.isinf(logits).any(), "Logits chứa giá trị Inf"

    def test_sigmoid_probability_range(self, model_fixture):
        """Kiểm tra xác suất qua hàm sigmoid nằm chặt chẽ trong khoảng [0.0, 1.0]."""
        x = torch.randn(1, 5, 17, 128, 128)
        with torch.no_grad():
            logits = model_fixture(x)
            probs = torch.sigmoid(logits)

        assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Xác suất vượt ra ngoài phạm vi [0, 1]"

    def test_loss_functions_computation(self):
        """Kiểm tra tính toán của 3 hàm Loss: Focal Loss, Dice Loss, Combined Loss."""
        logits = torch.randn(2, 1, 128, 128)
        targets = torch.randint(0, 2, (2, 1, 128, 128)).float()

        # 1. Focal Loss
        focal_fn = FocalLoss(alpha=0.75, gamma=2.0)
        focal_loss = focal_fn(logits, targets)
        assert focal_loss.dim() == 0, "Focal Loss phải là tensor vô hướng (scalar)"
        assert not torch.isnan(focal_loss) and not torch.isinf(focal_loss)
        assert focal_loss.item() >= 0.0

        # 2. Dice Loss
        dice_fn = DiceLoss()
        dice_loss = dice_fn(logits, targets)
        assert dice_loss.dim() == 0
        assert not torch.isnan(dice_loss) and not torch.isinf(dice_loss)
        assert 0.0 <= dice_loss.item() <= 1.0

        # 3. Combined Loss
        combined_fn = CombinedLoss(w_focal=1.0, w_dice=1.0)
        total_loss, loss_dict = combined_fn(logits, targets)
        assert total_loss.dim() == 0
        assert not torch.isnan(total_loss) and not torch.isinf(total_loss)
        assert "focal_loss" in loss_dict and "dice_loss" in loss_dict

    def test_backward_gradient_flow(self):
        """Kiểm tra khả năng tính đạo hàm và lan truyền ngược gradient."""
        model = WildfireUTAE(input_dim=17)
        model.train()

        x = torch.randn(1, 5, 17, 128, 128, requires_grad=True)
        targets = torch.randint(0, 2, (1, 1, 128, 128)).float()

        logits = model(x)
        loss_fn = CombinedLoss()
        loss, _ = loss_fn(logits, targets)
        loss.backward()

        # Kiểm tra tham số nhận được gradient hợp lệ
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "Không có gradient nào được tính"
        for g in grads:
            assert not torch.isnan(g).any(), "Gradient chứa giá trị NaN"

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason="Bỏ qua nếu chưa có tệp checkpoint best_model.pth",
    )
    def test_load_checkpoint_weights(self):
        """Kiểm tra nạp trọng số đã huấn luyện từ checkpoint."""
        model = WildfireUTAE(input_dim=17)
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        # Thử nghiệm forward pass với trọng số đã nạp
        x = torch.randn(1, 5, 17, 128, 128)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 1, 128, 128)
