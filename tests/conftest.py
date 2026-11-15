import os
import sys
import hashlib
import pytest
import numpy as np
import torch

# Ensure Wildfire-prediction-utae is on sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import WildfireUTAE


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def sample_valid_input():
    """Tạo batch đầu vào chuẩn (B=1, T=5, C=17, H=128, W=128)."""
    torch.manual_seed(42)
    # 17 channels: 0..12 dyn_cont, 13 fire, 14 frp, 15 w_mask, 16 v_mask
    x = torch.randn(1, 5, 17, 128, 128, dtype=torch.float32)
    # Fire mask binary
    x[:, :, 13:14] = (torch.rand(1, 5, 1, 128, 128) > 0.95).float()
    # Masks are 1
    x[:, :, 15:16] = 1.0
    x[:, :, 16:17] = 1.0
    # FRP non-negative clipped
    x[:, :, 14:15] = torch.clamp(x[:, :, 14:15], min=0.0, max=64.0)
    return x


@pytest.fixture(scope="session")
def sample_target():
    """Tạo nhãn thực tế ngày t+1 (B=1, C=1, H=128, W=128)."""
    torch.manual_seed(43)
    y = (torch.rand(1, 1, 128, 128, dtype=torch.float32) > 0.93).float()
    return y


@pytest.fixture(scope="session")
def model_instance():
    """Khởi tạo mô hình WildfireUTAE phục vụ kiểm thử."""
    model = WildfireUTAE(input_dim=17)
    model.eval()
    return model


@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Tạo file checkpoint hợp lệ kèm mã băm SHA256 tham chiếu."""
    model = WildfireUTAE(input_dim=17)
    ckpt_path = tmp_path / "test_checkpoint.pth"
    torch.save(model.state_dict(), ckpt_path)

    # Tính SHA256
    sha256 = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    valid_hash = sha256.hexdigest()

    return str(ckpt_path), valid_hash
