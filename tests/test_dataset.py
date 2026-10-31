"""
test_dataset.py
---------------
Kiểm tra toàn diện dữ liệu đầu vào cho mô hình U-TAE:
1. Đọc dữ liệu từ thư mục processed_v1 (và dữ liệu giả lập cho CI/CD).
2. Kích thước tensor đầu vào: x (B, T=5, C=17, H=128, W=128).
3. Kích thước và giá trị nhãn: y (B, 1, 128, 128), nhị phân {0.0, 1.0}.
4. Kiểm tra an toàn: Không chứa giá trị NaN hoặc Inf.
5. Kiểm tra DataLoader: Sinh batch hợp lệ, đúng kiểu dữ liệu.
"""

import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

# Đảm bảo đường dẫn gốc của dự án nằm trong sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.wildfire_dataset import (
    CHANNEL_NAMES,
    WildfireDataset,
    get_dataloader,
)

REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed_v1")


@pytest.fixture
def mock_dataset_dir(tmp_path) -> str:
    """
    Tạo môi trường dữ liệu giả lập nhẹ để chạy unit test độc lập
    ngay cả trên môi trường CI/CD (GitHub Actions) không có sẵn tập dữ liệu lớn.
    """
    data_dir = tmp_path / "processed_v1"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)

    # 1. Tạo tệp .npz mẫu với kích thước 128x128 và 10 bước thời gian
    num_days = 10
    h, w = 128, 128

    dyn_cont = np.random.uniform(0.0, 10.0, size=(num_days, 14, h, w)).astype(np.float32)
    fire = np.zeros((num_days, h, w), dtype=np.float32)
    # Giả lập vết cháy nhỏ ở giữa
    fire[4:, 50:70, 50:70] = 1.0

    weather_mask = np.ones((num_days, h, w), dtype=np.float32)
    vegetation_mask = np.ones((num_days, h, w), dtype=np.float32)

    npz_path = train_dir / "tile_001_001_2023.npz"
    np.savez_compressed(
        npz_path,
        dyn_cont=dyn_cont,
        fire=fire,
        weather_mask=weather_mask,
        vegetation_mask=vegetation_mask,
    )

    # 2. Tạo tệp index.csv mẫu
    index_data = {
        "file": ["tile_001_001_2023.npz", "tile_001_001_2023.npz"],
        "input_start": [0, 1],
        "input_end": [4, 5],
        "target": [5, 6],
        "target_date": ["2023-08-01", "2023-08-02"],
        "target_year": [2023, 2023],
        "fire_px_target": [400, 400],
        "fire_px_last": [400, 400],
    }
    df = pd.DataFrame(index_data)
    df.to_csv(train_dir / "index.csv", index=False)

    return str(data_dir)


class TestWildfireDataset:
    """Tập hợp các bài kiểm tra đọc dữ liệu đầu vào."""

    def test_channel_names_specification(self):
        """Xác nhận định nghĩa đầy đủ 17 kênh đầu vào."""
        assert len(CHANNEL_NAMES) == 17
        assert CHANNEL_NAMES[13] == "fire"
        assert CHANNEL_NAMES[15] == "weather_valid"
        assert CHANNEL_NAMES[16] == "vegetation_valid"

    def test_mock_dataset_initialization(self, mock_dataset_dir):
        """Kiểm tra khởi tạo Dataset trên thư mục hợp lệ."""
        dataset = WildfireDataset(root_dir=mock_dataset_dir, split="train")
        assert len(dataset) == 2

    def test_dataset_tensor_shapes_and_types(self, mock_dataset_dir):
        """
        Kiểm tra cấu trúc và kiểu dữ liệu của mẫu đọc ra:
        - x: (T=5, C=17, H=128, W=128), float32
        - y: (1, H=128, W=128), float32
        - persistence: (1, H=128, W=128), float32
        """
        dataset = WildfireDataset(root_dir=mock_dataset_dir, split="train")
        x, y, persistence, meta = dataset[0]

        # Kiểm tra shape
        assert x.shape == (5, 17, 128, 128), f"x shape không đúng: {x.shape}"
        assert y.shape == (1, 128, 128), f"y shape không đúng: {y.shape}"
        assert persistence.shape == (1, 128, 128), f"persistence shape không đúng: {persistence.shape}"

        # Kiểm tra kiểu dữ liệu
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
        assert persistence.dtype == torch.float32

        # Kiểm tra metadata
        assert isinstance(meta, dict)
        assert "file" in meta
        assert "target_date" in meta
        assert "fire_px_target" in meta

    def test_dataset_no_nan_or_inf(self, mock_dataset_dir):
        """Đảm bảo dữ liệu đầu vào và nhãn tuyệt đối không chứa NaN hoặc Inf."""
        dataset = WildfireDataset(root_dir=mock_dataset_dir, split="train")
        x, y, persistence, _ = dataset[0]

        assert not torch.isnan(x).any(), "Phát hiện NaN trong tensor đầu vào x"
        assert not torch.isinf(x).any(), "Phát hiện Inf trong tensor đầu vào x"
        assert not torch.isnan(y).any(), "Phát hiện NaN trong nhãn y"
        assert not torch.isinf(y).any(), "Phát hiện Inf trong nhãn y"
        assert not torch.isnan(persistence).any(), "Phát hiện NaN trong baseline persistence"

    def test_labels_are_binary(self, mock_dataset_dir):
        """Kiểm tra nhãn mục tiêu y chỉ chứa các giá trị nhị phân {0.0, 1.0}."""
        dataset = WildfireDataset(root_dir=mock_dataset_dir, split="train")
        _, y, persistence, _ = dataset[0]

        unique_y = torch.unique(y).tolist()
        for val in unique_y:
            assert val in [0.0, 1.0], f"Nhãn y chứa giá trị không hợp lệ: {val}"

        unique_p = torch.unique(persistence).tolist()
        for val in unique_p:
            assert val in [0.0, 1.0], f"Persistence chứa giá trị không hợp lệ: {val}"

    def test_dataloader_batch_generation(self, mock_dataset_dir):
        """Kiểm tra DataLoader ghép batch chuẩn xác."""
        dataloader = get_dataloader(
            root_dir=mock_dataset_dir,
            split="train",
            batch_size=2,
            shuffle=False,
        )

        batch = next(iter(dataloader))
        batch_x, batch_y, batch_p, batch_meta = batch

        assert batch_x.shape == (2, 5, 17, 128, 128)
        assert batch_y.shape == (2, 1, 128, 128)
        assert batch_p.shape == (2, 1, 128, 128)
        assert len(batch_meta["file"]) == 2

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(REAL_DATA_DIR, "train", "index.csv")),
        reason="Bỏ qua kiểm tra dữ liệu thật nếu chưa có thư mục data/processed_v1",
    )
    def test_real_dataset_integrity(self):
        """Kiểm tra ngẫu nhiên mẫu thực tế trên đĩa nếu có dữ liệu thật."""
        dataset = WildfireDataset(root_dir=REAL_DATA_DIR, split="train", max_samples=5)
        assert len(dataset) > 0

        x, y, p, meta = dataset[0]
        assert x.shape == (5, 17, 128, 128)
        assert y.shape == (1, 128, 128)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        assert not torch.isnan(y).any()
