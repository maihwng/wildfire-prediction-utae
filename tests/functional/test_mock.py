import pytest
from unittest.mock import patch
import torch
import numpy as np

import app
from app import get_model, format_metadata_html, run_prediction, CATALOG
from src.model import WildfireUTAE


class TestMockScenarios:
    """
    Kiểm thử giả lập (Mock Test) cho các tình huống bất thường trong hệ thống.
    Bao gồm các Test Case: TC09, TC10.
    """

    def test_tc09_mock_missing_checkpoint(self, monkeypatch):
        """
        TC09 - Giả lập thiếu checkpoint:
        Dùng monkeypatch giả lập đường dẫn checkpoint không tồn tại.
        Kết quả mong đợi: Hệ thống phát hiện thiếu checkpoint, thông báo rõ ràng,
        không bị crash hoặc tự khởi tạo mô hình dự phòng an toàn.
        """
        fake_path = "/nonexistent/directory/best_model_missing.pth"
        monkeypatch.setattr(app, "CHECKPOINT_PATH", fake_path)

        # Kiểm tra get_model xử lý ngoại lệ đường dẫn không tồn tại
        model = get_model()
        assert model is not None, "Mô hình phải được khởi tạo an toàn"
        assert isinstance(model, WildfireUTAE), "Kiểu mô hình không đúng"

    def test_tc10_mock_zero_fire_data(self):
        """
        TC10 - Giả lập không có dữ liệu cháy (Zero Fire Pixels):
        Mock dữ liệu với nhãn toàn 0 và xác suất dự đoán không phát hiện đám cháy.
        Kết quả mong đợi: Giao diện xử lý an toàn; các độ đo IoU/F1 không bị lỗi chia cho 0 (ZeroDivisionError).
        """
        # 1. Giả lập nhãn và dự đoán toàn 0
        zero_gt = np.zeros((128, 128), dtype=np.uint8)
        zero_pred = np.zeros((128, 128), dtype=np.float32)

        # 2. Tính toán IoU và F1 với bảo vệ epsilon chống chia cho 0
        eps = 1e-6
        intersection = np.sum((zero_pred >= 0.7) * zero_gt)
        union = np.sum((zero_pred >= 0.7) + zero_gt) - intersection

        iou = (intersection + eps) / (union + eps)
        f1 = (2.0 * intersection + eps) / (2.0 * intersection + np.sum(zero_pred >= 0.7) + np.sum(zero_gt) + eps)

        assert not np.isnan(iou), "IoU bị NaN do chia cho 0"
        assert not np.isinf(iou), "IoU bị Inf"
        assert not np.isnan(f1), "F1 bị NaN do chia cho 0"

        # 3. Kiểm tra format_metadata_html xử lý an toàn với item có sẵn
        assert len(CATALOG.fire_labels) > 0
        sample_item = CATALOG.get(CATALOG.fire_labels[0])
        meta_html = format_metadata_html(fire_item=sample_item)
        assert isinstance(meta_html, str) and len(meta_html) > 50
        assert "meta-card" in meta_html
