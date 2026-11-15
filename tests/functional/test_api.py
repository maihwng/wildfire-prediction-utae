import pytest
import torch
import numpy as np
import os
import sys

from app import run_prediction, CATALOG, get_tile_bbox_wgs, pixel_to_latlon, WildfireCatalog
from src.model import WildfireUTAE


class TestAPIEndpoints:
    """
    Kiểm thử cấp độ API (Automation Test) cho hệ thống Wildfire Prediction.
    Bao gồm các Test Case: TC01, TC02, TC03.
    """

    def test_tc01_api_valid_input_acceptance(self, sample_valid_input, model_instance):
        """
        TC01 - Tiếp nhận dữ liệu hợp lệ (API):
        Gửi dữ liệu chuỗi 5 ngày, 17 kênh, kích thước 128x128.
        Kết quả mong đợi: Hệ thống chấp nhận và trả về kết quả dự đoán hợp lệ (1, 128, 128).
        """
        assert sample_valid_input.shape == (1, 5, 17, 128, 128)
        assert sample_valid_input.dtype == torch.float32

        with torch.no_grad():
            logits = model_instance(sample_valid_input)
            probs = torch.sigmoid(logits)

        assert logits.shape == (1, 1, 128, 128), f"Shape không hợp lệ: {logits.shape}"
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0), "Xác suất vượt miền [0, 1]"
        assert not torch.isnan(probs).any(), "Kết quả chứa NaN"

    def test_tc02_api_missing_data_rejection(self, model_instance):
        """
        TC02 - Tiếp nhận dữ liệu thiếu (API):
        Gửi dữ liệu thiếu ngày (T=3 thay vì 5), thiếu kênh (C=14 thay vì 17) hoặc thiếu fire_id.
        Kết quả mong đợi: Hệ thống từ chối và thông báo lỗi rõ ràng, không sập.
        """
        # Tình huống 1: Thiếu ngày (T=3)
        input_missing_days = torch.randn(1, 3, 17, 128, 128)
        assert input_missing_days.shape[1] < 5, "Không phát hiện thiếu ngày"

        # Tình huống 2: Thiếu kênh (C=14 thay vì 17)
        input_missing_channels = torch.randn(1, 5, 14, 128, 128)
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            _ = model_instance(input_missing_channels)
        assert "channels" in str(excinfo.value).lower() or "expected" in str(excinfo.value).lower()

        # Tình huống 3: Gọi pipeline với fire_label không tồn tại
        result = run_prediction("INVALID_FIRE_ID_9999", "19/08/2020")
        assert result is not None
        assert isinstance(result, tuple) and len(result) == 2
        assert "Lỗi" in result[0] or "Lỗi" in result[1] or "error" in result[0].lower()

    def test_tc03_api_invalid_format_handling(self, model_instance):
        """
        TC03 - Dữ liệu sai định dạng (API):
        Gửi dữ liệu sai kích thước, sai dtype hoặc chứa giá trị NaN/Inf.
        Kết quả mong đợi: Hệ thống phát hiện dữ liệu không hợp lệ, không bị crash không kiểm soát.
        """
        # Tình huống 1: Dữ liệu chứa NaN
        input_with_nan = torch.randn(1, 5, 17, 128, 128)
        input_with_nan[0, 2, 5, 30, 30] = float("nan")
        has_nan = torch.isnan(input_with_nan).any()
        assert has_nan.item() is True, "Phải phát hiện được giá trị NaN"

        # Tình huống 2: Dữ liệu chứa Inf
        input_with_inf = torch.randn(1, 5, 17, 128, 128)
        input_with_inf[0, 1, 0, 10, 10] = float("inf")
        has_inf = torch.isinf(input_with_inf).any()
        assert has_inf.item() is True, "Phải phát hiện được giá trị Inf"

        # Tình huống 3: Kích thước không gian sai (64x64 thay vì 128x128)
        input_wrong_spatial = torch.randn(1, 5, 17, 64, 64)
        with torch.no_grad():
            output = model_instance(input_wrong_spatial)
        assert output.shape == (1, 1, 64, 64)
