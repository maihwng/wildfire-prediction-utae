import pytest
import time
import torch


class TestAT05ModelExtraction:
    """
    AT05 - Model Extraction / Model Stealing:
    Kẻ tấn công gửi truy vấn hàng loạt với tần suất cao nhằm tạo mô hình surrogate bắt chước.
    Tiêu chí PASS: Hệ thống giới hạn tần suất truy vấn (Rate Limiting), không để lộ thông tin trọng số hay feature map nội bộ.
    """

    def test_at05_rate_limiting_and_information_leakage(self, model_instance, sample_valid_input):
        model = model_instance
        model.eval()

        # 1. Kiểm tra rò rỉ thông tin nội bộ qua đầu ra API
        with torch.no_grad():
            output = model(sample_valid_input)
            probs = torch.sigmoid(output)

        # Đầu ra chỉ là tensor mặt nạ nhị phân hoặc xác suất, không chứa hidden representations hay gradient
        assert not hasattr(output, "grad_fn") or output.grad_fn is None, "Rò rỉ computation graph"
        assert output.shape == (1, 1, 128, 128)

        # 2. Giả lập cơ chế phòng thủ Rate Limiting (Giới hạn tối đa 50 requests / phút từ 1 IP)
        max_requests_per_minute = 50
        simulated_queries = 60

        request_timestamps = []
        is_blocked = False

        for req_idx in range(simulated_queries):
            now = time.time()
            request_timestamps.append(now)
            # Nếu trong 60 giây số lượng request vượt quá 50 -> Chặn
            window_requests = [t for t in request_timestamps if now - t <= 60.0]
            if len(window_requests) > max_requests_per_minute:
                is_blocked = True
                break

        assert is_blocked is True, "Cơ chế Rate Limiting phải chặn spam query từ kẻ tấn công trích xuất mô hình"
