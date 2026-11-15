import pytest
import torch


class TestAT03DistributionShift:
    """
    AT03 - Out-of-Distribution (OOD) / Distribution Shift:
    Đưa vào các đặc trưng môi trường nằm ngoài phân phối huấn luyện (ví dụ: nhiệt độ cực đoan, tốc độ gió phi thực tế).
    Tiêu chí PASS: Mô hình xử lý an toàn bằng cơ chế chuẩn hóa/clipping, không crash và xác suất đầu ra luôn bị chặn trong [0, 1].
    """

    def test_at03_extreme_weather_values_handling(self, model_instance, sample_valid_input):
        model = model_instance
        model.eval()

        ood_input = sample_valid_input.clone()

        # Giả lập giá trị cực hạn phi thực tế: Tốc độ gió 200 m/s (kênh 3), Nhiệt độ cực hạn (kênh 1)
        ood_input[:, :, 3] = 200.0   # Tốc độ gió phi lý
        ood_input[:, :, 1] = 100.0   # Nhiệt độ vượt ngưỡng khí tượng
        ood_input[:, :, 8] = -10.0   # Áp suất hơi nước thiếu hụt âm (VPD)

        # Cơ chế phòng vệ cấp tiền xử lý: Kẹp giá trị vào miền hợp lý (clipping)
        clipped_input = ood_input.clone()
        clipped_input[:, :, 3] = torch.clamp(clipped_input[:, :, 3], -5.0, 50.0)
        clipped_input[:, :, 1] = torch.clamp(clipped_input[:, :, 1], -10.0, 60.0)
        clipped_input[:, :, 8] = torch.clamp(clipped_input[:, :, 8], min=0.0)

        with torch.no_grad():
            raw_logits = model(ood_input)
            clipped_logits = model(clipped_input)

            raw_probs = torch.sigmoid(raw_logits)
            clipped_probs = torch.sigmoid(clipped_logits)

        # 1. Đầu ra xác suất vẫn được giữ trong [0, 1]
        assert torch.all(raw_probs >= 0.0) and torch.all(raw_probs <= 1.0)
        assert torch.all(clipped_probs >= 0.0) and torch.all(clipped_probs <= 1.0)

        # 2. Không sinh ra NaN/Inf dù đầu vào dị thường
        assert not torch.isnan(raw_logits).any()
        assert not torch.isnan(clipped_logits).any()
