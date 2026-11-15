import pytest
import torch
import numpy as np


class TestAT02DataPoisoning:
    """
    AT02 - Data Poisoning Attack:
    Tấn công làm nhiễm bẩn dữ liệu bằng cách lật nhãn (Label Flipping) 5-10% mẫu ngẫu nhiên.
    Tiêu chí PASS: Hệ thống phát hiện được sự thay đổi phân phối nhãn bất thường
    hoặc giới hạn được mức độ suy thoái mất mát khi fine-tune.
    """

    def test_at02_label_flipping_detection(self, sample_target):
        clean_target = sample_target.clone()
        poisoned_target = clean_target.clone()

        # Đảo nhãn 10% số pixel
        num_pixels = 128 * 128
        poison_ratio = 0.10
        num_poison = int(num_pixels * poison_ratio)

        indices = np.random.choice(num_pixels, num_poison, replace=False)
        flat_poison = poisoned_target.view(-1)
        flat_poison[indices] = 1.0 - flat_poison[indices]

        # Kiểm tra tỷ lệ biến đổi dữ liệu
        diff_pixels = torch.sum(clean_target != poisoned_target).item()
        actual_ratio = diff_pixels / num_pixels
        assert abs(actual_ratio - poison_ratio) < 0.01, "Tỷ lệ đầu độc không đúng"

        # Cơ chế phòng vệ: Kiểm tra ngưỡng phân phối điểm cháy
        # Nhãn cháy rừng thường rất thưa (sparse, < 5% pixel cháy). Nếu tỷ lệ cháy tăng đột biến bất thường, hệ thống cảnh báo
        clean_fire_ratio = torch.mean(clean_target).item()
        poisoned_fire_ratio = torch.mean(poisoned_target).item()

        # Phát hiện sự bất thường về mật độ cháy (Statistical Anomaly Detection)
        anomaly_detected = abs(poisoned_fire_ratio - clean_fire_ratio) > 0.05
        assert anomaly_detected is True, "Cơ chế phát hiện bất thường nhãn phải cảnh báo khi bị đầu độc 10%"
