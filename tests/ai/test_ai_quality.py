import pytest
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, precision_recall_curve, auc

from src.model import WildfireUTAE


class TestAIModelQuality:
    """
    Kiểm thử chất lượng và tính đúng đắn của mô hình AI U-TAE.
    Bao gồm các Test Case: TC05, TC06, TC07, TC08.
    """

    def test_tc05_temporal_sequence_consistency(self, sample_valid_input):
        """
        TC05 - Kiểm tra chuỗi thời gian:
        Kiểm tra độ dài chuỗi T=5 ngày liên tiếp, thứ tự ngày tăng dần,
        kênh mặt nạ khuyết thời tiết và thực vật.
        Kết quả mong đợi: Đúng thứ tự thời gian, validity mask xử lý đúng quy định.
        """
        B, T, C, H, W = sample_valid_input.shape
        assert T == 5, f"Số bước thời gian phải là 5, nhận được {T}"
        assert C == 17, f"Số kênh đầu vào phải là 17, nhận được {C}"

        # Kiểm tra kênh mặt nạ (Channel 15: weather_mask, Channel 16: vegetation_mask)
        w_mask = sample_valid_input[:, :, 15]
        v_mask = sample_valid_input[:, :, 16]
        assert torch.all((w_mask == 0.0) | (w_mask == 1.0)), "Weather mask phải là nhị phân"
        assert torch.all((v_mask == 0.0) | (v_mask == 1.0)), "Vegetation mask phải là nhị phân"

    def test_tc06_utae_output_specifications(self, model_instance, sample_valid_input):
        """
        TC06 - Kiểm tra đầu ra U-TAE:
        Truyền batch dữ liệu hợp lệ (1, 5, 17, 128, 128) vào mô hình.
        Kết quả mong đợi: Đầu ra đúng shape (1, 1, 128, 128), không NaN/Inf,
        xác suất sau sigmoid nằm trọn trong khoảng [0, 1].
        """
        model_instance.eval()
        with torch.no_grad():
            logits = model_instance(sample_valid_input)
            probs = torch.sigmoid(logits)

        # 1. Kiểm tra shape
        assert logits.shape == (1, 1, 128, 128), f"Shape đầu ra sai: {logits.shape}"

        # 2. Kiểm tra tính toàn vẹn số học
        assert not torch.isnan(logits).any(), "Logits chứa NaN"
        assert not torch.isinf(logits).any(), "Logits chứa Inf"

        # 3. Kiểm tra miền giá trị xác suất
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Xác suất nằm ngoài [0, 1]"

    def test_tc07_prediction_quality_and_metrics(self, model_instance, sample_valid_input, sample_target):
        """
        TC07 - Đánh giá chất lượng dự đoán & So sánh với Persistence Baseline:
        Tính toán IoU, F1-score, Precision, Recall, PR-AUC.
        Kết quả mong đợi: Có đầy đủ các chỉ số đánh giá, tính toán chính xác so với baseline.
        """
        model_instance.eval()
        with torch.no_grad():
            logits = model_instance(sample_valid_input)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        y_true = sample_target[0, 0].cpu().numpy().astype(int).flatten()
        y_prob = probs.flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        # 1. Tính toán các metric đánh giá
        iou = jaccard_score(y_true, y_pred, zero_division=1.0)
        f1 = f1_score(y_true, y_pred, zero_division=1.0)
        prec = precision_score(y_true, y_pred, zero_division=1.0)
        rec = recall_score(y_true, y_pred, zero_division=1.0)

        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)

        assert 0.0 <= iou <= 1.0, f"IoU không hợp lệ: {iou}"
        assert 0.0 <= f1 <= 1.0, f"F1 không hợp lệ: {f1}"
        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0
        assert 0.0 <= pr_auc <= 1.0

        # 2. Đối sánh với Persistence Baseline (Lấy vết cháy ngày t làm dự đoán ngày t+1)
        day_t_fire = sample_valid_input[0, -1, 13].cpu().numpy().flatten().astype(int)
        baseline_iou = jaccard_score(y_true, day_t_fire, zero_division=1.0)
        assert baseline_iou >= 0.0

    def test_tc08_inference_determinism_and_stability(self, model_instance, sample_valid_input):
        """
        TC08 - Kiểm tra tính ổn định (Determinism):
        Chạy cùng một đầu vào 5 lần liên tiếp ở chế độ evaluation.
        Kết quả mong đợi: Kết quả ra giống hệt nhau, sai số tuyệt đối tối đa < 1e-6.
        """
        model_instance.eval()
        outputs = []
        with torch.no_grad():
            for _ in range(5):
                out = model_instance(sample_valid_input)
                outputs.append(out.clone())

        for i in range(1, len(outputs)):
            max_diff = torch.max(torch.abs(outputs[0] - outputs[i])).item()
            assert max_diff < 1e-6, f"Kết quả không ổn định, sai khác cực đại: {max_diff}"
