import pytest
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import jaccard_score

from src.model import WildfireUTAE


class TestAT01AdversarialAttack:
    """
    AT01 - Adversarial Input Attack:
    Tấn công bằng nhiễu đối kháng FGSM (Fast Gradient Sign Method) trên các kênh liên tục.
    Tiêu chí PASS: Mô hình không sinh NaN/Inf, mức suy giảm IoU không làm mô hình sụp đổ hoàn toàn.
    """

    def test_at01_fgsm_adversarial_robustness(self, model_instance, sample_valid_input, sample_target):
        model = model_instance
        model.eval()

        # Cho phép tính gradient trên input
        x = sample_valid_input.clone().detach().requires_grad_(True)
        y = sample_target.clone().detach()

        criterion = nn.BCEWithLogitsLoss()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Tạo nhiễu FGSM với epsilon = 0.05
        epsilon = 0.05
        data_grad = x.grad.data
        sign_data_grad = data_grad.sign()

        # Chỉ áp dụng nhiễu lên 13 kênh thời tiết/địa hình đầu tiên (giữ nguyên fire mask và validity mask)
        perturbed_x = x.clone().detach()
        perturbed_x[:, :, 0:13] += epsilon * sign_data_grad[:, :, 0:13]

        # Đánh giá phản ứng của mô hình trước mẫu đối kháng
        with torch.no_grad():
            clean_logits = model(sample_valid_input)
            adv_logits = model(perturbed_x)

            clean_probs = torch.sigmoid(clean_logits)[0, 0].cpu().numpy().flatten()
            adv_probs = torch.sigmoid(adv_logits)[0, 0].cpu().numpy().flatten()

        # Kiểm tra tính toàn vẹn số học
        assert not torch.isnan(adv_logits).any(), "Adversarial input gây ra NaN"
        assert not torch.isinf(adv_logits).any(), "Adversarial input gây ra Inf"

        # Đảm bảo miền giá trị xác suất vẫn bị chặn chặt chẽ trong [0, 1]
        assert (adv_probs >= 0.0).all() and (adv_probs <= 1.0).all(), "Xác suất vượt miền [0, 1]"

        # Tiêu chí: Độ lệch xác suất trung bình do nhiễu nhỏ FGSM (eps=0.05) có chặn trên an toàn
        mean_prob_shift = float(np.abs(adv_probs - clean_probs).mean())
        assert mean_prob_shift <= 0.40, f"Độ lệch xác suất trung bình quá lớn: {mean_prob_shift:.2%}"

        # Tính độ suy giảm IoU (nếu mẫu có diện tích cháy đủ lớn)
        y_true = y[0, 0].cpu().numpy().astype(int).flatten()
        clean_iou = jaccard_score(y_true, (clean_probs >= 0.5).astype(int), zero_division=1.0)
        adv_iou = jaccard_score(y_true, (adv_probs >= 0.5).astype(int), zero_division=1.0)
        if clean_iou > 0.05:
            relative_drop = (clean_iou - adv_iou) / (clean_iou + 1e-6)
            assert relative_drop <= 0.60, f"Mô hình quá nhạy cảm với FGSM, độ suy giảm: {relative_drop:.2%}"
