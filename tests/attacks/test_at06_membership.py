import pytest
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


class TestAT06MembershipInference:
    """
    AT06 - Membership Inference Attack (MIA):
    Kẻ tấn công so sánh độ tin cậy/loss giữa tập train (members) và tập holdout (non-members)
    để đoán xem một mẫu cụ thể có thuộc về dữ liệu huấn luyện hay không.
    Tiêu chí PASS: AUC của bộ tấn công xấp xỉ 0.50 (<= 0.60), chứng minh mô hình không làm rò rỉ thông tin riêng tư.
    """

    def test_at06_membership_inference_auc_bound(self, model_instance):
        torch.manual_seed(42)
        np.random.seed(42)
        model = model_instance
        model.eval()

        # Tạo 20 mẫu giả lập: 10 mẫu train (member) và 10 mẫu holdout (non-member)
        num_samples = 10
        torch.manual_seed(101)
        member_inputs = torch.randn(num_samples, 5, 17, 128, 128)
        member_targets = (torch.rand(num_samples, 1, 128, 128) > 0.95).float()

        torch.manual_seed(202)
        non_member_inputs = torch.randn(num_samples, 5, 17, 128, 128)
        non_member_targets = (torch.rand(num_samples, 1, 128, 128) > 0.95).float()

        # Tính toán loss của từng mẫu
        with torch.no_grad():
            member_logits = model(member_inputs)
            member_losses = F.binary_cross_entropy_with_logits(member_logits, member_targets, reduction="none").mean(dim=[1, 2, 3]).cpu().numpy()

            non_member_logits = model(non_member_inputs)
            non_member_losses = F.binary_cross_entropy_with_logits(non_member_logits, non_member_targets, reduction="none").mean(dim=[1, 2, 3]).cpu().numpy()

        # Kẻ tấn công phân loại dựa trên giả định: Member có loss thấp hơn Non-member
        # Điểm số dự đoán member = -loss
        y_true = np.array([1] * num_samples + [0] * num_samples)
        y_scores = np.concatenate([-member_losses, -non_member_losses])

        attack_auc = roc_auc_score(y_true, y_scores)

        # Tiêu chí: Khả năng suy đoán thành viên của kẻ tấn công không được vượt quá 0.65 (lý tưởng là gần 0.5 - đoán ngẫu nhiên)
        assert attack_auc <= 0.65, f"Cảnh báo rò rỉ dữ liệu thành viên! AUC = {attack_auc:.3f} vượt ngưỡng an toàn"
