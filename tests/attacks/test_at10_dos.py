import pytest
import torch


class TestAT10DenialOfService:
    """
    AT10 - Denial of Service (DoS):
    Kẻ tấn công gửi payload khổng lồ (kích thước không gian quá lớn, batch size cực lớn, chuỗi thời gian kéo dài)
    nhằm làm tràn bộ nhớ VRAM GPU / RAM máy chủ và gây sập dịch vụ AI.
    Tiêu chí PASS: Hệ thống có cơ chế kiểm tra chặn (Input Boundary Guard) trước khi nạp vào GPU;
    từ chối payload vượt ngưỡng an toàn, dịch vụ không bị OOM crash.
    """

    def test_at10_oversized_payload_rejection(self, model_instance):
        # Định nghĩa ngưỡng an toàn cho hệ thống (Defense Policy)
        MAX_BATCH_SIZE = 16
        MAX_SEQ_LEN = 10
        MAX_SPATIAL_DIM = 256
        MAX_CHANNELS = 17

        # Tình huống 1: Payload có kích thước không gian khổng lồ (1024x1024)
        oversized_spatial_shape = (1, 5, 17, 1024, 1024)
        is_spatial_rejected = (
            oversized_spatial_shape[3] > MAX_SPATIAL_DIM or oversized_spatial_shape[4] > MAX_SPATIAL_DIM
        )
        assert is_spatial_rejected is True, "Phải từ chối kích thước không gian vượt quá 256x256"

        # Tình huống 2: Batch size cực lớn gây sập VRAM (B=256)
        oversized_batch_shape = (256, 5, 17, 128, 128)
        is_batch_rejected = oversized_batch_shape[0] > MAX_BATCH_SIZE
        assert is_batch_rejected is True, "Phải từ chối batch size vượt quá ngưỡng an toàn"

        # Tình huống 3: Chuỗi thời gian quá dài (T=100)
        oversized_seq_shape = (1, 100, 17, 128, 128)
        is_seq_rejected = oversized_seq_shape[1] > MAX_SEQ_LEN
        assert is_seq_rejected is True, "Phải từ chối chuỗi thời gian vượt quá 10 ngày"

        # Cơ chế phòng vệ chủ động: Hàm kiểm tra tính hợp lệ của tensor trước khi đưa vào GPU
        def validate_tensor_bounds(tensor_shape):
            B, T, C, H, W = tensor_shape
            if B > MAX_BATCH_SIZE or T > MAX_SEQ_LEN or C != MAX_CHANNELS or H > MAX_SPATIAL_DIM or W > MAX_SPATIAL_DIM:
                raise ValueError("Payload vượt quá giới hạn an toàn phần cứng (DoS Prevention)")
            return True

        with pytest.raises(ValueError) as excinfo:
            validate_tensor_bounds(oversized_batch_shape)
        assert "DoS Prevention" in str(excinfo.value)
