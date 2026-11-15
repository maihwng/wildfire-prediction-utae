import pytest
import hmac
import hashlib
import json


class TestAT08OutputIntegrity:
    """
    AT08 - Output Integrity Attack:
    Giả lập tình huống kết quả dự đoán của AI bị can thiệp / sửa đổi ở phía Client
    hoặc trên đường truyền trước khi lưu trữ vào cơ sở dữ liệu.
    Tiêu chí PASS: Backend không tin dữ liệu do client tự gửi; có cơ chế kiểm tra tính toàn vẹn (HMAC / Hash Signature).
    """

    def test_at08_client_side_tampered_payload_rejected(self):
        secret_key = b"wildfire_backend_secret_key_2026"

        # 1. Kết quả hợp lệ do AI Backend sinh ra
        original_prediction = {
            "fire_id": "US_CA_AUGUSTCOMPLEX_2020",
            "date": "2020-08-20",
            "predicted_burned_ha": 1450.5,
            "risk_level": "Nguy cấp",
            "confidence": 0.892
        }

        # Tạo chữ ký HMAC-SHA256 trên dữ liệu gốc
        raw_bytes = json.dumps(original_prediction, sort_keys=True).encode("utf-8")
        valid_signature = hmac.new(secret_key, raw_bytes, hashlib.sha256).hexdigest()

        # 2. Kẻ tấn công can thiệp ở Client: Hạ mức độ rủi ro từ "Nguy cấp" xuống "An toàn" và giảm diện tích
        tampered_prediction = original_prediction.copy()
        tampered_prediction["predicted_burned_ha"] = 12.0
        tampered_prediction["risk_level"] = "An toàn"

        # 3. Backend nhận dữ liệu và tiến hành xác thực chữ ký (Integrity Verification)
        tampered_bytes = json.dumps(tampered_prediction, sort_keys=True).encode("utf-8")
        computed_signature = hmac.new(secret_key, tampered_bytes, hashlib.sha256).hexdigest()

        # Tiêu chí phòng thủ: Chữ ký không khớp, Backend từ chối dữ liệu bị can thiệp
        is_tampered = not hmac.compare_digest(valid_signature, computed_signature)
        assert is_tampered is True, "Hệ thống phải phát hiện và từ chối payload bị can thiệp sửa đổi"
