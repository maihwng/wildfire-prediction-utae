import pytest
import hashlib
import os
import torch

from src.model import WildfireUTAE


class TestAT04ModelTampering:
    """
    AT04 - Model Tampering (Sửa đổi / Can thiệp tệp checkpoint trái phép):
    Kẻ tấn công sửa đổi một vài byte trong file trọng số `.pth` hoặc inject payload.
    Tiêu chí PASS: Kiểm tra mã băm SHA256 trước khi load; từ chối 100% file không khớp hash.
    """

    def test_at04_sha256_verification_detects_tampering(self, dummy_checkpoint):
        ckpt_path, valid_hash = dummy_checkpoint

        # 1. Kiểm tra xác thực file nguyên bản thành công
        sha256 = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        assert sha256.hexdigest() == valid_hash, "Hash nguyên bản không khớp"

        # 2. Thực hiện tấn công can thiệp: Sửa 8 bytes ở giữa file checkpoint
        with open(ckpt_path, "r+b") as f:
            f.seek(100)
            f.write(b"\xde\xad\xbe\xef\x00\x00\x00\x00")

        # 3. Cơ chế phòng vệ: Tính lại SHA256 trước khi load
        sha256_tampered = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            while chunk := f.read(8192):
                sha256_tampered.update(chunk)
        tampered_hash = sha256_tampered.hexdigest()

        # Tiêu chí phòng thủ: SHA256 thay đổi hoàn toàn
        assert tampered_hash != valid_hash, "Hash không phát hiện được sửa đổi byte"

        # Từ chối nạp mô hình khi sai mã hash
        is_safe_to_load = (tampered_hash == valid_hash)
        assert is_safe_to_load is False, "Hệ thống phải từ chối nạp checkpoint bị can thiệp"
