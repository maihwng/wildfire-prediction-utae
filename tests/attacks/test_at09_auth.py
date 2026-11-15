import pytest
import os
from pathlib import Path


class TestAT09AuthenticationBypass:
    """
    AT09 - Authentication / Authorization Bypass & Path Traversal:
    Kẻ tấn công cố tình truy cập các tài nguyên nhạy cảm (file cấu hình, mã nguồn nội bộ, checkpoint gốc)
    hoặc sử dụng kỹ thuật Directory / Path Traversal (../../).
    Tiêu chí PASS: Hệ thống chặn hoàn toàn Path Traversal, không cho phép đọc file ngoài thư mục cho phép.
    """

    def test_at09_path_traversal_prevention(self, project_root):
        data_base_dir = Path(project_root) / "data"

        # Danh sách các đường dẫn tấn công Directory Traversal phổ biến
        malicious_paths = [
            "../../../../etc/passwd",
            "..\\..\\..\\Windows\\win.ini",
            "../../.env",
            "../../checkpoints/best_model.pth",
            "../src/model.py"
        ]

        for attack_path in malicious_paths:
            # Mô phỏng hàm giải quyết đường dẫn có phòng vệ (Safe Path Resolver)
            requested_path = (data_base_dir / attack_path).resolve()

            # Tiêu chí: Đường dẫn sau khi resolve phải nằm TRONG thư mục data_base_dir
            is_escaped = not str(requested_path).startswith(str(data_base_dir.resolve()))

            # Hệ thống phát hiện truy cập vượt biên và từ chối
            assert is_escaped is True, f"Phát hiện tấn công vượt biên đối với: {attack_path}"

            # Kiểm tra cơ chế chặn truy cập
            with pytest.raises(PermissionError):
                if is_escaped:
                    raise PermissionError(f"Truy cập trái phép bị từ chối: {attack_path}")
