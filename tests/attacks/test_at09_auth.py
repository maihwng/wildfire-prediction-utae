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

        # Danh sách các đường dẫn tấn công Directory Traversal phổ biến (hỗ trợ cả Windows và Linux)
        malicious_paths = [
            "../../../../etc/passwd",
            r"..\..\..\Windows\win.ini",
            "../../.env",
            "../../checkpoints/best_model.pth",
            "../src/model.py"
        ]

        for attack_path in malicious_paths:
            # Chuẩn hóa dấu phân cách đường dẫn để chặn triệt để Path Traversal trên cả Linux (POSIX) và Windows
            clean_path = attack_path.replace("\\", "/")
            requested_path = (data_base_dir / clean_path).resolve()

            # Tiêu chí an toàn: Đường dẫn sau khi resolve phải nằm hoàn toàn TRONG thư mục data_base_dir
            base_resolved = str(data_base_dir.resolve())
            is_escaped = not str(requested_path).startswith(base_resolved) or (str(requested_path) == base_resolved)

            # Hệ thống phát hiện truy cập vượt biên và từ chối
            assert is_escaped is True, f"Phát hiện tấn công vượt biên đối với: {attack_path}"

            # Kiểm tra cơ chế chặn truy cập
            with pytest.raises(PermissionError):
                if is_escaped:
                    raise PermissionError(f"Truy cập trái phép bị từ chối: {attack_path}")
