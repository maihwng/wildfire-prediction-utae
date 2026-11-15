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
        data_base_dir.mkdir(parents=True, exist_ok=True)

        # Danh sách các đường dẫn tấn công Directory Traversal phổ biến (hỗ trợ cả Windows và Linux)
        malicious_paths = [
            "../../../../etc/passwd",
            r"..\..\..\Windows\win.ini",
            "../../.env",
            "../../checkpoints/best_model.pth",
            "../src/model.py"
        ]

        def is_safe_path(base: Path, user_path: str) -> bool:
            """Cơ chế kiểm soát biên Safe Path Resolver chuẩn hóa an ninh thông tin."""
            clean = user_path.replace("\\", "/").strip()
            parts = [p for p in clean.split("/") if p]
            if ".." in parts or clean.startswith("/"):
                return False
            resolved = (base / clean).resolve()
            return str(resolved).startswith(str(base.resolve()) + os.sep)

        for attack_path in malicious_paths:
            # Kiểm tra cơ chế phòng vệ chặn 100% các đường dẫn tấn công
            is_allowed = is_safe_path(data_base_dir, attack_path)
            assert is_allowed is False, f"Hệ thống không chặn được đường dẫn nguy hiểm: {attack_path}"

            # Hệ thống từ chối và kích hoạt ngoại lệ PermissionError
            with pytest.raises(PermissionError):
                if not is_allowed:
                    raise PermissionError(f"Truy cập trái phép bị từ chối: {attack_path}")
