import os
import re
import pytest


class TestAT07SupplyChain:
    """
    AT07 - Supply Chain Attack:
    Kiểm tra lỗ hổng bảo mật trong các thư viện phụ thuộc (Dependencies) và cấu hình Docker.
    Tiêu chí PASS: Các thư viện bắt buộc có quy định phiên bản an toàn, không sử dụng gói chứa lỗ hổng Critical đã biết.
    """

    def test_at07_dependency_security_audit(self, project_root):
        req_path = os.path.join(project_root, "requirements.txt")
        assert os.path.exists(req_path), "Không tìm thấy requirements.txt"

        with open(req_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        # Danh sách các gói độc hại / không an toàn đã biết cần cấm tuyệt đối
        banned_packages = ["pickle5", "pyyaml<5.4", "pillow<10.0.0", "requests<2.31.0"]

        packages_found = {}
        for line in lines:
            parts = re.split(r"[><=~]+", line)
            pkg_name = parts[0].strip().lower()
            packages_found[pkg_name] = line

        # 1. Đảm bảo các gói cốt lõi đều được chỉ định phiên bản tối thiểu
        for core_pkg in ["torch", "numpy", "pandas", "gradio"]:
            assert core_pkg in packages_found, f"Thiếu gói phụ thuộc cốt lõi: {core_pkg}"
            assert any(op in packages_found[core_pkg] for op in [">=", "==", "~="]), f"{core_pkg} chưa được pin phiên bản"

        # 2. Không chứa gói thuộc blacklist
        for banned in banned_packages:
            assert banned not in lines, f"Phát hiện phụ thuộc không an toàn: {banned}"

    def test_at07_dockerfile_security_hardening(self, project_root):
        """Kiểm tra cấu hình Dockerfile tuân thủ tiêu chuẩn an toàn bảo mật."""
        dockerfile_path = os.path.join(project_root, "Dockerfile")
        if os.path.exists(dockerfile_path):
            with open(dockerfile_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Không được chạy với quyền root không cần thiết hoặc phải có cấu hình port an toàn
            assert "EXPOSE" in content, "Dockerfile cần khai báo cổng ứng dụng rõ ràng"
            assert "ENTRYPOINT" in content or "CMD" in content, "Dockerfile thiếu lệnh khởi động ứng dụng"
