import pytest
import os
import gradio as gr

from app import (
    CATALOG,
    run_prediction,
    on_select_fire,
    on_select_date,
    build_app,
    build_leaflet_map_html,
    format_metadata_html
)


class TestSystemIntegration:
    """
    Kiểm thử cấp độ Hệ thống (System Test) cho toàn bộ ứng dụng AI.
    Bao gồm Test Case: TC04.
    """

    def test_tc04_full_system_pipeline(self):
        """
        TC04 - Kiểm thử toàn bộ hệ thống (System Level):
        Khởi tạo hệ thống, truy xuất danh mục ca cháy, chọn ngày quan sát,
        chạy toàn bộ pipeline dự báo lan truyền cháy rừng và hiển thị kết quả.
        Kết quả mong đợi: Dữ liệu được xử lý trọn vẹn, mô hình sinh bản đồ Leaflet
        và bảng siêu dữ liệu diện tích rủi ro đầy đủ.
        """
        # 1. Kiểm tra danh mục ca cháy từ CATALOG
        assert len(CATALOG.fire_labels) > 0, "Danh mục ca cháy trống"
        first_fire_label = CATALOG.fire_labels[0]
        assert first_fire_label is not None

        # 2. Kiểm tra cập nhật danh sách ngày quan sát
        fire_item = CATALOG.get(first_fire_label)
        assert fire_item is not None
        assert "date_keys" in fire_item and len(fire_item["date_keys"]) > 0
        test_date = fire_item["date_keys"][0]

        # 3. Chạy toàn bộ pipeline dự báo (End-to-End System Run)
        metadata_html, map_html = run_prediction(first_fire_label, test_date)

        # 4. Kiểm tra tính toàn vẹn của kết quả
        assert isinstance(map_html, str) and len(map_html) > 100, "Bản đồ HTML không hợp lệ"
        assert "leaflet" in map_html.lower() or "map" in map_html.lower(), "Thiếu thành phần bản đồ"
        assert isinstance(metadata_html, str) and len(metadata_html) > 20, "Siêu dữ liệu không đúng định dạng chuỗi"
        assert "meta-card" in metadata_html

        # 5. Kiểm tra khởi tạo ứng dụng Gradio
        demo = build_app()
        assert isinstance(demo, gr.Blocks), "Ứng dụng Gradio không khởi tạo được gr.Blocks"
