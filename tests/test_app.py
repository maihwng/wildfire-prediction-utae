"""
test_app.py
-----------
Kiểm tra ứng dụng demo Web Gradio:
1. Kiểm tra danh mục các vụ cháy WildfireCatalog.
2. Kiểm tra hàm chuyển đổi hệ tọa độ pixel sang WGS84 (pixel_to_latlon).
3. Kiểm tra trích xuất đa giác đường bao đám cháy từ mặt nạ (extract_polygons).
4. Kiểm tra sinh mã HTML bản đồ Leaflet vệ tinh tương tác.
5. Kiểm tra các hàm xử lý sự kiện UI: on_select_fire, on_select_date.
6. Kiểm tra chức năng suy luận dự đoán: run_prediction.
7. Kiểm tra khởi dựng giao diện Gradio Blocks (build_app).
"""

import os
import sys

import numpy as np
import pytest

# Đảm bảo đường dẫn gốc của dự án nằm trong sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gradio as gr
from app import (
    CATALOG,
    build_app,
    build_leaflet_map_html,
    extract_polygons,
    format_metadata_html,
    on_select_date,
    on_select_fire,
    pixel_to_latlon,
    render_map_component,
    run_prediction,
)


class TestWildfireApp:
    """Tập hợp các bài kiểm thử cho ứng dụng Web Demo Gradio và GIS."""

    def test_catalog_loaded(self):
        """Kiểm tra danh mục vụ cháy khởi tạo thành công."""
        assert hasattr(CATALOG, "fire_labels")
        assert isinstance(CATALOG.fire_labels, list)
        if len(CATALOG.fire_labels) > 0:
            top_fire = CATALOG.fire_labels[0]
            item = CATALOG.get(top_fire)
            assert item is not None
            assert "tx" in item and "ty" in item
            assert "dates_dict" in item

    def test_pixel_to_latlon_conversion(self):
        """
        Kiểm tra chuyển đổi tọa độ pixel lưới (tx=12, ty=22, Butte County CA)
        ra kinh vĩ độ WGS84 nằm trong ranh giới miền Tây Hoa Kỳ:
        - Vĩ độ Lat trong [31.0, 49.0]
        - Kinh độ Lon trong [-125.0, -102.0]
        """
        lat, lon = pixel_to_latlon(tx=12, ty=22, r=64.0, c=64.0)

        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert 31.0 <= lat <= 49.0, f"Vĩ độ không hợp lệ: {lat}"
        assert -125.0 <= lon <= -102.0, f"Kinh độ không hợp lệ: {lon}"

    def test_polygon_extraction(self):
        """Kiểm tra trích xuất đa giác viền đám cháy từ mặt nạ nhị phân."""
        # 1. Mặt nạ rỗng phải trả về danh sách rỗng
        empty_mask = np.zeros((128, 128), dtype=np.uint8)
        empty_polys = extract_polygons(empty_mask, tx=12, ty=22)
        assert empty_polys == []

        # 2. Mặt nạ có vết cháy nhân tạo
        fire_mask = np.zeros((128, 128), dtype=np.uint8)
        fire_mask[40:80, 40:80] = 1

        polygons = extract_polygons(fire_mask, tx=12, ty=22, sigma=1.0, threshold=0.1)
        assert isinstance(polygons, list)
        assert len(polygons) > 0

        # Kiểm tra cấu trúc mỗi đa giác
        first_poly = polygons[0]
        assert len(first_poly) >= 3, "Đa giác phải có ít nhất 3 đỉnh"
        first_pt = first_poly[0]
        assert len(first_pt) == 2
        pt_lat, pt_lon = first_pt
        assert 31.0 <= pt_lat <= 49.0
        assert -125.0 <= pt_lon <= -102.0

    def test_build_leaflet_map_html(self):
        """Kiểm tra sinh mã nguồn bản đồ Leaflet có đầy đủ các thành phần giao diện."""
        mock_polys = [[[39.75, -121.45], [39.78, -121.42], [39.72, -121.40]]]
        html_code = build_leaflet_map_html(
            center_lat=39.75,
            center_lon=-121.45,
            zoom=11,
            obs_polygons=mock_polys,
            pred_polygons=mock_polys,
            epicenter_lat=39.75,
            epicenter_lon=-121.45,
            show_pred=True,
        )

        assert isinstance(html_code, str)
        # Các thư viện và tài nguyên cốt lõi
        assert "leaflet.js" in html_code
        assert "leaflet.css" in html_code
        assert "Esri World Imagery" in html_code
        # Các Tab và chú thích
        assert "Vùng cháy ngày t" in html_code
        assert "AI dự đoán (ngày t+1)" in html_code
        assert "Vùng cháy quan sát" in html_code
        assert "Vùng cháy AI dự đoán" in html_code
        assert "Tâm cháy" in html_code

    @pytest.mark.skipif(
        len(CATALOG.fire_labels) == 0,
        reason="Bỏ qua nếu chưa nạp được danh mục dữ liệu vụ cháy",
    )
    def test_render_map_component_and_metadata(self):
        """Kiểm tra render iframe bản đồ và khung siêu dữ liệu cho vụ cháy đầu tiên."""
        fire_label = CATALOG.fire_labels[0]
        item = CATALOG.get(fire_label)
        date_keys = list(item["dates_dict"].keys())
        active_date = date_keys[0] if date_keys else "12/11/2018"

        iframe_html, coords = render_map_component(item, active_date, show_pred=False)
        assert "<iframe" in iframe_html
        assert "srcdoc=" in iframe_html
        assert isinstance(coords, tuple) and len(coords) == 2

        meta_html = format_metadata_html(item, coords=coords)
        assert "meta-card" in meta_html
        assert item["fire_code"] in meta_html

    @pytest.mark.skipif(
        len(CATALOG.fire_labels) == 0,
        reason="Bỏ qua nếu chưa nạp được danh mục dữ liệu vụ cháy",
    )
    def test_ui_selection_handlers(self):
        """Kiểm tra sự kiện chọn vụ cháy và ngày quan sát."""
        fire_label = CATALOG.fire_labels[0]
        dropdown_date, next_date, meta_html, map_html = on_select_fire(fire_label)

        assert next_date != ""
        assert "meta-card" in meta_html
        assert "<iframe" in map_html

        item = CATALOG.get(fire_label)
        date_keys = list(item["dates_dict"].keys())
        if date_keys:
            next_d, meta_h, map_h = on_select_date(fire_label, date_keys[0])
            assert next_d != ""
            assert "<iframe" in map_h

    @pytest.mark.skipif(
        len(CATALOG.fire_labels) == 0,
        reason="Bỏ qua nếu chưa nạp được danh mục dữ liệu vụ cháy",
    )
    def test_run_prediction_pipeline(self):
        """Kiểm tra pipeline dự đoán của mô hình U-TAE trên web."""
        fire_label = CATALOG.fire_labels[0]
        item = CATALOG.get(fire_label)
        date_keys = list(item["dates_dict"].keys())
        active_date = date_keys[0] if date_keys else "12/11/2018"

        meta_html, map_html = run_prediction(fire_label, active_date)
        assert "<iframe" in map_html
        assert "meta-card" in meta_html

    def test_build_app_gradio_instance(self):
        """Kiểm tra khởi dựng toàn bộ giao diện Gradio Blocks hoàn chỉnh."""
        demo = build_app()
        assert isinstance(demo, gr.Blocks)

