"""
app.py
------
Bản đồ Dự báo Lan truyền Cháy rừng Miền Tây Hoa Kỳ
Mô hình không gian – thời gian DINOv3 ViT-S/16 · U-TAE trên hạ tầng NVIDIA A10G (ZeroGPU)
Đưa trực tiếp dữ liệu 17 kênh từ data/processed_v1 vào mô hình U-TAE để dự đoán.
Tương thích Hugging Face Spaces & ZeroGPU.
"""

from datetime import datetime, timedelta
import html
import json
import math
import os
import re
import socket
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
import torch

# Hugging Face ZeroGPU compatibility
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False


def gpu_decorator(func):
    """Wrapper hỗ trợ Hugging Face ZeroGPU khi triển khai trên Spaces."""
    if HAS_SPACES:
        return spaces.GPU(func)
    return func


# Setup đường dẫn hệ thống
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src.model import WildfireUTAE

# Cấu hình thiết bị & đường dẫn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(APP_DIR, "checkpoints", "best_model.pth")
DATA_DIR = os.path.join(APP_DIR, "data", "processed_v1")

# Thông số hệ tọa độ lưới cho miền Tây Hoa Kỳ (EPSG:5070 Conus Albers)
BBOX = (-125.0, 31.0, -102.0, 49.0)
CRS = "EPSG:5070"
RES = 375
TILE_PX = 128
SPAN = RES * TILE_PX  # 48,000 meters

_TF_TO_PROJ = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
_TF_TO_WGS = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)

_W, _S, _E, _N = BBOX
_XS, _YS = _TF_TO_PROJ.transform([_W, _W, _E, _E], [_S, _N, _S, _N])
X0 = math.floor(min(_XS) / SPAN) * SPAN
Y0 = math.ceil(max(_YS) / SPAN) * SPAN


def get_tile_bbox_wgs(tx: int, ty: int) -> Tuple[float, float, float, float]:
    """Tính tọa độ WGS84 bounding box (min_lon, min_lat, max_lon, max_lat) từ (tx, ty)."""
    xmin = X0 + tx * SPAN
    ymax = Y0 - ty * SPAN
    xmax = xmin + SPAN
    ymin = ymax - SPAN
    lons, lats = _TF_TO_WGS.transform([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax])
    return min(lons), min(lats), max(lons), max(lats)


def pixel_to_latlon(tx: int, ty: int, r: float, c: float) -> Tuple[float, float]:
    """Chuyển đổi tọa độ pixel (hàng r, cột c) trong tile (tx, ty) sang (vĩ độ Lat, kinh độ Lon) WGS84."""
    xmin = X0 + tx * SPAN
    ymax = Y0 - ty * SPAN
    px_x = xmin + (c + 0.5) * RES
    px_y = ymax - (r + 0.5) * RES
    lon, lat = _TF_TO_WGS.transform(px_x, px_y)
    return round(float(lat), 5), round(float(lon), 5)


def extract_polygons(
    mask: np.ndarray,
    tx: int,
    ty: int,
    sigma: float = 1.0,
    threshold: float = 0.1,
    min_pts: int = 5,
) -> List[List[List[float]]]:
    """Trích xuất danh sách các đa giác viền từ mặt nạ 128x128 sang tọa độ [[lat, lon], ...]."""
    if np.count_nonzero(mask) == 0:
        return []
    smooth = gaussian_filter(mask.astype(float), sigma=sigma)
    binary = (smooth > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if len(cnt) >= min_pts and cv2.contourArea(cnt) > 8:
            pts = [list(pixel_to_latlon(tx, ty, pt[0][1], pt[0][0])) for pt in cnt]
            polys.append(pts)
    return polys


def resolve_us_state(lat: float, lon: float) -> Tuple[str, str]:
    """Xác định mã bang và tên bang Hoa Kỳ từ tọa độ địa lý."""
    if lat >= 46.2 and lon <= -117.0:
        return "WA", "Washington"
    elif 42.0 <= lat < 46.2 and lon <= -116.5:
        return "OR", "Oregon"
    elif 32.5 <= lat < 42.0 and lon <= -120.0:
        return "CA", "California"
    elif 32.5 <= lat < 39.0 and lon <= -114.5 and lat < 35.0 + (lon + 114.5) * (-1.5):
        return "CA", "California"
    elif 35.0 <= lat < 42.0 and -120.0 <= lon < -114.0:
        return "NV", "Nevada"
    elif 42.0 <= lat <= 49.0 and -117.2 <= lon < -111.0:
        return "ID", "Idaho"
    elif 44.4 <= lat <= 49.0 and -116.0 <= lon < -104.0:
        return "MT", "Montana"
    elif 41.0 <= lat < 45.0 and -111.0 <= lon < -104.0:
        return "WY", "Wyoming"
    elif 37.0 <= lat < 42.0 and -114.0 <= lon < -109.0:
        return "UT", "Utah"
    elif 37.0 <= lat < 41.0 and -109.0 <= lon < -102.0:
        return "CO", "Colorado"
    elif 31.3 <= lat < 37.0 and -114.8 <= lon < -109.0:
        return "AZ", "Arizona"
    elif 31.3 <= lat < 37.0 and -109.0 <= lon < -103.0:
        return "NM", "New Mexico"
    elif lon < -118.0:
        return "CA", "California"
    else:
        return "US", "Western US"


# Tên các vụ cháy lớn nổi tiếng gắn với tọa độ Tile
KNOWN_FIRE_NAMES: Dict[Tuple[int, int, int], Tuple[str, str]] = {
    (22, 12, 2018): ("Camp Fire Paradise", "Butte County CA - Rừng thông Sierra Nevada gió Diablo"),
    (16, 9, 2023): ("Smith River Complex", "Del Norte CA - Rừng quốc gia Six Rivers"),
    (17, 10, 2023): ("Happy Camp Complex", "Siskiyou County CA - Rừng quốc gia Klamath"),
    (12, 13, 2023): ("Lookout Fire", "Lane County OR - Rừng quốc gia Willamette"),
    (41, 33, 2023): ("Gila National Forest Fire", "Catron County NM - Rừng quốc gia Gila"),
    (36, 39, 2022): ("Hermits Peak / Calf Canyon", "San Miguel County NM - Vụ cháy lớn nhất lịch sử New Mexico"),
    (12, 27, 2022): ("Moose Fire", "Lemhi County ID - Rừng Salmon-Challis"),
    (11, 23, 2022): ("Cedar Creek Fire", "Lane County OR - Dãy Cascades"),
    (21, 13, 2021): ("Dixie Fire", "Plumas & Butte County CA - Vụ cháy đơn lẻ lớn nhất CA"),
    (31, 15, 2020): ("Creek Fire", "Fresno & Madera County CA - Rừng Sierra National Forest"),
    (21, 10, 2020): ("August Complex", "Mendocino & Tehama CA - Tổ hợp cháy kỷ lục 1 triệu mẫu Anh"),
    (18, 10, 2021): ("Monument Fire", "Trinity County CA - Rừng Shasta-Trinity"),
    (28, 14, 2020): ("SQF Complex", "Tulare County CA - Rừng Sequoia"),
    (3, 18, 2023): ("Sourdough Fire", "Whatcom County WA - Vườn quốc gia North Cascades"),
    (7, 27, 2023): ("Flathead Lake Fire", "Lake County MT - Khu vực hồ Flathead & Dãy Mission"),
}


# Model toàn cục
_MODEL: Optional[WildfireUTAE] = None


def get_model() -> WildfireUTAE:
    global _MODEL
    if _MODEL is None:
        _MODEL = WildfireUTAE(input_dim=17)
        if os.path.exists(CHECKPOINT_PATH):
            try:
                ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
                state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                _MODEL.load_state_dict(state)
                print(f"[app.py] Nạp thành công trọng số U-TAE từ: {CHECKPOINT_PATH}")
            except Exception as e:
                print(f"[app.py] Cảnh báo nạp checkpoint U-TAE: {e}")
        _MODEL.to(DEVICE)
        _MODEL.eval()
    return _MODEL


# =========================================================================
# LẬP CHỈ MỤC TẤT CẢ CÁC VỤ CHÁY TỪ TẬP DỮ LIỆU DATA/PROCESSED_V1
# =========================================================================

class WildfireCatalog:
    """Quản lý và lập chỉ mục toàn bộ các vụ cháy thực tế từ thư mục data/processed_v1."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.catalog: Dict[str, Dict] = {}
        self.fire_labels: List[str] = []
        self._build_catalog()

    def _build_catalog(self):
        splits = ["test", "val", "train"]
        raw_list = []

        for split in splits:
            idx_p = os.path.join(self.data_dir, split, "index.csv")
            if not os.path.exists(idx_p):
                continue
            df = pd.read_csv(idx_p)
            active_df = df[(df["fire_px_target"] > 0) | (df["fire_px_last"] > 0)]

            for fname, grp in active_df.groupby("file"):
                m = re.match(r"tile_(\d+)_(\d+)_(\d+)\.npz", fname)
                if not m:
                    continue
                ty, tx, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))

                min_lon, min_lat, max_lon, max_lat = get_tile_bbox_wgs(tx, ty)
                center_lat = round(float((min_lat + max_lat) / 2), 4)
                center_lon = round(float((min_lon + max_lon) / 2), 4)
                st_code, st_name = resolve_us_state(center_lat, center_lon)

                total_fire_px = int(grp["fire_px_target"].sum())
                max_fire_px = int(grp["fire_px_target"].max())

                # Tìm tên vụ cháy nếu có
                known = KNOWN_FIRE_NAMES.get((ty, tx, yr))
                if known:
                    fire_title = known[0]
                    desc = known[1]
                else:
                    fire_title = f"Vùng cháy Tile {ty:03d}_{tx:03d}"
                    desc = f"Khu vực {st_name} (Tọa độ lưới {ty:03d}_{tx:03d})"

                label = f"[{st_code}] {st_name} {fire_title} ({yr})"
                fire_code = f"US_{st_code}_{re.sub(r'[^A-Z0-9]', '', fire_title.upper())}_{yr}"
                state_split = f"{st_code} | {split.upper()}"
                coords_str = f"({center_lat:.4f}°N, {center_lon:.4f}°E)"

                # Xử lý các ngày có hoạt động cháy (dùng to_dict('records') cho tốc độ tức thì)
                dates_dict: Dict[str, Dict] = {}
                for row in grp.to_dict("records"):
                    td_str = str(row["target_date"])
                    try:
                        target_dt = datetime.strptime(td_str, "%Y-%m-%d")
                        curr_dt = target_dt - timedelta(days=1)
                        date_t_str = curr_dt.strftime("%d/%m/%Y")
                        date_next_str = target_dt.strftime("%d/%m/%Y")
                    except Exception:
                        continue

                    fire_target = int(row.get("fire_px_target", 0))
                    fire_last = int(row.get("fire_px_last", 0))

                    dates_dict[date_t_str] = {
                        "date_obs": date_t_str,
                        "date_pred": date_next_str,
                        "target_date_iso": td_str,
                        "input_start": int(row["input_start"]),
                        "input_end": int(row["input_end"]),
                        "target_idx": int(row["target"]),
                        "fire_px_target": fire_target,
                        "fire_px_last": fire_last,
                    }

                if not dates_dict:
                    continue

                date_keys = list(dates_dict.keys())
                # Chọn ngày mặc định có số lượng pixel cháy lớn nhất
                best_date = max(date_keys, key=lambda k: dates_dict[k]["fire_px_last"])

                # Ưu tiên đặt ngày 12/11/2018 cho Camp Fire nếu có
                if (ty, tx, yr) == (22, 12, 2018) and "12/11/2018" in dates_dict:
                    best_date = "12/11/2018"
                    center_lat, center_lon = 39.8100, -121.4370
                    coords_str = "(39.8100°N, -121.4370°E)"

                raw_list.append({
                    "label": label,
                    "file": fname,
                    "split": split,
                    "tx": tx,
                    "ty": ty,
                    "year": yr,
                    "fire_code": fire_code,
                    "state_split": state_split,
                    "coords_str": coords_str,
                    "epicenter": (center_lat, center_lon),
                    "desc": desc,
                    "dates_dict": dates_dict,
                    "date_keys": date_keys,
                    "active_date": best_date,
                    "total_fire_px": total_fire_px,
                    "max_fire_px": max_fire_px,
                    "is_camp_fire": (ty, tx, yr) == (22, 12, 2018),
                })

        # Sắp xếp: Đặt Camp Fire 2018 lên đầu, sau đó sắp xếp theo mức độ cháy mạnh nhất
        raw_list.sort(key=lambda x: (not x["is_camp_fire"], -x["total_fire_px"]))

        for item in raw_list:
            self.catalog[item["label"]] = item
            self.fire_labels.append(item["label"])

        print(f"[WildfireCatalog] Đã lập chỉ mục {len(self.fire_labels)} vụ cháy từ toàn bộ tập dữ liệu data/processed_v1.")

    def get(self, label: str) -> Optional[Dict]:
        return self.catalog.get(label)


CATALOG = WildfireCatalog(DATA_DIR)


# =========================================================================
# XÂY DỰNG BẢN ĐỒ LEAFLET TRONG IFRAME HTML
# =========================================================================

def build_leaflet_map_html(
    center_lat: float,
    center_lon: float,
    zoom: int,
    obs_polygons: List[List[List[float]]],
    pred_polygons: List[List[List[float]]],
    epicenter_lat: float,
    epicenter_lon: float,
    show_pred: bool = False,
) -> str:
    """Tạo mã nguồn HTML hoàn chỉnh chứa bản đồ Leaflet vệ tinh tương tác."""
    obs_json = json.dumps(obs_polygons)
    pred_json = json.dumps(pred_polygons)
    initial_show_pred_js = "true" if show_pred else "false"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body, html {{
      width: 100%;
      height: 100%;
      background: #0c1524;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      overflow: hidden;
    }}
    #app-container {{
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      padding: 0;
    }}
    /* Thanh chuyển đổi 2 Tab phía trên bản đồ */
    #tabs-bar {{
      display: flex;
      gap: 12px;
      margin-bottom: 12px;
      padding: 0 2px;
    }}
    .tab-btn {{
      flex: 1;
      background: #152238;
      border: 1px solid #23354e;
      border-radius: 6px;
      color: #8fa3bf;
      padding: 10px 16px;
      font-size: 13.5px;
      font-weight: 500;
      cursor: pointer;
      text-align: center;
      transition: all 0.2s ease;
      user-select: none;
    }}
    .tab-btn:hover {{
      background: #1c2e4a;
      color: #ffffff;
    }}
    .tab-btn.active {{
      background: #1a2a44;
      border-color: #3b82f6;
      color: #ffffff;
      font-weight: 600;
      box-shadow: 0 0 12px rgba(59, 130, 246, 0.25);
    }}
    /* Khung bản đồ Leaflet */
    #map {{
      flex: 1;
      width: 100%;
      border-radius: 8px;
      border: 1px solid #23354e;
      overflow: hidden;
      min-height: 480px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }}
    /* Hộp Chú thích Legend góc dưới phải */
    .map-legend {{
      background: rgba(12, 21, 36, 0.88);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 6px;
      padding: 10px 14px;
      color: #ffffff;
      font-size: 12px;
      line-height: 1.8;
      box-shadow: 0 4px 16px rgba(0,0,0,0.5);
      backdrop-filter: blur(4px);
      user-select: none;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 4px;
    }}
    .legend-row:last-child {{
      margin-bottom: 0;
    }}
    .legend-box-obs {{
      width: 14px;
      height: 14px;
      border: 2px solid #f97316;
      background: rgba(249, 115, 22, 0.1);
      border-radius: 2px;
      display: inline-block;
    }}
    .legend-box-pred {{
      width: 14px;
      height: 14px;
      background: #ea580c;
      border-radius: 2px;
      display: inline-block;
    }}
    .legend-dot-epicenter {{
      width: 10px;
      height: 10px;
      background: #ef4444;
      border: 2px solid #ffffff;
      border-radius: 50%;
      display: inline-block;
      box-shadow: 0 0 5px #ef4444;
    }}
    /* Tùy biến thanh điều khiển Leaflet khớp màu tối */
    .leaflet-bar {{
      border: 1px solid rgba(255, 255, 255, 0.2) !important;
      border-radius: 6px !important;
      overflow: hidden;
    }}
    .leaflet-bar a {{
      background-color: #152238 !important;
      color: #ffffff !important;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    }}
    .leaflet-bar a:hover {{
      background-color: #1e3252 !important;
    }}
    .leaflet-control-scale-line {{
      background: rgba(12, 21, 36, 0.8) !important;
      border-color: #ffffff !important;
      color: #ffffff !important;
      font-size: 11px !important;
      padding: 2px 5px !important;
    }}
    .leaflet-control-attribution {{
      background: rgba(12, 21, 36, 0.8) !important;
      color: #94a3b8 !important;
      font-size: 10px !important;
    }}
    .leaflet-control-attribution a {{
      color: #cbd5e1 !important;
    }}
  </style>
</head>
<body>
  <div id="app-container">
    <div id="tabs-bar">
      <div id="tab-obs" class="tab-btn" onclick="switchTab('obs')">Vùng cháy ngày t</div>
      <div id="tab-pred" class="tab-btn" onclick="switchTab('pred')">AI dự đoán (ngày t+1)</div>
    </div>
    <div id="map"></div>
  </div>

  <script>
    var map = L.map('map', {{
      zoomControl: false,
      attributionControl: false
    }}).setView([{center_lat}, {center_lon}], {zoom});

    L.control.zoom({{ position: 'topright' }}).addTo(map);
    L.control.scale({{ position: 'bottomleft', imperial: true, metric: true }}).addTo(map);

    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 18,
      attribution: 'Leaflet | Esri World Imagery'
    }}).addTo(map);

    var attrControl = L.control({{ position: 'bottomright' }});
    attrControl.onAdd = function() {{
      var div = L.DomUtil.create('div', 'leaflet-control-attribution');
      div.innerHTML = '<a href="https://leafletjs.com" target="_blank">Leaflet</a> | Esri World Imagery';
      return div;
    }};
    attrControl.addTo(map);

    var legend = L.control({{ position: 'bottomright' }});
    legend.onAdd = function() {{
      var div = L.DomUtil.create('div', 'map-legend');
      div.innerHTML = `
        <div class="legend-row"><span class="legend-box-obs"></span> Vùng cháy quan sát</div>
        <div class="legend-row"><span class="legend-box-pred"></span> Vùng cháy AI dự đoán</div>
        <div class="legend-row"><span class="legend-dot-epicenter"></span> Tâm cháy</div>
      `;
      return div;
    }};
    legend.addTo(map);

    var obsLayerGroup = L.layerGroup().addTo(map);
    var predLayerGroup = L.layerGroup();
    var epicenterLayerGroup = L.layerGroup().addTo(map);

    var obsPolygons = {obs_json};
    obsPolygons.forEach(function(pts) {{
      L.polygon(pts, {{
        color: '#f97316',
        weight: 2.6,
        fillColor: '#ea580c',
        fillOpacity: 0.12
      }}).addTo(obsLayerGroup);
    }});

    var predPolygons = {pred_json};
    predPolygons.forEach(function(pts) {{
      L.polygon(pts, {{
        color: '#ff5400',
        weight: 2,
        fillColor: '#ea580c',
        fillOpacity: 0.70
      }}).addTo(predLayerGroup);
    }});

    L.circleMarker([{epicenter_lat}, {epicenter_lon}], {{
      radius: 7,
      fillColor: '#ef4444',
      color: '#ffffff',
      weight: 2.5,
      fillOpacity: 1.0
    }}).addTo(epicenterLayerGroup);

    // Tự động căn chỉnh tầm nhìn vừa vặn và bao quát chính xác toàn bộ vùng cháy
    var allPolys = obsPolygons.concat(predPolygons);
    if (allPolys.length > 0) {{
      var bounds = L.latLngBounds();
      allPolys.forEach(function(pts) {{
        pts.forEach(function(pt) {{
          bounds.extend(pt);
        }});
      }});
      bounds.extend([{epicenter_lat}, {epicenter_lon}]);
      if (bounds.isValid()) {{
        map.fitBounds(bounds, {{ padding: [55, 55], maxZoom: 13 }});
      }}
    }}

    var showPredInitial = {initial_show_pred_js};
    if (showPredInitial) {{
      predLayerGroup.addTo(map);
      document.getElementById('tab-pred').classList.add('active');
      document.getElementById('tab-obs').classList.remove('active');
    }} else {{
      document.getElementById('tab-obs').classList.add('active');
      document.getElementById('tab-pred').classList.remove('active');
    }}

    function switchTab(mode) {{
      if (mode === 'obs') {{
        if (map.hasLayer(predLayerGroup)) {{
          map.removeLayer(predLayerGroup);
        }}
        document.getElementById('tab-obs').classList.add('active');
        document.getElementById('tab-pred').classList.remove('active');
      }} else {{
        if (!map.hasLayer(predLayerGroup)) {{
          map.addLayer(predLayerGroup);
        }}
        document.getElementById('tab-pred').classList.add('active');
        document.getElementById('tab-obs').classList.remove('active');
      }}
    }}
  </script>
</body>
</html>
"""
    return html_content


def render_map_component(
    fire_item: Dict,
    date_obs_str: str,
    pred_polygons: Optional[List[List[List[float]]]] = None,
    show_pred: bool = False,
) -> Tuple[str, Tuple[float, float]]:
    """Đọc dữ liệu từ file tile .npz, tính tâm cháy thực tế và sinh khung iframe Leaflet tương tác."""
    split = fire_item["split"]
    fname = fire_item["file"]
    tx = fire_item["tx"]
    ty = fire_item["ty"]

    npz_path = os.path.join(DATA_DIR, split, fname)
    if not os.path.exists(npz_path):
        return f"<div style='color: #ef4444; padding: 20px;'>Không tìm thấy tệp {fname}</div>", fire_item["epicenter"]

    date_entry = fire_item.get("dates_dict", {}).get(date_obs_str)
    with np.load(npz_path) as z:
        if date_entry:
            obs_mask = z["fire"][date_entry["input_end"]]
        else:
            obs_mask = z["fire"][min(56, len(z["fire"]) - 1)]

    # Trích xuất viền vùng cháy quan sát ngày t
    obs_polygons = extract_polygons(obs_mask, tx, ty, sigma=1.0, threshold=0.1)

    if pred_polygons is None:
        pred_polygons = []

    # Tính tọa độ tâm cháy (epicenter) chính xác từ mặt nạ cháy thực tế của ngày t (không bị lệch)
    ys, xs = np.where(obs_mask > 0)
    if len(ys) > 0:
        center_r = float(np.mean(ys))
        center_c = float(np.mean(xs))
        dists = (ys - center_r) ** 2 + (xs - center_c) ** 2
        best_idx = np.argmin(dists)
        epicenter_lat, epicenter_lon = pixel_to_latlon(tx, ty, ys[best_idx], xs[best_idx])
    else:
        epicenter_lat, epicenter_lon = fire_item["epicenter"]

    zoom = 11

    map_html = build_leaflet_map_html(
        center_lat=epicenter_lat,
        center_lon=epicenter_lon,
        zoom=zoom,
        obs_polygons=obs_polygons,
        pred_polygons=pred_polygons,
        epicenter_lat=epicenter_lat,
        epicenter_lon=epicenter_lon,
        show_pred=show_pred,
    )

    escaped_srcdoc = html.escape(map_html, quote=True)
    iframe_markup = f"""
    <iframe 
        srcdoc="{escaped_srcdoc}" 
        style="width: 100%; height: 560px; border: none; border-radius: 8px; display: block;"
        allowfullscreen>
    </iframe>
    """
    return iframe_markup, (epicenter_lat, epicenter_lon)


def format_metadata_html(fire_item: Dict, coords: Optional[Tuple[float, float]] = None) -> str:
    """Định dạng khung thông tin siêu dữ liệu phía dưới nút bấm."""
    coords_str = f"({coords[0]:.4f}°N, {coords[1]:.4f}°E)" if coords else fire_item["coords_str"]
    return f"""
    <div class="meta-card">
        <div class="meta-code">{fire_item['fire_code']}</div>
        <div class="meta-split">{fire_item['state_split']}</div>
        <div class="meta-coords">{coords_str}</div>
        <div class="meta-desc">{fire_item['desc']}</div>
    </div>
    """


# =========================================================================
# LOGIC XỬ LÝ SỰ KIỆN GIAO DIỆN
# =========================================================================

def on_select_fire(fire_label: str):
    """Khi người dùng chọn vụ cháy khác từ danh sách."""
    item = CATALOG.get(fire_label)
    if not item:
        return gr.Dropdown(), "", "", ""

    dates_dict = item.get("dates_dict", {})
    date_keys = list(dates_dict.keys())

    active_date = item.get("active_date", "")
    if active_date not in dates_dict and date_keys:
        active_date = date_keys[0]

    date_entry = dates_dict.get(active_date, {})
    next_date = date_entry.get("date_pred", "")

    # Hiển thị ban đầu vùng cháy ngày t với tọa độ tâm chuẩn xác
    map_html, real_coords = render_map_component(item, active_date, show_pred=False)
    meta_html = format_metadata_html(item, coords=real_coords)

    return (
        gr.Dropdown(choices=date_keys, value=active_date),
        next_date,
        meta_html,
        map_html,
    )


def on_select_date(fire_label: str, date_obs_str: str):
    """Khi người dùng chọn một ngày quan sát khác."""
    item = CATALOG.get(fire_label)
    if not item or not date_obs_str:
        return "", "", ""

    date_entry = item.get("dates_dict", {}).get(date_obs_str, {})
    next_date = date_entry.get("date_pred", "")
    map_html, real_coords = render_map_component(item, date_obs_str, show_pred=False)
    meta_html = format_metadata_html(item, coords=real_coords)

    return next_date, meta_html, map_html


@gpu_decorator
def run_prediction(fire_label: str, date_obs_str: str) -> Tuple[str, str]:
    """
    Đưa chuỗi dữ liệu 17 kênh từ data/processed_v1 vào mô hình Spatio-Temporal U-TAE
    để suy luận dự đoán vùng lan truyền cháy rừng ngày tiếp theo.
    """
    item = CATALOG.get(fire_label)
    if not item or not date_obs_str:
        return "<div class='meta-card'>Lỗi tham số</div>", "<div style='color: #ef4444;'>Lỗi tham số đầu vào.</div>"

    date_entry = item.get("dates_dict", {}).get(date_obs_str)
    if not date_entry:
        map_html, real_coords = render_map_component(item, date_obs_str, show_pred=False)
        meta_html = format_metadata_html(item, coords=real_coords)
        return meta_html, map_html

    # 1. Đọc đúng 17 kênh đầu vào từ tập dữ liệu
    npz_path = os.path.join(DATA_DIR, item["split"], item["file"])
    with np.load(npz_path) as z:
        s = date_entry["input_start"]
        e = date_entry["input_end"] + 1  # 5 time steps (T=5)

        dyn_cont = z["dyn_cont"][s:e]
        fire = z["fire"][s:e, np.newaxis]
        w_mask = z["weather_mask"][s:e, np.newaxis]
        v_mask = z["vegetation_mask"][s:e, np.newaxis]

        # Ghép 17 kênh đầu vào
        c0_12 = dyn_cont[:, 0:13]
        c13 = fire
        c14 = dyn_cont[:, 13:14]
        c15 = w_mask
        c16 = v_mask

        x_np = np.concatenate([c0_12, c13, c14, c15, c16], axis=1).astype(np.float32)

    # 2. Đưa dữ liệu vào mô hình U-TAE để suy luận dự đoán
    model = get_model()
    x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # 3. Tính toán mặt nạ dự đoán lan truyền trực tiếp từ đầu ra của U-TAE
    p_max = float(probs.max())
    if p_max > 0.505:
        # Ngưỡng động trích xuất vùng cháy lan mà mô hình U-TAE kích hoạt mạnh nhất
        cutoff = 0.50 + 0.35 * (p_max - 0.50)
        pred_binary = (probs >= cutoff).astype(np.uint8)
    else:
        pred_binary = (probs > 0.50).astype(np.uint8)

    # 4. Trích xuất đa giác dự đoán từ kết quả U-TAE
    pred_polygons = extract_polygons(pred_binary, item["tx"], item["ty"], sigma=1.3, threshold=0.08)

    # 5. Cập nhật bản đồ hiển thị kết quả AI dự đoán và tọa độ tâm chuẩn xác
    map_html, real_coords = render_map_component(item, date_obs_str, pred_polygons=pred_polygons, show_pred=True)
    meta_html = format_metadata_html(item, coords=real_coords)
    return meta_html, map_html


# =========================================================================
# CSS TUỲ BIẾN ĐẠT ĐỘ KHỚP HOÀN HẢO THEO ẢNH GIAO DIỆN MỚI
# =========================================================================

CUSTOM_CSS = """
/* Toàn bộ nền trang web màu đen xanh hải quân sâu */
body, .gradio-container {
    background-color: #0c1424 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    color: #ffffff !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 16px 20px !important;
}

/* Phần Tiêu đề Header & Huy hiệu mtthwng */
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding: 0 4px;
}
.header-left h1 {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0 !important;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.header-left p {
    font-size: 13px !important;
    color: #7b8ea3 !important;
    margin: 5px 0 0 0 !important;
}
.user-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background-color: #ffffff;
    padding: 6px 16px;
    border-radius: 9999px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}
.badge-dot {
    width: 10px;
    height: 10px;
    background-color: #eab308;
    border-radius: 50%;
}
.badge-text {
    color: #1e293b;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: -0.2px;
}

/* Thẻ khung điều khiển bên trái */
.control-card {
    background-color: #152238 !important;
    border: 1px solid #23354e !important;
    border-radius: 8px !important;
    padding: 22px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
}

/* Biểu tượng bánh răng góc trên thẻ */
.gear-header {
    color: #8fa3bf;
    font-size: 18px;
    margin-bottom: 12px;
    cursor: default;
}

/* Nhãn cho dropdown và ô nhập */
.control-card label span, .control-card .block-title {
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: #cbd5e1 !important;
    margin-bottom: 6px !important;
    display: block !important;
}

/* Ô nhập liệu và dropdown có nền trắng, chữ đen chuẩn theo ảnh thiết kế */
.control-card .wrap,
.control-card select,
.control-card input,
.control-card .secondary-wrap,
.control-card textarea {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

.control-card .wrap * {
    color: #0f172a !important;
}

/* Nút bấm Dự đoán ngày tiếp theo (ZeroGPU) màu xanh dương đậm */
.btn-predict {
    background-color: #0070f3 !important;
    color: #ffffff !important;
    font-size: 14.5px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 13px 20px !important;
    width: 100% !important;
    margin-top: 18px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(0, 112, 243, 0.4) !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 8px !important;
}
.btn-predict:hover {
    background-color: #0060df !important;
    box-shadow: 0 6px 20px rgba(0, 112, 243, 0.5) !important;
    transform: translateY(-1px) !important;
}
.btn-predict:active {
    transform: translateY(0) !important;
}

/* Khối thông tin siêu dữ liệu phía dưới nút bấm */
.meta-card {
    margin-top: 24px;
    padding: 4px 2px;
    color: #8fa3bf;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    font-size: 12.5px;
    line-height: 1.6;
}
.meta-code {
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: 0.5px;
}
.meta-split {
    color: #94a3b8;
    font-weight: 500;
}
.meta-coords {
    color: #94a3b8;
}
.meta-desc {
    color: #7b8ea3;
    margin-top: 4px;
    font-size: 12px;
}

/* Cột bản đồ bên phải */
.map-column {
    padding: 0 !important;
}

@media (max-width: 840px) {
    .header-container {
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
    }
}
"""


# =========================================================================
# KHỞI DỰNG GIAO DIỆN ỨNG DỤNG GRADIO
# =========================================================================

def build_app():
    # Lấy vụ cháy mặc định đầu tiên: Camp Fire Paradise (2018)
    initial_fire = CATALOG.fire_labels[0]
    initial_item = CATALOG.get(initial_fire)

    initial_dates = list(initial_item.get("dates_dict", {}).keys())
    initial_date_val = initial_item.get("active_date", "12/11/2018")
    initial_next_date = initial_item.get("dates_dict", {}).get(initial_date_val, {}).get("date_pred", "13/11/2018")

    initial_map, initial_coords = render_map_component(initial_item, initial_date_val, show_pred=False)
    initial_meta = format_metadata_html(initial_item, coords=initial_coords)

    with gr.Blocks(title="Bản đồ Dự báo Lan truyền Cháy rừng Miền Tây Hoa Kỳ") as demo:
        # Tiêm mã CSS tuỳ biến
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # Thanh Header đỉnh trang
        gr.HTML(
            """
            <div class="header-container">
                <div class="header-left">
                    <h1>🔥 Bản đồ Dự báo Lan truyền Cháy rừng Miền Tây Hoa Kỳ</h1>
                    <p>Mô hình không gian – thời gian DINOv3 ViT-S/16 · U-TAE trên hạ tầng NVIDIA A10G (ZeroGPU)</p>
                </div>
                <div class="header-right">
                    <div class="user-badge">
                        <span class="badge-dot"></span>
                        <span class="badge-text">mtthwng / wildfire-prediction</span>
                    </div>
                </div>
            </div>
            """
        )

        # Bố cục 2 cột chính
        with gr.Row():
            # Cột trái: Khung điều khiển
            with gr.Column(scale=4, elem_classes=["control-card"]):
                gr.HTML('<div class="gear-header">⚙</div>')

                dropdown_fire = gr.Dropdown(
                    label="Chọn vụ cháy (11 Bang)",
                    choices=CATALOG.fire_labels,
                    value=initial_fire,
                    interactive=True,
                )

                with gr.Row():
                    dropdown_date = gr.Dropdown(
                        label="Ngày quan sát (t)",
                        choices=initial_dates,
                        value=initial_date_val,
                        interactive=True,
                        scale=1,
                    )
                    txt_next_date = gr.Textbox(
                        label="Ngày dự đoán (t + 1)",
                        value=initial_next_date,
                        interactive=False,
                        scale=1,
                    )

                btn_predict = gr.Button(
                    "🔥 Dự đoán ngày tiếp theo",
                    elem_classes=["btn-predict"],
                    variant="primary",
                )

                meta_output = gr.HTML(value=initial_meta)

            # Cột phải: Bản đồ tương tác Leaflet
            with gr.Column(scale=6, elem_classes=["map-column"]):
                map_output = gr.HTML(value=initial_map)

        # Gán sự kiện tương tác
        dropdown_fire.change(
            fn=on_select_fire,
            inputs=[dropdown_fire],
            outputs=[dropdown_date, txt_next_date, meta_output, map_output],
        )

        dropdown_date.change(
            fn=on_select_date,
            inputs=[dropdown_fire, dropdown_date],
            outputs=[txt_next_date, meta_output, map_output],
        )

        btn_predict.click(
            fn=run_prediction,
            inputs=[dropdown_fire, dropdown_date],
            outputs=[meta_output, map_output],
        )

    return demo


def find_free_port(start_port: int = 7860) -> int:
    """Tìm cổng mạng trống bắt đầu từ start_port để tránh lỗi OSError port in use."""
    for p in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start_port


demo = build_app()

if __name__ == "__main__":
    host = "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"
    target_port = int(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or 7860)

    # Kiểm tra cổng nếu đang chạy cục bộ
    if not os.getenv("SPACE_ID"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", target_port)) == 0:
                target_port = find_free_port(target_port)

    print(f"Khởi chạy ứng dụng Web tại: http://{host}:{target_port}")
    demo.launch(
        server_name=host,
        server_port=target_port,
        share=False,
    )
