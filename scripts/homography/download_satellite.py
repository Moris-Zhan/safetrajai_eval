"""
依鏡頭設定的經緯度下載 Google Static Maps 衛星圖（change `camera-homography-georef`，design §5）。

用法：
    python scripts/homography/download_satellite.py --camera C000025 --purpose verify
    python scripts/homography/download_satellite.py --camera C000999 --purpose basemap
    python scripts/homography/download_satellite.py --camera C000025 --purpose basemap --replace

- verify：存到 models/homography/verification/<camera>/，只供比例尺檢查使用；
  優先使用 GOOGLE_MAPS_DEMO_API_KEY，沒有時用 GOOGLE_MAPS_API_KEY。
- basemap：存到 models/homography/satellite_images/，只接受 GOOGLE_MAPS_API_KEY
  （Maps Demo Key 依條款只能用於測試與原型開發）。既有底圖不會被覆寫，除非加 --replace；
  替換時舊檔移到 temp/，鏡頭設定的 matrix_tag 改記為 needs_recalibration。
金鑰從環境變數或 repo 根目錄的 .env 讀取，不會印出。
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import requests

from safetrajai.collision.homography.camera_config import CAMERA_DIR, camera_config_path, load_camera_config
from safetrajai.collision.homography.satellite import PURPOSES, download_satellite
from safetrajai.paths import PROJECT_ROOT
from safetrajai.secrets import google_maps_key

SATELLITE_DIR = PROJECT_ROOT / "models/homography/satellite_images"
VERIFICATION_DIR = PROJECT_ROOT / "models/homography/verification"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Google Static Maps satellite images for cameras.")
    parser.add_argument("--camera", nargs="+", required=True, help="鏡頭編號（可多個）")
    parser.add_argument("--purpose", choices=PURPOSES, required=True)
    parser.add_argument("--replace", action="store_true", help="basemap 專用：替換既有底圖")
    parser.add_argument("--camera-dir", type=Path, default=CAMERA_DIR)
    parser.add_argument("--satellite-dir", type=Path, default=SATELLITE_DIR)
    parser.add_argument("--verification-dir", type=Path, default=VERIFICATION_DIR)
    args = parser.parse_args()

    key = google_maps_key(args.purpose)
    print(f"使用金鑰：{key}")
    for camera in args.camera:
        cfg = load_camera_config(camera_config_path(args.camera_dir, camera))
        out = download_satellite(
            cfg, args.purpose, key=key, satellite_dir=args.satellite_dir,
            verification_dir=args.verification_dir, replace=args.replace,
            get=requests.get, today=date.today(),
        )
        print(f"[{camera}] {out}")


if __name__ == "__main__":
    main()
