"""
鏡頭衛星底圖的比例尺檢查，結果寫入 config/homography/<camera>.yaml 的 scale_check
（change `camera-homography-georef`，spec「比例尺檢查」、design §8）。

用法：
    python scripts/homography/check_homography_scale.py --camera C000025 \
        --points 612,540 655,541 --reference-m 3.5 --feature "車道寬"
    # 不下載、只做已知長度檢查：加 --skip-image-match
    # 只印結果、不寫入設定：加 --dry-run

1. 已知長度：--points 是衛星底圖上的兩個端點（像素），以 tag 推得的 m/px（含 cos(緯度)）
   換算後，必須落在 --reference-m 的 ±15% 以內。
2. 影像比對（有 GOOGLE_MAPS_DEMO_API_KEY 或 GOOGLE_MAPS_API_KEY 時）：以鏡頭經緯度下載
   verify 用途的底圖，與既有底圖做 ORB＋RANSAC 相似變換，縮放比例必須在 1±2% 以內。
   沒有金鑰時略過這一項，只依第 1 項判定。
兩項都通過才寫入 passed，否則寫入 failed；該鏡頭的公尺換算只在 passed 時可用。
"""

from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path
from typing import Tuple

import cv2
import requests

from safetrajai.collision.homography.camera_config import CAMERA_DIR, camera_config_path, load_camera_config, update_camera_config_fields
from safetrajai.collision.homography.georef import nominal_meters_per_pixel
from safetrajai.collision.homography.satellite import download_satellite, image_tag
from safetrajai.collision.homography.scale_match import (
    ScaleMatchError,
    decide_scale_check,
    estimate_similarity,
    known_length_ok,
)
from safetrajai.paths import PROJECT_ROOT
from safetrajai.secrets import MissingApiKeyError, google_maps_key

SATELLITE_DIR = PROJECT_ROOT / "models/homography/satellite_images"
VERIFICATION_DIR = PROJECT_ROOT / "models/homography/verification"


def parse_point(text: str) -> Tuple[float, float]:
    x, y = text.split(",")
    return float(x), float(y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the metre scale of a camera's satellite base map.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--points", nargs=2, type=parse_point, required=True, metavar="X,Y",
                        help="衛星底圖上已知長度地物的兩個端點（像素）")
    parser.add_argument("--reference-m", type=float, required=True, help="該地物的實際長度（公尺）")
    parser.add_argument("--feature", default="", help="地物描述，例如「車道寬」")
    parser.add_argument("--skip-image-match", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只印結果，不寫入鏡頭設定")
    parser.add_argument("--camera-dir", type=Path, default=CAMERA_DIR)
    parser.add_argument("--satellite-dir", type=Path, default=SATELLITE_DIR)
    parser.add_argument("--verification-dir", type=Path, default=VERIFICATION_DIR)
    args = parser.parse_args()

    cfg_path = camera_config_path(args.camera_dir, args.camera)
    cfg = load_camera_config(cfg_path)
    mpp = nominal_meters_per_pixel(cfg)
    if mpp is None or cfg.matrix_tag is None:
        raise SystemExit(f"{cfg_path}: 需要 matrix_tag、satellite 與 latlon 才能檢查比例尺")

    (x1, y1), (x2, y2) = args.points
    measured_m = math.hypot(x2 - x1, y2 - y1) * mpp
    length_ok = known_length_ok(measured_m, args.reference_m)
    print(f"[{args.camera}] m/px={mpp:.6f}  已知長度（{args.feature}）：量測 {measured_m:.3f} m，"
          f"參考 {args.reference_m:.3f} m，相對誤差 {(measured_m / args.reference_m - 1) * 100:+.1f}% "
          f"→ {'通過' if length_ok else '不通過'}（±15%）")

    image_scale = None
    image_match_failed = False
    if args.skip_image_match:
        print(f"[{args.camera}] 影像比對：略過（--skip-image-match）")
    else:
        try:
            key = google_maps_key("verify")
        except MissingApiKeyError:
            key = None
            print(f"[{args.camera}] 影像比對：略過（沒有 Google Maps 金鑰）")
        if key is not None:
            verify_png = download_satellite(
                cfg, "verify", key=key, satellite_dir=args.satellite_dir,
                verification_dir=args.verification_dir, get=requests.get, today=date.today(),
            )
            existing = cv2.imread(str(args.satellite_dir / f"{image_tag(cfg)}.png"))
            try:
                est = estimate_similarity(existing, cv2.imread(str(verify_png)))
            except ScaleMatchError as exc:
                image_match_failed = True
                print(f"[{args.camera}] 影像比對（{key}）：無法估計（{exc}），視為不通過")
            else:
                image_scale = est.scale
                print(f"[{args.camera}] 影像比對（{key}）：scale={est.scale:.4f} rotation={est.rotation_deg:+.2f}° "
                      f"shift=({est.tx_px:+.1f}, {est.ty_px:+.1f}) px inliers={est.inliers}/{est.matches}")

    result = decide_scale_check(length_ok, image_scale, image_match_failed)
    print(f"[{args.camera}] scale_check = {result}")
    if not args.dry_run:
        update_camera_config_fields(cfg_path, {"scale_check": result, "scale_check_date": date.today().isoformat()})
        print(f"[{args.camera}] 已寫入 {cfg_path}")


if __name__ == "__main__":
    main()
