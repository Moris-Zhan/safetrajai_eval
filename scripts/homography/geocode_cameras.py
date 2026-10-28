"""
取得鏡頭路口中心的經緯度（change `camera-homography-georef`，design §6）。

用法：
    python scripts/homography/geocode_cameras.py                      # 查詢全部鏡頭，寫候選 CSV（會連網）
    python scripts/homography/geocode_cameras.py --cameras C000025 C000085
    python scripts/homography/geocode_cameras.py --apply              # 只把 status=ok 的候選寫進鏡頭設定
    python scripts/homography/geocode_cameras.py --apply --accept C000033   # 另外套用使用者確認過的候選

查詢先用 OpenStreetMap Overpass（不需要金鑰）；查不到時，若 .env 或環境變數有
GOOGLE_MAPS_DEMO_API_KEY／GOOGLE_MAPS_API_KEY，以 Google Geocoding 備援（結果一律 needs_review）。
沒有 --apply 時不修改任何鏡頭設定。
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from datetime import date
from pathlib import Path

import requests

from safetrajai.collision.homography.camera_config import CAMERA_DIR, list_camera_configs
from safetrajai.collision.homography.geocode import (
    REQUEST_INTERVAL_S,
    apply_candidates,
    geocode_camera,
    read_candidates,
    write_candidates,
)
from safetrajai.paths import PROJECT_ROOT
from safetrajai.secrets import MissingApiKeyError, google_maps_key

CANDIDATES_CSV = PROJECT_ROOT / "models/homography/geocode_candidates.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Geocode camera intersection centres.")
    parser.add_argument("--cameras", nargs="*", help="只查詢這些鏡頭（預設全部）")
    parser.add_argument("--apply", action="store_true", help="依候選 CSV 寫入鏡頭設定")
    parser.add_argument("--accept", nargs="*", default=[], help="與 --apply 併用：另外套用這些 needs_review 的鏡頭")
    parser.add_argument("--no-google", action="store_true", help="不使用 Google Geocoding 備援")
    parser.add_argument("--camera-dir", type=Path, default=CAMERA_DIR)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_CSV)
    args = parser.parse_args()

    if args.apply:
        updated = apply_candidates(
            read_candidates(args.candidates), args.camera_dir, set(args.accept), date.today().isoformat(),
        )
        print(f"updated {len(updated)} camera(s): {', '.join(updated)}")
        return

    google_key = None
    if not args.no_google:
        try:
            google_key = google_maps_key("geocode")
            print(f"Google Geocoding 備援：使用 {google_key}")
        except MissingApiKeyError:
            print("Google Geocoding 備援：沒有金鑰，只使用 Overpass")

    configs = [c for c in list_camera_configs(args.camera_dir) if not args.cameras or c.camera in args.cameras]
    candidates = []
    for i, cfg in enumerate(configs):
        if i:
            time.sleep(REQUEST_INTERVAL_S)
        cand = geocode_camera(cfg, post=requests.post, get=requests.get, google_key=google_key, sleep=time.sleep)
        candidates.append(cand)
        print(f"[{cand.status:12s}] {cand.camera:8s} roads={cand.roads} n={cand.n_nodes} "
              f"latlon=({cand.lat}, {cand.lon}) spread={cand.spread_m} {cand.note}")

    write_candidates(args.candidates, candidates)
    print(f"\n候選表：{args.candidates}")
    print("summary:", dict(Counter(c.status for c in candidates)))


if __name__ == "__main__":
    main()
