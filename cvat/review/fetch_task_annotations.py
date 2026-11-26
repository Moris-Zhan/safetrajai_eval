#!/usr/bin/env python3
"""
把 CVAT task 的全部追蹤標註（track_id／label／每個關鍵幀的框／outside 標記）整份抓下來寫成 JSON，
供 cvat/review/serve_review.py 產生覆核用網頁。

分成兩支腳本是因為裝 cvat_sdk 的 venv（cvat/deploy/cvat_client_venv）沒有 cv2，
裝 cv2 的 conda env track_eval 沒有 cvat_sdk；這支只需要 cvat_sdk，不需要 safetrajai 套件。

用法（一定要用 cvat_client_venv 的 python）：
    cvat/deploy/cvat_client_venv/bin/python cvat/review/fetch_task_annotations.py \
        --task-id 3 --out runs/cvat_review/pilot_v1_annotations.json

CVAT 連線資訊從環境變數或 repo 根目錄 .env 讀取（CVAT_URL／CVAT_USERNAME／CVAT_PASSWORD），
不印出密碼值。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from cvat_sdk import make_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


def _read_env_file(path: Path) -> dict:
    entries = {}
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            entries[name.strip()] = value
    return entries


def _get_setting(name: str) -> str:
    value = os.environ.get(name) or _read_env_file(ENV_PATH).get(name)
    if not value:
        raise SystemExit(f"找不到設定：請在環境變數或 {ENV_PATH} 設定 {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    url = _get_setting("CVAT_URL")
    username = _get_setting("CVAT_USERNAME")
    password = _get_setting("CVAT_PASSWORD")
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)

    with make_client(host=f"{parts.scheme}://{parts.hostname}", port=port, credentials=(username, password)) as client:
        task = client.tasks.retrieve(args.task_id)
        labels = {label.id: label.name for label in task.get_labels()}
        annotations = task.get_annotations()

        tracks = [
            {
                "track_id": track.id,
                "label": labels.get(track.label_id),
                "shapes": [
                    {"frame": shape.frame, "points": shape.points, "outside": shape.outside}
                    for shape in track.shapes
                ],
            }
            for track in annotations.tracks
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"task_id": args.task_id, "tracks": tracks}, ensure_ascii=False),
        encoding="utf-8",
    )
    counts: dict = {}
    for t in tracks:
        counts[t["label"]] = counts.get(t["label"], 0) + 1
    print(f"wrote {args.out} — {len(tracks)} tracks, by class: {counts}")


if __name__ == "__main__":
    main()
