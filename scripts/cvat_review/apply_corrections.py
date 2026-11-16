#!/usr/bin/env python3
"""
把人工覆核／自動偵測出的 track 修正（刪除整條 track，或改標成另一個類別）套用到
CVAT task 的標註，取代 scripts/cvat_review 產生的覆核候選名單人工一一在 CVAT UI 操作。

套用前一定先把目前的標註完整備份一份（含時間戳），套用後重新抓一次標註驗證結果。

跟 fetch_task_annotations.py 一樣只需要 cvat_sdk，一定要用
ignore/cvat_deploy/cvat_client_venv 的 python 執行：

    ignore/cvat_deploy/cvat_client_venv/bin/python scripts/cvat_review/apply_corrections.py \
        --task-id 3 --corrections runs/cvat_review/corrections.csv

corrections.csv 欄位：track_id,action,new_label,reason
  action=delete    忽略 new_label，整條 track 刪除（沒有對應類別的誤判，例如紅綠燈）
  action=relabel   new_label 填目標類別名稱（例如 car），track 其他內容不變

CVAT 連線資訊從環境變數或 repo 根目錄 .env 讀取（CVAT_URL／CVAT_USERNAME／CVAT_PASSWORD），
不印出密碼值。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from cvat_sdk import make_client
from cvat_sdk.api_client import models
from cvat_sdk.core.proxies.annotations import AnnotationUpdateAction

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


def _dump_tracks(annotations, out_path: Path) -> None:
    tracks = [
        {
            "track_id": track.id,
            "label_id": track.label_id,
            "shapes": [
                {"frame": shape.frame, "points": shape.points, "outside": shape.outside}
                for shape in track.shapes
            ],
        }
        for track in annotations.tracks
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"tracks": tracks}, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=PROJECT_ROOT / "runs" / "cvat_review" / "backups")
    parser.add_argument("--log", type=Path, default=PROJECT_ROOT / "runs" / "cvat_review" / "corrections_applied.csv")
    parser.add_argument("--dry-run", action="store_true", help="只印出會做什麼，不呼叫 CVAT API")
    args = parser.parse_args()

    with args.corrections.open(newline="", encoding="utf-8") as f:
        corrections = list(csv.DictReader(f))
    if not corrections:
        raise SystemExit(f"{args.corrections} 沒有任何修正項目")

    url = _get_setting("CVAT_URL")
    username = _get_setting("CVAT_USERNAME")
    password = _get_setting("CVAT_PASSWORD")
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)

    with make_client(host=f"{parts.scheme}://{parts.hostname}", port=port, credentials=(username, password)) as client:
        task = client.tasks.retrieve(args.task_id)
        labels_by_name = {label.name: label.id for label in task.get_labels()}

        annotations = task.get_annotations()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = args.backup_dir / f"task{args.task_id}_pre_corrections_{timestamp}.json"
        _dump_tracks(annotations, backup_path)
        print(f"backed up current annotations ({len(annotations.tracks)} tracks) -> {backup_path}")

        tracks_by_id = {t.id: t for t in annotations.tracks}

        delete_ids = []
        relabel_requests = []
        log_rows = []
        for row in corrections:
            tid = int(row["track_id"])
            track = tracks_by_id.get(tid)
            if track is None:
                print(f"skip: track {tid} not found in current annotations")
                continue
            old_label_name = next((n for n, i in labels_by_name.items() if i == track.label_id), "?")
            action = row["action"].strip()
            if action == "delete":
                delete_ids.append(tid)
                log_rows.append({"track_id": tid, "action": "delete", "old_label": old_label_name,
                                  "new_label": "", "reason": row.get("reason", "")})
            elif action == "relabel":
                new_label = row["new_label"].strip()
                if new_label not in labels_by_name:
                    raise SystemExit(f"未知標籤：{new_label}")
                payload = track.to_dict()
                payload["label_id"] = labels_by_name[new_label]
                relabel_requests.append(models.LabeledTrackRequest(**payload))
                log_rows.append({"track_id": tid, "action": "relabel", "old_label": old_label_name,
                                  "new_label": new_label, "reason": row.get("reason", "")})
            else:
                raise SystemExit(f"未知 action：{action}")

        print(f"plan: delete {len(delete_ids)} tracks, relabel {len(relabel_requests)} tracks")
        if args.dry_run:
            print("dry-run，沒有呼叫 CVAT API")
            return

        if relabel_requests:
            task.update_annotations(
                models.PatchedLabeledDataRequest(tracks=relabel_requests),
                action=AnnotationUpdateAction.UPDATE,
            )
            print(f"relabeled {len(relabel_requests)} tracks")
        if delete_ids:
            task.remove_annotations(ids=delete_ids)
            print(f"deleted {len(delete_ids)} tracks")

        after = task.get_annotations()
        after_by_id = {t.id: t for t in after.tracks}
        for row in log_rows:
            tid = row["track_id"]
            if row["action"] == "delete":
                row["verified"] = tid not in after_by_id
            else:
                verified_label = next((n for n, i in labels_by_name.items() if i == after_by_id[tid].label_id), "?") \
                    if tid in after_by_id else "?"
                row["verified"] = verified_label == row["new_label"]

        args.log.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.log.exists()
        with args.log.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "action", "old_label", "new_label", "reason", "verified"])
            if write_header:
                writer.writeheader()
            writer.writerows(log_rows)
        failed = [r for r in log_rows if not r["verified"]]
        print(f"verified {len(log_rows) - len(failed)}/{len(log_rows)} corrections; log -> {args.log}")
        if failed:
            print("FAILED to verify:", failed)


if __name__ == "__main__":
    main()
