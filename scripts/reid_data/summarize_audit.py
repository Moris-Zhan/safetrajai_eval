#!/usr/bin/env python3
"""
計算抽查的不合格率（見 CONTEXT.md「抽查」「不合格率」、docs/adr/0007）。

讀取 generate_identity_review.py --audit-only 輸出的 audit_sample.csv 與使用者匯出的確認清單，
只看鏡頭抽查（camera_audit）的身分：需要整段排除（確認清單中 image 為空）的比例即不合格率，
只排除單張的身分不計入；bus 抽查與 person 審查不計算不合格率。
身分數佔全部 ≥ --major-share 的主要鏡頭，另外分別計算 car 與 scooter 的不合格率。
不合格率超過 --threshold（預設 0.1）的範圍會標記出來：主要鏡頭要調整規則並重新抽查，
其他鏡頭則以 build_reid_split.py --exclude-cameras 整支排除。只讀檔、只列印，不寫任何檔案。

用法：
    python scripts/reid_data/summarize_audit.py --confirm-csv review_camera.csv review_bus.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.reid.dataset_index import read_all_indexes
from safetrajai.reid.manifest import load_split_rules

PER_CLASS_SCOPES = ("car", "scooter")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def failure_rates(audit: list[dict], confirmed: list[dict], major_cameras: set[str], threshold: float = 0.1) -> list[dict]:
    failed = {(r["class"], r["identity"]) for r in confirmed if not r.get("image") and r.get("action", "remove") == "remove"}
    sampled = [r for r in audit if r["purpose"] == "camera_audit"]
    scopes: list[tuple[str, str, list[dict]]] = []
    for camera in sorted({r["camera"] for r in sampled}):
        members = [r for r in sampled if r["camera"] == camera]
        scopes.append((camera, "all", members))
        if camera in major_cameras:
            scopes += [(camera, cls, [r for r in members if r["class"] == cls]) for cls in PER_CLASS_SCOPES]
    results = []
    for scope, cls, members in scopes:
        if not members:
            continue
        failures = sum((r["class"], r["identity"]) in failed for r in members)
        rate = failures / len(members)
        results.append({"scope": scope, "class": cls, "sampled": len(members), "failures": failures,
                        "rate": round(rate, 4), "over_threshold": rate > threshold})
    return results


def major_cameras_from_index(crop_root: Path, min_crops: int, major_share: float) -> set[str]:
    identities = Counter((r["class"], r["identity"], r["camera"]) for r in read_all_indexes(crop_root))
    eligible = [cam for (_, _, cam), n in identities.items() if n >= min_crops]
    per_camera = Counter(eligible)
    return {cam for cam, n in per_camera.items() if n / len(eligible) >= major_share}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit-sample", default=str(PROJECT_ROOT / "dataset" / "reports" / "reid" / "review" / "audit_sample.csv"))
    parser.add_argument("--confirm-csv", nargs="+", required=True, help="審查頁匯出的確認清單（可多份）")
    parser.add_argument("--crop-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "crops"))
    parser.add_argument("--rules", default=str(CONFIG_DIR / "reid_split.yaml"))
    parser.add_argument("--major-share", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    min_crops = load_split_rules(Path(args.rules))["min_crops_per_identity"]
    major = major_cameras_from_index(Path(args.crop_root), min_crops, args.major_share)
    confirmed = [row for path in args.confirm_csv for row in read_rows(Path(path))]
    for r in failure_rates(read_rows(Path(args.audit_sample)), confirmed, major, args.threshold):
        flag = "  ← 超過門檻" if r["over_threshold"] else ""
        kind = "主要鏡頭" if r["scope"] in major else "其他鏡頭"
        print(f"{r['scope']:8s} {kind} {r['class']:8s} {r['failures']:3d}/{r['sampled']:3d} = {r['rate']:.1%}{flag}")


if __name__ == "__main__":
    main()
