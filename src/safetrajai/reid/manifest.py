"""資料集版本（manifest）的產生規則（見 CONTEXT.md「資料集版本」、「排除」，spec: reid-dataset）。

依序：套用排除清單 → 捨棄張數不足的身分 → 每個身分依時間均勻抽樣到上限 → 依鏡頭與日期切分。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

MANIFEST_FIELDS = ("crop_path", "identity", "class", "camera", "date", "split")
EXCLUSION_FIELDS = ("class", "identity", "image", "action")


def load_split_rules(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        rules = yaml.safe_load(f) or {}
    return {
        "val": [{"camera": v["camera"], "dates": [str(d) for d in v["dates"]]} for v in rules.get("val", [])],
        "min_crops_per_identity": int(rules["min_crops_per_identity"]),
        "max_crops_per_identity": int(rules["max_crops_per_identity"]),
    }


def _is_excluded(row: dict, removed_identities: set, removed_images: set) -> bool:
    key = (row["class"], row["identity"])
    image = PurePosixPath(row["crop_path"]).name
    return key in removed_identities or (key, image) in removed_images or (key, row["crop_path"]) in removed_images


def _uniform_sample(rows: list[dict], limit: int) -> list[dict]:
    if len(rows) <= limit:
        return rows
    picks = np.unique(np.round(np.linspace(0, len(rows) - 1, limit)).astype(int))
    return [rows[i] for i in picks]


def _split_for(camera: str, date: str, rules: dict) -> str:
    return "val" if any(camera == v["camera"] and date in v["dates"] for v in rules["val"]) else "train"


def build_manifest(
    index_rows: list[dict],
    exclusions: list[dict],
    rules: dict,
    eval_cameras: set[str],
) -> list[dict]:
    """由裁切索引、排除清單與切分規則產生 manifest 列，依 class、identity、time_s 排序。"""
    leaked = sorted({r["camera"] for r in index_rows if r["camera"] in eval_cameras})
    if leaked:
        raise ValueError(f"裁切索引含有評估鏡頭，不得產生資料集版本：{leaked}")

    removed = [e for e in exclusions if e.get("action", "remove") == "remove"]
    removed_identities = {(e["class"], e["identity"]) for e in removed if not e.get("image")}
    removed_images = {((e["class"], e["identity"]), e["image"]) for e in removed if e.get("image")}

    by_identity: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in index_rows:
        if not _is_excluded(row, removed_identities, removed_images):
            by_identity[(row["class"], row["identity"])].append(row)

    manifest = []
    for key in sorted(by_identity):
        rows = sorted(by_identity[key], key=lambda r: (float(r["time_s"]), r["crop_path"]))
        if len(rows) < rules["min_crops_per_identity"]:
            continue
        dates = {(r["camera"], r["date"]) for r in rows}
        if len(dates) != 1:
            raise ValueError(f"身分 {key} 跨越多個鏡頭或日期：{sorted(dates)}")
        camera, date = dates.pop()
        split = _split_for(camera, date, rules)
        manifest.extend(
            {"crop_path": r["crop_path"], "identity": r["identity"], "class": r["class"],
             "camera": camera, "date": date, "split": split}
            for r in _uniform_sample(rows, rules["max_crops_per_identity"])
        )
    return manifest


def summarize(manifest: list[dict]) -> dict:
    identities = {(r["class"], r["identity"], r["split"], r["camera"]) for r in manifest}
    crops = Counter((r["class"], r["split"], r["camera"]) for r in manifest)
    ids = Counter((c, s, cam) for c, _, s, cam in identities)
    return {
        "crops": len(manifest),
        "identities": len({(c, i) for c, i, _, _ in identities}),
        "by_class_split_camera": [
            {"class": c, "split": s, "camera": cam, "identities": ids[(c, s, cam)], "crops": crops[(c, s, cam)]}
            for c, s, cam in sorted(crops)
        ],
    }
