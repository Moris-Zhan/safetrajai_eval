"""自動排除規則：由污染標記產生排除紀錄（見 CONTEXT.md「自動排除」）。

- 靜止身分、運動跳動：整個身分排除（誤偵測身分、污染身分）。
- 外觀離群：只排除該張裁切（多為殘缺裁切）。
- 類別一致性：只在四輪與「人／二輪」之間不一致時整個身分排除；人與二輪之間
  （例如機車騎士被判為行人）不自動排除。
只處理裁切數 ≥ min_crops 的身分；不足的身分不會進資料集版本。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

from safetrajai.reid.contamination import COARSE_GROUPS

AUTO_EXCLUSION_FIELDS = ("class", "identity", "image", "action", "rule")
RULE_ORDER = ("static", "motion_jump", "class_group", "appearance_outlier")


def _is_true(value) -> bool:
    return str(value) == "True"


def _exclusion_group(cls: str) -> str:
    return "four_wheel" if COARSE_GROUPS[cls] == "four_wheel" else "person_or_two_wheel"


def build_auto_exclusions(
    index_rows: list[dict],
    motion_records: list[dict],
    crop_records: list[dict],
    min_crops: int,
) -> list[dict]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in index_rows:
        counts[(row["class"], row["identity"])] += 1
    eligible = {key for key, n in counts.items() if n >= min_crops}

    whole_rules: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in motion_records:
        key = (record["class"], record["identity"])
        if key not in eligible:
            continue
        if _is_true(record.get("static")):
            whole_rules[key].add("static")
        if _is_true(record.get("motion_jump")):
            whole_rules[key].add("motion_jump")

    mismatched: dict[tuple[str, str], list[bool]] = defaultdict(list)
    outlier_images: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in crop_records:
        key = (record["class"], record["identity"])
        if key not in eligible:
            continue
        mismatched[key].append(_exclusion_group(record["predicted_class"]) != _exclusion_group(record["class"]))
        if _is_true(record.get("is_outlier")):
            outlier_images[key].add(PurePosixPath(record["crop_path"]).name)
    for key, flags in mismatched.items():
        if sum(flags) > len(flags) / 2:
            whole_rules[key].add("class_group")

    rows = [
        {"class": cls, "identity": identity, "image": "", "action": "remove",
         "rule": "+".join(r for r in RULE_ORDER if r in rules)}
        for (cls, identity), rules in whole_rules.items()
    ]
    rows += [
        {"class": cls, "identity": identity, "image": image, "action": "remove", "rule": "appearance_outlier"}
        for (cls, identity), images in outlier_images.items() if (cls, identity) not in whole_rules
        for image in images
    ]
    return sorted(rows, key=lambda r: (r["class"], r["identity"], r["image"]))
