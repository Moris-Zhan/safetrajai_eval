"""Re-ID 裁切索引（見 CONTEXT.md「裁切」）。

每段序列的索引寫在 <crop_root>/_index/<seq>.csv，一列對應一張裁切。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

INDEX_FIELDS = (
    "crop_path", "identity", "class", "camera", "sequence", "date",
    "frame", "time_s", "x1", "y1", "x2", "y2", "score",
)
INDEX_DIR_NAME = "_index"

_SEQ_TIMESTAMP = re.compile(r"_(\d{8})\d{6}(?=$|_)")


def date_from_seq_name(seq_name: str) -> str:
    """序列名稱中的 _YYYYMMDDhhmmss 時間戳 → YYYYMMDD；取最後一個符合的時間戳。"""
    matches = _SEQ_TIMESTAMP.findall(seq_name)
    if not matches:
        raise ValueError(f"無法從序列名稱取得日期（缺少 _YYYYMMDDhhmmss）：{seq_name!r}")
    return matches[-1]


def index_path(crop_root: Path, seq_name: str) -> Path:
    return Path(crop_root) / INDEX_DIR_NAME / f"{seq_name}.csv"


def write_index(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_index(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_all_indexes(crop_root: Path) -> list[dict]:
    return [row for p in sorted((Path(crop_root) / INDEX_DIR_NAME).glob("*.csv")) for row in read_index(p)]
