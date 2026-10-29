"""Counting lines per camera (config/counting_lines/<camera>.yaml), in CCTV pixel coordinates.

    camera: C000025
    image_size: [640, 480]
    lines:
      - name: north_gate
        points: [[120, 300], [480, 310]]   # p1 -> p2
        forward_label: southbound          # optional, default "forward"
        backward_label: northbound         # optional, default "backward"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import yaml

from safetrajai.evaluation.dataset_layout import camera_id_from_seq_name
from safetrajai.paths import CONFIG_DIR

COUNTING_LINES_DIR = CONFIG_DIR / "counting_lines"
_FILE_KEYS = {"camera", "image_size", "lines"}
_LINE_KEYS = {"name", "points", "forward_label", "backward_label"}

Point = Tuple[float, float]


@dataclass(frozen=True)
class CountingLine:
    name: str
    p1: Point
    p2: Point
    forward_label: str = "forward"
    backward_label: str = "backward"


@dataclass(frozen=True)
class CountingLineSet:
    camera: str
    image_size: Tuple[int, int]
    lines: Tuple[CountingLine, ...]


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _point(value: Any, where: str) -> Point:
    if not (isinstance(value, list) and len(value) == 2 and all(_number(c) for c in value)):
        raise ValueError(f"{where} 必須是 [x, y]（兩個數字），目前是 {value!r}")
    return (float(value[0]), float(value[1]))


def _line(raw: Any, path: Path, index: int) -> CountingLine:
    where = f"{path}: lines[{index}]"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} 必須是對應表")
    unknown = sorted(set(raw) - _LINE_KEYS)
    if unknown:
        raise ValueError(f"{where} 有未知欄位 {unknown}")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where}.name 必須是非空白文字")
    points = raw.get("points")
    if not (isinstance(points, list) and len(points) == 2):
        raise ValueError(f"{where}.points 必須剛好兩個點")
    p1, p2 = _point(points[0], f"{where}.points[0]"), _point(points[1], f"{where}.points[1]")
    if p1 == p2:
        raise ValueError(f"{where}.points 兩個點相同，線段長度為 0")
    labels = {}
    for key, default in (("forward_label", "forward"), ("backward_label", "backward")):
        value = raw.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"{where}.{key} 必須是文字，目前是 {value!r}")
        labels[key] = value
    return CountingLine(name, p1, p2, labels["forward_label"], labels["backward_label"])


def load_counting_lines(path: Path) -> CountingLineSet:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 必須是對應表")
    unknown = sorted(set(raw) - _FILE_KEYS)
    if unknown:
        raise ValueError(f"{path} 有未知欄位 {unknown}")
    if raw.get("camera") != path.stem:
        raise ValueError(f"{path}: camera={raw.get('camera')!r} 與檔名 {path.stem!r} 不符")
    size = raw.get("image_size")
    if not (isinstance(size, list) and len(size) == 2
            and all(isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in size)):
        raise ValueError(f"{path}: image_size 必須是兩個正整數 [寬, 高]，目前是 {size!r}")
    raw_lines = raw.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError(f"{path}: lines 必須是至少一條線的清單")
    lines = tuple(_line(item, path, index) for index, item in enumerate(raw_lines))
    names = [line.name for line in lines]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"{path}: 計數線名稱重複 {duplicated}")
    return CountingLineSet(path.stem, (size[0], size[1]), lines)


def resolve_counting_lines(
    seq_name: str, image_size: Tuple[int, int], lines_dir: Path = COUNTING_LINES_DIR,
) -> Optional[Tuple[CountingLine, ...]]:
    """Lines of the sequence's camera, or None without a file; a size mismatch is an error (no rescale)."""
    path = Path(lines_dir) / f"{camera_id_from_seq_name(seq_name)}.yaml"
    if not path.exists():
        return None
    loaded = load_counting_lines(path)
    actual = (int(image_size[0]), int(image_size[1]))
    if loaded.image_size != actual:
        raise ValueError(
            f"{path} 的 image_size={list(loaded.image_size)} 與序列 {seq_name!r} 的實際影像尺寸 {list(actual)} 不符")
    return loaded.lines
