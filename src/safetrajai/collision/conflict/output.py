"""
Output files of the conflict analysis (spec「輸出檔與座標」, design §10).

Points are written in three coordinate systems: metres (`_m`), top-down plane pixels
(`_map_px`, metres / m_per_pixel) and CCTV pixels (`_cctv_px`, through the inverse
homography). CSV columns have a fixed order and floats are written with `repr`, so two
runs on the same input are byte-identical. JSON writes NaN and ±inf as null.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .tracks import project_points


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def json_value(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def point_columns(prefix: str, xy_m: np.ndarray, m_per_pixel: float, H_inv: np.ndarray) -> Dict[str, np.ndarray]:
    """{prefix_x_m, prefix_y_m, prefix_x_map_px, ...} for (N, 2) metre points; NaN rows stay NaN."""
    xy_m = np.asarray(xy_m, dtype=np.float64).reshape(-1, 2)
    map_px = xy_m / m_per_pixel
    cctv = np.full_like(map_px, np.nan)
    finite = np.isfinite(map_px).all(axis=1)
    if finite.any():
        cctv[finite] = project_points(map_px[finite], H_inv)
    columns = {}
    for suffix, values in (("m", xy_m), ("map_px", map_px), ("cctv_px", cctv)):
        columns[f"{prefix}_x_{suffix}"] = values[:, 0]
        columns[f"{prefix}_y_{suffix}"] = values[:, 1]
    return columns


def write_csv(columns: Mapping[str, Sequence[Any]], path: Path) -> None:
    """Columns in insertion order; every column has one value per row."""
    names = list(columns)
    n = len(columns[names[0]]) if names else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(names)
        writer.writerows([fmt(columns[name][r]) for name in names] for r in range(n))


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_value(data), ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def records(columns: Mapping[str, Sequence[Any]]) -> list:
    names = list(columns)
    n = len(columns[names[0]]) if names else 0
    return [{name: columns[name][r] for name in names} for r in range(n)]
