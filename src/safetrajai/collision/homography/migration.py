"""
One-off migration of tag-keyed homography data to camera-keyed files (design §2).

- models/homography/matrices/H_<tag>.npy        -> config/homography/<camera>.yaml
- config/homography_checkpoints/<tag>.yaml      -> config/homography_checkpoints/<camera>.yaml (verbatim copy)
- models/homography/checkpoints/check_pts_{cctv,sat}_<tag>.npy
                                                -> config/homography_checkpoints/<camera>.yaml

Sources are never modified. A target that already exists with different content is a
conflict; any conflict aborts the whole migration before anything is written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from safetrajai.evaluation.dataset_layout import camera_id_from_seq_name

from .camera_config import format_yaml_value, parse_tag_satellite

_TAG_SUFFIX_RE = re.compile(r"-\d+-\d+-\d+-\d+$")


class MigrationConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class MigrationStep:
    target: Path
    content: str
    source: str
    action: str  # "create" | "unchanged" | "conflict"


def _camera_config_text(tag: str) -> str:
    name = _TAG_SUFFIX_RE.sub("", tag)
    camera = camera_id_from_seq_name(name)
    satellite = parse_tag_satellite(tag)
    lines = [
        f"# config/homography/{camera}.yaml — 鏡頭 {camera} 的 Homography 設定"
        "（見 safetrajai/collision/homography/camera_config.py）",
        "# 由 scripts/migrate_homography_config.py 依既有矩陣 models/homography/matrices/H_<matrix_tag>.npy 產生。",
        "# - satellite：衛星底圖的 Google Static Maps 請求參數（輸出尺寸＝size × scale），由 tag 尾碼推得",
        "# - latlon／latlon_source／latlon_date：路口中心經緯度，由 scripts/geocode_cameras.py 寫入",
        "# - scale_check：passed／failed／unchecked，由 scripts/check_homography_scale.py 寫入；只有 passed 才換算公尺",
        "# - osm_roads：選填，覆寫經緯度查詢用的兩條路名",
        "# - cctv_points／map_points：選填人工對應點，只在沒有 matrix_tag 時使用",
        "# 每個欄位都寫在同一行，工具會逐欄更新並保留這段註解。",
        f"camera: {format_yaml_value(camera)}",
        f"name: {format_yaml_value(name)}",
        f"matrix_tag: {format_yaml_value(tag)}",
    ]
    if satellite is not None:
        sat = {"provider": satellite.provider, "zoom": satellite.zoom,
               "scale": satellite.scale, "size": list(satellite.size)}
        lines.append(f"satellite: {format_yaml_value(sat)}")
    lines.append("scale_check: unchecked")
    return "\n".join(lines) + "\n"


def _points_block(field: str, points: np.ndarray) -> List[str]:
    return [f"{field}:"] + [f"  - [{float(x)!r}, {float(y)!r}]" for x, y in points]


def _npy_checkpoints_text(tag: str, cctv: np.ndarray, sat: np.ndarray) -> str:
    camera = camera_id_from_seq_name(_TAG_SUFFIX_RE.sub("", tag))
    header = [
        f"# config/homography_checkpoints/{camera}.yaml — check points for {_TAG_SUFFIX_RE.sub('', tag)}",
        f"# 由 models/homography/checkpoints/check_pts_{{cctv,sat}}_{tag}.npy 轉入"
        "（scripts/migrate_homography_config.py），數值逐位元保留。",
    ]
    lines = header + _points_block("cctv_points", cctv) + _points_block("sat_points", sat)
    return "\n".join(lines) + "\n"


def _step(target: Path, content: str, source: str) -> MigrationStep:
    if not target.exists():
        return MigrationStep(target, content, source, "create")
    same = target.read_text(encoding="utf-8") == content
    return MigrationStep(target, content, source, "unchanged" if same else "conflict")


def _tags(matrices_dir: Path) -> List[str]:
    return sorted(p.stem[len("H_"):] for p in Path(matrices_dir).glob("H_*.npy"))


def plan_migration(
    matrices_dir: Path, npy_checkpoints_dir: Path, checkpoints_dir: Path, camera_dir: Path,
) -> Tuple[MigrationStep, ...]:
    """List every file the migration would write, with its action; reads sources only."""
    steps: List[MigrationStep] = []
    for tag in _tags(matrices_dir):
        camera = camera_id_from_seq_name(_TAG_SUFFIX_RE.sub("", tag))
        steps.append(_step(Path(camera_dir) / f"{camera}.yaml", _camera_config_text(tag), f"H_{tag}.npy"))

    checkpoints_dir = Path(checkpoints_dir)
    for src in sorted(checkpoints_dir.glob("*.yaml")):
        if not _TAG_SUFFIX_RE.search(src.stem):
            continue
        camera = camera_id_from_seq_name(_TAG_SUFFIX_RE.sub("", src.stem))
        steps.append(_step(checkpoints_dir / f"{camera}.yaml", src.read_text(encoding="utf-8"), src.name))

    npy_dir = Path(npy_checkpoints_dir)
    for cctv_path in sorted(npy_dir.glob("check_pts_cctv_*.npy")):
        tag = cctv_path.stem[len("check_pts_cctv_"):]
        sat_path = npy_dir / f"check_pts_sat_{tag}.npy"
        if not sat_path.exists():
            raise FileNotFoundError(f"{cctv_path} 沒有對應的 {sat_path.name}")
        cctv = np.load(cctv_path).reshape(-1, 2)
        sat = np.load(sat_path).reshape(-1, 2)
        if len(cctv) != len(sat):
            raise ValueError(f"{cctv_path.name} 與 {sat_path.name} 的點數不同")
        camera = camera_id_from_seq_name(_TAG_SUFFIX_RE.sub("", tag))
        steps.append(_step(checkpoints_dir / f"{camera}.yaml", _npy_checkpoints_text(tag, cctv, sat), cctv_path.name))

    targets = [s.target for s in steps]
    duplicated = sorted({str(t) for t in targets if targets.count(t) > 1})
    if duplicated:
        raise MigrationConflict(f"多個來源對應到同一個目標：{duplicated}")
    return tuple(steps)


def apply_migration(steps: Iterable[MigrationStep]) -> Tuple[Path, ...]:
    """Write all `create` steps; abort without writing if any step is a conflict."""
    steps = tuple(steps)
    conflicts = [str(s.target) for s in steps if s.action == "conflict"]
    if conflicts:
        raise MigrationConflict(f"目標已存在且內容不同，未寫入任何檔案：{conflicts}")
    created = []
    for step in steps:
        if step.action == "create":
            step.target.parent.mkdir(parents=True, exist_ok=True)
            step.target.write_text(step.content, encoding="utf-8")
            created.append(step.target)
    return tuple(created)
