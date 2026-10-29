"""
Camera-keyed homography configuration: config/homography/<camera>.yaml.

The camera id is the text before the first space of a sequence name (same key as
config/roi/<camera>.yaml). Every top-level value is written on a single line so
tools can update individual fields without losing the file's comments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from safetrajai.evaluation.dataset_layout import camera_id_from_seq_name
from safetrajai.paths import CONFIG_DIR

CAMERA_DIR = CONFIG_DIR / "homography"

SCALE_CHECK_VALUES = ("passed", "failed", "unchecked")
LATLON_SOURCES = ("osm_overpass", "google_geocoding", "manual")
SATELLITE_PROVIDERS = ("google_static",)

_TAG_SUFFIX_RE = re.compile(r"-(\d+)-(\d+)-(\d+)-(\d+)$")

_KNOWN_FIELDS = (
    "camera", "name", "matrix_tag", "needs_recalibration", "satellite",
    "latlon", "latlon_source", "latlon_date", "osm_roads",
    "scale_check", "scale_check_date", "cctv_points", "map_points",
)

Point = Tuple[float, float]


@dataclass(frozen=True)
class SatelliteParams:
    provider: str
    zoom: int
    scale: int
    size: Tuple[int, int]

    @property
    def output_size(self) -> Tuple[int, int]:
        return (self.size[0] * self.scale, self.size[1] * self.scale)


@dataclass(frozen=True)
class CameraHomographyConfig:
    path: Path
    camera: str
    name: Optional[str] = None
    matrix_tag: Optional[str] = None
    needs_recalibration: Optional[str] = None
    satellite: Optional[SatelliteParams] = None
    latlon: Optional[Tuple[float, float]] = None
    latlon_source: Optional[str] = None
    latlon_date: Optional[str] = None
    osm_roads: Optional[Tuple[str, str]] = None
    scale_check: str = "unchecked"
    scale_check_date: Optional[str] = None
    cctv_points: Optional[Tuple[Point, ...]] = None
    map_points: Optional[Tuple[Point, ...]] = None


def camera_config_path(camera_dir: Path, camera: str) -> Path:
    return Path(camera_dir) / f"{camera}.yaml"


def parse_tag_satellite(tag: str) -> Optional[SatelliteParams]:
    """Satellite parameters encoded in a matrix tag `<name>-<scale>-<zoom>-<w>-<h>`."""
    match = _TAG_SUFFIX_RE.search(tag)
    if match is None:
        return None
    scale, zoom, width, height = (int(g) for g in match.groups())
    if width % scale or height % scale:
        return None
    return SatelliteParams("google_static", zoom, scale, (width // scale, height // scale))


def _fail(path: Path, field: str, message: str) -> None:
    raise ValueError(f"{path}: 欄位 {field} {message}")


def _optional_str(raw: Mapping[str, Any], path: Path, field: str) -> Optional[str]:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        _fail(path, field, "必須是非空字串")
    return value


def _parse_points(raw: Mapping[str, Any], path: Path, field: str) -> Optional[Tuple[Point, ...]]:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(c, (int, float)) for c in p)
        for p in value
    ):
        _fail(path, field, "必須是 [[x, y], ...] 格式")
    return tuple((float(x), float(y)) for x, y in value)


def _parse_satellite(raw: Mapping[str, Any], path: Path) -> Optional[SatelliteParams]:
    value = raw.get("satellite")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"provider", "zoom", "scale", "size"}:
        _fail(path, "satellite", "必須包含 provider、zoom、scale、size 四個欄位")
    if value["provider"] not in SATELLITE_PROVIDERS:
        _fail(path, "satellite", f"的 provider 只支援 {SATELLITE_PROVIDERS}")
    size = value["size"]
    ints_ok = all(isinstance(value[k], int) and not isinstance(value[k], bool) for k in ("zoom", "scale"))
    size_ok = isinstance(size, list) and len(size) == 2 and all(isinstance(s, int) and s > 0 for s in size)
    if not ints_ok or not size_ok or not 0 <= value["zoom"] <= 22 or value["scale"] not in (1, 2, 4):
        _fail(path, "satellite", "的 zoom（0～22）、scale（1/2/4）、size（兩個正整數）格式錯誤")
    return SatelliteParams(value["provider"], value["zoom"], value["scale"], (size[0], size[1]))


def _parse_latlon(raw: Mapping[str, Any], path: Path) -> Optional[Tuple[float, float]]:
    value = raw.get("latlon")
    if value is None:
        return None
    if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)):
        _fail(path, "latlon", "必須是 [緯度, 經度]")
    lat, lon = float(value[0]), float(value[1])
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        _fail(path, "latlon", f"超出範圍：{value}")
    return (lat, lon)


def load_camera_config(path: Path) -> CameraHomographyConfig:
    """Load and validate one camera config; raises ValueError naming the file and field."""
    path = Path(path)
    return _config_from_text(path.read_text(encoding="utf-8"), path)


def _config_from_text(text: str, path: Path) -> CameraHomographyConfig:
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        _fail(path, "(root)", "必須是 mapping")

    unknown = sorted(set(raw) - set(_KNOWN_FIELDS))
    if unknown:
        _fail(path, ", ".join(unknown), "不是已知欄位")

    camera = raw.get("camera")
    if not isinstance(camera, str) or camera != path.stem:
        _fail(path, "camera", f"必須等於檔名 {path.stem!r}（目前 {camera!r}）")

    scale_check = raw.get("scale_check", "unchecked")
    if scale_check not in SCALE_CHECK_VALUES:
        _fail(path, "scale_check", f"必須是 {SCALE_CHECK_VALUES} 之一（目前 {scale_check!r}）")

    latlon_source = _optional_str(raw, path, "latlon_source")
    if latlon_source is not None and latlon_source not in LATLON_SOURCES:
        _fail(path, "latlon_source", f"必須是 {LATLON_SOURCES} 之一")

    osm_roads = raw.get("osm_roads")
    if osm_roads is not None and not (
        isinstance(osm_roads, list) and len(osm_roads) == 2 and all(isinstance(r, str) and r for r in osm_roads)
    ):
        _fail(path, "osm_roads", "必須是兩個路名")

    cctv_points = _parse_points(raw, path, "cctv_points")
    map_points = _parse_points(raw, path, "map_points")
    if (cctv_points is None) != (map_points is None):
        _fail(path, "cctv_points/map_points", "必須同時提供")
    if cctv_points is not None and len(cctv_points) != len(map_points):
        _fail(path, "cctv_points/map_points", f"數量不符（{len(cctv_points)} 對 {len(map_points)}）")
    if cctv_points is not None and len(cctv_points) < 4:
        _fail(path, "cctv_points/map_points", f"至少需要 4 組對應點（目前 {len(cctv_points)} 組）")

    return CameraHomographyConfig(
        path=path,
        camera=camera,
        name=_optional_str(raw, path, "name"),
        matrix_tag=_optional_str(raw, path, "matrix_tag"),
        needs_recalibration=_optional_str(raw, path, "needs_recalibration"),
        satellite=_parse_satellite(raw, path),
        latlon=_parse_latlon(raw, path),
        latlon_source=latlon_source,
        latlon_date=_optional_str(raw, path, "latlon_date"),
        osm_roads=tuple(osm_roads) if osm_roads is not None else None,
        scale_check=scale_check,
        scale_check_date=_optional_str(raw, path, "scale_check_date"),
        cctv_points=cctv_points,
        map_points=map_points,
    )


def load_camera_config_for_sequence(camera_dir: Path, seq_name: str) -> Optional[CameraHomographyConfig]:
    """Config of the camera a sequence belongs to, or None when the camera has no file."""
    path = camera_config_path(camera_dir, camera_id_from_seq_name(seq_name))
    return load_camera_config(path) if path.exists() else None


def list_camera_configs(camera_dir: Path) -> Tuple[CameraHomographyConfig, ...]:
    camera_dir = Path(camera_dir)
    if not camera_dir.exists():
        return ()
    return tuple(load_camera_config(p) for p in sorted(camera_dir.glob("*.yaml")))


def format_yaml_value(value: Any) -> str:
    """Single-line YAML (flow style) representation of a value."""
    text = yaml.safe_dump(value, allow_unicode=True, default_flow_style=True, width=10**6).strip()
    if text.endswith("..."):
        text = text[:-3].strip()
    return text


def update_camera_config_fields(path: Path, updates: Mapping[str, Any]) -> None:
    """Rewrite single-line top-level fields in place, keeping comments and other lines.

    Fields that are missing are appended. A field whose value spans several lines is
    refused, because a line-based rewrite could corrupt it.
    """
    path = Path(path)
    unknown = sorted(set(updates) - set(_KNOWN_FIELDS))
    if unknown:
        raise ValueError(f"{path}: 不能更新未知欄位 {unknown}")

    lines = path.read_text(encoding="utf-8").splitlines()
    pending: Dict[str, Any] = dict(updates)
    out = []
    for i, line in enumerate(lines):
        key = line.split(":", 1)[0] if re.match(r"^[A-Za-z_]+:", line) else None
        if key in pending:
            rest = line.split(":", 1)[1].split(" #", 1)[0].strip()
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if rest == "" and next_line.startswith((" ", "-")):
                raise ValueError(f"{path}: 欄位 {key} 是多行格式，無法自動更新")
            out.append(f"{key}: {format_yaml_value(pending.pop(key))}")
        else:
            out.append(line)
    out.extend(f"{key}: {format_yaml_value(value)}" for key, value in pending.items())
    new_text = "\n".join(out) + "\n"
    _config_from_text(new_text, path)
    path.write_text(new_text, encoding="utf-8")
