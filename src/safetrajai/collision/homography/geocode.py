"""
Intersection-centre geocoding for camera configs (spec `camera-homography`, design §6).

1. Split the camera name into two road names (or use `osm_roads` from the config).
2. Ask OpenStreetMap Overpass for nodes shared by both roads inside the Kaohsiung bbox;
   the mean of those nodes is the intersection centre.
3. When Overpass finds nothing and a Google key is available, fall back to the Google
   Geocoding API v4; such results always need review.

Results are written as a candidate CSV; only `--apply` writes coordinates into the camera
configs, and only for `ok` rows or cameras the user explicitly accepts.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Set, Tuple
from urllib.parse import quote

from safetrajai.secrets import ApiKey

from .camera_config import CameraHomographyConfig, camera_config_path, update_camera_config_fields

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
GEOCODE_V4_URL = "https://geocode.googleapis.com/v4/geocode/address/"
USER_AGENT = "safetrajai-eval/0.1 (camera intersection geocoding; offline research)"
KAOHSIUNG_BBOX = (22.45, 120.15, 22.95, 120.45)
SPREAD_LIMIT_M = 50.0
MAX_RETRIES = 3
BASE_DELAY_S = 2.0
REQUEST_INTERVAL_S = 2.0

_CORNER_SUFFIX_RE = re.compile(r"(東北角|西北角|東南角|西南角|口)$")
_SECTIONED_ROAD_RE = re.compile(r".+[一二三四五六七八九十]路$")
_ROAD_END_RE = re.compile(r".+(路|街|道)$")


class GeocodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    camera: str
    name: str
    roads: Optional[Tuple[str, str]]
    source: str
    n_nodes: int
    lat: Optional[float]
    lon: Optional[float]
    spread_m: Optional[float]
    status: str  # "ok" | "needs_review" | "error"
    note: str


# =============================================================================
# 1. Road names
# =============================================================================

def _strip_corner_suffix(token: str) -> str:
    previous = None
    while previous != token:
        previous, token = token, _CORNER_SUFFIX_RE.sub("", token)
    return token


def _split_compound(token: str) -> Tuple[str, ...]:
    head, sep, tail = token.partition("路")
    if sep and tail.endswith("路") and len(tail) > 1:
        return (head + sep, tail)
    return (token,)


def split_roads(name: str) -> Optional[Tuple[str, str]]:
    """Two road names from a camera name, or None when the name is not `<road> <road>`."""
    parts = name.strip().split(maxsplit=1)
    rest = parts[1] if len(parts) == 2 else ""
    rest = re.sub(r"[\(（][^)）]*[\)）]", " ", rest)
    rest = re.sub(r"[\[\]、]", " ", rest)
    tokens = [
        _strip_corner_suffix(t) for t in rest.split()
        if not t.startswith("通道") and not t.endswith("區") and t not in ("高雄", "高雄市")
    ]
    tokens = [piece for t in tokens if t for piece in _split_compound(t)]
    if len(tokens) != 2 or not all(_ROAD_END_RE.fullmatch(t) for t in tokens):
        return None
    return (tokens[0], tokens[1])


def road_regex(road: str) -> str:
    """Overpass name regex; an unsectioned `X路` also matches `X一路`, `X二路`, ..."""
    if _SECTIONED_ROAD_RE.fullmatch(road) or not road.endswith("路"):
        return "^" + re.escape(road)
    return "^" + re.escape(road[:-1]) + "[一二三四五六七八九十]?路"


def overpass_query(roads: Tuple[str, str], bbox: Tuple[float, float, float, float] = KAOHSIUNG_BBOX) -> str:
    box = "({},{},{},{})".format(*bbox)
    a, b = (road_regex(r) for r in roads)
    return (
        f'[out:json][timeout:25];way{box}["highway"]["name"~"{a}"]->.a;'
        f'way{box}["highway"]["name"~"{b}"]->.b;node(w.a)->.na;node(w.b)->.nb;node.na.nb;out;'
    )


# =============================================================================
# 2. HTTP
# =============================================================================

def request_json(send: Callable[[], Any], sleep: Callable[[float], None], label: str) -> Any:
    """Call `send()` and parse JSON, retrying 429/5xx/non-JSON replies with exponential back-off."""
    last = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = send()
        except Exception as exc:  # network errors may embed request details; keep only the type
            last = type(exc).__name__
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    last = "non-JSON response"
            elif response.status_code == 429 or response.status_code >= 500:
                last = f"HTTP {response.status_code}"
            else:
                raise GeocodeError(f"{label}: HTTP {response.status_code}")
        if attempt < MAX_RETRIES:
            sleep(BASE_DELAY_S * 2 ** attempt)
    raise GeocodeError(f"{label}: 重試 {MAX_RETRIES} 次後仍失敗（{last}）")


def summarize_nodes(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
    """(mean lat, mean lon, max distance in metres from the mean)."""
    lat = sum(p[0] for p in points) / len(points)
    lon = sum(p[1] for p in points) / len(points)
    k = 111_320.0
    spread = max(math.hypot((p[0] - lat) * k, (p[1] - lon) * k * math.cos(math.radians(lat))) for p in points)
    return lat, lon, spread


def query_overpass(roads: Tuple[str, str], post: Callable[..., Any], sleep: Callable[[float], None]) -> list:
    payload = request_json(
        lambda: post(OVERPASS_URL, data={"data": overpass_query(roads)},
                     headers={"User-Agent": USER_AGENT}, timeout=60),
        sleep, "Overpass",
    )
    return [(e["lat"], e["lon"]) for e in payload.get("elements", []) if e.get("type") == "node"]


def _parse_google_location(payload: Any) -> Optional[Tuple[float, float]]:
    """(lat, lon) from a v4 (`location.latitude`) or legacy (`geometry.location.lat`) reply;
    None for an empty or unexpected payload, so one odd reply cannot abort a batch."""
    try:
        results = payload.get("results") or []
        if not results:
            return None
        first = results[0]
        if "location" in first:
            return (float(first["location"]["latitude"]), float(first["location"]["longitude"]))
        location = first.get("geometry", {}).get("location")
        return (float(location["lat"]), float(location["lng"])) if location else None
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return None


def query_google_geocoding(
    roads: Tuple[str, str], key: ApiKey, get: Callable[..., Any], sleep: Callable[[float], None],
) -> Optional[Tuple[float, float]]:
    address = f"{roads[0]}與{roads[1]}交叉口, 高雄市"
    payload = request_json(
        lambda: get(GEOCODE_V4_URL + quote(address), params={"regionCode": "TW", "languageCode": "zh-TW"},
                    headers={"X-Goog-Api-Key": key.value, "User-Agent": USER_AGENT}, timeout=30),
        sleep, f"Google Geocoding（{key}）",
    )
    return _parse_google_location(payload)


# =============================================================================
# 3. One camera
# =============================================================================

def geocode_camera(
    cfg: CameraHomographyConfig,
    *,
    post: Callable[..., Any],
    get: Callable[..., Any],
    google_key: Optional[ApiKey],
    sleep: Callable[[float], None],
) -> Candidate:
    name = cfg.name or ""
    roads = cfg.osm_roads or split_roads(name)
    if roads is None:
        return Candidate(cfg.camera, name, None, "", 0, None, None, None, "needs_review",
                         "名稱無法拆成兩條路；請在設定加 osm_roads 或手動填 latlon")
    try:
        points = query_overpass(roads, post, sleep)
    except GeocodeError as exc:
        return Candidate(cfg.camera, name, roads, "osm_overpass", 0, None, None, None, "error", str(exc))

    if points:
        lat, lon, spread = summarize_nodes(points)
        status = "ok" if spread <= SPREAD_LIMIT_M else "needs_review"
        note = "" if status == "ok" else f"共用節點分散 {spread:.0f} m（> {SPREAD_LIMIT_M:.0f} m）"
        return Candidate(cfg.camera, name, roads, "osm_overpass", len(points), lat, lon, spread, status, note)

    if google_key is None:
        return Candidate(cfg.camera, name, roads, "osm_overpass", 0, None, None, None, "needs_review",
                         "Overpass 找不到共用節點，且沒有 Google 金鑰可備援")
    try:
        location = query_google_geocoding(roads, google_key, get, sleep)
    except GeocodeError as exc:
        return Candidate(cfg.camera, name, roads, "google_geocoding", 0, None, None, None, "error", str(exc))
    if location is None:
        return Candidate(cfg.camera, name, roads, "google_geocoding", 0, None, None, None, "needs_review",
                         "Overpass 與 Google Geocoding 都找不到")
    return Candidate(cfg.camera, name, roads, "google_geocoding", 0, location[0], location[1], None,
                     "needs_review", "Google Geocoding 備援結果，請確認位置")


# =============================================================================
# 4. Candidate CSV and apply
# =============================================================================

_FIELD_NAMES = tuple(f.name for f in fields(Candidate))


def write_candidates(path: Path, candidates: Iterable[Candidate]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_FIELD_NAMES)
        for c in candidates:
            writer.writerow([
                c.camera, c.name, "|".join(c.roads) if c.roads else "", c.source, c.n_nodes,
                "" if c.lat is None else repr(c.lat), "" if c.lon is None else repr(c.lon),
                "" if c.spread_m is None else repr(c.spread_m), c.status, c.note,
            ])


def _opt_float(text: str) -> Optional[float]:
    return float(text) if text != "" else None


def read_candidates(path: Path) -> Tuple[Candidate, ...]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return tuple(
        Candidate(
            camera=r["camera"], name=r["name"],
            roads=tuple(r["roads"].split("|")) if r["roads"] else None,
            source=r["source"], n_nodes=int(r["n_nodes"]),
            lat=_opt_float(r["lat"]), lon=_opt_float(r["lon"]), spread_m=_opt_float(r["spread_m"]),
            status=r["status"], note=r["note"],
        )
        for r in rows
    )


def apply_candidates(
    candidates: Iterable[Candidate], camera_dir: Path, accept: Set[str], today: str,
) -> Tuple[str, ...]:
    """Write latlon/latlon_source/latlon_date for `ok` rows and accepted cameras only."""
    selected = [c for c in candidates if c.status == "ok" or c.camera in accept]
    missing = [c.camera for c in selected if c.lat is None or c.lon is None]
    if missing:
        raise ValueError(f"這些鏡頭沒有候選座標，無法套用：{missing}")
    for c in selected:
        update_camera_config_fields(
            camera_config_path(camera_dir, c.camera),
            {"latlon": [c.lat, c.lon], "latlon_source": c.source, "latlon_date": today},
        )
    return tuple(c.camera for c in selected)
