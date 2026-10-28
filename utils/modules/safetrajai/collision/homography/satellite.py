"""
Google Static Maps satellite downloads for camera configs (spec `camera-homography`, design §5).

Two purposes:
- `verify`: written under the verification directory; accepts demo or billing keys; only used
  to cross-check the scale of the existing base map, never by analyses.
- `basemap`: the base map used with the homography; requires a billing key, never overwrites
  an existing image unless `replace=True`, in which case the old one is quarantined to temp/.

Key values are only sent as a request parameter; errors mention the key name and kind only.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any, Callable, Tuple

import cv2
import numpy as np

from safetrajai.secrets import ApiKey

from .camera_config import CameraHomographyConfig, update_camera_config_fields

STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
PURPOSES = ("verify", "basemap")
MIN_GRAY_STD = 5.0


class SatelliteError(RuntimeError):
    pass


def image_tag(cfg: CameraHomographyConfig) -> str:
    """File stem of a camera's base map: its matrix tag, or `<name>-<scale>-<zoom>-<w>-<h>`."""
    if cfg.matrix_tag is not None:
        return cfg.matrix_tag
    width, height = cfg.satellite.output_size
    return f"{cfg.name or cfg.camera}-{cfg.satellite.scale}-{cfg.satellite.zoom}-{width}-{height}"


def static_map_params(cfg: CameraHomographyConfig) -> dict:
    lat, lon = cfg.latlon
    sat = cfg.satellite
    return {
        "center": f"{lat!r},{lon!r}",
        "zoom": sat.zoom,
        "size": f"{sat.size[0]}x{sat.size[1]}",
        "scale": sat.scale,
        "maptype": "satellite",
        "format": "png",
    }


def sanity_check(img: Any, expected_size: Tuple[int, int]) -> None:
    """Reject images with the wrong size or (almost) a single colour, e.g. error tiles."""
    if img is None:
        raise SatelliteError("回應不是可解碼的影像")
    height, width = img.shape[:2]
    if (width, height) != tuple(expected_size):
        raise SatelliteError(f"影像尺寸 {width}x{height} 與設定的 {expected_size[0]}x{expected_size[1]} 不符")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if float(np.std(gray)) < MIN_GRAY_STD:
        raise SatelliteError("影像幾乎是單一顏色，可能是錯誤圖或沒有衛星影像")


def _quarantine(png: Path, today: date, reason: str) -> None:
    temp_dir = png.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    moved = []
    for src in (png, Path(f"{png}.meta.json")):
        if src.exists():
            dst = temp_dir / src.name
            if dst.exists():
                raise SatelliteError(f"{dst} 已存在，請先處理舊的隔離檔")
            shutil.move(str(src), str(dst))
            moved.append((src, dst))
    with (temp_dir / "DO_NOT_USE.md").open("a", encoding="utf-8") as f:
        for src, dst in moved:
            f.write(
                f"\n## {src.name}\n- 原路徑：{src}\n- 日期：{today.isoformat()}\n- 原因：{reason}\n"
                f"- 還原：mv \"{dst}\" \"{src}\"\n"
            )


def download_satellite(
    cfg: CameraHomographyConfig,
    purpose: str,
    *,
    key: ApiKey,
    satellite_dir: Path,
    verification_dir: Path,
    replace: bool = False,
    get: Callable[..., Any],
    today: date,
) -> Path:
    """Download one camera's satellite image; returns the written PNG path."""
    if purpose not in PURPOSES:
        raise SatelliteError(f"未知的用途 {purpose!r}，可用：{PURPOSES}")
    if purpose == "basemap" and key.kind != "billing":
        raise SatelliteError(f"正式底圖只能用 billing 金鑰下載（目前是 {key}）；Demo Key 只能用於 verify")
    if replace and purpose != "basemap":
        raise SatelliteError("--replace 只適用於 basemap")
    if cfg.latlon is None or cfg.satellite is None:
        raise SatelliteError(f"{cfg.path}: 需要 latlon 與 satellite 才能下載")

    tag = image_tag(cfg)
    if purpose == "verify":
        out = Path(verification_dir) / cfg.camera / f"{tag}.png"
    else:
        out = Path(satellite_dir) / f"{tag}.png"
        if out.exists() and not replace:
            raise SatelliteError(f"{out} 已存在；要替換正式底圖請加 --replace")

    try:
        response = get(STATIC_MAPS_URL, params={**static_map_params(cfg), "key": key.value}, timeout=60)
    except Exception as exc:  # the exception text may contain the request URL including the key
        raise SatelliteError(f"Static Maps 請求失敗（{key}）：{type(exc).__name__}") from None
    if response.status_code != 200:
        detail = str(getattr(response, "text", "") or "").replace(key.value, "<redacted>")[:200]
        raise SatelliteError(f"Static Maps 回應 HTTP {response.status_code}（{key}）：{detail}")

    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    sanity_check(img, cfg.satellite.output_size)

    if purpose == "basemap" and out.exists():
        _quarantine(out, today, "以 scripts/homography/download_satellite.py --purpose basemap --replace 換成新底圖；"
                                "舊矩陣的對應點對新底圖無效，需要重新標定")
        update_camera_config_fields(cfg.path, {
            "matrix_tag": None, "needs_recalibration": cfg.matrix_tag, "scale_check": "unchecked",
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(response.content)
    sat = cfg.satellite
    meta = {
        "camera": cfg.camera, "provider": sat.provider, "zoom": sat.zoom, "scale": sat.scale,
        "size": list(sat.size), "center_latlon": list(cfg.latlon), "downloaded_at": today.isoformat(),
        "purpose": purpose, "key_type": key.kind,
    }
    Path(f"{out}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out
