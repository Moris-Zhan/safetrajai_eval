"""
Metres per pixel of a camera's top-down (satellite) plane.

Web Mercator ground resolution of a Google Static Maps image:
    m_per_pixel = 156543.03392 * cos(latitude) / 2**zoom / scale
156543.03392 = 2 * pi * 6378137 / 256 (WGS84 equatorial radius, 256-px tiles).
The reference implementation (Sub_Module) omitted cos(latitude), which overstates
distances by about 8% in Kaohsiung.
"""

from __future__ import annotations

import math
from typing import Optional

from .camera_config import CameraHomographyConfig

EQUATOR_M_PER_PIXEL_Z0 = 156543.03392


def nominal_meters_per_pixel(cfg: CameraHomographyConfig) -> Optional[float]:
    """Formula value without the scale-check gate; only for running the scale check itself."""
    if cfg.satellite is None or cfg.latlon is None:
        return None
    return EQUATOR_M_PER_PIXEL_Z0 * math.cos(math.radians(cfg.latlon[0])) / 2 ** cfg.satellite.zoom / cfg.satellite.scale


def meters_per_pixel(cfg: CameraHomographyConfig) -> Optional[float]:
    """None unless the camera has a satellite matrix, satellite parameters, a latitude
    and a passed scale check; never falls back to an equatorial or default value."""
    if cfg.matrix_tag is None or cfg.scale_check != "passed" or cfg.needs_recalibration is not None:
        return None
    return nominal_meters_per_pixel(cfg)
