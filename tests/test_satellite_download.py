"""Unit tests for satellite base-map downloads (purpose rules, key handling), with mocked HTTP."""

import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

from safetrajai.collision.homography.camera_config import load_camera_config
from safetrajai.collision.homography.satellite import SatelliteError, download_satellite, sanity_check
from safetrajai.secrets import ApiKey

TAG = "C000025 中華三路 民生二路(東北角)-2-20-1280-1280"
DEMO = ApiKey(name="GOOGLE_MAPS_DEMO_API_KEY", kind="demo", value="fake-demo-abc")
BILLING = ApiKey(name="GOOGLE_MAPS_API_KEY", kind="billing", value="fake-billing-def")


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": "image/png"}


class Recorder:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


def png_bytes(size=(1280, 1280), textured: bool = True) -> bytes:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8) if textured else np.full(
        (size[1], size[0], 3), 200, np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture()
def setup(tmp_path: Path) -> dict:
    cameras = tmp_path / "config" / "homography"
    cameras.mkdir(parents=True)
    path = cameras / "C000025.yaml"
    path.write_text(
        "# keep me\ncamera: C000025\nname: C000025 中華三路 民生二路(東北角)\n"
        f"matrix_tag: {TAG}\n"
        "satellite: {provider: google_static, zoom: 20, scale: 2, size: [640, 640]}\n"
        "latlon: [22.625869, 120.297201]\nscale_check: passed\n",
        encoding="utf-8",
    )
    sat_dir = tmp_path / "models" / "satellite_images"
    sat_dir.mkdir(parents=True)
    return {"cfg_path": path, "sat_dir": sat_dir, "verify_dir": tmp_path / "models" / "verification"}


def call(setup: dict, purpose: str, key: ApiKey, get, replace: bool = False) -> Path:
    return download_satellite(
        load_camera_config(setup["cfg_path"]), purpose, key=key,
        satellite_dir=setup["sat_dir"], verification_dir=setup["verify_dir"],
        replace=replace, get=get, today=date(2026, 10, 13),
    )


def test_demo_key_cannot_download_basemap(setup: dict) -> None:
    get = Recorder(FakeResponse(200, png_bytes()))
    with pytest.raises(SatelliteError) as excinfo:
        call(setup, "basemap", DEMO, get)
    assert get.calls == []
    assert DEMO.value not in str(excinfo.value)


def test_verify_download_writes_verification_dir_only(setup: dict) -> None:
    existing = setup["sat_dir"] / f"{TAG}.png"
    existing.write_bytes(b"old")
    out = call(setup, "verify", DEMO, Recorder(FakeResponse(200, png_bytes())))
    assert out == setup["verify_dir"] / "C000025" / f"{TAG}.png" and out.exists()
    assert existing.read_bytes() == b"old"

    meta_text = Path(f"{out}.meta.json").read_text(encoding="utf-8")
    meta = json.loads(meta_text)
    assert meta["purpose"] == "verify" and meta["key_type"] == "demo"
    assert meta["downloaded_at"] == "2026-10-13"
    assert meta["center_latlon"] == [22.625869, 120.297201]
    assert DEMO.value not in meta_text


def test_existing_basemap_is_not_overwritten(setup: dict) -> None:
    existing = setup["sat_dir"] / f"{TAG}.png"
    existing.write_bytes(b"old")
    get = Recorder(FakeResponse(200, png_bytes()))
    with pytest.raises(SatelliteError):
        call(setup, "basemap", BILLING, get)
    assert get.calls == [] and existing.read_bytes() == b"old"


def test_flat_image_fails_sanity_check(setup: dict) -> None:
    with pytest.raises(SatelliteError):
        call(setup, "verify", DEMO, Recorder(FakeResponse(200, png_bytes(textured=False))))
    with pytest.raises(SatelliteError):
        sanity_check(np.zeros((640, 640, 3), np.uint8), (1280, 1280))


def test_http_error_hides_key(setup: dict) -> None:
    response = FakeResponse(403)
    response.text = f"This API is not activated on your API project (key={DEMO.value})"
    with pytest.raises(SatelliteError) as excinfo:
        call(setup, "verify", DEMO, Recorder(response))
    message = str(excinfo.value)
    assert "403" in message and "not activated" in message and DEMO.value not in message


def test_replace_quarantines_old_basemap(setup: dict) -> None:
    existing = setup["sat_dir"] / f"{TAG}.png"
    existing.write_bytes(b"old")
    out = call(setup, "basemap", BILLING, Recorder(FakeResponse(200, png_bytes())), replace=True)
    assert out == existing and existing.read_bytes() != b"old"
    assert (setup["sat_dir"] / "temp" / f"{TAG}.png").read_bytes() == b"old"
    assert TAG in (setup["sat_dir"] / "temp" / "DO_NOT_USE.md").read_text(encoding="utf-8")

    cfg = load_camera_config(setup["cfg_path"])
    assert cfg.matrix_tag is None and cfg.needs_recalibration == TAG and cfg.scale_check == "unchecked"
    assert setup["cfg_path"].read_text(encoding="utf-8").startswith("# keep me\n")
