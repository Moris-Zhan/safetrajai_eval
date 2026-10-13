"""Unit tests for camera geocoding (Overpass + Google fallback), with mocked HTTP."""

from pathlib import Path
from typing import List

import pytest

from safetrajai.collision.homography.camera_config import load_camera_config
from safetrajai.collision.homography.geocode import (
    Candidate,
    apply_candidates,
    geocode_camera,
    read_candidates,
    split_roads,
    write_candidates,
)
from safetrajai.secrets import ApiKey

FAKE_KEY = ApiKey(name="GOOGLE_MAPS_DEMO_API_KEY", kind="demo", value="fake-value-xyz")


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class Recorder:
    def __init__(self, responses: List[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.responses.pop(0)


def nodes(points):
    return {"elements": [{"type": "node", "lat": lat, "lon": lon} for lat, lon in points]}


def write_camera(tmp_path: Path, camera: str, name: str, extra: str = "") -> Path:
    path = tmp_path / f"{camera}.yaml"
    path.write_text(f"# header comment\ncamera: {camera}\nname: {name}\nscale_check: unchecked\n{extra}", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("name", "roads"),
    [
        ("C000025 中華三路 民生二路(東北角)", ("中華三路", "民生二路")),
        ("C000026 中華三路 民生二路口東南角", ("中華三路", "民生二路")),
        ("037- [翠華路、海功東路 通道-01]", ("翠華路", "海功東路")),
        ("C000096 前鎮區 中山三路 凱旋四路(輕軌旁)", ("中山三路", "凱旋四路")),
        ("BEV0001 高雄 大順路富民路", ("大順路", "富民路")),
        ("C000011 中正路交流道南下入口", None),
        ("C000294 一心凱旋西南角", None),
    ],
)
def test_split_roads(name: str, roads) -> None:
    assert split_roads(name) == roads


def test_overpass_centroid_ok(tmp_path: Path) -> None:
    cfg = load_camera_config(write_camera(tmp_path, "C000025", "C000025 中華三路 民生二路(東北角)"))
    pts = [(22.6258 + i * 1e-5, 120.2972 + i * 1e-5) for i in range(16)]
    post = Recorder([FakeResponse(200, nodes(pts))])
    cand = geocode_camera(cfg, post=post, get=Recorder([]), google_key=None, sleep=lambda s: None)
    assert cand.status == "ok" and cand.source == "osm_overpass" and cand.n_nodes == 16
    assert cand.lat == pytest.approx(sum(p[0] for p in pts) / 16)
    assert cand.spread_m < 50


def test_large_spread_needs_review(tmp_path: Path) -> None:
    cfg = load_camera_config(write_camera(tmp_path, "C000033", "C000033 大中一路 民族一路"))
    post = Recorder([FakeResponse(200, nodes([(22.678, 120.319), (22.681, 120.319)]))])
    cand = geocode_camera(cfg, post=post, get=Recorder([]), google_key=None, sleep=lambda s: None)
    assert cand.status == "needs_review" and cand.spread_m > 50


def test_osm_roads_override(tmp_path: Path) -> None:
    cfg = load_camera_config(
        write_camera(tmp_path, "C000085", "C000085 中正路 三多路(建軍)", "osm_roads: [中正一路, 三多二路]\n")
    )
    post = Recorder([FakeResponse(200, nodes([(22.6, 120.3)]))])
    cand = geocode_camera(cfg, post=post, get=Recorder([]), google_key=None, sleep=lambda s: None)
    assert cand.roads == ("中正一路", "三多二路")
    assert "中正一路" in post.calls[0][1]["data"]["data"]


def test_unparsable_name_needs_review_without_requests(tmp_path: Path) -> None:
    cfg = load_camera_config(write_camera(tmp_path, "C000011", "C000011 中正路交流道南下入口"))
    post, get = Recorder([]), Recorder([])
    cand = geocode_camera(cfg, post=post, get=get, google_key=FAKE_KEY, sleep=lambda s: None)
    assert cand.status == "needs_review" and post.calls == [] and get.calls == []


def test_rate_limit_retries_then_error(tmp_path: Path) -> None:
    cfg = load_camera_config(write_camera(tmp_path, "C000054", "C000054 民族一路 天祥二路"))
    delays: list = []
    post = Recorder([FakeResponse(429), FakeResponse(200, None, "<html>busy</html>"),
                     FakeResponse(429), FakeResponse(429)])
    cand = geocode_camera(cfg, post=post, get=Recorder([]), google_key=None, sleep=delays.append)
    assert cand.status == "error" and len(post.calls) == 4
    assert sorted(delays) == delays and delays[-1] > delays[0]


def test_google_fallback_needs_review_and_hides_key(tmp_path: Path) -> None:
    cfg = load_camera_config(write_camera(tmp_path, "C000221", "C000221 中山四路 中平路(西北角)"))
    post = Recorder([FakeResponse(200, nodes([]))])
    get = Recorder([FakeResponse(200, {"results": [{"location": {"latitude": 22.57, "longitude": 120.33}}]})])
    cand = geocode_camera(cfg, post=post, get=get, google_key=FAKE_KEY, sleep=lambda s: None)
    assert cand.status == "needs_review" and cand.source == "google_geocoding"
    assert (cand.lat, cand.lon) == (22.57, 120.33)
    assert FAKE_KEY.value not in repr(cand)
    assert all(FAKE_KEY.value not in str(args) for args, _ in get.calls)


def test_unexpected_google_payload_needs_review(tmp_path: Path) -> None:
    cfg = load_camera_config(write_camera(tmp_path, "C000221", "C000221 中山四路 中平路(西北角)"))
    post = Recorder([FakeResponse(200, nodes([]))])
    get = Recorder([FakeResponse(200, {"results": [{"location": {"lat": "x"}}]})])
    cand = geocode_camera(cfg, post=post, get=get, google_key=FAKE_KEY, sleep=lambda s: None)
    assert cand.status == "needs_review" and cand.lat is None


def test_candidates_csv_roundtrip(tmp_path: Path) -> None:
    rows = (Candidate("C000025", "n", ("a", "b"), "osm_overpass", 3, 22.1, 120.2, 5.0, "ok", ""),)
    write_candidates(tmp_path / "c.csv", rows)
    assert read_candidates(tmp_path / "c.csv") == rows


def test_apply_writes_only_ok_or_accepted(tmp_path: Path) -> None:
    for cam in ("C000025", "C000033", "C000011"):
        write_camera(tmp_path, cam, f"{cam} x路 y路")
    rows = (
        Candidate("C000025", "", ("x路", "y路"), "osm_overpass", 16, 22.62, 120.29, 3.0, "ok", ""),
        Candidate("C000033", "", ("x路", "y路"), "osm_overpass", 5, 22.67, 120.31, 331.0, "needs_review", ""),
        Candidate("C000011", "", None, "", 0, None, None, None, "needs_review", ""),
    )
    before_033 = (tmp_path / "C000033.yaml").read_text(encoding="utf-8")
    updated = apply_candidates(rows, tmp_path, accept=set(), today="2026-10-13")
    assert updated == ("C000025",)
    assert (tmp_path / "C000033.yaml").read_text(encoding="utf-8") == before_033

    cfg = load_camera_config(tmp_path / "C000025.yaml")
    assert cfg.latlon == (22.62, 120.29) and cfg.latlon_source == "osm_overpass" and cfg.latlon_date == "2026-10-13"
    assert (tmp_path / "C000025.yaml").read_text(encoding="utf-8").startswith("# header comment\n")

    assert apply_candidates(rows, tmp_path, accept={"C000033"}, today="2026-10-13") == ("C000025", "C000033")


def test_accepting_row_without_coordinates_is_error(tmp_path: Path) -> None:
    write_camera(tmp_path, "C000011", "C000011 x")
    rows = (Candidate("C000011", "", None, "", 0, None, None, None, "needs_review", ""),)
    with pytest.raises(ValueError):
        apply_candidates(rows, tmp_path, accept={"C000011"}, today="2026-10-13")
