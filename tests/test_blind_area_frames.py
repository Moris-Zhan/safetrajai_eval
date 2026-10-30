"""Per-frame blind-area ratio from occluder tracks: heading handling, columns and skip counts."""

import numpy as np
import pytest

from safetrajai.collision.conflict.blind_area import blind_area_px, body_rectangle, compute_blind_area
from safetrajai.collision.conflict.config import ConflictConfig
from safetrajai.collision.conflict.kinematics import compute_kinematics
from safetrajai.collision.conflict.tracks import Track

MPP = 0.1                       # metres per map pixel
FPS = 10.0
SHAPE = (400, 400)
ROI = np.ones(SHAPE, dtype=bool)
VIEWPOINT = (10.0, 200.0)
LENGTHS = {"car": 5.0, "bus": 12.0}
WIDTHS = {"car": 1.8, "bus": 2.5}
CFG = ConflictConfig(smooth_window_s=0.1)   # window 1 frame: positions are used as they are


def track(class_name: str, track_id: int, xs, ys, first_frame: int = 1) -> Track:
    xy = np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])
    frames = np.arange(first_frame, first_frame + len(xy), dtype=np.int64)
    return Track(class_name, track_id, frames, xy, xy, xy * MPP)


def run(tracks, frames=None, viewpoint=VIEWPOINT):
    kins = [compute_kinematics(t, FPS, CFG) for t in tracks]
    frame_list = frames if frames is not None else sorted({int(f) for t in tracks for f in t.frames})
    return compute_blind_area(kins, viewpoint, ROI, MPP, FPS, frame_list, LENGTHS, WIDTHS)


def east_car(n: int = 8, start_x: float = 100.0, step: float = 5.0, y: float = 200.0, track_id: int = 1) -> Track:
    return track("car", track_id, [start_x + step * k for k in range(n)], [y] * n)


def test_columns_and_units() -> None:
    result = run([east_car()])

    assert list(result.columns) == ["frame", "time_s", "num_occluders", "blind_area_m2", "roi_area_m2", "blind_ratio"]
    assert result.columns["frame"] == list(range(1, 9))
    assert result.columns["time_s"] == pytest.approx([f / FPS for f in range(1, 9)])
    assert result.columns["roi_area_m2"] == [pytest.approx(SHAPE[0] * SHAPE[1] * MPP**2)] * 8
    ratio = np.array(result.columns["blind_ratio"])
    assert ratio == pytest.approx(np.array(result.columns["blind_area_m2"]) / np.array(result.columns["roi_area_m2"]))


def test_the_area_is_the_shadow_of_the_body_with_the_track_heading() -> None:
    result = run([east_car()])
    frame4 = result.columns["frame"].index(4)
    body = body_rectangle((115.0, 200.0), 0.0, 5.0 / MPP, 1.8 / MPP)     # x = 100 + 5 * 3

    expected_px, _ = blind_area_px(VIEWPOINT, [body], ROI)

    assert result.columns["blind_area_m2"][frame4] == pytest.approx(expected_px * MPP**2)
    assert result.columns["num_occluders"][frame4] == 1


def test_the_first_frame_has_no_heading_yet_and_is_skipped_but_counted() -> None:
    result = run([east_car()])

    assert result.columns["num_occluders"] == [0, 1, 1, 1, 1, 1, 1, 1]     # the last frame reuses the previous heading
    assert result.columns["blind_area_m2"][0] == 0.0
    assert result.frames_without_heading == 1
    assert result.occluders_without_heading == 0


def test_a_stopped_vehicle_keeps_its_last_valid_heading() -> None:
    moving = [100 + 5.0 * k for k in range(6)]
    stopped = track("car", 1, moving + [moving[-1]] * 6, [200.0] * 12)

    result = run([stopped])
    body = body_rectangle((moving[-1], 200.0), 0.0, 5.0 / MPP, 1.8 / MPP)
    expected_px, _ = blind_area_px(VIEWPOINT, [body], ROI)

    assert result.columns["blind_area_m2"][9] == pytest.approx(expected_px * MPP**2)    # frame 10, standing still
    assert result.columns["num_occluders"][9] == 1


def test_an_object_that_never_moves_is_skipped_and_counted_once() -> None:
    parked = track("car", 7, [150.0] * 6, [200.0] * 6)

    result = run([parked])

    assert result.occluders_without_heading == 1
    assert result.columns["num_occluders"] == [0] * 6
    assert result.columns["blind_area_m2"] == [0.0] * 6


def test_a_viewpoint_inside_a_body_gives_no_shadow_and_is_counted() -> None:
    result = run([east_car(start_x=0.0, step=2.0, n=12)], viewpoint=(10.0, 200.0))

    assert result.viewpoint_inside_skips >= 1
    assert result.columns["num_occluders"][3] == 0 and result.columns["blind_area_m2"][3] == 0.0    # frame 4: centre x = 6


def test_frames_without_any_occluder_are_listed_with_zero_area() -> None:
    result = run([east_car()], frames=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    assert result.columns["frame"] == list(range(1, 11))
    assert result.columns["blind_area_m2"][9] == 0.0 and result.columns["blind_ratio"][9] == 0.0
    assert result.columns["num_occluders"][9] == 0


def test_the_bodies_use_the_class_length_and_width() -> None:
    car = run([east_car()]).columns["blind_area_m2"][3]
    bus = run([track("bus", 1, [100 + 5.0 * k for k in range(8)], [200.0] * 8)]).columns["blind_area_m2"][3]

    assert bus > car


def test_a_zero_area_roi_gives_a_nan_ratio() -> None:
    kins = [compute_kinematics(east_car(), FPS, CFG)]

    result = compute_blind_area(kins, VIEWPOINT, np.zeros(SHAPE, dtype=bool), MPP, FPS, [1, 2], LENGTHS, WIDTHS)

    assert all(np.isnan(r) for r in result.columns["blind_ratio"])
