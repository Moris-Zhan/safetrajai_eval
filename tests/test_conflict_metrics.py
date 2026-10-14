"""Unit tests for the stage3 conflict-metrics analysis (change conflict-metrics, ADR 0011)."""

import math

import numpy as np
import pytest

from safetrajai.collision.conflict.config import ConflictConfig, load_conflict_config


# =============================================================================
# 1. Config
# =============================================================================

def test_missing_block_is_disabled() -> None:
    cfg = load_conflict_config(None)
    assert cfg.enabled is False


def test_defaults_follow_reference_implementation() -> None:
    cfg = load_conflict_config({"enabled": True})
    assert cfg.ttc_distance_m == 2.0
    assert cfg.ttc_distance_mode == "fixed"
    assert cfg.pet_max_gap_s == 0.0
    assert [label for _, label in cfg.angle_types] == ["Rear-end", "Lane-change", "Crossing"]
    assert len(cfg.class_pairs) == 20
    assert ("person", "person") not in cfg.class_pairs
    assert ("car", "person") in cfg.class_pairs


@pytest.mark.parametrize(
    "block",
    [
        {"smooth_window_s": 0},
        {"min_fps_for_acceleration": 0},
        {"ttc_distance_m": -1},
        {"event_threshold_s": -0.5},
        {"angle_types": [[30, "Rear-end"], [20, "Lane-change"], [180, "Crossing"]]},
        {"angle_types": [[30, "Rear-end"], [85, "Lane-change"], [170, "Crossing"]]},
        {"class_pairs": [["car", "tram"]]},
        {"object_length_m": {"car": 5.0}},
        {"object_mass_kg": {"person": 70}},
        {"ttc_distance_mode": "box"},
        {"unknown_key": 1},
    ],
)
def test_invalid_config_is_rejected(block: dict) -> None:
    with pytest.raises(ValueError):
        load_conflict_config({"enabled": True, **block})


# =============================================================================
# 2. Tracks
# =============================================================================

SEQ = "C000025 test_seq"


def write_mot(seq_dir, class_name: str, rows) -> None:
    path = seq_dir / "trackers" / "mot_challenge" / "CCTV-val" / f"YOLO11x__{class_name}" / "data" / f"{seq_dir.name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(",".join(str(v) for v in r) + "\n" for r in rows), encoding="utf-8")


def test_tracks_use_bottom_centre_roi_centre_and_class_keys(tmp_path) -> None:
    from safetrajai.collision.conflict.tracks import load_tracks

    seq_dir = tmp_path / SEQ
    write_mot(seq_dir, "car", [(1, 7, 100, 200, 40, 20, 0.9), (2, 7, 102, 200, 40, 20, 0.9), (1, 9, 900, 900, 10, 10, 0.9)])
    write_mot(seq_dir, "scooter", [(1, 7, 300, 100, 10, 30, 0.8)])
    roi = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype=np.int32)
    tracks = load_tracks(seq_dir, "CCTV", "val", "YOLO11x", roi, np.eye(3), 0.1)

    keys = [(t.class_name, t.track_id) for t in tracks]
    assert keys == [("car", 7), ("scooter", 7)]
    car = tracks[0]
    np.testing.assert_array_equal(car.frames, [1, 2])
    np.testing.assert_allclose(car.cctv_xy[0], [120.0, 220.0])
    np.testing.assert_allclose(car.map_xy[0], [120.0, 220.0])
    np.testing.assert_allclose(car.xy_m[0], [12.0, 22.0])


def test_tracks_project_through_homography(tmp_path) -> None:
    from safetrajai.collision.conflict.tracks import load_tracks

    seq_dir = tmp_path / SEQ
    write_mot(seq_dir, "car", [(1, 1, 10, 10, 20, 10, 0.9)])
    roi = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype=np.int32)
    H = np.array([[2.0, 0, 5.0], [0, 2.0, 0], [0, 0, 1.0]])
    (track,) = load_tracks(seq_dir, "CCTV", "val", "YOLO11x", roi, H, 0.5)
    np.testing.assert_allclose(track.map_xy[0], [45.0, 40.0])
    np.testing.assert_allclose(track.xy_m[0], [22.5, 20.0])


def make_track(class_name: str, track_id: int, frames, xy_m) -> "Track":
    from safetrajai.collision.conflict.tracks import Track

    xy = np.asarray(xy_m, dtype=float)
    return Track(class_name, track_id, np.asarray(frames, dtype=np.int64), xy * 10.0, xy * 10.0, xy)


# =============================================================================
# 3. Kinematics
# =============================================================================

@pytest.mark.parametrize(("fps", "frames"), [(4.9, 3), (30.0, 19), (1.0, 1)])
def test_smoothing_window_conversion(fps: float, frames: int) -> None:
    from safetrajai.collision.conflict.kinematics import smoothing_window_frames

    assert smoothing_window_frames(0.6, fps) == frames


def test_constant_velocity() -> None:
    from safetrajai.collision.conflict.kinematics import compute_kinematics

    fps = 30.0
    frames = np.arange(30)
    track = make_track("car", 1, frames, np.c_[10.0 * frames / fps, np.zeros(30)])
    kin = compute_kinematics(track, fps, ConflictConfig())
    assert np.all(np.isnan(kin.vel[[0, -1]]))
    np.testing.assert_allclose(kin.speed[1:-1], 10.0, atol=1e-6)
    np.testing.assert_allclose(kin.acc[2:-2], 0.0, atol=1e-6)
    np.testing.assert_allclose(kin.heading_deg[1:-1], 0.0, atol=1e-9)


def test_constant_acceleration_interior() -> None:
    from safetrajai.collision.conflict.kinematics import compute_kinematics

    fps = 30.0
    t = np.arange(90) / fps
    track = make_track("car", 1, np.arange(90), np.c_[0.5 * 2.0 * t ** 2, np.zeros(90)])
    kin = compute_kinematics(track, fps, ConflictConfig())
    np.testing.assert_allclose(kin.acc[15:-15, 0], 2.0, atol=1e-6)


def test_low_fps_has_no_acceleration() -> None:
    from safetrajai.collision.conflict.kinematics import compute_kinematics

    fps = 4.9
    frames = np.arange(20)
    track = make_track("car", 1, frames, np.c_[10.0 * frames / fps, np.zeros(20)])
    kin = compute_kinematics(track, fps, ConflictConfig())
    assert np.all(np.isnan(kin.acc))
    np.testing.assert_allclose(kin.speed[1:-1], 10.0, atol=1e-6)
    assert kin.acceleration_available is False


def test_gap_splits_segments() -> None:
    from safetrajai.collision.conflict.kinematics import compute_kinematics

    frames = np.r_[np.arange(0, 11), np.arange(30, 41)]
    xy = np.c_[frames * 1.0, np.zeros(len(frames))]
    kin = compute_kinematics(make_track("car", 1, frames, xy), 10.0, ConflictConfig(max_gap_frames=5))
    assert np.isnan(kin.vel[10]).all() and np.isnan(kin.vel[11]).all()
    np.testing.assert_allclose(kin.speed[[5, 15]], 10.0, atol=1e-6)


def test_slow_track_has_no_heading() -> None:
    from safetrajai.collision.conflict.kinematics import compute_kinematics

    frames = np.arange(10)
    kin = compute_kinematics(make_track("person", 1, frames, np.c_[frames * 0.01, np.zeros(10)]), 10.0, ConflictConfig())
    assert np.all(np.isnan(kin.heading_deg))


# =============================================================================
# 4. Pairs, conflict angle and type
# =============================================================================

def line_track(class_name: str, track_id: int, start_xy, velocity, n: int = 20, fps: float = 10.0, first_frame: int = 0):
    frames = np.arange(first_frame, first_frame + n)
    t = (frames - first_frame) / fps
    xy = np.asarray(start_xy, float) + np.outer(t, np.asarray(velocity, float))
    return make_track(class_name, track_id, frames, xy)


def kinematics_of(tracks, fps: float = 10.0, cfg: ConflictConfig = ConflictConfig()):
    from safetrajai.collision.conflict.kinematics import compute_kinematics

    return [compute_kinematics(t, fps, cfg) for t in tracks]


def pair_keys(pairs, kins) -> set:
    return {(kins[i].track.key, kins[j].track.key) for i, j in zip(pairs.i, pairs.j)}


def test_pairs_respect_radius_classes_and_stationary() -> None:
    from safetrajai.collision.conflict.pairs import build_pairs

    tracks = [
        line_track("car", 1, (0, 0), (10, 0)),
        line_track("car", 2, (80, 0), (10, 0)),          # 80 m away from car 1
        line_track("person", 3, (3, 3), (1, 0)),
        line_track("person", 4, (4, 3), (1, 0)),         # person-person, excluded by default
        line_track("scooter", 5, (200, 0), (0, 0)),
        line_track("car", 6, (201, 0), (0, 0)),          # both stationary
    ]
    kins = kinematics_of(tracks)
    keys = pair_keys(build_pairs(kins, ConflictConfig()), kins)
    assert (("car", 1), ("car", 2)) not in keys
    assert (("person", 3), ("person", 4)) not in keys
    assert (("car", 1), ("person", 3)) in keys
    assert (("car", 6), ("scooter", 5)) not in keys and (("scooter", 5), ("car", 6)) not in keys


@pytest.mark.parametrize(("angle", "label"), [(0.0, "Rear-end"), (30.0, "Rear-end"), (30.5, "Lane-change"),
                                              (85.0, "Lane-change"), (90.0, "Crossing"), (170.0, "Crossing")])
def test_classify_angle(angle: float, label: str) -> None:
    from safetrajai.collision.conflict.pairs import classify_angle

    assert classify_angle(np.array([angle]), ConflictConfig().angle_types)[0] == label


def test_angle_is_clipped_and_undefined_when_slow() -> None:
    from safetrajai.collision.conflict.pairs import classify_angle, conflict_angle_deg

    v = np.array([[0.1 + 1e-17, 0.3]])
    assert not np.isnan(conflict_angle_deg(v, v * (1 + 1e-15))).any()
    assert classify_angle(np.array([np.nan]), ConflictConfig().angle_types)[0] == "Undefined"


def test_one_slow_track_is_undefined_type() -> None:
    from safetrajai.collision.conflict.pairs import build_pairs

    kins = kinematics_of([line_track("car", 1, (0, 0), (10, 0)), line_track("person", 2, (5, 2), (0.1, 0))])
    pairs = build_pairs(kins, ConflictConfig())
    assert len(pairs.frame) > 0 and set(pairs.conflict_type) == {"Undefined"}


# =============================================================================
# 5. TTC / MTTC / leading and following
# =============================================================================

def ttc_rows(tracks, fps: float = 10.0, cfg: ConflictConfig = ConflictConfig()):
    from safetrajai.collision.conflict.pairs import build_pairs
    from safetrajai.collision.conflict.ttc import ttc_for_pairs

    kins = kinematics_of(tracks, fps, cfg)
    pairs = build_pairs(kins, cfg)
    return kins, pairs, ttc_for_pairs(kins, pairs, cfg)


def first_row(pairs, frame: int) -> int:
    return int(np.flatnonzero(pairs.frame == frame)[0])


def test_head_on_ttc() -> None:
    # At frame 1 (t = 0.1 s) the cars are at x = 0 and x = 22, closing at 20 m/s.
    kins, pairs, res = ttc_rows([line_track("car", 1, (-1, 0), (10, 0)), line_track("car", 2, (23, 0), (-10, 0))])
    r = first_row(pairs, 1)
    assert res.ttc[r] == pytest.approx(1.0, abs=0.02 + 1e-9)   # within one step
    np.testing.assert_allclose(res.collision_xy_m[r], [11.0, 0.0], atol=0.3)


def test_class_length_mode_distance() -> None:
    cfg = ConflictConfig(ttc_distance_mode="class_length")
    assert cfg.collision_distance_m("car", "car") == pytest.approx(5.0)
    assert cfg.collision_distance_m("car", "scooter") == pytest.approx(3.4)
    assert ConflictConfig().collision_distance_m("car", "bus") == 2.0


def test_overlap_at_start() -> None:
    kins, pairs, res = ttc_rows([line_track("car", 1, (0, 0), (10, 0)), line_track("car", 2, (1.5, 0), (10, 0.0))])
    r = first_row(pairs, 1)
    assert res.overlap_at_start[r] and np.isnan(res.ttc[r]) and np.isnan(res.mttc[r])


def test_separating_pair_is_infinite() -> None:
    kins, pairs, res = ttc_rows([line_track("car", 1, (0, 0), (-10, 0)), line_track("car", 2, (5, 0), (10, 0))])
    r = first_row(pairs, 1)
    assert res.ttc[r] == np.inf and res.mttc[r] == np.inf


def test_acceleration_only_gives_no_mttc_and_ttc_finite_gives_mttc() -> None:
    fps = 30.0
    frames = np.arange(60)
    t = frames / fps
    rear = make_track("car", 1, frames, np.c_[10 * t + 0.5 * 3.0 * t ** 2, np.zeros(60)])
    # At frame 20 the rear car moves at 10 + 3 * 20/30 = 12 m/s, the same as the front car.
    front = make_track("car", 2, frames, np.c_[8.0 + 12 * t, np.zeros(60)])
    _, pairs, res = ttc_rows([rear, front], fps=fps)
    r = first_row(pairs, 20)
    assert res.ttc[r] == np.inf and res.mttc[r] == np.inf

    slow_front = make_track("car", 2, frames, np.c_[20.0 + 5 * t, np.zeros(60)])
    _, pairs, res = ttc_rows([rear, slow_front], fps=fps)
    r = first_row(pairs, 20)
    assert np.isfinite(res.ttc[r]) and np.isfinite(res.mttc[r]) and res.mttc[r] <= res.ttc[r]


def test_mttc_undefined_without_acceleration() -> None:
    _, pairs, res = ttc_rows([line_track("car", 1, (0, 0), (15, 0)), line_track("car", 2, (10, 0), (10, 0))], fps=4.9)
    r = first_row(pairs, 1)
    assert np.isfinite(res.ttc[r]) and np.isnan(res.mttc[r])


def test_rear_end_leading_is_front_car() -> None:
    kins, pairs, res = ttc_rows([line_track("car", 1, (0, 0), (15, 0)), line_track("car", 2, (10, 0), (10, 0))])
    r = first_row(pairs, 1)
    assert np.isfinite(res.ttc[r])
    assert res.leading[r] == 2   # track j (the front car) leads


# =============================================================================
# 6. Front/rear and rear-end metrics
# =============================================================================

def test_closest_front_and_rear() -> None:
    from safetrajai.collision.conflict.following import closest_front_rear
    from safetrajai.collision.conflict.pairs import build_pairs

    kins = kinematics_of([line_track("car", 1, (0, 0), (10, 0)), line_track("car", 2, (10, 0), (10, 0)),
                          line_track("car", 3, (20, 0), (10, 0))])
    table = closest_front_rear(kins, build_pairs(kins, ConflictConfig()))
    assert table[(1, 5)] == (2, 0)      # B: front = C (index 2), rear = A (index 0)
    assert table[(0, 5)] == (1, -1)     # A: front = B, no rear
    assert table[(2, 5)] == (-1, 1)     # C: no front, rear = B


def rear_end(rear_v: float, front_v: float, gap: float, fps: float = 10.0):
    from safetrajai.collision.conflict.following import rear_end_metrics

    kins, pairs, res = ttc_rows([line_track("car", 1, (0, 0), (rear_v, 0)), line_track("car", 2, (gap, 0), (front_v, 0))], fps=fps)
    return pairs, rear_end_metrics(kins, pairs, res, ConflictConfig())


def test_drac_uses_centre_distance() -> None:
    pairs, met = rear_end(15.0, 10.0, 10.0)
    r = first_row(pairs, 1)
    gap = 10.0 - 5.0 * 0.1                   # at frame 1 the rear car has closed 0.5 m
    assert met.rear_side[r] == 1
    assert met.delta_s[r] == pytest.approx(gap)
    assert met.delta_v[r] == pytest.approx(5.0)
    assert met.drac[r] == pytest.approx(25.0 / (2 * gap))


def test_drac_undefined_when_not_closing() -> None:
    pairs, met = rear_end(10.0, 15.0, 10.0)
    assert np.all(np.isnan(met.drac))


def test_drac_exceedance() -> None:
    from safetrajai.collision.conflict.following import drac_exceedance

    exceeds, excess = drac_exceedance(np.array([1.25, 0.5, np.nan]), np.array([0.8, 0.8, 0.8]))
    np.testing.assert_array_equal(exceeds[:2], [1.0, 0.0])
    assert excess[0] == pytest.approx(0.45) and np.isnan(exceeds[2]) and np.isnan(excess[2])


def test_low_fps_has_no_acceleration_metrics() -> None:
    pairs, met = rear_end(15.0, 10.0, 10.0, fps=4.9)
    r = first_row(pairs, 1)
    assert np.isfinite(met.drac[r])
    for name in ("dr", "maxd", "drac_exceeds", "drac_excess", "ci", "ci_m"):
        assert np.isnan(getattr(met, name)[r]), name


def test_crash_index_uses_magnitudes() -> None:
    from safetrajai.collision.conflict.following import crash_index

    assert crash_index(15.0, 10.0, 0.0, 0.0, 2.0) == pytest.approx(31.25)
    assert crash_index(15.0, 10.0, 0.0, abs(-2.0), 2.0) == crash_index(15.0, 10.0, 0.0, abs(2.0), 2.0)
    assert crash_index(15.0, 10.0, 0.0, 0.0, 2.0, m_follow=2150, m_lead=200) == pytest.approx(
        (2150 * 15 ** 2 - 200 * 10 ** 2) / 4.0)


# =============================================================================
# 7. PET
# =============================================================================

def crossing_tracks(a_frames, b_frames):
    """Track a moves along +x and passes the origin at frame 28; b moves along +y and passes it at frame 44."""
    fa, fb = np.asarray(a_frames), np.asarray(b_frames)
    a = make_track("car", 1, fa, np.c_[(fa - 28) * 1.0, np.zeros(len(fa))])
    b = make_track("scooter", 2, fb, np.c_[np.zeros(len(fb)), (fb - 44) * 1.0])
    return [a, b]


def test_pet_for_co_present_tracks() -> None:
    from safetrajai.collision.conflict.pet import compute_pet

    (row,) = compute_pet(crossing_tracks(range(10, 41), range(25, 60)), 10.0, ConflictConfig())
    assert row.pet == pytest.approx(1.2)
    assert row.first == 1 and (row.t1_out, row.t2_in) == (pytest.approx(3.0), pytest.approx(4.2))
    np.testing.assert_allclose(row.conflict_xy_m, [0.0, 0.0])


def test_pet_requires_co_presence_by_default() -> None:
    from safetrajai.collision.conflict.pet import compute_pet

    tracks = crossing_tracks(range(10, 31), range(40, 60))
    assert compute_pet(tracks, 10.0, ConflictConfig()) == ()
    (row,) = compute_pet(tracks, 10.0, ConflictConfig(pet_max_gap_s=5.0))
    assert row.pet == pytest.approx(1.2) and row.first == 1


def test_pet_zero_when_occupancy_overlaps() -> None:
    from safetrajai.collision.conflict.pet import compute_pet

    fa = np.arange(20, 60)
    a = make_track("car", 1, fa, np.c_[(fa - 40) * 0.2, np.zeros(len(fa))])   # inside the zone frames 30..50
    b = make_track("car", 2, fa, np.c_[np.zeros(len(fa)), (fa - 44) * 1.0])   # inside the zone frames 42..46
    (row,) = compute_pet([a, b], 10.0, ConflictConfig())
    assert row.pet == 0.0


def test_no_pet_when_paths_stay_apart() -> None:
    from safetrajai.collision.conflict.pet import compute_pet

    frames = np.arange(10, 40)
    a = make_track("car", 1, frames, np.c_[frames * 1.0, np.zeros(len(frames))])
    b = make_track("car", 2, frames, np.c_[frames * 1.0, np.full(len(frames), 5.0)])   # parallel, 5 m apart
    assert compute_pet([a, b], 10.0, ConflictConfig()) == ()


# =============================================================================
# 8. Rear-end chains and travel direction
# =============================================================================

def chains_of(tracks, cfg: ConflictConfig = ConflictConfig()):
    from safetrajai.collision.conflict.chains import build_chains
    from safetrajai.collision.conflict.following import rear_end_metrics

    kins, pairs, res = ttc_rows(tracks, cfg=cfg)
    return build_chains(kins, pairs, rear_end_metrics(kins, pairs, res, cfg), cfg)


def test_platoon_forms_one_chain_ranked_rear_to_front() -> None:
    rows = chains_of([line_track("car", 1, (16, 0), (10, 0)), line_track("car", 2, (8, 0), (10, 0)),
                      line_track("car", 3, (0, 0), (10, 0))])
    assert len({r.chain_id for r in rows}) == 1 and all(r.chain_id > 0 for r in rows)
    assert {r.direction for r in rows} == {"E"}
    rank = {r.track_index: r.rank for r in rows}
    assert rank == {2: 1, 1: 2, 0: 3}


def test_spaced_cars_form_separate_chains() -> None:
    rows = chains_of([line_track("car", 1, (0, 0), (10, 0)), line_track("car", 2, (30, 0), (10, 0))])
    assert len({r.chain_id for r in rows}) == 2


@pytest.mark.parametrize(("heading", "direction"), [(0.0, "E"), (44.9, "E"), (45.0, "S"), (135.0, "W"),
                                                    (225.0, "N"), (315.0, "E"), (np.nan, "")])
def test_travel_direction_sectors(heading: float, direction: str) -> None:
    from safetrajai.collision.conflict.chains import travel_direction

    assert travel_direction(heading) == direction


# =============================================================================
# 9. Events
# =============================================================================

def synthetic_rows(frames, ttc, overlap=None, i=0, j=1):
    from safetrajai.collision.conflict.following import RearEndMetrics
    from safetrajai.collision.conflict.pairs import PairFrames
    from safetrajai.collision.conflict.ttc import TtcResult

    m = len(frames)
    ints = lambda v: np.full(m, v, dtype=np.int64)
    nan = np.full(m, np.nan)
    pairs = PairFrames(np.asarray(frames, np.int64), ints(i), ints(j), ints(0), ints(0), np.zeros(m),
                       np.full(m, "Rear-end", dtype=object))
    overlap = np.zeros(m, bool) if overlap is None else np.asarray(overlap)
    ttc_arr = np.asarray(ttc, float)
    res = TtcResult(ttc_arr, np.where(np.isfinite(ttc_arr), ttc_arr, np.inf), overlap,
                    np.zeros((m, 2)), ints(2))
    met = RearEndMetrics(ints(1), *([nan] * 10))
    return pairs, res, met


def two_tracks():
    return kinematics_of([line_track("car", 1, (0, 0), (10, 0)), line_track("scooter", 2, (5, 0), (5, 0))])


def test_consecutive_hits_form_one_event() -> None:
    from safetrajai.collision.conflict.events import build_events, threshold_counts

    events = build_events(two_tracks(), *synthetic_rows(range(10, 22), [1.2] * 12), ConflictConfig())
    assert len(events) == 1 and (events[0].start_frame, events[0].end_frame) == (10, 21)
    counts = threshold_counts(events, ConflictConfig().stat_thresholds_s, "min_ttc")
    assert counts["< 1.5 s"] == 1 and counts["< 1.0 s"] == 0


def test_gap_handling_between_hits() -> None:
    from safetrajai.collision.conflict.events import build_events

    one_gap = list(range(10, 15)) + list(range(16, 20))       # one missing frame
    five_gap = list(range(10, 15)) + list(range(20, 25))      # five missing frames
    assert len(build_events(two_tracks(), *synthetic_rows(one_gap, [1.0] * 9), ConflictConfig())) == 1
    assert len(build_events(two_tracks(), *synthetic_rows(five_gap, [1.0] * 10), ConflictConfig())) == 2


def test_overlap_rows_are_not_events() -> None:
    from safetrajai.collision.conflict.events import build_events, overlap_at_start_pairs

    rows = synthetic_rows(range(10, 15), [np.nan] * 5, overlap=[True] * 5)
    assert build_events(two_tracks(), *rows, ConflictConfig()) == ()
    assert overlap_at_start_pairs(rows[0], rows[1]) == 1


def test_event_id_and_order_are_stable() -> None:
    from safetrajai.collision.conflict.events import build_events

    (event,) = build_events(two_tracks(), *synthetic_rows(range(10, 13), [2.0, 0.8, 1.5]), ConflictConfig())
    assert event.event_id == "car:1-scooter:2@10"
    assert event.min_ttc == pytest.approx(0.8) and event.min_ttc_frame == 11


def test_app_config_reads_block() -> None:
    from safetrajai.evaluation.config_loader import AppConfig

    assert AppConfig({}).conflict_analysis.enabled is False
    assert AppConfig({"conflict_analysis": {"enabled": True, "pair_radius_m": 30}}).conflict_analysis.pair_radius_m == 30


# =============================================================================
# 10. Sequence entry and output files
# =============================================================================

OUT_SEQ = "C999001 synthetic"
OUT_FILES = (
    "chains.csv", "conflict_events.csv", "conflict_events.json", "conflict_heatmap.png",
    "pair_frames.csv", "pet.csv", "summary.json",
)


def write_box_track(seq_dir, class_name: str, track_id: int, xs, y: float, size: float = 20.0) -> None:
    rows = [(f, track_id, x - size / 2, y - size, size, size, 0.9) for f, x in enumerate(xs, start=1)]
    path = seq_dir / "trackers" / "mot_challenge" / "CCTV-val" / f"YOLO11x__{class_name}" / "data" / f"{seq_dir.name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.writelines(",".join(repr(float(v)) if isinstance(v, float) else str(v) for v in r) + "\n" for r in rows)


def sequence_setup(tmp_path, camera_yaml: "str | None", enabled: bool = True):
    import types

    import cv2

    seq_dir = tmp_path / "stage1" / OUT_SEQ
    seq_dir.mkdir(parents=True)
    for class_name in ("person", "car", "scooter", "small_truck", "bus", "truck"):
        write_box_track(seq_dir, class_name, 0, [], 0.0)
    m_per_px = 156543.03392 / 2 ** 20 / 2                       # latitude 0, zoom 20, scale 2
    step = 1.0 / 10.0 / m_per_px                                # px per frame at 1 m/s and 10 fps
    write_box_track(seq_dir, "car", 1, [100 + 10 * step * k for k in range(30)], 300.0)
    write_box_track(seq_dir, "car", 2, [100 + 15 / m_per_px + 5 * step * k for k in range(30)], 300.0)
    cv2.imwrite(str(seq_dir / "first_frame_with_box.jpg"), np.zeros((480, 640, 3), np.uint8))

    camera_dir = tmp_path / "homography"
    matrices_dir = tmp_path / "matrices"
    camera_dir.mkdir()
    matrices_dir.mkdir()
    np.save(matrices_dir / "H_T.npy", np.eye(3))
    if camera_yaml is not None:
        (camera_dir / "C999001.yaml").write_text(camera_yaml, encoding="utf-8")
    cfg = types.SimpleNamespace(
        conflict_analysis=ConflictConfig(enabled=enabled),
        homography_camera_dir=camera_dir, homography_matrices_dir=matrices_dir,
        collision_output_dir=str(tmp_path / "collision"),
        benchmark_name="CCTV", split_name="val", tracker_name="YOLO11x",
    )
    return seq_dir, cfg, tmp_path / "collision" / OUT_SEQ / "conflict"


GEOREF_CAMERA = (
    "camera: C999001\nmatrix_tag: T\nsatellite: {provider: google_static, scale: 2, size: [640, 640], zoom: 20}\n"
    "scale_check: passed\nlatlon: [0.0, 120.0]\n"
)


def run_entry(seq_dir, cfg, fps: float = 10.0):
    from safetrajai.collision.conflict import run_conflict_analysis_for_sequence

    return run_conflict_analysis_for_sequence(seq_dir, cfg, fps=fps, bounds=(480, 640))


def test_output_files_and_coordinate_columns(tmp_path) -> None:
    import json

    seq_dir, cfg, out_dir = sequence_setup(tmp_path, GEOREF_CAMERA)
    run_entry(seq_dir, cfg)
    assert tuple(sorted(p.name for p in out_dir.iterdir())) == OUT_FILES
    header = (out_dir / "conflict_events.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    for suffix in ("_m", "_map_px", "_cctv_px"):
        assert f"collision_x{suffix}" in header and f"collision_y{suffix}" in header
    pet_header = (out_dir / "pet.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {"conflict_x_m", "conflict_x_map_px", "conflict_x_cctv_px"} <= set(pet_header)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["skipped_reason"] is None and summary["camera"] == "C999001"
    assert summary["m_per_pixel"] == pytest.approx(156543.03392 / 2 ** 21)
    assert summary["num_events"] >= 1 and summary["acceleration_available"] is True
    events = json.loads((out_dir / "conflict_events.json").read_text(encoding="utf-8"))
    assert len(events) == summary["num_events"] and events[0]["conflict_type"] == "Rear-end"


def test_low_fps_is_recorded_in_limitations(tmp_path) -> None:
    import json

    seq_dir, cfg, out_dir = sequence_setup(tmp_path, GEOREF_CAMERA)
    run_entry(seq_dir, cfg, fps=4.9)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["acceleration_available"] is False
    assert "no_acceleration_low_fps" in {item["code"] for item in summary["limitations"]}


@pytest.mark.parametrize(
    ("camera_yaml", "reason"),
    [
        (None, "no_homography"),
        ("camera: C999001\nmatrix_tag: T\nsatellite: {provider: google_static, scale: 2, size: [640, 640], zoom: 20}\n"
         "scale_check: passed\n", "no_meters_per_pixel"),
    ],
)
def test_unavailable_scale_writes_only_summary(tmp_path, camera_yaml, reason: str) -> None:
    import json

    seq_dir, cfg, out_dir = sequence_setup(tmp_path, camera_yaml)
    run_entry(seq_dir, cfg)
    assert [p.name for p in out_dir.iterdir()] == ["summary.json"]
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["skipped_reason"] == reason


def test_disabled_writes_nothing(tmp_path) -> None:
    seq_dir, cfg, out_dir = sequence_setup(tmp_path, GEOREF_CAMERA, enabled=False)
    assert run_entry(seq_dir, cfg) is None
    assert not out_dir.parent.exists()


def test_two_runs_are_byte_identical(tmp_path) -> None:
    seq_dir, cfg, out_dir = sequence_setup(tmp_path, GEOREF_CAMERA)
    run_entry(seq_dir, cfg)
    first = {p.name: p.read_bytes() for p in out_dir.iterdir()}
    run_entry(seq_dir, cfg)
    assert {p.name: p.read_bytes() for p in out_dir.iterdir()} == first


# =============================================================================
# 11. Golden comparison category
# =============================================================================

def test_golden_compare_puts_conflict_outputs_in_own_category(monkeypatch) -> None:
    import importlib.util
    import sys

    from safetrajai.paths import PROJECT_ROOT

    spec = importlib.util.spec_from_file_location("golden_compare", PROJECT_ROOT / "tests/golden/compare.py")
    compare = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "golden_compare", compare)
    spec.loader.exec_module(compare)
    assert "conflict" in compare.CATEGORY_ORDER
    assert compare.categorize("conflict_analysis/collision_output/C000025 x/conflict/summary.json") == "conflict"
    assert compare.categorize("conflict_analysis/collision_output/C000025 x/ttc_events.csv") == "conflict_events"


def test_golden_compare_treats_equal_infinities_as_identical(monkeypatch) -> None:
    import importlib.util
    import sys

    from safetrajai.paths import PROJECT_ROOT

    spec = importlib.util.spec_from_file_location("golden_compare", PROJECT_ROOT / "tests/golden/compare.py")
    compare = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "golden_compare", compare)
    spec.loader.exec_module(compare)
    assert compare.compare_scalars("inf", "inf").identical
    assert compare.compare_scalars("-inf", "-inf").identical
    assert not compare.compare_scalars("inf", "-inf").identical
    assert not compare.compare_scalars("inf", "1.0").identical
