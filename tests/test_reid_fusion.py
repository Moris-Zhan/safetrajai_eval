"""Scenarios from specs/reid-fusion: appearance fusion in ClassAwareBOTSORT.get_dists_class_aware."""

import numpy as np
import pytest
from ultralytics.trackers.bot_sort import BOTrack

from safetrajai.tracking import class_aware_botsort, matching
from safetrajai.tracking.class_aware_botsort import ClassAwareBOTSORT

CAR = 1
BASE_PARAMS = {"appearance_thresh": 0.1, "proximity_thresh": 0.5, "use_reid": True,
               "center_dist_weight": 0.0, "norm_factor": 80.0, "fuse_score": False}


def make_tracker(frame_id: int = 20, frame_rate: int = 5) -> ClassAwareBOTSORT:
    tracker = ClassAwareBOTSORT.__new__(ClassAwareBOTSORT)
    tracker.appearance_thresh, tracker.proximity_thresh = 0.4, 0.5
    tracker.cls_encoder = {CAR: object()}
    tracker.frame_id, tracker.frame_rate = frame_id, frame_rate
    return tracker


def make_box(cx: float, cy: float, w: float, h: float, feat: np.ndarray, score: float = 0.9) -> BOTrack:
    return BOTrack(np.array([cx, cy, w, h, 0.0]), score, CAR, feat)


def make_track(cx, cy, w, h, feat, frame_id, lost=False, velocity=(0.0, 0.0)) -> BOTrack:
    track = make_box(cx, cy, w, h, feat)
    track.activate(BOTrack.shared_kalman, frame_id)
    track.mean[4:6] = velocity
    if lost:
        track.mark_lost()
    BOTrack.multi_predict([track])
    return track


def unit(rng: np.random.Generator, dim: int = 16) -> np.ndarray:
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def random_scene(seed: int):
    rng = np.random.default_rng(seed)
    tracks = [make_track(*rng.uniform(50, 400, 2), *rng.uniform(20, 80, 2), unit(rng), frame_id=int(rng.integers(10, 20)),
                         lost=bool(i % 3 == 0), velocity=rng.uniform(-8, 8, 2)) for i in range(7)]
    dets = [make_box(*rng.uniform(50, 400, 2), *rng.uniform(20, 80, 2), unit(rng), score=float(rng.uniform(0.2, 1)))
            for _ in range(6)]
    for i, t in enumerate(tracks[:3]):
        x, y, w, h = t.xywh
        dets[i] = make_box(x + rng.uniform(-3, 3), y + rng.uniform(-3, 3), w, h, t.smooth_feat + 0.1 * unit(rng))
    return tracks, dets


def legacy_dists(tracker: ClassAwareBOTSORT, tracks, detections, params: dict) -> np.ndarray:
    """Verbatim algorithm of get_dists_class_aware before change reid-center-gate-fusion."""
    appearance_thresh = params.get("appearance_thresh", tracker.appearance_thresh)
    proximity_thresh = params.get("proximity_thresh", tracker.proximity_thresh)
    dists = matching.iou_distance(tracks, detections)
    dists_mask = dists > (1 - proximity_thresh)
    center_w = params.get("center_dist_weight", 0.0)
    if center_w > 0:
        center_dists = tracker._center_distance(tracks, detections, params.get("norm_factor", 200.0))
        dists = (1 - center_w) * dists + center_w * center_dists
    if params.get("fuse_score", False):
        dists = matching.fuse_score(dists, detections)
    if params.get("use_reid", False) and tracker.cls_encoder.get(CAR) is not None:
        emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        emb_dists[emb_dists > (1 - appearance_thresh)] = 1.0
        emb_dists[dists_mask] = 1.0
        dists = np.minimum(dists, emb_dists)
    return dists


@pytest.fixture
def params(monkeypatch):
    current = dict(BASE_PARAMS)
    monkeypatch.setattr(class_aware_botsort, "get_params_for_class", lambda cls_id: current)
    return current


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("overrides", [
    {}, {"use_reid": False}, {"center_dist_weight": 0.6}, {"fuse_score": True},
    {"center_dist_weight": 0.4, "fuse_score": True, "appearance_thresh": 0.8, "proximity_thresh": 0.1},
])
def test_unset_new_params_is_bitwise_unchanged(params, seed, overrides) -> None:
    params.update(overrides)
    tracker = make_tracker()
    tracks, dets = random_scene(seed)
    expected = legacy_dists(tracker, tracks, dets, params)
    assert np.array_equal(tracker.get_dists_class_aware(tracks, dets, CAR), expected)


def test_regression_scene_exercises_appearance(params) -> None:
    tracker = make_tracker()
    tracks, dets = random_scene(0)
    with_reid = legacy_dists(tracker, tracks, dets, params)
    without_reid = legacy_dists(tracker, tracks, dets, {**params, "use_reid": False})
    assert (with_reid < without_reid).any()


def fuse_one(motion=0.9, appearance=0.1, iou_far=True, lost=True, center=60.0, radius=80.0,
             appearance_thresh=0.8, veto_sim=None, gate=True) -> float:
    kwargs = {"is_lost": np.array([lost]), "center_px": np.array([[center]]), "gate_radius": np.array([radius])} if gate else {}
    fused = matching.fuse_appearance(np.array([[motion]]), np.array([[appearance]]), np.array([[iou_far]]),
                                     appearance_thresh, veto_sim=veto_sim, **kwargs)
    return float(fused[0, 0])


def test_lost_track_inside_gate_uses_appearance_even_without_iou() -> None:
    assert fuse_one() == pytest.approx(0.1)


def test_outside_gate_and_iou_uses_motion_only() -> None:
    assert fuse_one(center=120.0) == pytest.approx(0.9)


def test_iou_overlap_outside_gate_still_uses_appearance() -> None:
    assert fuse_one(center=500.0, iou_far=False) == pytest.approx(0.1)


def test_widened_gate_radius_is_applied() -> None:
    assert fuse_one(center=120.0, radius=80.0 + 40.0 * 1.5) == pytest.approx(0.1)


def test_non_lost_track_ignores_gate() -> None:
    assert fuse_one(lost=False, center=40.0, appearance=0.05) == pytest.approx(0.9)


def test_dissimilar_appearance_inside_gate_uses_motion() -> None:
    assert fuse_one(appearance=0.3, appearance_thresh=0.8) == pytest.approx(0.9)


def test_veto_rejects_dissimilar_pair_regardless_of_motion() -> None:
    assert fuse_one(motion=0.05, appearance=0.5, iou_far=False, lost=False, veto_sim=0.6) == 1.0


def test_no_veto_keeps_motion_pair() -> None:
    assert fuse_one(motion=0.05, appearance=0.5, iou_far=False, lost=False) == pytest.approx(0.05)


def test_veto_only_without_gate_matches_legacy_fusion() -> None:
    rng = np.random.default_rng(0)
    motion, appearance = rng.uniform(0, 1, (5, 4)), rng.uniform(0, 1, (5, 4))
    mask = rng.uniform(0, 1, (5, 4)) > 0.5
    legacy = appearance.copy()
    legacy[legacy > 0.2] = 1.0
    legacy[mask] = 1.0
    expected = np.minimum(motion, legacy)
    expected[appearance > 0.3] = 1.0
    assert np.array_equal(matching.fuse_appearance(motion, appearance, mask, 0.8, veto_sim=0.7), expected)


def stopped_then_lost_scene(params: dict, record_last: bool = True, det_x: float = 145.0, vx: float = 30.0,
                            lost_frames: int = 5, frame_rate: int = 5):
    """A car tracked at (100,100), lost, then extrapolated by vx px/frame; a same-looking detection at (det_x,100)."""
    feat = unit(np.random.default_rng(1))
    tracker = make_tracker(frame_id=10, frame_rate=frame_rate)
    track = make_track(100, 100, 40, 40, feat, frame_id=10)
    det = make_box(det_x, 100, 40, 40, feat)
    if record_last:
        tracker.get_dists_class_aware([track], [det], CAR)
        assert np.allclose(track._reid_last_center, track.xywh[:2])
    track.mark_lost()
    track.mean[4] = vx
    for _ in range(lost_frames):
        BOTrack.multi_predict([track])
    tracker.frame_id = 10 + lost_frames
    return tracker, track, det


def test_tracked_track_records_last_center(params) -> None:
    params.update(reid_center_gate_px=80.0)
    stopped_then_lost_scene(params)


def test_lost_track_uses_last_center_when_prediction_drifted(params) -> None:
    params.update(reid_center_gate_px=80.0, appearance_thresh=0.8)
    tracker, track, det = stopped_then_lost_scene(params)
    assert track.xywh[0] > 200
    assert tracker.get_dists_class_aware([track], [det], CAR)[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_lost_track_without_record_uses_predicted_center_only(params) -> None:
    params.update(reid_center_gate_px=80.0, appearance_thresh=0.8)
    tracker, track, det = stopped_then_lost_scene(params, record_last=False)
    assert tracker.get_dists_class_aware([track], [det], CAR)[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("frame_rate", [5, 10])
def test_gate_widens_with_lost_frames(params, frame_rate) -> None:
    params.update(reid_center_gate_px=80.0, reid_center_gate_px_per_frame=8.0, appearance_thresh=0.8)
    # prediction drifts away (vx < 0); the last center is 120 px from the detection
    tracker, track, det = stopped_then_lost_scene(params, det_x=220.0, vx=-30.0, lost_frames=7, frame_rate=frame_rate)
    # lost 7 frames -> radius 80 + 8 * 7 = 136 px >= 120 px, regardless of frame rate
    assert tracker.get_dists_class_aware([track], [det], CAR)[0, 0] == pytest.approx(0.0, abs=1e-6)
    params.update(reid_center_gate_px_per_frame=5.0)  # 80 + 5 * 7 = 115 px < 120 px
    assert tracker.get_dists_class_aware([track], [det], CAR)[0, 0] == pytest.approx(1.0)


def test_use_reid_false_ignores_new_params(params) -> None:
    params.update(use_reid=False, reid_center_gate_px=500.0, appearance_veto_sim=0.9)
    tracker = make_tracker()
    tracks, dets = random_scene(3)
    expected = legacy_dists(tracker, tracks, dets, params)
    assert np.array_equal(tracker.get_dists_class_aware(tracks, dets, CAR), expected)


@pytest.mark.parametrize(("key", "value"), [
    ("appearance_veto_sim", 1.5), ("appearance_veto_sim", -0.1), ("reid_center_gate_px_per_frame", -1.0),
    ("reid_center_gate_px", 0), ("reid_center_gate_px", "80"),
])
def test_invalid_fusion_params_rejected_before_tracking(tmp_path, key, value) -> None:
    import yaml

    from safetrajai.paths import CONFIG_DIR
    from safetrajai.tracking import config as tracking_config

    cfg = yaml.safe_load((CONFIG_DIR / "reid" / "reid_crops.yaml").read_text(encoding="utf-8"))
    cfg["class_tracking"]["class_tracking_params"]["scooter"][key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    before = dict(tracking_config.CLASS_TRACKING_PARAMS)
    with pytest.raises(ValueError, match=rf"scooter.*{key}"):
        tracking_config.load_track_config(path)
    assert tracking_config.CLASS_TRACKING_PARAMS == before


def test_valid_fusion_params_load(tmp_path) -> None:
    import yaml

    from safetrajai.paths import CONFIG_DIR
    from safetrajai.tracking import config as tracking_config

    cfg = yaml.safe_load((CONFIG_DIR / "reid" / "reid_crops.yaml").read_text(encoding="utf-8"))
    cfg["class_tracking"]["class_tracking_params"]["car"].update(
        reid_center_gate_px=80, reid_center_gate_px_per_frame=0, appearance_veto_sim=0.6)
    path = tmp_path / "ok.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    tracking_config.load_track_config(path)
    assert tracking_config.get_params_for_class(CAR)["reid_center_gate_px"] == 80


def test_legacy_per_second_gate_param_rejected(tmp_path) -> None:
    import yaml

    from safetrajai.paths import CONFIG_DIR
    from safetrajai.tracking import config as tracking_config

    cfg = yaml.safe_load((CONFIG_DIR / "reid" / "reid_crops.yaml").read_text(encoding="utf-8"))
    cfg["class_tracking"]["class_tracking_params"]["car"]["reid_center_gate_px_per_s"] = 40
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    before = dict(tracking_config.CLASS_TRACKING_PARAMS)
    with pytest.raises(ValueError, match=r"car.*reid_center_gate_px_per_s.*reid_center_gate_px_per_frame"):
        tracking_config.load_track_config(path)
    assert tracking_config.CLASS_TRACKING_PARAMS == before
