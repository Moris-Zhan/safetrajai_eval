"""Frame-based retention behavior and atomic config validation (ADR 0009)."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from ultralytics.trackers.basetrack import TrackState

from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.tracking import config, create_tracker


@pytest.fixture
def restore_config():
    path = config.get_track_config_path()
    yield
    config.load_track_config(path)


@pytest.mark.parametrize("tracker_type", ["botsort", "bytetrack"])
@pytest.mark.parametrize("fps", [5, 10])
@pytest.mark.parametrize("buffer", [15, 30])
@pytest.mark.parametrize("reappear", [False, True])
def test_retention_boundary(monkeypatch, tracker_type, fps, buffer, reappear):
    params = {**config.get_params_for_class(1), "track_buffer": buffer, "use_reid": False}
    monkeypatch.setitem(config.CLASS_TRACKING_PARAMS, "car", params)
    # Disable all encoders; this test runs real updates without model or video I/O.
    for name, values in list(config.CLASS_TRACKING_PARAMS.items()):
        monkeypatch.setitem(config.CLASS_TRACKING_PARAMS, name, {**values, "use_reid": False})
    tracker = create_tracker(tracker_type, frame_rate=fps, gmc_method="none")
    detection = SimpleNamespace(conf=np.array([0.99]), xywh=np.array([[100., 100., 40., 40.]]), cls=np.array([1.]))
    empty = SimpleNamespace(conf=np.empty(0), xywh=np.empty((0, 4)), cls=np.empty(0))
    tracker.update(detection)
    [track] = tracker.tracked_stracks
    original_id = track.track_id
    for gap in range(1, buffer + 1):
        if reappear and gap == buffer:
            tracker.update(detection)
            assert track.state == TrackState.Tracked
            assert track.track_id == original_id and track in tracker.tracked_stracks
            return
        tracker.update(empty)
        assert track.state == TrackState.Lost
        assert track in tracker.lost_stracks
        assert track not in tracker.removed_stracks
    tracker.update(empty)
    assert track.state == TrackState.Removed
    assert track in tracker.removed_stracks


def config_state():
    return copy.deepcopy((config.CLASS_NAMES, config.CLASS_TRACKING_PARAMS, config.DEFAULT_TRACKING_PARAMS,
                          config.CLASS_CONF_THRESH, config.DEFAULT_CONF_THRESH, config.get_track_config_path()))


@pytest.mark.parametrize("value", [None, 0, -1, 2.5, 90.0, True, "90"])
@pytest.mark.parametrize("use_default", [False, True])
def test_invalid_buffer_rejected_atomically(tmp_path, restore_config, value, use_default):
    data = yaml.safe_load((CONFIG_DIR / "botsort_noreid.yaml").read_text())
    section = data["class_tracking"]
    # Keep unrelated classes valid while exercising one invalid effective value.
    for params in section["class_tracking_params"].values():
        params["track_buffer"] = 90
    if use_default:
        del section["class_tracking_params"]["car"]
        params = section["default_tracking_params"]
    else:
        params = section["class_tracking_params"]["car"]
        section["default_tracking_params"]["track_buffer"] = 120
    if value is None:
        params.pop("track_buffer", None)
    else:
        params["track_buffer"] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data))
    before = config_state()
    refs = (config.CLASS_NAMES, config.CLASS_TRACKING_PARAMS, config.DEFAULT_TRACKING_PARAMS, config.CLASS_CONF_THRESH)
    with pytest.raises(ValueError, match="car.*track_buffer"):
        config.load_track_config(path)
    assert config_state() == before
    assert all(a is b for a, b in zip(refs, (config.CLASS_NAMES, config.CLASS_TRACKING_PARAMS,
                                             config.DEFAULT_TRACKING_PARAMS, config.CLASS_CONF_THRESH)))


def test_missing_class_uses_default_buffer(tmp_path, restore_config):
    data = yaml.safe_load((CONFIG_DIR / "botsort_noreid.yaml").read_text())
    section = data["class_tracking"]
    for params in section["class_tracking_params"].values():
        params["track_buffer"] = 90
    del section["class_tracking_params"]["car"]
    section["default_tracking_params"]["track_buffer"] = 17
    path = tmp_path / "default.yaml"
    path.write_text(yaml.safe_dump(data))
    config.load_track_config(path)
    assert config.get_params_for_class(1)["track_buffer"] == 17


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")) + sorted((PROJECT_ROOT / "tests/golden/configs").glob("*.yaml")))
def test_scenario_buffers_are_positive_integer_frames(path, restore_config):
    if "class_tracking" not in (yaml.safe_load(path.read_text()) or {}):
        pytest.skip("not a tracking scenario")
    config.load_track_config(path)
    for cls_id in config.CLASS_NAMES:
        value = config.get_params_for_class(cls_id)["track_buffer"]
        assert type(value) is int and value > 0
