"""Unit tests for tracking/config.py (spec track-config-loading).

Every test runs inside restoring_config_globals(), because load_track_config() mutates module-level
dicts shared by the whole pytest session.
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from safetrajai.paths import CONFIG_DIR
from safetrajai.tracking import config, create_tracker

DICT_GLOBAL_NAMES = ("CLASS_NAMES", "CLASS_TRACKING_PARAMS", "DEFAULT_TRACKING_PARAMS", "CLASS_CONF_THRESH")


def capture_state() -> dict[str, Any]:
    return copy.deepcopy({
        **{name: getattr(config, name) for name in DICT_GLOBAL_NAMES},
        "default_conf_thresh": config._DEFAULT_CONF_THRESH[0],
        "resolved_path": config.get_track_config_path(),
    })


@contextlib.contextmanager
def restoring_config_globals() -> Iterator[None]:
    saved = capture_state()
    try:
        yield
    finally:
        # Restore in place: other modules hold references to these dict objects.
        for name in DICT_GLOBAL_NAMES:
            target = getattr(config, name)
            target.clear()
            target.update(saved[name])
        config._DEFAULT_CONF_THRESH[0] = saved["default_conf_thresh"]
        config._TRACK_CONFIG_RESOLVED = saved["resolved_path"]


@pytest.fixture(autouse=True)
def isolated_config_globals() -> Iterator[None]:
    with restoring_config_globals():
        yield


def minimal_section() -> dict[str, Any]:
    return {
        "class_names": {0: "person", 1: "car"},
        "class_tracking_params": {
            "person": {"track_buffer": 30, "match_thresh": 0.8, "use_reid": False},
            "car": {"track_buffer": 45, "match_thresh": 0.7, "use_reid": False},
        },
        "default_tracking_params": {"track_buffer": 20, "match_thresh": 0.5},
        "class_conf_thresh": {"person": 0.3, "car": 0.2},
    }


def write_config(directory: Path, data: Any, name: str = "track.yaml") -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_restoring_config_globals_undoes_a_load(tmp_path: Path) -> None:
    before = capture_state()
    other = minimal_section()
    other["class_names"] = {0: "only_class"}
    other["class_tracking_params"] = {"only_class": {"track_buffer": 7}}
    other["class_conf_thresh"] = {"only_class": 0.9}
    with restoring_config_globals():
        config.load_track_config(write_config(tmp_path, other))
        assert config.CLASS_NAMES == {0: "only_class"}
    assert capture_state() == before


def test_restoring_config_globals_keeps_dict_identity(tmp_path: Path) -> None:
    references = [getattr(config, name) for name in DICT_GLOBAL_NAMES]
    with restoring_config_globals():
        config.load_track_config(write_config(tmp_path, minimal_section()))
    assert all(reference is getattr(config, name) for reference, name in zip(references, DICT_GLOBAL_NAMES))


MISSING = object()


def without_resolved_path(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key != "resolved_path"}


@pytest.mark.parametrize(("key", "value"), [
    ("class_names", MISSING),
    ("class_names", {}),
    ("class_tracking_params", MISSING),
    ("class_tracking_params", {}),
    ("class_conf_thresh", MISSING),
    ("class_conf_thresh", {}),
    ("default_tracking_params", MISSING),
    ("default_tracking_params", ["not", "a", "mapping"]),
])
def test_invalid_section_is_rejected_and_names_the_section(tmp_path: Path, key: str, value: Any) -> None:
    section = minimal_section()
    if value is MISSING:
        del section[key]
    else:
        section[key] = value
    before = capture_state()
    with pytest.raises(ValueError, match=key):
        config.load_track_config(write_config(tmp_path, section))
    assert capture_state() == before


@pytest.mark.parametrize("content", [[1, 2], "just a string", None])
def test_non_mapping_root_is_rejected(tmp_path: Path, content: Any) -> None:
    with pytest.raises(ValueError, match="mapping"):
        config.load_track_config(write_config(tmp_path, content))


def test_sectioned_and_flat_formats_are_equivalent(tmp_path: Path) -> None:
    flatPath = write_config(tmp_path, minimal_section(), "flat.yaml")
    sectionedPath = write_config(tmp_path, {"class_tracking": minimal_section()}, "sectioned.yaml")
    config.load_track_config(flatPath)
    flatState = without_resolved_path(capture_state())
    config.load_track_config(sectionedPath)
    sectionedState = without_resolved_path(capture_state())
    assert flatState["CLASS_NAMES"] == {0: "person", 1: "car"}
    assert flatState == sectionedState


def test_non_integer_class_name_key_is_rejected_atomically(tmp_path: Path) -> None:
    section = minimal_section()
    section["class_names"] = {"zero": "person", 1: "car"}
    before = capture_state()
    with pytest.raises(ValueError):
        config.load_track_config(write_config(tmp_path, section))
    assert capture_state() == before


def test_existing_references_see_new_content(tmp_path: Path) -> None:
    reference = config.CLASS_TRACKING_PARAMS
    config.load_track_config(write_config(tmp_path, minimal_section()))
    assert reference is config.CLASS_TRACKING_PARAMS
    assert set(reference) == {"person", "car"}
    assert reference["car"]["track_buffer"] == 45


def test_confidence_thresholds_are_converted_to_float(tmp_path: Path) -> None:
    section = minimal_section()
    section["class_conf_thresh"] = {"person": "0.3", "car": 1}
    config.load_track_config(write_config(tmp_path, section))
    assert config.CLASS_CONF_THRESH == {"person": 0.3, "car": 1.0}
    assert all(type(value) is float for value in config.CLASS_CONF_THRESH.values())


@pytest.mark.parametrize(("extra", "expected"), [({}, 0.15), ({"default_conf_thresh": 0.4}, 0.4)])
def test_default_confidence_threshold(tmp_path: Path, extra: dict[str, Any], expected: float) -> None:
    config.load_track_config(write_config(tmp_path, {**minimal_section(), **extra}))
    assert config.DEFAULT_CONF_THRESH == expected


def test_absolute_path_is_returned_resolved_and_recorded(tmp_path: Path) -> None:
    path = write_config(tmp_path, minimal_section())
    assert config.load_track_config(path) == path.resolve()
    assert config.get_track_config_path() == path.resolve()


def test_relative_path_is_resolved_from_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_config(tmp_path, minimal_section(), "relative.yaml")
    monkeypatch.chdir(tmp_path)
    assert config.load_track_config("relative.yaml") == path.resolve()


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_path_loads_default_scenario(missing: str | None) -> None:
    assert config.load_track_config(missing) == (CONFIG_DIR / "tracking" / "bytetrack.yaml").resolve()


def test_missing_file_raises_and_leaves_globals_unchanged(tmp_path: Path) -> None:
    before = capture_state()
    with pytest.raises(FileNotFoundError):
        config.load_track_config(tmp_path / "does_not_exist.yaml")
    assert capture_state() == before


@pytest.fixture
def minimal_loaded(tmp_path: Path) -> None:
    section = minimal_section()
    section["class_names"] = {0: "person", 1: "car", 2: "bus"}
    config.load_track_config(write_config(tmp_path, section))


@pytest.mark.parametrize("identifier", [1, np.int64(1), 1.0, "car"])
def test_params_lookup_accepts_ids_and_names(minimal_loaded: None, identifier: Any) -> None:
    assert config.get_params_for_class(identifier)["track_buffer"] == 45


@pytest.mark.parametrize("identifier", [99, "unknown", 2])
def test_params_lookup_falls_back_to_defaults(minimal_loaded: None, identifier: Any) -> None:
    # Class id 2 ("bus") is listed in class_names but has no entry in class_tracking_params.
    assert config.get_params_for_class(identifier) == config.DEFAULT_TRACKING_PARAMS


def test_class_name_lookup(minimal_loaded: None) -> None:
    assert config.get_class_name(0) == "person"
    assert config.get_class_name(7) == "class_7"


def test_confidence_threshold_lookup_and_minimum(minimal_loaded: None) -> None:
    assert config.get_conf_thresh_for_class(1) == 0.2
    assert config.get_conf_thresh_for_class("person") == 0.3
    # "bus" has no dedicated threshold, so the default (0.15) applies.
    assert config.get_conf_thresh_for_class(2) == config.DEFAULT_CONF_THRESH == 0.15
    assert config.get_min_conf_thresh() == 0.2


def test_unknown_module_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        config.NO_SUCH_ATTRIBUTE


def test_reload_from_cli_reads_only_track_config(tmp_path: Path) -> None:
    chosen = write_config(tmp_path, minimal_section(), "y.yaml")
    result = config.reload_track_config_from_cli(["--config", "x.yaml", "--track-config", str(chosen), "--foo"])
    assert result == chosen.resolve()
    assert config.CLASS_NAMES == {0: "person", 1: "car"}


def test_reload_from_cli_without_option_uses_default() -> None:
    assert config.reload_track_config_from_cli(["--config", "x.yaml"]) == (CONFIG_DIR / "tracking" / "bytetrack.yaml").resolve()


def three_class_section() -> dict[str, Any]:
    section = minimal_section()
    section["class_names"] = {0: "person", 1: "car", 2: "bus"}
    section["class_tracking_params"]["bus"] = {"track_buffer": 25, "match_thresh": 0.6, "use_reid": False}
    section["class_conf_thresh"]["bus"] = 0.25
    return section


def test_botsort_keeps_the_class_set_it_was_created_with(tmp_path: Path) -> None:
    config.load_track_config(write_config(tmp_path, minimal_section(), "two.yaml"))
    tracker = create_tracker("botsort", frame_rate=10, gmc_method="none")
    config.load_track_config(write_config(tmp_path, three_class_section(), "three.yaml"))
    assert sorted(tracker.CLASS_KEYS) == ["car", "person"]
    rebuilt = create_tracker("botsort", frame_rate=10, gmc_method="none")
    assert sorted(rebuilt.CLASS_KEYS) == ["bus", "car", "person"]


def test_non_mapping_params_for_unlisted_class_leave_globals_unchanged(tmp_path: Path) -> None:
    section = minimal_section()
    section["class_tracking_params"]["bus"] = 5
    before = capture_state()
    with pytest.raises(ValueError):
        config.load_track_config(write_config(tmp_path, section))
    assert capture_state() == before


def test_non_numeric_confidence_threshold_leaves_globals_unchanged(tmp_path: Path) -> None:
    section = minimal_section()
    section["class_conf_thresh"] = {"person": "not a number", "car": 0.2}
    before = capture_state()
    with pytest.raises(ValueError):
        config.load_track_config(write_config(tmp_path, section))
    assert capture_state() == before


@pytest.mark.parametrize("value", [None, [0.3], "not a number"])
def test_non_numeric_default_confidence_threshold_leaves_globals_unchanged(tmp_path: Path, value: Any) -> None:
    section = {**minimal_section(), "default_conf_thresh": value}
    before = capture_state()
    with pytest.raises(ValueError, match="default_conf_thresh"):
        config.load_track_config(write_config(tmp_path, section))
    assert capture_state() == before
