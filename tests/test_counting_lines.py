"""config/counting_lines/<camera>.yaml: loading, validation, and read-back of the web picker output."""

from pathlib import Path

import pytest

from safetrajai.collision.counting.lines import CountingLine, load_counting_lines, resolve_counting_lines
from safetrajai.picker.yaml_output import counting_lines_yaml

VALID = """camera: C000025
image_size: [640, 480]
lines:
  - name: north_gate
    points: [[120, 300], [480, 310]]
    forward_label: southbound
    backward_label: northbound
  - name: side_road
    points: [[10, 20], [30, 40]]
"""
SEQ_NAME = "C000025 中華三路 民生二路(東北角)_20240101"


def write(tmp_path: Path, text: str, camera: str = "C000025") -> Path:
    path = tmp_path / f"{camera}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_lines_are_read_with_labels_only_when_typed(tmp_path: Path) -> None:
    lines = load_counting_lines(write(tmp_path, VALID))

    assert lines.camera == "C000025"
    assert lines.image_size == (640, 480)
    assert lines.lines == (
        CountingLine("north_gate", (120.0, 300.0), (480.0, 310.0), "southbound", "northbound"),
        CountingLine("side_road", (10.0, 20.0), (30.0, 40.0), None, None),
    )


def test_picker_output_is_read_back_unchanged(tmp_path: Path) -> None:
    picked = [
        {"name": "north gate: 北門", "points": [[120, 300], [480, 310]], "forward_label": "往南", "backward_label": "往北"},
        {"name": "side_road", "points": [[10.5, 20], [30, 40]]},
    ]

    loaded = load_counting_lines(write(tmp_path, counting_lines_yaml("C000025", (640, 480), picked)))

    assert [(l.name, l.p1, l.p2) for l in loaded.lines] == [
        ("north gate: 北門", (120.0, 300.0), (480.0, 310.0)), ("side_road", (10.5, 20.0), (30.0, 40.0))]
    assert (loaded.lines[0].forward_label, loaded.lines[0].backward_label) == ("往南", "往北")
    assert (loaded.lines[1].forward_label, loaded.lines[1].backward_label) == (None, None)


@pytest.mark.parametrize("text, message", [
    (VALID.replace("camera: C000025", "camera: C000043"), "camera"),
    (VALID.replace("image_size: [640, 480]", "image_size: [640]"), "image_size"),
    (VALID.replace("[[120, 300], [480, 310]]", "[[120, 300]]"), "points"),
    (VALID.replace("[[120, 300], [480, 310]]", "[[5, 5], [5, 5]]"), "長度"),
    (VALID.replace("[[10, 20], [30, 40]]", "[[10, 20], [a, 40]]"), "points"),
    (VALID.replace("name: side_road", "name: north_gate"), "重複"),
    (VALID.replace("name: side_road", "name: ' '"), "name"),
    (VALID.replace("forward_label: southbound", "forward_label: 5"), "forward_label"),
    (VALID.replace("    forward_label: southbound\n", "    forwad_label: southbound\n"), "forwad_label"),
    (VALID + "extra: 1\n", "extra"),
    ("camera: C000025\nimage_size: [640, 480]\nlines: []\n", "lines"),
    ("camera: C000025\nimage_size: [640, 480]\n", "lines"),
])
def test_invalid_files_are_rejected_naming_the_problem(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_counting_lines(write(tmp_path, text))


def test_a_sequence_uses_the_file_of_its_camera(tmp_path: Path) -> None:
    write(tmp_path, VALID)

    resolved = resolve_counting_lines(SEQ_NAME, (640, 480), tmp_path)

    assert [line.name for line in resolved] == ["north_gate", "side_road"]


def test_a_camera_without_a_file_has_no_lines(tmp_path: Path) -> None:
    assert resolve_counting_lines(SEQ_NAME, (640, 480), tmp_path) is None


def test_image_size_mismatch_is_an_error_not_a_rescale(tmp_path: Path) -> None:
    write(tmp_path, VALID)

    with pytest.raises(ValueError, match="image_size"):
        resolve_counting_lines(SEQ_NAME, (1280, 720), tmp_path)


def test_counting_block_defaults_to_disabled_and_is_read_by_app_config() -> None:
    from safetrajai.evaluation.config_loader import AppConfig

    assert AppConfig({}).counting.enabled is False
    assert AppConfig({"counting": {}}).counting.enabled is False
    assert AppConfig({"counting": {"enabled": True}}).counting.enabled is True


@pytest.mark.parametrize("block, message", [
    ({"enabled": "yes"}, "enabled"),
    ({"enabled": True, "threshold": 1}, "threshold"),
    ("on", "counting"),
])
def test_bad_counting_blocks_are_rejected(block, message: str) -> None:
    from safetrajai.collision.counting.config import load_counting_config

    with pytest.raises(ValueError, match=message):
        load_counting_config(block)


def test_existing_scenario_configs_keep_counting_disabled() -> None:
    from safetrajai.evaluation.config_loader import load_config
    from safetrajai.paths import CONFIG_DIR

    for path in sorted((CONFIG_DIR / "collision").glob("*.yaml")):
        assert load_config(str(path)).counting.enabled is False, path
