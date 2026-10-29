"""YAML text produced from the points clicked in the web picker, read back by the existing loaders."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from safetrajai.collision.homography.camera_config import load_camera_config, update_camera_config_fields
from safetrajai.collision.homography.validation import load_checkpoints
from safetrajai.picker.yaml_output import (
    camera_position_yaml,
    checkpoints_yaml,
    counting_lines_yaml,
    homography_yaml,
)

CCTV = [[100, 200], [500, 210], [480, 400], [120, 380]]
SAT = [[300.5, 310], [700, 305.25], [690, 600], [310, 590]]


def load_homography_fragment(tmp_path: Path, fragment: str):
    path = tmp_path / "C000054.yaml"
    path.write_text("camera: C000054\n" + fragment, encoding="utf-8")
    return load_camera_config(path)


def test_homography_points_are_read_back_unchanged(tmp_path: Path) -> None:
    cfg = load_homography_fragment(tmp_path, homography_yaml("C000054", CCTV, SAT))

    assert [list(p) for p in cfg.cctv_points] == [[float(x), float(y)] for x, y in CCTV]
    assert [list(p) for p in cfg.map_points] == [[float(x), float(y)] for x, y in SAT]


def test_homography_fragment_keeps_every_top_level_value_on_one_line(tmp_path: Path) -> None:
    fragment = homography_yaml("C000054", CCTV, SAT)
    path = tmp_path / "C000054.yaml"
    path.write_text("camera: C000054\n" + fragment, encoding="utf-8")

    update_camera_config_fields(path, {"cctv_points": CCTV[::-1]})

    assert [list(p) for p in load_camera_config(path).cctv_points] == [[float(x), float(y)] for x, y in CCTV[::-1]]


@pytest.mark.parametrize("cctv, sat", [
    (CCTV[:3], SAT[:3]),            # fewer than four pairs cannot define a homography
    (CCTV, SAT[:3]),                # different counts
    ([[1, 2, 3]] * 4, SAT),         # wrong point shape
    ([["a", "b"]] * 4, SAT),        # not numbers
])
def test_homography_rejects_unusable_points(cctv, sat) -> None:
    with pytest.raises(ValueError):
        homography_yaml("C000054", cctv, sat)


def test_checkpoints_are_read_back_by_the_validation_loader(tmp_path: Path) -> None:
    (tmp_path / "C000054.yaml").write_text(checkpoints_yaml("C000054", CCTV, SAT), encoding="utf-8")

    cctv, sat = load_checkpoints(tmp_path, "C000054")

    np.testing.assert_array_equal(cctv, np.array(CCTV, dtype=float))
    np.testing.assert_array_equal(sat, np.array(SAT, dtype=float))


def test_checkpoints_use_the_existing_sat_points_key() -> None:
    raw = yaml.safe_load(checkpoints_yaml("C000054", CCTV, SAT))

    assert set(raw) == {"cctv_points", "sat_points"}


@pytest.mark.parametrize("cctv, sat", [([], []), (CCTV, SAT[:2]), ([[1, 2, 3]], [[1, 2]])])
def test_checkpoints_reject_unusable_points(cctv, sat) -> None:
    with pytest.raises(ValueError):
        checkpoints_yaml("C000054", cctv, sat)


def test_camera_position_is_read_back(tmp_path: Path) -> None:
    cfg = load_homography_fragment(tmp_path, camera_position_yaml("C000054", [412.5, 233]))

    assert cfg.camera_view_point == (412.5, 233.0)


@pytest.mark.parametrize("point", [[1], [1, 2, 3], ["a", 2], None])
def test_camera_position_rejects_a_bad_point(point) -> None:
    with pytest.raises(ValueError):
        camera_position_yaml("C000054", point)


LINES = [
    {"name": "north_gate", "points": [[120, 300], [480, 310]], "forward_label": "southbound", "backward_label": "northbound"},
    {"name": "side_road", "points": [[10, 20], [30, 40]]},
]


def test_counting_lines_keep_the_documented_structure() -> None:
    raw = yaml.safe_load(counting_lines_yaml("C000025", (640, 480), LINES))

    assert raw == {
        "camera": "C000025",
        "image_size": [640, 480],
        "lines": [
            {"name": "north_gate", "points": [[120, 300], [480, 310]],
             "forward_label": "southbound", "backward_label": "northbound"},
            {"name": "side_road", "points": [[10, 20], [30, 40]]},
        ],
    }


def test_counting_line_text_values_are_json_quoted_and_empty_labels_are_dropped() -> None:
    text = counting_lines_yaml("C000025", (640, 480), [
        {"name": "north gate: 北門", "points": [[1, 2], [3, 4]], "forward_label": "", "backward_label": "a\"b"},
    ])

    assert '  - name: "north gate: 北門"' in text
    assert '    backward_label: "a\\"b"' in text
    assert "forward_label" not in text
    assert yaml.safe_load(text)["lines"][0] == {"name": "north gate: 北門", "points": [[1, 2], [3, 4]], "backward_label": 'a"b'}


@pytest.mark.parametrize("lines", [
    [],                                                                           # no lines
    [{"name": "a", "points": [[1, 2]]}],                                          # one point
    [{"name": "a", "points": [[1, 2], [3, 4], [5, 6]]}],                          # three points
    [{"name": "a", "points": [[5, 5], [5, 5]]}],                                  # zero length
    [{"name": "", "points": [[1, 2], [3, 4]]}],                                   # empty name
    [{"name": "a", "points": [[1, 2], [3, 4]]}, {"name": "a", "points": [[5, 6], [7, 8]]}],   # duplicate name
    [{"name": "a", "points": [[1, 2], [3, 4]], "forward_label": 5}],              # label is not text
])
def test_counting_lines_reject_unusable_lines(lines) -> None:
    with pytest.raises(ValueError):
        counting_lines_yaml("C000025", (640, 480), lines)
