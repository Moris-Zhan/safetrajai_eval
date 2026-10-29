"""Web picker page: rendered HTML and the JavaScript it embeds (executed with Node)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from safetrajai.picker.page import LOGIC_BEGIN, LOGIC_END, MODES, PickerImage, render_picker_page
from safetrajai.picker.yaml_output import (
    camera_position_yaml,
    checkpoints_yaml,
    counting_lines_yaml,
    homography_yaml,
)

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

CCTV_IMAGE = PickerImage(name="cctv.mp4", width=640, height=480, jpeg_b64="Q0NUVg==")
SATELLITE_IMAGE = PickerImage(name="sat.png", width=1280, height=1280, jpeg_b64="U0FU")
IMAGES_BY_MODE = {
    "homography": {"cctv": CCTV_IMAGE, "satellite": SATELLITE_IMAGE},
    "checkpoints": {"cctv": CCTV_IMAGE, "satellite": SATELLITE_IMAGE},
    "counting-lines": {"cctv": CCTV_IMAGE},
    "camera-position": {"satellite": SATELLITE_IMAGE},
}


def page_config(page: str) -> dict:
    match = re.search(r"^const CONFIG = (.*);$", page, re.MULTILINE)
    assert match, "CONFIG line not found"
    return json.loads(match.group(1))


def inline_scripts(page: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", page, re.DOTALL)


def logic_section(page: str) -> str:
    start = page.index(LOGIC_BEGIN)
    return page[start : page.index(LOGIC_END, start)]


def run_logic(tmp_path: Path, name: str, *args):
    """Call one function of the page's pure-logic section in Node and return its JSON result."""
    page = render_picker_page("homography", "C000054", IMAGES_BY_MODE["homography"], [])
    harness = (
        logic_section(page)
        + "\nconst fns = {homographyYaml, checkpointsYaml, cameraPositionYaml, countingLinesYaml, nextSide,"
        " addPairedPoint, undoPoint, splitPairs, buildLines, readiness};\n"
        "const [name, args] = JSON.parse(process.argv[2]);\n"
        "process.stdout.write(JSON.stringify(fns[name](...args)));\n"
    )
    script = tmp_path / "logic.js"
    script.write_text(harness, encoding="utf-8")
    result = subprocess.run([NODE, str(script), json.dumps([name, list(args)])], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_renders_its_config_and_images(mode: str) -> None:
    page = render_picker_page(mode, "C000054", IMAGES_BY_MODE[mode], ["note one"])

    config = page_config(page)
    assert config["mode"] == mode
    assert config["camera"] == "C000054"
    assert config["notes"] == ["note one"]
    assert set(config["images"]) == set(IMAGES_BY_MODE[mode])
    for key, image in IMAGES_BY_MODE[mode].items():
        assert config["images"][key] == {"name": image.name, "width": image.width, "height": image.height,
                                         "b64": image.jpeg_b64}


def test_unknown_mode_and_missing_images_are_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        render_picker_page("roi", "C000054", IMAGES_BY_MODE["homography"], [])
    with pytest.raises(ValueError, match="satellite"):
        render_picker_page("homography", "C000054", {"cctv": CCTV_IMAGE}, [])
    with pytest.raises(ValueError, match="cctv"):
        render_picker_page("counting-lines", "C000054", {"satellite": SATELLITE_IMAGE}, [])


def test_text_is_escaped_in_html_and_in_the_embedded_config() -> None:
    nasty = "</script><b>x</b>"
    page = render_picker_page("camera-position", f"C{nasty}", IMAGES_BY_MODE["camera-position"], [nasty])

    assert nasty not in page
    assert page_config(page)["notes"] == [nasty]
    assert "<title>" in page and "&lt;" in page


@needs_node
def test_embedded_scripts_have_valid_syntax(tmp_path: Path) -> None:
    page = render_picker_page("homography", "C000054", IMAGES_BY_MODE["homography"], [])
    for index, script in enumerate(inline_scripts(page)):
        path = tmp_path / f"script{index}.js"
        path.write_text(script, encoding="utf-8")
        result = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr


CCTV_POINTS = [[100, 200], [500, 210], [480, 400], [120, 380]]
SAT_POINTS = [[300, 310], [700, 305], [690, 600], [310, 590]]
LINES = [
    {"name": "north gate: 北門", "points": [[120, 300], [480, 310]], "forward_label": "southbound",
     "backward_label": "northbound"},
    {"name": "side_road", "points": [[10, 20], [30, 40]]},
]


@needs_node
def test_javascript_yaml_matches_python_yaml_exactly(tmp_path: Path) -> None:
    assert run_logic(tmp_path, "homographyYaml", "C000054", CCTV_POINTS, SAT_POINTS) == homography_yaml(
        "C000054", CCTV_POINTS, SAT_POINTS)
    assert run_logic(tmp_path, "checkpointsYaml", "C000054", CCTV_POINTS, SAT_POINTS) == checkpoints_yaml(
        "C000054", CCTV_POINTS, SAT_POINTS)
    assert run_logic(tmp_path, "cameraPositionYaml", "C000054", [412, 233]) == camera_position_yaml("C000054", [412, 233])
    assert run_logic(tmp_path, "countingLinesYaml", "C000025", [640, 480], LINES) == counting_lines_yaml(
        "C000025", (640, 480), LINES)


@needs_node
def test_points_must_be_clicked_left_then_right(tmp_path: Path) -> None:
    assert run_logic(tmp_path, "nextSide", []) == "first"
    assert run_logic(tmp_path, "nextSide", [[1, 1]]) == "second"
    assert run_logic(tmp_path, "addPairedPoint", [], "second", [5, 5]) == []          # wrong side is ignored
    assert run_logic(tmp_path, "addPairedPoint", [], "first", [5, 5]) == [[5, 5]]
    assert run_logic(tmp_path, "addPairedPoint", [[5, 5]], "first", [6, 6]) == [[5, 5]]


@needs_node
def test_undo_removes_the_last_point_and_pairs_are_split_by_side(tmp_path: Path) -> None:
    stack = [[1, 1], [2, 2], [3, 3]]

    assert run_logic(tmp_path, "undoPoint", stack) == [[1, 1], [2, 2]]
    assert run_logic(tmp_path, "undoPoint", []) == []
    assert run_logic(tmp_path, "splitPairs", stack) == {"first": [[1, 1]], "second": [[2, 2]], "pending": [3, 3]}
    assert run_logic(tmp_path, "splitPairs", stack[:2])["pending"] is None


@needs_node
def test_lines_are_built_from_consecutive_point_pairs(tmp_path: Path) -> None:
    stack = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 9]]
    meta = [{"name": "first", "forward": "in", "backward": "out"}, {}]

    lines = run_logic(tmp_path, "buildLines", stack, meta)

    assert lines == [
        {"name": "first", "points": [[1, 2], [3, 4]], "forward_label": "in", "backward_label": "out"},
        {"name": "line_2", "points": [[5, 6], [7, 8]]},
    ]


@needs_node
@pytest.mark.parametrize("mode, state, ok, has_warning", [
    ("homography", {"stack": [[1, 1], [2, 2]] * 4}, True, False),     # four pairs
    ("homography", {"stack": [[1, 1], [2, 2]] * 3}, False, False),    # three pairs
    ("homography", {"stack": [[1, 1], [2, 2]] * 3 + [[9, 9]]}, False, False),   # a pending point does not count
    ("checkpoints", {"stack": []}, False, False),
    ("checkpoints", {"stack": [[1, 1], [2, 2]]}, True, True),
    ("checkpoints", {"stack": [[1, 1], [2, 2]] * 4}, True, False),
    ("camera-position", {"point": None}, False, False),
    ("camera-position", {"point": [3, 4]}, True, False),
    ("counting-lines", {"stack": [[1, 2], [3, 4]], "meta": [{"name": " "}]}, False, False),
    ("counting-lines", {"stack": [[1, 2], [3, 4], [5, 6], [7, 8]], "meta": [{"name": "a"}, {"name": "a"}]}, False, False),
    ("counting-lines", {"stack": [[1, 2], [3, 4]], "meta": [{"name": "a"}]}, True, False),
    ("counting-lines", {"stack": [[1, 2]], "meta": []}, False, False),
])
def test_readiness_rules(tmp_path: Path, mode: str, state: dict, ok: bool, has_warning: bool) -> None:
    result = run_logic(tmp_path, "readiness", mode, state)

    assert result["ok"] is ok
    assert bool(result.get("warning")) is has_warning
    assert ok or result["message"]
