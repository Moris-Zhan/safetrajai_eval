"""GT overlay drawing shared by the two visualize_gt_* scripts."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).parent / "fixtures" / "gt_overlay" / "draw_gt_on_frame.json"
SCRIPTS = ("visualize_gt_on_video", "visualize_gt_on_class_video")
VARIANTS = {
    "class": {},
    "by_id": {"color_by": "id"},
    "alpha": {"alpha": 0.4},
    "labels_off": {"show_id": False, "show_class": False},
    "conf": {"show_conf": True},
}


def synthetic_frame() -> np.ndarray:
    return (np.random.RandomState(1).rand(120, 160, 3) * 255).astype(np.uint8)


def synthetic_annotations() -> list[dict]:
    """Known and unknown class ids, boxes partly outside the frame, and ids beyond the palette size."""
    boxes = [(10, 10, 30, 40), (-5, 20, 30, 40), (130, 60, 50, 80), (60, 0, 20, 20), (80, 70, 25, 30)]
    class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 9, -1]
    return [
        {"id": 7 * index + 3, "x": boxes[index % 5][0] + index, "y": boxes[index % 5][1], "w": boxes[index % 5][2],
         "h": boxes[index % 5][3], "conf": 0.5 + index / 100, "class_id": class_id, "class_name": f"c{class_id}"}
        for index, class_id in enumerate(class_ids)
    ]


def frame_digest(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


def draw_digests(draw) -> dict[str, str]:
    return {
        name: frame_digest(draw(synthetic_frame(), synthetic_annotations(), **kwargs))
        for name, kwargs in VARIANTS.items()
    }


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "visualization" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_draw_matches_the_snapshot_taken_from_the_scripts() -> None:
    from safetrajai.visualization.gt_overlay import draw_gt_on_frame

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert draw_digests(draw_gt_on_frame) == expected


def test_color_for_id_cycles_through_the_palette() -> None:
    from safetrajai.visualization.gt_overlay import color_for_id

    assert color_for_id(0) == (255, 56, 56)
    assert color_for_id(20) == color_for_id(0)
    assert color_for_id(-1) == (255, 55, 199)


def test_class_colors_keep_the_current_mapping() -> None:
    from safetrajai.visualization.gt_overlay import CLASS_COLORS

    assert CLASS_COLORS == {
        0: (255, 200, 50), 1: (50, 200, 255), 2: (50, 255, 130), 3: (255, 80, 200), 4: (80, 130, 255),
        5: (255, 150, 30), 6: (255, 50, 50), 7: (150, 255, 50), -1: (200, 200, 200),
    }


def test_parse_seqinfo_reads_the_sequence_section(tmp_path: Path) -> None:
    from safetrajai.visualization.gt_overlay import parse_seqinfo

    ini = tmp_path / "seqinfo.ini"
    ini.write_text("[Sequence]\nname=seq1\nimWidth=640\nimHeight=480\nframeRate=10\nseqLength=99\nimDir=img\nimExt=.png\n")
    assert parse_seqinfo(ini) == {
        "name": "seq1", "imWidth": 640, "imHeight": 480, "frameRate": 10.0, "seqLength": 99, "imDir": "img", "imExt": ".png",
    }


def test_parse_seqinfo_falls_back_to_defaults(tmp_path: Path) -> None:
    from safetrajai.visualization.gt_overlay import parse_seqinfo

    ini = tmp_path / "seqinfo.ini"
    ini.write_text("")
    assert parse_seqinfo(ini) == {
        "name": tmp_path.name, "imWidth": 1920, "imHeight": 1080, "frameRate": 30.0, "seqLength": 0,
        "imDir": "img1", "imExt": ".jpg",
    }


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_use_the_shared_module(script: str) -> None:
    from safetrajai.visualization import gt_overlay

    module = load_script(script)
    assert module.draw_gt_on_frame is gt_overlay.draw_gt_on_frame
    assert module.parse_seqinfo is gt_overlay.parse_seqinfo
    assert module.CLASS_COLORS is gt_overlay.CLASS_COLORS
