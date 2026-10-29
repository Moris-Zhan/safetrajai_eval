"""Inputs of the web picker: first frame, satellite base map, matrix_tag note, and the make_picker.py CLI."""

import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from safetrajai.paths import PROJECT_ROOT
from safetrajai.picker.build import PickerInputError, build_picker_page, write_picker_page

CAMERA = "C000054"
SATELLITE_TAG = "C000054-2-20-1280-1280"
SCRIPT = PROJECT_ROOT / "scripts" / "picker" / "make_picker.py"


@pytest.fixture
def inputs(tmp_path: Path) -> dict:
    videos = tmp_path / "videos"
    (videos / CAMERA).mkdir(parents=True)
    writer = cv2.VideoWriter(str(videos / CAMERA / "seq.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for _ in range(3):
        writer.write(np.full((48, 64, 3), 90, dtype=np.uint8))
    writer.release()

    satellite_dir = tmp_path / "satellite"
    satellite_dir.mkdir()
    cv2.imwrite(str(satellite_dir / f"{SATELLITE_TAG}.png"), np.full((80, 96, 3), 150, dtype=np.uint8))

    config_dir = tmp_path / "homography"
    config_dir.mkdir()
    (config_dir / f"{CAMERA}.yaml").write_text(
        f"camera: {CAMERA}\nname: {CAMERA}\nsatellite: {{provider: google_static, scale: 2, size: [640, 640], zoom: 20}}\n", encoding="utf-8")
    return {"videos_root": videos, "satellite_dir": satellite_dir, "camera_config_dir": config_dir}


def page_config(page: str) -> dict:
    match = re.search(r"^const CONFIG = (.*);$", page, re.MULTILINE)
    assert match
    return json.loads(match.group(1))


def test_homography_page_embeds_the_first_frame_and_the_satellite_image(inputs: dict) -> None:
    config = page_config(build_picker_page(CAMERA, "homography", **inputs))

    assert (config["images"]["cctv"]["width"], config["images"]["cctv"]["height"]) == (64, 48)
    assert config["images"]["cctv"]["name"] == "seq.mp4"
    assert (config["images"]["satellite"]["width"], config["images"]["satellite"]["height"]) == (96, 80)
    assert config["images"]["satellite"]["name"] == f"{SATELLITE_TAG}.png"
    assert config["notes"] == []


def test_each_mode_loads_only_the_images_it_needs(inputs: dict) -> None:
    assert set(page_config(build_picker_page(CAMERA, "counting-lines", **inputs))["images"]) == {"cctv"}
    assert set(page_config(build_picker_page(CAMERA, "camera-position", **inputs))["images"]) == {"satellite"}


def test_counting_lines_needs_no_satellite_image_or_camera_config(inputs: dict, tmp_path: Path) -> None:
    inputs["satellite_dir"] = tmp_path / "nowhere"
    inputs["camera_config_dir"] = tmp_path / "nowhere"

    assert set(page_config(build_picker_page(CAMERA, "counting-lines", **inputs))["images"]) == {"cctv"}


def test_missing_satellite_image_points_to_download_satellite(inputs: dict) -> None:
    (inputs["satellite_dir"] / f"{SATELLITE_TAG}.png").unlink()

    with pytest.raises(PickerInputError, match="download_satellite.py"):
        build_picker_page(CAMERA, "homography", **inputs)


def test_missing_camera_config_is_an_error_for_satellite_modes(inputs: dict) -> None:
    (inputs["camera_config_dir"] / f"{CAMERA}.yaml").unlink()

    with pytest.raises(PickerInputError, match=f"{CAMERA}.yaml"):
        build_picker_page(CAMERA, "camera-position", **inputs)


def test_missing_video_names_the_directory(inputs: dict) -> None:
    with pytest.raises(PickerInputError, match="other"):
        build_picker_page("other", "counting-lines", **inputs)


def test_matrix_tag_is_noted_only_for_homography_points(inputs: dict) -> None:
    config_path = inputs["camera_config_dir"] / f"{CAMERA}.yaml"
    config_path.write_text(f"camera: {CAMERA}\nmatrix_tag: {SATELLITE_TAG}\n", encoding="utf-8")

    homography = page_config(build_picker_page(CAMERA, "homography", **inputs))
    other = page_config(build_picker_page(CAMERA, "checkpoints", **inputs))

    assert len(homography["notes"]) == 1 and "matrix_tag" in homography["notes"][0]
    assert other["notes"] == []
    assert "matrix_tag" in config_path.read_text(encoding="utf-8")   # the config is never modified


def test_write_picker_page_creates_the_output_directory(inputs: dict, tmp_path: Path) -> None:
    out_path = write_picker_page(CAMERA, "homography", out_dir=tmp_path / "runs" / "picker", **inputs)

    assert out_path == tmp_path / "runs" / "picker" / f"{CAMERA}_homography.html"
    assert out_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_cli_help_lists_every_mode() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0
    for mode in ("homography", "checkpoints", "counting-lines", "camera-position"):
        assert mode in result.stdout


def test_cli_reports_a_missing_camera_without_a_traceback(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--camera", "C999999", "--mode", "counting-lines", "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=120)

    assert result.returncode != 0
    assert "C999999" in result.stderr and "Traceback" not in result.stderr
