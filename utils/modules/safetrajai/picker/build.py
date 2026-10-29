"""Collect the images and notes for one web picker page (scripts/picker/make_picker.py)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List

import cv2

from safetrajai.collision.homography.camera_config import camera_config_path, load_camera_config
from safetrajai.collision.homography.points import prebuilt_satellite_image_path
from safetrajai.collision.homography.satellite import image_tag
from safetrajai.evaluation.dataset_layout import load_dataset_registry
from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.picker.page import MODES, REQUIRED_IMAGES, PickerImage, render_picker_page

DEFAULT_SATELLITE_DIR = PROJECT_ROOT / "models" / "homography" / "satellite_images"
DEFAULT_CAMERA_CONFIG_DIR = CONFIG_DIR / "homography"
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs" / "picker"
JPEG_QUALITY = 92


class PickerInputError(RuntimeError):
    pass


def raw_videos_root() -> Path:
    registry = load_dataset_registry()
    raw_roots = {entry["root"] for entry in registry.values() if entry.get("layout") == "raw"}
    if len(raw_roots) != 1:
        raise PickerInputError(f"datasets.yaml 中 layout: raw 的根目錄應該恰好一個，實際為 {sorted(raw_roots)}")
    return (PROJECT_ROOT / raw_roots.pop()).resolve()


def list_cameras(videos_root: Path | None = None) -> List[str]:
    """Camera folders under the raw video root that hold at least one mp4 (the `temp` quarantine is skipped)."""
    root = videos_root if videos_root is not None else raw_videos_root()
    return sorted(
        d.name for d in Path(root).iterdir()
        if d.is_dir() and d.name != "temp" and next(d.glob("*.mp4"), None) is not None)


def _picker_image(name: str, frame) -> PickerImage:
    height, width = frame.shape[:2]
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise PickerInputError(f"JPEG 編碼失敗：{name}")
    return PickerImage(name=name, width=width, height=height, jpeg_b64=base64.b64encode(jpeg.tobytes()).decode("ascii"))


def _first_frame_image(camera: str, videos_root: Path) -> PickerImage:
    videos = sorted((videos_root / camera).glob("*.mp4"))
    if not videos:
        raise PickerInputError(f"找不到鏡頭 {camera} 的原始影片（{videos_root / camera}）")
    capture = cv2.VideoCapture(str(videos[0]))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise PickerInputError(f"無法讀取首幀：{videos[0]}")
    return _picker_image(videos[0].name, frame)


def build_picker_page(
    camera: str,
    mode: str,
    *,
    videos_root: Path | None = None,
    satellite_dir: Path = DEFAULT_SATELLITE_DIR,
    camera_config_dir: Path = DEFAULT_CAMERA_CONFIG_DIR,
) -> str:
    if mode not in MODES:
        raise PickerInputError(f"mode 必須是 {list(MODES)} 其中之一，目前是 {mode!r}")
    needed = REQUIRED_IMAGES[mode]
    images: Dict[str, PickerImage] = {}
    notes: List[str] = []

    if "cctv" in needed:
        images["cctv"] = _first_frame_image(camera, videos_root if videos_root is not None else raw_videos_root())

    if "satellite" in needed:
        config_path = camera_config_path(camera_config_dir, camera)
        if not config_path.is_file():
            raise PickerInputError(f"找不到鏡頭設定 {config_path}，無法決定衛星圖")
        config = load_camera_config(config_path)
        if config.matrix_tag is None and config.satellite is None:
            raise PickerInputError(f"{config_path} 沒有 matrix_tag 也沒有 satellite 區塊，無法決定衛星圖")
        image_path = prebuilt_satellite_image_path(satellite_dir, image_tag(config))
        if not image_path.is_file():
            raise PickerInputError(
                f"找不到衛星圖 {image_path}；請先執行 python scripts/homography/download_satellite.py "
                f"--camera {camera} --purpose basemap")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise PickerInputError(f"無法讀取衛星圖：{image_path}")
        images["satellite"] = _picker_image(image_path.name, frame)
        if mode == "homography" and config.matrix_tag is not None:
            notes.append(
                f"{config_path.name} 已指定 matrix_tag: {config.matrix_tag}；這組點要移除 matrix_tag 才會生效"
                "（本工具不會修改設定檔）")

    return render_picker_page(mode, camera, images, notes)


def write_picker_page(camera: str, mode: str, *, out_dir: Path = DEFAULT_OUT_DIR, **inputs) -> Path:
    page = build_picker_page(camera, mode, **inputs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{camera}_{mode}.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path
