"""
Homography 矩陣解析：以鏡頭為鍵（config/homography/<camera>.yaml，見 camera_config.py）。

解析順序（spec `camera-homography`）：
    1. 鏡頭設定的 `matrix_tag` 指向的既有矩陣 `models/homography/matrices/H_<tag>.npy`；
    2. 鏡頭設定內的人工對應點 `cctv_points`／`map_points` 算出的矩陣；
    3. 都沒有時回傳 None，下游改用 CCTV 原始像素座標。

`matrix_tag` 指向的檔案不存在時直接報錯，不會靜默退回第 3 步。
舊的序列鍵對應點檔 `config/homography_points/<seq>.yaml` 不再讀取，存在時報錯。

人工對應點的取得方式：本專案 headless 環境（無 $DISPLAY）不能用滑鼠點選，
改用 pick_aid.py 產生的並排圖量測座標，填進鏡頭設定的 `cctv_points`／`map_points`。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from safetrajai.paths import CONFIG_DIR

from . import compute_homography, load_homography
from .camera_config import CameraHomographyConfig, load_camera_config_for_sequence

LEGACY_POINTS_DIR = CONFIG_DIR / "homography_points"


# =============================================================================
# 1. 既有矩陣（models/homography/）
# =============================================================================

def list_available_tags(matrices_dir: Path) -> List[str]:
    """掃描 matrices_dir 下的 H_<tag>.npy，回傳所有可用的 tag 清單。"""
    matrices_dir = Path(matrices_dir)
    if not matrices_dir.exists():
        return []
    return sorted(p.stem[len("H_"):] for p in matrices_dir.glob("H_*.npy"))


def matrix_path(matrices_dir: Path, tag: str) -> Path:
    return Path(matrices_dir) / f"H_{tag}.npy"


def load_prebuilt_homography(matrices_dir: Path, tag: str) -> np.ndarray:
    """讀取 models/homography/matrices/H_<tag>.npy。"""
    return load_homography(str(matrix_path(matrices_dir, tag)))


def prebuilt_satellite_image_path(satellite_images_dir: Path, tag: str) -> Path:
    """對應的衛星底圖路徑（models/homography/satellite_images/<tag>.png）。"""
    return Path(satellite_images_dir) / f"{tag}.png"


# =============================================================================
# 2. 依鏡頭解析
# =============================================================================

def assert_no_legacy_point_files(legacy_dir: Path = LEGACY_POINTS_DIR) -> None:
    """舊的序列鍵對應點檔不再使用；README 以外的 .yaml 一律報錯。"""
    legacy_dir = Path(legacy_dir)
    if not legacy_dir.exists():
        return
    legacy_files = sorted(p.name for p in legacy_dir.glob("*.yaml"))
    if legacy_files:
        raise ValueError(
            f"{legacy_dir} 下的序列鍵對應點檔已不再使用：{legacy_files}。"
            "請把 cctv_points／map_points 搬到 config/homography/<camera>.yaml"
        )


def homography_from_config(
    camera_cfg: CameraHomographyConfig, matrices_dir: Path,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """依鏡頭設定取得 (H, matrix_tag)；人工對應點產生的矩陣 tag 為 None。"""
    if camera_cfg.matrix_tag is not None:
        path = matrix_path(matrices_dir, camera_cfg.matrix_tag)
        if not path.exists():
            raise FileNotFoundError(
                f"{camera_cfg.path}: matrix_tag 指向的矩陣不存在，預期路徑為 {path}"
            )
        return load_homography(str(path)), camera_cfg.matrix_tag
    if camera_cfg.cctv_points is not None:
        return compute_homography(camera_cfg.cctv_points, camera_cfg.map_points), None
    return None, None


def resolve_camera_homography(
    seq_name: str,
    camera_dir: Path,
    matrices_dir: Path,
    legacy_points_dir: Path = LEGACY_POINTS_DIR,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """序列所屬鏡頭的 (H, matrix_tag)；鏡頭沒有設定檔或設定中沒有矩陣來源時為 (None, None)。"""
    assert_no_legacy_point_files(legacy_points_dir)
    camera_cfg = load_camera_config_for_sequence(camera_dir, seq_name)
    if camera_cfg is None:
        return None, None
    return homography_from_config(camera_cfg, matrices_dir)


# =============================================================================
# 3. 目視驗證圖
# =============================================================================

def save_verification_image(
    first_frame_path: Path, cctv_points, output_path: Path, radius: int = 6,
) -> None:
    """把 cctv_points 標在 first_frame 圖上另存一張圖，供人工目視確認座標有沒有標錯位置。

    只標 CCTV 側的點——這是給人工對應點流程用的快速檢查；若要對既有矩陣
    做正式的 reprojection error 驗證，用 safetrajai/collision/homography/validation.py。
    """
    img = cv2.imread(str(first_frame_path))
    if img is None:
        raise FileNotFoundError(f"無法讀取 {first_frame_path}")

    for i, (x, y) in enumerate(cctv_points):
        pt = (int(x), int(y))
        cv2.circle(img, pt, radius, (0, 0, 255), -1)
        cv2.putText(img, f"P{i + 1}", (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Homography 對應點檢查圖已存到: {output_path}")
