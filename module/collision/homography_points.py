"""
Homography 矩陣載入：優先用既有已算好的矩陣，沒有對應路口才退回人工點選流程。

## 1. 既有矩陣（優先，見 models/homography/）
`models/homography/matrices/H_<tag>.npy` 是從 markdown/reference_results/homongraphy_val/
搬過來的既有成果，涵蓋 26 個路口（tag 命名規則、資料來源見 markdown/ARCHITECTURE.md
「Homography 資料」章節）。`find_tag_for_sequence()` 依序列名比對路口名前綴找出對應 tag，
`resolve_homography()` 會優先用這份；只有 26 個路口以外的新路口才需要走第 2 節的
人工點選流程。⚠️ 這 26 個路口裡，只有本專案實際有 CCTV 影片的 6 個（C000025、
C000043、C000054、C000085、C000294、C000297）才有辦法真的驗證矩陣準確度（見
module/collision/homography_validation.py）；其餘 20 個只有矩陣本身，沒有 CCTV 影片
可以選 check points，屬於「未驗證」狀態，套用時要留意。

## 2. 人工點選（fallback：路口不在既有 26 個之列時使用）
原始 SafeTrajAI 的選點是 cv2 滑鼠互動視窗（見 markdown/HOMOGRAPHY.md），本專案
headless 環境（無 $DISPLAY）無法直接沿用，改成人工流程：

    1. 打開 stage1 已存的 <seq>/first_frame_with_box.jpg，自己在圖上
       量測/估計 4 個以上對應點的像素座標。
    2. 把座標填進 <homography_points_dir>/<seq>.yaml（cctv_points + map_points，
       見下方格式）。
    3. resolve_homography() 找不到既有矩陣時，會自動偵測該序列有沒有這份檔案：
       有 → 算 Homography；沒有 → 回傳 None，該序列跳過 homography，只用原始像素座標。

本專案目前沒有真實地圖底圖，map_points 是使用者自行在同一張 CCTV 畫面上定義的
「校正後」座標（例如把透視變形的路口攤平成矩形），不是地理座標，詳見
markdown/HOMOGRAPHY.md、markdown/TRAJECTORY_FORMAT.md 第 3 節。

yaml 格式（cctv_points / map_points 需一一對應、順序相同，至少 4 組）：
    cctv_points:
      - [x1, y1]
      - [x2, y2]
      - [x3, y3]
      - [x4, y4]
    map_points:
      - [x1, y1]
      - [x2, y2]
      - [x3, y3]
      - [x4, y4]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

from .homography import compute_homography, load_homography

_TAG_SUFFIX_RE = re.compile(r"-\d+-\d+-\d+-\d+$")


# =============================================================================
# 1. 既有矩陣（優先，見 models/homography/）
# =============================================================================

def list_available_tags(matrices_dir: Path) -> List[str]:
    """掃描 matrices_dir 下的 H_<tag>.npy，回傳所有可用的 tag 清單。"""
    matrices_dir = Path(matrices_dir)
    if not matrices_dir.exists():
        return []
    return sorted(p.stem[len("H_"):] for p in matrices_dir.glob("H_*.npy"))


def find_tag_for_sequence(seq_name: str, matrices_dir: Path) -> Optional[str]:
    """依序列名找出對應的既有路口 tag。

    tag 命名規則是 "{路口名}-{SCALE}-{ZOOM}-{寬}-{高}"（例如
    "C000025 中華三路 民生二路(東北角)-2-20-1280-1280"），本專案序列名（來源影片檔名）
    是 "{路口名}_{時間戳記}..."（例如
    "C000025 中華三路 民生二路(東北角)_20251209152639_15s_1"）。兩者的「路口名」
    部分完全一致，只是後綴不同，用 seq_name 是否以「tag 去掉數字後綴」開頭來比對。
    """
    for tag in list_available_tags(matrices_dir):
        intersection_name = _TAG_SUFFIX_RE.sub("", tag)
        if seq_name.startswith(intersection_name):
            return tag
    return None


def load_prebuilt_homography(matrices_dir: Path, tag: str) -> np.ndarray:
    """讀取 models/homography/matrices/H_<tag>.npy。"""
    return load_homography(str(Path(matrices_dir) / f"H_{tag}.npy"))


def prebuilt_satellite_image_path(satellite_images_dir: Path, tag: str) -> Path:
    """對應的衛星底圖路徑（models/homography/satellite_images/<tag>.png）。"""
    return Path(satellite_images_dir) / f"{tag}.png"


def resolve_homography(
    seq_name: str, matrices_dir: Path, homography_points_dir: Path,
) -> Optional[np.ndarray]:
    """依序列名找出可用的 Homography 矩陣：優先用既有矩陣，找不到才退回人工點選檔案。

    Returns:
        3x3 矩陣，或 None（兩種來源都沒有時，呼叫端據此退回原始像素座標）。
    """
    tag = find_tag_for_sequence(seq_name, matrices_dir)
    if tag is not None:
        return load_prebuilt_homography(matrices_dir, tag)
    return load_homography_for_sequence(homography_points_dir, seq_name)


# =============================================================================
# 2. 人工點選（fallback：路口不在既有 26 個之列時使用）
# =============================================================================

def homography_points_path(homography_points_dir: Path, seq_name: str) -> Path:
    """<homography_points_dir>/<seq_name>.yaml 的完整路徑。"""
    return Path(homography_points_dir) / f"{seq_name}.yaml"


def load_homography_for_sequence(homography_points_dir: Path, seq_name: str) -> Optional[np.ndarray]:
    """若 <homography_points_dir>/<seq_name>.yaml 存在，讀取對應點並算出 Homography 矩陣。

    不存在則回傳 None——呼叫端據此判斷該序列是否啟用 homography。
    """
    points_path = homography_points_path(homography_points_dir, seq_name)
    if not points_path.exists():
        return None

    with points_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cctv_points = raw.get("cctv_points")
    map_points = raw.get("map_points")
    if not cctv_points or not map_points:
        print(f"⚠️  {points_path} 缺少 cctv_points/map_points，略過 homography")
        return None

    return compute_homography(cctv_points, map_points)


# =============================================================================
# 3. 目視驗證圖
# =============================================================================

def save_verification_image(
    first_frame_path: Path, cctv_points, output_path: Path, radius: int = 6,
) -> None:
    """把 cctv_points 標在 first_frame 圖上另存一張圖，供人工目視確認座標有沒有標錯位置。

    只標 CCTV 側的點——這是給第 2 節「人工點選」流程用的快速檢查；若要對既有矩陣
    做正式的 reprojection error 驗證，用 module/collision/homography_validation.py。
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
