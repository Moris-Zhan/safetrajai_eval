# evaluation/dataset_layout.py
"""
Dataset 路徑組裝：集中 SOURCE 掃描與 OUTPUT 序列目錄命名，供
track_eval.py、scripts/grid_search/grid_search_*.py、safetrajai/collision/ 共用。

命名慣例（見 docs/superpowers/specs/2026-09-03-dataset-path-refactor-design.md）：
    SOURCE（唯讀，既有資料，命名不變）：
        SOURCE_VIDEO_DIR/<run_det_eval_dir>/annotation_<seq>/<seq>.mp4
        SOURCE_VIDEO_DIR/mot_annotations_all8/mot_annotation_<seq>/gt/
    OUTPUT（本專案產生，無前綴）：
        OUTPUT_ROOT/<run_det_eval_dir>/<seq>/...      （stage1）
        OUTPUT_ROOT/<run_track_eval_dir>/<seq>/...    （stage2）
"""

from __future__ import annotations

import glob
from pathlib import Path

import yaml

from safetrajai.paths import CONFIG_DIR

_DATASETS_YAML_PATH = CONFIG_DIR / "datasets.yaml"


def load_dataset_registry(path: Path | str = _DATASETS_YAML_PATH) -> dict[str, dict]:
    """讀取 config/datasets.yaml，回傳 {dataset_id: {root, run_det_eval_dir, run_track_eval_dir}}。
    檔案不存在時回傳空 dict（呼叫端自行決定是否視為錯誤）。"""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("datasets", {}) or {}


def seq_name_from_video_path(video_path: str | Path) -> str:
    """序列名稱的單一來源：影片檔名（不含副檔名）。"""
    return Path(video_path).stem


def camera_id_from_seq_name(seq_name: str) -> str:
    """鏡頭編號（見 CONTEXT.md）：序列名稱第一個空白之前的字串，不限定格式。"""
    parts = seq_name.strip().split(maxsplit=1)
    if len(parts) < 2:
        raise ValueError(f"無法從序列名稱取得鏡頭編號（名稱中沒有空白）：{seq_name!r}")
    return parts[0]


def discover_source_sequences(
    source_video_dir: Path,
    run_det_eval_dir: str,
    layout: str = "annotated",
    cameras: list[str] | None = None,
) -> list[Path]:
    """依 dataset registry 的 layout 列出來源影片，跳過 Demo.mp4。

    - annotated：SOURCE_VIDEO_DIR/<run_det_eval_dir>/annotation_*/**/*.mp4
    - raw：SOURCE_VIDEO_DIR/<camera>/*.mp4，只列 cameras 清單內的鏡頭
    """
    if layout == "annotated":
        pattern = str(Path(source_video_dir) / run_det_eval_dir / "annotation_*" / "**" / "*.mp4")
        candidates = [Path(p) for p in glob.glob(pattern, recursive=True)]
    elif layout == "raw":
        if not cameras:
            raise ValueError("layout: raw 的資料集必須在 datasets.yaml 列出 cameras")
        candidates = [p for cam in cameras for p in (Path(source_video_dir) / cam).glob("*.mp4")]
    else:
        raise ValueError(f"未知的 dataset layout：{layout!r}（只支援 annotated、raw）")
    return sorted(p for p in candidates if not p.stem.startswith("Demo"))


def stage1_seq_dir(det_output_dir: Path, seq_name: str) -> Path:
    """OUTPUT_ROOT/<run_det_eval_dir>/<seq_name>/"""
    return Path(det_output_dir) / seq_name


def stage2_seq_dir(track_output_dir: Path, seq_name: str) -> Path:
    """OUTPUT_ROOT/<run_track_eval_dir>/<seq_name>/"""
    return Path(track_output_dir) / seq_name


def source_gt_dir(source_video_dir: Path, run_track_eval_dir: str, seq_name: str) -> Path:
    """SOURCE_VIDEO_DIR/<run_track_eval_dir>/mot_annotation_<seq_name>/gt/
    （SOURCE 端既有命名不變，此函式只是讀取，不產生新命名。回傳值已含尾端 /gt。
    <run_track_eval_dir> 來自 dataset registry 該筆的 run_track_eval_dir 欄位，
    不同 dataset 可能是不同子目錄名稱（例：cctv_all8 是 mot_annotations_all8，
    reid_sources 是 mot_annotations_video）——呼叫端不可再硬編碼 "mot_annotations_all8"。
    注意：並非所有 dataset 都有 GT（例如 reid_sources 是沒有人工 GT 的來源資料集），回傳的路徑
    不保證存在，呼叫端需自行判斷是否要略過。）"""
    return Path(source_video_dir) / run_track_eval_dir / f"mot_annotation_{seq_name}" / "gt"


def list_stage1_sequences(det_output_dir: Path) -> list[Path]:
    """列舉 OUTPUT_ROOT/<run_det_eval_dir>/ 下所有已產生的序列目錄（無前綴，直接是子目錄）。
    目錄不存在時回傳空清單，不拋例外。"""
    det_output_dir = Path(det_output_dir)
    if not det_output_dir.is_dir():
        return []
    return sorted(p for p in det_output_dir.iterdir() if p.is_dir())
