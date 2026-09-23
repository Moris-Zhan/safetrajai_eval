# evaluation/dataset_layout.py
"""
Dataset 路徑組裝：集中 SOURCE 掃描與 OUTPUT 序列目錄命名，供
track_eval.py、module/grid_search/*.py、module/collision/ 共用。

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


def discover_source_sequences(source_video_dir: Path, run_det_eval_dir: str) -> list[Path]:
    """掃描 SOURCE_VIDEO_DIR/<run_det_eval_dir>/annotation_*/**/*.mp4，跳過 Demo.mp4。
    取代舊版 track_eval.py 的 glob + 補幀相關檔名過濾（該過濾邏輯已隨補幀邏輯一併移除，
    見 spec 第 3.6 節）。"""
    pattern = str(Path(source_video_dir) / run_det_eval_dir / "annotation_*" / "**" / "*.mp4")
    return sorted(
        Path(p) for p in glob.glob(pattern, recursive=True)
        if not Path(p).stem.startswith("Demo")
    )


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
    video_seqs 是 mot_annotations_video）——呼叫端不可再硬編碼 "mot_annotations_all8"。
    注意：並非所有 dataset 都有 GT（例如 video_seqs 本來就沒有標註），回傳的路徑
    不保證存在，呼叫端需自行判斷是否要略過。）"""
    return Path(source_video_dir) / run_track_eval_dir / f"mot_annotation_{seq_name}" / "gt"


def list_stage1_sequences(det_output_dir: Path) -> list[Path]:
    """列舉 OUTPUT_ROOT/<run_det_eval_dir>/ 下所有已產生的序列目錄（無前綴，直接是子目錄）。
    目錄不存在時回傳空清單，不拋例外。"""
    det_output_dir = Path(det_output_dir)
    if not det_output_dir.is_dir():
        return []
    return sorted(p for p in det_output_dir.iterdir() if p.is_dir())
