# evaluation/config_loader.py
# -*- coding: utf-8 -*-
"""
設定檔載入器。

用法：
    from safetrajai.evaluation.config_loader import load_config, AppConfig

    cfg = load_config()                          # 預設讀 config/bytetrack.yaml
    cfg = load_config("config/botsort_reid.yaml") # 指定路徑

    # 以屬性存取（推薦）
    print(cfg.base_dir)
    print(cfg.det_model_path)
    print(cfg.classes_to_run)   # dict

    # 也可當 dict 存取
    print(cfg["benchmark_name"])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError as e:
    raise ImportError("請先安裝 PyYAML：pip install pyyaml") from e


# =========================================================
# 預設設定（與 config.yaml 同步；當 yaml 缺欄位時使用）
# =========================================================
_DEFAULTS: dict = {
    # paths
    "SOURCE_VIDEO_DIR" :   "./data/",
    "TRACKEVAL_OUT_DIR":  "trackeval_output",
    "COLLISION_OUT_DIR":  "collision_output",
    "OUTPUT_ROOT"      :  "./runs/track_eval",
    "run_det_eval_dir":      "annotations_all8",
    "run_track_eval_dir":    "mot_annotations_all8",

    # models
    "det_model_path":        "./Yolo/yolo11x-reid-ultra.pt",
    "reid_model_path":       "./Yolo/yolo11x-reid.pt",
    # trackeval
    "benchmark_name":        "CCTV",
    "split_name":            "val",
    "tracker_name":          "YOLO11x",
    # flags
    "run_det_eval":          False,
    "run_track_eval":        True,
    # 兩階段主流程開關（run_stage3 預設關閉，需在 config 明確開啟，見 collision_detect.yaml）
    "run_stage1":            True,
    "run_stage2":            True,
    "run_stage3":            False,
    # classes
    "classes_to_run": {
        "person":      0,
        "car":         1,
        "scooter":     2,
        "small_truck": 3,
        "bus":         4,
        "truck":       5,
    },
    # logging
    "verbose": True,   # False = 關掉 GT 處理過程的詳細 log
    "debug":   False,  # True  = 開啟 TrackEval tracker 檔案逐項檢查 log
    # Re-ID 訓練資料裁切
    "save_reid_crops": False,
    "reid_crop_dir":   "./dataset/reid/crop_dataset",
    # 碰撞偵測（safetrajai/collision/）
    "homography_points_dir":      "config/homography_points",
    # 既有已算好的 Homography 矩陣（26 個路口，見 markdown/ARCHITECTURE.md「Homography 資料」），
    # resolve_homography() 優先用這份，找不到對應路口才退回 homography_points_dir 的人工點選
    "homography_matrices_dir":    "models/homography/matrices",
    "homography_satellite_dir":   "models/homography/satellite_images",
    "collision_max_ttc_frames":   30,
    "collision_iou_threshold":    0.8,
    "collision_lookback_sec":     3.0,
    "collision_overlap_threshold": 30.0,
}


# =========================================================
# AppConfig：扁平化存取介面
# =========================================================
class AppConfig:
    """
    將巢狀 YAML 攤平為單一層屬性，方便直接以 cfg.xxx 存取。
    同時支援 cfg["xxx"] dict 風格存取。
    """

    def __init__(self, raw: dict):
        self._data: dict = {}
        self._load(raw)

    # ------------------------------------------------------------------
    # 內部載入：把巢狀區塊攤平到頂層
    # ------------------------------------------------------------------
    def _load(self, raw: dict):
        d = dict(_DEFAULTS)  # 先填預設值

        # paths 區塊
        paths = raw.get("paths", {}) or {}

        # dataset id 對照表（見 evaluation/dataset_layout.py、config/datasets.yaml）：
        # 有 paths.dataset 時先查表取得 root/run_det_eval_dir/run_track_eval_dir 當起始值，
        # 顯式欄位（下方 paths.get(..., registry_value)）仍可逐欄位覆蓋（escape hatch）。
        dataset_id = paths.get("dataset")
        registry_entry: dict = {}
        if dataset_id is not None:
            from safetrajai.evaluation.dataset_layout import load_dataset_registry
            registry = load_dataset_registry()
            if dataset_id not in registry:
                raise ValueError(
                    f"paths.dataset={dataset_id!r} 不存在於 config/datasets.yaml，"
                    f"可用的 dataset id：{sorted(registry.keys())}"
                )
            registry_entry = registry[dataset_id]

        d["SOURCE_VIDEO_DIR"]     = paths.get("SOURCE_VIDEO_DIR", registry_entry.get("root", d["SOURCE_VIDEO_DIR"]))
        d["SOURCE_VIDEO_DIR"]     = Path(d["SOURCE_VIDEO_DIR"])

        d["run_det_eval_dir"]     = paths.get("run_det_eval_dir", registry_entry.get("run_det_eval_dir", d["run_det_eval_dir"]))
        d["run_track_eval_dir"]   = paths.get("run_track_eval_dir", registry_entry.get("run_track_eval_dir", d["run_track_eval_dir"]))
        d["dataset_layout"]       = registry_entry.get("layout", "annotated")
        d["dataset_cameras"]      = list(registry_entry.get("cameras", []) or [])
        d["TRACKEVAL_OUT_DIR"]    = paths.get("TRACKEVAL_OUT_DIR", d["TRACKEVAL_OUT_DIR"])
        
        d["OUTPUT_ROOT"]          = paths.get("OUTPUT_ROOT",    d["OUTPUT_ROOT"])
        d["OUTPUT_ROOT"]          = Path(d["OUTPUT_ROOT"])

        # base_dir / trackeval_output_dir：run_eval_pipeline() 讀取的欄位，衍生自 OUTPUT_ROOT，
        # 這裡直接算好放進 AppConfig，呼叫方（無論是 track_eval.py 還是獨立呼叫）都不用自己組
        d["base_dir"]             = str(d["OUTPUT_ROOT"])
        d["trackeval_output_dir"] = str(d["OUTPUT_ROOT"] / d["TRACKEVAL_OUT_DIR"])

        d["COLLISION_OUT_DIR"]    = paths.get("COLLISION_OUT_DIR", d["COLLISION_OUT_DIR"])
        d["collision_output_dir"] = str(d["OUTPUT_ROOT"] / d["COLLISION_OUT_DIR"])


        # models 區塊
        models = raw.get("models", {}) or {}
        d["det_model_path"]  = models.get("det_model_path",  d["det_model_path"])
        d["reid_model_path"] = models.get("reid_model_path", d["reid_model_path"])

        # trackeval 區塊
        te = raw.get("trackeval", {}) or {}
        d["benchmark_name"] = te.get("benchmark_name", d["benchmark_name"])
        d["split_name"]     = te.get("split_name",     d["split_name"])
        d["tracker_name"]   = te.get("tracker_name",   d["tracker_name"])

        # eval_flags 區塊
        flags = raw.get("eval_flags", {}) or {}
        d["run_det_eval"]   = bool(flags.get("run_det_eval",   d["run_det_eval"]))
        d["run_track_eval"] = bool(flags.get("run_track_eval", d["run_track_eval"]))

        # stages 區塊（兩階段主流程開關；沒有這個區塊時預設 stage1/stage2 都跑、stage3 不跑，向後相容）
        stages = raw.get("stages", {}) or {}
        d["run_stage1"] = bool(stages.get("run_stage1", d["run_stage1"]))
        d["run_stage2"] = bool(stages.get("run_stage2", d["run_stage2"]))
        d["run_stage3"] = bool(stages.get("run_stage3", d["run_stage3"]))

        # collision 區塊（safetrajai/collision/ 的碰撞偵測參數）
        collision = raw.get("collision", {}) or {}
        d["homography_points_dir"]       = collision.get("homography_points_dir", d["homography_points_dir"])
        d["homography_matrices_dir"]     = collision.get("homography_matrices_dir", d["homography_matrices_dir"])
        d["homography_satellite_dir"]    = collision.get("homography_satellite_dir", d["homography_satellite_dir"])
        d["collision_max_ttc_frames"]    = collision.get("max_ttc_frames",   d["collision_max_ttc_frames"])
        d["collision_iou_threshold"]     = collision.get("iou_threshold",    d["collision_iou_threshold"])
        d["collision_lookback_sec"]      = collision.get("lookback_sec",     d["collision_lookback_sec"])
        d["collision_overlap_threshold"] = collision.get("overlap_threshold", d["collision_overlap_threshold"])

        # classes_to_run（直接取頂層 key）
        if "classes_to_run" in raw and raw["classes_to_run"]:
            d["classes_to_run"] = {str(k): int(v) for k, v in raw["classes_to_run"].items()}

        if "det_corner" in raw:
            raise ValueError(
                "情境 config 不再支援頂層 det_corner；ROI 改為以鏡頭為鍵，"
                "請移到 config/roi/<camera>.yaml（見 CONTEXT.md「關注區域」、docs/adr/0005）"
            )


        # logging 區塊
        logging = raw.get("logging", {}) or {}
        d["verbose"] = bool(logging.get("verbose", d["verbose"]))
        d["debug"]   = bool(logging.get("debug",   d["debug"]))

        # tracking_params 區塊
        tracking_params = raw.get("tracking_params", {}) or {}
        d["tracker"]        = tracking_params.get("tracker", None)
        d["conf"]           = tracking_params.get("conf", None)
        d["iou"]            = tracking_params.get("iou", None)
        d["imgsz"]          = tracking_params.get("imgsz", None)
        d["frame_rate"]     = tracking_params.get("frame_rate", None)
        d["gmc_method"]     = tracking_params.get("gmc_method", None)

        # save params 區塊
        save_params = raw.get("save params", {}) or {}
        d["show"]           = save_params.get("show", None)
        d["save"]           = save_params.get("save", None)
        d["save_txt"]       = save_params.get("save_txt", None)
        d["show_unmatched"] = save_params.get("show_unmatched", None)
        d["show_removed"]   = save_params.get("show_removed", None)
        d["save_reid_crops"] = bool(save_params.get("save_reid_crops", d["save_reid_crops"]))
        d["reid_crop_dir"]    = save_params.get("reid_crop_dir", d["reid_crop_dir"])

        self._data = d

    # ------------------------------------------------------------------
    # 屬性 / dict 存取
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"AppConfig has no attribute '{name}'")

    def __getitem__(self, key: str):
        return self._data[key]

    def __contains__(self, key: str):
        return key in self._data

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def as_dict(self) -> dict:
        """回傳完整設定的淺層副本（dict）。"""
        return dict(self._data)

    def __repr__(self) -> str:
        lines = ["AppConfig("]
        for k, v in self._data.items():
            lines.append(f"  {k} = {v!r}")
        lines.append(")")
        return "\n".join(lines)


# =========================================================
# 公開 API
# =========================================================
def load_config(config_path: str | Path = "config/bytetrack.yaml") -> AppConfig:
    """
    載入 YAML 設定檔並回傳 AppConfig 實例。

    Parameters
    ----------
    config_path : str | Path
        YAML 設定檔路徑，預設為專案根目錄的 config.yaml。
        若檔案不存在，會以內建預設值建立 AppConfig 並印出警告。

    Returns
    -------
    AppConfig
    """
    p = Path(config_path)

    if not p.exists():
        print(f"⚠️  Config file not found: {p.resolve()}")
        print("   Using built-in defaults.")
        return AppConfig({})

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = AppConfig(raw)
    print(f"✅ Config loaded: {p.resolve()}")
    return cfg