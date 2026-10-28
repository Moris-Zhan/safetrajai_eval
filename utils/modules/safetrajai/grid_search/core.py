"""
scripts/grid_search_*.py 三支腳本（bytetrack/botsort/botsort_reid）共用邏輯。

三支腳本各自只保留 PARAM_GRIDS 常數 + main()（各自 argparse 定義與預設值），
其餘 90% 以上逐字重複的邏輯（ParameterGrid、scoring、GT 處理、GridSearchRunner）
全部收斂到這裡。詳見 docs/superpowers/specs/2026-09-03-grid-search-dedup-design.md。

【safetrajai/detection/ 整合】score_params()/_run_tracking_for_video() 的偵測邏輯
改呼叫 safetrajai/detection/detector.py::Detector，取代原本三支腳本各自手刻的
model.predict() + 逐偵測套用類別門檻過濾（與 Detector.detect() 內容逐字重複）。
詳見 docs/superpowers/specs/2026-09-03-detection-module-design.md。

【路徑組裝】discover_grid_search_videos()/_resolve_gt_source_root()/score_params()
的 trial_dir 組裝一律呼叫 evaluation/dataset_layout.py（dataset-path-refactor 已合併），
OUTPUT 端序列目錄不含前綴，SOURCE 端既有命名不變。
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import math
import random
import shutil
import string
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from safetrajai.detection.detector import Detector
from safetrajai.tracking import create_tracker
from safetrajai.tracking import config as tracker_config_module
from safetrajai.evaluation import dataset_layout
from safetrajai.evaluation.mot_format import save_mot_format
from safetrajai.evaluation.trackeval_runner import run_trackeval_mot


# =============================================================================
# 1. ParameterGrid — 產生所有參數組合（等同 sklearn.model_selection.ParameterGrid）
# =============================================================================

class ParameterGrid:
    """
    把 {param: [v1, v2, ...], ...} 展開成所有排列組合的清單。

    Examples
    --------
    >>> list(ParameterGrid({"a": [1, 2], "b": [3, 4]}))
    [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """

    def __init__(self, param_grid: dict[str, list]):
        self.param_grid = param_grid
        self._keys   = sorted(param_grid.keys())          # 固定順序以利 resume 比對
        self._values = [param_grid[k] for k in self._keys]

    def __iter__(self):
        for combo in itertools.product(*self._values):
            yield dict(zip(self._keys, combo))

    def __len__(self):
        n = 1
        for v in self._values:
            n *= len(v)
        return n

    def combo_id(self, params: dict) -> str:
        """回傳參數組合的唯一字串 ID（供 CSV resume 比對用）。"""
        return json.dumps({k: params[k] for k in self._keys}, sort_keys=True)


# =============================================================================
# 2. Scoring Function — 從 TrackEval results dict 萃取 HOTA.AssA_mean
# =============================================================================

def extract_assa_mean(
    trackeval_results: dict,
    target_class: str,
    tracker_variant: str,
) -> float | None:
    """
    從 TrackEval evaluator.evaluate() 的原始回傳結果中，
    萃取指定類別的 HOTA.AssA_mean（所有 alpha 門檻的平均值）。

    TrackEval result dict 的巢狀結構：
      results
        [dataset_name]          e.g. "MotChallenge2DBox"
          [tracker_name]        e.g. "YOLO11x__scooter"
            [seq_name]          e.g. "C000025_..."  or "COMBINED_SEQ"
              [class_name]      e.g. "scooter"
                [metric_family] e.g. "HOTA"
                  [metric_key]  e.g. "AssA"  → np.ndarray(shape=(19,)) 各 alpha 分數

    AssA_mean = mean(AssA array over all alpha thresholds)

    Returns
    -------
    float | None
        萃取成功回傳 float；任何步驟失敗回傳 None。
    """
    if trackeval_results is None:
        return None

    # evaluator.evaluate() 回傳 (results_dict, messages_dict) tuple
    # run_trackeval_mot 直接 return 這個 tuple，因此取 [0] 得到 results dict
    if isinstance(trackeval_results, tuple):
        trackeval_results = trackeval_results[0]

    if not isinstance(trackeval_results, dict):
        return None

    try:
        # 第一層：dataset（通常只有一個）
        for dataset_name, dataset_val in trackeval_results.items():
            if not isinstance(dataset_val, dict):
                continue

            # 第二層：tracker variant
            trk_val = dataset_val.get(tracker_variant)
            if trk_val is None:
                # fallback: 取第一個 tracker
                for v in dataset_val.values():
                    if isinstance(v, dict):
                        trk_val = v
                        break
            if trk_val is None:
                continue

            # 第三層：sequence（取 COMBINED_SEQ 優先，否則取第一個）
            seq_val = trk_val.get("COMBINED_SEQ")
            if seq_val is None:
                for v in trk_val.values():
                    if isinstance(v, dict):
                        seq_val = v
                        break
            if seq_val is None:
                continue

            # 第四層：class
            cls_val = seq_val.get(target_class)
            if cls_val is None:
                continue

            # 第五層：metric family → HOTA → AssA
            hota_val = cls_val.get("HOTA")
            if hota_val is None:
                continue

            assa = hota_val.get("AssA")
            if assa is None:
                continue

            # AssA 是 shape=(19,) 的 np.ndarray（對應 alpha=0.05~0.95 step 0.05）
            if isinstance(assa, np.ndarray):
                return float(assa.mean())
            return float(assa)

    except Exception as e:
        print(f"  [score] extract_assa_mean failed: {e}")

    return None


# =============================================================================
# 3. GT 處理輔助函式
# =============================================================================

def _resolve_gt_source_root(cfg, seq_stem: str) -> str | None:
    """
    從 cfg 的 run_track_eval_dir 下，找出指定序列的原始 GT 目錄。

    預期結構（無前綴，見 evaluation/dataset_layout.py）：
      base_dir/mot_annotations_all8/
        <seq_stem>/
          gt/
            gt.txt        ← flat GT（9 columns MOT format）
          seqinfo.ini     ← 可選
    """
    base    = Path(cfg.OUTPUT_ROOT)
    mot_dir = base / cfg.run_track_eval_dir

    candidate = dataset_layout.stage2_seq_dir(mot_dir, seq_stem)
    if candidate.exists():
        return str(candidate)

    # fallback: glob 比對前綴；有歧義（多個候選）時不猜，避免取到別段序列的 GT
    matches = sorted(mot_dir.glob(f"{seq_stem[:20]}*"))
    if len(matches) == 1:
        return str(matches[0])

    return None


def _setup_trial_gt(
    gt_src_root: str,
    trial_dir: Path,
) -> bool:
    """
    將 GT 來源目錄中的 flat GT 檔案複製進 trial_dir，
    使 trial_dir 具備 run_trackeval_mot 所需的完整輸入結構：

      trial_dir/                         ← 作為 trackeval_root
        gt/
          gt.txt                         ← 從 gt_src_root/gt/gt.txt 複製
        seqinfo.ini                      ← 從 gt_src_root/seqinfo.ini 複製（若存在）
        trackers/...                     ← save_mot_format 已寫入

    prepare_mot_gt_shadow 會在執行時自動將 gt/gt.txt 展開成
    gt/mot_challenge/<benchmark>-<split>/<seq>/gt/gt.txt 的 shadow layout，
    因此這裡只需提供 flat 的 gt/gt.txt 即可。

    Returns
    -------
    bool
        True = 成功；False = gt.txt 來源不存在，無法繼續。
    """
    src_root = Path(gt_src_root)

    # ── GT txt ────────────────────────────────────────────────────────────────
    src_gt = src_root / "gt" / "gt.txt"
    if not src_gt.exists():
        print(f"  [setup_gt] gt.txt not found: {src_gt}")
        return False

    dst_gt_dir = trial_dir / "gt"
    dst_gt_dir.mkdir(parents=True, exist_ok=True)
    dst_gt = dst_gt_dir / "gt.txt"

    # 每次 trial 都強制覆寫，避免上一次試驗留下被修改過的 GT
    shutil.copy2(src_gt, dst_gt)

    # ── seqinfo.ini（可選，prepare_mot_gt_shadow 會從影片自動生成） ───────────
    for candidate in [src_root / "seqinfo.ini", src_root / "gt" / "seqinfo.ini"]:
        if candidate.exists():
            shutil.copy2(candidate, trial_dir / "seqinfo.ini")
            break

    return True


def discover_grid_search_videos(
    cfg,
    video_glob_override: str | None,
    max_videos: int,
) -> list[str]:
    """
    收集 grid search 用的影片路徑清單。

    預設來源是 stage1 已產生的輸出樹（OUTPUT_ROOT/run_det_eval_dir/<seq>/<seq>.mp4），
    不是 SOURCE_VIDEO_DIR——grid search 需要 stage1 已經跑過（trackers/ 與 GT 都已就緒）。
    video_glob_override 有值時直接 glob，取代預設掃描邏輯。
    不含「找不到影片」的錯誤處理（呼叫端的 main() 自行檢查回傳的 list 是否為空並印
    錯誤/sys.exit，維持現行行為）。
    """
    import glob as _glob

    if video_glob_override:
        video_paths = sorted(_glob.glob(video_glob_override, recursive=True))
    else:
        det_output_dir = Path(cfg.OUTPUT_ROOT) / cfg.run_det_eval_dir
        video_paths = sorted(
            str(seq_dir / f"{seq_dir.name}.mp4")
            for seq_dir in dataset_layout.list_stage1_sequences(det_output_dir)
            if (seq_dir / f"{seq_dir.name}.mp4").exists()
        )

    return video_paths[:max_videos]


# =============================================================================
# 4. Tracking（單一影片）
# =============================================================================

def _run_tracking_for_video(
    detector: Detector,
    video_path: str,
    labels_path: Path,
    images_path: Path,
    trial_dir: Path,
    tracker_type: str,
    conf: float,
    iou: float,
    imgsz: int,
    frame_rate: int,
    gmc_method: str,
    verbose: bool,
):
    """
    對單一影片執行 class-aware tracking，輸出 label txt 和 jpg 到指定目錄。
    這是 score_params 的內部函式，刻意精簡（不存影片、不繪框）以加速 grid search。

    統一簽章，一律接受 conf/iou/imgsz/frame_rate/gmc_method；gmc_method 只在
    tracker_type == "botsort" 時加入 tracker_kwargs（沿用現行 grid_search_botsort.py
    的判斷邏輯，行為對兩種 tracker 都不變）。

    conf 只影響 create_tracker() 的 track_high_thresh/track_low_thresh（追蹤器的軌跡
    確認門檻），不影響偵測信心過濾——偵測信心過濾一律由 detector.detect() 內部的
    get_min_conf_thresh()/get_conf_thresh_for_class() 決定，與這裡的 conf 參數無關
    （這是現行程式碼本來就有的行為，非本次變動）。

    【safetrajai/detection/ 整合】原本各腳本手刻的 model = YOLO(model_path) +
    model.predict(...) + 逐偵測套用 get_conf_thresh_for_class() 過濾，改成呼叫
    safetrajai/detection/detector.py 的 Detector.detect(frame, iou=iou, imgsz=imgsz)
    （與 track_eval.py 共用同一份偵測+信心過濾邏輯，不再各自重寫）。
    """
    from copy import deepcopy

    stem_name = Path(video_path).stem.replace(" 中華三路 民生二路(東北角)_", "_")

    tracker_kwargs = {
        "track_high_thresh": conf,
        "track_low_thresh": conf * 0.3,
    }
    if tracker_type == "botsort":
        tracker_kwargs["gmc_method"] = gmc_method

    tracker = create_tracker(
        tracker_type     = tracker_type,
        frame_rate       = frame_rate,
        **tracker_kwargs,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        clean_frame = deepcopy(frame)
        frame_id += 1

        boxes = detector.detect(frame, iou=iou, imgsz=imgsz)

        if len(boxes) > 0:
            tracks, _, _ = tracker.update(boxes.cpu().numpy(), frame)
        else:
            tracks = np.array([])

        # 儲存 label
        label_file = labels_path / f"{stem_name}_f{frame_id:06d}.txt"
        img_file   = images_path / f"{stem_name}_f{frame_id:06d}.jpg"
        cv2.imwrite(str(img_file), clean_frame)

        with open(label_file, "w") as f:
            for track in tracks:
                if len(track) >= 7:
                    x1, y1, x2, y2 = track[:4]
                    track_id = int(track[4])
                    score    = track[5]
                    cls_id   = int(track[6])
                    cx = (x1 + x2) / 2 / width
                    cy = (y1 + y2) / 2 / height
                    w  = (x2 - x1) / width
                    h  = (y2 - y1) / height
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {track_id} {score:.4f}\n")

        if verbose and frame_id % 30 == 0:
            print(f"      tracking {frame_id}/{total}", end="\r")

    cap.release()


# =============================================================================
# 5. Scoring Function（完整流程）
# =============================================================================

def score_params(
    params: dict,
    target_class: str,
    video_paths: list[str],
    model_path: str,
    cfg,                        # AppConfig
    tracker_type: str = "bytetrack",
    conf: float = 0.15,
    iou: float = 0.50,
    imgsz: int = 640,
    frame_rate: int = 30,
    gmc_method: str = "ecc",
    output_root: Path = Path("./runs/gs_tmp"),
    verbose: bool = False,
) -> float:
    """
    Scoring function（sklearn 慣例：分數越高越好）。

    流程：
      1. Patch CLASS_TRACKING_PARAMS[target_class] with params
      2. 對所有 video_paths 執行 tracking → save_mot_format
      3. 呼叫 run_trackeval_mot → 取得 HOTA.AssA_mean
      4. 還原 CLASS_TRACKING_PARAMS
      5. 回傳平均分數

    Parameters
    ----------
    params        : 本次試驗的參數字典
    target_class  : 要調優的類別名稱（如 "scooter"）
    video_paths   : 評估用的影片路徑清單
    model_path    : YOLO 模型路徑
    cfg           : AppConfig（含 benchmark_name, split_name 等）
    tracker_type  : "botsort" 或 "bytetrack"
    conf/iou/imgsz/frame_rate/gmc_method : 傳給 _run_tracking_for_video()，
        呼叫端（GridSearchRunner.run()）依各腳本現行行為決定實際值。
    output_root   : 暫存 tracking 輸出的根目錄
    verbose       : 是否印出逐幀進度

    Returns
    -------
    float
        所有序列的 HOTA.AssA_mean 平均值；若無有效結果回傳 -inf。
    """
    # ── 1. Patch CLASS_TRACKING_PARAMS ───────────────────────────────────────
    had_entry = target_class in tracker_config_module.CLASS_TRACKING_PARAMS
    original_params = copy.deepcopy(
        tracker_config_module.CLASS_TRACKING_PARAMS.get(target_class, {})
    )
    patched = copy.deepcopy(original_params)
    patched.update(params)
    tracker_config_module.CLASS_TRACKING_PARAMS[target_class] = patched

    scores: list[float] = []

    try:
        detector = Detector(model_path)

        for video_path in video_paths:
            seq_stem  = dataset_layout.seq_name_from_video_path(video_path)
            trial_dir = dataset_layout.stage2_seq_dir(output_root / "mot_annotations_all8", seq_stem)
            trial_dir.mkdir(parents=True, exist_ok=True)

            labels_path = trial_dir / "labels" / seq_stem
            images_path = trial_dir / "images" / seq_stem
            labels_path.mkdir(parents=True, exist_ok=True)
            images_path.mkdir(parents=True, exist_ok=True)

            # ── 2a. Tracking ──────────────────────────────────────────────────
            _run_tracking_for_video(
                detector    = detector,
                video_path  = video_path,
                labels_path = labels_path,
                images_path = images_path,
                trial_dir   = trial_dir,
                tracker_type= tracker_type,
                conf        = conf,
                iou         = iou,
                imgsz       = imgsz,
                frame_rate  = frame_rate,
                gmc_method  = gmc_method,
                verbose     = verbose,
            )

            # ── 2b. MOT format + per-class tracker txt ────────────────────────
            cap    = cv2.VideoCapture(video_path)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            save_mot_format(
                imgs_path   = images_path,
                labels_path = labels_path,
                output_file = trial_dir / f"{trial_dir.stem}.txt",
                img_width   = width,
                img_height  = height,
            )

            # ── 2c. 將 GT 複製進 trial_dir，使 trial_dir 自成完整的 trackeval_root ──
            #
            # run_trackeval_mot 的設計是以單一目錄（trackeval_root）同時讀取：
            #   trackeval_root/gt/gt.txt          ← flat GT（prepare_mot_gt_shadow 讀取）
            #   trackeval_root/seqinfo.ini        ← 序列資訊（可選）
            #   trackeval_root/trackers/...       ← tracker prediction txt
            #
            # save_mot_format 已把 tracker txt 寫入 trial_dir/trackers/...
            # 因此只需把 GT 來源的 gt/gt.txt 和 seqinfo.ini 複製進 trial_dir，
            # 再以 trial_dir 作為 trackeval_root 呼叫 run_trackeval_mot 即可。
            # 這樣每次 trial 完全隔離，不會污染原始 GT 目錄。
            gt_src_root = _resolve_gt_source_root(cfg, seq_stem)
            if gt_src_root is None or not Path(gt_src_root).exists():
                print(f"  [score] GT source root not found for seq={seq_stem}, skip.")
                continue

            ok = _setup_trial_gt(
                gt_src_root    = gt_src_root,
                trial_dir      = trial_dir,
            )
            if not ok:
                print(f"  [score] GT setup failed for seq={seq_stem}, skip.")
                continue

            # ── 3. TrackEval（以 trial_dir 作為 trackeval_root）──────────────
            tracker_variant = f"{cfg.tracker_name}__{target_class}"
            out_root        = str(trial_dir / "trackeval_out")

            results = run_trackeval_mot(
                cfg,
                trackeval_root         = str(trial_dir),
                benchmark_name         = cfg.benchmark_name,
                split_name             = cfg.split_name,
                seq_name               = seq_stem,
                tracker_names_to_eval  = [tracker_variant],
                out_root               = out_root,
                video_path             = video_path,
                classes_to_eval        = [target_class],
                allow_empty_tracker_file = True,
                verbose                = False,
                DEBUG                  = False,
            )

            score = extract_assa_mean(results, target_class, tracker_variant)
            if score is not None:
                scores.append(score)
                if verbose:
                    print(f"    seq={seq_stem}  AssA_mean={score:.4f}")
            else:
                print(f"  [score] AssA_mean=None for seq={seq_stem}")

    except Exception:
        traceback.print_exc()
    finally:
        # ── 4. 還原 CLASS_TRACKING_PARAMS ────────────────────────────────────
        if had_entry:
            tracker_config_module.CLASS_TRACKING_PARAMS[target_class] = original_params
        else:
            tracker_config_module.CLASS_TRACKING_PARAMS.pop(target_class, None)

    if not scores:
        return float("-inf")

    # 回傳所有影片 AssA_mean 的第 25 百分位作為此參數組合的總體分數（較偏保守評估，避免極端高分干擾）
    # 先過濾掉分數為 0 的值
    filtered_scores = [s for s in scores if s != 0]
    if not filtered_scores:
        return float("-inf")
    return float(np.percentile(filtered_scores, 25))


# =============================================================================
# 6. GridSearchRunner
# =============================================================================

class GridSearchRunner:
    """
    Tracker 參數 Grid Search 執行器。

    Parameters
    ----------
    param_grid    : 參數搜尋空間 {param_name: [v1, v2, ...]}
    target_class  : 調優目標類別（如 "scooter"）
    video_paths   : 評估用影片路徑清單
    model_path    : YOLO 模型路徑
    cfg           : AppConfig
    tracker_type  : "botsort" 或 "bytetrack"
    conf/iou/imgsz/frame_rate/gmc_method : 傳給 score_params()，各呼叫端（腳本 main()）
        依現行行為決定實際值（bytetrack/botsort_reid 傳寫死常數，botsort 傳 CLI 值）。
    output_root   : 暫存 tracking 結果的根目錄
    result_csv    : 結果存檔路徑（可 None）
    resume        : True = 跳過已有結果的組合
    verbose       : 是否印出逐幀進度
    """

    def __init__(
        self,
        param_grid:   dict[str, list],
        target_class: str,
        video_paths:  list[str],
        model_path:   str,
        cfg,
        tracker_type: str = "botsort",
        conf: float = 0.15,
        iou: float = 0.50,
        imgsz: int = 640,
        frame_rate: int = 30,
        gmc_method: str = "ecc",
        output_root:  Path = Path("./runs/grid_search/botsort"),
        result_csv:   Path | None = None,
        resume:       bool = False,
        verbose:      bool = False,
    ):
        self.grid         = ParameterGrid(param_grid)
        self.target_class = target_class
        self.video_paths  = video_paths
        self.model_path   = model_path
        self.cfg          = cfg
        self.tracker_type = tracker_type
        self.conf         = conf
        self.iou          = iou
        self.imgsz        = imgsz
        self.frame_rate   = frame_rate
        self.gmc_method   = gmc_method
        self.output_root  = Path(output_root) / tracker_type
        rand_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

        self.result_csv   = result_csv or (
            self.output_root / f"gs_{target_class}_{datetime.now():%Y%m%d_%H%M%S}_{rand_id}.csv"
        )
        self.resume       = resume
        self.verbose      = verbose

        self.results_: list[dict] = []          # 所有試驗結果
        self.best_score_: float   = float("-inf")
        self.best_params_: dict   = {}

        self.output_root.mkdir(parents=True, exist_ok=True)

        # 讀取已有的 CSV（resume 模式）
        self._done_ids: set[str] = set()
        if resume and self.result_csv.exists():
            self._load_existing_results()

    def _load_existing_results(self):
        """讀取已成功完成的試驗結果，供 resume 時跳過；失敗列（-inf / nan）略過，該組合會重跑。"""
        failed_count = 0
        with open(self.result_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                combo_id = row.get("combo_id", "")
                score    = float(row.get("score", float("-inf")))
                params   = json.loads(row.get("params", "{}"))
                if not math.isfinite(score):
                    failed_count += 1
                    continue
                self._done_ids.add(combo_id)
                self.results_.append({"combo_id": combo_id, "score": score, "params": params})
                if score > self.best_score_:
                    self.best_score_  = score
                    self.best_params_ = params

        print(f"  [resume] Loaded {len(self._done_ids)} completed trials from {self.result_csv}"
              f" (ignored {failed_count} failed trials, they will be rerun)")

    def _append_csv(self, combo_id: str, score: float, params: dict, elapsed: float):
        """將一筆試驗結果追加寫入 CSV。"""
        is_new = not self.result_csv.exists()
        with open(self.result_csv, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["combo_id", "score", "elapsed_sec", "params"] + sorted(params.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            row = {
                "combo_id":    combo_id,
                "score":       f"{score:.6f}",
                "elapsed_sec": f"{elapsed:.1f}",
                "params":      json.dumps(params, sort_keys=True),
            }
            row.update({k: v for k, v in params.items()})
            writer.writerow(row)

    def run(self) -> dict:
        """
        執行完整的 Grid Search。

        Returns
        -------
        dict
            {"best_params": ..., "best_score": ..., "all_results": ...}
        """
        total  = len(self.grid)
        done   = len(self._done_ids)
        remain = total - done

        print(f"\n{'='*70}")
        print(f"  Grid Search: class={self.target_class}  total={total}  remain={remain}")
        print(f"  tracker_type  : {self.tracker_type}")
        print(f"  conf          : {self.conf}")
        print(f"  iou           : {self.iou}")
        print(f"  imgsz         : {self.imgsz}")
        print(f"  frame_rate    : {self.frame_rate}")
        print(f"  gmc_method    : {self.gmc_method}")
        print(f"  videos        : {len(self.video_paths)}")
        print(f"  result_csv    : {self.result_csv}")
        print(f"{'='*70}")

        trial_idx = 0
        for params in self.grid:
            combo_id = self.grid.combo_id(params)

            # Resume: 跳過已完成
            if self.resume and combo_id in self._done_ids:
                continue

            trial_idx += 1
            print(f"\n[{trial_idx}/{remain}] Testing: {params}")

            trial_output = self.output_root / f"trial_{trial_idx:04d}"
            t0 = time.time()

            score = score_params(
                params       = params,
                target_class = self.target_class,
                video_paths  = self.video_paths,
                model_path   = self.model_path,
                cfg          = self.cfg,
                tracker_type = self.tracker_type,
                conf         = self.conf,
                iou          = self.iou,
                imgsz        = self.imgsz,
                frame_rate   = self.frame_rate,
                gmc_method   = self.gmc_method,
                output_root  = trial_output,
                verbose      = self.verbose,
            )

            elapsed = time.time() - t0
            status  = "✅" if score > float("-inf") else "❌"
            print(f"  {status} score (AssA_mean) = {score:.4f}  ({elapsed:.1f}s)")

            # 記錄結果
            entry = {"combo_id": combo_id, "score": score, "params": copy.deepcopy(params)}
            self.results_.append(entry)
            self._append_csv(combo_id, score, params, elapsed)

            if score > self.best_score_:
                self.best_score_  = score
                self.best_params_ = copy.deepcopy(params)
                print(f"  🏆 New best! AssA_mean={self.best_score_:.4f}")

            # 每次試驗結束後清除暫存（節省磁碟空間）
            if trial_output.exists():
                shutil.rmtree(trial_output, ignore_errors=True)

        return {
            "best_params": self.best_params_,
            "best_score":  self.best_score_,
            "all_results": self.results_,
        }

    def print_summary(self):
        """在 terminal 印出搜尋結果摘要（Top-10 + 最佳參數）。"""
        if not self.results_:
            print("No results to summarize.")
            return

        sorted_results = sorted(
            self.results_, key=lambda r: r["score"], reverse=True
        )

        GREEN = "\033[32m"
        RESET = "\033[0m"
        print(f"\n{GREEN}{'='*70}{RESET}")
        print(f"{GREEN}  Grid Search Summary — class={self.target_class}{RESET}")
        print(f"{GREEN}{'='*70}{RESET}")
        print(f"{GREEN}  Total trials : {len(self.results_)}{RESET}")
        print(f"{GREEN}  Best score   : {self.best_score_:.4f}  (HOTA.AssA_mean){RESET}")
        print(f"{GREEN}  Best params  :{RESET}")
        for k, v in sorted(self.best_params_.items()):
            print(f"{GREEN}    {k:<30} = {v}{RESET}")

        print(f"\n{GREEN}  Top-10 trials:{RESET}")
        header = f"  {'Rank':<5} {'AssA_mean':<12} params"
        print(f"{GREEN}{header}{RESET}")
        print(f"{GREEN}  {'-'*65}{RESET}")
        for i, r in enumerate(sorted_results[:10], 1):
            param_str = "  ".join(f"{k}={v}" for k, v in sorted(r["params"].items()))
            print(f"{GREEN}  {i:<5} {r['score']:<12.4f} {param_str}{RESET}")

        print(f"{GREEN}{'='*70}{RESET}")


# =============================================================================
# 7. 便利函式：產生 config.py 的更新建議
# =============================================================================

def generate_config_patch(
    target_class: str,
    best_params: dict,
    original_params: dict,
) -> str:
    """
    產生可直接貼入 config.py 的 CLASS_TRACKING_PARAMS 更新片段。
    保留原始參數中 grid search 未涉及的欄位（如 use_reid、model 等）。
    """
    merged = copy.deepcopy(original_params)
    merged.update(best_params)

    lines = [f"    '{target_class}': {{"]
    for k, v in sorted(merged.items()):
        if isinstance(v, str):
            lines.append(f"        '{k}': '{v}',")
        elif isinstance(v, bool):
            lines.append(f"        '{k}': {v},")
        else:
            lines.append(f"        '{k}': {v},")
    lines.append("    },")
    return "\n".join(lines)
