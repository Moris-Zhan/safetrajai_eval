"""
safetrajai.pipeline.tracking_pipeline — YOLO class-aware tracking + TrackEval evaluation pipeline
（入口：python track_eval.py 或 python scripts/pipeline/run_pipeline.py，CLI 見 safetrajai/cli/run_pipeline.py）

目錄結構設計
============
來源（唯讀）:
  SOURCE_VIDEO_DIR/
    annotations_all8/
      annotation_<seq>/
        <seq>.mp4

第一階段輸出（tracking 結果）:
  OUTPUT_ROOT/
    annotations_all8/              ← run_det_eval_dir
      annotation_<seq>/
        images/<seq>/*.jpg
        labels/<seq>/*.txt
        trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt
        <seq>.mp4  (原始影片複本)
        Demo.mp4   (標注影片，save=True 時)
    mot_annotations_all8/          ← run_track_eval_dir
      mot_annotation_<seq>/
        trackers/...  (從 annotations_all8 symlink / copy 過來)

第二階段輸出（eval 結果）:
  OUTPUT_ROOT/
    trackeval_output/              ← trackeval_output_dir
      mot_annotation_<seq>/
        det_metrics_<seq>_new.xlsx
        trackeval_metrics_<seq>_MULTI_new.xlsx

第三階段輸出（碰撞偵測結果，需 stages.run_stage3: true，預設關閉，見 config/collision/collision_detect.yaml）:
  OUTPUT_ROOT/
    collision_output/              ← collision_output_dir
      <seq>/
        ttc_events.json / ttc_events.csv / collision_ttc_log.csv / ttc_collision_heatmap.png
        pet_events.json / pet_events.csv / collision_pet_log.csv / pet_collision_heatmap.png

架構
====
Detector（safetrajai/detection/detector.py）  — YOLO 推論 + 逐類別信心過濾
VideoSequenceTracker（safetrajai/pipeline/sequence_tracker.py） — 單一影片的 class-aware tracking 迴圈（偵測→追蹤→標註→存檔）
TrackingEvalPipeline（本檔）                               — 三階段主流程（stage1 tracking → stage2 TrackEval 評估 → stage3 碰撞偵測）
"""

import shutil
from pathlib import Path

import cv2

from safetrajai.collision import run_collision_detection_for_sequence
from safetrajai.collision.conflict import run_conflict_analysis_for_sequence
from safetrajai.collision.counting.runner import run_counting_for_sequence
from safetrajai.evaluation.mot_format import save_mot_format
from safetrajai.evaluation.pipeline import run_eval_pipeline
from safetrajai.evaluation import dataset_layout
from safetrajai.evaluation.utils import probe_video
from safetrajai.pipeline.sequence_tracker import VideoSequenceTracker
from safetrajai.roi import ROI_DIR


# =============================================================================
# TrackingEvalPipeline — 三階段主流程（stage1 追蹤、stage2 評估、stage3 衝突事件）
# =============================================================================

def check_crop_sources(video_paths: list[str], roi_dir: Path = ROI_DIR, registry: dict | None = None) -> None:
    """裁切蒐集前的檢查：評估鏡頭一律拒絕，每支鏡頭都必須有 ROI 檔（spec: reid-dataset）。"""
    registry = dataset_layout.load_dataset_registry() if registry is None else registry
    eval_cameras = {cam for entry in registry.values() if entry.get("role") == "eval" for cam in entry.get("cameras", [])}
    cameras = sorted({dataset_layout.camera_id_from_seq_name(Path(p).stem) for p in video_paths})
    forbidden = [cam for cam in cameras if cam in eval_cameras]
    missing_roi = [cam for cam in cameras if not (Path(roi_dir) / f"{cam}.yaml").exists()]
    if forbidden:
        raise ValueError(f"評估鏡頭不得用來蒐集 Re-ID 裁切：{forbidden}")
    if missing_roi:
        raise ValueError(f"蒐集 Re-ID 裁切前，以下鏡頭必須先有 config/roi/<camera>.yaml：{missing_roi}")


def resolve_frame_rate(setting, video_path: str) -> int:
    """情境設定的 frame_rate：數值照用；"auto" 讀影片實際幀率並四捨五入。"""
    if isinstance(setting, str) and setting.strip().lower() == "auto":
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps or fps <= 0:
            raise ValueError(f"frame_rate: auto 無法取得影片幀率：{video_path}")
        return int(round(fps))
    return int(setting)


class TrackingEvalPipeline:
    """
    stage1：掃描 SOURCE_VIDEO_DIR 下所有影片，逐一執行 class-aware tracking，
            並將 trackers/ 輸出複製到 mot_annotations_all8/ 下供第二階段讀取。
    stage2：把第一階段輸出轉成 TrackEval 期望的目錄結構，計算 HOTA/CLEAR/Identity，
            合併結果輸出 Excel。
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def _process_single_video(self, video_path: str, det_output_dir: Path, track_output_dir: Path) -> None:
        """對單一影片：建立 VideoSequenceTracker → run() → save_mot_format() →
        複製 trackers/ 到 stage2 → 複製 SOURCE GT 到 stage2
        （GT 來源不存在時略過並警告，不中斷整條 pipeline——見本文件 Global Constraints）。
        """
        cfg = self.cfg

        print(f"\n{'='*70}")
        print(f"  VIDEO: {video_path}")
        print(f"{'='*70}")

        seq_tracker = VideoSequenceTracker(
            model_path   = cfg.det_model_path,
            tracker_type = cfg.tracker,
            conf         = cfg.conf,
            iou          = cfg.iou,
            imgsz        = cfg.imgsz,
            frame_rate   = resolve_frame_rate(cfg.frame_rate, video_path),
            gmc_method   = cfg.gmc_method,
            save         = cfg.save,
            save_txt     = cfg.save_txt,
            show         = cfg.show,
            show_unmatched = cfg.show_unmatched,
            show_removed   = cfg.show_removed,
            save_reid_crops = cfg.save_reid_crops,
            reid_crop_dir   = cfg.reid_crop_dir,
        )
        output_path, images_path, labels_path = seq_tracker.run(video_path, det_output_dir)

        # 取影片尺寸（從已複製的來源影片）
        width, height, _ = probe_video(video_path)

        # 儲存 MOT 格式（同時在 output_path/trackers/ 下建立 per-class txt）
        save_mot_format(
            imgs_path   = images_path,
            labels_path = labels_path,
            output_file = output_path / f"{output_path.stem}.txt",
            img_width   = width,
            img_height  = height,
        )

        # 將 trackers/ 由 annotations_all8 複製到 runs/track_eval/mot_annotations_all8/<seq>/trackers/
        mot_seq_dir = dataset_layout.stage2_seq_dir(track_output_dir, output_path.stem)
        shutil.copytree(
            output_path / "trackers",  # annotations_all8/<seq>
            mot_seq_dir / "trackers",  # mot_annotations_all8/<seq>
            dirs_exist_ok=True,
        )
        # 準備 GT 驗證資料集（SOURCE 端既有命名不變，source_gt_dir() 回傳值已含尾端 /gt）。
        # 並非所有 dataset 都有 GT（例如 reid_sources 是沒有人工 GT 的來源資料集，見 config/datasets.yaml），
        # 來源不存在時略過並警告，不當成錯誤中斷整條 pipeline。
        gt_src = dataset_layout.source_gt_dir(cfg.SOURCE_VIDEO_DIR, cfg.run_track_eval_dir, output_path.stem)
        if gt_src.exists():
            shutil.copytree(gt_src, mot_seq_dir / "gt", dirs_exist_ok=True)
        else:
            print(f"  ⚠️  {output_path.stem}: 找不到 GT 來源 {gt_src}，略過（此 dataset 可能本來就沒有 GT 標註）")
        print(f"  trackers → {mot_seq_dir / 'trackers'}")

    def run_stage1(self) -> Path:
        """
        Returns
        -------
        det_output_dir : Path
            annotations_all8 目錄（第二階段的 base_dir/run_det_eval_dir）
        """
        cfg = self.cfg
        det_output_dir   = cfg.OUTPUT_ROOT / cfg.run_det_eval_dir
        track_output_dir = cfg.OUTPUT_ROOT / cfg.run_track_eval_dir
        det_output_dir.mkdir(parents=True, exist_ok=True)
        track_output_dir.mkdir(parents=True, exist_ok=True)

        # 搜尋來源影片（跳過 Demo.mp4）
        mp4s = [str(p) for p in dataset_layout.discover_source_sequences(
            cfg.SOURCE_VIDEO_DIR, cfg.run_det_eval_dir, cfg.dataset_layout, cfg.dataset_cameras,
        )]

        if not mp4s:
            print(f"⚠️  No source videos found under: {cfg.SOURCE_VIDEO_DIR / cfg.run_det_eval_dir}")
            return det_output_dir

        print(f"\n✅ Found {len(mp4s)} source video(s)")

        if cfg.save_reid_crops:
            check_crop_sources(mp4s)

        for video_path in mp4s:
            self._process_single_video(video_path, det_output_dir, track_output_dir)

        return det_output_dir

    def run_stage2(self):
        """
        第二階段：讀取 OUTPUT_ROOT 內第一階段的產出，跑 TrackEval 評估，
        結果寫到 OUTPUT_ROOT/trackeval_output/ 下。
        """
        cfg = self.cfg

        print(f"\n{'='*70}")
        print(f"  STAGE 2 — TrackEval")
        print(f"  base_dir           : {cfg.base_dir}")
        print(f"  run_det_eval_dir   : {cfg.run_det_eval_dir}")
        print(f"  run_track_eval_dir : {cfg.run_track_eval_dir}")
        print(f"  trackeval_output   : {cfg.trackeval_output_dir}")
        print(f"{'='*70}\n")

        run_eval_pipeline(cfg)

    def _ensure_stage1_output_exists(self, det_output_dir: Path) -> list[Path]:
        """回傳 det_output_dir 下所有序列輸出目錄。沒有任何序列時印警告並回傳空清單——
        這是合法的 soft-skip 情境（見 config/collision/collision_detect.yaml 的「只重跑 stage3、
        沿用之前 stage1 舊輸出」用法：run_stage1: false + run_stage3: true）。
        """
        seq_dirs = dataset_layout.list_stage1_sequences(det_output_dir)
        if not seq_dirs:
            print(f"⚠️  No stage1 output found under: {det_output_dir}（stage3 需要先跑過 stage1）")
        return seq_dirs

    def run_stage3(self):
        """
        第三階段：軌跡碰撞偵測（TTC/PET，safetrajai/collision/）。

        讀取 stage1 產生的逐類別軌跡檔（annotation_<seq>/trackers/mot_challenge/
        <benchmark>-<split>/<tracker_name>__<class>/data/<seq>.txt，格式見
        CLAUDE.md「軌跡格式契約」），對每支序列跑一次
        safetrajai.collision.run_collision_detection_for_sequence()，輸出寫到
        cfg.collision_output_dir/<seq_name>/ 底下。config 的 conflict_analysis.enabled 為 true 時，
        接著跑 safetrajai.collision.conflict.run_conflict_analysis_for_sequence()（衝突指標），
        輸出寫到 cfg.collision_output_dir/<seq_name>/conflict/，不影響既有輸出。

        需要 stage1 先跑過（run_det_eval_dir 下要有 annotation_<seq>/trackers/... 與
        first_frame_with_box.jpg），若同一次執行 stages.run_stage1 也是 true，
        會直接讀到本次剛產生的結果；若只想單獨重跑 stage3，把 run_stage1 關掉即可
        沿用上一次的 stage1 輸出。
        """
        cfg = self.cfg
        det_output_dir = cfg.OUTPUT_ROOT / cfg.run_det_eval_dir

        seq_dirs = self._ensure_stage1_output_exists(det_output_dir)
        if not seq_dirs:
            return

        print(f"\n{'='*70}")
        print(f"  STAGE 3 — Collision Detection (TTC/PET)")
        print(f"  collision_output   : {cfg.collision_output_dir}")
        print(f"  homography_cameras : {cfg.homography_camera_dir}")
        print(f"{'='*70}\n")

        for seq_dir in seq_dirs:
            seq_name = seq_dir.stem
            video_path = seq_dir / f"{seq_name}.mp4"
            if not video_path.exists():
                print(f"  ⚠️  {seq_name}: 找不到 {video_path}，略過（stage1 是否有跑完？）")
                continue

            width, height, probed_fps = probe_video(video_path)
            fps = probed_fps or cfg.frame_rate or 30

            run_collision_detection_for_sequence(
                seq_output_dir=seq_dir,
                cfg=cfg,
                fps=fps,
                bounds=(height, width),
            )
            if cfg.conflict_analysis.enabled:
                run_conflict_analysis_for_sequence(
                    seq_output_dir=seq_dir,
                    cfg=cfg,
                    fps=fps,
                    bounds=(height, width),
                )
            if cfg.counting.enabled:
                run_counting_for_sequence(
                    seq_output_dir=seq_dir,
                    cfg=cfg,
                    fps=fps,
                    bounds=(height, width),
                )

    def run(self):
        """依 config 的 stages.run_stage1 / run_stage2 / run_stage3 決定要跑哪幾階段
        （stage1/stage2 預設都跑，stage3 預設關閉，需在 config 明確開啟）。"""
        if getattr(self.cfg, "run_stage1", True):
            self.run_stage1()
        else:
            print("⏭️  stage1 skipped（config 的 stages.run_stage1: false）")

        if getattr(self.cfg, "run_stage2", True):
            self.run_stage2()
        else:
            print("⏭️  stage2 skipped（config 的 stages.run_stage2: false）")

        if getattr(self.cfg, "run_stage3", False):
            self.run_stage3()
        else:
            print("⏭️  stage3 skipped（config 的 stages.run_stage3: false，預設關閉）")
