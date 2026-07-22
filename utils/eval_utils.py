# main.py
# -*- coding: utf-8 -*-
"""
One-file YOLO DET + MOT TrackEval evaluation (clean & robust).

Key fixes vs original version:
- Avoid accidentally using dataset GT labels (labels/val) as tracker predictions.
- Optionally run Ultralytics model.track() to generate tracking labels under runs/track/...
- Convert Ultralytics track labels -> MOT tracker txt (7 cols) per class (tracker variants).
- Run TrackEval per class (CLASSES_TO_EVAL=[cls]) to avoid class mismatch noise.
- Merge all per-class TrackEval results into ONE Excel report.

模組說明：
  utils.py           - 基礎工具（型別轉換、MD5、bbox 座標、檔案探查等）
  excel_export.py    - TrackEval / DET 結果輸出到 Excel
  mot_converter.py   - Ultralytics 追蹤標籤 -> MOT txt 轉換、GT 重複 ID 修正
  trackeval_runner.py - TrackEval 執行（GT shadow 準備、評估呼叫）
  dataset_prep.py    - 序列探索、DET yaml 生成、標籤修正、追蹤標籤生成
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import numpy as np

# numpy backward compat（部分函式庫仍使用 np.float / np.int）
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

from ultralytics import YOLO

# ---- 各功能模組 ----
from utils.track_eval.utils import print_file_probe, ensure_parent
from utils.track_eval.excel_export import (
    export_det_metrics_to_excel_openpyxl,
    export_trackeval_results_multi_to_excel,
    print_eval_summary,
)
from utils.track_eval.trackeval_runner import (
    build_trackeval_paths,
    verify_gt_basic,
    run_trackeval_mot,
)
from utils.track_eval.dataset_prep import (
    discover_sequences,
    ensure_det_yaml_for_seq,
    fix_det_labels_to_5cols,
    # run_ultralytics_track_to_labels,  # ← 若需自動生成追蹤標籤則啟用
)

from utils.track_eval.config_loader import load_config

def run_eval_pipeline(cfg):
    """
    執行 MOT/TrackEval pipeline，可被外部 main.py 調用。
    傳入參數 cfg（可為 dict 或已解析的 config 物件）。
    """
    import types

    # 若傳入為 dict 則轉為物件
    if isinstance(cfg, dict):
        cfg = types.SimpleNamespace(**cfg)

    det_model_path = cfg.det_model_path
    base_dir = cfg.base_dir
    benchmark_name = cfg.benchmark_name
    split_name = cfg.split_name
    tracker_name = cfg.tracker_name
    run_det_eval = cfg.run_det_eval
    run_track_eval = cfg.run_track_eval
    classes_to_run = cfg.classes_to_run
    trackeval_output_dir = cfg.trackeval_output_dir

    verbose = getattr(cfg, "verbose", True) 
    debug   = getattr(cfg, "debug",   False)

    seqs = discover_sequences(base_dir, cfg)
    print(f"\n✅ Discovered {len(seqs)} sequences under base_dir={Path(base_dir).resolve()}")

    det_model = YOLO(det_model_path)

    for i, item in enumerate(seqs[:6], 1):
        seq_name = item["seq_name"]
        video_path = item["video_path"]
        det_yaml = item["det_yaml"]
        trackeval_root = item["trackeval_root"]

        video_path = str(Path(video_path).resolve())
        det_yaml = str(Path(det_yaml).resolve())
        trackeval_root = str(Path(trackeval_root).resolve())

        print("\n" + "=" * 90)
        print(f"[{i}/{len(seqs)}] SEQ = {seq_name}")
        if verbose:
            print("video_path    :", Path(video_path).resolve())
            print("det_yaml      :", Path(det_yaml).resolve())
            print("trackeval_root:", Path(trackeval_root).resolve())
        print("=" * 90)

        if run_det_eval and not Path(det_yaml).exists():
            print(f"⚠️ DET yaml missing, skip DET: {Path(det_yaml).resolve()}")

        if run_track_eval and not Path(trackeval_root).exists():
            print(f"⚠️ trackeval_root missing, skip TRACK: {Path(trackeval_root).resolve()}")

        if (
            (run_det_eval and not Path(det_yaml).exists())
            and (run_track_eval and not Path(trackeval_root).exists())
        ):
            print("⚠️ Both DET yaml and trackeval_root missing -> SKIP this seq.")
            continue

        trackeval_output = Path(trackeval_output_dir) / Path(trackeval_root).stem
        out_base = Path(trackeval_output)
        out_base.mkdir(parents=True, exist_ok=True)

        det_xlsx = ensure_parent(out_base / f"det_metrics_{seq_name}_new.xlsx")
        track_xlsx = ensure_parent(out_base / f"trackeval_metrics_{seq_name}_MULTI_new.xlsx")

        if Path(trackeval_root).exists():
            paths = build_trackeval_paths(
                trackeval_root, benchmark_name, split_name, seq_name, tracker_name
            )
            out_root = str(Path(trackeval_root) / "trackeval_out")

            if verbose:
                print("\n=== PATHS (single source of truth) ===")
                print("gt_txt     :", paths["gt_txt"].resolve())
                print("seqinfo.ini:", paths["seq_ini"].resolve())
                print("trk_base   :", paths["trk_base"].resolve())
                print("out_root   :", Path(out_root).resolve())

            if verbose:
                print("\n=== GT probe ===")
                try:
                    verify_gt_basic(paths["gt_txt"], paths["seq_ini"])
                except Exception as e:
                    print("⚠️ GT verify failed:", repr(e))
        else:
            paths = None
            out_root = None

        if run_det_eval:
            det_yaml_p = Path(det_yaml)
            det_root = det_yaml_p.parent

            def _yaml_has_train_val(p: Path) -> bool:
                if not p.exists():
                    return False
                txt = p.read_text(encoding="utf-8", errors="ignore")
                return ("train:" in txt) and ("val:" in txt)

            if (not det_yaml_p.exists()) or (not _yaml_has_train_val(det_yaml_p)):
                try:
                    ensure_det_yaml_for_seq(
                        det_root=det_root,
                        seq_name=seq_name,
                        names=det_model.names,
                        yaml_path=det_yaml_p,
                        overwrite=True,
                    )
                    print(f"✅ DET yaml fixed/generated: {det_yaml_p.resolve()}")
                except Exception as e:
                    print(f"⚠️ DET yaml auto-fix failed -> skip DET: {det_yaml_p.resolve()}")
                    print("   reason:", repr(e))
                    det_yaml_p = None

            if det_yaml_p is None or (not det_yaml_p.exists()):
                print("⏭️ DET skipped.")
            else:
                try:
                    fix_det_labels_to_5cols(det_root, seq_name)
                except Exception as e:
                    print(f"⚠️ DET label fix failed -> skip DET: {det_root.resolve()}")
                    print("   reason:", repr(e))
                    print("⏭️ DET skipped.")
                    det_yaml_p = None

                if det_yaml_p is not None:
                    print("\n=== DET val ===")
                    det_metrics = det_model.val(
                        data=str(det_yaml_p),
                        split="val",
                        batch=8,
                        imgsz=640,
                        cache=False,
                    )
                    export_det_metrics_to_excel_openpyxl(
                        det_metrics,
                        det_model.names,
                        str(det_xlsx),
                        include_curves=True,
                    )
                    print("✅ DET Excel :", det_xlsx.resolve())

        else:
            print("⏭️ DET skipped.")

        if run_track_eval and Path(trackeval_root).exists():
            all_results = {}
            notes = {}
            tracker_variants = []

            for cname, cid in classes_to_run.items():
                tracker_variant = f"{tracker_name}__{cname}"
                tracker_variants.append(tracker_variant)

                trk_txt = (
                    Path(trackeval_root)
                    / "trackers" / "mot_challenge" / f"{benchmark_name}-{split_name}"
                    / tracker_variant / "data" / f"{seq_name}.txt"
                )

                if not trk_txt.exists():
                    notes[cname] = f"missing precomputed tracker file: {trk_txt}"
                    continue

                results = run_trackeval_mot(
                    cfg,
                    trackeval_root=str(Path(trackeval_root)),
                    benchmark_name=benchmark_name,
                    split_name=split_name,
                    seq_name=seq_name,
                    tracker_names_to_eval=[tracker_variant],
                    out_root=out_root,
                    video_path=video_path,
                    classes_to_eval=[cname],
                    allow_empty_tracker_file=True,
                    verbose=verbose,
                    DEBUG=debug,
                )

                if results is not None:
                    all_results[cname] = results

            if not all_results:
                print("❌ No class produced valid TrackEval results. (skip Excel)")
            else:
                means_data, mid_data = export_trackeval_results_multi_to_excel(
                    all_results,
                    str(track_xlsx),
                    classes_expected=list(classes_to_run.keys()),
                    tracker_base_hint=tracker_name,
                )
                # print("✅ TRK Excel :", track_xlsx.resolve())
                # if notes:
                #     print("\nNotes:")
                #     for k, v in notes.items():
                #         print(f"  - {k}: {v}")
                # print("Trackers created:", tracker_variants)
                print_eval_summary(means_data, mid_data, seq_name=seq_name)
        else:
            print("⏭️ TRACK skipped.")

    print("\n✅ ALL DONE (batch).")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run MOT/TrackEval pipeline with specified config.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config yaml file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_eval_pipeline(cfg)


if __name__ == "__main__":
    main()


# 關注Crop範圍前    
# ==========================================================================================
#   序列：C000025 中華三路 民生二路(東北角)_20251209152639_15s_2
# ==========================================================================================
# 類別                [mean]HOTA     [mid]HOTA    [mean]AssA     [mid]AssA
# ----------------------------------------------------------------------
# bus                   0.0437        0.0000        0.0478        0.0000
# car                   0.1221        0.1114        0.2583        0.2729
# person                0.3815        0.3694        0.8542        0.8585
# scooter               0.0881        0.0845        0.1519        0.1429
# small_truck           0.2267        0.2984        0.4198        0.6005
# truck                 0.0000        0.0000        0.0000        0.0000
# ==========================================================================================

# 關注Crop範圍後
# ==========================================================================================
#   序列：C000025 中華三路 民生二路(東北角)_20251209152639_15s_2
# ==========================================================================================
# 類別                [mean]HOTA     [mid]HOTA    [mean]AssA     [mid]AssA
# ----------------------------------------------------------------------
# bus                   0.0629        0.0000        0.1001        0.0000
# car                   0.0392        0.0270        0.1756        0.1578
# person                0.0431        0.0338        0.1093        0.0814
# scooter               0.0866        0.0828        0.1351        0.1271
# small_truck           0.2401        0.3158        0.4198        0.6005
# truck                 0.0000        0.0000        0.0000        0.0000
# ==========================================================================================