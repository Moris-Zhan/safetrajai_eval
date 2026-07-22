# dataset_prep.py
# -*- coding: utf-8 -*-
"""
資料集準備工具：
- 自動探索序列（discover_sequences）
- 生成 DET yaml（ensure_det_yaml_for_seq）
- 修正 DET 標籤格式（fix_det_labels_to_5cols）
- 執行 Ultralytics tracking 生成標籤（run_ultralytics_track_to_labels）
"""

from pathlib import Path

from ultralytics import YOLO

from utils.track_eval.utils import _assert_exists

# =========================================================
# 序列探索
# =========================================================
def discover_sequences(base_dir: str, cfg):
    """
    在 base_dir 下遞迴搜尋所有序列，回傳包含路徑資訊的 list[dict]。

    預期目錄結構：
      base_dir/
        annotations_all8/
          annotation_<seq_name>/
            <seq_name>.mp4
            data.yaml
        mot_annotations_all8/
          mot_annotation_<seq_name>/
            (TrackEval GT 資料)
    """
    run_det_eval_dir   = cfg.run_det_eval_dir
    run_track_eval_dir = cfg.run_track_eval_dir

    base = Path(base_dir).resolve()   # ✅ 固定為絕對路徑
    if not base.exists():
        raise FileNotFoundError(f"base_dir not found: {base}")

        
    mp4s = sorted([
        p for p in base.glob(f"{run_det_eval_dir}/annotation_*/**/*.mp4")
        if not p.stem.startswith("Demo")
    ])
    if not mp4s:
        raise RuntimeError(
            f"No mp4 found under: {base} (pattern: annotation_*/**/*.mp4)"
        )

    seqs = []
    for mp4 in mp4s:
        mp4 = mp4.resolve()
        seq_name = mp4.stem

        det_yaml = (
            base / run_det_eval_dir / f"annotation_{seq_name}" / "data.yaml"
        ).resolve()
        trackeval_root = (
            base / run_track_eval_dir / f"mot_annotation_{seq_name}"
        ).resolve()

        if "_org" not in seq_name:
            seqs.append({
                "seq_name": seq_name,
                "video_path": str(mp4),
                "det_yaml": str(det_yaml),
                "trackeval_root": str(trackeval_root),
            })
    return seqs


# =========================================================
# DET yaml 生成
# =========================================================
def ensure_det_yaml_for_seq(
    det_root: Path,
    seq_name: str,
    names: dict,
    yaml_path: Path,
    overwrite: bool = False,
):
    """
    為單一序列生成 Ultralytics DET yaml 設定檔。

    預期 det_root 結構：
      det_root/
        images/<seq_name>/*.jpg
        labels/<seq_name>/*.txt
    """
    img_dir = det_root / "images" / seq_name
    lbl_dir = det_root / "labels" / seq_name

    if not img_dir.exists():
        raise FileNotFoundError(f"DET images dir not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"DET labels dir not found: {lbl_dir}")

    if yaml_path.exists() and (not overwrite):
        return

    det_root_posix = det_root.resolve().as_posix()
    img_rel = f"images/{seq_name}"

    lines = [
        f"path: {det_root_posix}",
        f"train: {img_rel}",
        f"val: {img_rel}",
        "",
        "names:",
    ]
    for k in sorted(names.keys()):
        lines.append(f"  {int(k)}: {names[k]}")

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================================================
# DET 標籤修正（5 cols）
# =========================================================
def fix_det_labels_to_5cols(det_root: Path, seq_name: str):
    """
    將 DET 標籤統一修正為 Ultralytics 需要的 5 欄格式：
      cls cx cy w h

    支援處理的格式變體：
      - 6 欄 (cls tid x y w h) 或 (cls x y w h tid)
      - 7 欄 (cls x y w h tid conf)
      - 其他欄數的行直接丟棄

    同時刪除對應的 .cache 檔案，避免 Ultralytics 讀到舊快取。
    """
    det_root = Path(det_root)
    lbl_dir = det_root / "labels" / seq_name
    if not lbl_dir.exists():
        raise FileNotFoundError(f"labels dir not found: {lbl_dir}")

    # 刪除 Ultralytics cache，避免 "labels require 5 columns" 錯誤
    cache = det_root / "labels" / f"{seq_name}.cache"
    if cache.exists():
        cache.unlink()

    fixed_files = 0
    dropped = 0

    def _is_norm4(arr):
        """檢查 4 個值是否都在 [0, 1] 之間（正規化座標）。"""
        try:
            vs = [float(x) for x in arr]
        except Exception:
            return False
        return all(0.0 <= v <= 1.0 for v in vs)

    for txt in sorted(lbl_dir.glob("*.txt")):
        lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        out = []
        changed = False

        for line in lines:
            s = line.strip()
            if not s:
                continue
            p = s.split()

            if len(p) == 5:
                # 已是正確格式
                out.append(" ".join(p))
                continue

            if len(p) == 6:
                # A) cls tid x y w h
                if _is_norm4(p[2:6]):
                    out.append(" ".join([p[0], p[2], p[3], p[4], p[5]]))
                    changed = True
                    continue
                # B) cls x y w h tid
                if _is_norm4(p[1:5]):
                    out.append(" ".join([p[0], p[1], p[2], p[3], p[4]]))
                    changed = True
                    continue

                # fallback: 取前 5 欄
                out.append(" ".join(p[:5]))
                changed = True
                continue

            if len(p) == 7:
                # cls x y w h tid conf
                if _is_norm4(p[1:5]):
                    out.append(" ".join([p[0], p[1], p[2], p[3], p[4]]))
                    changed = True
                    continue

            # 其他長度 -> 丟棄
            dropped += 1
            changed = True

        if changed:
            txt.write_text(
                "\n".join(out) + ("\n" if out else ""),
                encoding="utf-8",
            )
            fixed_files += 1

    print(f"✅ DET labels fixed: files_changed={fixed_files}, dropped_lines={dropped}")

