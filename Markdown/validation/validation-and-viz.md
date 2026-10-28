# 驗證與視覺化工具

## 1. Homography 驗證

```bash
python scripts/homography/validate_homography.py
```

- 沒有任何參數（也不支援 `--help`，**直接執行就會開始驗證**）。它會掃描全部序列，對有 checkpoint 的序列輸出比對影片與誤差統計，其餘印出 `skipped (no_checkpoints)`。
- 輸出到 `models/homography/validation_output/`（`<seq>_compare.mp4`、`summary.csv`）。
- 目前只有 C000025 有 checkpoint，而且少於 4 點，所以會跳過誤差熱力圖。

## 2. 在序列影片上疊 GT／追蹤結果

```bash
python scripts/visualization/visualize_gt_on_video.py \
  --trackeval_root <mot_annotation_<seq> 根目錄> \
  --video_path <來源影片，可自動偵測> \
  --output_video <輸出.mp4>
```

常用選項：`--show_class_filter person car`、`--color_by {class,id}`、`--no_id`、`--no_class`、`--show_conf`、`--no_legend`、`--bbox_alpha 0~1`、`--max_frames N`（除錯用）。`scripts/visualization/visualize_gt_on_class_video.py` 的介面相同。
