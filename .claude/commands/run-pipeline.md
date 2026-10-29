---
description: 依指定情境執行或規劃追蹤評估 pipeline
argument-hint: [config 路徑] [--stage stage1|stage2|stage3]
---

閱讀 $ARGUMENTS 指定的設定檔；未指定時使用 `config/tracking/bytetrack.yaml`。先整理這次會執行的階段、資料來源、模型、輸出位置與預期耗時。

執行前確認：

1. `stages.run_stage1`、`run_stage2`、`run_stage3` 與使用者意圖一致
2. `SOURCE_VIDEO_DIR`、模型權重與既有輸出路徑存在，且不會意外覆寫重要結果
3. `class_tracking` 已內嵌於 config，並會在建立 tracker 前載入
4. 要跑完整影片或大量序列時，先說明範圍與預期成本，等待確認
5. 已在 conda env `track_eval` 安裝本專案（`pip install -e . --no-deps`）；未安裝時 `track_eval.py` 會印出安裝指令並以 exit 1 結束

再提供或執行對應指令：

```bash
python track_eval.py --config <config 路徑>
# 等同：python scripts/pipeline/run_pipeline.py --config <config 路徑>
```

完成後回報各 stage 是否成功、輸出目錄、每序列主要結果，以及任何跳過或警告。
