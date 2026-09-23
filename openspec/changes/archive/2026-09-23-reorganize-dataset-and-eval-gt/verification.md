# 驗證紀錄：reorganize-dataset-and-eval-gt

日期：2026-09-23。環境：conda env `track_eval`、RTX 5090。golden 基準位於 `runs/refactor_golden/`，比對輸出與中間紀錄位於 `runs/reorganize-dataset-and-eval-gt/`（皆為 gitignore）。

## 一、新基準的穩定性

以最終程式碼連跑兩次 `tests/golden/run_all.sh`（`after1`、`after2`），`compare.py after1 after2` 八個類別全部為 0 差異（`identical`）。新基準的自比容忍度仍為 0。

## 二、與舊基準 before1 的比對

`compare.py before1 after1`：

| 類別 | 結果 | 說明 |
|---|---|---|
| conflict_events | 0 差異 | 必須相同：stage3 改為讀入時套用 ROI，事件與改版前完全一致 |
| reid_crops | 0 差異 | 必須相同：Re-ID 裁切改用以鏡頭為鍵的 ROI，裁切清單不變 |
| homography | 0 差異 | 必須相同：不受本 change 影響 |
| yolo_labels | 0 差異 | stage1 逐幀追蹤結果不變，代表追蹤本身沒有被動到；`botsort_reid` 換用暫代權重後，這組資料的 stage1 追蹤結果也與 before1 相同 |
| mot_tracks | 預期變動 | stage1 軌跡不再依 ROI 過濾。180 個 stage1 軌跡檔中，before1 的每一列都存在於 after1，只多出 ROI 外的列 |
| trackeval | 預期變動 | 評估範圍改為忽略區域（UA-DETRAC 式），見第三節 |
| grid_search | 預期變動 | 分數隨 stage2 改變；最佳參數（`best_params_scooter.txt`）與 before1 相同 |
| other | 預期變動 | 少了 30 份 stage1 `data.yaml`（舊 detection 評估專用，已移除）；多了 108 份 `trackers_raw/` 備份（ignore 前處理的原始 tracker 副本） |

## 三、stage2 指標對照（同一批 stage1 輸出，三種評估方式）

- **舊**：before1，對稱硬過濾（GT 與預測都依中心點砍掉 ROI 外）。
- **MOT17**：`after1_mot17`，最初實作的忽略語意（只移除匹配到 ignore GT 的預測）。
- **UA**：after1，最終採用的 UA-DETRAC 式（ROI 外且未匹配到 ROI 內 GT 的預測也不計分）。

tracking 的人工 GT 內容三者相同（`mot_annotations_all8` 與 `(orign)` 的 gt.txt 本來就一致），差異完全來自評估方式。FN／FP／IDSW 為 6 段序列加總，HOTA／AssA 為 6 段序列平均。

| 情境 | 類別 | FN 舊／MOT17／UA | FP 舊／MOT17／UA | IDSW 舊／MOT17／UA | HOTA 舊／MOT17／UA | AssA 舊／MOT17／UA |
|---|---|---|---|---|---|---|
| bytetrack | person | 10／8／8 | 12／47／10 | 0／0／0 | 69.7／66.4／71.0 | 70.0／71.4／71.4 |
| bytetrack | car | 84／66／66 | 60／551／45 | 12／13／13 | 78.3／69.3／79.3 | 80.4／76.5／81.4 |
| bytetrack | scooter | 110／94／94 | 84／330／76 | 9／10／10 | 72.0／63.7／73.0 | 77.0／69.7／77.9 |
| bytetrack | small_truck | 6／6／6 | 1／16／0 | 4／4／4 | 50.5／52.0／55.6 | 50.1／52.0／55.1 |
| bytetrack | bus | 0／0／0 | 0／2／0 | 0／0／0 | 14.0／12.8／14.0 | 14.0／12.8／14.0 |
| bytetrack | truck | 5／4／4 | 6／29／4 | 0／0／0 | 33.8／28.7／35.5 | 34.3／29.2／36.0 |
| botsort_noreid | person | 4／2／2 | 7／39／4 | 0／0／0 | 72.2／69.4／73.7 | 72.2／73.8／73.8 |
| botsort_noreid | car | 44／22／22 | 16／390／1 | 14／14／14 | 83.8／77.9／85.1 | 84.7／83.7／86.0 |
| botsort_noreid | scooter | 88／69／69 | 59／270／51 | 7／8／8 | 74.1／67.9／75.3 | 78.3／73.6／79.5 |
| botsort_noreid | small_truck | 6／5／5 | 1／8／0 | 4／4／4 | 52.4／53.2／57.7 | 52.0／56.2／57.3 |
| botsort_noreid | bus | 0／0／0 | 0／1／0 | 0／0／0 | 15.9／15.1／15.9 | 15.9／15.1／15.9 |
| botsort_noreid | truck | 2／1／1 | 1／7／0 | 0／0／0 | 42.7／42.2／43.9 | 42.8／42.2／44.0 |
| botsort_reid | person | 4／2／2 | 7／39／4 | 0／0／0 | 72.2／69.6／73.7 | 72.2／73.8／73.8 |
| botsort_reid | car | 44／22／22 | 16／570／1 | 14／14／14 | 83.8／74.8／85.1 | 84.7／81.7／86.0 |
| botsort_reid | scooter | 87／68／68 | 58／276／50 | 7／8／8 | 74.0／67.7／75.2 | 78.2／73.4／79.3 |
| botsort_reid | small_truck | 5／4／4 | 1／16／0 | 3／3／3 | 53.1／45.8／58.4 | 52.9／46.1／58.3 |
| botsort_reid | bus | 0／0／0 | 0／1／0 | 0／0／0 | 15.9／15.1／15.9 | 15.9／15.1／15.9 |
| botsort_reid | truck | 2／1／1 | 1／7／0 | 0／0／0 | 42.7／42.2／43.9 | 42.8／42.2／44.0 |

解讀：
- UA-DETRAC 式相對舊流程，FN 減少（跨越 ROI 邊界、原本被誤砍的物件改計為 TP），FP 持平或略少，HOTA 普遍略升。這是評估方式修正造成的，不代表模型進步。
- 少數類別 IDSW 增加 1（例如 bytetrack 的 car、scooter）：邊界上的軌跡片段現在被納入計分，原本被硬過濾切掉的 ID 切換因此浮現。
- MOT17 式的 FP 暴增來自 ROI 外未標註或定位偏差的物件，這是改用 UA-DETRAC 式的原因（代價與風險見 `specs/evaluation-roi-filtering/spec.md`）。
- `botsort_reid` 的數值與 `botsort_noreid` 幾乎相同：Re-ID 權重目前全部是 detector backbone 暫代，等後續 change 換上新模型。

## 四、stage3 的回歸問題與修正

after1 第一次執行時，PET 事件在 6 段序列都與 before1 不同，但軌跡的列完全一致。原因是 stage3 先依 track_id 分組、再以 ROI 過濾，ROI 外先出現的軌跡改變了軌跡順序，而 PET 計算與順序有關。改為分組前逐列過濾後，只重跑 stage3，24 份事件檔全部與 before1 相同；最終的 after1、after2 亦同。

另有 3 列軌跡的框中心點恰好落在 ROI 邊界上：舊流程以未四捨五入的座標判斷，新流程以軌跡檔（小數兩位）判斷，結果不同。這 3 列不影響任何衝突事件。

## 五、detection 評估改用人工 GT

以 after1 bytetrack 的 stage1 輸出，另外跑一次 `run_det_eval: true`（輸出在 `runs/reorganize-dataset-and-eval-gt/det_eval_check/`，不混入 golden）：

| 序列 | 影像 | GT 框數 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| 145630_15s_1 | 72 | 1773 | 0.941 | 0.945 | 0.961 | 0.844 |
| 145630_15s_2 | 71 | 1356 | 0.967 | 0.967 | 0.987 | 0.877 |
| 150634_15s_1 | 74 | 2187 | 0.978 | 0.920 | 0.974 | 0.811 |
| 150634_15s_2 | 75 | 1535 | 0.993 | 0.985 | 0.993 | 0.903 |
| 152639_15s_1 | 74 | 2144 | 0.969 | 0.972 | 0.992 | 0.859 |
| 152639_15s_2 | 75 | 1406 | 0.977 | 0.981 | 0.990 | 0.906 |

GT 框數加總 10401，與 MOT GT 一致。每份報表的第一頁標示「ROI：未套用（全畫面）」。

第一次執行時以 symlink 放影像，ultralytics 解析 symlink 後把 6 個 `.cache` 寫進了來源 GT 目錄；已刪除這 6 個檔案（搬移前清單中不存在，屬本次新增），改用 hardlink 後重跑，來源目錄沒有任何檔案被寫入。

## 六、資料與來源完整性

- 搬移前後只比對路徑與檔案數（不讀內容，見 CLAUDE.md「禁止大量 I/O」）：總數皆為 1,127,543；少了 4 支內容相同的重複影片，多了 4 份 `DO_NOT_USE.md`，其餘檔名全部對得上。
- 人工 GT：YOLO 與 MOT 逐幀框數一致，共 10401 框（`checks/verify_gt_consistency.py`）。
- 跑完 stage1、stage2、stage3 與 detection 評估後，6 份來源 `gt.txt` 的雜湊與搬移前相同，評估資料集底下（`temp/` 除外）沒有任何新增或修改的檔案。
- `pytest`：128 passed、1 skipped。
