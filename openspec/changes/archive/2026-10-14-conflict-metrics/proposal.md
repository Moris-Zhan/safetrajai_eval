# Proposal

## Why

現行 stage3 只算兩種指標：以 IoU 外推的 TTC，以及最近點 PET；而且每對軌跡只回報一次，單位是像素。`Sub_Module/trajectoryanalyst2` 涵蓋了交通衝突分析常用的整組替代安全指標：MTTC、衝突角度與類型、前導／跟隨、追撞車鏈、DRAC、PICUD、Crash Index。但它有多處單位錯誤與邏輯錯誤（見 `Sub_Module/trajectoryanalyst2/CONFLICT_METRICS.md`），不能直接移植。前置 change `camera-homography-georef` 提供了鏡頭的公尺換算，現在可以用公尺單位，把這組指標重寫進流程。

## What Changes

- 新增 stage3 的選用子步驟「衝突指標分析」，由新的 config 區塊 `conflict_analysis:` 控制，預設關閉。新增情境 `config/conflict_analysis.yaml`（stage1＋stage3，開啟分析）。
- 只對有 Homography 且 `m_per_pixel` 可用的鏡頭計算；其餘鏡頭在摘要中記錄略過原因，不退回像素。
- 位置取框的**底邊中點**，投影到俯視平面再換算成公尺；ROI 仍以框中心點過濾（ADR 0005）；軌跡以「類別＋track_id」識別。
- 運動學：先平滑位置（視窗以秒數設定），再以中心差分求速度與加速度；端點與斷點不補 0，低速時航向視為未定義。序列 fps 低於 `min_fps_for_acceleration`（預設 10）時，不計算加速度類指標（MTTC、DR、MaxD、CI、CI_m）。評估鏡頭 C000025 的影片約 4.9 fps，所以這次在真實資料上只會產生速度類指標。
- 逐幀兩兩配對的指標：
  - 衝突角度與類型（Rear-end／Lane-change／Crossing 三類，區間連續、可設定）；
  - 等速 TTC 與等加速 MTTC：碰撞距離預設為中心距離 2 m（依類別長度為選用模式）；MTTC 只在 TTC 命中時計算；只算正在接近的配對；t=0 時已經重疊的配對標成 `overlap_at_start`，不算事件；
  - 要分析的類別組合由 config 指定（預設排除 person–person），兩條軌跡都靜止時不配對；
  - 前導／跟隨；
  - 每條軌跡的最近前車與後車。
- 追撞配對的後車指標：ΔS（中心距離）、ΔV、DR／MaxD、DRAC、DRAC 超限（參考實作稱為 PICUD 的量）、Crash Index（速度與加速度取向量大小）、質量版 CI_m。
- 追撞鏈：以 Floyd-Warshall 合併同一串跟車的軌跡，並依行進方向（E/S/W/N）分組排序。
- PET：預設只算至少同時出現過一幀的軌跡配對（不要求同時出現的版本為選用模式）；以衝突區進出時間計算。
- 指標的核心定義以參考實作為準（ADR 0011），只修正參考實作中的錯誤（單位、分組、區間缺口、NaN 處理）；各項限制寫入每個序列的 summary。
- 逐幀結果合併成**衝突事件**（同一對軌跡、連續命中的幀合併為一筆），統計以事件計數。
- 輸出到 `collision_output/<seq>/conflict/`：逐幀配對指標、事件、PET、追撞鏈、摘要、事件熱區圖。事件座標同時提供公尺、俯視平面像素與 CCTV 像素三種。
- 現行 stage3 的 TTC／PET 與輸出完全不變。

## Capabilities

### New Capabilities
- `conflict-analysis`：以公尺為單位的逐幀配對衝突指標、追撞鏈、PET、衝突事件彙總，以及輸出格式。

### Modified Capabilities
（無。現行 stage3 的行為與 `project-structure` 的「stage3 事件等價」要求不變。）

## Impact

- 前置條件：change `camera-homography-georef` 已實作（需要以鏡頭解析 Homography 與 `meters_per_pixel()`），而且至少一台評估鏡頭（C000025）已填入經緯度。
- 程式：
  - 新增 `src/safetrajai/collision/conflict/` 子套件；
  - `src/safetrajai/collision/__init__.py` 新增對外入口；
  - `src/safetrajai/pipeline/tracking_pipeline.py::run_stage3()` 在開啟時呼叫；
  - `src/safetrajai/evaluation/config_loader.py` 讀取 `conflict_analysis:`。
- 設定：新增 `config/conflict_analysis.yaml`；既有情境不加這個區塊，等同關閉。
- 依賴：只用 numpy、scipy（cKDTree），不新增依賴；不使用 pandas（pandas 只是 ultralytics 的間接依賴，沒有登記在 `pyproject.toml`）。
- 測試：新增 `tests/test_conflict_metrics.py`，以合成軌跡驗證各指標的解析解；smoke 測試涵蓋新 config。
- golden：既有 5 個情境不受影響，因為沒有開啟就不會多出任何輸出檔。新情境在實作期間以連跑兩次結果相同來驗收；使用者目視確認抽查的事件後，再加入 golden 成為第 6 個情境（新基準 `conflict1`）。
- 文件：CONTEXT.md 新增指標術語，CLAUDE.md 的 stage3 說明補上子步驟。`Sub_Module/trajectoryanalyst2/CONFLICT_METRICS.md` 作為重寫依據，不搬進 repo。
