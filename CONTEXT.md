# SafeTrajAI 評估

從路口 CCTV 影片偵測並追蹤道路使用者，評估追蹤品質，再從追蹤結果找出交通衝突事件的離線評估情境。

## Language

### 影像來源

**鏡頭（Camera）**：
一台固定視角的路口 CCTV，以編號加視角描述識別（例如 `C000025 中華三路 民生二路(東北角)`）。Homography 與 ROI 在領域上都屬於鏡頭。
_Avoid_: tag、路口、intersection

**鏡頭編號（Camera ID）**：
鏡頭的短識別碼，是序列名稱第一個空白之前的字串（例如 `C000025`、`BEV0001`），格式不限於 `C` 加數字。
_Avoid_: camera tag、路口編號

**序列（Sequence）**：
某一台鏡頭錄下的一段連續影片（例如 `..._20251209145630_15s_1`），是追蹤、評估與衝突分析的基本單位；序列名稱的前綴對應到它的鏡頭。
_Avoid_: video、clip、seq（僅限程式碼縮寫）

**道路使用者（Road User）**：
畫面中一個真實存在、被偵測與追蹤的人或車輛，各屬於一個類別。
_Avoid_: 物件、object、target、實體

**類別（Class）**：
固定 6 種道路使用者：person、car、scooter、small_truck、bus、truck，各有固定編號，是跨模組契約。
_Avoid_: category、label

**關注區域（ROI, Region of Interest）**：
屬於鏡頭的畫面多邊形範圍，界定評估、衝突分析與訓練資料蒐集只考慮的區域；物件是否在區域內以框中心點判斷。追蹤本身不受它限制。
_Avoid_: 偵測區域、detection zone、det_corner、corner

**忽略區域（Ignore Region）**：
ROI 以外的範圍；落在其中的人工 GT，以及落在其中或與 ignore GT 匹配、但沒有匹配到 ROI 內 GT 的預測，在評估時都不計分。
_Avoid_: 過濾區、crop 範圍、限縮範圍

### 資料與模型

**人工 GT（Ground Truth）**：
由人工標註、作為評估基準的偵測框與軌跡；內容不可修改，範圍限縮只能在評估時進行。
_Avoid_: 權威 GT、原始 GT、orign、限縮版

**評估資料集（Evaluation Set）**：
附有人工 GT、只用於評估的序列集合；它涵蓋的鏡頭稱為評估鏡頭。
_Avoid_: 驗證集、all8

**來源資料集（Source Set）**：
沒有人工 GT、用來產生訓練資料的序列集合；不得包含任何評估鏡頭。
_Avoid_: video_seqs、bus_video、raw set

**工具模型（Tool Model）**：
用來產生衍生資料（Re-ID 裁切、偵測預標）的偵測模型，本身不是訓練或評估的對象。
_Avoid_: helper model、teacher

**基底模型（Base Model）**：
fine-tune 的起點權重；同一份權重可以同時是工具模型與基底模型。
_Avoid_: pretrained model、backbone（backbone 指網路結構的一部分）

### 追蹤

**軌跡（Track）**：
在同一序列、同一類別內，追蹤器賦予同一 `track_id` 的一串偵測框；跨序列不唯一，完整識別需「序列＋類別＋track_id」。
_Avoid_: trajectory、tracklet

**裁切（Crop）**：
從某段序列的某一幀、依某條軌跡的框切出的影像，是 Re-ID 資料的最小單位；每張裁切都能追溯到它的序列、幀與框。
_Avoid_: 截圖、patch、crop image

**身分（Identity）**：
Re-ID 資料集中的一個單位，由單一條軌跡的裁切組成；不跨軌跡合併，train／val 依身分互斥切分。同一個道路使用者可能對應到多個身分，身分之間不保證是不同物體。
_Avoid_: ID、person ID、虛擬身分、pseudo identity

**污染身分（Contaminated Identity）**：
裁切混入了不只一個道路使用者的身分，通常來自追蹤器的 ID 切換；是污染偵測與人工審查的對象。
_Avoid_: 髒資料、ID switch（ID switch 是追蹤指標，不是資料狀態）

**資料集版本（Dataset Version）**：
由一份 manifest 定義的 Re-ID 訓練資料，列出每張裁切所屬的身分與 split；同一批裁切可依不同的排除清單與規則產生多個版本。
_Avoid_: split、snapshot、單獨使用「資料集」（易與評估資料集、來源資料集混淆）

**排除（Exclusion）**：
人工審查後決定某個身分或某張裁切不進入資料集版本；只記錄在排除清單，檔案留在原處。
_Avoid_: 刪除、quarantine（與隔離混淆）

**隔離（Quarantine）**：
把有疑慮、整個專案都不得引用的權重或資料搬到所在目錄的 `temp/` 並標記不可使用；與排除不同，隔離會移動檔案。
_Avoid_: 排除、封存、archive

**情境設定（Scenario Config）**：
一份自包含的設定檔，完整描述一次實驗要跑哪些階段、用哪個追蹤器與參數。
_Avoid_: experiment config、profile

**階段（Stage）**：
流程的三個步驟：stage1 偵測＋追蹤、stage2 追蹤品質評估、stage3 衝突分析。
_Avoid_: step、phase

### 衝突分析

**衝突事件（Conflict Event）**：
兩條軌跡的 TTC 或 PET 低於門檻的接近事件，代表潛在風險，並非實際碰撞；程式碼與設定檔沿用 `collision` 命名。未經實測校正前只能當作技術訊號。
_Avoid_: 碰撞事件、collision event、accident

**TTC（Time-to-Collision）**：
在目前速度與方向不變的假設下，兩個道路使用者到碰撞前還剩的時間。

**PET（Post-Encroachment Time）**：
第一個道路使用者離開衝突點、到第二個道路使用者抵達同一點之間的時間差。

**俯視平面（Ground Plane）**：
經 Homography 消除透視變形後的座標系。既有鏡頭矩陣的俯視平面是衛星圖像素；人工點選時可以是自訂矩形。單位是像素，不是公尺。
_Avoid_: 衛星座標、map、地圖座標

**對應點（Correspondence Points）**：
用來計算 Homography 的成對座標（CCTV 像素 ↔ 俯視平面）。
_Avoid_: 訓練點、校正點

**驗證點（Check Points）**：
與對應點不同、另外挑選的成對座標，只用來量測既有 Homography 的誤差。
_Avoid_: 測試點、GT 點

## Relationships

- 一台**鏡頭**有多段**序列**；一個 **Homography** 與一個 **ROI** 各屬於一台**鏡頭**。
- 一台**鏡頭**要嘛是評估鏡頭、要嘛只出現在**來源資料集**，兩者互斥。
- 一個道路使用者可對應多個**身分**（跨軌跡、跨序列）。
- 一段**序列**在每個**類別**下產生零到多條**軌跡**。
- 一個**身分**恰好來自一條**軌跡**，由多張**裁切**組成；**身分**只來自**來源資料集**。
- 一個**資料集版本**引用既有的**裁切**，並套用**排除**；**隔離**的資產不會出現在任何**資料集版本**。
- 一個**衝突事件**涉及同一**序列**內的兩條**軌跡**。

## Flagged ambiguities

- 人工對應點檔案目前以**序列**為鍵（`config/homography_points/<seq>.yaml`），既有矩陣則以**鏡頭**為鍵；依領域模型應以鏡頭為鍵，這是待另開變更處理的已知落差。
- 「collision」在程式碼中指**衝突事件**，不是實際碰撞。
- 既有 Homography 矩陣以完整鏡頭名稱（含視角描述與參數尾碼）為鍵，ROI 以**鏡頭編號**為鍵；兩種鏡頭鍵並存是已知落差。
