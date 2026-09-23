# SafeTrajAI 評估

從路口 CCTV 影片偵測並追蹤道路使用者，評估追蹤品質，再從追蹤結果找出交通衝突事件的離線評估情境。

## Language

### 影像來源

**鏡頭（Camera）**：
一台固定視角的路口 CCTV，以編號加視角描述識別（例如 `C000025 中華三路 民生二路(東北角)-2-20-1280-1280`）。Homography 在領域上屬於鏡頭。
_Avoid_: tag、路口、intersection

**序列（Sequence）**：
某一台鏡頭錄下的一段連續影片（例如 `..._20251209145630_15s_1`），是追蹤、評估與衝突分析的基本單位；序列名稱的前綴對應到它的鏡頭。
_Avoid_: video、clip、seq（僅限程式碼縮寫）

**類別（Class）**：
固定 6 種道路使用者：person、car、scooter、small_truck、bus、truck，各有固定編號，是跨模組契約。
_Avoid_: category、label

**偵測區域（Detection Zone）**：
畫面上的多邊形範圍，偵測框中心點落在區域外的就不納入追蹤輸出。設定檔中的鍵名為 `det_corner`。
_Avoid_: corner、ROI

### 追蹤

**軌跡（Track）**：
在同一序列、同一類別內，追蹤器賦予同一 `track_id` 的一串偵測框；跨序列不唯一，完整識別需「序列＋類別＋track_id」。
_Avoid_: trajectory、tracklet

**身分（Identity）**：
Re-ID 資料集中的一個單位，由單一條軌跡的裁切影像組成；不跨軌跡合併，train／val 依身分互斥切分。
_Avoid_: ID、person ID

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

- 一台**鏡頭**有多段**序列**；一個 **Homography** 屬於一台**鏡頭**。
- 一段**序列**在每個**類別**下產生零到多條**軌跡**。
- 一個**身分**恰好來自一條**軌跡**。
- 一個**衝突事件**涉及同一**序列**內的兩條**軌跡**。

## Flagged ambiguities

- 人工對應點檔案目前以**序列**為鍵（`config/homography_points/<seq>.yaml`），既有矩陣則以**鏡頭**為鍵；依領域模型應以鏡頭為鍵，這是待另開變更處理的已知落差。
- 「collision」在程式碼中指**衝突事件**，不是實際碰撞。
