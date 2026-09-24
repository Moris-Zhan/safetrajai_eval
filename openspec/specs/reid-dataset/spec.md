# reid-dataset Specification

## Purpose

定義 Re-ID 訓練資料如何從來源資料集的影片產生、標記、審查與切分，讓訓練資料可重現、可追溯，並且永遠不包含評估鏡頭。

## Requirements

### Requirement: 裁切只來自來源資料集並套用鏡頭 ROI

Re-ID 裁切 SHALL 只從 `config/datasets.yaml` 中 `role: source` 的資料集蒐集。處理的序列若屬於任何 `role: eval` 資料集所列的鏡頭，蒐集流程 MUST 拒絕並以錯誤停止。每支要處理的鏡頭 MUST 已有 `config/roi/<camera>.yaml`；只要有一支缺少，蒐集流程 MUST 在開始追蹤前以錯誤停止，不得退回以整張影像為 ROI。框中心點位於該鏡頭 ROI 外的偵測 MUST NOT 被裁切。

#### Scenario: 評估鏡頭被拒絕
- **WHEN** 以開啟裁切蒐集的情境設定處理鏡頭 C000025 的序列
- **THEN** 流程以錯誤停止，而且沒有寫出任何裁切

#### Scenario: 缺 ROI 的來源鏡頭被拒絕
- **WHEN** 開啟裁切蒐集，而要處理的某支來源鏡頭沒有 ROI 檔
- **THEN** 流程在開始追蹤前以錯誤停止，錯誤訊息指出缺少 ROI 的鏡頭，而且沒有寫出任何裁切

#### Scenario: ROI 外的偵測不裁切
- **WHEN** 某條軌跡在某幀的框中心點位於該鏡頭 ROI 外
- **THEN** 該幀不產生裁切，也不寫入索引

### Requirement: 裁切依時間抽樣並符合最小尺寸

每條軌跡 SHALL 依影片時間每 0.5 秒最多保留 1 張裁切（第一張在軌跡首次出現在 ROI 內時取得）。最短邊小於 32 像素的裁切 MUST NOT 被寫出，也不計入抽樣間隔。抽樣以影片的實際時間戳計算，不假設固定幀率。

#### Scenario: 10fps 影片的抽樣
- **WHEN** 一條軌跡在 10fps 影片中連續出現 3 秒，且每幀的框都符合最小尺寸
- **THEN** 該軌跡產生 6 或 7 張裁切，相鄰兩張的時間差不小於 0.5 秒

#### Scenario: 過小的框不寫出
- **WHEN** 某幀的框最短邊為 20 像素
- **THEN** 該幀不產生裁切，下一幀若符合尺寸即可取樣

### Requirement: 每張裁切都有 metadata 索引

蒐集流程 SHALL 為每張寫出的裁切在索引中寫入一列，欄位至少包含：`crop_path`（相對裁切根目錄）、`identity`、`class`、`camera`、`sequence`、`date`（YYYYMMDD，由序列名稱的時間戳取得）、`frame`、`time_s`、`x1`、`y1`、`x2`、`y2`、`score`。索引中的每一列 MUST 對應到一個存在的裁切檔案，反之亦然。身分以「序列名稱＋類別＋track_id」唯一識別。

#### Scenario: 索引與檔案一致
- **WHEN** 對一段序列完成裁切蒐集
- **THEN** 索引列數等於該序列寫出的裁切檔案數，每一列的 `crop_path` 都指向存在的檔案

#### Scenario: 日期由序列名稱取得
- **WHEN** 序列名稱為 `C000054 民族一路 天祥二路_20250629080441`
- **THEN** 該序列所有裁切的 `camera` 為 `C000054`、`date` 為 `20250629`

### Requirement: 污染標記與人工審查只產生紀錄

類別一致性檢查、軌跡內外觀離群值偵測、靜止身分與運動跳動偵測 SHALL 只輸出標記檔，MUST NOT 搬移、改名或刪除任何裁切。類別一致性 SHALL 只在人、二輪、四輪三個粗分組之間不一致時標記。人工審查的結果 SHALL 寫成排除清單（欄位 `class, identity, image, action`，`image` 為空代表整個身分），同樣不改動裁切檔案。審查頁 SHALL 只包含裁切數達到 manifest 最少張數的身分，並涵蓋其中所有被任一檢查標記的身分、bus 類別的全部身分，以及每支鏡頭以固定種子從未標記身分中抽出的隨機抽查身分。

#### Scenario: 標記不改動檔案
- **WHEN** 執行類別一致性檢查與外觀離群值偵測
- **THEN** 裁切根目錄內的檔案清單與執行前相同，只新增標記檔

#### Scenario: 審查結果成為排除清單
- **WHEN** 使用者在審查頁勾選移除某個身分並匯出，再套用審查結果
- **THEN** 排除清單新增該身分，對應的裁切檔案仍留在原處

#### Scenario: bus 全數進入審查
- **WHEN** 產生審查頁
- **THEN** 裁切數達到最少張數的每個 bus 身分都出現在審查頁，無論是否被標記

#### Scenario: 不足最少張數的身分不進審查
- **WHEN** 某個被標記的身分只有 3 張裁切，而 manifest 最少張數為 4
- **THEN** 該身分不出現在審查頁

#### Scenario: 靜止身分被標記
- **WHEN** 某身分持續 10 秒，首末兩張的框中心點位移小於框尺寸的 0.5 倍
- **THEN** 該身分被標記為靜止身分

#### Scenario: 四輪子類別誤判不標記
- **WHEN** 某個 car 身分的裁切多數被判定為 small_truck
- **THEN** 類別一致性檢查不標記該身分

#### Scenario: 隨機抽查可重現
- **WHEN** 以相同的索引與標記連續產生兩次審查頁
- **THEN** 兩次的隨機抽查身分完全相同

### Requirement: 以版本化 manifest 定義訓練資料與切分

訓練資料 SHALL 由 `dataset/reid/splits/<version>/manifest.csv` 定義，欄位為 `crop_path, identity, class, camera, date, split`，`split` 為 `train` 或 `val`。manifest 產生時 SHALL 依序套用：
1. 移除排除清單中的身分與影像。
2. 捨棄剩餘少於 4 張的身分。
3. 每個身分最多保留 30 張，依 `time_s` 均勻抽樣。
4. 切分：鏡頭 C000054 日期為 20250623、20250625、20250627 者為 `train`，20250629 者為 `val`；其他來源鏡頭全部為 `train`。

同一個身分 MUST 只出現在一個 split。manifest 只引用既有裁切，MUST NOT 複製影像。相同的輸入（索引、排除清單、規則）SHALL 產生逐位元組相同的 manifest。

#### Scenario: 身分張數規則
- **WHEN** 某身分套用排除清單後剩 3 張，另一身分剩 80 張
- **THEN** 前者不出現在 manifest，後者在 manifest 中恰好有 30 列

#### Scenario: 依日期切分
- **WHEN** 產生 manifest
- **THEN** 所有 `date=20250629` 的 C000054 身分都在 `val`，其他日期的 C000054 身分與其他來源鏡頭的身分都在 `train`

#### Scenario: 評估鏡頭永遠不出現
- **WHEN** 產生 manifest
- **THEN** manifest 中沒有任何 `camera` 屬於 `role: eval` 資料集的列

#### Scenario: 可重現
- **WHEN** 以相同輸入連續產生兩次 manifest
- **THEN** 兩份 manifest 內容完全相同

#### Scenario: 不複製影像
- **WHEN** 產生 manifest
- **THEN** 裁切根目錄與 splits 目錄內都沒有新增任何影像檔
