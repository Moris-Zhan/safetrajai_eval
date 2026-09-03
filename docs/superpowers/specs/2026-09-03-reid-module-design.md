# module/reid/ — 設計文件

- 日期:2026-09-03
- 狀態:待審查
- 範圍:五模組(detection/collision/grid_search/reid/tracker)+ `track_eval.py` 主流程整體重構的**第七個子專案**。本輪不處理 model 架構/訓練迴圈本身(已定案且已驗證,見第 1 節),聚焦三件事:修正過時待辦記錄、舊資料歸檔、Re-ID 資料過濾機制的架構規劃。

## 1. 背景:NEED.md 有一則過時的待辦記錄

逐檔審查 `module/reid/training/dataset.py`、`training/train.py`、`yolo_reid_trainer.py` 後確認:**model 架構與訓練方式其實已經定案且已實作、已驗證**——YOLO-as-encoder(`ReIDHookManager` 掛 hook 取倒數第二層特徵)+ 獨立 `ReIDProjector` + `BatchHardTripletLoss`(batch-hard triplet)+ `PKSampler`(P×K 抽樣)+ 依身分(而非影像)切 train/val(`split_identities()`,避免同一身分照片同時出現在 train/val 造成資料外洩)。

`markdown/NEED.md` 目前有兩則相關記錄互相矛盾:
- 已勾選(`[x]`,2026-07-24):「完整跑過 `module/reid/training/train.py` 訓練 smoke test(loss 下降、`load_reid_model()`/`encoders.py::ReID` 兩條路徑皆能還原)」。
- 未勾選(`[ ]`):「讀取邏輯、model 架構(是否延用 YOLO-as-encoder...)、訓練迴圈都需要重新規劃,**尚未定案**」。

後者已經是過時描述,本輪修正。

## 2. 目標 / 非目標

**目標**
- 修正 `markdown/NEED.md` 的過時待辦,反映「model 架構與訓練方式已定案並驗證」的實際狀態,只保留真正未解決的部分。
- 決定舊版扁平 Re-ID 資料(444 身分,無類別標籤)的去向。
- 規劃 Re-ID 資料品質過濾機制的架構(掛載點),不決定具體門檻值。

**非目標**
- 不改變 `ReIDHookManager`/`BatchHardTripletLoss`/`PKSampler`/`ReIDCropDataset` 的既有實作。
- 不重新設計 model 架構或訓練迴圈。
- 不決定過濾機制的具體數值門檻(尺寸/模糊度/截斷等)——這些需要實際資料調校,留待後續。
- 不執行舊資料的實際搬移(見第 4 節,搬移本身是批次檔案操作,需另外確認後才執行)。

## 3. NEED.md 修正內容

原「Re-ID 資料讀取、model 架構、訓練方式重新設計中」條目改寫為兩則:

1. `[x]` **model 架構與訓練方式已定案並驗證**:YOLO-as-encoder(`ReIDHookManager`)+ `ReIDProjector` + `BatchHardTripletLoss` + `PKSampler`,依身分切 train/val(`training/dataset.py::split_identities()`)。已用 `training/train.py` smoke test 驗證(loss 下降、`load_reid_model()`/`encoders.py::ReID` 兩條路徑皆能還原,見 2026-07-24 記錄)。讀取邏輯服務 `dataset/reid/crop_dataset/{class}/{identity}/` 這個分類別結構,不支援舊版扁平結構(見下一項)。
2. `[ ]`(維持未定案,細節見本文件)**舊版扁平 Re-ID 資料(`dataset/reid/crop_dataset/train/`,444 身分,無類別標籤)如何處理**:已決定搬到 `archive/` 做區隔(見第 4 節),尚未實際執行搬移。

## 4. 舊資料歸檔

`dataset/reid/crop_dataset/train/`(444 個身分,資料夾直接命名為 6 碼數字如 `000100`,無類別標籤)是與現行 6 類別資料夾(`bus`/`car`/`person`/`scooter`/`small_truck`/`truck`)**不同產生流程**的舊資料(`dataset/reid/README.md` 已記載此事實,`build_reid_split.py` 也已正確地不讀取這個子目錄)。

**技術限制**:由於沒有類別標籤,無法自動判斷這 444 個身分各自屬於哪一類,無法自動合併進現行分類別結構——若要合併,需要人工逐一重新分類(資料標註工作,不在本次程式碼重構範圍)。

**決定**:搬到 `archive/`(比照專案既有的 `archive/` 慣例,用於保留歷史/非現行流程程式碼與資料),讓 `dataset/reid/crop_dataset/` 底下只剩現行 6 類別資料夾,避免新舊實體混在同一層造成誤讀。具體搬移目標路徑建議 `archive/reid_legacy_flat_dataset/`(比照 `archive/track_config_variants/`、`archive/trackers_trackeval_duplicate/` 等既有命名慣例),但實際搬移(含 `numbers.txt`/`res.txt`/`train.cache`/`val.cache`)是批次檔案操作,依專案規則需先確認才執行,本文件只記錄決定,不在這輪執行。

搬移後需同步更新 `dataset/reid/README.md` 對應段落(移除「舊資料在 `crop_dataset/train/` 底下」的描述,改指向新位置)。

## 5. Re-ID 資料過濾機制架構

現況:`ReidCropCollector`(見 `2026-09-03-track-eval-pipeline-refactor-design.md`)蒐集裁切時只做多邊形區域過濾(是否落在 `det_corner` 內),沒有影像品質過濾;`training/dataset.py::ReIDCropDataset` 只做身分層級的「影像數低於 `min_images_per_identity` 就整個身分捨棄」,沒有單張影像層級的品質判斷。

**兩層架構**(本輪只定義掛載點與職責分工,不實作具體過濾條件):

### Tier 1:蒐集時單張影像過濾(修訂 `ReidCropCollector`)

在已核准的 `2026-09-03-track-eval-pipeline-refactor-design.md` 的 `ReidCropCollector` 設計上新增一個可選掛載點:

```python
class ReidCropCollector:
    def __init__(
        self,
        crop_dir: Path,
        det_corner: np.ndarray | None,
        quality_filters: list[Callable[[np.ndarray], bool]] | None = None,
    ):
        """quality_filters：每個 filter 接收裁切後的影像陣列(BGR, np.ndarray)，
        回傳 True＝保留、False＝丟棄（不寫檔）。預設 None（不過濾，行為與現行完全一致）。
        具體 filter 函式（尺寸/模糊度/邊界截斷等）留待後續依實際資料調校門檻再實作，
        本輪只定義介面。"""
```

`collect()` 內部在多邊形區域過濾之後、寫檔之前,依序套用 `quality_filters`,任一 filter 回傳 `False` 就跳過該張(不寫檔、不計入該身分張數)。這類**單張影像可獨立判斷**的品質問題(尺寸過小、過度模糊、bbox 貼近畫面邊緣被截斷)適合放在這裡——省去蒐集不合格影像浪費的磁碟空間與後續人工複查成本。

### Tier 2:訓練前身分/跨圖層級過濾(新增獨立工具,不隨蒐集流程執行)

新增 `dataset/reid/filter_crop_dataset.py`(比照 `build_reid_split.py` 的獨立工具定位),讀取已蒐集的 `crop_dataset/{class}/{identity}/`,套用**需要跨圖比對或人工複查**的規則(例如:同一身分資料夾內的裁切彼此視覺差異過大,疑似追蹤誤配到不同物體)。這類判斷主觀性較高,設計原則是**只標記/搬移到 `crop_dataset_rejected/{class}/{identity}/` 供人工複查,不自動刪除**——自動刪除的誤判成本高,可能誤刪真正的身分資料。

身分層級「影像數是否足夠」不需要在這裡重複做,`training/dataset.py::ReIDCropDataset` 的 `min_images_per_identity` 已經在訓練時動態處理(未達門檻的身分整個跳過,不影響其餘身分)。

### 兩層分工原則
- **可單張獨立判斷的品質問題**(尺寸、模糊度、邊界截斷)→ Tier 1,蒐集時濾掉,省空間。
- **需要跨圖比對或視覺主觀判斷的問題**(疑似誤配、身分不一致)→ Tier 2,訓練前工具,只標記不自動刪除。
- **身分層級的「數量是否足夠」**→ 已由 `training/dataset.py` 處理,不重複實作。

## 6. 影響檔案清單

| 檔案 | 改動 |
|---|---|
| `markdown/NEED.md` | 修正過時待辦(見第 3 節) |
| `2026-09-03-track-eval-pipeline-refactor-design.md`(`ReidCropCollector` 設計) | 新增 `quality_filters` 可選掛載點(見第 5 節 Tier 1),需回頭修訂 |
| `dataset/reid/crop_dataset/train/` → `archive/reid_legacy_flat_dataset/` | 待確認後執行的檔案搬移(第 4 節),本輪不執行 |
| `dataset/reid/README.md` | 待搬移執行後同步更新 |
| `dataset/reid/filter_crop_dataset.py`(新,Tier 2 工具) | 具體過濾規則與門檻待後續設計,本輪只定架構定位 |

## 7. 錯誤處理與邊界情況

- `ReidCropCollector` 的 `quality_filters` 為 `None`(預設)時行為必須與現行完全一致——這是行為保留的介面新增,不是行為變更。
- Tier 2 工具的「標記/搬移到 `crop_dataset_rejected/`」設計,需確保不會被 `build_reid_split.py`/`training/dataset.py::discover_identities()` 誤讀為正常身分資料夾(命名或路徑需明確區隔於 `crop_dataset/{class}/` 之外)。

## 8. 驗證計畫

1. NEED.md 修正:純文件變更,人工覆核內容正確性即可。
2. `ReidCropCollector` 新增 `quality_filters` 掛載點後,`quality_filters=None`(預設)情境下,重跑 `config/botsort_save_reid_crops.yaml` 的 stage1,確認裁切輸出數量/內容與新增掛載點前逐位元組一致。
3. 舊資料搬移執行後(另外確認時機),確認 `build_reid_split.py`/`training/dataset.py::discover_identities()` 對 6 個類別資料夾的行為不受影響(本來就不讀 `train/` 子目錄)。

## 9. 風險總結

- **低風險**:本輪產出主要是文件修正與架構規劃,唯一涉及實際檔案異動的是舊資料搬移,且明確標注為「待確認後執行」,不在本次直接動作。
- **中風險**:`ReidCropCollector` 新增 `quality_filters` 掛載點需要與 `2026-09-03-track-eval-pipeline-refactor-design.md` 的既有設計保持一致,避免兩份文件對 `ReidCropCollector` 建構參數的描述不同步。
