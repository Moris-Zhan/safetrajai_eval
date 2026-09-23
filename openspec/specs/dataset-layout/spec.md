# dataset-layout Specification

## Purpose

定義 `dataset/` 與 `models/` 的資料分層、評估用人工 GT 的權威性，以及有疑慮資產的隔離慣例，讓評估與後續訓練資料都有可追溯、不被汙染的來源。

## Requirements

### Requirement: 資料依角色分層存放

`dataset/` SHALL 依角色分成下列區域：
- 原始影片：`dataset/raw/videos/<camera>/`，`<camera>` 是序列名稱開頭的鏡頭編號，例如 `C000054`。
- 評估資料集：`dataset/eval/<dataset-id>/`，內含來源影片與人工 GT。
- Re-ID 衍生資料：`dataset/reid/`。
- 稽核報表：`dataset/reports/`。

repo 根目錄 MUST NOT 再有獨立的影片目錄（例如 `bus_video/`），`dataset/` 根目錄 MUST NOT 有散落的影片檔。每個資料集 SHALL 在 `config/datasets.yaml` 登記，並標註角色：`eval` 表示評估用，`source` 表示訓練資料來源。

#### Scenario: 原始影片依鏡頭歸位
- **WHEN** 整理完成後列出 `dataset/raw/videos/`
- **THEN** 每支影片都位於與其序列名稱開頭鏡頭編號相同的子目錄下，repo 根目錄不存在 `bus_video/`

#### Scenario: 資料集登記角色
- **WHEN** 讀取 `config/datasets.yaml`
- **THEN** 每個資料集項目都有 `role`，值為 `eval` 或 `source`

### Requirement: 評估鏡頭不得作為訓練資料來源

登記為 `eval` 角色的資料集所涵蓋的鏡頭，MUST NOT 出現在任何 `source` 角色資料集中。目前的評估鏡頭是 C000025。

#### Scenario: 鏡頭角色互斥檢查
- **WHEN** 執行 smoke 測試
- **THEN** 若某個鏡頭同時出現在 `eval` 與 `source` 資料集，測試失敗並指出該鏡頭

### Requirement: 評估 GT 以原始人工標註為準且不可修改

評估資料集 `cctv_all8` 的 GT SHALL 是原始人工標註：
- detection GT：5 欄 YOLO，共 10401 框，來自 `annotations_all8(orign)`。
- tracking GT：MOT 格式，來自 `mot_annotations_all8(orign)`。

任何 stage 的執行 MUST NOT 修改、覆寫或新增檔案到評估資料集的來源目錄。範圍限縮只能在評估階段、在輸出目錄中進行。

#### Scenario: 執行完整流程後 GT 不變
- **WHEN** 以任一 config 跑完 stage1 與 stage2
- **THEN** 評估資料集來源目錄內所有檔案的內容雜湊值與執行前相同，而且沒有新增檔案

#### Scenario: detection GT 與 tracking GT 一致
- **WHEN** 逐幀比對 detection GT 的框數與 tracking GT 同一幀的列數
- **THEN** 所有幀的數量相同

### Requirement: 有疑慮資產以 temp 隔離並標示不可使用

被判定為有疑慮或不可使用的權重與資料 SHALL 以搬移（不刪除）的方式放到其所在目錄下的 `temp/`。每個 `temp/` MUST 附一份 `DO_NOT_USE.md`，逐項記錄原路徑、隔離日期與原因。本次隔離範圍：
- `models/yolo-reid/` 全部權重與 `models/class_reid_weights/` 全部權重
- `dataset/reid/` 全部既有內容
- `cctv_all8` 的限縮版與重複版標註：`annotations_all8`、`annotations_all8(compasate)`、`mot_annotations_all8`、`mot_annotations_all8(Ver2)`

#### Scenario: 隔離後可追溯
- **WHEN** 開啟任一 `temp/DO_NOT_USE.md`
- **THEN** 能找到同目錄下每個被隔離項目的原路徑、日期與原因

#### Scenario: 引用被隔離資產時明確失敗
- **WHEN** config 引用了已被隔離的權重路徑並執行
- **THEN** 流程以找不到檔案的錯誤停止，不會以其他權重或空特徵悄悄繼續

#### Scenario: 設定檔不引用隔離資產
- **WHEN** 執行 smoke 測試
- **THEN** `config/` 與 `tests/golden/configs/` 下沒有任何設定值指向 `temp/` 內的路徑
