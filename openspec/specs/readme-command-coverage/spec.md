# readme-command-coverage Specification

## Purpose

定義 `README.md` 對可執行入口的涵蓋義務：每支腳本、每份情境設定與測試指令都要能在 README 找到用途與執行方式，並由自動檢查在新增入口卻漏寫文件時失敗，避免文件與 repo 內容再次脫節。

## Requirements

### Requirement: README 涵蓋所有腳本入口
`README.md` SHALL 為 `scripts/` 之下每一支 `.py` 腳本（含子目錄）提供至少一處以檔名指名的說明，內容包含用途與一個可直接執行的指令範例。明確列入豁免清單的腳本除外，豁免必須附理由。

#### Scenario: 新增腳本但 README 沒有提到
- **WHEN** `scripts/` 新增一支腳本，其檔名不在 README，也不在豁免清單
- **THEN** 覆蓋檢查失敗，訊息列出缺漏的腳本路徑

#### Scenario: 既有腳本全部有說明
- **WHEN** 對目前的 repo 執行覆蓋檢查
- **THEN** 26 支腳本皆出現在 README 或豁免清單，檢查通過

### Requirement: README 涵蓋所有情境設定
`README.md` SHALL 為 `config/*.yaml` 之下每一份設定檔說明其用途與對應的執行指令；`config/` 子目錄中的鏡頭鍵設定（`roi/`、`homography/`、`homography_checkpoints/`）以目錄為單位說明即可。

#### Scenario: 新增情境設定但 README 沒有提到
- **WHEN** `config/` 新增一份 `.yaml`，其檔名不在 README，也不在豁免清單
- **THEN** 覆蓋檢查失敗，訊息列出缺漏的設定檔

#### Scenario: 設定檔對應到執行指令
- **WHEN** 讀者在 README 找到 `config/conflict_analysis.yaml`
- **THEN** 同處說明它啟用的階段（stage1＋stage3＋衝突指標），以及指令 `python track_eval.py --config config/conflict_analysis.yaml`

### Requirement: 需要外部條件的指令必須標明前提
README 中的指令若需要連網、外部服務金鑰、原始影片、模型權重，或只有完整資料（而非 HF 還原的資料）才能執行，MUST 在該指令旁明確標示前提，並指出缺少前提時的結果，讓讀者不必先執行才發現失敗。

#### Scenario: 會連網或用金鑰的腳本
- **WHEN** 讀者查看 `geocode_cameras.py`、`check_homography_scale.py`、`download_satellite.py`
- **THEN** README 標明需連網與所需的環境變數名稱（`GOOGLE_MAPS_API_KEY`、`GOOGLE_MAPS_DEMO_API_KEY`），並說明 Demo Key 只能用於 `--purpose verify`；說明中不出現任何金鑰值

#### Scenario: 只有 HF 資料時無法執行的腳本
- **WHEN** 讀者在僅還原 HF 資料的機器上查看 `detect_contamination.py`、`generate_identity_review.py`、`extract_fragment_features.py`
- **THEN** README 標明它們需要個別裁切 JPEG 或來源影片，不能只靠 HF 資料執行

### Requirement: README 說明測試分層與執行時機
`README.md` SHALL 說明測試的三個層級——smoke 測試、單元測試、golden 比對——各自的涵蓋範圍、大約耗時、是否需要資料或權重，以及何時必須執行。

#### Scenario: 結構調整後的必跑測試
- **WHEN** 讀者查看測試小節，想知道調整專案結構後該跑什麼
- **THEN** 小節指出必須執行 `pytest`（smoke 與單元測試），並說明改動演算法或 stage 行為時另需 golden 比對

#### Scenario: golden 比對的成本與容忍度
- **WHEN** 讀者查看 golden 比對的說明
- **THEN** 小節標明需要完整資料與權重、全量執行約 16 分鐘、基準自比容忍度為 0，並指向 `tests/golden/TOLERANCE.md`

### Requirement: 覆蓋檢查不得讀取大型資料目錄
README 覆蓋檢查 MUST 只讀 `README.md` 與 `scripts/`、`config/` 的檔名清單，MUST NOT 讀取或遍歷 `dataset/`、`models/`、`runs/`，也不得需要網路、GPU 或模型權重；在無資料的乾淨 checkout 上也要能執行。

#### Scenario: 乾淨 checkout 執行檢查
- **WHEN** 在沒有 `dataset/`、`models/`、`runs/` 的環境執行 `pytest tests/test_readme_coverage.py`
- **THEN** 檢查正常完成，不因缺少資料而跳過或失敗
