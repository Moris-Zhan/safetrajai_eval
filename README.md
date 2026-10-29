# safetrajai_eval

離線 CCTV 交通監控的偵測、追蹤與衝突事件分析流程：

```
影片 ─▶ stage1：YOLO11 偵測 ─▶ class-aware BoTSORT／ByteTrack 追蹤 ─▶ 7 欄 MOT 軌跡
     ─▶ stage2：TrackEval（HOTA／CLEAR／Identity）─▶ Excel 報表
     ─▶ stage3：衝突事件分析（TTC／PET＋Homography）
```

另有一條 Re-ID 支線：從來源鏡頭蒐集裁切 → 品質把關 → manifest → 訓練 DINOv2 Re-ID 模型，供 BoTSORT 的逐類別 `use_reid` 使用。

固定 6 類，類別 ID 是跨模組契約（`utils/modules/safetrajai/classes.py`）：`person=0`、`car=1`、`scooter=2`、`small_truck=3`、`bus=4`、`truck=5`。

- 術語（鏡頭、序列、軌跡、身分、衝突事件…）：[`CONTEXT.md`](CONTEXT.md)
- 開發規則與完整指令表：[`CLAUDE.md`](CLAUDE.md)
- 詳細使用說明：[`Markdown/USAGE.md`](Markdown/USAGE.md)

---

## 目錄

1. [專案結構](#1-專案結構)
2. [安裝](#2-安裝)
3. [資料與模型總覽](#3-資料與模型總覽)
4. [Hugging Face 存取前置作業](#4-hugging-face-存取前置作業)
5. [追蹤評估資料：`safetrajai-eval`](#5-追蹤評估資料safetrajai-eval)
6. [Re-ID 資料與模型：`safetrajai-source`](#6-re-id-資料與模型safetrajai-source)
7. [執行](#7-執行)
8. [資料使用規則](#8-資料使用規則)

---

## 1. 專案結構

```
track_eval.py              主入口（相容 wrapper → safetrajai.cli.run_pipeline）
utils/modules/safetrajai/            主套件
  pipeline/                stage1/2/3 調度、單序列追蹤、Re-ID 裁切蒐集
  detection/               YOLO 推論＋逐類別信心門檻
  tracking/                class-aware ByteTrack／BoTSORT（override 方法凍結）
  reid/                    Re-ID 推論、資料品質工具、training/
  evaluation/              stage2：路徑組裝、MOT 轉換、ROI 忽略區域、TrackEval、Excel
  collision/               stage3：TTC／PET、熱區圖、homography/
  grid_search/             追蹤參數 grid search 共用邏輯
scripts/                   CLI，依功能分資料夾：pipeline/、grid_search/、homography/、visualization/、picker/、reid_data/、reid_train/
config/                    情境設定檔（tracking/、grid_search/、reid/、collision/）、datasets.yaml、roi/<camera>.yaml、homography_*
tests/                     smoke 測試、單元測試、golden 比對
utils/modules/_vendor/trackeval/     修改過的 vendored TrackEval（唯讀）
```

`dataset/`、`models/`、`runs/` 不進版控，由第 5、6 節從 Hugging Face 還原。

## 2. 安裝

只使用 conda env `track_eval`（不要用 uv 或另建 venv）。安裝一律加 `--no-deps`，避免 pip 把 CUDA 版 torch 換成 PyPI 版：

```bash
conda activate track_eval
pip install -e . --no-deps
pip install pytest huggingface_hub   # 測試與資料下載用
pytest                               # smoke 測試，幾秒內完成
```

依賴以 `pyproject.toml` 為準；`requirement.txt` 是舊腳本，不是權威來源。

## 3. 資料與模型總覽

| 資料 | 本機位置 | 取得方式 |
|---|---|---|
| C000025 人工 GT（評估用，6 段 15 秒序列） | `dataset/eval/cctv_all8/` | HF `Moris555/safetrajai-eval`，tag `gt-2026-09-23`（[第 5 節](#5-追蹤評估資料safetrajai-eval)） |
| 調參／保留子集（tune 4 段、holdout 2 段） | `dataset/eval/cctv_all8_subsets/` | 由 `subsets.yaml` 重建 symlink |
| C000025 完整影片 | `dataset/raw/videos/C000025/` | 同上 |
| Re-ID manifest（v1／v2／v3） | `dataset/reid/splits/<v>/` | HF `Moris555/safetrajai-source`，tag `reid-v3-2026-10-08`（[第 6 節](#6-re-id-資料與模型safetrajai-source)） |
| Re-ID 影像打包檔（v2／v3） | `~/.cache/safetrajai/reid_packs/<v>/` | 同上 |
| 排除／合併清單、裁切 metadata 索引 | `dataset/reid/` | 同上 |
| 品質報表 CSV、碎片特徵快取 | `dataset/reports/reid/` | 同上 |
| Re-ID 模型權重（6 組） | `models/class_reid_weights/<run>/` | 同上 |
| **YOLO 偵測權重** | `models/yolo/yolo11x_dennis.pt`、`models/yolo/yolo11x.pt` | **不在 HF**，需另外向專案維護者取得 |
| 來源鏡頭原始影片（9 支，約 10.8 GB） | `dataset/raw/videos/<camera>/` | **不在 HF**，需向來源重新取得 |
| 個別 Re-ID 裁切 JPEG（約 28.8 萬張） | `dataset/reid/crops/{class}/...` | **不上傳**；訓練改用打包檔 |

兩個 HF repo 依 `config/datasets.yaml` 的 `role` 分開存放：評估鏡頭（C000025）只在 `safetrajai-eval`，來源鏡頭只在 `safetrajai-source`，避免訓練程式誤讀評估資料。

## 4. Hugging Face 存取前置作業

兩個 dataset repo 都是 **private**，需要有讀取權限的帳號：

1. 到 <https://huggingface.co/settings/tokens> 建立 token（只下載的話 read 權限即可）。
2. 在**真正的終端機**登入（Claude Code 的 `!` 沒有 TTY，會出現 `EOFError`；也不要用 `--token hf_xxx`，token 會以明碼留在紀錄中）：

   ```bash
   conda activate track_eval
   hf auth login        # 舊版 huggingface_hub 為 huggingface-cli login
   hf auth whoami       # 確認帳號
   ```

下載**一律指定 tag**，不要讀 `main`，確保拿到的是固定版本。

## 5. 追蹤評估資料：`safetrajai-eval`

Repo：<https://huggingface.co/datasets/Moris555/safetrajai-eval>（tag `gt-2026-09-23`，約 150 MB、1,356 個檔案）

### 5.1 Repo 內容

```
annotations_all8/annotation_<seq>/        YOLO 格式：<seq>.mp4、<seq>.txt、data.yaml、images/、labels/
mot_annotations_all8/mot_annotation_<seq>/ MOT 格式：gt/gt.txt、gt/labels.txt、img1/*.jpg（只有 145630_15s_1 附 seqinfo.ini）
subsets.yaml                               tune（145630、150634 時段）／holdout（152639 時段）
videos/C000025/<video>.mp4                 評估鏡頭的完整影片
```

`<seq>` 例如 `C000025 中華三路 民生二路(東北角)_20251209145630_15s_1`（含空白與中文，路徑要加引號）。

### 5.2 下載並放回專案位置

在 repo 根目錄執行：

```bash
conda activate track_eval
python - <<'EOF'
from huggingface_hub import snapshot_download

REPO, REV = "Moris555/safetrajai-eval", "gt-2026-09-23"

# 人工 GT 與子集清單 → dataset/eval/cctv_all8/
snapshot_download(REPO, repo_type="dataset", revision=REV,
                  local_dir="dataset/eval/cctv_all8",
                  allow_patterns=["annotations_all8/**", "mot_annotations_all8/**", "subsets.yaml"])

# 完整影片 → dataset/raw/videos/C000025/
snapshot_download(REPO, repo_type="dataset", revision=REV,
                  local_dir="dataset/raw",
                  allow_patterns=["videos/**"])
EOF
```

完成後的結構與 `config/datasets.yaml` 的 `cctv_all8` 一致：

```
dataset/eval/cctv_all8/
  annotations_all8/annotation_<seq>/
  mot_annotations_all8/mot_annotation_<seq>/
  subsets.yaml
dataset/raw/videos/C000025/<video>.mp4
```

### 5.3 重建 tune／holdout 子集

本機的子集原本是 symlink，HF 不保存 symlink，所以依 `subsets.yaml` 重建（`scripts/reid_train/fusion_distance_stats.py` 預設讀 `dataset/eval/cctv_all8_subsets/tune/`）：

```bash
python - <<'EOF'
import os, yaml
from pathlib import Path

root = Path("dataset/eval/cctv_all8")
subsets = yaml.safe_load((root / "subsets.yaml").read_text(encoding="utf-8"))
for name, seqs in subsets.items():
    for kind, prefix in [("annotations_all8", "annotation_"), ("mot_annotations_all8", "mot_annotation_")]:
        dst_dir = Path("dataset/eval/cctv_all8_subsets") / name / kind
        dst_dir.mkdir(parents=True, exist_ok=True)
        for seq in seqs:
            src, dst = root / kind / f"{prefix}{seq}", dst_dir / f"{prefix}{seq}"
            assert src.is_dir(), src
            if not dst.exists():
                os.symlink(os.path.relpath(src, dst_dir), dst)
    print(name, len(seqs))
EOF
```

### 5.4 檢查

只比對檔案數，不要對資料目錄做全量 checksum（見 [第 8 節](#8-資料使用規則)）：

```bash
ls dataset/eval/cctv_all8/annotations_all8 | wc -l        # 6
ls dataset/eval/cctv_all8/mot_annotations_all8 | wc -l    # 6
ls dataset/eval/cctv_all8_subsets/{tune,holdout}/annotations_all8
```

## 6. Re-ID 資料與模型：`safetrajai-source`

Repo：<https://huggingface.co/datasets/Moris555/safetrajai-source>（tag `reid-v3-2026-10-08`，約 2.13 GB、91 個檔案）

`reid-v2`、`reid-v3`、`reid-v3-2026-10-08` 三個 tag 指向**同一份**完整快照（同時包含 v2 與 v3），不是各版本的篩選子集。

### 6.1 Repo 內容

| Repo 路徑 | 內容 | 還原到 |
|---|---|---|
| `dataset/reid/splits/{v1,v2,v3}/` | `manifest.csv`、`split_rules.yaml`、`summary.json`（v1 只有 metadata，沒有打包檔） | `dataset/reid/splits/` |
| `reid_packs/{v2,v3}/` | `crops.bin`、`index.csv`、`pack.json`（v3 約 624 MB，156,186 張） | `~/.cache/safetrajai/reid_packs/` |
| `dataset/reid/crops/_index/` | 39 份來源序列的框座標索引（干擾物取樣用；不含 JPEG） | `dataset/reid/crops/_index/` |
| `dataset/reid/{auto_exclusions,exclusions,fragment_merges}.csv` | 自動排除、人工排除、碎片合併清單 | `dataset/reid/` |
| `dataset/reports/reid/` | 標記與抽查 CSV、`fragment_features.npz` | `dataset/reports/reid/` |
| `models/class_reid_weights/<run>/` | `model.pt`＋`train_record.json`，共 6 組（tag 快照；`main` 上只有 `dinov2s_v3` 在原位，其餘 5 組在 `temp/`） | `models/class_reid_weights/` |
| `config/` | 9 支來源鏡頭的 ROI 與資料準備設定（上傳當時的快照） | **不覆蓋**，repo 內已有版控版本 |
| `inventory.json` | 每個檔案的路徑與位元組數 | 供下載後核對 |

資料集版本：

| 版本 | 列數 | 說明 |
|---|---:|---|
| `v1` | 174,344 | 審查前的基準，只供比較，無打包檔 |
| `v2` | 156,141 | 套用排除清單（22,164 個身分） |
| `v3` | 156,186 | v2＋碎片合併（22,223 個身分），**目前採用** |

模型：

| Run | 狀態 |
|---|---|
| `dinov2s_v3` | 唯一保留的模型（DINOv2 ViT-S/14，val mAP 0.783）；`fusion_distance_stats.py` 的預設權重，`config/tracking/botsort_reid_timm.yaml` 只在註解中列為候選，沒有任何類別載入 |
| `dinov2s_v2` | v2 對照組，追蹤上與 v3 平手；**已隔離** |
| `dinov2s_v3_seed1` | 換種子的雜訊對照；**已隔離** |
| `dinov2s_v3_distractor` | 實驗候選，未通過外觀分離度關卡；**已隔離** |
| `dinov2s_v3_lowres`、`dinov2s_v3_both` | 實驗候選，改以物體配對判定後通過外觀分離度關卡，但 `_both` 的追蹤驗收沒有任何類別通過；**已隔離** |

2026-10-12 起，已隔離的 5 組在本機與 HF `main` 都移到 `models/class_reid_weights/temp/`（檔案內容不變，原因與還原指令見 `temp/DO_NOT_USE.md`），不可用於情境設定。tag `reid-v3-2026-10-08` 仍是原本的佈局，下面的下載指令會把 6 組都放在原位；只需要目前模型時，請照 6.2 的說明縮小 `allow_patterns`。

val mAP 只是參考，不等於追蹤驗收；目前追蹤 config 的 6 類 ReID 仍全部關閉。

### 6.2 下載並放回專案位置

在 repo 根目錄執行：

```bash
conda activate track_eval
python - <<'EOF'
from pathlib import Path
from huggingface_hub import snapshot_download

REPO, REV = "Moris555/safetrajai-source", "reid-v3-2026-10-08"

# manifest、清單、metadata 索引、報表、模型權重 → 專案根目錄下的 dataset/、models/
snapshot_download(REPO, repo_type="dataset", revision=REV, local_dir=".",
                  allow_patterns=["dataset/**", "models/**"])

# 打包檔 → ~/.cache/safetrajai/reid_packs/{v2,v3}/
snapshot_download(REPO, repo_type="dataset", revision=REV,
                  local_dir=str(Path.home() / ".cache" / "safetrajai"),
                  allow_patterns=["reid_packs/**"])

# 清單與核對用
snapshot_download(REPO, repo_type="dataset", revision=REV,
                  local_dir="dataset/reid", allow_patterns=["inventory.json"])
EOF
```

- 只需要追蹤推論時，可把第一段的 `allow_patterns` 縮成 `["models/class_reid_weights/dinov2s_v3/**"]`，並略過打包檔。
- `snapshot_download` 會在 `local_dir` 下建立 `.cache/huggingface/` 存放下載 metadata，可以保留（重新下載時用來跳過已完成的檔案）。
- `config/` 刻意不下載：repo 內 `config/roi/`、`config/reid_*.yaml`、`config/reid/fragment_merge.yaml` 已有版控版本。若要確認上傳當時的設定，可另外下載到暫存目錄比對。

### 6.3 核對

依 `inventory.json` 只比對路徑與檔案大小（共 91 個檔案，不讀內容）：

```bash
python - <<'EOF'
import json
from pathlib import Path

inv = json.loads(Path("dataset/reid/inventory.json").read_text())
cache = Path.home() / ".cache" / "safetrajai"
bad = []
for rel, size in inv["files"].items():
    if rel.startswith("config/") or rel in ("README.md",):
        continue
    p = (cache / rel) if rel.startswith("reid_packs/") else Path(rel)
    if not p.is_file() or p.stat().st_size != size:
        bad.append(rel)
print("missing/mismatch:", bad or "none")
EOF
```

### 6.4 還原後的注意事項

- **不要再執行 `pack_manifest.py`**：打包檔已還原，而且個別裁切 JPEG 不在 HF，重新打包會失敗。`v3/pack.json` 記錄的 `reused_from: v2` 只是歷史，v3 打包檔本身是完整的。
- 訓練讀「打包檔＋manifest」；干擾物取樣另外讀 `dataset/reid/crops/_index/` 與 `config/roi/`，兩者都已還原。
- `train_record.json` 中的絕對路徑是原機器的紀錄，不影響載入。
- 需要讀取個別裁切的工具（`detect_contamination.py`、`generate_identity_review.py`、`extract_fragment_features.py`，以及用 `build_reid_split.py` 產生新版本）無法只靠 HF 資料執行，必須先取得來源影片，用 `config/reid/reid_crops.yaml` 重新蒐集裁切。
- 排除與合併清單是上傳當時的快照；用它們重算 v1／v2 不保證與凍結的 manifest 一致，請直接使用下載的 manifest。

## 7. 執行

所有指令都在 repo 根目錄、`track_eval` 環境中執行。

### 7.1 追蹤＋評估（需要第 5 節資料與 YOLO 權重）

```bash
python track_eval.py --config config/tracking/bytetrack.yaml         # stage1 追蹤 + stage2 TrackEval
python track_eval.py --config config/tracking/botsort_noreid.yaml
python track_eval.py --config config/tracking/botsort_reid_timm.yaml # Re-ID 驗收基準（6 類 ReID 都關閉）
python track_eval.py --config config/collision/collision_detect.yaml  # stage1 + stage3（TTC／PET）
python scripts/pipeline/run_evaluation.py --config <config>           # 只對既有 stage1 輸出重跑 stage2
```

輸出在 `runs/track_eval/`。要縮小範圍，在 config 的 `paths:` 加 `SOURCE_VIDEO_DIR` 指定單一序列。調整追蹤參數用 `scripts/grid_search/grid_search_*.py`（`--dry-run` 先看組合數）。

#### 評測資料集

追蹤評估一律使用 `config/datasets.yaml` 的 `cctv_all8`（`role: eval`，附人工 GT，只用於評估，不能進任何訓練資料）：

| 項目 | 內容 |
|---|---|
| dataset id | `cctv_all8`（情境 config 以 `paths.dataset: cctv_all8` 引用） |
| 本機路徑 | `dataset/eval/cctv_all8/`：`annotations_all8/`（YOLO 格式＋序列影片）、`mot_annotations_all8/`（MOT GT） |
| 來源 | HF `Moris555/safetrajai-eval`，tag `gt-2026-09-23`（見第 5 節） |
| 鏡頭 | C000025（中華三路 民生二路，東北角），2025-12-09 拍攝 |
| 序列 | 6 段 15 秒片段，約 5 fps（每段 71～75 幀），640×480 |
| 類別 | 6 類（person、car、scooter、small_truck、bus、truck） |

6 段序列依時段分成兩個子集（`subsets.yaml`，本機以 symlink 放在 `dataset/eval/cctv_all8_subsets/`，重建方式見 5.3）：

| 子集 | 時段 | 序列 | 幀數 | GT 列數 | 用途 |
|---|---|---|---:|---:|---|
| `tune` | 145630、150634 | `…_145630_15s_1`、`_15s_2`、`…_150634_15s_1`、`_15s_2` | 292 | 6,851 | 調參與驗收比較，下表的基準分數來自這裡 |
| `holdout` | 152639 | `…_152639_15s_1`、`_15s_2` | 149 | 3,550 | 保留序列，不用於調參；`fusion_distance_stats.py` 會拒絕讀取 |

- GT 列數是 `gt/gt.txt` 的總列數，含 ROI 外的框。stage2 會把 ROI 外的 GT 標成 ignore 再評估，所以下表的 GT 數比這裡少。
- **bus 的軌跡驗證只能靠 holdout**：GT 中的 bus 只有 2 條軌跡。tune 的 `…_150634_15s_2`（id 29，62 列）全部在 ROI 外，評估時被忽略；holdout 的 `…_152639_15s_2`（id 21，42 列，由上往下駛過畫面）有 21 列的中心點在 ROI 內，是唯一能實際評估 bus 的資料。
- 用 `SOURCE_VIDEO_DIR` 指向子集目錄，就能只評估該子集，例如 `./dataset/eval/cctv_all8_subsets/tune/`。不設定時會評估全部 6 段序列。

#### 目前的追蹤基準分數

Re-ID 追蹤驗收的基準（2026-10-12）：設定 `runs/reid_domain/accept/configs/baseline_{tune,holdout}.yaml`，由 `acceptance_grid.py configs --name baseline` 以 `config/tracking/botsort_reid_timm.yaml` 為底產生（追蹤參數相同，6 類 ReID 都關閉）。偵測模型為 `yolo11x_dennis.pt`，兩個子集的設定只差 `SOURCE_VIDEO_DIR` 與輸出位置。逐序列 Excel 在 `runs/reid_domain/accept/baseline/{tune,holdout}/`，彙總在 `runs/reid_domain/accept/grid_tune.md` 與 `baseline_holdout.md`。

**tune 子集**（4 段序列）：

| 類別 | HOTA | AssA | IDSW | IDP | GT |
|---|---:|---:|---:|---:|---:|
| person | 87.10 | 87.15 | 0 | 0.978 | 178 |
| car | 88.82 | 92.19 | 2 | 0.996 | 1,355 |
| scooter | 76.68 | 80.73 | 4 | 0.967 | 865 |
| small_truck | 88.70 | 90.66 | 0 | 1.000 | 70 |
| bus | — | — | 0 | — | 0 |
| truck | 93.54 | 93.86 | 0 | 1.000 | 81 |

**holdout 子集**（2 段序列；bus 只能在這裡驗證）：

| 類別 | HOTA | AssA | IDSW | IDP | GT |
|---|---:|---:|---:|---:|---:|
| person | 92.10 | 92.22 | 0 | 1.000 | 56 |
| car | 83.75 | 85.96 | 1 | 0.992 | 729 |
| scooter | 76.41 | 82.85 | 2 | 0.948 | 459 |
| small_truck | 96.72 | 96.80 | 0 | 1.000 | 76 |
| **bus** | **94.97** | **94.97** | **0** | **1.000** | **21** |
| truck | — | — | 0 | — | 0 |

- tune 的 bus 為 0：tune 子集唯一的 bus（`…_150634_15s_2`，GT id 29，62 列）是停在畫面上緣（中心 y 約 39～54、框約 16×23 px）的遠處車輛，中心點全部在 ROI 外，被 stage2 標成 ignore。GT 欄是 TrackEval 輸出的 GT 數（ROI 處理後）。
- holdout 的 bus：`…_152639_15s_2` 的 GT id 21，原有 42 列，其中 21 列的中心點在 ROI 外被忽略，剩下 21 列全部配對成功（TP 21），沒有 ID switch。只有 1 條軌跡、21 幀，分數只能當作單一案例的訊號，不足以代表 bus 類別的整體表現。holdout 沒有 truck 的 GT。
- holdout 的基準最早在 2026-09-28 由 change `train-reid-timm` 執行過（見該 change 的 `verification.md` 3.1）；2026-10-12 重跑的結果逐類別完全相同。holdout 只作為最終確認，不可拿來調參。
- `bytetrack.yaml` 目前沒有保存的評估結果；`botsort_noreid.yaml` 的結果見下一小節。
- 這是 stage2 的 ROI 忽略區域處理後的分數。新設定要和這個基準比較時，請在同一個子集、同一個偵測模型上重跑。

#### 開啟 ReID 的比較（2026-10-12，全部 6 段序列）

以主流程 `track_eval.py` 在 `cctv_all8` 全部 6 段序列（tune＋holdout）上跑兩組「ReID 關／開」對照。設定放在 `runs/reid_all6/configs/`，不改動 `config/`；彙總在 `runs/reid_all6/summary_timm.md`、`summary_legacy.md`。

| 組別 | 來源設定 | 差異 |
|---|---|---|
| `timm_off` | `config/tracking/botsort_reid_timm.yaml` 原樣 | 主 baseline，6 類 ReID 關閉 |
| `timm_v3_on` | 同上 | 只改 6 類 `use_reid: true`、`model: models/class_reid_weights/dinov2s_v3/model.pt`；融合維持原本設定（沒有閘門、沒有否決） |
| `legacy_off` | `config/tracking/botsort_noreid.yaml` | 舊情境（`frame_rate: 30`、`gmc_method: ecc`、`track_buffer` 90／60） |
| `legacy_on` | `config/tracking/botsort_reid.yaml` | 舊情境開 ReID（YOLO 當編碼器），`track_buffer` 也改成 120，不是純 ReID 的差異 |

HOTA 與 IDSW：

| 類別 | timm_off | timm_v3_on | legacy_off | legacy_on |
|---|---:|---:|---:|---:|
| person | 88.30 | 88.30 | 88.36 | 88.36 |
| car | 87.05 | 87.06 | 85.68 | 85.68 |
| scooter | 76.58 | 76.52 | 76.24 | 76.09 |
| small_truck | 92.87 | 92.87 | 88.48 | 89.54 |
| bus | 94.97 | 94.97 | 95.14 | 95.14 |
| truck | 93.54 | 93.54 | 92.93 | 92.93 |
| IDSW（car／scooter／small_truck） | 3／6／0 | 3／6／0 | 14／8／4 | 14／8／3 |
| 每幀追蹤時間 | 0.008 s | 0.077 s | 0.839 s | 1.181 s |

結論：**開啟 ReID 沒有讓分數超過 baseline，`botsort_reid_timm.yaml` 維持 6 類 ReID 關閉。**

- `dinov2s_v3`：6 類的 IDSW、IDP、TP 完全不變，依 spec `reid-model` 全部不通過；HOTA 只有 ±0.06 的雜訊級變動，追蹤時間增加約 9.5 倍。原本的融合只在 IoU 已重疊的配對上使用外觀距離，而這些配對靠 IoU 本來就會配對正確。
- 舊設定：開 ReID 後 scooter 的 IDP、AssA 下降；small_truck 的 IDSW 4 → 3，依規則判為通過，但只差 1 次，而且仍不如主 baseline（IDSW 0、HOTA 92.87），判定也用到了 holdout，所以不作為啟用依據。
- 舊設定整體較差，主要來自為 30 fps 設計的追蹤參數（C000025 約 5 fps），不是 ReID 造成的。
- 剩下的 ID switch（car 3、scooter 6）屬於遮擋或長時間遺失。先前用閘門擴大外觀作用範圍反而讓 IDSW 增加，瓶頸在模型與資料（見 `CLAUDE.md` 的 Re-ID 一節）。下一步應先用疊圖影片檢查這些 ID switch 的情境，而且只在 tune 子集上分析。

### 7.2 調參（grid search）

針對**單一類別**掃描追蹤參數。這是參數調整，不是模型訓練；分數是 `HOTA.AssA_mean`（關聯指標，不等於 HOTA）。

| 腳本 | 掃描對象 | 預設設定檔 | 前提 |
|---|---|---|---|
| `scripts/grid_search/grid_search_bytetrack.py` | ByteTrack | `config/grid_search/grid_search_bytetrack.yaml` | 第 5 節資料、YOLO 權重、長時間 |
| `scripts/grid_search/grid_search_botsort.py` | BoTSORT（ReID 關閉） | `config/grid_search/grid_search_botsort.yaml` | 同上 |
| `scripts/grid_search/grid_search_botsort_reid.py` | BoTSORT（ReID 開啟） | `config/tracking/botsort_reid.yaml` | 同上，另需 Re-ID 權重 |

```bash
python scripts/grid_search/grid_search_bytetrack.py --target-class scooter --dry-run   # 只印組合數，不執行
python scripts/grid_search/grid_search_botsort.py --target-class scooter --tracker botsort --max-videos 1   # 最小驗證
python scripts/grid_search/grid_search_botsort_reid.py --target-class person --resume --result-csv <csv>    # 續跑
```

- `--params-subset a,b` 只搜尋指定參數，其餘沿用目前設定；`--resume` 只跳過完全相同、已完成的組合。
- 全量掃描會跑完整的 stage1＋stage2，耗時，也會寫入 `runs/`，執行前先確認範圍與輸出位置。
- 每個 trial 結束後都會還原 `CLASS_TRACKING_PARAMS`（成功、跳過或例外皆然）。

### 7.3 衝突事件與 Homography（stage3）

程式碼中的 `collision` 指**衝突事件**（TTC／PET 低於門檻的接近事件），不是實際碰撞。

| 設定檔 | 階段 | 說明 |
|---|---|---|
| `config/collision/collision_detect.yaml` | stage1＋stage3 | TTC／PET 事件，輸出在 `collision_output/<seq>/` |
| `config/collision/conflict_analysis.yaml` | stage1＋stage3 | 同上，另外計算公尺單位的衝突指標（輸出在 `collision_output/<seq>/conflict/`）；只在 `scale_check: passed` 的鏡頭計算，其餘序列只寫 `summary.json` 並記錄略過原因 |

```bash
python track_eval.py --config config/collision/conflict_analysis.yaml
```

計數線流量統計是選用的 `counting:` 區塊（預設關閉）：先用 `scripts/picker/make_picker.py --mode counting-lines` 畫線並存成 `config/counting_lines/<camera>.yaml`，再在情境 config 設 `counting: {enabled: true}`；輸出在 `collision_output/<seq>/counting/`。步驟與規則見 [`Markdown/counting/counting-guide.md`](Markdown/counting/counting-guide.md)。

衝突指標只是技術訊號，不能當成正式的安全結論。

**Homography 鏡頭設定**（`config/homography/<camera>.yaml`）依下列順序操作；腳本都在 repo 根目錄執行：

| 步驟 | 腳本 | 用途 | 前提 |
|---|---|---|---|
| 驗證 | `scripts/homography/validate_homography.py` | 批次驗證既有矩陣，輸出到 `models/homography/validation_output/`（**會覆寫該目錄既有輸出**；沒有參數，不要用 `--help` 試探） | `models/homography/`；比對影片需要來源影片 |
| 1 遷移 | `scripts/homography/migrate_homography_config.py` | 舊的 tag 鍵資料 → 鏡頭鍵設定，只寫新檔，衝突時不寫；先用 `--dry-run` | 無 |
| 2 經緯度 | `scripts/homography/geocode_cameras.py` | 查詢鏡頭經緯度，先產生候選表 `models/homography/geocode_candidates.csv`，再用 `--apply [--accept <cam...>]` 寫入 | 連網（OSM Overpass）；Google 備援需 `GOOGLE_MAPS_API_KEY` 或 `GOOGLE_MAPS_DEMO_API_KEY`，`--no-google` 可關閉備援 |
| 3 比例尺 | `scripts/homography/check_homography_scale.py` | 以已知長度地物（如車道寬）核對比例尺，結果寫入 `scale_check`；`--dry-run` 只印結果 | 連網與 Google 金鑰（影像比對；`--skip-image-match` 可略過） |
| 底圖 | `scripts/homography/download_satellite.py` | 下載衛星圖，`--purpose verify\|basemap` | 連網；`basemap` 只接受 `GOOGLE_MAPS_API_KEY`，Demo Key 只能用於 `verify` |

- 金鑰從環境變數或 repo 根目錄的 `.env`（已被 git 忽略）讀取，程式不會印出金鑰值；README 與紀錄中也不要貼金鑰。
- 這三個連網腳本都是手動執行，不在 pipeline 中。Google Static Maps 的條款可能限制離線保存與衍生使用，底圖不納入 git。
- 沒有 `scale_check: passed` 的鏡頭，單位維持像素，不會退回預設比例。

### 7.4 Re-ID 資料

資料流程依序是 **蒐集 → 標記 → 自動排除 → 抽查 → 人工排除 → 碎片合併 → manifest**。各步驟的細節、輸出目錄與資料量見 [`scripts/reid_data/README.md`](scripts/reid_data/README.md)。腳本都在 repo 根目錄執行；寫入類的腳本不帶 `--apply` 時只列印計畫（dry-run）。

| 步驟 | 腳本／設定 | 用途 | 前提 |
|---|---|---|---|
| 1 蒐集 | `config/reid/reid_crops.yaml`（`python track_eval.py --config config/reid/reid_crops.yaml`） | 只跑 stage1，從來源鏡頭蒐集裁切到 `dataset/reid/crops/` | 來源影片（第 3 節，不在 HF）、YOLO 權重、長時間 |
| 2 標記 | `scripts/reid_data/detect_contamination.py` | 以 DINOv2 標記污染身分，輸出到 `dataset/reports/reid/` | 個別裁切 JPEG；第一次使用從 HF Hub 下載 DINOv2（連網）；HF 還原的資料不足以執行 |
| 3 自動排除 | `scripts/reid_data/apply_auto_exclusions.py [--apply]` | 由標記整份重算 `auto_exclusions.csv` | 只讀報表與 CSV |
| 4 抽查 | `scripts/reid_data/generate_identity_review.py --audit-only` | 產生抽查頁與 `audit_sample.csv`（勾選＝排除） | 個別裁切 JPEG；HF 還原的資料不足以執行 |
| 4 抽查 | `scripts/reid_data/summarize_audit.py --confirm-csv <csv...>` | 每支鏡頭的不合格率（> 10% 要處理） | 只讀 CSV |
| 5 人工排除 | `scripts/reid_data/apply_review_exclusions.py --confirm-csv <csv> [--apply]` | 合併進 `exclusions.csv` | 只讀 CSV |
| 6 碎片合併 | `scripts/reid_data/extract_fragment_features.py` | 只讀碎片候選端點裁切的 DINOv2 特徵 | 個別裁切 JPEG；HF 還原的資料不足以執行 |
| 6 碎片合併 | `scripts/reid_data/build_fragment_merges.py [--sample N --seed S \| --calibrate <csv> \| --apply]` | 產生碎片合併清單 `fragment_merges.csv`；`--sample` 會複製裁切到確認頁，所以需要個別裁切 JPEG | `--calibrate`／`--apply` 只讀特徵與 CSV |
| 7 manifest | `scripts/reid_data/build_reid_split.py --version <v> [--merges dataset/reid/fragment_merges.csv]` | 兩份排除清單都套用，依 `config/reid/reid_split.yaml` 切分 train／val | 產生新版本需要 `crops/`，HF 還原的資料不足以執行（見 6.4）；直接使用下載的 manifest 即可 |

蒐集用的其他情境設定：

| 設定檔 | 說明 |
|---|---|
| `config/reid/botsort_save_reid_crops.yaml` | BoTSORT＋`save_reid_crops`，只跑 stage1。資料來源是**評估資料集** `cctv_all8`，裁切不得用於訓練 |
| `config/reid/botsort_save_reid_crops_video.yaml` | 同上，來源改為 `reid_sources`、`frame_rate: 30`、輸出放在 `runs/`；只保留作為對照，正式訓練資料請用 `config/reid/reid_crops.yaml` |
| `config/reid/reid_split.yaml` | 資料集版本的切分規則（哪些鏡頭＋日期為 val、每身分最少／最多張數），由 `build_reid_split.py` 讀取 |

### 7.5 Re-ID 訓練與驗證（需要第 6 節資料）

```bash
# val 評估既有模型（mAP／Rank-1，只作參考）
python scripts/reid_train/eval_reid.py \
  --weights models/class_reid_weights/dinov2s_v3/model.pt --version v3

# 重新訓練（30 epoch 約 25～45 分鐘，建議背景執行）
python scripts/reid_train/train_reid.py --version v3 \
  --output models/class_reid_weights/<new_run>

# 調參序列 GT 上的外觀距離分布（需要第 5 節的 tune 子集；保留序列會被拒絕）
python scripts/reid_train/fusion_distance_stats.py --out runs/reid_stats/<name>.json
```

新模型要上線，須依 spec `reid-model` 的條件在調參序列上與驗收基準比較：結果不變差，而且 IDSW 減少或 IDP 上升。只有 val mAP 上升不算通過。

其餘兩支腳本：

| 腳本 | 用途 | 前提 |
|---|---|---|
| `scripts/reid_train/pack_manifest.py --version <v> [--reuse-version <v0>]` | 依 manifest 把裁切打包成本機快取 `~/.cache/safetrajai/reid_packs/<v>/`（只從掛載磁碟各讀一次） | **HF 還原後不要執行**：打包檔已還原，而且個別裁切 JPEG 不在 HF，重新打包會失敗（見 6.4） |
| `scripts/reid_train/acceptance_grid.py {configs,summarize}` | 追蹤驗收：`configs` 以 `config/tracking/botsort_reid_timm.yaml` 為底，產生每個子集的驗收設定；`summarize` 彙總逐類別結果並依 spec 判定（輸出 CSV＋Markdown） | 第 5 節的 tune／holdout 子集、YOLO 權重；執行驗收設定要跑 stage1＋stage2，耗時 |

```bash
# 產生驗收設定 → 以 track_eval.py 執行 → 彙總
python scripts/reid_train/acceptance_grid.py configs --name <run> --set CLASS.KEY=VALUE
python track_eval.py --config runs/reid_domain/accept/configs/<run>_tune.yaml > runs/reid_domain/accept/<run>/tune/run.log 2>&1
python scripts/reid_train/acceptance_grid.py summarize --baseline baseline --runs <run> --out <prefix>
```

holdout 子集只作最終確認，不可拿來調參。目前 `config/tracking/botsort_reid_timm.yaml` 的 6 類 ReID 都關閉。

### 7.6 工具

| 腳本 | 用途 | 前提 |
|---|---|---|
| `scripts/pipeline/run_pipeline.py` | 與 `python track_eval.py` 等價的入口（同一個 `safetrajai.cli.run_pipeline.main()`），參數相同：`--config <path>` | 同 7.1 |
| `scripts/picker/make_roi_picker.py --camera <camera>` | 產生 ROI 離線點選頁（單檔 HTML，首幀內嵌），輸出到 `runs/roi_picker/`；在本機瀏覽器點出多邊形頂點後，把產生的 YAML 放回 `config/roi/<camera>.yaml`。完整步驟見 [`Markdown/roi/roi-guide.md`](Markdown/roi/roi-guide.md) 第 2 節 | `dataset/raw/videos/<camera>/` 的來源影片；不需要網路 |
| `scripts/picker/make_picker.py --camera <camera> --mode <mode>` | 產生離線點選頁（單檔 HTML，影像內嵌），輸出到 `runs/picker/<camera>_<mode>.html`；`--mode` 為 `homography`（成對點選 ≥ 4 組）、`checkpoints`（驗證點）、`counting-lines`（兩點成一條計數線）、`camera-position`（衛星圖上的相機位置）。頁面即時產生 YAML 可複製或下載，工具本身不修改任何 config。步驟見 [`Markdown/picker/picker-guide.md`](Markdown/picker/picker-guide.md) | CCTV 首幀來自 `dataset/raw/videos/<camera>/`；`homography`／`checkpoints`／`camera-position` 另需 `models/homography/satellite_images/` 的衛星圖（缺少時依訊息執行 `download_satellite.py`） |
| `scripts/visualization/visualize_gt_on_video.py` | 把**人工 GT**（`gt/gt.txt`，可帶 class 欄）疊在序列影片上 | 第 5 節資料 |
| `scripts/visualization/visualize_gt_on_class_video.py` | 把**追蹤結果**（`trackers/mot_challenge/<benchmark>-<split>/YOLO11x__<class>/data/<seq>.txt`，每類一檔）合併疊在序列影片上，用來檢查 ID switch 與跨類別誤配 | 已跑過 stage1／stage2 的輸出 |

```bash
python scripts/picker/make_roi_picker.py --camera C000054
python scripts/picker/make_picker.py --camera C000054 --mode homography
python scripts/visualization/visualize_gt_on_video.py --trackeval_root <path> --video_path <path> --output_video <path>
python scripts/visualization/visualize_gt_on_class_video.py --trackeval_root <path> --video_path <path> --output_video <path>
```

兩支疊圖腳本另有 `--color_by {class,id}`、`--class_names`、`--show_class_filter`、`--no_id`、`--no_legend` 等顯示選項，見各自的 `--help`。

### 7.7 測試

測試分三層。前兩層都在 `track_eval` 環境、repo 根目錄執行 `pytest`，不需要資料集、權重或 GPU；第三層才需要完整資料。

| 層級 | 指令 | 耗時 | 前提 | 何時跑 |
|---|---|---|---|---|
| smoke | `pytest tests/test_smoke.py` | 秒級 | 安裝即可 | 任何結構調整後 |
| 單元 | `pytest`（含 smoke 與下列單元測試） | 秒級～分鐘 | 不需要資料／權重 | 提交前 |
| golden 比對 | `tests/golden/run_all.sh <run>`，再 `python tests/golden/compare.py <基準> <run>` | 約 16 分鐘 | 完整資料（第 5 節）、YOLO 權重 | 改動演算法或 stage 行為後 |

**smoke**（`tests/test_smoke.py`）：全部模組可 import、全部 config 可載入、類別 ID 契約、tracker override 方法的 AST 不變性、`sys.path` 掃描、有 argparse 的腳本 `--help` 可執行。其中「被隔離的 Re-ID 權重仍可載入」在權重檔不存在時會被略過（`skipif`），不是失敗。另有少數參數組合不適用（例如不迭代類別的腳本、不是追蹤情境的設定）而被略過，`pytest` 結尾出現 `skipped` 屬正常。

**單元測試**依領域分組：

| 領域 | 測試檔 |
|---|---|
| Re-ID 資料與訓練 | `tests/test_reid_*.py`（manifest、切分、污染標記、自動排除、抽查、碎片合併、編碼、融合、外觀距離統計、訓練、驗收） |
| Homography／georef | `tests/test_homography_*.py`、`test_geocode_cameras.py`、`test_satellite_download.py`、`test_scale_match.py`、`test_secrets.py` |
| 衝突指標 | `tests/test_conflict_metrics.py` |
| ROI／資料集路徑／忽略區域 | `tests/test_roi_and_layout.py`、`test_ignore_region.py` |
| tracker 保留期限 | `tests/test_tracker_retention.py` |
| 追蹤設定與 grid search | `tests/test_track_config.py`（設定載入與驗證、全域參數更新、查表退回預設、載入順序契約）、`test_grid_search_core.py`（參數組合、AssA 萃取、trial 參數修補與還原、resume） |
| 文件 | `tests/test_readme_coverage.py`：`scripts/` 的每支腳本與 `config/` 的每份設定都必須出現在本 README；新增入口卻沒寫文件時會失敗，確實不適合列入的才寫進該檔的 `EXEMPT`（要附理由） |

**golden 比對**確認重構或調整後的輸出與基準逐位元一致：`run_all.sh` 以 6 個情境＋grid search＋homography 驗證跑一次完整流程，輸出在 `runs/refactor_golden/<run>/`（先用 `--dry-run` 看會執行哪些步驟）。基準自比的容忍度為 0，任何非零差異都代表行為改變；基準名稱、更換基準的原因與預期差異見 [`tests/golden/TOLERANCE.md`](tests/golden/TOLERANCE.md)。全量執行耗時，也會寫入 `runs/`，執行前先確認範圍。

## 8. 資料使用規則

- **人工 GT 不可修改**；讀評估資料一律指定 tag `gt-2026-09-23`。
- **評估鏡頭 C000025 不可進任何訓練資料**；與它同一路口的 C000026 也刻意排除。smoke 測試會檢查 `datasets.yaml` 中 eval 與 source 的鏡頭不重疊。
- **禁止大量 I/O**：`dataset/`、`models/`、`runs/` 可能位於掛載磁碟。不要做全量 `sha256sum`／`du`、`find … -exec cat` 或 `rsync --checksum`；核對只比對路徑清單、檔案數或大小。
- **隔離不刪除**：有疑慮的資料或權重用 `mv` 移到同目錄的 `temp/`，並記錄在 `temp/DO_NOT_USE.md`；config 不得引用 `temp/`。
- 上傳流程、HF 限制與 repo 規劃見 [`Markdown/workflow/HF_UPLOAD_WORKFLOW.md`](Markdown/workflow/HF_UPLOAD_WORKFLOW.md)。

## License

見 [`LICENSE`](LICENSE)。
