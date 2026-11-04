# 從 Hugging Face 還原資料與模型

新機器還原 `dataset/`、`models/` 的步驟：存取權限 → 評估資料 `safetrajai-eval` → Re-ID 資料與模型 `safetrajai-source`。資料與模型的總覽表在 README 的「資料與模型」一節。

## 1. Hugging Face 存取前置作業

兩個 dataset repo 都是 **private**，需要有讀取權限的帳號：

1. 到 <https://huggingface.co/settings/tokens> 建立 token（只下載的話 read 權限即可）。
2. 在**真正的終端機**登入（Claude Code 的 `!` 沒有 TTY，會出現 `EOFError`；也不要用 `--token hf_xxx`，token 會以明碼留在紀錄中）：

   ```bash
   conda activate track_eval
   hf auth login        # 舊版 huggingface_hub 為 huggingface-cli login
   hf auth whoami       # 確認帳號
   ```

下載**一律指定 tag**，不要讀 `main`，確保拿到的是固定版本。

## 2. 追蹤評估資料：`safetrajai-eval`

Repo：<https://huggingface.co/datasets/Moris555/safetrajai-eval>（tag `gt-2026-09-23`，約 150 MB、1,356 個檔案）

### 2.1 Repo 內容

```
annotations_all8/annotation_<seq>/        YOLO 格式：<seq>.mp4、<seq>.txt、data.yaml、images/、labels/
mot_annotations_all8/mot_annotation_<seq>/ MOT 格式：gt/gt.txt、gt/labels.txt、img1/*.jpg（只有 145630_15s_1 附 seqinfo.ini）
subsets.yaml                               tune（145630、150634 時段）／holdout（152639 時段）
videos/C000025/<video>.mp4                 評估鏡頭的完整影片
```

`<seq>` 例如 `C000025 中華三路 民生二路(東北角)_20251209145630_15s_1`（含空白與中文，路徑要加引號）。

### 2.2 下載並放回專案位置

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

### 2.3 重建 tune／holdout 子集

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

### 2.4 檢查

只比對檔案數，不要對資料目錄做全量 checksum（見 README 的「資料使用規則」）：

```bash
ls dataset/eval/cctv_all8/annotations_all8 | wc -l        # 6
ls dataset/eval/cctv_all8/mot_annotations_all8 | wc -l    # 6
ls dataset/eval/cctv_all8_subsets/{tune,holdout}/annotations_all8
```

## 3. Re-ID 資料與模型：`safetrajai-source`

Repo：<https://huggingface.co/datasets/Moris555/safetrajai-source>（tag `reid-v3-2026-10-08`，約 2.13 GB、91 個檔案）

`reid-v2`、`reid-v3`、`reid-v3-2026-10-08` 三個 tag 指向**同一份**完整快照（同時包含 v2 與 v3），不是各版本的篩選子集。

### 3.1 Repo 內容

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

2026-10-12 起，已隔離的 5 組在本機與 HF `main` 都移到 `models/class_reid_weights/temp/`（檔案內容不變，原因與還原指令見 `temp/DO_NOT_USE.md`），不可用於情境設定。tag `reid-v3-2026-10-08` 仍是原本的佈局，下面的下載指令會把 6 組都放在原位；只需要目前模型時，請照 3.2 的說明縮小 `allow_patterns`。

val mAP 只是參考，不等於追蹤驗收；目前追蹤 config 的 6 類 ReID 仍全部關閉。

### 3.2 下載並放回專案位置

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

### 3.3 核對

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

### 3.4 還原後的注意事項

- **不要再執行 `pack_manifest.py`**：打包檔已還原，而且個別裁切 JPEG 不在 HF，重新打包會失敗。`v3/pack.json` 記錄的 `reused_from: v2` 只是歷史，v3 打包檔本身是完整的。
- 訓練讀「打包檔＋manifest」；干擾物取樣另外讀 `dataset/reid/crops/_index/` 與 `config/roi/`，兩者都已還原。
- `train_record.json` 中的絕對路徑是原機器的紀錄，不影響載入。
- 需要讀取個別裁切的工具（`detect_contamination.py`、`generate_identity_review.py`、`extract_fragment_features.py`，以及用 `build_reid_split.py` 產生新版本）無法只靠 HF 資料執行，必須先取得來源影片，用 `config/reid/reid_crops.yaml` 重新蒐集裁切。
- 排除與合併清單是上傳當時的快照；用它們重算 v1／v2 不保證與凍結的 manifest 一致，請直接使用下載的 manifest。
