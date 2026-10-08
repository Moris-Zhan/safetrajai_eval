# Hugging Face 資料上傳流程（HF Private Repo Upload Workflow）

> 整理日期：2026-10-08
> 依據：2026-10-08 把評估 GT 上傳到 HF private repo 的過程與決策。GitHub 的程式碼推送見 `Markdown/GITHUB_PUSH_WORKFLOW.md`。

---

## 0. 現況

| 項目 | 設定 |
|---|---|
| HF 帳號 | `Moris555`（免費方案，private 儲存空間上限 **100 GB**） |
| 已上傳 | `Moris555/safetrajai-eval`（**private** dataset repo）：C000025 人工 GT，commit `cc5fbd7`，tag `gt-2026-09-23` |
| 本機 CLI | `track_eval` env 的 `huggingface_hub` 0.36.2：`/root/miniconda3/envs/track_eval/bin/hf` |
| 目前使用的 token | `Moris555` 的 fine-grained token，需要 `repo.write` 權限 |
| 其他帳號 | `moris1234fff` 的 token 是別的專案用的，仍保留在 `hf auth list` 中，要用時以 `hf auth switch` 切換 |

---

## 1. 哪些資料上傳、哪些不上傳

| 資料 | 大小 | 決定 | 原因 |
|---|---:|---|---|
| `dataset/eval/cctv_all8/annotations_all8`、`mot_annotations_all8`（C000025 人工 GT） | 0.14 GB | ✅ **已上傳** | 人工標註無法重新產生，而且是所有驗收的基準 |
| `dataset/eval/cctv_all8_subsets/`（tune／holdout） | — | ✅ 改成 `subsets.yaml` | 本機是 symlink，HF repo 不保存 symlink |
| `dataset/raw/videos/C000025/` | 11 MB | ✅ 已上傳（放在 eval repo） | 評估鏡頭的完整影片 |
| `dataset/eval/cctv_all8/temp/` | 0.32 GB | ❌ 不上傳 | 已隔離（DO_NOT_USE） |
| `dataset/raw/videos/C000026/` | 0.07 GB | ❌ 不上傳 | 與評估鏡頭同一路口，刻意排除（使用者決定） |
| `dataset/raw/videos/` 9 支來源鏡頭 | 10.8 GB | ⏸ 先不上傳 | 可以重新向來源取得 |
| `dataset/reid/`（manifest、打包檔、排除清單）與模型權重 | v3 打包檔約 0.62 GB | ✅ **2026-10-08 已授權上傳目前快照** | 已備份 v2／v3 packs 與六組模型；change 的文件收尾仍未完成 |
| `dataset/reports/reid/` | 0.2 GB | ✅ CSV／fragment features 已上傳 | 大多可以重新產生 |
| `dataset/reid/crops/`（28.8 萬張裁切） | — | ❌ **不逐檔上傳** | 超過 HF 每個 repo 的檔案數限制，而且等於把掛載磁碟整個讀一遍 |

資料授權：CCTV 畫面允許存放在 HF（2026-10-08 使用者確認）。

---

## 2. Repo 規劃

評估資料與訓練資料**放在不同的 repo**，跟本機的 `role: eval`／`role: source` 一致，避免訓練程式誤讀評估鏡頭：

```
Moris555/safetrajai-eval     ← C000025 人工 GT、subsets.yaml、C000025 完整影片（已建立）
Moris555/safetrajai-source   ← 之後的 Re-ID manifest、打包檔、模型權重（已建立；2026-10-08 上傳完成）
```

`safetrajai-eval` 的結構：

```
annotations_all8/annotation_<seq>/...        (900 個檔案)
mot_annotations_all8/mot_annotation_<seq>/... (454 個檔案：gt/gt.txt、img1/*.jpg、seqinfo.ini)
subsets.yaml                                  (tune 4 支、holdout 2 支)
videos/C000025/<seq>.mp4
```

---

## 3. HF 的限制

| 項目 | 限制 |
|---|---|
| Private 儲存空間 | 免費 100 GB；PRO（US$9／月）1 TB，超出按 US$18／TB／月計費 |
| 每個 repo 的檔案數 | 建議少於 10 萬 |
| 每個資料夾的檔案數 | 最多 1 萬 |
| 單一檔案 | 建議小於 200 GB，硬上限 500 GB |

大量小檔（例如 Re-ID 裁切）要先打包成 tar 分片（WebDataset），或直接上傳 `pack_manifest.py` 產生的打包檔，不要逐檔上傳。

來源：[Hugging Face Hub — Storage limits](https://huggingface.co/docs/hub/en/storage-limits)

---

## 4. 登入

### 4.1 建立 token

到 https://huggingface.co/settings/tokens（以 `Moris555` 登入）建立 **Fine-grained** token，勾選：

- **Write access to contents/settings of all repos under your personal namespace**

只有讀取權限（`repo.content.read`）的 token 無法建立 repo，也無法上傳。權限可以事後用 **⋮ → Edit permissions** 修改，token 字串不變，不需要重新登入。

### 4.2 登入 CLI

**必須在真正的終端機執行**（`docker exec -it` 或 SSH 進入這台機器）：

```bash
/root/miniconda3/envs/track_eval/bin/hf auth login
# 貼上 token；詢問 "Add token as git credential?" 時回答 n
```

- Claude Code 的 `! <指令>` **沒有 TTY**，`hf auth login` 讀取 token 時會出現 `EOFError` 而失敗。不要改用 `--token hf_xxx` 在 `!` 裡登入，token 會以明碼留在對話紀錄裡。
- 回答 n 是為了讓 HF 不寫入 git 憑證；上傳走 HTTP API，不需要 git 憑證，也不會影響 GitHub 的憑證。
- 新 token 會加入清單並成為目前使用的帳號，原本的 token 會保留。

### 4.3 上傳前確認帳號與權限

```bash
conda activate track_eval
python - <<'EOF'
from huggingface_hub import HfApi
w = HfApi().whoami(); a = w["auth"]["accessToken"]
perms = [p for s in a.get("fineGrained", {}).get("scoped", [])
         if s.get("entity", {}).get("name") == "Moris555" for p in s.get("permissions", [])]
print(w["name"], perms)
assert w["name"] == "Moris555" and "repo.write" in perms
EOF
```

帳號不是 `Moris555`，或沒有 `repo.write` 時，**停下來，不要上傳**。

---

## 5. 上傳流程

所有指令都在 `dataset/` 目錄下、`track_eval` 環境中執行。

### 5.1 建立 private repo，並確認是 private

```python
from huggingface_hub import HfApi
api = HfApi(); repo = "Moris555/<repo>"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=False)
assert api.repo_info(repo, repo_type="dataset").private is True
```

### 5.2 一次 commit 上傳（直接從原處讀取，不複製）

```python
import os
from huggingface_hub import HfApi, CommitOperationAdd
api = HfApi(); repo = "Moris555/safetrajai-eval"
ops = []
for src in ["eval/cctv_all8/annotations_all8", "eval/cctv_all8/mot_annotations_all8"]:
    dst = os.path.basename(src)
    for root, _, files in os.walk(src):
        for f in files:
            p = os.path.join(root, f)
            ops.append(CommitOperationAdd(path_in_repo=os.path.join(dst, os.path.relpath(p, src)),
                                          path_or_fileobj=p))
# 單一檔案：影片、subsets.yaml 等
ops.append(CommitOperationAdd(path_in_repo="subsets.yaml", path_or_fileobj="<path>/subsets.yaml"))
api.create_commit(repo, repo_type="dataset", operations=ops, commit_message="<message>")
```

2026-10-08 的實際結果：1,356 個檔案、150 MB，一次 commit，約 50 秒。

### 5.3 比對檔案清單與大小（不算 hash）

```python
info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
remote = {s.rfilename: s.size for s in info.siblings if s.rfilename != ".gitattributes"}
# local = {repo 內路徑: os.path.getsize(本機路徑)}，與 5.2 用同樣的方式組出
assert set(local) == set(remote) and all(local[k] == remote[k] for k in local)
```

依 CLAUDE.md「禁止大量 I/O」，只比對路徑清單和檔案大小，不對掛載磁碟上的檔案做全量 checksum。

### 5.4 打 tag 固定版本

比對一致後才打 tag：

```python
api.create_tag(repo, repo_type="dataset", tag="gt-2026-09-23", revision=info.sha,
               tag_message="C000025 human GT as fixed on 2026-09-23")
```

人工 GT 依規定不可修改，以後讀取評估資料一律指定這個 tag（`revision="gt-2026-09-23"`）。

---

## 6. 下載使用

```python
from huggingface_hub import snapshot_download
path = snapshot_download("Moris555/safetrajai-eval", repo_type="dataset",
                         revision="gt-2026-09-23", local_dir="<目標路徑>")
```

下載後放回 `dataset/eval/cctv_all8/` 時，`subsets.yaml` 要依清單重建成 `dataset/eval/cctv_all8_subsets/{tune,holdout}/` 的 symlink。

---

## 7. Re-ID 資料上傳規劃（已於 2026-10-08 執行，結果見第 8 節）

條件：使用者確認 Re-ID 資料已全部完成。

- 上傳的單位是**資料集版本**：`dataset/reid/splits/<version>/`（`manifest.csv`、`split_rules.yaml`、`summary.json`）、`~/.cache/safetrajai/reid_packs/<version>/` 打包檔、`auto_exclusions.csv`、`exclusions.csv`、`fragment_merges.csv`，以及對應的 `models/class_reid_weights/<run>/`（`model.pt`、`train_record.json`）。
- 打包檔在本機快取（不在掛載磁碟上），直接上傳它就不必再讀一次 `crops/`。
- 放在 `safetrajai-source`，不要放進 `safetrajai-eval`；每個版本打一個 tag（例如 `reid-v3`）。
- 評估鏡頭（C000025）和 C000026 不能出現在這個 repo。


## 8. Re-ID 私人備份完成紀錄（2026-10-08）

使用者明確授權將 91 個檔案（約 2.13 GB）上傳至 `Moris555/safetrajai-source` 私人 dataset repo 並建立版本標籤。

- Repo：[Moris555/safetrajai-source](https://huggingface.co/datasets/Moris555/safetrajai-source)
- Commit：`30b13cbcb56f5b84069163d9eacd2e2821331591`
- 檔案：91 個，共 2,129,317,756 bytes（不含 Hub 管理的 `.gitattributes`）。
- 已確認 `private=True`，遠端路徑集合與逐檔大小完全符合上傳清單。
- Tags：`reid-v2`、`reid-v3`、`reid-v3-2026-10-08`，全部指向同一份包含兩個版本的完整快照，不是各版本的篩選視圖。
- 實作來源 commit：`f723ba0`；GitHub 的程式碼、OpenSpec 結果與當時文件已推送至 `8db4ce2`。

### 上傳內容

- `dataset/reid/splits/{v1,v2,v3}/`：manifest、split_rules、summary。v1 只保存歷史 metadata，沒有現成影像 pack。
- `reid_packs/{v2,v3}/`：既有 `crops.bin`、`index.csv`、`pack.json`。未重讀原始裁切影像。
- `dataset/reid/crops/_index/`：39 份來源框座標索引，用於干擾物取樣。
- 排除與合併清單：`auto_exclusions.csv`、`exclusions.csv`、`fragment_merges.csv`。
- `dataset/reports/reid/` 的品質／抽查 CSV 與 `fragment_features.npz`；不含可重建的 HTML／review assets。
- 九個來源鏡頭的 ROI，以及 reid_crops、reid_split、fragment_merge 設定。
- 六組 `model.pt`／`train_record.json`：`dinov2s_v2`、`dinov2s_v3`、`dinov2s_v3_lowres`、`dinov2s_v3_distractor`、`dinov2s_v3_both`、`dinov2s_v3_seed1`。
- HF README 與 `inventory.json`（檔案路徑及大小）。

上傳前已確認 v2／v3 pack 索引與 manifest 的路徑、列數一致，offset 在檔案大小範圍內；manifest 與裁切 metadata 只有九個來源鏡頭。C000025 評估影像／GT、C000026、原始影片、temp 隔離資料、冒煙與 OOM 權重未上傳。

新候選未通過外觀分離度關卡，備份不代表通過追蹤驗收或批准部署；追蹤 config 維持不變。

### 下載與還原

登入可存取私人 repo 的帳號後：

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "Moris555/safetrajai-source",
    repo_type="dataset",
    revision="reid-v3-2026-10-08",
    local_dir="<download-dir>",
)
```

將 `dataset/`、`models/` 與所需的來源 `config/` 放回對應專案位置；`reid_packs/{v2,v3}/` 放到 `~/.cache/safetrajai/reid_packs/{v2,v3}/`。訓練讀取 pack，不需重新執行 `pack_manifest.py`。v3 pack 自成一份完整資料，不依赖 v2 pack；train_record 中的絕對路徑是原機器的歷史紀錄。
