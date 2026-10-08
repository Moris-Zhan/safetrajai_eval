# GitHub 推送流程（GitHub Push Workflow）

> 整理日期：2026-10-08
> 依據：2026-10-06～10-08 首次把專案推到 GitHub 的過程與決策。

---

## 0. 現況

| 項目 | 設定 |
|---|---|
| 遠端 repo | `https://github.com/Moris-Zhan/safetrajai_eval`（**Private**） |
| remote | `origin` |
| 分支 | 只有 `main`；本機 `main` 追蹤 `origin/main`（原本叫 `master`，已改名） |
| 開發方式 | 直接在 `main` 上 commit 後 push，不開 feature branch、不走 PR |
| 驗證工具 | GitHub CLI（`gh`），裝在 conda base：`/root/miniconda3/bin/gh`（**不在 PATH**，要用完整路徑） |
| 憑證 | `gh auth login`（瀏覽器 device code、HTTPS），再用 `gh auth setup-git` 交給 git 使用；token 以明文存在 `/root/.config/gh/hosts.yml` |
| 授權 | 遠端 Initial commit 帶進的 MIT `LICENSE` 與 `README.md` 已合併保留 |

---

## 1. 日常推送

```bash
cd /mnt/moris/Dcoker_Project/Project/safetrajai_eval
conda activate track_eval

# 1. commit 前先跑測試
python -m pytest -q

# 2. 看 diff 摘要，只 add 這次要提交的檔案（不要用 git add -A）
git status
git diff --stat
git add <files...>
git commit          # 英文、conventional commits 格式

# 3. 推送
git push            # 已設定追蹤，等同 git push origin main
```

推送後確認本機和遠端指到同一個 commit：

```bash
git status -sb                  # 應顯示 main...origin/main，沒有 ahead/behind
git ls-remote origin main       # hash 應與 git rev-parse HEAD 相同
```

---

## 2. 禁止事項

| 不要做 | 原因 |
|---|---|
| `git push --all`／`--mirror` | 本機有 Codex checkpoint ref（`refs/codex/...`），它引用了 `sumo.zip`（1.77 GB）與 `SSAM.zip`（283 MB）。這兩個檔案超過 GitHub 單檔 100 MB 上限，推送會被拒絕。**只推 `main`。** |
| `git rebase` 已推送的歷史、`git push --force` | 已推送的 commit 被改寫後，遠端與本機會分岔，必須 force push 才能同步。另外，smoke 測試會用 `tests/golden/BASE_COMMIT`（`9c6414b`，`main` 之前 32 個 commit）的 hash，以 `git show` 讀回凍結的 tracker 方法；只要改寫到這個 commit 或更早的歷史，hash 就會失效，AST 檢查也會失敗。 |
| 把 `dataset/`、`models/`、`runs/`、`video/` 加進 git | 這些目錄已在 gitignore 中（權重、影片、實驗輸出，檔案數以百萬計），而且位於掛載磁碟，不能做大量 I/O。 |
| 推送前全量掃描這些資料目錄 | 見 CLAUDE.md「禁止大量 I/O」；推送前的檢查只看 git 物件（下一節）。 |

---

## 3. 推送前檢查（大檔與機密）

只檢查 `main` 能追溯到的 git 物件，不讀取資料目錄：

```bash
# main 上最大的檔案（GitHub 單檔上限 100 MB，超過 50 MB 會警告）
git rev-list --objects main \
  | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk '$1=="blob"' | sort -k2 -n -r | head -5

# 常見的 token／私鑰格式
git grep -nIE "(ghp_[A-Za-z0-9]{30,}|github_pat_|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)" -- . ':!third_party'
```

2026-10-08 首次推送時：`main` 共 109 個 commit、8.2 MB，最大單檔 0.2 MB，沒有找到機密資料。

---

## 4. 在新機器上重新設定

```bash
# 1. 安裝 gh（裝在 conda base，不碰 track_eval）
conda install -y -n base -c conda-forge gh

# 2. 登入：會顯示 8 碼代碼，到 https://github.com/login/device 輸入
/root/miniconda3/bin/gh auth login -h github.com -p https -w

# 3. 讓 git 使用 gh 的憑證
/root/miniconda3/bin/gh auth setup-git

# 4. 確認 repo 仍是 private
/root/miniconda3/bin/gh repo view Moris-Zhan/safetrajai_eval --json visibility,defaultBranchRef

# 5. 取得程式碼
git clone https://github.com/Moris-Zhan/safetrajai_eval.git
```

注意事項：

- **登入必須由你在終端機互動完成**。在 Claude Code 裡用 `! <指令>` 執行；如果 120 秒內沒有完成授權，指令會被移到背景，重新執行一次即可，記得停掉殘留的背景程序。
- 容器裡沒有剪貼簿工具，`gh` 會顯示「Failed to copy one-time code」，直接手動抄寫代碼即可。
- clone 下來的只有程式碼。資料、權重與實驗輸出（`dataset/`、`models/`、`runs/`、`video/`）都不在 repo 裡，需要另外準備；安裝方式見 CLAUDE.md（conda env `track_eval`、`pip install -e . --no-deps`）。

---

## 5. 首次推送紀錄（2026-10-08）

1. 安裝 `gh` 2.102.0 到 conda base，以 `Moris-Zhan` 身分登入，token 權限為 `repo`、`gist`、`read:org`。
2. 確認 repo 是 private 時發現遠端並非空的：有一個 GitHub 自動產生的 `Initial commit`（`c360d90`：Python 範本 `.gitignore`、MIT `LICENSE`、`README.md`），與本機歷史沒有共同祖先。
3. 為了不改變本機 commit hash（`BASE_COMMIT` 依賴），不用 rebase 也不 force push，改為：
   ```bash
   git branch -m master main
   git merge --no-commit --no-ff --allow-unrelated-histories origin/main
   git checkout --ours .gitignore      # 保留專案自己的 .gitignore
   git add .gitignore
   python -m pytest -q                 # 368 passed、11 skipped
   git commit                          # 51f8281 chore: merge GitHub initial commit with LICENSE and README
   git push -u origin main
   ```
4. 結果：遠端 `main` 與本機同為 `51f8281`，預設分支 `main`，visibility `PRIVATE`。
