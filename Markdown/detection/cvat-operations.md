# CVAT 操作手冊（偵測資料集覆核用）

用途：pilot_v1 偵測資料集的人工覆核使用本機自架的 CVAT（task id 3，`safetrajai detection pilot v1`）。這份文件記錄如何手動啟動、停止、檢查它，以及覆核時用到的腳本。部署目錄在 `ignore/cvat_deploy/`，被 gitignore，不進版控。

## 部署位置

| 項目 | 位置 |
| --- | --- |
| CVAT 原始碼與 compose 檔 | `ignore/cvat_deploy/cvat/`（clone 自 `cvat-ai/cvat`） |
| compose 專案名稱 | `cvat` |
| compose 設定檔 | `docker-compose.yml` ＋ `components/serverless/docker-compose.serverless.yml` |
| SDK 的獨立 venv | `ignore/cvat_deploy/cvat_client_venv/`（不動 conda env `track_eval`） |
| 網址 | `http://localhost:8080` |
| 連線設定 | 環境變數或 repo 根目錄 `.env` 的 `CVAT_URL`、`CVAT_USERNAME`、`CVAT_PASSWORD`（不要把值寫進文件或回覆） |

## 啟動、停止、檢查

以下指令都在 `ignore/cvat_deploy/cvat/` 執行。

```bash
cd ignore/cvat_deploy/cvat

# 啟動既有容器（不重建，資料庫與 volume 不動）
docker compose -p cvat -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml start

# 停止（保留容器與資料）
docker compose -p cvat -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml stop

# 檢查：回應 200 代表 API 正常（啟動後約需 10～60 秒）
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8080/api/server/about
docker compose ls -a          # 專案狀態
docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'cvat|traefik'
```

只啟動網頁介面（`cvat_ui` 預設可能是停止的；用 SDK 套用修正時不需要它，要在瀏覽器看標註才啟動）：

```bash
docker compose -p cvat -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml start cvat_ui
```

### 不要做的事

- **不要用 `docker compose down -v`**：`-v` 會刪掉資料庫 volume，所有標註與修正結果都會消失。
- **不要用 `up -d --force-recreate` 或重建容器**，除非已先備份。一般重啟只用 `start`。
- 不要把帳號密碼寫進文件、commit 或回覆。

## 覆核用的腳本（`scripts/cvat_review/`）

CVAT SDK 腳本一律用獨立 venv 的 Python：`ignore/cvat_deploy/cvat_client_venv/bin/python`。

| 腳本 | 用途 |
| --- | --- |
| `fetch_task_annotations.py --task-id 3 --out runs/cvat_review/pilot_v1_annotations.json` | 把 CVAT 目前的標註匯出成 JSON（每次套用修正後重新匯出核對） |
| `apply_corrections.py --task-id 3 --corrections <csv> [--dry-run]` | 套用修正檔（欄位 `track_id,action,new_label,reason`；`action` 是 `delete` 或 `relabel`）。先自動備份，全部成功後才寫入紀錄 |

套用修正的固定流程：先 `--dry-run` 確認計畫的刪除與改類別數量，再正式套用，最後用 `fetch_task_annotations.py` 重新匯出，核對 track 總數與各類別數。

產出位置（都在 `runs/cvat_review/`，gitignore）：

- 備份：`backups/task3_pre_corrections_<UTC 時間>.json`
- 已套用紀錄：`corrections_applied.csv`（欄位 `track_id,action,old_label,new_label,reason,verified`）
- 框位置與群框清單：`box_issues.csv`（第二階段用）

## 已知問題

- `cvat_vector`（日誌收集）啟動時掛載 `vector.toml` 失敗，一直是 Created 狀態；不影響標註作業，可以不理它。
- `cvat_ui` 可能是停止的，見上面「只啟動網頁介面」。
- 曾發現整組 CVAT 容器同時乾淨停止（Exited 0），主機上同時有另一個不相關的 docker 專案；原因不明。用上面的 `start` 即可恢復，資料不受影響。
