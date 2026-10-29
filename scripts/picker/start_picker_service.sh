#!/usr/bin/env bash
# 在背景啟動選點器網頁服務（serve_picker.py）；已經有服務在該連接埠回應時什麼都不做。
#   bash scripts/picker/start_picker_service.sh
# 環境變數：PICKER_PORT（預設 8765）、PICKER_LOG（預設 runs/picker/server.log，權限 600，被 git 忽略）。
# 日誌第一行含帶權杖的網址，所以只留在 runs/ 內；pod 重啟後需要再執行一次。
set -euo pipefail
cd "$(dirname "$0")/../.."

PORT="${PICKER_PORT:-8765}"
LOG="${PICKER_LOG:-runs/picker/server.log}"

# 沒帶權杖的請求會得到 401，代表服務在線；沒有人在聽時 curl 會失敗。
if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${PORT}/"; then
  echo "picker service already running on port ${PORT}"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
umask 077
nohup conda run -n track_eval --no-capture-output python scripts/picker/serve_picker.py --port "$PORT" >> "$LOG" 2>&1 &
echo "picker service started on port ${PORT} (log: ${LOG}; the URL with the token is printed at its top)"
