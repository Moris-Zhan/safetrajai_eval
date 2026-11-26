#!/bin/bash
# 部署 cvat/nuclio/ 下的 nuclio 函式到本機 CVAT（--platform local）。
#
# 前置條件：
# - CVAT 要用含 serverless 疊加檔案啟動過，否則沒有 nuclio dashboard 可以註冊函式：
#     cd cvat/deploy/cvat && docker compose -f docker-compose.yml \
#       -f components/serverless/docker-compose.serverless.yml up -d
# - nuctl CLI 放在 cvat/bin/nuctl（gitignore），版本要跟 nuclio dashboard image 一致
#   （見 cvat/deploy/cvat/components/serverless/docker-compose.serverless.yml 的
#   quay.io/nuclio/dashboard:<version>，目前 1.16.3）。缺少時：
#     curl -sSfL -o cvat/bin/nuctl https://github.com/nuclio/nuclio/releases/download/1.16.3/nuctl-1.16.3-linux-amd64 && chmod +x cvat/bin/nuctl
# - docker 資料在容器內、重啟後消失，函式映像也會跟著消失；重啟後重跑本腳本即可重建（約 10～20 分鐘）。
set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd )"
CVAT_SERVERLESS_DIR="$REPO_ROOT/cvat/deploy/cvat/serverless"
export PATH="$REPO_ROOT/cvat/bin:$PATH"

if ! command -v nuctl >/dev/null 2>&1; then
    echo "nuctl not found in cvat/bin/ or on PATH. Install the version matching" >&2
    echo "cvat/deploy/cvat/components/serverless/docker-compose.serverless.yml's nuclio dashboard image tag." >&2
    exit 1
fi

# 權重檔不進版控（*.pt 已在 .gitignore），每次部署前從 models/ 複製一份進函式目錄。
cp "$REPO_ROOT/models/yolo/yolo11x_base.pt" "$SCRIPT_DIR/yolo11_detector/model/yolo11x_base.pt"

"$CVAT_SERVERLESS_DIR/deploy_gpu.sh" "$SCRIPT_DIR"
