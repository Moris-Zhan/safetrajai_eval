#!/bin/bash
# 部署 scripts/cvat_nuclio/ 下的 nuclio 函式到本機 CVAT（--platform local）。
#
# 前置條件：
# - CVAT 要用含 serverless 疊加檔案啟動過，否則沒有 nuclio dashboard 可以註冊函式：
#     cd ignore/cvat_deploy/cvat && docker compose -f docker-compose.yml \
#       -f components/serverless/docker-compose.serverless.yml up -d
# - 要裝跟 nuclio dashboard image 版本一致的 nuctl CLI
#   （版本見 ignore/cvat_deploy/cvat/components/serverless/docker-compose.serverless.yml 的
#   quay.io/nuclio/dashboard:<version> 這一行）。
set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd )"
CVAT_SERVERLESS_DIR="$REPO_ROOT/ignore/cvat_deploy/cvat/serverless"

if ! command -v nuctl >/dev/null 2>&1; then
    echo "nuctl not found on PATH. Install the version matching" >&2
    echo "ignore/cvat_deploy/cvat/components/serverless/docker-compose.serverless.yml's nuclio dashboard image tag." >&2
    exit 1
fi

# 權重檔不進版控（*.pt 已在 .gitignore），每次部署前從 models/ 複製一份進函式目錄。
cp "$REPO_ROOT/models/yolo/yolo11x_dennis.pt" "$SCRIPT_DIR/yolo11_detector/model/yolo11x_dennis.pt"

"$CVAT_SERVERLESS_DIR/deploy_gpu.sh" "$SCRIPT_DIR"
