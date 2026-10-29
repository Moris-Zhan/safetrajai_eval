#!/usr/bin/env python3
"""
把選點器部署成網頁服務，可從遠端瀏覽器開啟（頁面在點選時才產生，影像不離開這台主機）。

服務需要存取權杖：預設讀 runs/picker/server_token（第一次啟動時隨機產生，權限 600，在 gitignore 的 runs/ 內）；
也可用 --token 或環境變數 PICKER_TOKEN 指定。網址帶 ?token=<權杖> 開啟一次，之後瀏覽器以 cookie 記住。
服務是純 HTTP，沒有加密：只在信任的網路（內網、VPN、SSH 通道或 kubectl port-forward）使用，不要直接暴露到公開網路。

用法：
    python scripts/picker/serve_picker.py                 # 0.0.0.0:8765
    python scripts/picker/serve_picker.py --port 9000 --host 127.0.0.1   # 只開本機（配合 SSH 通道）
"""
from __future__ import annotations

import argparse
import os
import socket

from safetrajai.paths import PROJECT_ROOT
from safetrajai.picker.build import build_picker_page, list_cameras
from safetrajai.picker.server import load_or_create_token, make_server

DEFAULT_TOKEN_PATH = PROJECT_ROOT / "runs" / "picker" / "server_token"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=os.environ.get("PICKER_TOKEN"), help="存取權杖（預設讀 runs/picker/server_token）")
    args = parser.parse_args()

    token = args.token or load_or_create_token(DEFAULT_TOKEN_PATH)
    server = make_server(args.host, args.port, token, build_picker_page, list_cameras)
    print(f"選點器服務：{args.host}:{args.port}（Ctrl+C 結束）", flush=True)
    print(f"網址（以可連到這台主機的位址取代主機名稱）：http://{socket.gethostname()}:{args.port}/?token={token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
