"""Token-protected web service that builds the picker pages on demand (scripts/picker/serve_picker.py).

Routes: `/` lists the cameras and modes, `/page/<camera>/<mode>` builds one page. Every request needs
the access token, as `?token=...` or as the cookie set after one valid request. Only listed cameras
and known modes are served, so the request path never reaches the file system. The CCTV and
satellite images stay on this host and are only sent to the browser that holds the token.
"""

from __future__ import annotations

import hmac
import html
import os
import re
import secrets
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import parse_qs, urlsplit

from safetrajai.picker.build import PickerInputError
from safetrajai.picker.page import MODES

COOKIE_NAME = "picker_token"
_TOKEN_IN_LOG = re.compile(r"(token=)[^&\s]*")


def redact_token(text: str) -> str:
    return _TOKEN_IN_LOG.sub(r"\1***", text)


def load_or_create_token(path: Path) -> str:
    """The saved access token, or a new random one written with owner-only permissions."""
    path = Path(path)
    if path.exists():
        if path.stat().st_mode & 0o077:
            raise PermissionError(f"{path} 的權限過寬，其他使用者可讀取權杖：請執行 chmod 600 {path}")
        return path.read_text(encoding="utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    return token


def _index_html(cameras: Sequence[str]) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(camera)}</th>"
        + "".join(f'<td><a href="/page/{html.escape(camera)}/{mode}">{mode}</a></td>' for mode in MODES)
        + "</tr>"
        for camera in cameras)
    return ('<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8"><title>選點器</title>'
            '<style>body{font-family:sans-serif;margin:16px}td,th{padding:6px 12px;text-align:left}</style></head>'
            f"<body><h1>選點器</h1><p>選鏡頭與模式；頁面在點選後會產生可複製的 YAML。</p><table>{rows}</table></body></html>")


def make_server(
    host: str, port: int, token: str,
    build_page: Callable[[str, str], str], list_cameras: Callable[[], Sequence[str]],
) -> ThreadingHTTPServer:
    if not token:
        raise ValueError("token 不可為空：沒有權杖的服務會把 CCTV 影像開放給任何連得到的人")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8", cookie: str = "") -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Robots-Tag", "noindex")
            self.send_header("Referrer-Policy", "no-referrer")
            if cookie:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(data)

        def _presented_tokens(self, query: dict) -> list:
            jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
            return query.get("token", [])[:1] + ([jar[COOKIE_NAME].value] if COOKIE_NAME in jar else [])

        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            presented = self._presented_tokens(parse_qs(parts.query))
            if not any(hmac.compare_digest(value.encode("utf-8"), token.encode("utf-8")) for value in presented):
                self._send(401, "需要存取權杖：請使用啟動服務時印出的網址。")
                return
            cookie = f"{COOKIE_NAME}={token}; HttpOnly; SameSite=Strict; Path=/"
            segments = parts.path.split("/")
            if parts.path == "/":
                self._send(200, _index_html(list(list_cameras())), cookie=cookie)
            elif len(segments) == 4 and segments[1] == "page" and segments[3] in MODES and segments[2] in list_cameras():
                try:
                    self._send(200, build_page(segments[2], segments[3]), cookie=cookie)
                except PickerInputError as error:
                    self._send(400, f"<p>無法產生頁面：{html.escape(str(error))}</p>")
                except Exception as error:
                    self.log_error("page build failed: %r", error)
                    self._send(500, "<p>伺服器產生頁面時發生錯誤，詳見伺服器日誌。</p>")
            else:
                self._send(404, "找不到")

        def log_message(self, format: str, *args) -> None:
            print(redact_token(f"{self.address_string()} - {format % args}"), flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
