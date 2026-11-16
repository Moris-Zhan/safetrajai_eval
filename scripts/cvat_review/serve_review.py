#!/usr/bin/env python3
"""
把 CVAT pilot_v1 task 指定類別（預設 small_truck／bus，降幅最大的兩類，見
Markdown/training_eval_audit.md 的「人工覆核前的抽樣檢視」）的完整追蹤標註，
做成網頁服務讓使用者從遠端瀏覽器逐 track 覆核（沿用 scripts/picker/serve_picker.py
的部署模式：token 保護、頁面在請求時才產生、影像不離開這台主機）。

前置：先用 ignore/cvat_deploy/cvat_client_venv 跑 fetch_task_annotations.py 產生
runs/cvat_review/pilot_v1_annotations.json（這支 serve_review.py 只需要 cv2，
用 conda env track_eval 執行即可，不需要 cvat_sdk）。

用法：
    python scripts/cvat_review/serve_review.py                      # 0.0.0.0:8766
    python scripts/cvat_review/serve_review.py --host 127.0.0.1 --port 9001  # 配合 SSH 通道

服務是純 HTTP：只在信任的網路（內網、VPN、SSH 通道或 kubectl port-forward）使用。
"""
from __future__ import annotations

import argparse
import base64
import csv
import hmac
import html
import socket
from collections import defaultdict
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlsplit

import cv2

from safetrajai.paths import PROJECT_ROOT
from safetrajai.picker.server import load_or_create_token, redact_token

ANNOTATIONS_PATH = PROJECT_ROOT / "runs" / "cvat_review" / "pilot_v1_annotations.json"
MANIFEST_PATH = PROJECT_ROOT / "dataset" / "detection" / "splits" / "pilot_v1" / "manifest.csv"
IMAGES_ROOT = PROJECT_ROOT / "dataset" / "detection" / "images"
DEFAULT_TOKEN_PATH = PROJECT_ROOT / "runs" / "cvat_review" / "server_token"
DEFAULT_CLASSES = ["small_truck", "bus"]
PAGE_SIZE = 25
JPEG_QUALITY = 85
COOKIE_NAME = "cvat_review_token"

HIGHLIGHT_COLOR = {"small_truck": (0, 140, 255), "bus": (255, 0, 255)}  # BGR


class ReviewData:
    """Indexes the fetched CVAT annotations and the manifest, for page building."""

    def __init__(self, annotations_path: Path, manifest_path: Path, dedup_dir: Path | None = None) -> None:
        import json

        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        self.tracks = payload["tracks"]

        with manifest_path.open(newline="", encoding="utf-8") as f:
            self.manifest_rows = list(csv.DictReader(f))

        self.dedup_groups: list[dict] = []
        if dedup_dir is not None:
            for name, kind in [("small_truck_clusters.csv", "identity"), ("small_truck_appearance_sample.csv", "appearance")]:
                path = dedup_dir / name
                if path.exists():
                    with path.open(newline="", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            self.dedup_groups.append({
                                "kind": kind, "size": int(row["size"]),
                                "representative_track_id": int(row["representative_track_id"]),
                                "member_track_ids": row["member_track_ids"],
                            })
            self.dedup_groups.sort(key=lambda g: -g["size"])

        # position (0-idx, == CVAT task frame) -> every non-outside shape on that frame,
        # so a track's page can draw the other objects in the same frame for context.
        self.by_frame: Dict[int, List[dict]] = defaultdict(list)
        # label -> list of tracks (each with only its non-outside shapes), for the index page.
        self.tracks_by_label: Dict[str, List[dict]] = defaultdict(list)
        for track in self.tracks:
            real_shapes = [s for s in track["shapes"] if not s["outside"]]
            for shape in real_shapes:
                self.by_frame[shape["frame"]].append(
                    {"track_id": track["track_id"], "label": track["label"], "points": shape["points"]}
                )
            self.tracks_by_label[track["label"]].append({
                "track_id": track["track_id"], "label": track["label"], "shapes": real_shapes,
            })
        for label in self.tracks_by_label:
            self.tracks_by_label[label].sort(key=lambda t: len(t["shapes"]), reverse=True)

    def track(self, label: str, track_id: int) -> dict | None:
        for t in self.tracks_by_label.get(label, []):
            if t["track_id"] == track_id:
                return t
        return None

    def render_frame(self, position: int, target_track_id: int, target_label: str) -> bytes:
        row = self.manifest_rows[position]
        img_path = IMAGES_ROOT / row["seq"] / f"{int(row['frame_idx']):06d}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(img_path)
        for box in self.by_frame.get(position, []):
            x1, y1, x2, y2 = (int(round(v)) for v in box["points"])
            is_target = box["track_id"] == target_track_id and box["label"] == target_label
            color = HIGHLIGHT_COLOR.get(target_label, (0, 140, 255)) if is_target else (160, 160, 160)
            thickness = 3 if is_target else 1
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            if is_target:
                cv2.putText(image, f"{box['label']} #{box['track_id']}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"JPEG encode failed for frame {position}")
        return jpeg.tobytes()


def _index_html(data: ReviewData, classes: List[str]) -> str:
    sections = []
    for label in classes:
        tracks = data.tracks_by_label.get(label, [])
        rows = []
        for t in tracks:
            n = len(t["shapes"])
            first_position = t["shapes"][0]["frame"] if t["shapes"] else None
            seq = html.escape(data.manifest_rows[first_position]["camera"]) if first_position is not None else "?"
            rows.append(
                f'<tr><td><a href="/track/{html.escape(label)}/{t["track_id"]}">#{t["track_id"]}</a></td>'
                f'<td>{n}</td><td>{seq}</td></tr>'
            )
        sections.append(
            f"<h2>{html.escape(label)}（{len(tracks)} 條 track）</h2>"
            '<table><tr><th>track</th><th>真實關鍵幀數</th><th>第一個鏡頭</th></tr>'
            + "".join(rows) + "</table>"
        )
    dedup_link = '<p><a href="/dedup">→ 去重候選群覆核（把剩餘未覆核 track 壓縮成少量代表）</a></p>' if data.dedup_groups else ""
    return (
        '<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8"><title>CVAT 標註覆核</title>'
        '<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}'
        'td,th{padding:4px 10px;text-align:left;border-bottom:1px solid #ddd}</style></head>'
        '<body><h1>pilot_v1 標註覆核</h1><p>按 track 數由多到少排序；點進去逐張看真實關鍵幀。</p>'
        + dedup_link + "".join(sections) + "</body></html>"
    )


def _dedup_html(data: "ReviewData", label: str) -> str:
    rows = []
    for g in data.dedup_groups:
        kind_zh = "身分去重（同一實體物件）" if g["kind"] == "identity" else "外觀分群抽樣（非同一實體，僅供參考）"
        n_members = g["member_track_ids"].count(";") + 1
        rows.append(
            f'<tr><td>{kind_zh}</td><td>{g["size"]}</td>'
            f'<td><a href="/track/{html.escape(label)}/{g["representative_track_id"]}">#{g["representative_track_id"]}</a></td>'
            f'<td>{html.escape(g["member_track_ids"])}</td></tr>'
        )
    return (
        '<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8"><title>去重候選群</title>'
        '<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}'
        'td,th{padding:4px 10px;text-align:left;border-bottom:1px solid #ddd;vertical-align:top}'
        'td:last-child{max-width:480px;word-break:break-all;font-size:12px;color:#555}</style></head>'
        '<body><p><a href="/">« 回索引</a></p><h1>去重候選群（共 ' + str(len(rows)) + ' 群）</h1>'
        '<p>「身分去重」：同鏡頭＋同位置＋同尺寸＋幾乎靜止＋外觀高度相似，判定為同一個實體物件被拆成多條 track，'
        '看完代表 track 的結論可以套用到整群（確認有誤才會一次修正整群）。</p>'
        '<p>「外觀分群抽樣」：單純外觀相近，不是同一個實體物件，代表 track 的結論<b>不能</b>套用到整群，'
        '只能當作這個外觀類型大致的參考，群內其餘 track 仍是未知風險。</p>'
        '<table><tr><th>類型</th><th>群大小</th><th>代表 track</th><th>群內全部 track id</th></tr>'
        + "".join(rows) + "</table></body></html>"
    )


def _track_html(data: ReviewData, label: str, track_id: int, offset: int) -> str:
    track = data.track(label, track_id)
    if track is None:
        return f"<p>找不到 track：{html.escape(label)} #{track_id}</p>"
    shapes = track["shapes"]
    page = shapes[offset:offset + PAGE_SIZE]
    figures = []
    for shape in page:
        position = shape["frame"]
        row = data.manifest_rows[position]
        jpeg = data.render_frame(position, track_id, label)
        b64 = base64.b64encode(jpeg).decode("ascii")
        caption = f"position {position} · {html.escape(row['camera'])} · frame_idx {row['frame_idx']}"
        figures.append(f'<figure><img src="data:image/jpeg;base64,{b64}"><figcaption>{caption}</figcaption></figure>')

    nav = []
    if offset > 0:
        nav.append(f'<a href="/track/{html.escape(label)}/{track_id}?offset={max(0, offset - PAGE_SIZE)}">« 上一頁</a>')
    if offset + PAGE_SIZE < len(shapes):
        nav.append(f'<a href="/track/{html.escape(label)}/{track_id}?offset={offset + PAGE_SIZE}">下一頁 »</a>')
    nav_html = " ｜ ".join(nav)

    return (
        '<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">'
        f'<title>{html.escape(label)} #{track_id}</title>'
        '<style>body{font-family:sans-serif;margin:16px}figure{display:inline-block;margin:8px;text-align:center}'
        'figcaption{font-size:12px;color:#555}img{max-width:704px}</style></head>'
        f'<body><p><a href="/">« 回索引</a></p><h1>{html.escape(label)} #{track_id}</h1>'
        f'<p>共 {len(shapes)} 個真實關鍵幀，顯示第 {offset + 1}–{min(offset + PAGE_SIZE, len(shapes))} 個。{nav_html}</p>'
        + "".join(figures) + f'<p>{nav_html}</p></body></html>'
    )


def make_server(host: str, port: int, token: str, data: ReviewData, classes: List[str]) -> ThreadingHTTPServer:
    if not token:
        raise ValueError("token 不可為空：沒有權杖的服務會把 CCTV 影像開放給任何連得到的人")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: str, cookie: str = "") -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Robots-Tag", "noindex")
            self.send_header("Referrer-Policy", "no-referrer")
            if cookie:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(payload)

        def _presented_tokens(self, query: dict) -> list:
            jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
            return query.get("token", [])[:1] + ([jar[COOKIE_NAME].value] if COOKIE_NAME in jar else [])

        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            presented = self._presented_tokens(query)
            if not any(hmac.compare_digest(v.encode("utf-8"), token.encode("utf-8")) for v in presented):
                self._send(401, "需要存取權杖：請使用啟動服務時印出的網址。")
                return
            cookie = f"{COOKIE_NAME}={token}; HttpOnly; SameSite=Strict; Path=/"
            segments = parts.path.split("/")
            try:
                if parts.path == "/":
                    self._send(200, _index_html(data, classes), cookie=cookie)
                elif parts.path == "/dedup":
                    self._send(200, _dedup_html(data, classes[0]), cookie=cookie)
                elif len(segments) == 4 and segments[1] == "track" and segments[2] in classes:
                    offset = int(query.get("offset", ["0"])[0])
                    self._send(200, _track_html(data, segments[2], int(segments[3]), offset), cookie=cookie)
                else:
                    self._send(404, "找不到")
            except Exception as error:
                self.log_error("page build failed: %r", error)
                self._send(500, "<p>伺服器產生頁面時發生錯誤，詳見伺服器日誌。</p>")

        def log_message(self, format: str, *args) -> None:
            print(redact_token(f"{self.address_string()} - {format % args}"), flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_PATH)
    parser.add_argument("--dedup-dir", type=Path, default=PROJECT_ROOT / "runs" / "cvat_review",
                         help="找 small_truck_clusters.csv／small_truck_appearance_sample.csv 的目錄，不存在時 /dedup 頁面留空")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    data = ReviewData(args.annotations, MANIFEST_PATH, args.dedup_dir)
    token = args.token or load_or_create_token(DEFAULT_TOKEN_PATH)
    server = make_server(args.host, args.port, token, data, args.classes)
    print(f"CVAT 標註覆核服務：{args.host}:{args.port}（Ctrl+C 結束）", flush=True)
    print(f"網址（以可連到這台主機的位址取代主機名稱）：http://{socket.gethostname()}:{args.port}/?token={token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
