#!/usr/bin/env python3
"""
Token-protected web page for the user's first review task (audit of Claude's own judgments).

Reads runs/cvat_review/audit_sample.json (from build_audit_sample.py). Each item shows crops of
one track and asks "what should this box be now?"; answers are appended to
runs/cvat_review/audit_responses.jsonl (the latest answer per item wins). /summary reports the
disagreement rate per sample group against the 5% threshold.

Deployment follows serve_review.py: token required, pages built on request, images stay on this host.
Run with the conda env track_eval (needs cv2):
    python scripts/cvat_review/serve_audit.py --host 0.0.0.0 --port 8767
"""
from __future__ import annotations

import argparse
import csv
import hmac
import html
import json
import socket
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlsplit

import cv2

from safetrajai.paths import PROJECT_ROOT
from safetrajai.picker.server import load_or_create_token, redact_token

REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
SAMPLE_PATH = REVIEW_DIR / "audit_sample.json"
RESPONSES_PATH = REVIEW_DIR / "audit_responses.jsonl"
CURRENT_ANNOTATIONS_PATH = REVIEW_DIR / "pilot_v1_annotations.json"
BACKUP_PATH = REVIEW_DIR / "backups" / "task3_pre_corrections_20261113T033126Z.json"
MANIFEST_PATH = PROJECT_ROOT / "dataset" / "detection" / "splits" / "pilot_v1" / "manifest.csv"
IMAGES_ROOT = PROJECT_ROOT / "dataset" / "detection" / "images"
TOKEN_PATH = REVIEW_DIR / "server_token"
COOKIE_NAME = "cvat_audit_token"
CROP_FRAMES = 3
CROP_HEIGHT_PX = 300
JPEG_QUALITY = 85
ERROR_THRESHOLD = 0.05

ANSWER_CHOICES = [
    ("person", "person 行人"),
    ("car", "car 轎車／休旅／廂型"),
    ("scooter", "scooter 機車＋騎士"),
    ("small_truck", "small_truck 輕型貨車"),
    ("truck", "truck 中大型貨車"),
    ("bus", "bus 公車／遊覽車"),
    ("delete", "不是道路使用者（應刪除）"),
    ("unsure", "看不出來"),
]
VALID_ANSWERS = {value for value, _ in ANSWER_CHOICES}
GROUP_NAMES = {
    "applied_correction": "已套用修正",
    "unchanged_representative": "判為『真』的代表（未修正）",
    "ambiguous_representative": "不確定／疑似標錯的代表（待你裁決）",
}
ADJUDICATION_KINDS = {"ambiguous_representative"}


class AuditData:
    def __init__(self, sample_path: Path, tracks_path: Path, backup_path: Path, manifest_path: Path) -> None:
        self.items: List[dict] = json.loads(sample_path.read_text(encoding="utf-8"))["items"]
        self.items_by_id: Dict[str, dict] = {item["item_id"]: item for item in self.items}
        wanted = {item["track_id"] for item in self.items}
        self.shapes_by_track: Dict[int, List[dict]] = {}
        for source in (backup_path, tracks_path):
            for track in json.loads(source.read_text(encoding="utf-8"))["tracks"]:
                if track["track_id"] in wanted and track["track_id"] not in self.shapes_by_track:
                    self.shapes_by_track[track["track_id"]] = [s for s in track["shapes"] if not s["outside"]]
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            self.manifest_rows = list(csv.DictReader(handle))

    def chosen_shapes(self, track_id: int) -> List[dict]:
        shapes = self.shapes_by_track.get(track_id, [])
        if len(shapes) <= CROP_FRAMES:
            return shapes
        last = len(shapes) - 1
        return [shapes[round(last * fraction)] for fraction in (0.1, 0.5, 0.9)]

    def render(self, item_id: str, index: int, context: bool) -> bytes:
        item = self.items_by_id[item_id]
        shapes = self.chosen_shapes(item["track_id"])
        shape = shapes[min(index, len(shapes) - 1)]
        row = self.manifest_rows[shape["frame"]]
        image = cv2.imread(str(IMAGES_ROOT / row["seq"] / f"{int(row['frame_idx']):06d}.jpg"))
        if image is None:
            raise FileNotFoundError(row["seq"])
        x1, y1, x2, y2 = (int(round(v)) for v in shape["points"])
        if context:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            output = image
        else:
            height, width = image.shape[:2]
            pad = max(40, 2 * max(x2 - x1, y2 - y1))
            left, top = max(0, x1 - pad), max(0, y1 - pad)
            right, bottom = min(width, x2 + pad), min(height, y2 + pad)
            crop = image[top:bottom, left:right].copy()
            cv2.rectangle(crop, (x1 - left, y1 - top), (x2 - left, y2 - top), (0, 0, 255), 1)
            scale = CROP_HEIGHT_PX / crop.shape[0]
            output = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ok, jpeg = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return jpeg.tobytes()


def load_latest_answers(path: Path) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                answers[record["item_id"]] = record["answer"]
    return answers


def summarize(items: List[dict], answers: Dict[str, str]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for kind in GROUP_NAMES:
        group = [item for item in items if item["kind"] == kind]
        answered = [item for item in group if item["item_id"] in answers]
        decided = [item for item in answered if answers[item["item_id"]] != "unsure"]
        disagreed = [item for item in decided if answers[item["item_id"]] != item["claimed_final"]]
        summary[kind] = {
            "total": len(group), "answered": len(answered), "unsure": len(answered) - len(decided),
            "disagreed": len(disagreed),
            "rate": (len(disagreed) / len(decided)) if decided else None,
        }
    return summary


def _item_html(item: dict, index: int, current: Optional[str]) -> str:
    images = "".join(f'<img loading="lazy" src="/img/{item["item_id"]}/{k}.jpg">' for k in range(CROP_FRAMES))
    context = f'<img loading="lazy" class="ctx" src="/img/{item["item_id"]}/ctx.jpg">'
    if item["kind"] == "applied_correction":
        claim = f'原標籤 {html.escape(item["old_label"])} → <b>{html.escape(item["claimed_final"])}</b>'
    elif item["kind"] == "ambiguous_representative":
        claim = f'目前標籤 <b>{html.escape(item["claimed_final"])}</b>（我不確定：{html.escape(item["reason"])}）'
    else:
        claim = f'目前標籤 <b>{html.escape(item["claimed_final"])}</b>（未修正）'
    options = "".join(
        f'<label><input type="radio" name="{item["item_id"]}" value="{value}"'
        f'{" checked" if current == value else ""} onchange="save(this)"> {html.escape(text)}</label>'
        for value, text in ANSWER_CHOICES
    )
    return (
        f'<section id="{item["item_id"]}"><h3>#{index} · track {item["track_id"]} · {claim}</h3>'
        f'<div class="imgs">{images}{context}</div><div class="opts">{options}</div></section>'
    )


def _page_html(data: AuditData, answers: Dict[str, str]) -> str:
    sections = "".join(_item_html(item, n + 1, answers.get(item["item_id"])) for n, item in enumerate(data.items))
    return (
        '<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>標註稽核</title><style>body{font-family:sans-serif;margin:16px;max-width:1100px}'
        'section{border-top:2px solid #888;padding:8px 0}.imgs img{height:200px;margin:2px;border:1px solid #ccc}'
        '.imgs img.ctx{height:240px}.opts label{display:inline-block;margin:4px 12px 4px 0}'
        '#bar{position:sticky;top:0;background:#fff;padding:6px;border-bottom:1px solid #aaa}</style></head><body>'
        f'<div id="bar">已答 <span id="n">{len(answers)}</span> / {len(data.items)} ｜ <a href="/summary">結果摘要</a></div>'
        '<h1>標註稽核</h1><p>每題請依你自己的判斷，選出<b>這個紅框現在應該是什麼</b>。看不清楚就選「看不出來」。'
        '標題的標籤是我（Claude）的判斷，請不要被它影響。前三張是同一條軌跡的放大裁切，最後一張是整幀。</p>'
        + sections +
        '<script>async function save(el){const r=await fetch("/answer",{method:"POST",headers:{"Content-Type":"application/json"},'
        'body:JSON.stringify({item_id:el.name,answer:el.value})});if(r.ok){const j=await r.json();'
        'document.getElementById("n").textContent=j.answered}else{alert("儲存失敗："+r.status)}}</script>'
        '</body></html>'
    )


def _summary_html(data: AuditData, answers: Dict[str, str]) -> str:
    rows = []
    for kind, stats in summarize(data.items, answers).items():
        if kind in ADJUDICATION_KINDS:
            rate, verdict = "不計錯誤率（逐條裁決，答案見 audit_responses.jsonl）", ""
        else:
            rate = "—" if stats["rate"] is None else f'{stats["rate"]:.1%}'
            verdict = "" if stats["rate"] is None else ("；超過 5%，需重審" if stats["rate"] > ERROR_THRESHOLD else "；通過")
        rows.append(
            f'<tr><td>{GROUP_NAMES[kind]}</td><td>{stats["answered"]}/{stats["total"]}</td>'
            f'<td>{stats["unsure"]}</td><td>{stats["disagreed"]}</td><td>{rate}{verdict}</td></tr>'
        )
    return (
        '<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8"><title>稽核結果</title></head><body>'
        '<p><a href="/">« 回題目</a></p><h1>稽核結果</h1>'
        '<table border="1" cellpadding="6"><tr><th>群組</th><th>已答</th><th>看不出來</th><th>與我不同</th><th>不同意率（不含看不出來）</th></tr>'
        + "".join(rows) + '</table></body></html>'
    )


def make_server(host: str, port: int, token: str, data: AuditData, responses_path: Path) -> ThreadingHTTPServer:
    if not token:
        raise ValueError("token 不可為空：沒有權杖的服務會把 CCTV 影像開放給任何連得到的人")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str = "text/html; charset=utf-8", cookie: str = "") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Robots-Tag", "noindex")
            self.send_header("Referrer-Policy", "no-referrer")
            if cookie:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self, query: dict) -> bool:
            jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
            presented = query.get("token", [])[:1] + ([jar[COOKIE_NAME].value] if COOKIE_NAME in jar else [])
            return any(hmac.compare_digest(v.encode("utf-8"), token.encode("utf-8")) for v in presented)

        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            if not self._authorized(query):
                self._send(401, "需要存取權杖：請使用啟動服務時印出的網址。".encode("utf-8"))
                return
            cookie = f"{COOKIE_NAME}={token}; HttpOnly; SameSite=Strict; Path=/"
            segments = parts.path.split("/")
            try:
                if parts.path == "/":
                    self._send(200, _page_html(data, load_latest_answers(responses_path)).encode("utf-8"), cookie=cookie)
                elif parts.path == "/summary":
                    self._send(200, _summary_html(data, load_latest_answers(responses_path)).encode("utf-8"), cookie=cookie)
                elif len(segments) == 4 and segments[1] == "img" and segments[2] in data.items_by_id and segments[3].endswith(".jpg"):
                    stem = segments[3][:-4]
                    jpeg = data.render(segments[2], 0 if stem == "ctx" else int(stem), stem == "ctx")
                    self._send(200, jpeg, "image/jpeg", cookie)
                else:
                    self._send(404, "找不到".encode("utf-8"))
            except Exception as error:
                self.log_error("page build failed: %r", error)
                self._send(500, "<p>伺服器產生頁面時發生錯誤，詳見伺服器日誌。</p>".encode("utf-8"))

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/answer":
                self._send(404, b"")
                return
            if not self._authorized({}):
                self._send(401, b"")
                return
            try:
                record = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                item_id, answer = record["item_id"], record["answer"]
                if item_id not in data.items_by_id or answer not in VALID_ANSWERS:
                    raise ValueError("unknown item or answer")
            except (ValueError, KeyError, TypeError):
                self._send(400, b"")
                return
            with responses_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"item_id": item_id, "answer": answer, "ts": time.time()}, ensure_ascii=False) + "\n")
            body = json.dumps({"answered": len(load_latest_answers(responses_path))}).encode("utf-8")
            self._send(200, body, "application/json")

        def log_message(self, format: str, *args) -> None:
            print(redact_token(f"{self.address_string()} - {format % args}"), flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--responses", type=Path, default=RESPONSES_PATH)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    data = AuditData(args.sample, CURRENT_ANNOTATIONS_PATH, BACKUP_PATH, MANIFEST_PATH)
    token = args.token or load_or_create_token(TOKEN_PATH)
    server = make_server(args.host, args.port, token, data, args.responses)
    print(f"標註稽核服務：{args.host}:{args.port}（Ctrl+C 結束）", flush=True)
    print(f"網址（以可連到這台主機的位址取代主機名稱）：http://{socket.gethostname()}:{args.port}/?token={token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
