#!/usr/bin/env python3
"""
Token-protected web page to spot-check detector false positives on the validation split: are they real
mistakes, or objects the ground truth missed?

Input is a false-positive list from count_detection_errors.py --dump-fp (one camera is kept).
The list is split into confidence strata (--strata) and --per-stratum items are drawn from each with --seed;
the sample is saved once and reused on restart. Each item shows a crop around the false-positive box (red)
with the ground-truth boxes of every class (green, labelled), and asks one question:
    missed_gt    it is the target class and the ground truth has no box for it
    false        not a road user at all (a real false positive)
    other_class  a road user of another class (e.g. a rider, which the labeling guide counts as scooter)
    unsure       cannot tell
    localization the ground truth has a box for it, but the box differs too much (IoU < the matching threshold)
Answers are appended to --responses (latest answer per item wins). The ground truth itself is never changed.

--summary prints per-stratum counts and the rate of each verdict weighted by stratum size.

Run with the conda env track_eval (needs cv2):
    python scripts/cvat_review/serve_fp_review.py --fp runs/finetune/errors_v3/fp_base_person.json --camera C000056
    python scripts/cvat_review/serve_fp_review.py --fp runs/finetune/errors_v3/fp_base_person.json --camera C000056 --summary
"""
from __future__ import annotations

import argparse
import hmac
import json
import random
import socket
import time
from collections import Counter
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import parse_qs, urlsplit

import cv2
import yaml

from safetrajai.paths import PROJECT_ROOT
from safetrajai.picker.server import load_or_create_token, redact_token

REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
EXPORTS = PROJECT_ROOT / "dataset" / "detection" / "exports"
TOKEN_PATH = REVIEW_DIR / "server_token"
COOKIE_NAME = "fp_review_token"
VERDICTS = [("missed_gt", "是目標類別，GT 漏標"), ("false", "不是道路使用者，真的誤報"),
            ("other_class", "是其他類別的道路使用者"), ("unsure", "看不清"),
            ("localization", "GT 已有框，只是大小／位置不合")]
MIN_CROP_HEIGHT_PX = 360
MAX_CROP_WIDTH_PX = 1000


def stratum_of(conf: float, edges: Sequence[float]) -> int:
    return sum(conf >= edge for edge in edges)


def draw_sample(items: List[dict], edges: Sequence[float], per_stratum: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    sample = []
    for stratum in range(len(edges) + 1):
        pool = [dict(item, stratum=stratum) for item in items if stratum_of(item["conf"], edges) == stratum]
        sample.extend(rng.sample(pool, min(per_stratum, len(pool))))
    rng.shuffle(sample)
    return [dict(item, item_id=f"q{n}") for n, item in enumerate(sample)]


def render_item(item: dict, export_root: Path, names: Dict[int, str]) -> bytes:
    image = cv2.imread(str(export_root / "images" / "val" / item["image"]))
    if image is None:
        raise RuntimeError(f"cannot read {item['image']}")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = item["box"]
    pad = max(60.0, 1.5 * max(x2 - x1, y2 - y1))
    left, top = max(0, int(x1 - pad)), max(0, int(y1 - pad))
    right, bottom = min(width, int(x2 + pad)), min(height, int(y2 + pad))
    scale = max(1.0, MIN_CROP_HEIGHT_PX / (bottom - top))
    if (right - left) * scale > MAX_CROP_WIDTH_PX:
        scale = MAX_CROP_WIDTH_PX / (right - left)
    crop = cv2.resize(image[top:bottom, left:right], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def to_crop(x: float, y: float):
        return int((x - left) * scale), int((y - top) * scale)

    label_path = export_root / "labels" / "val" / f"{Path(item['image']).stem}.txt"
    lines = label_path.read_text(encoding="utf-8").split("\n") if label_path.exists() else []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:])
        a = to_crop((cx - w / 2) * width, (cy - h / 2) * height)
        b = to_crop((cx + w / 2) * width, (cy + h / 2) * height)
        cv2.rectangle(crop, a, b, (60, 200, 60), 1)
        cv2.putText(crop, names.get(cls, str(cls)), (a[0], max(12, a[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 200, 60), 1)
    cv2.rectangle(crop, to_crop(x1, y1), to_crop(x2, y2), (40, 40, 230), 2)
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def load_latest(path: Path) -> Dict[str, str]:
    latest: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["item_id"]] = record["verdict"]
    return latest


def summarize(items: List[dict], sample: List[dict], answers: Dict[str, str], edges: Sequence[float]) -> str:
    sizes = Counter(stratum_of(item["conf"], edges) for item in items)
    bounds = [0.0, *edges, 1.0]
    lines = [f"false positives: {len(items)}; sample: {len(sample)}; answered: {sum(i['item_id'] in answers for i in sample)}"]
    weighted: Counter = Counter()
    covered = 0
    for stratum in sorted(sizes):
        got = Counter(answers[i["item_id"]] for i in sample if i["stratum"] == stratum and i["item_id"] in answers)
        total = sum(got.values())
        lines.append(f"stratum conf [{bounds[stratum]:.2f}, {bounds[stratum + 1]:.2f}): size {sizes[stratum]}, answered {total}, "
                     + ", ".join(f"{v} {got[v]}" for v, _ in VERDICTS))
        if total:
            covered += sizes[stratum]
            for verdict, _ in VERDICTS:
                weighted[verdict] += sizes[stratum] * got[verdict] / total
    if covered:
        lines.append("weighted rate (over answered strata): " + ", ".join(f"{v} {weighted[v] / covered:.1%}" for v, _ in VERDICTS))
    return "\n".join(lines)


_STYLE = ("body{font-family:sans-serif;max-width:1050px;margin:auto}section{border-bottom:1px solid #ccc;padding:8px 0}"
          "img{max-width:100%;display:block}.opt{margin-right:6px}.opt input{display:none}"
          ".opt span{display:inline-block;padding:5px 12px;border:1px solid #888;border-radius:4px;cursor:pointer}"
          ".opt input:checked+span{background:#2a6;color:#fff;border-color:#2a6}section.active{background:#fffbe0}")

_SCRIPT = r"""
let active=null;
function save(id,v){return fetch('/answer',{method:'POST',body:JSON.stringify({item_id:id,verdict:v})})
 .then(r=>r.json()).then(j=>{document.getElementById('done').textContent=j.answered})}
document.addEventListener('change',e=>{if(e.target.type==='radio')save(e.target.name,e.target.value)});
document.addEventListener('mouseover',e=>{const s=e.target.closest('section');if(s&&s!==active){
 if(active)active.classList.remove('active');active=s;s.classList.add('active')}});
document.addEventListener('keydown',e=>{if(!active)return;const k=parseInt(e.key,10)-1;
 const inputs=active.querySelectorAll('input[type=radio]');if(k>=0&&k<inputs.length){inputs[k].checked=true;save(active.id,inputs[k].value)}});
"""


def _page_html(sample: List[dict], answers: Dict[str, str], class_name: str) -> str:
    sections = ""
    for number, item in enumerate(sample, 1):
        buttons = "".join(f'<label class="opt"><input type="radio" name="{item["item_id"]}" value="{v}"'
                          f'{" checked" if answers.get(item["item_id"]) == v else ""}><span>{n}. {label}</span></label>'
                          for n, (v, label) in enumerate(VERDICTS, 1))
        sections += (f'<section id="{item["item_id"]}"><h3>#{number} {item["image"]}（信心 {item["conf"]:.2f}）</h3>'
                     f'<img loading="lazy" src="/img/{item["item_id"]}.jpg"><div>{buttons}</div></section>')
    return ('<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8"><title>誤報抽查</title>'
            f'<style>{_STYLE}</style></head><body><h1>誤報抽查（{class_name}）</h1>'
            f'<p>共 {len(sample)} 題；已答 <span id="done">{sum(i["item_id"] in answers for i in sample)}</span> 題。'
            f'紅框是模型的 {class_name} 預測（沒有對上任何同類別 GT），綠框是 GT（標類別）。判斷紅框裡是什麼：'
            '是目標類別但 GT 沒框＝漏標；不是任何道路使用者＝真的誤報；是其他類別（例如騎車的人依指南算 scooter）＝其他類別。'
            '點選即儲存；滑鼠移到某題後可按 1～5。灰色區域是遮罩。</p>'
            f'{sections}<script>{_SCRIPT}</script></body></html>')


def make_server(host: str, port: int, token: str, sample: List[dict], export_root: Path, names: Dict[int, str],
                responses_path: Path, class_name: str) -> ThreadingHTTPServer:
    if not token:
        raise ValueError("token 不可為空：沒有權杖的服務會把 CCTV 影像開放給任何連得到的人")
    by_id = {item["item_id"]: item for item in sample}
    cache: Dict[str, bytes] = {}

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
            if not self._authorized(parse_qs(parts.query)):
                self._send(401, "需要存取權杖：請使用啟動服務時印出的網址。".encode("utf-8"))
                return
            cookie = f"{COOKIE_NAME}={token}; HttpOnly; SameSite=Strict; Path=/"
            try:
                if parts.path == "/":
                    self._send(200, _page_html(sample, load_latest(responses_path), class_name).encode("utf-8"), cookie=cookie)
                elif parts.path.startswith("/img/") and parts.path.endswith(".jpg") and parts.path[5:-4] in by_id:
                    item_id = parts.path[5:-4]
                    if item_id not in cache:
                        cache[item_id] = render_item(by_id[item_id], export_root, names)
                    self._send(200, cache[item_id], "image/jpeg", cookie)
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
                item_id, verdict = record["item_id"], record["verdict"]
                if item_id not in by_id or verdict not in dict(VERDICTS):
                    raise ValueError("bad answer")
            except (ValueError, KeyError, TypeError):
                self._send(400, b"")
                return
            with responses_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"item_id": item_id, "verdict": verdict, "ts": time.time()}) + "\n")
            answered = sum(i in by_id for i in load_latest(responses_path))
            self._send(200, json.dumps({"answered": answered}).encode("utf-8"), "application/json")

        def log_message(self, format: str, *args) -> None:
            print(redact_token(f"{self.address_string()} - {format % args}"), flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp", type=Path, required=True, help="count_detection_errors.py --dump-fp output")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--class-name", default="person", help="class the false positives belong to (page text only)")
    parser.add_argument("--export", default="ft_v3", help="export whose images/val and labels/val the list refers to")
    parser.add_argument("--strata", type=float, nargs="+", default=[0.25, 0.5], help="confidence edges between strata")
    parser.add_argument("--per-stratum", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", type=Path, default=REVIEW_DIR / "fp_review_sample.json", help="created on first run, then reused")
    parser.add_argument("--responses", type=Path, default=REVIEW_DIR / "fp_review_responses.jsonl")
    parser.add_argument("--summary", action="store_true", help="print the weighted verdict rates and exit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    items = [item for item in json.loads(args.fp.read_text(encoding="utf-8")) if item["camera"] == args.camera]
    if not items:
        raise SystemExit(f"no false positives for camera {args.camera} in {args.fp}")
    edges = sorted(args.strata)
    if args.sample.exists():
        sample = json.loads(args.sample.read_text(encoding="utf-8"))
    else:
        sample = draw_sample(items, edges, args.per_stratum, args.seed)
        args.sample.write_text(json.dumps(sample, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.summary:
        print(summarize(items, sample, load_latest(args.responses), edges))
        return

    export_root = EXPORTS / args.export
    names = yaml.safe_load((export_root / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    token = args.token or load_or_create_token(TOKEN_PATH)
    server = make_server(args.host, args.port, token, sample, export_root, names, args.responses, args.class_name)
    print(f"誤報抽查服務：{args.host}:{args.port}，{len(sample)} 題（Ctrl+C 結束）", flush=True)
    print(f"網址：http://{socket.gethostname()}:{args.port}/?token={token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
