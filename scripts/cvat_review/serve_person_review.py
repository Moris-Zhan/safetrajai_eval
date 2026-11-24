#!/usr/bin/env python3
"""
Token-protected web page to review person candidates mined from raw videos (mine_person_frames.py).

Each frame shows a zoomed crop around dennis' person boxes (numbered) and asks, per box: class buttons plus a status dropdown (normal / false detection / can't tell / partial box), plus one checkbox for "a person in this frame has no box".
A box can be corrected by dragging a new rectangle (stored as `corrections`, original image pixels).
Answers are appended to runs/cvat_review/person_review_responses.jsonl (latest answer per frame wins).

Run with the conda env track_eval (needs cv2):
    python scripts/cvat_review/serve_person_review.py --host 0.0.0.0 --port 8770
"""
from __future__ import annotations

import argparse
import html
import hmac
import json
import socket
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

import cv2

from safetrajai.paths import PROJECT_ROOT
from safetrajai.picker.server import load_or_create_token, redact_token

REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
CANDIDATES_PATH = REVIEW_DIR / "person_candidates.json"
RESPONSES_PATH = REVIEW_DIR / "person_review_responses.jsonl"
VIDEOS = PROJECT_ROOT / "dataset" / "raw" / "videos"
TOKEN_PATH = REVIEW_DIR / "server_token"
COOKIE_NAME = "person_review_token"
CLASS_BUTTONS = [("person", "人"), ("scooter", "機車"), ("car", "car"), ("other", "其他")]
OTHER_CLASSES = [("small_truck", "small_truck"), ("bus", "bus"), ("truck", "truck")]
STATUS_CHOICES = [("normal", "正常"), ("none", "誤檢"), ("unsure", "看不清"), ("partial", "框不完整")]
DEFAULT_VERDICT = "person"
LEGACY_VERDICTS = {"not_person": "none"}
CLASS_VALUES = ("person", "scooter", "car", "small_truck", "bus", "truck")
VERDICTS = (set(CLASS_VALUES) | {"none", "unsure", "partial"} | {f"partial:{c}" for c in CLASS_VALUES} | set(LEGACY_VERDICTS))
MIN_CROP_HEIGHT_PX = 420
PAD_PX = 50
MAX_CROP_WIDTH_PX = 1100


def split_verdict(value: str) -> Tuple[str, str]:
    value = LEGACY_VERDICTS.get(value, value) or DEFAULT_VERDICT
    if value in ("none", "unsure"):
        return value, DEFAULT_VERDICT
    if value == "partial":
        return "partial", DEFAULT_VERDICT
    if value.startswith("partial:"):
        return "partial", value.split(":", 1)[1]
    return "normal", value


def parse_corrections(raw: Dict[str, List[float]], box_count: int) -> Dict[str, List[float]]:
    corrections: Dict[str, List[float]] = {}
    for key, value in raw.items():
        index = int(key)
        x1, y1, x2, y2 = (float(v) for v in value)
        if not 0 <= index < box_count or not (x2 > x1 and y2 > y1):
            raise ValueError("bad correction")
        corrections[str(index)] = [x1, y1, x2, y2]
    return corrections


def crop_geometry(boxes: List[List[float]], width: int, height: int) -> Tuple[int, int, int, int, float]:
    left = max(0, int(min(b[0] for b in boxes)) - PAD_PX)
    top = max(0, int(min(b[1] for b in boxes)) - PAD_PX)
    right = min(width, int(max(b[2] for b in boxes)) + PAD_PX)
    bottom = min(height, int(max(b[3] for b in boxes)) + PAD_PX)
    scale = max(1.0, MIN_CROP_HEIGHT_PX / (bottom - top))
    if (right - left) * scale > MAX_CROP_WIDTH_PX:
        scale = MAX_CROP_WIDTH_PX / (right - left)
    return left, top, right, bottom, scale


def render_frame(candidate: dict) -> Tuple[bytes, dict]:
    capture = cv2.VideoCapture(str(VIDEOS / candidate["camera"] / candidate["video"]))
    capture.set(cv2.CAP_PROP_POS_FRAMES, candidate["frame_idx"])
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {candidate['frame_idx']} of {candidate['video']}")
    height, width = image.shape[:2]
    left, top, right, bottom, scale = crop_geometry(candidate["persons"], width, height)
    crop = cv2.resize(image[top:bottom, left:right], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
    return jpeg, {"left": left, "top": top, "scale": scale}


def load_latest(path: Path) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["frame_id"]] = record
    return latest


def _box_row(frame_id: str, index: int, conf: float, current: str, starred: bool) -> str:
    other_values = {"small_truck", "bus", "truck"}
    status, chosen = split_verdict(current)
    chosen_button = "other" if chosen in other_values else chosen
    name = f"{frame_id}-{index}"
    buttons = "".join(
        f'<label class="opt"><input type="radio" name="{name}" value="{value}"{" checked" if chosen_button == value else ""}'
        f' onchange="save(\'{frame_id}\')"><span>{label}</span></label>' for value, label in CLASS_BUTTONS)
    options = "".join(f'<option value="{value}"{" selected" if chosen == value else ""}>{label}</option>' for value, label in OTHER_CLASSES)
    states = "".join(f'<option value="{value}"{" selected" if status == value else ""}>{label}</option>' for value, label in STATUS_CHOICES)
    star = '<b style="color:#c00">★</b> ' if starred else ""
    dim = " dim" if status in ("none", "unsure") else ""
    return (f'<div class="row{dim}" data-box="{index}">{star}框 {index + 1}（{conf:.2f}）{buttons}'
            f'<select class="other" onchange="pickOther(this)">{options}</select>'
            f' 狀態：<select class="status" onchange="pickStatus(this)">{states}</select>'
            f' <button type="button" onclick="startEdit(this)">改框</button>'
            f'<button type="button" class="reset" onclick="resetCorr(this)">重設</button><b class="corrnote"></b></div>')


def _frame_html(frame_id: str, number: int, candidate: dict, answer: dict, targets: Dict[str, str]) -> str:
    boxes = answer.get("boxes", {})
    rows = ""
    for index, (_, _, _, _, conf) in enumerate(candidate["persons"]):
        rows += _box_row(frame_id, index, conf, boxes.get(str(index), ""), str(index) in targets)
    missed = " checked" if answer.get("missed") else ""
    originals = html.escape(json.dumps([[round(v, 1) for v in b[:4]] for b in candidate["persons"]]), quote=True)
    corrections = html.escape(json.dumps(answer.get("corrections", {})), quote=True)
    return (f'<section id="{frame_id}" data-boxes="{originals}" data-corr="{corrections}">'
            f'<h3>#{number} {candidate["camera"]} 幀 {candidate["frame_idx"]}</h3><div class="hint"></div>'
            f'<div class="wrap"><img loading="lazy" src="/img/{frame_id}.jpg"><canvas></canvas></div>{rows}'
            f'<div><label><input type="checkbox" name="{frame_id}-missed"{missed} onchange="save(\'{frame_id}\')">'
            f'有人沒被框到</label></div></section>')


_STYLE = ("body{font-family:sans-serif;max-width:1150px;margin:auto}section{border-bottom:1px solid #ccc;padding:8px 0}"
          ".wrap{position:relative;display:inline-block;max-width:100%}.wrap img{max-width:100%;display:block}"
          ".wrap canvas{position:absolute;left:0;top:0;pointer-events:none}.hint{color:#06c;min-height:1.2em}"
          ".row{margin:4px 0}.row.active{background:#fffbe0}.opt{margin-right:6px}.opt input{display:none}"
          ".opt span{display:inline-block;padding:5px 12px;border:1px solid #888;border-radius:4px;cursor:pointer}"
          ".opt input:checked+span{background:#2a6;color:#fff;border-color:#2a6}"
          ".row.dim .opt,.row.dim .other{opacity:.35}select{padding:4px}.corrnote{color:#080;margin-left:6px}"
          ".row .reset{display:none}.row.fixed .reset{display:inline}")

_SCRIPT = r"""
let active=null,editing=null,drag=null;const state={};
const sec=id=>document.getElementById(id);
function boxValue(row){const st=row.querySelector('.status').value;
 if(st==='none'||st==='unsure')return st;
 const r=row.querySelector('input[type=radio]:checked').value;
 const cls=r==='other'?row.querySelector('.other').value:r;
 return st==='partial'?'partial:'+cls:cls}
function save(id){const s=sec(id);const boxes={};
 s.querySelectorAll('.row').forEach(row=>{boxes[row.dataset.box]=boxValue(row)});
 const missed=s.querySelector('input[type=checkbox]').checked;
 return fetch('/answer',{method:'POST',body:JSON.stringify({frame_id:id,boxes:boxes,missed:missed,corrections:state[id].corr})})
 .then(r=>r.json()).then(j=>{document.getElementById('done').textContent=j.answered})}
function dimRow(row){const st=row.querySelector('.status').value;row.classList.toggle('dim',st==='none'||st==='unsure')}
function pickOther(sel){const row=sel.closest('.row');row.querySelector('input[value=other]').checked=true;save(row.closest('section').id)}
function pickStatus(sel){const row=sel.closest('.row');dimRow(row);save(row.closest('section').id)}
function geom(id){const s=state[id];const img=sec(id).querySelector('img');
 return s.meta?{f:s.meta.scale*img.clientWidth/img.naturalWidth,left:s.meta.left,top:s.meta.top,img:img}:null}
function draw(id){const s=state[id],g=geom(id);if(!g)return;const cv=sec(id).querySelector('canvas');
 cv.width=g.img.clientWidth;cv.height=g.img.clientHeight;const c=cv.getContext('2d');c.clearRect(0,0,cv.width,cv.height);
 const rect=(b,color,dash)=>{c.strokeStyle=color;c.lineWidth=2;c.setLineDash(dash?[5,4]:[]);
  c.strokeRect((b[0]-g.left)*g.f,(b[1]-g.top)*g.f,(b[2]-b[0])*g.f,(b[3]-b[1])*g.f)};
 s.boxes.forEach((b,i)=>{const fixed=s.corr[i];rect(b,'#e22',!!fixed);
  c.font='bold 14px sans-serif';c.fillStyle='#ff0';c.strokeStyle='#000';c.lineWidth=3;c.setLineDash([]);
  const x=(b[0]-g.left)*g.f,y=Math.max(14,(b[1]-g.top)*g.f-3);c.strokeText(i+1,x,y);c.fillText(i+1,x,y);
  if(fixed)rect(fixed,'#0c0',false)});
 if(drag)rect(drag.rect,'#38f',false);
 sec(id).querySelectorAll('.row').forEach(row=>{const f=!!s.corr[row.dataset.box];
  row.classList.toggle('fixed',f);row.querySelector('.corrnote').textContent=f?'已修正':''})}
function startEdit(btn){const row=btn.closest('.row'),id=row.closest('section').id;
 if(!state[id].meta)return;editing={id:id,box:row.dataset.box};
 const cv=sec(id).querySelector('canvas');cv.style.pointerEvents='auto';cv.style.cursor='crosshair';
 sec(id).querySelector('.hint').textContent='在圖上拖出框 '+(+row.dataset.box+1)+' 的正確範圍（Esc 取消）'}
function endEdit(){if(!editing)return;const id=editing.id;const cv=sec(id).querySelector('canvas');
 cv.style.pointerEvents='none';sec(id).querySelector('.hint').textContent='';editing=null;drag=null;draw(id)}
function resetCorr(btn){const row=btn.closest('.row'),id=row.closest('section').id;
 delete state[id].corr[row.dataset.box];draw(id);save(id)}
function initFrame(s){const id=s.id;state[id]={boxes:JSON.parse(s.dataset.boxes),corr:JSON.parse(s.dataset.corr),meta:null};
 const img=s.querySelector('img'),cv=s.querySelector('canvas');
 const ready=()=>fetch('/meta/'+id+'.json').then(r=>r.json()).then(m=>{state[id].meta=m;draw(id)});
 if(img.complete&&img.naturalWidth)ready();else img.addEventListener('load',ready);
 const pos=e=>{const r=cv.getBoundingClientRect();return[Math.min(Math.max(e.clientX-r.left,0),cv.width),Math.min(Math.max(e.clientY-r.top,0),cv.height)]};
 const toOrig=(x,y,g)=>[x/g.f+g.left,y/g.f+g.top];
 cv.addEventListener('mousedown',e=>{if(!editing||editing.id!==id)return;const p=pos(e);drag={x:p[0],y:p[1],rect:[0,0,0,0]};e.preventDefault()});
 cv.addEventListener('mousemove',e=>{if(!drag)return;const g=geom(id),p=pos(e);
  const a=toOrig(drag.x,drag.y,g),b=toOrig(p[0],p[1],g);
  drag.rect=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];drag.px=[Math.abs(p[0]-drag.x),Math.abs(p[1]-drag.y)];draw(id)});
 window.addEventListener('mouseup',()=>{if(!drag||!editing||editing.id!==id)return;
  const ok=drag.px&&drag.px[0]>=6&&drag.px[1]>=6;
  if(ok){const r=drag.rect.map(v=>Math.round(v*10)/10);state[id].corr[editing.box]=r;
   const row=s.querySelector('.row[data-box="'+editing.box+'"]');row.querySelector('.status').value='normal';dimRow(row);endEdit();save(id)}
  else endEdit()});
 window.addEventListener('resize',()=>draw(id))}
document.querySelectorAll('section').forEach(initFrame);
function saveAll(){const ids=[...document.querySelectorAll('section')].map(s=>s.id);
 ids.reduce((p,id)=>p.then(()=>save(id)),Promise.resolve()).then(()=>alert('已把全部畫面上的答案（含預設的「人」）存起來'))}
document.addEventListener('mouseover',e=>{const row=e.target.closest('.row');if(row&&row!==active){
 if(active)active.classList.remove('active');active=row;row.classList.add('active')}});
document.addEventListener('keydown',e=>{
 if(e.key==='Escape'){endEdit();return}
 if(!active||e.target.tagName==='SELECT')return;
 const classes=['person','scooter','car','other'];const k=parseInt(e.key,10)-1;const key=e.key.toLowerCase();
 const states={q:'none',w:'unsure',e:'partial'};const st=active.querySelector('.status');
 if(k>=0&&k<classes.length){if(st.value==='none'||st.value==='unsure')st.value='normal';dimRow(active);
  active.querySelector('input[value='+classes[k]+']').checked=true;if(classes[k]==='other')active.querySelector('.other').focus()}
 else if(states[key]){st.value=st.value===states[key]?'normal':states[key];dimRow(active)}
 else if(key==='r'){startEdit(active.querySelector('button'));return}
 else return;save(active.closest('section').id)});
"""


def _page_html(candidates: List[dict], latest: Dict[str, dict], subset: Optional[Dict[str, Dict[str, str]]] = None) -> str:
    shown = [(n, c) for n, c in enumerate(candidates) if subset is None or f"f{n}" in subset]
    sections = "".join(_frame_html(f"f{n}", k + 1, c, latest.get(f"f{n}", {}), (subset or {}).get(f"f{n}", {}))
                       for k, (n, c) in enumerate(shown))
    return ('<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8"><title>行人候選審查</title>'
            f'<style>{_STYLE}</style></head><body>'
            f'<h1>行人候選審查</h1><p>共 {len(shown)} 張；已儲存 <span id="done">{len(latest)}</span> 張（全部累計）。'
            + ('★ 標示要回答的框。' if subset is not None else '') +
            '每個框預設「人」，只需改例外；點任何一個選項時，該張的所有框（含預設）一起儲存。'
            '左邊按鈕選類別（「其他」旁的下拉選 small_truck／bus／truck），右邊「狀態」下拉標例外：誤檢＝框裡沒有該標註的東西（類別不算）、看不清＝塗灰（類別不算）、框不完整＝類別照樣有效，但沒修框的整張會排除。'
            '要修框：按該框的「改框」，在圖上拖出正確範圍，修好的框（綠色）會變「正常」，整張可用；紅色虛線是原框，「重設」可還原。'
            '滑鼠移到某個框後可按鍵：1 人、2 機車、3 car、4 其他；Q 誤檢、W 看不清、E 框不完整（再按一次回正常）、R 改框、Esc 取消。'
            '<button onclick="saveAll()">全部確認（沒改的都當「人」）</button></p>'
            f'{sections}<script>{_SCRIPT}</script></body></html>')


def make_server(host: str, port: int, token: str, candidates: List[dict], responses_path: Path,
                subset: Optional[Dict[str, Dict[str, str]]] = None) -> ThreadingHTTPServer:
    if not token:
        raise ValueError("token 不可為空：沒有權杖的服務會把 CCTV 影像開放給任何連得到的人")
    cache: Dict[str, Tuple[bytes, dict]] = {}

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
                    self._send(200, _page_html(candidates, load_latest(responses_path), subset).encode("utf-8"), cookie=cookie)
                elif parts.path.startswith(("/img/f", "/meta/f")) and parts.path.endswith((".jpg", ".json")):
                    frame_id = parts.path.rsplit("/", 1)[1].rsplit(".", 1)[0]
                    index = int(frame_id[1:])
                    if frame_id not in cache:
                        cache[frame_id] = render_frame(candidates[index])
                    if parts.path.startswith("/img/"):
                        self._send(200, cache[frame_id][0], "image/jpeg", cookie)
                    else:
                        self._send(200, json.dumps(cache[frame_id][1]).encode("utf-8"), "application/json", cookie)
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
                frame_id, boxes, missed = record["frame_id"], record["boxes"], bool(record["missed"])
                index = int(frame_id[1:])
                if not frame_id.startswith("f") or not 0 <= index < len(candidates):
                    raise ValueError("unknown frame")
                if any(v not in VERDICTS or not 0 <= int(k) < len(candidates[index]["persons"]) for k, v in boxes.items()):
                    raise ValueError("bad box answer")
                corrections = parse_corrections(record.get("corrections", {}), len(candidates[index]["persons"]))
            except (ValueError, KeyError, TypeError):
                self._send(400, b"")
                return
            with responses_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"frame_id": frame_id, "boxes": boxes, "missed": missed, "corrections": corrections, "ts": time.time()}, ensure_ascii=False) + "\n")
            self._send(200, json.dumps({"answered": len(load_latest(responses_path))}).encode("utf-8"), "application/json")

        def log_message(self, format: str, *args) -> None:
            print(redact_token(f"{self.address_string()} - {format % args}"), flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--responses", type=Path, default=RESPONSES_PATH)
    parser.add_argument("--token", default=None)
    parser.add_argument("--subset", type=Path, default=None,
                        help='JSON {"f12": {"0": "pending"}}: show only these frames and mark these boxes')
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    subset = json.loads(args.subset.read_text(encoding="utf-8")) if args.subset else None
    token = args.token or load_or_create_token(TOKEN_PATH)
    server = make_server(args.host, args.port, token, candidates, args.responses, subset)
    print(f"行人候選審查服務：{args.host}:{args.port}（Ctrl+C 結束）", flush=True)
    print(f"網址：http://{socket.gethostname()}:{args.port}/?token={token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
