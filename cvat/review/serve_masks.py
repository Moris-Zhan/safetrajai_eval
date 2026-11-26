#!/usr/bin/env python3
"""
Token-protected web page to confirm and edit the suggested "unreliable annotation" mask polygons
(from suggest_masks.py), one camera at a time.

The page loads the suggested polygons (or the saved confirmed ones) on top of a sampled frame; the
user drags vertices, deletes a vertex (double-click) or a whole polygon, adds a vertex on an edge
(shift-click), draws a new polygon and saves. Saved masks go to
runs/cvat_review/phase2/masks_confirmed/<camera>.json; nothing is written to CVAT.

Deployment follows serve_audit.py: token required (runs/cvat_review/server_token), pages built on
request, images stay on this host, plain HTTP so use it on a trusted network or an SSH tunnel.
Run with the conda env track_eval (needs cv2):
    python cvat/review/serve_masks.py --host 0.0.0.0 --port 8768
"""
from __future__ import annotations

import argparse
import hmac
import html
import json
import socket
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlsplit

import cv2

from safetrajai.paths import PROJECT_ROOT
from safetrajai.picker.server import load_or_create_token, redact_token

REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
PHASE2_DIR = REVIEW_DIR / "phase2"
SUGGESTED_DIR = PHASE2_DIR / "masks_suggested"
CONFIRMED_DIR = PHASE2_DIR / "masks_confirmed"
IMAGES_ROOT = PROJECT_ROOT / "dataset" / "detection" / "images"
TOKEN_PATH = REVIEW_DIR / "server_token"
COOKIE_NAME = "cvat_masks_token"
JPEG_QUALITY = 88
MAX_POLYGONS = 100
MAX_POINTS = 400


def camera_of(sequence: str) -> str:
    return sequence.split(" ")[0]


class MaskData:
    def __init__(self, phase2_dir: Path, suggested_dir: Path, confirmed_dir: Path) -> None:
        self.phase2_dir = phase2_dir
        self.confirmed_dir = confirmed_dir
        self.suggested: Dict[str, dict] = {
            path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(suggested_dir.glob("*.json"))
        }
        self.frames: Dict[str, List[dict]] = {camera: [] for camera in self.suggested}
        for item in json.loads((phase2_dir / "sample.json").read_text(encoding="utf-8")):
            camera = camera_of(item["seq"])
            if camera in self.frames:
                self.frames[camera].append(item)

    def current(self, camera: str) -> dict:
        saved = self.confirmed_dir / f"{camera}.json"
        if saved.exists():
            return {**json.loads(saved.read_text(encoding="utf-8")), "confirmed": True}
        return {**self.suggested[camera], "confirmed": False}

    def validate_and_save(self, camera: str, polygons: object) -> int:
        width, height = self.suggested[camera]["image_size"]
        if not isinstance(polygons, list) or len(polygons) > MAX_POLYGONS:
            raise ValueError("polygons must be a list")
        clean: List[List[List[float]]] = []
        for polygon in polygons:
            if not isinstance(polygon, list) or not 3 <= len(polygon) <= MAX_POINTS:
                raise ValueError("each polygon needs 3..%d points" % MAX_POINTS)
            points = []
            for point in polygon:
                x, y = float(point[0]), float(point[1])
                if not (-1 <= x <= width + 1 and -1 <= y <= height + 1):
                    raise ValueError("point outside the image")
                points.append([round(min(max(x, 0.0), float(width)), 1), round(min(max(y, 0.0), float(height)), 1)])
            clean.append(points)
        self.confirmed_dir.mkdir(parents=True, exist_ok=True)
        record = {"camera": camera, "image_size": [width, height], "polygons": clean,
                  "source": "suggest_masks.py edited in serve_masks.py", "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (self.confirmed_dir / f"{camera}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        return len(clean)

    def frame_jpeg(self, camera: str, index: int, boxes: bool) -> bytes:
        item = self.frames[camera][index]
        if boxes:
            image = cv2.imread(str(self.phase2_dir / item["image"]))
        else:
            image = cv2.imread(str(IMAGES_ROOT / item["seq"] / f"{item['frame_idx']:06d}.jpg"))
        if image is None:
            raise FileNotFoundError(item["image"])
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return jpeg.tobytes()


def _index_html(data: MaskData) -> str:
    rows = []
    for camera in data.suggested:
        state = data.current(camera)
        rows.append(f'<tr><td><a href="/cam/{camera}">{html.escape(camera)}</a></td><td>{len(data.frames[camera])}</td>'
                    f'<td>{len(state["polygons"])}</td><td>{"已確認" if state["confirmed"] else "建議（未確認）"}</td></tr>')
    done = sum(1 for camera in data.suggested if (data.confirmed_dir / f"{camera}.json").exists())
    return ('<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>遮罩確認</title></head><body style="font-family:sans-serif;margin:16px">'
            f'<h1>標註不可靠區域（遮罩）確認</h1><p>已確認 {done} / {len(data.suggested)} 個鏡頭。點鏡頭進入編輯；'
            '紅色區域在訓練時會塗灰，區域內的框會移除。</p>'
            '<table border="1" cellpadding="6"><tr><th>鏡頭</th><th>抽樣幀</th><th>多邊形</th><th>狀態</th></tr>'
            + "".join(rows) + '</table></body></html>')


PAGE_TEMPLATE = """<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>遮罩確認 __CAMERA__</title><style>
body{font-family:sans-serif;margin:12px}#bar{position:sticky;top:0;background:#fff;padding:6px;border-bottom:1px solid #aaa;z-index:2}
button{margin:2px;padding:6px 10px}canvas{width:100%;max-width:1400px;border:1px solid #888;touch-action:none;cursor:crosshair}
#msg{margin-left:8px;color:#06c}.help{font-size:13px;color:#444;max-width:1400px}</style></head><body>
<div id="bar"><a href="/">« 鏡頭清單</a> ｜ <b>__CAMERA__</b>
 <button id="prev">上一幀</button><span id="fl"></span><button id="next">下一幀</button>
 <label><input type="checkbox" id="boxes"> 顯示現有框</label>
 <button id="draw">新增多邊形</button><button id="del">刪除選取的多邊形</button><button id="reset">還原建議</button>
 <button id="save"><b>儲存（確認此鏡頭）</b></button><span id="msg"></span></div>
<p class="help">紅色＝訓練時要遮掉的「標註不可靠區域」，應該只涵蓋<b>有物體沒被框到</b>的地方（遠處車隊、天橋陰影下等）。
拖動頂點調整；雙擊頂點刪除；Shift＋點邊線新增頂點；點多邊形內部選取，Delete 鍵刪除選取的多邊形。「新增多邊形」後依序點出頂點，
點回第一個點或按 Enter 完成，Esc 取消。換幀時多邊形不變（只是換背景看）。儲存後狀態變為「已確認」，可再修改再存。</p>
<canvas id="c"></canvas>
<script>
const CAM="__CAMERA__";let W=0,H=0,polys=[],suggested=[],sel=-1,drag=null,drawing=null,frame=0,N=0,dirty=false,img=new Image();
const cv=document.getElementById("c"),ctx=cv.getContext("2d"),msg=document.getElementById("msg");
function say(t){msg.textContent=t}
function pos(e){const r=cv.getBoundingClientRect();return [(e.clientX-r.left)*W/r.width,(e.clientY-r.top)*H/r.height]}
function tol(){return 10*W/cv.getBoundingClientRect().width}
function inside(p,poly){let c=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const a=poly[i],b=poly[j];
 if((a[1]>p[1])!=(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])c=!c}return c}
function segDist(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],l=dx*dx+dy*dy;let t=l?((p[0]-a[0])*dx+(p[1]-a[1])*dy)/l:0;t=Math.max(0,Math.min(1,t));
 return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)}
function draw(){ctx.clearRect(0,0,W,H);ctx.drawImage(img,0,0,W,H);
 polys.forEach((poly,i)=>{ctx.beginPath();poly.forEach((p,k)=>k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();
  ctx.fillStyle=i==sel?"rgba(255,0,0,0.40)":"rgba(255,0,0,0.25)";ctx.fill();ctx.lineWidth=i==sel?3:2;ctx.strokeStyle="#f00";ctx.stroke();
  poly.forEach(p=>{ctx.fillStyle=i==sel?"#ff0":"#fff";ctx.strokeStyle="#000";ctx.lineWidth=1;ctx.beginPath();ctx.arc(p[0],p[1],5,0,7);ctx.fill();ctx.stroke()})});
 if(drawing){ctx.strokeStyle="#0af";ctx.lineWidth=2;ctx.beginPath();drawing.forEach((p,k)=>k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();
  drawing.forEach(p=>{ctx.fillStyle="#0af";ctx.beginPath();ctx.arc(p[0],p[1],5,0,7);ctx.fill()})}}
function loadFrame(){img=new Image();img.onload=()=>{draw()};img.src="/img/"+CAM+"/"+frame+".jpg?boxes="+(document.getElementById("boxes").checked?1:0);
 document.getElementById("fl").textContent=" "+(frame+1)+" / "+N+" "}
function hitVertex(p){for(let i=0;i<polys.length;i++)for(let k=0;k<polys[i].length;k++)if(Math.hypot(p[0]-polys[i][k][0],p[1]-polys[i][k][1])<tol())return [i,k];return null}
function finish(){if(drawing&&drawing.length>=3){polys.push(drawing);sel=polys.length-1;dirty=true}drawing=null;draw()}
cv.addEventListener("mousedown",e=>{const p=pos(e);
 if(drawing){if(drawing.length>=3&&Math.hypot(p[0]-drawing[0][0],p[1]-drawing[0][1])<tol()){finish()}else{drawing.push(p);draw()}return}
 const v=hitVertex(p);if(v){sel=v[0];drag=v;draw();return}
 if(e.shiftKey&&sel>=0){const poly=polys[sel];for(let k=0;k<poly.length;k++){const a=poly[k],b=poly[(k+1)%poly.length];
  if(segDist(p,a,b)<tol()){poly.splice(k+1,0,p);drag=[sel,k+1];dirty=true;draw();return}}}
 sel=-1;for(let i=polys.length-1;i>=0;i--)if(inside(p,polys[i])){sel=i;break}draw()});
cv.addEventListener("mousemove",e=>{if(!drag)return;const p=pos(e);polys[drag[0]][drag[1]]=[Math.max(0,Math.min(W,p[0])),Math.max(0,Math.min(H,p[1]))];dirty=true;draw()});
window.addEventListener("mouseup",()=>{drag=null});
cv.addEventListener("dblclick",e=>{const v=hitVertex(pos(e));if(v&&polys[v[0]].length>3){polys[v[0]].splice(v[1],1);dirty=true;draw()}});
window.addEventListener("keydown",e=>{if(e.key=="Delete"&&sel>=0){polys.splice(sel,1);sel=-1;dirty=true;draw()}
 else if(e.key=="Enter"&&drawing)finish();else if(e.key=="Escape"&&drawing){drawing=null;draw()}});
document.getElementById("prev").onclick=()=>{frame=(frame+N-1)%N;loadFrame()};
document.getElementById("next").onclick=()=>{frame=(frame+1)%N;loadFrame()};
document.getElementById("boxes").onchange=loadFrame;
document.getElementById("draw").onclick=()=>{drawing=[];sel=-1;say("依序點出頂點；點回第一個點或按 Enter 完成")};
document.getElementById("del").onclick=()=>{if(sel>=0){polys.splice(sel,1);sel=-1;dirty=true;draw()}};
document.getElementById("reset").onclick=()=>{if(confirm("放棄目前編輯，還原成自動建議？")){polys=JSON.parse(JSON.stringify(suggested));sel=-1;dirty=true;draw()}};
document.getElementById("save").onclick=async()=>{const r=await fetch("/save/"+CAM,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({polygons:polys})});
 if(r.ok){dirty=false;say("已儲存（"+polys.length+" 個多邊形）")}else{say("儲存失敗："+r.status)}};
window.addEventListener("beforeunload",e=>{if(dirty){e.preventDefault();e.returnValue=""}});
fetch("/api/"+CAM).then(r=>r.json()).then(j=>{W=j.image_size[0];H=j.image_size[1];cv.width=W;cv.height=H;polys=j.polygons;suggested=j.suggested;N=j.frames;
 say(j.confirmed?"已確認過的版本":"自動建議（尚未確認）");loadFrame()});
</script></body></html>"""


def make_server(host: str, port: int, token: str, data: MaskData) -> ThreadingHTTPServer:
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
                    self._send(200, _index_html(data).encode("utf-8"), cookie=cookie)
                elif len(segments) == 3 and segments[1] == "cam" and segments[2] in data.suggested:
                    self._send(200, PAGE_TEMPLATE.replace("__CAMERA__", segments[2]).encode("utf-8"), cookie=cookie)
                elif len(segments) == 3 and segments[1] == "api" and segments[2] in data.suggested:
                    state = data.current(segments[2])
                    body = {"image_size": state["image_size"], "polygons": state["polygons"], "confirmed": state["confirmed"],
                            "suggested": data.suggested[segments[2]]["polygons"], "frames": len(data.frames[segments[2]])}
                    self._send(200, json.dumps(body).encode("utf-8"), "application/json", cookie)
                elif len(segments) == 4 and segments[1] == "img" and segments[2] in data.suggested and segments[3].endswith(".jpg"):
                    index = int(segments[3][:-4])
                    if not 0 <= index < len(data.frames[segments[2]]):
                        self._send(404, "找不到".encode("utf-8"))
                        return
                    jpeg = data.frame_jpeg(segments[2], index, query.get("boxes", ["0"])[0] == "1")
                    self._send(200, jpeg, "image/jpeg", cookie)
                else:
                    self._send(404, "找不到".encode("utf-8"))
            except Exception as error:
                self.log_error("page build failed: %r", error)
                self._send(500, "<p>伺服器產生頁面時發生錯誤，詳見伺服器日誌。</p>".encode("utf-8"))

        def do_POST(self) -> None:
            segments = urlsplit(self.path).path.split("/")
            if len(segments) != 3 or segments[1] != "save" or segments[2] not in data.suggested:
                self._send(404, b"")
                return
            if not self._authorized({}):
                self._send(401, b"")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 2_000_000:
                    raise ValueError("body too large")
                count = data.validate_and_save(segments[2], json.loads(self.rfile.read(length))["polygons"])
            except (ValueError, KeyError, TypeError, IndexError):
                self._send(400, b"")
                return
            self._send(200, json.dumps({"saved": count}).encode("utf-8"), "application/json")

        def log_message(self, format: str, *args) -> None:
            print(redact_token(f"{self.address_string()} - {format % args}"), flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    data = MaskData(PHASE2_DIR, SUGGESTED_DIR, CONFIRMED_DIR)
    token = args.token or load_or_create_token(TOKEN_PATH)
    server = make_server(args.host, args.port, token, data)
    print(f"遮罩確認服務：{args.host}:{args.port}（Ctrl+C 結束）", flush=True)
    print(f"網址（以可連到這台主機的位址取代主機名稱）：http://{socket.gethostname()}:{args.port}/?token={token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
