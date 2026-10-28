#!/usr/bin/env python3
"""
產生鏡頭 ROI 的離線點選頁（單檔 HTML，首幀以 base64 內嵌）。

伺服器沒有圖形介面，無法用 OpenCV／matplotlib 的滑鼠事件點選；改為輸出 HTML，
在任何瀏覽器開啟後依序點出多邊形頂點，頁面會即時產生 config/roi/<camera>.yaml 的內容，
可複製或下載。輸出放在 gitignore 的 runs/roi_picker/，CCTV 畫面不進 git、不上傳。

用法：
    python scripts/picker/make_roi_picker.py --camera C000054
    python scripts/picker/make_roi_picker.py --camera C000054 --out-dir runs/roi_picker
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2

from safetrajai.evaluation.dataset_layout import load_dataset_registry
from safetrajai.paths import PROJECT_ROOT

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>ROI 點選：__CAMERA__</title>
<style>
  body { font-family: sans-serif; margin: 16px; background: #fafafa; color: #222; }
  #wrap { position: relative; display: inline-block; max-width: 100%; }
  canvas { max-width: 100%; height: auto; cursor: crosshair; border: 1px solid #999; }
  button { margin-right: 8px; padding: 6px 12px; font-size: 14px; }
  pre { background: #fff; border: 1px solid #ccc; padding: 8px; }
  .hint { color: #555; }
</style>
</head>
<body>
<h1>ROI 點選：__CAMERA__</h1>
<p class="hint">依序點出多邊形頂點（建議順時針）。影像尺寸 __WIDTH__×__HEIGHT__，座標為原始像素。來源：__SOURCE__</p>
<p>
  <button id="undo">復原上一點</button>
  <button id="clear">清除重畫</button>
  <button id="copy">複製 YAML</button>
  <button id="download">下載 __CAMERA__.yaml</button>
</p>
<div id="wrap"><canvas id="cv" width="__WIDTH__" height="__HEIGHT__"></canvas></div>
<h2>config/roi/__CAMERA__.yaml</h2>
<pre id="yaml"></pre>
<script>
const CAMERA = __CAMERA_JSON__;
const W = __WIDTH__, H = __HEIGHT__;
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const img = new Image();
const pts = [];

function toYaml() {
  const lines = [
    "# config/roi/" + CAMERA + ".yaml — 鏡頭 " + CAMERA + " 的 ROI（關注區域，見 CONTEXT.md、docs/adr/0005）",
    "camera: " + CAMERA,
    "image_size: [" + W + ", " + H + "]",
    "polygon:",
  ];
  for (const [x, y] of pts) lines.push("  - [" + x + ", " + y + "]");
  return lines.join("\\n") + "\\n";
}

function draw() {
  ctx.drawImage(img, 0, 0, W, H);
  if (pts.length) {
    ctx.lineWidth = Math.max(2, W / 400);
    ctx.strokeStyle = "#00ff00";
    ctx.fillStyle = "rgba(0, 255, 0, 0.15)";
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (pts.length >= 3) { ctx.closePath(); ctx.fill(); }
    ctx.stroke();
    ctx.fillStyle = "#ff0000";
    ctx.font = Math.max(12, W / 80) + "px sans-serif";
    pts.forEach(([x, y], i) => {
      ctx.beginPath(); ctx.arc(x, y, Math.max(3, W / 300), 0, 2 * Math.PI); ctx.fill();
      ctx.fillText("P" + (i + 1) + "(" + x + "," + y + ")", x + 6, y - 6);
    });
  }
  document.getElementById("yaml").textContent = pts.length >= 3 ? toYaml() : "（至少需要 3 個頂點）";
}

cv.addEventListener("click", (e) => {
  const r = cv.getBoundingClientRect();
  const x = Math.round((e.clientX - r.left) * W / r.width);
  const y = Math.round((e.clientY - r.top) * H / r.height);
  pts.push([Math.min(Math.max(x, 0), W), Math.min(Math.max(y, 0), H)]);
  draw();
});
document.getElementById("undo").onclick = () => { pts.pop(); draw(); };
document.getElementById("clear").onclick = () => { pts.length = 0; draw(); };
document.getElementById("copy").onclick = () => {
  if (pts.length >= 3) navigator.clipboard.writeText(toYaml());
};
document.getElementById("download").onclick = () => {
  if (pts.length < 3) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([toYaml()], { type: "text/yaml" }));
  a.download = CAMERA + ".yaml";
  a.click();
  URL.revokeObjectURL(a.href);
};
img.onload = draw;
img.src = "data:image/jpeg;base64,__IMAGE_B64__";
</script>
</body>
</html>
"""


def raw_videos_root() -> Path:
    registry = load_dataset_registry()
    raw_roots = {entry["root"] for entry in registry.values() if entry.get("layout") == "raw"}
    if len(raw_roots) != 1:
        raise ValueError(f"datasets.yaml 中 layout: raw 的根目錄應該恰好一個，實際為 {sorted(raw_roots)}")
    return (PROJECT_ROOT / raw_roots.pop()).resolve()


def first_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"無法讀取首幀：{video_path}")
    return frame


def render_page(camera: str, frame, source_name: str) -> str:
    height, width = frame.shape[:2]
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("首幀 JPEG 編碼失敗")
    replacements = {
        "__CAMERA_JSON__": json.dumps(camera),
        "__CAMERA__": camera,
        "__WIDTH__": str(width),
        "__HEIGHT__": str(height),
        "__SOURCE__": source_name,
        "__IMAGE_B64__": base64.b64encode(jpeg.tobytes()).decode("ascii"),
    }
    page = PAGE_TEMPLATE
    for key, value in replacements.items():
        page = page.replace(key, value)
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="鏡頭編號，例如 C000054")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "runs" / "roi_picker"))
    args = parser.parse_args()

    videos = sorted((raw_videos_root() / args.camera).glob("*.mp4"))
    if not videos:
        raise SystemExit(f"找不到鏡頭 {args.camera} 的原始影片（{raw_videos_root() / args.camera}）")

    frame = first_frame(videos[0])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.camera}.html"
    out_path.write_text(render_page(args.camera, frame, videos[0].name), encoding="utf-8")
    height, width = frame.shape[:2]
    print(f"{args.camera}: {width}x{height} -> {out_path}")


if __name__ == "__main__":
    main()
