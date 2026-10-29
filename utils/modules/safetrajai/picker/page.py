"""Single-file HTML page for the web point picker (scripts/picker/make_picker.py).

The page works offline: the images are embedded as base64 and the JavaScript builds the YAML itself.
Its YAML functions live in the "pure logic" section, which has no DOM access so that the tests can run
it in Node and require the same text as safetrajai.picker.yaml_output.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

MODES = ("homography", "checkpoints", "counting-lines", "camera-position")
REQUIRED_IMAGES = {
    "homography": ("cctv", "satellite"),
    "checkpoints": ("cctv", "satellite"),
    "counting-lines": ("cctv",),
    "camera-position": ("satellite",),
}
MODE_TITLES = {
    "homography": "Homography 對應點",
    "checkpoints": "Homography 驗證點",
    "counting-lines": "計數線",
    "camera-position": "相機位置",
}

LOGIC_BEGIN = "// ==== BEGIN pure logic ===="
LOGIC_END = "// ==== END pure logic ===="


@dataclass(frozen=True)
class PickerImage:
    name: str
    width: int
    height: int
    jpeg_b64: str


PURE_LOGIC_JS = LOGIC_BEGIN + r"""
function formatPoint(p) { return "[" + p[0] + ", " + p[1] + "]"; }
function formatPoints(points) { return "[" + points.map(formatPoint).join(", ") + "]"; }

function homographyYaml(camera, cctv, sat) {
  return "# " + camera + "：Homography 對應點（CCTV 首幀像素 ↔ 衛星圖像素），貼進 config/homography/" + camera + ".yaml\n" +
    "cctv_points: " + formatPoints(cctv) + "\n" +
    "map_points: " + formatPoints(sat) + "\n";
}

function checkpointsYaml(camera, cctv, sat) {
  const lines = ["# " + camera + "：Homography 驗證點（check points），存成 config/homography_checkpoints/" + camera + ".yaml",
                 "cctv_points:"];
  cctv.forEach(function (p) { lines.push("  - " + formatPoint(p)); });
  lines.push("sat_points:");
  sat.forEach(function (p) { lines.push("  - " + formatPoint(p)); });
  return lines.join("\n") + "\n";
}

function cameraPositionYaml(camera, point) {
  return "# " + camera + "：相機在衛星圖上的位置（衛星圖像素），貼進 config/homography/" + camera + ".yaml\n" +
    "camera_view_point: " + formatPoint(point) + "\n";
}

function countingLinesYaml(camera, size, lines) {
  const out = ["# config/counting_lines/" + camera + ".yaml — 鏡頭 " + camera + " 的計數線（CCTV 首幀像素座標）",
               "camera: " + camera, "image_size: " + formatPoint(size), "lines:"];
  lines.forEach(function (line) {
    out.push("  - name: " + JSON.stringify(line.name));
    out.push("    points: " + formatPoints(line.points));
    if (line.forward_label) out.push("    forward_label: " + JSON.stringify(line.forward_label));
    if (line.backward_label) out.push("    backward_label: " + JSON.stringify(line.backward_label));
  });
  return out.join("\n") + "\n";
}

// Pair modes: points alternate between the first image (CCTV) and the second (satellite).
function nextSide(stack) { return stack.length % 2 === 0 ? "first" : "second"; }
function addPairedPoint(stack, side, point) { return side === nextSide(stack) ? stack.concat([point]) : stack; }
function undoPoint(stack) { return stack.slice(0, -1); }
function splitPairs(stack) {
  const first = [], second = [];
  stack.forEach(function (p, i) { (i % 2 === 0 ? first : second).push(p); });
  return { first: first.slice(0, second.length), second: second,
           pending: stack.length % 2 === 1 ? stack[stack.length - 1] : null };
}

// Counting lines: every two consecutive points are one line (p1 -> p2); meta[i] holds its name and labels.
function buildLines(stack, meta) {
  const lines = [];
  for (let i = 0; i + 1 < stack.length; i += 2) {
    const index = i / 2, info = (meta && meta[index]) || {};
    const line = { name: info.name !== undefined ? info.name : "line_" + (index + 1), points: [stack[i], stack[i + 1]] };
    if (info.forward) line.forward_label = info.forward;
    if (info.backward) line.backward_label = info.backward;
    lines.push(line);
  }
  return lines;
}

function readiness(mode, state) {
  const ok = function (warning) { return { ok: true, message: "", warning: warning || "" }; };
  const no = function (message) { return { ok: false, message: message, warning: "" }; };
  if (mode === "homography" || mode === "checkpoints") {
    const n = splitPairs(state.stack || []).second.length;
    if (mode === "homography") return n >= 4 ? ok() : no("至少需要 4 組對應點（目前 " + n + " 組）");
    if (n < 1) return no("至少需要 1 組驗證點");
    return ok(n < 4 ? "建議至少 4 組驗證點（目前 " + n + " 組）" : "");
  }
  if (mode === "camera-position") return state.point ? ok() : no("請在衛星圖上點出相機位置");
  const lines = buildLines(state.stack || [], state.meta || []);
  if (lines.length < 1) return no("至少需要 1 條計數線（每條線兩個點）");
  const names = lines.map(function (line) { return line.name.trim(); });
  if (names.some(function (name) { return name === ""; })) return no("計數線名稱不可空白");
  if (new Set(names).size !== names.length) return no("計數線名稱重複");
  return ok();
}
""" + LOGIC_END

UI_JS = r"""
const MODE = CONFIG.mode;
const PAIR_MODE = MODE === "homography" || MODE === "checkpoints";
const TARGETS = {
  "homography": "貼進 config/homography/" + CONFIG.camera + ".yaml",
  "checkpoints": "存成 config/homography_checkpoints/" + CONFIG.camera + ".yaml",
  "counting-lines": "存成 config/counting_lines/" + CONFIG.camera + ".yaml",
  "camera-position": "貼進 config/homography/" + CONFIG.camera + ".yaml",
};
const HINTS = {
  "homography": "先點左側 CCTV 的一個地面特徵，再點右側衛星圖的同一個地點，成對進行（至少 4 組，建議 8 組以上）。",
  "checkpoints": "與「對應點」相同的成對點法，但必須是和訓練點不同的額外地點（建議 8 組以上）。",
  "counting-lines": "在 CCTV 首幀每兩點畫一條線（p1 → p2）。黃色箭頭是 forward 的穿越方向（面向 p2，從左側穿越到右側）；方向標籤由程式依線的方向自動判斷，不必填寫。",
  "camera-position": "在衛星圖上點出相機所在位置（只需一個點，再點一次會取代）。",
};
const state = { stack: [], point: null, meta: [] };
const canvases = {}, images = {};

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }
function byId(id) { return document.getElementById(id); }

function setStatus(text) { byId("status").textContent = text; }

function makePanel(key) {
  const info = CONFIG.images[key];
  const panel = el("div", undefined, "panel");
  panel.appendChild(el("h3", (key === "cctv" ? "CCTV 首幀" : "衛星圖") + "：" + info.name + "（" + info.width + "×" + info.height + "）"));
  const cv = document.createElement("canvas");
  cv.width = info.width; cv.height = info.height;
  cv.addEventListener("click", function (event) { onClick(key, event, cv); });
  panel.appendChild(cv);
  byId("panels").appendChild(panel);
  canvases[key] = cv;
  const img = new Image();
  img.onload = redraw;
  img.src = "data:image/jpeg;base64," + info.b64;
  images[key] = img;
}

function onClick(key, event, cv) {
  const rect = cv.getBoundingClientRect(), info = CONFIG.images[key];
  const point = [clamp(Math.round((event.clientX - rect.left) * info.width / rect.width), 0, info.width),
                 clamp(Math.round((event.clientY - rect.top) * info.height / rect.height), 0, info.height)];
  if (PAIR_MODE) {
    const next = addPairedPoint(state.stack, key === "cctv" ? "first" : "second", point);
    if (next === state.stack) {
      setStatus("請先在" + (nextSide(state.stack) === "first" ? "左側 CCTV" : "右側衛星圖") + "點下一個點");
    } else { state.stack = next; setStatus(""); }
  } else if (MODE === "counting-lines") {
    state.stack = state.stack.concat([point]);
    renderLineForm();
  } else {
    state.point = point;
  }
  redraw();
}

function undo() {
  state.stack = undoPoint(state.stack);
  state.point = null;
  if (MODE === "counting-lines") renderLineForm();
  setStatus("");
  redraw();
}
function clearAll() {
  state.stack = []; state.point = null; state.meta = [];
  if (MODE === "counting-lines") renderLineForm();
  setStatus("");
  redraw();
}

function marker(ctx, point, label, color, size) {
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(point[0], point[1], size, 0, 2 * Math.PI); ctx.fill();
  ctx.font = Math.max(12, size * 4) + "px sans-serif";
  ctx.fillText(label + " (" + point[0] + "," + point[1] + ")", point[0] + size + 3, point[1] - size);
}

function drawBase(key) {
  const ctx = canvases[key].getContext("2d");
  ctx.drawImage(images[key], 0, 0, CONFIG.images[key].width, CONFIG.images[key].height);
}

function drawPairs() {
  const pairs = splitPairs(state.stack);
  [["cctv", pairs.first], ["satellite", pairs.second]].forEach(function (entry) {
    const key = entry[0], size = Math.max(3, CONFIG.images[key].width / 300);
    entry[1].forEach(function (p, i) { marker(canvases[key].getContext("2d"), p, "P" + (i + 1), "#ff2020", size); });
  });
  if (pairs.pending) {
    marker(canvases.cctv.getContext("2d"), pairs.pending, "P" + (pairs.first.length + 1) + "（待配對）", "#ff9900",
           Math.max(3, CONFIG.images.cctv.width / 300));
  }
}

// Arrow at the middle of the line along the forward travel direction (-dy, dx): left of p1 -> p2 to its right.
function drawForwardArrow(ctx, line, size) {
  const dx = line.points[1][0] - line.points[0][0], dy = line.points[1][1] - line.points[0][1];
  const length = Math.hypot(dx, dy);
  if (length === 0) return;
  const ux = -dy / length, uy = dx / length;
  const mx = (line.points[0][0] + line.points[1][0]) / 2, my = (line.points[0][1] + line.points[1][1]) / 2;
  const span = Math.max(20, CONFIG.images.cctv.width / 20), head = span / 3;
  const tx = mx + ux * span, ty = my + uy * span;
  ctx.lineWidth = Math.max(2, size / 1.5); ctx.strokeStyle = "#ffee00"; ctx.fillStyle = "#ffee00";
  ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(tx, ty); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(tx - ux * head + uy * head / 2, ty - uy * head - ux * head / 2);
  ctx.lineTo(tx - ux * head - uy * head / 2, ty - uy * head + ux * head / 2);
  ctx.closePath(); ctx.fill();
  ctx.font = Math.max(12, CONFIG.images.cctv.width / 80) + "px sans-serif";
  ctx.fillText("forward", tx + ux * 4 + 4, ty + uy * 4 + 4);
}

function drawLines() {
  const ctx = canvases.cctv.getContext("2d"), size = Math.max(3, CONFIG.images.cctv.width / 300);
  buildLines(state.stack, state.meta).forEach(function (line) {
    ctx.lineWidth = Math.max(2, size / 1.5); ctx.strokeStyle = "#00ff66";
    ctx.beginPath(); ctx.moveTo(line.points[0][0], line.points[0][1]); ctx.lineTo(line.points[1][0], line.points[1][1]); ctx.stroke();
    marker(ctx, line.points[0], line.name + " p1", "#ff2020", size);
    marker(ctx, line.points[1], "p2", "#ff2020", size);
    drawForwardArrow(ctx, line, size);
  });
  if (state.stack.length % 2 === 1) marker(ctx, state.stack[state.stack.length - 1], "p1（待 p2）", "#ff9900", size);
}

function drawViewPoint() {
  if (!state.point) return;
  marker(canvases.satellite.getContext("2d"), state.point, "相機", "#00e5ff", Math.max(5, CONFIG.images.satellite.width / 150));
}

function currentYaml() {
  if (PAIR_MODE) {
    const pairs = splitPairs(state.stack);
    return (MODE === "homography" ? homographyYaml : checkpointsYaml)(CONFIG.camera, pairs.first, pairs.second);
  }
  if (MODE === "camera-position") return cameraPositionYaml(CONFIG.camera, state.point);
  const info = CONFIG.images.cctv;
  return countingLinesYaml(CONFIG.camera, [info.width, info.height], buildLines(state.stack, state.meta));
}

function updateOutput() {
  const result = readiness(MODE, state);
  byId("yaml").textContent = result.ok ? currentYaml() : "（" + result.message + "）";
  byId("warning").textContent = result.warning;
  byId("copy").disabled = !result.ok;
  byId("download").disabled = !result.ok;
}

function redraw() {
  Object.keys(canvases).forEach(drawBase);
  if (PAIR_MODE) drawPairs(); else if (MODE === "counting-lines") drawLines(); else drawViewPoint();
  updateOutput();
}

function renderLineForm() {
  const form = byId("lineForm");
  form.textContent = "";
  buildLines(state.stack, state.meta).forEach(function (line, index) {
    const row = el("div", undefined, "line-row");
    [["name", "名稱", line.name]].forEach(function (field) {
      const label = el("label", field[1] + " ");
      const input = document.createElement("input");
      input.value = field[2];
      input.addEventListener("input", function () {
        state.meta[index] = state.meta[index] || {};
        state.meta[index][field[0]] = input.value;
        updateOutput();
      });
      label.appendChild(input);
      row.appendChild(label);
    });
    form.appendChild(row);
  });
}

byId("title").textContent = CONFIG.camera + "：" + TITLES[MODE];
byId("hint").textContent = HINTS[MODE];
byId("target").textContent = TARGETS[MODE];
CONFIG.notes.forEach(function (note) { byId("notes").appendChild(el("div", note, "note")); });
Object.keys(CONFIG.images).forEach(makePanel);
byId("undo").onclick = undo;
byId("clear").onclick = clearAll;
byId("copy").onclick = function () { navigator.clipboard.writeText(currentYaml()); setStatus("已複製"); };
byId("download").onclick = function () {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([currentYaml()], { type: "text/yaml" }));
  a.download = CONFIG.camera + "_" + MODE + ".yaml";
  a.click();
  URL.revokeObjectURL(a.href);
};
document.addEventListener("keydown", function (event) {
  if (event.target && event.target.tagName === "INPUT") return;
  if (event.key === "z" || event.key === "Z") undo();
  if (event.key === "Escape" && MODE === "counting-lines" && state.stack.length % 2 === 1) undo();
});
redraw();
"""

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>選點：__TITLE__</title>
<style>
  body { font-family: sans-serif; margin: 16px; background: #fafafa; color: #222; }
  #panels { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }
  .panel { flex: 1 1 420px; min-width: 0; }
  canvas { max-width: 100%; height: auto; cursor: crosshair; border: 1px solid #999; }
  button { margin-right: 8px; padding: 6px 12px; font-size: 14px; }
  pre { background: #fff; border: 1px solid #ccc; padding: 8px; overflow-x: auto; }
  .note { background: #fff3cd; border: 1px solid #e0c36a; padding: 6px 10px; margin: 6px 0; }
  .hint { color: #555; }
  #warning { color: #b45f06; }
  .line-row label { margin-right: 12px; }
</style>
</head>
<body>
<h1 id="title">__TITLE__</h1>
<div id="notes"></div>
<p class="hint" id="hint"></p>
<p>
  <button id="undo">復原上一點 (z)</button>
  <button id="clear">清除重畫</button>
  <button id="copy">複製 YAML</button>
  <button id="download">下載 YAML</button>
  <span id="status"></span> <span id="warning"></span>
</p>
<div id="lineForm"></div>
<div id="panels"></div>
<h2>輸出（<span id="target"></span>）</h2>
<pre id="yaml"></pre>
<script>
const CONFIG = __CONFIG_JSON__;
const TITLES = __TITLES_JSON__;
__PURE_LOGIC__
__UI__
</script>
</body>
</html>
"""


def _script_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render_picker_page(
    mode: str, camera: str, images: Mapping[str, PickerImage], notes: Sequence[str]
) -> str:
    if mode not in MODES:
        raise ValueError(f"未知的 mode {mode!r}，必須是 {MODES} 之一")
    missing = [key for key in REQUIRED_IMAGES[mode] if key not in images]
    if missing:
        raise ValueError(f"mode {mode} 需要影像 {missing}（目前只有 {sorted(images)}）")

    config = {
        "mode": mode,
        "camera": camera,
        "images": {
            key: {"name": image.name, "width": image.width, "height": image.height, "b64": image.jpeg_b64}
            for key, image in images.items() if key in REQUIRED_IMAGES[mode]
        },
        "notes": list(notes),
    }
    replacements = {
        "__TITLE__": html.escape(f"{camera}：{MODE_TITLES[mode]}"),
        "__CONFIG_JSON__": _script_json(config),
        "__TITLES_JSON__": _script_json(MODE_TITLES),
        "__PURE_LOGIC__": PURE_LOGIC_JS,
        "__UI__": UI_JS,
    }
    page = PAGE_TEMPLATE
    for key, value in replacements.items():
        page = page.replace(key, value)
    return page
