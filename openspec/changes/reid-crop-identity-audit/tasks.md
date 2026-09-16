## 1. 身分跨類別一致性檢查（`dataset/reid/check_class_consistency.py`）

> 對應 `specs/reid/identity-class-consistency/spec.md`；設計依據見 `design.md` Decision 1/2。

- [ ] 1.1 確認既有介面簽章沒有過期，再動手寫程式：檢視 `module/detection/detector.py::Detector.__init__(model_path)` / `Detector.detect(frame, iou, imgsz)`，`module/tracker/config.py::load_track_config(path)` / `CLASS_NAMES`，`evaluation/config_loader.py::load_config(config_path)` 回傳的 `AppConfig` 是否仍有 `det_model_path`／`iou`／`imgsz` 屬性（`track_eval.py` 第 526-530 行的呼叫方式為準）。若簽章已變動，本檔其餘任務要跟著調整，不要照抄下面程式碼硬套。
- [ ] 1.2 建立 `dataset/reid/check_class_consistency.py`，內容如下（`discover_identities`/`list_images` 複製自 `filter_crop_dataset.py`，比照 `generate_quality_report.py` 已有的重複慣例，不額外抽共用模組）：

```python
#!/usr/bin/env python3
"""
dataset/reid/crop_dataset/ 身分跨類別一致性稽核（Tier 3）。

對每個 CCTV 來源身分逐張裁切重新呼叫本專案既有 YOLO 偵測器，比對偵測結果的最高信心
類別是否等於資料夾所屬類別，用來找出「整個身分可能歸錯類別」或「身分內因追蹤器 ID
switch 混入其他物體」的候選項目。只標記、不刪除／搬移任何原始影像檔案，對應
openspec/changes/reid-crop-identity-audit/specs/reid/identity-class-consistency/spec.md。

DSEC 來源身分（資料夾名稱前綴 dsecdet_）一律略過評分——reid-training-data-pipeline
的 pilot 已證實本專案偵測器在 DSEC 視角對 bus 類別系統性誤判，用同一顆偵測器驗證
同視角資料沒有意義，見 design.md Decision 2。

用法：
    python dataset/reid/check_class_consistency.py
    python dataset/reid/check_class_consistency.py --classes person small_truck
    python dataset/reid/check_class_consistency.py --config config/botsort_noreid.yaml \
        --out dataset/reid/class_consistency_flags.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from evaluation.config_loader import load_config
from module.detection.detector import Detector
from module.tracker.config import CLASS_NAMES, load_track_config

CLASSES = ["bus", "car", "person", "scooter", "small_truck", "truck"]
IMG_SUFFIXES = (".jpg", ".jpeg", ".png")
DSEC_PREFIX = "dsecdet_"


def discover_identities(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.is_dir())


def list_images(identity_dir: Path) -> list[Path]:
    return sorted(p for p in identity_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)


def is_dsec_identity(identity_dir: Path) -> bool:
    return identity_dir.name.startswith(DSEC_PREFIX)


def predicted_class_for_crop(detector: Detector, crop, iou: float, imgsz: int):
    """對單張裁切推論，回傳信心最高偵測框的類別名稱與信心值；沒有框通過門檻則回傳 (None, 0.0)。"""
    boxes = detector.detect(crop, iou=iou, imgsz=imgsz)
    if len(boxes) == 0:
        return None, 0.0
    best_idx = int(boxes.conf.argmax())
    cls_id = int(boxes.cls[best_idx])
    conf = float(boxes.conf[best_idx])
    return CLASS_NAMES.get(cls_id, f"class_{cls_id}"), conf


def check_identity(identity_dir: Path, assigned_class: str, detector: Detector, iou: float, imgsz: int) -> list[dict]:
    rows = []
    for img_path in list_images(identity_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        predicted_class, conf = predicted_class_for_crop(detector, img, iou, imgsz)
        agree = predicted_class == assigned_class
        rows.append({
            "class": assigned_class,
            "identity": identity_dir.name,
            "image": str(img_path),
            "predicted_class": predicted_class or "",
            "confidence": round(conf, 4),
            "agree": agree,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop-root", default="dataset/reid/crop_dataset")
    parser.add_argument("--classes", nargs="+", default=CLASSES, choices=CLASSES)
    parser.add_argument("--config", default="config/botsort_noreid.yaml")
    parser.add_argument("--out", default="dataset/reid/class_consistency_flags.csv")
    args = parser.parse_args()

    # 必須先載入 track config 才能取得正確的 CLASS_NAMES／class_conf_thresh，
    # 順序比照 track_eval.py 的 CLI 入口（load_track_config 先於 Detector 建立）。
    load_track_config(args.config)
    cfg = load_config(args.config)
    detector = Detector(cfg.det_model_path)

    crop_root = Path(args.crop_root)
    all_rows: list[dict] = []
    identity_summaries: list[dict] = []

    for cls in args.classes:
        class_dir = crop_root / cls
        if not class_dir.is_dir():
            print(f"skip {cls}: {class_dir} not found")
            continue
        identity_dirs = discover_identities(class_dir)
        n_dsec = 0
        for identity_dir in identity_dirs:
            if is_dsec_identity(identity_dir):
                n_dsec += 1
                identity_summaries.append({
                    "class": cls, "identity": identity_dir.name,
                    "n_images": len(list_images(identity_dir)),
                    "agreement_rate": "", "skipped_reason": "dsec_source",
                })
                continue
            rows = check_identity(identity_dir, cls, detector, cfg.iou, cfg.imgsz)
            all_rows.extend(rows)
            n = len(rows)
            n_agree = sum(1 for r in rows if r["agree"])
            identity_summaries.append({
                "class": cls, "identity": identity_dir.name, "n_images": n,
                "agreement_rate": round(n_agree / n, 4) if n else "",
                "skipped_reason": "",
            })
        print(f"  {cls}: {len(identity_dirs)} identities scanned ({n_dsec} DSEC skipped)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "identity", "image", "predicted_class", "confidence", "agree"])
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = out_path.with_name(out_path.stem + "_identities.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "identity", "n_images", "agreement_rate", "skipped_reason"])
        writer.writeheader()
        scored = [r for r in identity_summaries if r["agreement_rate"] != ""]
        skipped = [r for r in identity_summaries if r["agreement_rate"] == ""]
        scored.sort(key=lambda r: r["agreement_rate"])
        writer.writerows(scored + skipped)

    n_mismatch = sum(1 for r in all_rows if not r["agree"])
    print(
        f"\ndone: scanned {len(all_rows)} images across {len(identity_summaries)} identities, "
        f"{n_mismatch} image-level mismatches -> {out_path}\n"
        f"identity summary (sorted by agreement_rate) -> {summary_path}"
    )
    print("只標記，未刪除／搬移任何原始檔案。")


if __name__ == "__main__":
    main()
```

驗證：`python -c "import ast; ast.parse(open('dataset/reid/check_class_consistency.py').read())"` 確認語法正確；`python dataset/reid/check_class_consistency.py --help` 確認 CLI 可執行且參數列表符合預期。

- [ ] 1.3 小規模驗證（先跑資料量最小的類別，對應 `design.md` Migration Plan 第 2 步）：執行 `python dataset/reid/check_class_consistency.py --classes small_truck`，確認：(a) 終端機輸出的 `n_dsec` 對 `small_truck` 應為 0（`small_truck` 目前沒有 DSEC 來源身分，見 README 資料來源說明）；(b) `dataset/reid/class_consistency_flags.csv` 與 `_identities.csv` 兩份檔案都有產生且非空；(c) 打開 `_identities.csv`，依 `agreement_rate` 排序最前面（最低）的 3-5 個身分，用檔案總管或圖片檢視器打開該身分資料夾內的圖片，人工確認 agreement_rate 低是否真的對應「歸錯類別」或「局部污染」，記錄下這幾筆的目視判斷結果（供 1.4 決定門檻參考，若判斷理由站不住腳要回頭檢查 1.2 的邏輯或考慮 crop 在偵測器上的行為是否如預期）。
- [ ] 1.4 依 1.3 的人工核對結果，在 `design.md` 的 Open Questions 段落填入實際採用的 agreement_rate 參考門檻（例如「低於 0.5 建議整身分複核」），若核對結果顯示訊號不可靠（人工看起來沒問題的身分卻被排到最前面），記錄下原因並評估是否需要調整 1.2 的推論邏輯（如信心門檻、`imgsz`），而非直接假設分數可用。

## 2. 全量互動審查報告（`dataset/reid/generate_identity_review.py`）

> 對應 `specs/reid/crop-review-report/spec.md`；設計依據見 `design.md` Decision 3/5/6。

- [ ] 2.1 建立 `dataset/reid/generate_identity_review.py`（`list_images`/`make_thumbnail`/`thumb_name` 複製自 `generate_quality_report.py` 既有慣例）：

```python
#!/usr/bin/env python3
"""
dataset/reid/crop_dataset/ 全量身分互動審查報告（Tier 3 報告）。

讀取 check_class_consistency.py 的身分層級輸出（*_identities.csv），為每個類別產生
本機互動 HTML 報告：CCTV 來源身分依 agreement_rate 由低到高排序（越可疑排越前面），
DSEC 來源身分（無自動分數）獨立呈現在每個類別報告的最後一頁。每個身分卡片可勾選
「整身分移除」，展開後可對單張影像個別勾選「僅此張移除」；頁面上的「產生確認清單」
按鈕用純前端 JS 把目前頁面的勾選結果匯出成 dataset/reid/quarantine_flagged.py 可讀的
CSV（欄位：class,identity,image,action）。只產生報告與縮圖，不搬移／刪除任何原始
影像，對應 openspec/changes/reid-crop-identity-audit/specs/reid/crop-review-report/spec.md。

用法：
    python dataset/reid/generate_identity_review.py
    python dataset/reid/generate_identity_review.py \
        --identities-csv dataset/reid/class_consistency_flags_identities.csv \
        --out-dir dataset/reid/identity_review --page-size 100
"""
from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

import cv2

CLASSES = ["bus", "car", "person", "scooter", "small_truck", "truck"]
IMG_SUFFIXES = (".jpg", ".jpeg", ".png")
THUMB_MAX_SIDE = 160


def list_images(identity_dir: Path) -> list[Path]:
    return sorted(p for p in identity_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)


def make_thumbnail(src: Path, dst: Path) -> bool:
    img = cv2.imread(str(src))
    if img is None:
        return False
    h, w = img.shape[:2]
    scale = THUMB_MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)
    return True


def thumb_name(cls: str, identity: str, image_path: Path) -> str:
    safe_identity = "".join(c if c.isalnum() or c in "-_." else "_" for c in identity)
    return f"{cls}__{safe_identity}__{image_path.stem}.jpg"


def load_identities(csv_path: Path) -> dict[str, list[dict]]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_class[row["class"]].append(row)
    return by_class


IDENTITY_CARD_TEMPLATE = """
<div class="identity-card" data-class="{cls}" data-identity="{identity_attr}">
  <label class="identity-header">
    <input type="checkbox" class="identity-checkbox">
    <img src="{assets_rel}/{thumb}" loading="lazy">
    <span class="identity-name">{identity_label}</span>
    <span class="agreement">{agreement_label}</span>
  </label>
  <details>
    <summary>展開全部 {n_images} 張</summary>
    <div class="image-grid">
      {image_thumbs}
    </div>
  </details>
</div>
"""

IMAGE_THUMB_TEMPLATE = """
      <label class="image-item">
        <input type="checkbox" class="image-checkbox" data-image="{image_name}">
        <img src="{assets_rel}/{thumb}" loading="lazy">
        <span>{image_name}</span>
      </label>"""

PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: sans-serif; margin: 16px; }}
  .identity-card {{ border: 1px solid #ccc; border-radius: 6px; padding: 8px; margin-bottom: 8px; }}
  .identity-header {{ display: flex; align-items: center; gap: 8px; cursor: pointer; }}
  .identity-header img {{ max-height: 80px; }}
  .agreement {{ margin-left: auto; font-family: monospace; color: #a33; }}
  .image-grid {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
  .image-item {{ display: flex; flex-direction: column; align-items: center; font-size: 11px; }}
  .image-item img {{ max-height: 60px; }}
  #export-btn {{ position: sticky; top: 0; background: #fff; padding: 8px 0; font-size: 16px; }}
  section.dsec {{ border-top: 3px solid #999; margin-top: 24px; padding-top: 8px; }}
</style>
</head>
<body>
<button id="export-btn">產生確認清單（下載 CSV）</button>
<h1>{title}</h1>
{sections}
<script>
function exportCsv() {{
  const rows = [["class", "identity", "image", "action"]];
  document.querySelectorAll(".identity-card").forEach(card => {{
    const cls = card.dataset.class;
    const identity = card.dataset.identity;
    const identityChecked = card.querySelector(".identity-checkbox").checked;
    if (identityChecked) {{
      rows.push([cls, identity, "", "remove"]);
      return;
    }}
    card.querySelectorAll(".image-checkbox").forEach(cb => {{
      if (cb.checked) {{
        rows.push([cls, identity, cb.dataset.image, "remove"]);
      }}
    }});
  }});
  const csvText = rows.map(r => r.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(",")).join("\\n");
  const blob = new Blob([csvText], {{ type: "text/csv;charset=utf-8;" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "quarantine_confirm.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}
document.getElementById("export-btn").addEventListener("click", exportCsv);
</script>
</body>
</html>
"""


def render_identity_card(cls: str, identity: str, crop_root: Path, assets_dir: Path, agreement_label: str) -> str:
    identity_dir = crop_root / cls / identity
    imgs = list_images(identity_dir) if identity_dir.is_dir() else []
    rep_img = imgs[len(imgs) // 2] if imgs else None
    rep_thumb = thumb_name(cls, identity, rep_img) if rep_img else ""
    if rep_img:
        make_thumbnail(rep_img, assets_dir / rep_thumb)
    assets_rel = assets_dir.name

    image_thumbs = []
    for img_path in imgs:
        t_name = thumb_name(cls, identity, img_path)
        make_thumbnail(img_path, assets_dir / t_name)
        image_thumbs.append(IMAGE_THUMB_TEMPLATE.format(
            image_name=html.escape(img_path.name), assets_rel=assets_rel, thumb=t_name,
        ))

    return IDENTITY_CARD_TEMPLATE.format(
        cls=html.escape(cls), identity_attr=html.escape(identity), identity_label=html.escape(identity),
        assets_rel=assets_rel, thumb=rep_thumb, agreement_label=agreement_label,
        n_images=len(imgs), image_thumbs="".join(image_thumbs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop-root", default="dataset/reid/crop_dataset")
    parser.add_argument("--identities-csv", default="dataset/reid/class_consistency_flags_identities.csv")
    parser.add_argument("--out-dir", default="dataset/reid/identity_review")
    parser.add_argument("--classes", nargs="+", default=CLASSES, choices=CLASSES)
    parser.add_argument("--page-size", type=int, default=100)
    args = parser.parse_args()

    crop_root = Path(args.crop_root)
    out_dir = Path(args.out_dir)
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_class = load_identities(Path(args.identities_csv))

    for cls in args.classes:
        rows = by_class.get(cls, [])
        scored = [r for r in rows if r["skipped_reason"] != "dsec_source"]
        dsec = [r for r in rows if r["skipped_reason"] == "dsec_source"]

        pages = [scored[i:i + args.page_size] for i in range(0, len(scored), args.page_size)] or [[]]

        for page_idx, page_rows in enumerate(pages, start=1):
            is_last_page = page_idx == len(pages)
            cards = [
                render_identity_card(cls, r["identity"], crop_root, assets_dir, f"agreement_rate={r['agreement_rate']}")
                for r in page_rows
            ]
            sections = [f"<section>{''.join(cards)}</section>"]

            if is_last_page and dsec:
                dsec_cards = [
                    render_identity_card(cls, r["identity"], crop_root, assets_dir, "DSEC 來源，無自動分數")
                    for r in dsec
                ]
                sections.append(f'<section class="dsec"><h2>DSEC 來源身分（純人工審查）</h2>{"".join(dsec_cards)}</section>')

            title = f"{cls} 身分審查（第 {page_idx}/{len(pages)} 頁）"
            html_content = PAGE_TEMPLATE.format(title=html.escape(title), sections="".join(sections))
            out_path = out_dir / f"identity_review_{cls}_page{page_idx}.html"
            out_path.write_text(html_content, encoding="utf-8")
            print(f"  {cls} page {page_idx}/{len(pages)}: {len(page_rows)} identities -> {out_path}")

    print(f"\ndone: reports -> {out_dir}")
    print("只產生報告與縮圖，未搬移／刪除任何原始裁切影像。")


if __name__ == "__main__":
    main()
```

驗證：`python -c "import ast; ast.parse(open('dataset/reid/generate_identity_review.py').read())"` 確認語法正確；`python dataset/reid/generate_identity_review.py --help` 確認 CLI 可執行。

- [ ] 2.2 對任務 1.3 已產生的 `small_truck` 一致性結果執行 `python dataset/reid/generate_identity_review.py --classes small_truck`，用瀏覽器打開 `dataset/reid/identity_review/identity_review_small_truck_page1.html`，確認：(a) 身分卡片依 agreement_rate 由低到高排列；(b) 展開一張卡片可看到該身分全部裁切縮圖；(c) 勾選 1-2 個身分（含至少一次「整身分」與一次「單張影像」）後點「產生確認清單」，瀏覽器下載出 `quarantine_confirm.csv`，用文字編輯器打開確認欄位為 `class,identity,image,action`、內容與勾選的項目一致。

## 3. 隔離搬移（`dataset/reid/quarantine_flagged.py`）

> 對應 `specs/reid/crop-quarantine/spec.md`；設計依據見 `design.md` Decision 4。

- [ ] 3.1 建立 `dataset/reid/quarantine_flagged.py`：

```python
#!/usr/bin/env python3
"""
dataset/reid/crop_dataset/ 問題身分／影像隔離腳本（Tier 3 移除機制）。

讀取人工確認過的清單 CSV（欄位：class,identity,image,action；image 留空代表整個身分，
非空代表僅該張影像；目前唯一支援的 action 是 "remove"），把對應項目從 crop_dataset/
搬到隔離目錄（保留 {class}/{identity}/ 結構），而非直接刪除，讓誤判可回復。預設
dry-run（只列印將搬移的項目，不動任何檔案），需明確帶 --apply 才實際搬移。每次
--apply 執行後在隔離目錄寫入一份 manifest 記錄本次操作。單張影像搬移後若原身分
資料夾清空，一併移除該空資料夾。對應
openspec/changes/reid-crop-identity-audit/specs/reid/crop-quarantine/spec.md。

confirm CSV 可手動撰寫，或用 generate_identity_review.py 產生的報告頁面勾選後匯出
（欄位 schema 兩者一致）。

用法：
    python dataset/reid/quarantine_flagged.py --confirm-csv path/to/quarantine_confirm.csv
    python dataset/reid/quarantine_flagged.py --confirm-csv path/to/quarantine_confirm.csv --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def load_confirm_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("action") != "remove":
            raise ValueError(f"不支援的 action: {r.get('action')!r}（目前只支援 'remove'）: {r}")
        if not r.get("class") or not r.get("identity"):
            raise ValueError(f"確認清單缺少 class 或 identity: {r}")
    return rows


def plan_moves(rows: list[dict], crop_root: Path, quarantine_root: Path) -> list[dict]:
    """把確認清單展開成實際的 (src, dst) 搬移計畫，不做任何 I/O。"""
    moves = []
    for r in rows:
        cls, identity, image = r["class"], r["identity"], r.get("image") or ""
        src_identity_dir = crop_root / cls / identity
        dst_identity_dir = quarantine_root / cls / identity
        if image:
            src = src_identity_dir / image
            moves.append({"class": cls, "identity": identity, "image": image, "src": src, "dst": dst_identity_dir / image})
        else:
            if not src_identity_dir.is_dir():
                print(f"警告：找不到身分資料夾，略過：{src_identity_dir}")
                continue
            for img_path in sorted(src_identity_dir.iterdir()):
                if img_path.is_file():
                    moves.append({
                        "class": cls, "identity": identity, "image": img_path.name,
                        "src": img_path, "dst": dst_identity_dir / img_path.name,
                    })
    return moves


def apply_moves(moves: list[dict], quarantine_root: Path, confirm_csv: Path) -> Path:
    quarantined_identity_dirs: set[Path] = set()
    manifest_entries = []
    for m in moves:
        src, dst = m["src"], m["dst"]
        if not src.is_file():
            print(f"警告：來源檔案不存在，略過：{src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        quarantined_identity_dirs.add(src.parent)
        manifest_entries.append({"class": m["class"], "identity": m["identity"], "image": m["image"], "src": str(src), "dst": str(dst)})

    # 清理搬移後變空的原身分資料夾，避免下游 discover_identities() 誤判
    for identity_dir in quarantined_identity_dirs:
        if identity_dir.is_dir() and not any(identity_dir.iterdir()):
            identity_dir.rmdir()

    manifest = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "confirm_csv": str(confirm_csv),
        "n_moved": len(manifest_entries),
        "entries": manifest_entries,
    }
    quarantine_root.mkdir(parents=True, exist_ok=True)
    manifest_path = quarantine_root / f"manifest_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm-csv", required=True)
    parser.add_argument("--crop-root", default="dataset/reid/crop_dataset")
    parser.add_argument("--quarantine-root", default="dataset/reid/crop_dataset_quarantine")
    parser.add_argument("--apply", action="store_true", help="實際搬移檔案；不帶此旗標只列印計畫（dry-run）")
    args = parser.parse_args()

    confirm_csv = Path(args.confirm_csv)
    crop_root = Path(args.crop_root)
    quarantine_root = Path(args.quarantine_root)

    rows = load_confirm_rows(confirm_csv)
    moves = plan_moves(rows, crop_root, quarantine_root)

    if not args.apply:
        print(f"[dry-run] {len(moves)} 個檔案將被搬移到 {quarantine_root}（未實際執行，加 --apply 才會搬移）：")
        for m in moves[:20]:
            print(f"  {m['src']} -> {m['dst']}")
        if len(moves) > 20:
            print(f"  ...（共 {len(moves)} 筆，僅列出前 20 筆）")
        return

    manifest_path = apply_moves(moves, quarantine_root, confirm_csv)
    print(f"done: 搬移 {len(moves)} 個檔案到 {quarantine_root}，manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
```

驗證：`python -c "import ast; ast.parse(open('dataset/reid/quarantine_flagged.py').read())"` 確認語法正確；`python dataset/reid/quarantine_flagged.py --help` 確認 CLI 可執行。

- [ ] 3.2 用假造 fixture 驗證四種情境（不對真實 `crop_dataset/` 執行，避免任何誤動真實資料的風險）。在 `$CLAUDE_JOB_DIR/tmp`（或等效暫存目錄）寫一支驗證腳本，內容比照下列邏輯（用 `tempfile.TemporaryDirectory` 建立假 `crop_root`/`quarantine_root`，`Path.write_bytes(b"fake")` 產生佔位檔案即可，`quarantine_flagged.py` 不會真的解碼影像內容）：

```python
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "dataset/reid")
from quarantine_flagged import apply_moves, load_confirm_rows, plan_moves  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    crop_root = tmp / "crop_dataset"
    quarantine_root = tmp / "crop_dataset_quarantine"

    # 情境準備：bus/id_a 有 2 張（整身分移除用）、car/id_b 有 2 張（單張移除用）
    (crop_root / "bus" / "id_a").mkdir(parents=True)
    (crop_root / "bus" / "id_a" / "000001.jpg").write_bytes(b"fake")
    (crop_root / "bus" / "id_a" / "000002.jpg").write_bytes(b"fake")
    (crop_root / "car" / "id_b").mkdir(parents=True)
    (crop_root / "car" / "id_b" / "000001.jpg").write_bytes(b"fake")
    (crop_root / "car" / "id_b" / "000002.jpg").write_bytes(b"fake")
    # 情境準備：small_truck/id_c 只有 1 張，該張被移除後應清空整個資料夾
    (crop_root / "small_truck" / "id_c").mkdir(parents=True)
    (crop_root / "small_truck" / "id_c" / "000001.jpg").write_bytes(b"fake")

    confirm_csv = tmp / "confirm.csv"
    with open(confirm_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "identity", "image", "action"])
        writer.writeheader()
        writer.writerow({"class": "bus", "identity": "id_a", "image": "", "action": "remove"})
        writer.writerow({"class": "car", "identity": "id_b", "image": "000001.jpg", "action": "remove"})
        writer.writerow({"class": "small_truck", "identity": "id_c", "image": "000001.jpg", "action": "remove"})

    rows = load_confirm_rows(confirm_csv)
    moves = plan_moves(rows, crop_root, quarantine_root)
    assert len(moves) == 4, f"expected 4 moves (2 for id_a + 1 + 1), got {len(moves)}"

    # dry-run：檔案不應被搬動
    assert (crop_root / "bus" / "id_a" / "000001.jpg").is_file()

    manifest_path = apply_moves(moves, quarantine_root, confirm_csv)
    assert manifest_path.is_file()

    # 情境 1：整身分搬移 -> bus/id_a 底下兩張都到隔離目錄，原資料夾應被清空移除
    assert not (crop_root / "bus" / "id_a").exists(), "整身分移除後原資料夾應被清空移除"
    assert (quarantine_root / "bus" / "id_a" / "000001.jpg").is_file()
    assert (quarantine_root / "bus" / "id_a" / "000002.jpg").is_file()

    # 情境 2：單張移除 -> car/id_b 只搬走 000001.jpg，000002.jpg 留在原地，資料夾不清空
    assert (quarantine_root / "car" / "id_b" / "000001.jpg").is_file()
    assert (crop_root / "car" / "id_b" / "000002.jpg").is_file(), "未勾選的影像應留在原資料夾"
    assert (crop_root / "car" / "id_b").is_dir(), "還有剩餘影像的資料夾不應被移除"

    # 情境 3：單張移除致資料夾清空 -> small_truck/id_c 應整個消失
    assert not (crop_root / "small_truck" / "id_c").exists(), "唯一影像移除後空資料夾應被清理"
    assert (quarantine_root / "small_truck" / "id_c" / "000001.jpg").is_file()

    print("ALL PASS")
```

執行該腳本並確認終端機印出 `ALL PASS`；若任一 `assert` 失敗，回頭修正 `quarantine_flagged.py` 的 `plan_moves`/`apply_moves` 邏輯，而不是放寬驗證腳本的斷言。

## 4. 全量執行與人工審查

> 對 `dataset/reid/crop_dataset/`（真實資料）執行，依 CLAUDE.md「涉及刪除資料...一律先確認」——本組任務除了最後一步 `--apply`，其餘都是唯讀或只新增報告檔案；4.5 執行前務必先把 4.4 產出的確認清單拿給使用者過目。

- [ ] 4.1 重跑既有 Tier 2 工具取得涵蓋 DSEC 併入後全量資料的基準線：`python dataset/reid/filter_crop_dataset.py && python dataset/reid/generate_quality_report.py`，確認終端機輸出的「掃描 N 身分」數字與 `dataset/reid/README.md`「目前資料量」表一致（尤其 `bus` 應為 156，而非舊報告的 5）；`quality_report.md` 的總覽表也應反映新數字。
- [ ] 4.2 對全部 6 個類別執行 `python dataset/reid/check_class_consistency.py`（不加 `--classes` 限制），確認 `dataset/reid/class_consistency_flags_identities.csv` 涵蓋全部類別、`bus` 類別裡 `dsecdet_` 前綴身分的 `skipped_reason` 皆為 `dsec_source`。此步驟需逐張跑 YOLO 推論，車流量大的類別（car/scooter）耗時較長，可視情況先跑一個類別確認無誤再跑其餘。
- [ ] 4.3 對全部 6 個類別執行 `python dataset/reid/generate_identity_review.py`，確認 `dataset/reid/identity_review/` 底下每個類別都產生對應頁數的 HTML（car/scooter 應有多頁），且每個類別最後一頁若該類別有 DSEC 身分（目前只有 bus）應包含「DSEC 來源身分」區塊。
- [ ] 4.4 人工依 4.3 產生的報告逐類別審查（此步驟由使用者執行，非自動化）：瀏覽每個類別的 HTML 頁面，勾選判定要移除的整身分或單張影像，用頁面「產生確認清單」按鈕逐頁匯出 CSV；多頁需要的話，把各頁下載的 CSV 合併成一份（保留同一組欄位 `class,identity,image,action`，去除重複表頭）。
- [ ] 4.5 拿到使用者確認的合併清單後，先執行 `python dataset/reid/quarantine_flagged.py --confirm-csv <合併後的CSV>`（dry-run），把預計搬移的筆數與範例路徑貼給使用者過目；使用者明確同意後才加 `--apply` 執行實際搬移，執行完印出的 manifest 路徑要記錄下來。

## 5. 文件收尾

- [ ] 5.1 更新 `dataset/reid/README.md`：在既有「目前資料量」表後補一段說明三支新工具（`check_class_consistency.py`／`generate_identity_review.py`／`quarantine_flagged.py`）的用途與執行順序，並補上 `dataset/reid/crop_dataset_quarantine/` 目錄的說明（含 manifest 檔案格式）；若 4.5 已實際執行過 `--apply`，同步更新「目前資料量」表反映搬移後的身分數／張數。
- [ ] 5.2 依專案既有慣例，在 `markdown/EXPERIMENTS.md` 新增一則本次稽核記錄：涵蓋 4.1-4.5 的執行結果（重跑 Tier2 後的統計、一致性檢查掃描到的身分/影像數、人工審查後實際隔離的筆數、agreement_rate 門檻的人工核對結論），避免之後重工。
