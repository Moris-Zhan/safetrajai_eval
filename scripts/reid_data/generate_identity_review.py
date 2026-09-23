#!/usr/bin/env python3
"""
dataset/reid/crop_dataset/ 全量身分互動審查報告（Tier 3 報告）。

讀取 check_class_consistency.py 的身分層級輸出（*_identities.csv），為每個類別產生
本機互動 HTML 報告：身分依 agreement_rate 由低到高排序（越可疑排越前面）。每個身分卡片可勾選
「整身分移除」，展開後可對單張影像個別勾選「僅此張移除」；頁面上的「產生確認清單」
按鈕用純前端 JS 把目前頁面的勾選結果匯出成 scripts/reid_data/quarantine_flagged.py 可讀的
CSV（欄位：class,identity,image,action）。只產生報告與縮圖，不搬移／刪除任何原始
影像，對應 openspec/changes/reid-crop-identity-audit/specs/reid/crop-review-report/spec.md。

用法：
    python scripts/reid_data/generate_identity_review.py
    python scripts/reid_data/generate_identity_review.py \
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

from safetrajai.classes import CLASS_NAMES

CLASSES = sorted(CLASS_NAMES.values())
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
        if not make_thumbnail(rep_img, assets_dir / rep_thumb):
            print(f"warn: failed to read {rep_img}")
    assets_rel = assets_dir.name

    image_thumbs = []
    for img_path in imgs:
        t_name = thumb_name(cls, identity, img_path)
        if not make_thumbnail(img_path, assets_dir / t_name):
            print(f"warn: failed to read {img_path}")
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

        pages = [rows[i:i + args.page_size] for i in range(0, len(rows), args.page_size)] or [[]]

        for page_idx, page_rows in enumerate(pages, start=1):
            cards = [
                render_identity_card(cls, r["identity"], crop_root, assets_dir, f"agreement_rate={r['agreement_rate']}")
                for r in page_rows
            ]
            sections = [f"<section>{''.join(cards)}</section>"]

            title = f"{cls} 身分審查（第 {page_idx}/{len(pages)} 頁）"
            html_content = PAGE_TEMPLATE.format(title=html.escape(title), sections="".join(sections))
            out_path = out_dir / f"identity_review_{cls}_page{page_idx}.html"
            out_path.write_text(html_content, encoding="utf-8")
            print(f"  {cls} page {page_idx}/{len(pages)}: {len(page_rows)} identities -> {out_path}")

    print(f"\ndone: reports -> {out_dir}")
    print("只產生報告與縮圖，未搬移／刪除任何原始裁切影像。")


if __name__ == "__main__":
    main()
