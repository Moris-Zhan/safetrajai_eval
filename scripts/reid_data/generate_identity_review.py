#!/usr/bin/env python3
"""
Re-ID 身分人工審查頁（spec: reid-dataset，design D7）。

讀取裁切索引與 detect_contamination.py 的身分層級標記，挑出「任一檢查標記的身分＋bus 的全部身分
＋每支鏡頭的隨機抽查身分」，只包含裁切數 ≥ min_crops_per_identity（--rules）的身分，
依標記原因分頁產生本機互動 HTML：靜止 → 運動跳動 → 外觀離群 → 類別不一致 → bus → 隨機抽查；
同一身分有多個原因時放在優先序最高的分頁，並列出全部原因。靜止分頁提供整頁全選。

隨機抽查：每支鏡頭以固定種子（--seed）從未被標記的身分中抽 --audit-per-camera 個，用來估計漏網率。

每個身分只產生代表縮圖與最多 12 張依時間均勻抽樣的縮圖（不為每一張裁切產生縮圖）。
頁面上的「產生確認清單」按鈕匯出 CSV（欄位：class,identity,image,action），交給
apply_review_exclusions.py 合併進排除清單。只產生報告與縮圖，不改動任何裁切。

用法：
    python scripts/reid_data/generate_identity_review.py
    python scripts/reid_data/generate_identity_review.py --out-dir dataset/reports/reid/review --audit-per-camera 20
"""
from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from safetrajai.paths import PROJECT_ROOT
from safetrajai.reid.dataset_index import read_all_indexes
from safetrajai.reid.manifest import load_split_rules

THUMB_MAX_SIDE = 160
MAX_THUMBS_PER_IDENTITY = 12
ALWAYS_REVIEW = ("bus",)
REASON_ORDER = ("static", "motion_jump", "appearance_outlier", "class_inconsistent", "bus", "audit")
REASON_TITLES = {
    "static": "靜止身分（疑似號誌等靜止物）",
    "motion_jump": "運動跳動（疑似 ID 閃跳）",
    "appearance_outlier": "外觀離群（疑似混入其他物件）",
    "class_inconsistent": "類別不一致（人／二輪／四輪）",
    "bus": "bus 全數審查",
    "audit": "隨機抽查（未被標記）",
}


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


def thumb_name(cls: str, identity: str, image_name: str) -> str:
    safe_identity = "".join(c if c.isalnum() or c in "-_." else "_" for c in identity)
    return f"{cls}__{safe_identity}__{Path(image_name).stem}.jpg"


def read_flags(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {(r["class"], r["identity"]): r for r in csv.DictReader(f)}


def sample_rows(rows: list[dict], limit: int = MAX_THUMBS_PER_IDENTITY) -> list[dict]:
    rows = sorted(rows, key=lambda r: float(r["time_s"]))
    if len(rows) <= limit:
        return rows
    return [rows[i] for i in np.unique(np.round(np.linspace(0, len(rows) - 1, limit)).astype(int))]


def is_true(record: dict | None, field: str) -> bool:
    return record is not None and str(record.get(field)) == "True"


def identity_reasons(key: tuple[str, str], consistency: dict, outliers: dict, motion: dict) -> list[str]:
    checks = {
        "static": is_true(motion.get(key), "static"),
        "motion_jump": is_true(motion.get(key), "motion_jump"),
        "appearance_outlier": is_true(outliers.get(key), "flagged"),
        "class_inconsistent": is_true(consistency.get(key), "flagged"),
        "bus": key[0] in ALWAYS_REVIEW,
    }
    return [reason for reason in REASON_ORDER if checks.get(reason)]


def select_for_review(
    index_rows: list[dict],
    consistency: dict,
    outliers: dict,
    motion: dict,
    min_crops: int,
    audit_per_camera: int,
    seed: int,
) -> dict[tuple[str, str], tuple[list[dict], list[str]]]:
    by_identity: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in index_rows:
        by_identity[(row["class"], row["identity"])].append(row)
    eligible = {k: v for k, v in sorted(by_identity.items()) if len(v) >= min_crops}

    selected = {}
    unflagged_by_camera: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, rows in eligible.items():
        reasons = identity_reasons(key, consistency, outliers, motion)
        if reasons:
            selected[key] = (rows, reasons)
        else:
            unflagged_by_camera[rows[0]["camera"]].append(key)

    rng = np.random.default_rng(seed)
    for camera in sorted(unflagged_by_camera):
        candidates = unflagged_by_camera[camera]
        picks = rng.choice(len(candidates), size=min(audit_per_camera, len(candidates)), replace=False)
        for i in sorted(picks):
            selected[candidates[i]] = (eligible[candidates[i]], ["audit"])
    return selected


def review_label(key: tuple[str, str], reasons: list[str], consistency: dict, outliers: dict, motion: dict) -> str:
    parts = ["+".join(reasons)]
    if key in motion:
        parts.append(f"duration={motion[key]['duration_s']}s displacement={motion[key]['displacement']} "
                     f"velocity_change={motion[key]['max_velocity_change']}")
    if key in consistency:
        parts.append(f"coarse_agreement={consistency[key].get('coarse_agreement_rate', '')}")
    if key in outliers:
        parts.append(f"outliers={outliers[key]['n_outliers']}/{outliers[key]['n_crops']}")
    return "; ".join(parts)


IDENTITY_CARD_TEMPLATE = """
<div class="identity-card" data-class="{cls}" data-identity="{identity_attr}">
  <label class="identity-header">
    <input type="checkbox" class="identity-checkbox">
    <img src="{assets_rel}/{thumb}" loading="lazy">
    <span class="identity-name">{identity_label}</span>
    <span class="agreement">{agreement_label}</span>
  </label>
  <details>
    <summary>展開抽樣縮圖（共 {n_images} 張）</summary>
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
{extra_controls}
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
  a.download = "review_exclusions.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}
document.getElementById("export-btn").addEventListener("click", exportCsv);
const selectAll = document.getElementById("select-all");
if (selectAll) {{
  selectAll.addEventListener("click", () => {{
    document.querySelectorAll(".identity-checkbox").forEach(cb => {{ cb.checked = true; }});
  }});
}}
</script>
</body>
</html>
"""

def render_identity_card(cls: str, identity: str, rows: list[dict], crop_root: Path, assets_dir: Path, label: str) -> str:
    picks = sample_rows(rows)
    assets_rel = assets_dir.name
    thumbs = []
    for row in picks:
        image_name = Path(row["crop_path"]).name
        t_name = thumb_name(cls, identity, image_name)
        if not make_thumbnail(crop_root / row["crop_path"], assets_dir / t_name):
            print(f"warn: failed to read {row['crop_path']}")
        thumbs.append(IMAGE_THUMB_TEMPLATE.format(image_name=html.escape(image_name), assets_rel=assets_rel, thumb=t_name))
    rep = thumb_name(cls, identity, Path(picks[len(picks) // 2]["crop_path"]).name)
    return IDENTITY_CARD_TEMPLATE.format(
        cls=html.escape(cls), identity_attr=html.escape(identity), identity_label=html.escape(identity),
        assets_rel=assets_rel, thumb=rep, agreement_label=html.escape(label),
        n_images=len(rows), image_thumbs="".join(thumbs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "crops"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "dataset" / "reports" / "reid"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "dataset" / "reports" / "reid" / "review"))
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--audit-per-camera", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rules", default=str(PROJECT_ROOT / "config" / "reid_split.yaml"))
    args = parser.parse_args()

    crop_root, reports = Path(args.crop_root), Path(args.reports_dir)
    consistency = read_flags(reports / "class_consistency.csv")
    outliers = read_flags(reports / "appearance_outliers.csv")
    motion = read_flags(reports / "motion_flags.csv")
    min_crops = load_split_rules(Path(args.rules))["min_crops_per_identity"]
    selected = select_for_review(read_all_indexes(crop_root), consistency, outliers, motion,
                                 min_crops, args.audit_per_camera, args.seed)

    out_dir = Path(args.out_dir)
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for reason in REASON_ORDER:
        keys = sorted(k for k, (_, reasons) in selected.items() if reasons[0] == reason)
        pages = [keys[i:i + args.page_size] for i in range(0, len(keys), args.page_size)]
        for page_idx, page_keys in enumerate(pages, start=1):
            cards = [render_identity_card(k[0], k[1], selected[k][0], crop_root, assets_dir,
                                          review_label(k, selected[k][1], consistency, outliers, motion))
                     for k in page_keys]
            title = f"{REASON_TITLES[reason]} 身分審查（第 {page_idx}/{len(pages)} 頁）"
            controls = '<button id="select-all">本頁全部勾選</button>' if reason == "static" else ""
            page = PAGE_TEMPLATE.format(title=html.escape(title), extra_controls=controls,
                                        sections=f"<section>{''.join(cards)}</section>")
            (out_dir / f"identity_review_{reason}_page{page_idx}.html").write_text(page, encoding="utf-8")
            print(f"  {reason} page {page_idx}/{len(pages)}: {len(page_keys)} identities")
    print(f"done: {len(selected)} identities (>= {min_crops} crops) -> {out_dir}")
    print("只產生報告與縮圖，未改動任何裁切。")


if __name__ == "__main__":
    main()
