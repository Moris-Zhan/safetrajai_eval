#!/usr/bin/env python3
"""
遮擋碎片合併清單（change merge-occlusion-fragments，spec: reid-dataset）。
讀取裁切索引、兩份排除清單、碎片端點特徵（extract_fragment_features.py）與 config/reid/fragment_merge.yaml，
在同一序列、同一類別內以時間、外插位置與外觀一對一配對碎片，串成合併後的身分。
只寫合併紀錄與報告，不搬移、改名或刪除任何裁切，也不修改索引與排除清單。

模式：
  （預設）dry-run       印出各鏡頭的候選數與合併數（門檻未校準時印出相似度分布），不寫檔。
  --apply               寫 <out>（預設 dataset/reid/fragment_merges.csv）：class,identity,merged_identity,sequence,score；
                        只列出被改名的原始身分，依 class、identity 排序。門檻未校準時報錯。
  --sample N --seed S   依 design D4 分層抽 N 組候選，寫到 <review-dir>：
                          merge_sample.csv（sample_id,stratum,class,camera,sequence,identity_a,identity_b,score）
                          fragment_review.html（A 最後 4 張、B 最前 4 張並排；只勾選「不是同一實體」，
                          匯出 fragment_confirm.csv：sample_id,class,identity_a,identity_b,verdict）
                        頁面不顯示相似度與分層，避免影響判斷。
  --sample N --min-score X --exclude-sample CSV --start-id K
                        門檻以上樣本不足時的補抽（design D4）：從相似度 ≥ X、不在既有抽樣清單中的候選抽 N 組，
                        sample_id 從 K 起編，stratum 為 extra。
  --calibrate CSV...    讀 --sample-csv（可多份，預設 <review-dir>/merge_sample.csv）與匯出的確認 CSV（可多份），
                        計算讓門檻以上準確率 ≥ target 的最低門檻。

用法：
    python scripts/reid_data/build_fragment_merges.py
    python scripts/reid_data/build_fragment_merges.py --sample 80 --seed 0
    python scripts/reid_data/build_fragment_merges.py --calibrate dataset/reports/reid/fragment_review/fragment_confirm.csv
    python scripts/reid_data/build_fragment_merges.py --sample 30 --seed 1 --min-score 0.90 --start-id 81 \
        --exclude-sample dataset/reports/reid/fragment_review/merge_sample.csv --review-dir dataset/reports/reid/fragment_review_extra
    python scripts/reid_data/build_fragment_merges.py --apply
"""
from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.reid.dataset_index import read_all_indexes
from safetrajai.reid.fragment_merge import (
    MERGE_FIELDS,
    SAMPLE_FIELDS,
    STRATA,
    allocate_proportional,
    calibrate_threshold,
    collect_tracks,
    find_candidates,
    head_paths,
    load_merge_config,
    load_retention,
    merge_records,
    score_candidates,
    select_pairs,
    stratified_sample,
    supplementary_sample,
    tail_paths,
)

THUMB_MAX_SIDE = 160
REVIEW_CROPS = 4


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_features(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"找不到碎片端點特徵，請先執行 extract_fragment_features.py：{path}")
    data = np.load(path)
    return dict(zip(data["crop_path"].tolist(), data["features"]))


def make_thumbnail(src: Path, dst: Path) -> None:
    img = cv2.imread(str(src))
    if img is None:
        print(f"warn: failed to read {src}")
        return
    h, w = img.shape[:2]
    scale = THUMB_MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)


PAIR_CARD_TEMPLATE = """
<div class="pair-card" data-sample-id="{sample_id}" data-class="{cls}" data-identity-a="{identity_a}" data-identity-b="{identity_b}">
  <label class="pair-header"><input type="checkbox" class="not-same"> #{sample_id} 不是同一實體
    <span class="meta">{cls} · {camera}</span></label>
  <div class="pair-row">
    <div class="side"><div class="side-title">A 最後 {n_a} 張</div>{thumbs_a}</div>
    <div class="arrow">→</div>
    <div class="side"><div class="side-title">B 最前 {n_b} 張</div>{thumbs_b}</div>
  </div>
</div>
"""

PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>碎片合併候選確認</title>
<style>
  body {{ font-family: sans-serif; margin: 16px; }}
  .pair-card {{ border: 1px solid #ccc; border-radius: 6px; padding: 8px; margin-bottom: 10px; }}
  .pair-header {{ font-size: 16px; cursor: pointer; }}
  .meta {{ margin-left: 12px; color: #666; font-family: monospace; }}
  .pair-row {{ display: flex; align-items: center; gap: 12px; margin-top: 6px; }}
  .side {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: flex-end; }}
  .side-title {{ width: 100%; font-size: 12px; color: #444; }}
  .side img {{ max-height: 110px; }}
  .arrow {{ font-size: 28px; color: #888; }}
  #export-btn {{ position: sticky; top: 0; background: #fff; padding: 8px 0; font-size: 16px; }}
</style>
</head>
<body>
<button id="export-btn">產生確認清單（下載 CSV）</button>
<h1>碎片合併候選確認（共 {n_pairs} 組）</h1>
<p>每組左邊是軌跡 A 結束前的裁切，右邊是軌跡 B 出現後的裁切。只勾選「不是同一實體」的配對，未勾選視為同一實體。</p>
{cards}
<script>
document.getElementById("export-btn").addEventListener("click", () => {{
  const rows = [["sample_id", "class", "identity_a", "identity_b", "verdict"]];
  document.querySelectorAll(".pair-card").forEach(card => {{
    if (card.querySelector(".not-same").checked) {{
      rows.push([card.dataset.sampleId, card.dataset.class, card.dataset.identityA, card.dataset.identityB, "not_same"]);
    }}
  }});
  const csvText = rows.map(r => r.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(",")).join("\\n");
  const url = URL.createObjectURL(new Blob([csvText], {{ type: "text/csv;charset=utf-8;" }}));
  const a = document.createElement("a");
  a.href = url;
  a.download = "fragment_confirm.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}});
</script>
</body>
</html>
"""


def render_review(sample: list[dict], tracks: dict, crop_root: Path, review_dir: Path) -> None:
    assets = review_dir / "assets"
    cards = []
    for s in sample:
        sides = {}
        for side, paths in (("a", tail_paths(tracks[(s["class"], s["identity_a"])], REVIEW_CROPS)),
                            ("b", head_paths(tracks[(s["class"], s["identity_b"])], REVIEW_CROPS))):
            imgs = []
            for k, rel in enumerate(paths):
                name = f"{int(s['sample_id']):03d}_{side}{k}.jpg"
                make_thumbnail(crop_root / rel, assets / name)
                imgs.append(f'<img src="assets/{name}" loading="lazy" title="{html.escape(rel)}">')
            sides[side] = imgs
        cards.append(PAIR_CARD_TEMPLATE.format(
            sample_id=s["sample_id"], cls=html.escape(s["class"]), camera=html.escape(s["camera"]),
            identity_a=html.escape(s["identity_a"]), identity_b=html.escape(s["identity_b"]),
            n_a=len(sides["a"]), n_b=len(sides["b"]), thumbs_a="".join(sides["a"]), thumbs_b="".join(sides["b"]),
        ))
    (review_dir / "fragment_review.html").write_text(
        PAGE_TEMPLATE.format(n_pairs=len(sample), cards="".join(cards)), encoding="utf-8")


def report_calibration(sample_csvs: list[Path], confirm_csvs: list[Path], sample_cfg: dict) -> None:
    sample = [r for path in sample_csvs for r in read_rows(path)]
    if not sample:
        raise SystemExit(f"找不到抽樣清單：{sample_csvs}")
    duplicated = [k for k, n in Counter(s["sample_id"] for s in sample).items() if n > 1]
    if duplicated:
        raise SystemExit(f"抽樣清單的 sample_id 重複：{sorted(duplicated)}")
    rejected = {r["sample_id"] for path in confirm_csvs for r in read_rows(path)
                if r.get("verdict", "not_same") == "not_same"}
    unknown = rejected - {s["sample_id"] for s in sample}
    if unknown:
        raise SystemExit(f"確認清單含有不在抽樣清單中的 sample_id：{sorted(unknown)}")
    labels = [(float(s["score"]), s["sample_id"] not in rejected) for s in sample]
    for stratum in (*STRATA, "extra"):
        members = [s for s in sample if s["stratum"] == stratum]
        same = sum(s["sample_id"] not in rejected for s in members)
        print(f"  {stratum}: {same}/{len(members)} same entity")
    result = calibrate_threshold(labels, sample_cfg["target_accuracy"])
    if result is None:
        print(f"NOT CALIBRATED: 沒有任何門檻能讓準確率 ≥ {sample_cfg['target_accuracy']:.0%}，不可產生資料集版本。")
        return
    print(f"threshold={result['threshold']:.6f} accuracy={result['accuracy']:.4f} "
          f"n_above={result['n_above']} wrong_above={result['n_wrong_above']}")
    if result["n_above"] < sample_cfg["min_above"]:
        print(f"WARNING: 門檻以上只有 {result['n_above']} 組（< {sample_cfg['min_above']}），需補抽。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "crops"))
    parser.add_argument("--exclusions", default=str(PROJECT_ROOT / "dataset" / "reid" / "exclusions.csv"))
    parser.add_argument("--auto-exclusions", default=str(PROJECT_ROOT / "dataset" / "reid" / "auto_exclusions.csv"))
    parser.add_argument("--features", default=str(PROJECT_ROOT / "dataset" / "reports" / "reid" / "fragment_features.npz"))
    parser.add_argument("--config", default=str(CONFIG_DIR / "reid" / "fragment_merge.yaml"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "dataset" / "reid" / "fragment_merges.csv"))
    parser.add_argument("--review-dir", default=str(PROJECT_ROOT / "dataset" / "reports" / "reid" / "fragment_review"))
    parser.add_argument("--apply", action="store_true", help="寫出合併清單（預設只 dry-run）")
    parser.add_argument("--sample", type=int, default=0, help="抽出 N 組候選產生確認頁")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=None, help="補抽：只抽相似度 ≥ 此值的候選")
    parser.add_argument("--exclude-sample", nargs="*", default=[], help="補抽：排除這些抽樣清單中已抽過的配對")
    parser.add_argument("--start-id", type=int, default=1, help="補抽：sample_id 起始值")
    parser.add_argument("--calibrate", nargs="*", default=None, help="確認頁匯出的 CSV（可多份），計算門檻")
    parser.add_argument("--sample-csv", nargs="*", default=None, help="校準用的抽樣清單（可多份）")
    args = parser.parse_args()

    cfg = load_merge_config(Path(args.config))
    review_dir = Path(args.review_dir)
    if args.calibrate:
        sample_csvs = [Path(p) for p in args.sample_csv] if args.sample_csv else [review_dir / "merge_sample.csv"]
        report_calibration(sample_csvs, [Path(p) for p in args.calibrate], cfg["sample"])
        return

    crop_root = Path(args.crop_root)
    tracks = collect_tracks(read_all_indexes(crop_root),
                            read_rows(Path(args.exclusions)) + read_rows(Path(args.auto_exclusions)))
    track_buffers = load_retention(cfg["track_config"])
    candidates = find_candidates(tracks, track_buffers, cfg["margin_frames"], cfg["position_factor"])
    features = load_features(Path(args.features))
    missing = {p for c in candidates for p in tail_paths(tracks[(c.cls, c.identity_a)], cfg["endpoint_crops"])
               + head_paths(tracks[(c.cls, c.identity_b)], cfg["endpoint_crops"]) if p not in features}
    if missing:
        raise SystemExit(f"特徵檔缺少 {len(missing)} 張端點裁切，請重跑 extract_fragment_features.py")
    scored = score_candidates(candidates, tracks, features, cfg["endpoint_crops"])

    if args.sample:
        if args.min_score is not None:
            seen = {(r["class"], r["identity_a"], r["identity_b"])
                    for path in args.exclude_sample for r in read_rows(Path(path))}
            sample = supplementary_sample(scored, args.min_score, seen, args.sample, args.seed, args.start_id)
        else:
            sample_cfg = {**cfg["sample"], "counts": allocate_proportional(args.sample, cfg["sample"]["counts"])}
            sample = stratified_sample(scored, sample_cfg, args.seed)
        review_dir.mkdir(parents=True, exist_ok=True)
        write_rows(review_dir / "merge_sample.csv", SAMPLE_FIELDS, sample)
        render_review(sample, tracks, crop_root, review_dir)
        print(f"sampled {len(sample)} pairs: {dict(Counter(s['stratum'] for s in sample))}, "
              f"non-major cameras={sum(s['camera'] not in cfg['sample']['major_cameras'] for s in sample)} "
              f"-> {review_dir}")
        return

    by_camera = Counter(s.candidate.camera for s in scored)
    threshold = cfg["threshold"]
    records = merge_records(select_pairs(scored, threshold), tracks) if threshold is not None else []
    merged = Counter(tracks[(r["class"], r["identity"])].camera for r in records)
    chains: dict[str, set] = defaultdict(set)
    for r in records:
        chains[tracks[(r["class"], r["identity"])].camera].add((r["class"], r["merged_identity"]))
    print(f"tracks={len(tracks)} candidates={len(scored)} threshold={threshold}")
    for camera in sorted(by_camera):
        print(f"  {camera}: candidates={by_camera[camera]} merged_identities={merged[camera]} "
              f"chains={len(chains[camera])}")
    if scored:
        qs = np.quantile([s.score for s in scored], [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        print("  score quantiles p10/p25/p50/p75/p90/p99: " + " ".join(f"{q:.4f}" for q in qs))

    if not args.apply:
        print("dry-run：未寫入合併清單（加 --apply 才寫入）。")
        return
    if threshold is None:
        raise SystemExit("config 的 threshold 尚未校準，不可寫入合併清單。")
    write_rows(Path(args.out), MERGE_FIELDS, records)
    print(f"wrote {len(records)} merged identities -> {args.out}")


if __name__ == "__main__":
    main()
