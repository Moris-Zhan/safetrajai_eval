#!/usr/bin/env python3
"""
Re-ID 追蹤驗收：產生驗收設定與彙總結果（spec: reid-model；change separability-gate-object-pairs D4）。

configs：以情境設定（預設 config/botsort_reid_timm.yaml）為底，套用 --set CLASS.KEY=VALUE，為每個子集
  （dataset/eval/cctv_all8_subsets/<subset>/）各寫一份設定到 <out-root>/configs/<name>_<subset>.yaml，
  輸出目錄（OUTPUT_ROOT）為 <out-root>/<name>/<subset>/；寫出後以 load_track_config() 驗證。
  執行：python track_eval.py --config <out-root>/configs/<name>_<subset>.yaml > <out-root>/<name>/<subset>/run.log 2>&1
summarize：讀取各組的 TrackEval <class>_summary.txt 與 run.log，彙總（IDSW、CLEAR TP 加總；IDP 由
  IDTP／IDFP 加總計算；HOTA、AssA 依 GT 框數加權；每幀追蹤時間依幀數加權），逐類別與基準比較並依
  spec 判定，寫出 <out>.csv 與 <out>.md。

用法：
    python scripts/reid_train/acceptance_grid.py configs --name baseline
    python scripts/reid_train/acceptance_grid.py configs --name a_car_1w --subsets tune \\
        --set car.use_reid=true --set car.model=models/class_reid_weights/dinov2s_v3_both/model.pt \\
        --set car.appearance_thresh=0.75 --set car.reid_center_gate_px=80
    python scripts/reid_train/acceptance_grid.py summarize --baseline baseline --runs a_car_1w c_car_p99 \\
        --subset tune --out runs/reid_domain/accept/summary_tune
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from safetrajai.classes import CLASS_NAMES
from safetrajai.evaluation.dataset_layout import load_dataset_registry, stage2_seq_dir
from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.reid.acceptance import aggregate, apply_overrides, judge_class, parse_summary, parse_track_time
from safetrajai.tracking.config import load_track_config

DEFAULT_OUT_ROOT = PROJECT_ROOT / "runs" / "reid_domain" / "accept"
SUBSET_ROOT = Path("dataset") / "eval" / "cctv_all8_subsets"
METRICS = ("IDSW", "CLR_TP", "IDP", "AssA", "HOTA", "GT_Dets")


def write_configs(args: argparse.Namespace) -> None:
    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    out_root = Path(args.out_root)
    for subset in args.subsets:
        config = apply_overrides(base, args.set or [])
        config["paths"]["SOURCE_VIDEO_DIR"] = f"./{(SUBSET_ROOT / subset).as_posix()}/"
        config["paths"]["OUTPUT_ROOT"] = str(out_root / args.name / subset)
        path = out_root / "configs" / f"{args.name}_{subset}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        load_track_config(path)
        print(f"[configs] {path}")


def run_metrics(run_dir: Path, config: dict) -> dict[str, dict]:
    registry = load_dataset_registry()
    track_dir = run_dir / registry[config["paths"]["dataset"]]["run_track_eval_dir"]
    te = config["trackeval"]
    split = f"{te['benchmark_name']}-{te['split_name']}"
    seqs = sorted(p.name for p in track_dir.iterdir() if p.is_dir())
    out = {}
    for cls in CLASS_NAMES.values():
        rows = [parse_summary(stage2_seq_dir(track_dir, seq) / "trackers" / "mot_challenge" / split
                              / f"{te['tracker_name']}__{cls}" / f"{cls}_summary.txt") for seq in seqs]
        out[cls] = aggregate(rows)
    return out


def summarize(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    names = [args.baseline, *args.runs]
    results, timing = {}, {}
    for name in names:
        config = yaml.safe_load((out_root / "configs" / f"{name}_{args.subset}.yaml").read_text(encoding="utf-8"))
        run_dir = out_root / name / args.subset
        results[name] = run_metrics(run_dir, config)
        log = run_dir / "run.log"
        timing[name] = parse_track_time(log.read_text(encoding="utf-8", errors="replace"))[1] if log.exists() else None

    rows = []
    for name in names:
        for cls in CLASS_NAMES.values():
            m = results[name][cls]
            if name == args.baseline:
                verdict = {"passed": None, "reason": "基準"}
            else:
                verdict = judge_class(m, results[args.baseline][cls])
            rows.append({"run": name, "class": cls, **{k: m[k] for k in METRICS},
                         "track_s_per_frame": timing[name], "passed": verdict["passed"], "reason": verdict["reason"]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value) -> str:
        if value is None:
            return "—"
        return f"{value:.4f}" if isinstance(value, float) and not float(value).is_integer() else f"{value:g}"

    lines = ["| run | class | IDSW | TP | IDP | AssA | HOTA | GT | s/frame | 判定 |", "|" + "---|" * 10]
    for r in rows:
        mark = {True: "通過", False: "不通過", None: "基準"}[r["passed"]]
        lines.append(f"| {r['run']} | {r['class']} | {fmt(r['IDSW'])} | {fmt(r['CLR_TP'])} | {fmt(r['IDP'])} | "
                     f"{fmt(r['AssA'])} | {fmt(r['HOTA'])} | {fmt(r['GT_Dets'])} | {fmt(r['track_s_per_frame'])} | "
                     f"{mark}：{r['reason']} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    configs = sub.add_parser("configs", help="產生驗收設定")
    configs.add_argument("--name", required=True, help="組名，例如 baseline、a_car_1w")
    configs.add_argument("--set", action="append", help="CLASS.KEY=VALUE（VALUE 以 YAML 解析），可重複")
    configs.add_argument("--subsets", nargs="+", default=["tune", "holdout"])
    configs.add_argument("--base", default=str(CONFIG_DIR / "botsort_reid_timm.yaml"))
    configs.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))

    summ = sub.add_parser("summarize", help="彙總並逐類別判定")
    summ.add_argument("--baseline", required=True)
    summ.add_argument("--runs", nargs="*", default=[])
    summ.add_argument("--subset", default="tune")
    summ.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    summ.add_argument("--out", required=True, help="輸出路徑前綴（寫出 .csv 與 .md）")

    args = parser.parse_args()
    {"configs": write_configs, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
