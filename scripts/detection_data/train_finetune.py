#!/usr/bin/env python3
"""
Fine-tune the detector on an exported dataset (export_training_set.py) and record the run.

Round 1 uses Ultralytics defaults on purpose (no tuning); later rounds change one option at a time
(--lr0, --freeze, --epochs). The best checkpoint is chosen by the validation split of the export
(C000012 and one C000054 sequence), never by the evaluation camera C000025.

Output: runs/finetune/<name>/ (Ultralytics run folder, weights/best.pt) and train_record.json

Usage (conda env track_eval; run in the background, it takes a long time):
    nohup python scripts/detection_data/train_finetune.py --export ft_v1 --name ft_v1_r1 > runs/finetune/ft_v1_r1.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORTS = PROJECT_ROOT / "dataset" / "detection" / "exports"
RUNS = PROJECT_ROOT / "runs" / "finetune"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "yolo11x_base.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, help="export version under dataset/detection/exports/")
    parser.add_argument("--name", required=True, help="run folder name under runs/finetune/")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=None, help="initial learning rate (Ultralytics default if unset)")
    parser.add_argument("--optimizer", default=None, help="SGD, Adam, AdamW...; with the default auto, Ultralytics picks its own and ignores --lr0")
    parser.add_argument("--freeze", type=int, default=None, help="freeze the first N layers")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue the existing run --name from its weights/last.pt (all other options come from that run)")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers (lower it if training segfaults)")
    args = parser.parse_args()

    data_yaml = EXPORTS / args.export / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found; run export_training_set.py first")
    if (RUNS / args.name).exists() and not args.resume:
        raise SystemExit(f"{RUNS / args.name} exists; use a new --name for every run")

    from ultralytics import YOLO

    options = {"data": str(data_yaml), "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
               "patience": args.patience, "seed": args.seed, "project": str(RUNS), "name": args.name,
               "exist_ok": False, "workers": args.workers}
    if args.optimizer is not None:
        options["optimizer"] = args.optimizer
    if args.lr0 is not None:
        options["lr0"] = args.lr0
    if args.freeze is not None:
        options["freeze"] = args.freeze

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.resume:
        import yaml

        # The run's real options live in its args.yaml; the argparse defaults above do not apply.
        run_args = yaml.safe_load((RUNS / args.name / "args.yaml").read_text(encoding="utf-8"))
        options = {key: run_args.get(key) for key in ("data", "epochs", "imgsz", "batch", "patience", "seed",
                                                       "project", "name", "workers", "optimizer", "lr0", "freeze")}
        options["resumed"] = True
        YOLO(str(RUNS / args.name / "weights" / "last.pt")).train(resume=True)
    else:
        YOLO(str(args.weights)).train(**options)

    run_dir = RUNS / args.name
    export_record = json.loads((EXPORTS / args.export / "export_record.json").read_text(encoding="utf-8"))
    (run_dir / "train_record.json").write_text(json.dumps({
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": str(args.weights), "export": args.export, "options": options, "git_commit": commit,
        "export_record": export_record}, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
