#!/usr/bin/env python3
"""
Draw the seeded audit sample for the user's first web task (Q4 of the pilot_v1 review plan):
N applied corrections that Claude judged on its own, plus N appearance-sample representatives
that Claude kept unchanged (judged genuine, or left ambiguous and never corrected).

Output: runs/cvat_review/audit_sample.json, read by serve_audit.py.

Usage (plain python3, stdlib only):
    python3 scripts/cvat_review/build_audit_sample.py --seed 0 --per-group 40
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
APPEARANCE_GLOB = "*_appearance_sample*.csv"
USER_CONFIRMED_MARKER = "user-confirmed"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_representatives(review_dir: Path) -> List[Dict[str, str]]:
    reps: List[Dict[str, str]] = []
    for path in sorted(review_dir.glob(APPEARANCE_GLOB)):
        label = path.name.split("_appearance_sample")[0]
        for row in read_csv(path):
            reps.append({
                "track_id": row["representative_track_id"],
                "source_label": label,
                "source_file": path.name,
                "cluster_size": row["size"],
            })
    return reps


def build_sample(review_dir: Path, per_group: int, seed: int, ambiguous_path: Path) -> Dict[str, object]:
    rng = random.Random(seed)
    applied = read_csv(review_dir / "corrections_applied.csv")
    applied_ids = {row["track_id"] for row in applied}

    eligible_applied = [row for row in applied if USER_CONFIRMED_MARKER not in row["reason"]]
    sampled_applied = rng.sample(eligible_applied, min(per_group, len(eligible_applied)))

    ambiguous_rows = read_csv(ambiguous_path) if ambiguous_path.exists() else []
    ambiguous_ids = {row["track_id"] for row in ambiguous_rows}
    eligible_reps = [
        rep for rep in load_representatives(review_dir)
        if rep["track_id"] not in applied_ids and rep["track_id"] not in ambiguous_ids
    ]
    sampled_reps = rng.sample(eligible_reps, min(per_group, len(eligible_reps)))

    items = [
        {
            "item_id": f"applied-{row['track_id']}",
            "kind": "applied_correction",
            "track_id": int(row["track_id"]),
            "old_label": row["old_label"],
            "claimed_final": row["new_label"] if row["action"] == "relabel" else "delete",
            "reason": row["reason"],
        }
        for row in sampled_applied
    ] + [
        {
            "item_id": f"rep-{rep['source_label']}-{rep['track_id']}",
            "kind": "unchanged_representative",
            "track_id": int(rep["track_id"]),
            "old_label": rep["source_label"],
            "claimed_final": rep["source_label"],
            "reason": f"{rep['source_file']} cluster size {rep['cluster_size']}",
        }
        for rep in sampled_reps
    ] + [
        {
            "item_id": f"ambiguous-{row['track_id']}",
            "kind": "ambiguous_representative",
            "track_id": int(row["track_id"]),
            "old_label": row["current_label"],
            "claimed_final": row["current_label"],
            "reason": f"{row['verdict']}: {row['note']}",
        }
        for row in ambiguous_rows
    ]
    return {
        "seed": seed,
        "per_group": per_group,
        "eligible_applied": len(eligible_applied),
        "eligible_representatives": len(eligible_reps),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-group", type=int, default=40)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    parser.add_argument("--ambiguous-csv", type=Path, default=REVIEW_DIR / "ambiguous_reps.csv",
                        help="track_id,current_label,verdict,note; all rows go to the page, none are in the genuine draw")
    parser.add_argument("--out", type=Path, default=REVIEW_DIR / "audit_sample.json")
    args = parser.parse_args()

    sample = build_sample(args.review_dir, args.per_group, args.seed, args.ambiguous_csv)
    args.out.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(sample['items'])} items "
          f"(eligible applied {sample['eligible_applied']}, eligible representatives {sample['eligible_representatives']})")


if __name__ == "__main__":
    main()
