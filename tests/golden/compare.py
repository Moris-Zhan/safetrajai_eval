"""Compare two golden runs under runs/refactor_golden/ and print a markdown report.

Usage:
    python tests/golden/compare.py <runA> <runB> [--json report.json]

Exit status is 0 when the runs are identical, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import openpyxl

GOLDEN_ROOT = Path(__file__).resolve().parents[2] / "runs" / "refactor_golden"

SKIPPED_TOP_DIRS = frozenset({"_configs", "logs"})
PATH_ONLY_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".pdf", ".mp4", ".avi", ".npy", ".pt"})
IGNORED_CSV_COLUMNS = frozenset({"elapsed_sec"})
IGNORED_LINE_PATTERN = re.compile(r"^#\s*Generated:")
TOKEN_SPLIT = re.compile(r"[,\s]+")
# Outputs may embed their own run directory (e.g. data.yaml `path:`); mask the run name.
RUN_PREFIX = re.compile(r"runs/refactor_golden/[^/\s\"']+/")

CATEGORY_ORDER = (
    "mot_tracks",
    "yolo_labels",
    "trackeval",
    "grid_search",
    "conflict_events",
    "conflict",
    "reid_crops",
    "homography",
    "other",
)


@dataclass(frozen=True)
class FileDiff:
    max_abs_diff: float = 0.0
    value_mismatches: int = 0
    structural: str | None = None

    @property
    def identical(self) -> bool:
        return self.max_abs_diff == 0.0 and self.value_mismatches == 0 and self.structural is None


@dataclass
class CategorySummary:
    files_a: int = 0
    files_b: int = 0
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)
    differing: list[tuple[str, FileDiff]] = field(default_factory=list)
    max_abs_diff: float = 0.0
    value_mismatches: int = 0

    @property
    def identical(self) -> bool:
        return not (self.only_a or self.only_b or self.differing)


def categorize(relative_path: str) -> str:
    parts = relative_path.split("/")
    if "reid_crops" in parts:
        return "reid_crops"
    if parts[0] == "grid_search":
        return "grid_search"
    if parts[0] == "homography_validation":
        return "homography"
    if "collision_output" in parts and "conflict" in parts:
        return "conflict"
    if "collision_output" in parts:
        return "conflict_events"
    if "trackeval_output" in parts or relative_path.endswith("_summary.txt"):
        return "trackeval"
    if "trackers" in parts and "data" in parts and relative_path.endswith(".txt"):
        return "mot_tracks"
    if "labels" in parts and relative_path.endswith(".txt"):
        return "yolo_labels"
    return "other"


def list_files(run_dir: Path) -> dict[str, Path]:
    return {
        path.relative_to(run_dir).as_posix(): path
        for path in run_dir.rglob("*")
        if path.is_file() and path.relative_to(run_dir).parts[0] not in SKIPPED_TOP_DIRS
    }


def to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def merge(diffs: Iterable[FileDiff]) -> FileDiff:
    max_abs_diff = 0.0
    value_mismatches = 0
    structural = None
    for diff in diffs:
        max_abs_diff = max(max_abs_diff, diff.max_abs_diff)
        value_mismatches += diff.value_mismatches
        structural = structural or diff.structural
    return FileDiff(max_abs_diff, value_mismatches, structural)


def compare_scalars(value_a: Any, value_b: Any) -> FileDiff:
    number_a, number_b = to_float(value_a), to_float(value_b)
    if number_a is not None and number_b is not None:
        if (math.isnan(number_a) and math.isnan(number_b)) or number_a == number_b:
            return FileDiff()
        delta = abs(number_a - number_b)
        if math.isnan(delta):
            return FileDiff(value_mismatches=1)
        return FileDiff(max_abs_diff=delta, value_mismatches=int(delta > 0))
    if isinstance(value_a, str) and isinstance(value_b, str):
        value_a = RUN_PREFIX.sub("runs/refactor_golden/<run>/", value_a)
        value_b = RUN_PREFIX.sub("runs/refactor_golden/<run>/", value_b)
    return FileDiff(value_mismatches=int(value_a != value_b))


def compare_values(value_a: Any, value_b: Any, where: str = "$") -> FileDiff:
    if isinstance(value_a, dict) and isinstance(value_b, dict):
        if set(value_a) != set(value_b):
            return FileDiff(structural=f"{where}: keys differ")
        return merge(compare_values(value_a[k], value_b[k], f"{where}.{k}") for k in value_a)
    if isinstance(value_a, list) and isinstance(value_b, list):
        if len(value_a) != len(value_b):
            return FileDiff(structural=f"{where}: length {len(value_a)} vs {len(value_b)}")
        return merge(compare_values(a, b, f"{where}[{i}]") for i, (a, b) in enumerate(zip(value_a, value_b)))
    return compare_scalars(value_a, value_b)


def parse_json_cell(value: str) -> Any:
    if value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def compare_text(path_a: Path, path_b: Path) -> FileDiff:
    def rows(path: Path) -> list[list[str]]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return [
            [token for token in TOKEN_SPLIT.split(line.strip()) if token]
            for line in lines
            if not IGNORED_LINE_PATTERN.match(line)
        ]

    rows_a, rows_b = rows(path_a), rows(path_b)
    if len(rows_a) != len(rows_b):
        return FileDiff(structural=f"line count {len(rows_a)} vs {len(rows_b)}")
    return compare_values(rows_a, rows_b)


def compare_csv(path_a: Path, path_b: Path) -> FileDiff:
    def rows(path: Path) -> list[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8-sig") as f:
            return [
                {k: parse_json_cell(v) for k, v in row.items() if k not in IGNORED_CSV_COLUMNS}
                for row in csv.DictReader(f)
            ]

    return compare_values(rows(path_a), rows(path_b))


def compare_json(path_a: Path, path_b: Path) -> FileDiff:
    return compare_values(
        json.loads(path_a.read_text(encoding="utf-8")),
        json.loads(path_b.read_text(encoding="utf-8")),
    )


def compare_xlsx(path_a: Path, path_b: Path) -> FileDiff:
    def sheets(path: Path) -> dict[str, list[list[Any]]]:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            return {ws.title: [list(row) for row in ws.iter_rows(values_only=True)] for ws in workbook.worksheets}
        finally:
            workbook.close()

    return compare_values(sheets(path_a), sheets(path_b))


def comparator_for(path: Path) -> Callable[[Path, Path], FileDiff] | None:
    suffix = path.suffix.lower()
    if suffix in PATH_ONLY_SUFFIXES:
        return None
    return {
        ".csv": compare_csv,
        ".json": compare_json,
        ".xlsx": compare_xlsx,
    }.get(suffix, compare_text)


def compare_runs(run_a: Path, run_b: Path) -> dict[str, CategorySummary]:
    files_a, files_b = list_files(run_a), list_files(run_b)
    summaries = {name: CategorySummary() for name in CATEGORY_ORDER}

    for rel in files_a:
        summaries[categorize(rel)].files_a += 1
    for rel in files_b:
        summaries[categorize(rel)].files_b += 1
    for rel in sorted(files_a.keys() - files_b.keys()):
        summaries[categorize(rel)].only_a.append(rel)
    for rel in sorted(files_b.keys() - files_a.keys()):
        summaries[categorize(rel)].only_b.append(rel)

    for rel in sorted(files_a.keys() & files_b.keys()):
        comparator = comparator_for(files_a[rel])
        if comparator is None:
            continue
        try:
            diff = comparator(files_a[rel], files_b[rel])
        except Exception as error:  # noqa: BLE001 - report unreadable files as structural diffs
            diff = FileDiff(structural=f"{type(error).__name__}: {error}")
        summary = summaries[categorize(rel)]
        summary.max_abs_diff = max(summary.max_abs_diff, diff.max_abs_diff)
        summary.value_mismatches += diff.value_mismatches
        if not diff.identical:
            summary.differing.append((rel, diff))

    return summaries


def render_markdown(run_a: str, run_b: str, summaries: dict[str, CategorySummary], detail_limit: int) -> str:
    lines = [
        f"# Golden compare: `{run_a}` vs `{run_b}`",
        "",
        "| category | files A | files B | only A | only B | differing files | value mismatches | max abs diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | {s.files_a} | {s.files_b} | {len(s.only_a)} | {len(s.only_b)} "
            f"| {len(s.differing)} | {s.value_mismatches} | {s.max_abs_diff:.6g} |"
        )

    for name, s in summaries.items():
        if s.identical:
            continue
        lines += ["", f"## {name}"]
        lines += [f"- only in A: `{rel}`" for rel in s.only_a[:detail_limit]]
        lines += [f"- only in B: `{rel}`" for rel in s.only_b[:detail_limit]]
        ranked = sorted(s.differing, key=lambda item: (item[1].structural is None, -item[1].max_abs_diff))
        for rel, diff in ranked[:detail_limit]:
            note = diff.structural or f"{diff.value_mismatches} values, max abs diff {diff.max_abs_diff:.6g}"
            lines.append(f"- `{rel}`: {note}")
        hidden = len(s.only_a) + len(s.only_b) + len(s.differing) - 3 * detail_limit
        if hidden > 0:
            lines.append(f"- ... and up to {hidden} more")

    identical = all(s.identical for s in summaries.values())
    lines += ["", f"**Result:** {'identical' if identical else 'differences found'}"]
    return "\n".join(lines)


def resolve_run(name: str) -> Path:
    candidate = Path(name)
    run_dir = candidate if candidate.is_dir() else GOLDEN_ROOT / name
    if not run_dir.is_dir():
        sys.exit(f"run not found: {name}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--json", type=Path, default=None, help="also write a machine-readable summary")
    parser.add_argument("--detail-limit", type=int, default=20)
    args = parser.parse_args()

    summaries = compare_runs(resolve_run(args.run_a), resolve_run(args.run_b))
    print(render_markdown(args.run_a, args.run_b, summaries, args.detail_limit))

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    name: {
                        "files_a": s.files_a,
                        "files_b": s.files_b,
                        "only_a": s.only_a,
                        "only_b": s.only_b,
                        "differing": [
                            {"path": rel, "max_abs_diff": d.max_abs_diff,
                             "value_mismatches": d.value_mismatches, "structural": d.structural}
                            for rel, d in s.differing
                        ],
                        "max_abs_diff": s.max_abs_diff,
                        "value_mismatches": s.value_mismatches,
                    }
                    for name, s in summaries.items()
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    sys.exit(0 if all(s.identical for s in summaries.values()) else 1)


if __name__ == "__main__":
    main()
