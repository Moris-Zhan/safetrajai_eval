"""Cell-by-cell snapshots of the Excel reports written by evaluation.excel_export."""

import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

import safetrajai.evaluation.excel_export as excel_export

SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "excel_export"
DATASET = "MotChallenge2DBox"
ALPHAS = [round(0.05 * i, 2) for i in range(1, 20)]


def _class_metrics(scale: float, with_counts: bool = True) -> dict:
    metrics = {
        "HOTA": {
            "HOTA": np.array([scale * a for a in ALPHAS]),
            "DetA": [scale * a / 2 for a in ALPHAS],
            "AssA": [scale * (1 - a) for a in ALPHAS],
        },
        "CLEAR": {"sMOTA": scale * 0.5, "CLR_FP": int(scale * 100), "CLR_TP": int(scale * 200)},
        "Identity": {"IDF1": scale * 0.25},
    }
    if with_counts:
        metrics["Count"] = {"GT_Dets": int(scale * 300), "Dets": int(scale * 280)}
    return metrics


def synthetic_trackeval_results() -> dict:
    """Per-class TrackEval results: two sequences, a COMBINED_SEQ row, one class without counts, two classes absent."""
    results = {}
    for scale, name, with_counts in ((0.9, "person", True), (0.6, "car", True), (0.3, "bus", False), (0.1, "scooter", True)):
        tracker = f"YOLO11x__{name}"
        results[name] = {DATASET: {tracker: {
            "seqA": {name: _class_metrics(scale, with_counts)},
            "seqB": {name: _class_metrics(scale / 2, with_counts)},
            "COMBINED_SEQ": {name: _class_metrics(scale / 3, with_counts)},
        }}}
    return results


def dump_workbook(path: Path) -> dict[str, list[list]]:
    workbook = load_workbook(path)
    return {sheet.title: [list(row) for row in sheet.iter_rows(values_only=True)] for sheet in workbook.worksheets}


def export_trackeval_snapshot(tmp_dir: Path) -> dict:
    out = tmp_dir / "trackeval.xlsx"
    means_data, mid_data = excel_export.export_trackeval_results_multi_to_excel(
        synthetic_trackeval_results(), str(out), tracker_base_hint="YOLO11x"
    )
    sheets = dump_workbook(out)
    flat_rows = len(sheets.pop("flat_all"))  # raw dump; only its size is pinned to keep the fixture small
    return {"flat_all_rows": flat_rows, "sheets": sheets, "means_data": means_data, "mid_data": mid_data}


def test_trackeval_workbook_matches_snapshot(tmp_path: Path) -> None:
    expected = json.loads((SNAPSHOT_DIR / "trackeval_multi.json").read_text(encoding="utf-8"))
    actual = json.loads(json.dumps(export_trackeval_snapshot(tmp_path)))
    assert actual == expected


def test_trackeval_snapshot_covers_the_key_sheets(tmp_path: Path) -> None:
    sheets = export_trackeval_snapshot(tmp_path)["sheets"]
    assert {"key_means", "key_mid"} <= set(sheets)
    header = sheets["key_means"][0]
    count_column = header.index("Count.GT_Dets_mean")
    assert any(row[3] == "bus" and row[count_column] is None for row in sheets["key_means"][1:])
    assert any(row[4] and row[4].startswith("MISSING") for row in sheets["key_mid"][1:])
