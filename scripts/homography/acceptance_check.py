#!/usr/bin/env python3
"""
Homography 驗收 spec（training_eval_audit.md 補強優先順序 #4 的 spec 骨架）。

讀取 validate_homography.py 已產生的 models/homography/validation_output/summary.csv，
套用誤差門檻判定每支鏡頭是否「驗收通過」，輸出 CSV＋Markdown 報表（做法仿照
scripts/reid_train/acceptance_grid.py 的 summarize：讀既有輸出、套門檻、出報表，
不重新執行驗證本身）。

現行 validate_camera()／validate_homography.py 的 validated 欄位只檢查 check points
數量是否 >= MIN_CHECKPOINTS，完全沒有誤差大小的門檻（2026-11-11 調查印證了這個弱點：
見 Markdown/homography/checkpoint_validation_2026-11-11.md）。這支腳本刻意不改動
validate_homography.py 既有的輸出與 validated 欄位語意（避免連動影響 golden 比對），
而是另外疊加一層誤差門檻判定，單獨輸出報表。

誤差門檻（ACCEPTANCE_RMSE_M）目前刻意留 None（待定）——2026-11-11 使用者明確決定
「先不定閾值，等補完點實測數據再決定」。門檻是 None 時，點數檢查合格但無門檻可比對的
鏡頭會回報 threshold_tbd，不會被誤報為通過或失敗；之後決定門檻時，只需要把這個常數
改成實際數值即可，不需要改動其他邏輯或報表格式。

用法：
    python scripts/homography/acceptance_check.py
    python scripts/homography/acceptance_check.py --out Markdown/homography/acceptance_report
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from safetrajai.collision.homography.validation import MIN_CHECKPOINTS
from safetrajai.paths import PROJECT_ROOT

SUMMARY_CSV = PROJECT_ROOT / "models" / "homography" / "validation_output" / "summary.csv"

# 只有這 4 支鏡頭的 meters_per_pixel() 會被 conflict_analysis 實際使用（見 CLAUDE.md
# 「衝突指標分析」一節），其餘有矩陣的鏡頭跟衝突指標的可信度無關，不在這份驗收 spec 範圍內。
CONFLICT_CAMERAS = ("C000025", "C000043", "C000054", "C000085")

# 待定：使用者 2026-11-11 決定先不設定誤差門檻，等累積更多鏡頭的實測資料後再決定。
# 單位是公尺（對應 summary.csv 的 check_dist_rmse_m 欄）。None 代表「尚未定案」，
# 不是「無限寬鬆」——verdict() 會回報 threshold_tbd，不會冒充 pass。
ACCEPTANCE_RMSE_M: float | None = None


def verdict(row: dict) -> str:
    check_n = row.get("check_N")
    if not check_n:
        return "no_checkpoints"
    if int(float(check_n)) < MIN_CHECKPOINTS:
        return "too_few_checkpoints"
    rmse_m = row.get("check_dist_rmse_m")
    if not rmse_m:
        return "no_meters_per_pixel"
    if ACCEPTANCE_RMSE_M is None:
        return "threshold_tbd"
    return "pass" if float(rmse_m) <= ACCEPTANCE_RMSE_M else "fail"


def build_rows(summary_csv: Path) -> list[dict]:
    with summary_csv.open(encoding="utf-8") as f:
        by_camera = {row["camera"]: row for row in csv.DictReader(f)}

    rows = []
    for camera in CONFLICT_CAMERAS:
        row = by_camera.get(camera)
        if row is None:
            rows.append({"camera": camera, "check_N": "", "rmse_m": "", "verdict": "no_matrix"})
            continue
        rows.append({
            "camera": camera,
            "check_N": row.get("check_N", ""),
            "rmse_m": row.get("check_dist_rmse_m", ""),
            "verdict": verdict(row),
        })
    return rows


def write_report(rows: list[dict], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["camera", "check_N", "rmse_m", "verdict"])
        writer.writeheader()
        writer.writerows(rows)

    threshold_text = "待定（見腳本註解 ACCEPTANCE_RMSE_M）" if ACCEPTANCE_RMSE_M is None else f"{ACCEPTANCE_RMSE_M} m"
    lines = [
        "# Homography 驗收 spec 報表",
        "",
        f"誤差門檻（`ACCEPTANCE_RMSE_M`）：{threshold_text}",
        "",
        "| 鏡頭 | check_N | RMSE (m) | 判定 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['camera']} | {row['check_N']} | {row['rmse_m']} | {row['verdict']} |")
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "models" / "homography" / "validation_output" / "acceptance_report"),
        help="輸出路徑前綴（寫出 .csv 與 .md）",
    )
    parser.add_argument("--summary-csv", default=str(SUMMARY_CSV))
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    if not summary_csv.exists():
        raise SystemExit(f"找不到 {summary_csv}，請先執行 python scripts/homography/validate_homography.py")

    rows = build_rows(summary_csv)
    write_report(rows, Path(args.out))
    for row in rows:
        print(f"[{row['camera']}] check_N={row['check_N']} rmse_m={row['rmse_m']} -> {row['verdict']}")


if __name__ == "__main__":
    main()
