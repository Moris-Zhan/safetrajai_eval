#!/usr/bin/env bash
# Golden-output regression run for the restructure-project-layout change.
#
# Usage:
#   tests/golden/run_all.sh <run> [--dry-run]
#
# All outputs land in runs/refactor_golden/<run>/. Entry points can be overridden
# per phase via environment variables:
#   PIPELINE_CMD       (default: "python track_eval.py")
#   GRID_SEARCH_CMD    (default: "python scripts/grid_search/grid_search_bytetrack.py")
#   HOMOGRAPHY_SCRIPT  (default: "scripts/homography/validate_homography.py")
#
# Steps run in parallel (the pipelines are CPU-bound on a single core each);
# grid search starts once the bytetrack pipeline has produced its stage1 output.
# Set GOLDEN_PARALLEL=0 to run the steps one at a time.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <run> [--dry-run]" >&2
    exit 2
fi

RUN="$1"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1

if [[ ! "$RUN" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "invalid run name: $RUN" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GOLDEN_ROOT="runs/refactor_golden"
OUT="$GOLDEN_ROOT/$RUN"
SCENARIOS=(bytetrack botsort_noreid botsort_reid collision_detect botsort_save_reid_crops conflict_analysis)

PIPELINE_CMD="${PIPELINE_CMD:-python track_eval.py}"
GRID_SEARCH_CMD="${GRID_SEARCH_CMD:-python scripts/grid_search/grid_search_bytetrack.py}"
HOMOGRAPHY_SCRIPT="${HOMOGRAPHY_SCRIPT:-scripts/homography/validate_homography.py}"
PARALLEL="${GOLDEN_PARALLEL:-1}"

if [[ $DRY_RUN -eq 0 && -e "$OUT" ]]; then
    echo "refusing to overwrite existing $OUT" >&2
    exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
    CFG_DIR="$(mktemp -d)"
    trap 'rm -rf "$CFG_DIR"' EXIT
else
    CFG_DIR="$OUT/_configs"
    mkdir -p "$CFG_DIR" "$OUT/logs"
fi

for scenario in "${SCENARIOS[@]}"; do
    sed "s#__RUN__#$RUN#g" "tests/golden/configs/$scenario.yaml" > "$CFG_DIR/$scenario.yaml"
done

# Every output path declared by a rendered config must stay under $OUT.
for scenario in "${SCENARIOS[@]}"; do
    while read -r value; do
        case "$value" in
            "./$OUT/"*) ;;
            *) echo "output path escapes $OUT in $scenario: $value" >&2; exit 1 ;;
        esac
    done < <(grep -E '^\s*(OUTPUT_ROOT|reid_crop_dir):' "$CFG_DIR/$scenario.yaml" | awk '{print $2}')
done

declare -A STEP_PIDS=()
FAILED=0

wait_step() {
    local name="$1"
    [[ $DRY_RUN -eq 1 ]] && return 0
    local pid="${STEP_PIDS[$name]}"
    unset "STEP_PIDS[$name]"
    wait "$pid" || FAILED=1
}

start_step() {
    local name="$1"; shift
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] $name: $*"
        return
    fi
    echo "=== start $name"
    (
        set +e
        started=$SECONDS
        "$@" > "$OUT/logs/$name.log" 2>&1
        rc=$?
        if [[ $rc -eq 0 ]]; then
            echo "    $name done in $((SECONDS - started))s"
        else
            echo "    $name FAILED (exit $rc) after $((SECONDS - started))s (log: $OUT/logs/$name.log)"
        fi
        exit $rc
    ) &
    STEP_PIDS[$name]=$!
    if [[ $PARALLEL -eq 0 ]]; then
        wait_step "$name"
    fi
}

trap 'kill $(jobs -p) 2>/dev/null' INT TERM

for scenario in "${SCENARIOS[@]}"; do
    # shellcheck disable=SC2086
    start_step "pipeline_$scenario" $PIPELINE_CMD --config "$CFG_DIR/$scenario.yaml"
done

# The validation script hard-codes OUTPUT_DIR; load it as a module and redirect it.
start_step homography_validation python -c '
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("golden_homography_validation", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUTPUT_DIR = Path(sys.argv[2])
module.run_all()
' "$HOMOGRAPHY_SCRIPT" "$OUT/homography_validation"

# Grid search reuses the bytetrack stage1 output (videos + GT) as its source.
if [[ -n "${STEP_PIDS[pipeline_bytetrack]:-}" ]]; then
    wait_step pipeline_bytetrack
fi
if [[ $FAILED -eq 0 ]]; then
    # shellcheck disable=SC2086
    start_step grid_search $GRID_SEARCH_CMD \
        --config "$CFG_DIR/bytetrack.yaml" \
        --target-class scooter \
        --tracker bytetrack \
        --max-videos 1 \
        --params-subset fuse_score \
        --output-root "$OUT/grid_search" \
        --result-csv "$OUT/grid_search/result.csv"
else
    echo "    grid_search skipped: pipeline_bytetrack failed"
fi

for name in "${!STEP_PIDS[@]}"; do
    wait_step "$name"
done

if [[ $FAILED -ne 0 ]]; then
    echo "one or more steps failed under $OUT" >&2
    exit 1
fi
if [[ $DRY_RUN -eq 0 ]]; then
    echo "all outputs under $OUT"
fi
