#!/usr/bin/env bash
# Golden-output regression run for the restructure-project-layout change.
#
# Usage:
#   tests/golden/run_all.sh <run> [--dry-run]
#
# All outputs land in runs/refactor_golden/<run>/. Entry points can be overridden
# per phase via environment variables:
#   PIPELINE_CMD       (default: "python track_eval.py")
#   GRID_SEARCH_CMD    (default: "python module/grid_search/grid_search_bytetrack.py")
#   HOMOGRAPHY_SCRIPT  (default: "module/collision/run_homography_validation.py")
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
SCENARIOS=(bytetrack botsort_noreid botsort_reid collision_detect botsort_save_reid_crops)

PIPELINE_CMD="${PIPELINE_CMD:-python track_eval.py}"
GRID_SEARCH_CMD="${GRID_SEARCH_CMD:-python module/grid_search/grid_search_bytetrack.py}"
HOMOGRAPHY_SCRIPT="${HOMOGRAPHY_SCRIPT:-module/collision/run_homography_validation.py}"

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

run_step() {
    local name="$1"; shift
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] $name: $*"
        return
    fi
    echo "=== $name"
    local started=$SECONDS
    "$@" > "$OUT/logs/$name.log" 2>&1
    echo "    done in $((SECONDS - started))s (log: $OUT/logs/$name.log)"
}

for scenario in "${SCENARIOS[@]}"; do
    # shellcheck disable=SC2086
    run_step "pipeline_$scenario" $PIPELINE_CMD --config "$CFG_DIR/$scenario.yaml"
done

# Grid search reuses the bytetrack stage1 output (videos + GT) as its source.
# shellcheck disable=SC2086
run_step grid_search $GRID_SEARCH_CMD \
    --config "$CFG_DIR/bytetrack.yaml" \
    --target-class scooter \
    --tracker bytetrack \
    --max-videos 1 \
    --params-subset fuse_score \
    --output-root "$OUT/grid_search" \
    --result-csv "$OUT/grid_search/result.csv"

# The validation script hard-codes OUTPUT_DIR; load it as a module and redirect it.
run_step homography_validation python -c '
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("golden_homography_validation", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUTPUT_DIR = Path(sys.argv[2])
module.run_all()
' "$HOMOGRAPHY_SCRIPT" "$OUT/homography_validation"

if [[ $DRY_RUN -eq 0 ]]; then
    echo "all outputs under $OUT"
fi
