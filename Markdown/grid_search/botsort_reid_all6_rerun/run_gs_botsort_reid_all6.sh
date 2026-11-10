#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate track_eval
for cls in person car scooter small_truck bus truck; do
  echo "=== $(date) START $cls ==="
  python scripts/grid_search/grid_search_botsort_reid.py \
    --config runs/grid_search_coverage/configs/gs_botsort_reid_all6.yaml \
    --target-class "$cls" \
    --params-subset match_thresh,second_match_thresh,unconfirmed_match_thresh \
    --result-csv "runs/grid_search_coverage/gs_botsort_reid/results_all6/${cls}.csv"
  echo "=== $(date) DONE $cls ==="
done
echo "ALL DONE"
