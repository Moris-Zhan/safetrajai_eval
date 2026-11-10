#!/bin/bash
set -e
cd /mnt/moris/Dcoker_Project/Project/safetrajai_eval
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate track_eval

names="baseline reid_person reid_car reid_scooter reid_small_truck reid_bus reid_truck"
for name in $names; do
  for subset in tune holdout; do
    cfg="runs/reid_domain/accept/configs/${name}_${subset}.yaml"
    outdir="runs/reid_domain/accept/${name}/${subset}"
    mkdir -p "$outdir"
    echo "=== running $cfg -> $outdir/run.log ==="
    python track_eval.py --config "$cfg" > "$outdir/run.log" 2>&1
    echo "=== done $cfg (exit $?) ==="
  done
done
echo "ALL_ACCEPTANCE_RUNS_DONE"
