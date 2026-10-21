#!/bin/bash
# Evaluate one rollout run with responsibility attribution (CPU).
# Input: $RUNS/$RUN/shard*/data.hdf5
# Output: $RUNS/$RUN/eval/{replay_results_laneaware.csv,kinematics.csv,metrics.json}
set -euo pipefail
source "$(dirname "$0")/../config.sh"
R=$RUNS/${RUN:-cars_released}
"$REPO/infra/run.sh" -m cars.run.evaluate --hdf5 "$R"/shard*/data.hdf5 --nusc "$NUSC" \
    --out "$R/eval" --total_attempts "${TOTAL_ATTEMPTS:-679}"
