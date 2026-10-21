#!/bin/bash
# Train and evaluate the adversary selector (CPU, about 5 minutes).
# Output: $WORK/selector/{trees.npz,samples.npz,metrics.json}
set -euo pipefail
source "$(dirname "$0")/../config.sh"
OUT=$WORK/selector
"$REPO/infra/run.sh" -m cars.run.train_selector --nusc "$NUSC" --out "$OUT" \
    --index "$WORK/nusc_index.npz"
