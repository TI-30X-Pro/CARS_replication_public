#!/bin/bash
# Train the diffusion prior for 15,000 steps by default (GPU).
# Pass overrides through PRIOR_ARGS, for example "--max_steps 30000".
# Output: $RUNS/$RUN/{iter*.ckpt,latest.ckpt,history.jsonl}
set -euo pipefail
source "$(dirname "$0")/../config.sh"
RUN=${RUN:-prior}
"$REPO/infra/run.sh" -m cars.run.train_prior --nusc "$NUSC" --index "$WORK/nusc_index.npz" \
    --out "$RUNS/$RUN" --workers "${WORKERS:-8}" ${PRIOR_ARGS:-}
