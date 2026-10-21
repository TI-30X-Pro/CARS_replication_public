#!/bin/bash
# Generate closed-loop rollouts with the released checkpoints (GPU).
# Set SHARD to run one of NSHARDS shards; otherwise all shards run sequentially.
# Output: $RUNS/$RUN/shard*/{data.hdf5,summary.json}
set -euo pipefail
source "$(dirname "$0")/../config.sh"
NSHARDS=${NSHARDS:-24}
RUN=${RUN:-cars_released}
SHARD=${SHARD:-}

run_shard() {
    local out=$RUNS/$RUN/shard$1
    if [ -f "$out/done" ]; then echo "shard $1 already done"; return; fi
    "$REPO/infra/run.sh" -m cars.run.rollout --nusc "$NUSC" --index "$WORK/nusc_index.npz" \
        --ckpt "$CKPT" --out "$out" --shard "$1" --num_shards "$NSHARDS" ${ROLLOUT_ARGS:-}
    touch "$out/done"
}

if [ -n "$SHARD" ]; then
    run_shard "$SHARD"
else
    for s in $(seq 1 "$NSHARDS"); do run_shard "$s"; done
fi
