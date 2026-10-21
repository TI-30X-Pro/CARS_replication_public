#!/bin/bash
# Fine-tune the adversarial policy from g8 with PPO (GPU, long-running).
# Pass overrides through PPO_ARGS, for example "--epochs 5".
# Output: $RUNS/$RUN/{adversary_best.ckpt,adversary_last.ckpt,history.jsonl}
set -euo pipefail
source "$(dirname "$0")/../config.sh"
RUN=${RUN:-ppo_g8}
"$REPO/infra/run.sh" -m cars.run.train_ppo --nusc "$NUSC" --index "$WORK/nusc_index.npz" \
    --ckpt "$CKPT" --out "$RUNS/$RUN" ${PPO_ARGS:-}
