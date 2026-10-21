# Stage 4 — Run and evaluate the full pipeline

## Pipeline

`python run_pipeline.py` validates the selector, runs the complete scenario pipeline for each
labelled pair, and evaluates the collision episodes. The target and other agents use `g8`, the
selected adversary uses `cars_adversary`, and each agent samples eight plans before executing the
one closest to the logged trajectory. Re-planning happens every five steps. The prior and PPO
models are loaded from released checkpoints and are not trained by this command.

Settings are stored in `configs/rollout_nusc.json`: 311 rollout scenes, 676 runnable pairs,
seed 42, 120 simulation steps, five-step re-planning, eight samples, and 24 shards.

## Smoke test

```bash
python run_pipeline.py --nusc /path/to/nuscenes --run cars_smoke --shard 1 --num-scenes 2
```

## Full run

```bash
python run_pipeline.py --nusc /path/to/nuscenes
```

A GPU is strongly recommended because diffusion sampling dominates the runtime.

## Replication results

The completed 676-pair run on an RTX 3090 produced:

| Metric | Result |
|---|---:|
| Collisions | 427 (63.2%) |
| FSM avoidable | 87.1% |
| CC-JP avoidable | 80.3% |
| RSS avoidable | 92.5% |
| Hard / Medium / Easy | 47 / 224 / 156 |
| H_crit (95% lower bound) | 0.790 (0.743) |
| BD+ | 23.4% |
| IP | 0.065% |

These results are retained as a record of the replication run; large rollout files and
checkpoints remain under the Git-ignored `work/` directory.
