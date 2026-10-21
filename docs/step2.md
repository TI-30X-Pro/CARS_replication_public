# Stage 2 — Generate adversarial trajectories

## Components

`cars/generation/` contains:

- `nets.py`: map, history, attention, and denoising network blocks;
- `gmdm.py`: the Gaussian-mixture diffusion model, training loss, and 20-step DDIM sampling;
- `policy.py`: multi-plan rollout policies and role assignment;
- `reward.py`: the adversary approach reward;
- `ppo.py` and `ppo_env.py`: PPO over the denoising chain;
- `data.py`: prior-training data.

Released `g8.ckpt` and `cars_adversary.ckpt` checkpoints load directly into the replication.
The architecture configuration is retained in `CARS_original/checkpoints/config.json`.

## Train the diffusion prior

```bash
python -m cars.run.train_prior \
  --nusc "$NUSC" \
  --index "$WORK/nusc_index.npz" \
  --out "$RUNS/prior"
```

The default recipe uses batch size 64, AdamW with learning rate `1e-5` and weight decay `1e-4`,
mixed precision on GPU, periodic validation, and checkpoints every 5,000 steps. It stops at
15,000 steps by default.

## Fine-tune the adversary with PPO

```bash
python -m cars.run.train_ppo \
  --nusc "$NUSC" \
  --index "$WORK/nusc_index.npz" \
  --ckpt "$CKPT" \
  --out "$RUNS/ppo_g8"
```

PPO starts from `g8.ckpt` unless another prior is selected. It writes the best and last
adversary checkpoints together with `history.jsonl`.

## Tests

Tests in `tests/test_generation.py` cover the diffusion schedule, checkpoint loading, plan
selection with missing logged steps, generalised-advantage termination handling, and minibatch
construction.
