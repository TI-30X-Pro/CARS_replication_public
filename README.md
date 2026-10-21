# CARS replication

Python replication of CARS, *Learning Responsibility-Attributed Adversarial Scenarios
for Testing Autonomous Vehicles* (Xiao et al. 2026,
[arXiv:2605.13751](https://arxiv.org/abs/2605.13751)).

## Repository layout

The repository contains:

- `cars/`: the replication, organised by method stage;
- `run_pipeline.py`: one command for selector validation, rollout, and evaluation;
- `configs/`: nuScenes labels and rollout settings;
- `tests/`: self-contained unit and integration tests;
- `infra/`: installation and run scripts for a regular Python environment;
- `docs/`: implementation notes and replication results;
- `CARS_original/`: the official release, retained unchanged for provenance and the shipped
  model configuration and selector assets.

## 1. Install dependencies

Python 3.11 or newer is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
PIP=.venv/bin/pip bash infra/install_cars.sh
export CARS_PYTHON=.venv/bin/python
```

## 2. Prepare data and checkpoints

The nuScenes trainval metadata and map expansion are required. For an unpacked dataset:

```bash
NUSC=/path/to/nuscenes bash infra/steps/setup.sh
```

For a tar containing `nuscenes/v1.0-trainval` and `nuscenes/maps`:

```bash
NUSC_TAR=/path/to/nuscenes.tar bash infra/steps/setup.sh
```

Setup downloads the released checkpoints into `work/checkpoints`. The `work/` directory is
ignored by Git because it contains datasets, checkpoints, and run outputs.

## 3. Run the pipeline

```bash
python run_pipeline.py --nusc /path/to/nuscenes
```

This validates the selector, runs the released prior and adversary checkpoints, and evaluates
the resulting collisions. It does not train the prior or PPO models. For a short smoke run:

```bash
python run_pipeline.py --nusc /path/to/nuscenes --run cars_smoke --shard 1 --num-scenes 2
```

Completed shards are skipped automatically. Use `python run_pipeline.py --help` for stage,
shard, device, and selector options. See `infra/README.md` and `docs/` for details.

## Verify the installation

```bash
infra/run.sh -m pytest -q
```

Checkpoint-loading tests skip when the released checkpoints are unavailable.
