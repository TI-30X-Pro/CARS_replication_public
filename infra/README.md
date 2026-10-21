# Replication run guide

The scripts in `steps/` run in a regular Python environment. They do not require a container,
Slurm, or a site-specific cluster configuration.

## 1. Create the environment

Create and activate a Python 3.11+ virtual environment, then install the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
PIP=.venv/bin/pip bash infra/install_cars.sh
```

By default, the scripts call `python`. Set `CARS_PYTHON` when a different interpreter should be
used:

```bash
export CARS_PYTHON=$PWD/.venv/bin/python
```

## 2. Configure paths

`infra/config.sh` defines the shared paths. Environment variables override its defaults:

- `WORK`: local data and outputs, default `./work`;
- `NUSC`: unpacked nuScenes root, default `$WORK/nuscenes`;
- `CKPT`: released checkpoints, default `$WORK/checkpoints`;
- `RUNS`: rollout and training outputs, default `$WORK/runs`.

## 3. Prepare data and checkpoints

If nuScenes is already unpacked:

```bash
NUSC=/path/to/nuscenes bash infra/steps/setup.sh
```

Otherwise, provide a tar containing `nuscenes/v1.0-trainval` and `nuscenes/maps`:

```bash
NUSC_TAR=/path/to/nuscenes.tar bash infra/steps/setup.sh
```

The setup step also downloads the released checkpoints.

## 4. Run selection, rollout, and evaluation

From the repository root, the top-level runner performs all three stages:

```bash
python run_pipeline.py --nusc /path/to/nuscenes
```

It uses the released prior, adversary, and selector checkpoints; it does not train the prior or
PPO models. Rollouts default to 24 sequential shards. Run one shard with:

```bash
python run_pipeline.py --nusc /path/to/nuscenes --shard 1
```

A short smoke run can limit the number of scenes:

```bash
python run_pipeline.py --nusc /path/to/nuscenes --run cars_smoke --shard 1 --num-scenes 2
```

Use `--steps rollout evaluate` to omit selector validation, or `--selector trained` to use the
newly fitted selector from `work/selector/trees.npz` in the rollouts. The scripts in `steps/`
remain available for running individual stages.

## 5. Train models

```bash
bash infra/steps/train_prior.sh
bash infra/steps/train_ppo.sh
```

Use `WORKERS` to control the prior data-loader workers, and `PRIOR_ARGS`, `PPO_ARGS`, or
`ROLLOUT_ARGS` to pass additional command-line options to the corresponding entry point.
