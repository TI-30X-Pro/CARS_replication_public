# Shared paths and Python interpreter. Environment variables override every default.

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# Data and output paths
WORK=${WORK:-$REPO/work}
NUSC=${NUSC:-$WORK/nuscenes}            # v1.0-trainval metadata + maps (with the v1.3 expansion)
CKPT=${CKPT:-$WORK/checkpoints}         # g8.ckpt, cars_adversary.ckpt, ctg.ckpt
RUNS=${RUNS:-$WORK/runs}

# Replication interpreter
CARS_PYTHON=${CARS_PYTHON:-python}

export REPO WORK NUSC CKPT RUNS CARS_PYTHON
