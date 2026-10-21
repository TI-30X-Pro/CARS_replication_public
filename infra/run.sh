#!/bin/bash
# Run Python from the repository root with the replication on PYTHONPATH.
# Usage: infra/run.sh -m pytest tests
set -euo pipefail
source "$(dirname "$0")/config.sh"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=${WANDB_MODE:-offline}
exec "$CARS_PYTHON" "$@"
