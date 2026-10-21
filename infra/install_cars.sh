#!/bin/bash
# Install all replication dependencies into a Python 3.11+ environment.
# Example: PIP=.venv/bin/pip bash infra/install_cars.sh
set -euo pipefail
PIP=${PIP:-pip}
HERE=$(cd "$(dirname "$0")" && pwd)

$PIP install -r "$HERE/requirements-cars.txt"

# nuscenes-devkit 1.2.0 declares numpy<2, but the scene-split and map APIs used here support
# NumPy 2. Install it without its legacy dependency pins so the environment remains compatible
# with Python 3.11+ and current matplotlib versions.
$PIP install --no-deps nuscenes-devkit==1.2.0
