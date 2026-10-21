#!/bin/bash
# Prepare nuScenes and download the released checkpoints.
# Set NUSC to an unpacked dataset, or set NUSC_TAR to a tar containing
# nuscenes/v1.0-trainval and nuscenes/maps.
set -euo pipefail
source "$(dirname "$0")/../config.sh"

if [ ! -d "$NUSC/v1.0-trainval" ]; then
    : "${NUSC_TAR:?set NUSC to an unpacked nuScenes or NUSC_TAR to nuscenes.tar}"
    mkdir -p "$(dirname "$NUSC")"
    tar -C "$(dirname "$NUSC")" -xf "$NUSC_TAR"
fi

mkdir -p "$CKPT"
"$REPO/infra/run.sh" -c "
from huggingface_hub import snapshot_download
snapshot_download('Visier-1/cars-checkpoints', local_dir='$CKPT')"

ls -lh "$CKPT"
