# Stage 0 — Prepare data and simulation

## Runtime

The replication targets Python 3.11+, NumPy 2, and PyTorch 2. The nuScenes devkit is installed
by `infra/install_cars.sh`; data, checkpoints, and outputs live under `work/` by default.

## Build the nuScenes index and tracks

`cars/data/nuscenes.py` streams the nuScenes tables and builds a compact index:

```bash
python -m cars.data.nuscenes --root "$NUSC" --out "$WORK/nusc_index.npz"
```

`cars/data/tracks.py` builds per-scene agent tables. At 2 Hz it retains vehicles and humans
with at least two annotations, fills internal gaps linearly, constructs the ego track from the
front-camera ego poses, and derives velocity and acceleration by backward differences. The
tables are then interpolated to 10 Hz. Agents are ordered with ego first and the remaining
agents by first appearance.

## Build maps and model inputs

`cars/data/maps.py` rasterises the nuScenes map expansion at one pixel per metre with three
layers: drivable area, lane edges, and pedestrian areas.

`cars/data/batch.py` constructs the model input:

- 2 seconds of history and 2 seconds of future at 10 Hz;
- nearby vehicles within 50 metres, ordered by distance;
- a 224-pixel map crop oriented with the agent facing positive x;
- transforms and auxiliary future trajectories used by the policy.

## Simulate closed-loop scenes

`cars/sim/` contains the closed-loop scene, rollout loop, unicycle dynamics, collision tests,
episode logging, and HDF5 writer. Every five simulation steps, the policy produces a two-second
plan for each agent. Up to five points of that plan are applied before re-planning. Episodes stop
at the first detected target-adversary collision or after 120 steps.

Collision episodes are written to HDF5 with arrays shaped `(agents, steps, ...)`. The format is
consumed directly by `cars.run.evaluate`.

Tests for this stage are in `tests/test_tracks.py`, `tests/test_maps.py`,
`tests/test_dynamics.py`, and `tests/test_sim.py`.
