# Stage 3 — Attribute collision responsibility

## Components

`cars/attribution/` contains:

- `drivers.py`: FSM, lane-aware CC-JP, and RSS reference drivers;
- `replay.py`: path-constrained replay and rollout-file reading;
- `metrics.py`: attribution rates, trigger breakdowns, criticality, and kinematics.

## Run the evaluation

Run attribution over one or more rollout files with:

```bash
python -m cars.run.evaluate \
  --hdf5 "$RUNS"/cars_released/shard*/data.hdf5 \
  --nusc "$NUSC" \
  --out "$RUNS/cars_released/eval" \
  --total_attempts 679
```

The command writes `replay_results_laneaware.csv`, `kinematics.csv`, and `metrics.json`.

## Attribution rule

A collision is attributed to the system under test when a reference driver, placed in the
target's position, avoids it while the adversary follows its logged trajectory. The target is
re-driven along its logged path, with speed reduced according to the selected driver. The first
collision ends the replay.

## Reported metrics

Reported metrics include the attribution rate for each driver, FSM criticality tiers and
`H_crit`, bootstrap confidence bounds, braking-demand statistics, TTC, collision speed and
angle, and the kinematic-infraction percentage.

Important implementation details include deterministic episode ordering for the bootstrap,
explicit endpoint handling for Savitzky-Golay smoothing, and historical percentile behaviour
for float32 inputs.

## Tests

Self-contained attribution tests are in `tests/test_attribution.py`.
