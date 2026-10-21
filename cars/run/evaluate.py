"""Evaluate rollout files with responsibility attribution and kinematic metrics.

    python -m cars.run.evaluate --hdf5 run/shard*/data.hdf5 --nusc $NUSC --out run/eval \
        --total_attempts 679

Inputs: one or more rollout HDF5 files and a nuScenes root.
Outputs: replay_results_laneaware.csv, kinematics.csv, and metrics.json.
"""
import argparse
import csv
import json
import os

import h5py
import numpy as np
import pandas as pd

from cars.attribution.metrics import kinematics_file, per_tier, summary, triggers
from cars.attribution.replay import COLUMNS, replay_file

LOCATIONS = ("boston-seaport", "singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage")


def scene_locations(nusc_root):
    from cars.data.nuscenes import iter_records
    from pathlib import Path
    d = Path(nusc_root) / "v1.0-trainval"
    logs = {r["token"]: r["location"] for r in iter_records(d / "log.json")}
    return {r["name"]: logs[r["log_token"]] for r in iter_records(d / "scene.json")}


def main():
    ap = argparse.ArgumentParser(description="Evaluate rollout files with responsibility attribution.")
    ap.add_argument("--hdf5", nargs="+", required=True, help="rollout files (e.g. one per shard)")
    ap.add_argument("--nusc", required=True, help="nuScenes root (scene locations, map expansion)")
    ap.add_argument("--out", required=True, help="directory for CSV and JSON results")
    ap.add_argument("--total_attempts", type=int, default=0, help="number of (target, adversary) pairs tried")
    ap.add_argument("--bootstrap_b", type=int, default=10000, help="bootstrap samples for H_crit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from nuscenes.map_expansion.map_api import NuScenesMap
    locs = scene_locations(a.nusc)
    maps = {loc: NuScenesMap(dataroot=a.nusc, map_name=loc) for loc in LOCATIONS}

    rows, kin, n_col = [], [], 0
    for path in a.hdf5:
        rows += replay_file(path, locs, maps)
        kin.append(kinematics_file(path))
        with h5py.File(path, "r") as f:
            n_col += len(f.keys())
    # one sorted episode order over all files, as if they were merged first (the H_crit
    # bootstrap resamples by row, so the order matters)
    driver_order = {"FSM": 0, "CC-JP-LaneAware": 1, "RSS": 2}
    rows.sort(key=lambda r: (r["episode"], driver_order[r["controller"]]))
    with open(os.path.join(a.out, "replay_results_laneaware.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    kin = pd.concat(kin, ignore_index=True).sort_values("episode", ignore_index=True)
    kin.to_csv(os.path.join(a.out, "kinematics.csv"), index=False)

    df = pd.read_csv(os.path.join(a.out, "replay_results_laneaware.csv"))
    m = summary(df, n_col, a.total_attempts or n_col, bootstrap_b=a.bootstrap_b)
    m["IP%"] = float(kin["any_viol"].mean())
    m["per_tier"] = per_tier(df)
    m["triggers"] = triggers(df)
    with open(os.path.join(a.out, "metrics.json"), "w") as fp:
        json.dump(m, fp, indent=1, default=float)

    print(f"{m['N_collisions']} collisions of {m['N_total_attempts']} attempts")
    for k in ("FSM", "CCJP", "RSS"):
        print(f"  {k:5s} avoidable {100 * (1 - m[f'{k}_unprev%']):5.1f} %")
    print(f"  H_crit {m['H_crit_prev']:.3f} (95 % LCB {m['H_crit_LCB']:.3f})   BD+ {100 * m['BD_pos%']:.1f} %   "
          f"IP {100 * m['IP%']:.2f} %")


if __name__ == "__main__":
    main()
