"""Run closed-loop scenarios for the configured target-adversary pairs.

    python -m cars.run.rollout --nusc $NUSC --index $WORK/nusc_index.npz --ckpt $CKPT \
        --out $RUNS/cars/shard1 --shard 1 --num_shards 24 [--device cuda] [--num_scenes 2]

The target and background agents use the prior; the selected adversary uses the adversarial
checkpoint. Each agent samples eight plans and executes the one closest to its logged future.
Collision episodes are written to `<out>/data.hdf5`; every run also writes `summary.json`.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from cars.data.tracks import scene_tracks
from cars.generation.gmdm import GMDM
from cars.generation.policy import Planner, TargetAdvOthers
from cars.run.train_selector import Observer
from cars.selection.dataset import load_pairs
from cars.selection.model import SHIPPED, Trees
from cars.selection.selector import TreeSelector
from cars.sim.rollout import run_episode, write_episode
from cars.sim.scene import SimScene

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "rollout_nusc.json"


def shard(items, k, n):
    """Contiguous items assigned to one-based shard k of n."""
    m = len(items)
    return items[(m * (k - 1)) // n:(m * k) // n]


def main():
    ap = argparse.ArgumentParser(description="Run closed-loop CARS scenarios.")
    ap.add_argument("--nusc", required=True, help="unpacked nuScenes root")
    ap.add_argument("--index", help="cached nuScenes index (.npz); built if missing")
    ap.add_argument("--rasters", help="optional cached map rasters (.npz)")
    ap.add_argument("--ckpt", default=str(REPO / "work" / "checkpoints"), help="checkpoint directory")
    ap.add_argument("--prior", default="g8.ckpt", help="model of the target and the other agents")
    ap.add_argument("--adversary", default="cars_adversary.ckpt", help="adversarial-policy checkpoint")
    ap.add_argument("--selector", default=str(SHIPPED), help="pickled classifier, or trees .npz")
    ap.add_argument("--out", required=True, help="directory for data.hdf5 and summary.json")
    ap.add_argument("--shard", type=int, help="one-based shard to run")
    ap.add_argument("--num_shards", type=int, help="total number of shards")
    ap.add_argument("--num_scenes", type=int, help="limit the selected shard to its first N scenes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="torch device")
    ap.add_argument("--save_all", action="store_true", help="also keep episodes without a collision")
    ap.add_argument("--seed", type=int, help="instead of the config's 42, e.g. to measure run-to-run spread")
    a = ap.parse_args()
    cfg = json.load(open(CONFIG))
    seed = cfg["seed"] if a.seed is None else a.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(a.out, exist_ok=True)
    h5 = os.path.join(a.out, "data.hdf5")

    obs = Observer(a.nusc, a.index, a.rasters)
    pairs = load_pairs(REPO / "configs" / "labels_nusc.json")
    scenes = cfg["scenes"]
    if a.shard is not None:
        scenes = shard(scenes, a.shard, a.num_shards)
    if a.num_scenes is not None:
        scenes = scenes[:a.num_scenes]

    load = lambda name: GMDM.from_checkpoint(os.path.join(a.ckpt, name), device=a.device)  # noqa: E731
    k = cfg["num_samples"]
    target, adversary, others = Planner(load(a.prior), k), Planner(load(a.adversary), k), Planner(load(a.prior), k)
    trees = Trees.load(a.selector) if a.selector.endswith(".npz") else Trees.from_pickle(a.selector)

    summary = {"pairs": 0, "collisions": 0, "skipped": [], "episodes": []}
    for scene in scenes:
        tr = scene_tracks(obs.idx, obs.names.index(scene))
        for pair_idx, (target_name, _) in enumerate(pairs[scene]):
            t0 = time.time()
            sim = SimScene(tr, obs.rasters[tr.location], start=cfg["start"])
            if target_name not in sim.agents:
                summary["skipped"].append([scene, target_name])
                continue
            policy = TargetAdvOthers(target, adversary, others, TreeSelector(trees, cfg["confirm_k"]), target_name)
            out = run_episode(sim, policy, target_name, n_step=cfg["n_step"], horizon=cfg["horizon"],
                              log=True, scene_index=obs.scene_index[scene])
            hit = bool(out["had_collision"])
            if hit or a.save_all:
                write_episode(h5, scene, cfg["start"], out["target_idx"], out["buffer"], pair_idx)
            summary["pairs"] += 1
            summary["collisions"] += hit
            summary["episodes"].append({"scene": scene, "pair": pair_idx, "target": out["target_idx"],
                                        "collision": hit, "steps": out["steps"], "agents": len(sim.agents),
                                        "seconds": round(time.time() - t0, 1)})
            print(f"{scene} pair {pair_idx + 1}/{len(pairs[scene])}: {'collision' if hit else 'none'} after "
                  f"{out['steps']} steps, {len(sim.agents)} agents, {time.time() - t0:.0f} s", flush=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"{summary['collisions']} collisions in {summary['pairs']} pairs")


if __name__ == "__main__":
    main()
