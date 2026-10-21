"""Fine-tune the adversarial diffusion policy with PPO.

    python -m cars.run.train_ppo --nusc $NUSC --index $WORK/nusc_index.npz --out $RUNS/ppo \
        [--prior g8.ckpt] [--epochs 500] [--seed 42]

Starts from `g8.ckpt` by default. Writes `adversary_best.ckpt`, `adversary_last.ckpt`, and
`history.jsonl`.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from cars.generation.gmdm import GMDM
from cars.generation.ppo import DiffusionPPO, PPOConfig
from cars.generation.ppo_env import AdversaryEnv, train
from cars.run.train_selector import Observer
from cars.selection.dataset import START, load_pairs
from cars.sim.scene import SimScene
from cars.data.tracks import scene_tracks

REPO = Path(__file__).resolve().parents[2]


def training_scenes(first=0, last=20, config=REPO / "configs" / "rollout_nusc.json"):
    """The configured rollout scenes in the half-open interval first..last."""
    return json.load(open(config))["scenes"][first:last]


def main():
    ap = argparse.ArgumentParser(description="Fine-tune the adversarial diffusion policy with PPO.")
    ap.add_argument("--nusc", required=True, help="unpacked nuScenes root")
    ap.add_argument("--index", help="cached nuScenes index (.npz)")
    ap.add_argument("--rasters", help="optional cached map rasters (.npz)")
    ap.add_argument("--ckpt", default=str(REPO / "work" / "checkpoints"), help="checkpoint directory")
    ap.add_argument("--prior", default="g8.ckpt", help="start of the adversary (and the target / others model)")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--epochs", type=int, default=500, help="training epochs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenes", type=int, nargs=2, default=(0, 20), help="half-open scene-index range")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="torch device")
    a = ap.parse_args()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    obs = Observer(a.nusc, a.index, a.rasters)
    pairs = load_pairs(REPO / "configs" / "labels_nusc.json")
    scenes = training_scenes(*a.scenes)

    def make_scene(name):
        tr = scene_tracks(obs.idx, obs.names.index(name))
        return SimScene(tr, obs.rasters[tr.location], start=START)

    load = lambda name: GMDM.from_checkpoint(os.path.join(a.ckpt, name), device=a.device)  # noqa: E731
    adversary, target, others = load(a.prior), load("g8.ckpt"), load("g8.ckpt")
    ppo = DiffusionPPO(adversary, PPOConfig())

    def env():
        return AdversaryEnv(scenes, pairs, make_scene, adversary, target, others)

    os.makedirs(a.out, exist_ok=True)
    log_path = open(os.path.join(a.out, "history.jsonl"), "a")

    def log(row):
        print(row, flush=True)
        log_path.write(json.dumps(row) + "\n")
        log_path.flush()

    train(ppo, env(), env(), a.out, epochs=a.epochs, log=log)


if __name__ == "__main__":
    main()
