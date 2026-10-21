#!/usr/bin/env python3
"""Run the CARS replication with the released trajectory-model checkpoints.

The default pipeline trains and checks the selector, runs all rollout shards, and
evaluates the resulting collision episodes. It does not train the diffusion prior
or the PPO adversary model.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent
ALL_STEPS = ("selector", "rollout", "evaluate")


def run_module(module: str, *arguments: object) -> None:
    """Run one pipeline stage in a fresh process with this repository importable."""
    command = [sys.executable, "-m", module, *(str(value) for value in arguments)]
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPO) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    environment.setdefault("WANDB_MODE", "offline")
    print(f"\n$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO, env=environment, check=True)


def existing_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing {description}: {path}")
    return path


def selector_path(value: str, work: Path) -> Path:
    """Resolve the released/trained selector aliases or a user-supplied path."""
    if value == "released":
        return REPO / "CARS_original" / "checkpoints" / "decision_tree_model.pkl"
    if value == "trained":
        return work / "selector" / "trees.npz"
    return Path(value).expanduser().resolve()


def selected_shards(shard: int | None, count: int) -> list[int]:
    if count < 1:
        raise SystemExit("--num-shards must be at least 1")
    if shard is not None and not 1 <= shard <= count:
        raise SystemExit(f"--shard must be between 1 and {count}")
    return [shard] if shard is not None else list(range(1, count + 1))


def attempted_pairs(run_dir: Path, shards: list[int]) -> int:
    """Sum completed pair counts so smoke and partial runs get the right denominator."""
    total = 0
    for shard in shards:
        summary = run_dir / f"shard{shard}" / "summary.json"
        if summary.exists():
            total += int(json.loads(summary.read_text())["pairs"])
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run selector validation, released-checkpoint rollouts, and evaluation. "
            "Prior and PPO training are not included."
        )
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=list(ALL_STEPS),
        metavar="STEP",
        help="stages to run in pipeline order (default: selector rollout evaluate)",
    )
    parser.add_argument("--work", help="data and output root (default: WORK or ./work)")
    parser.add_argument("--nusc", help="unpacked nuScenes root (default: NUSC or WORK/nuscenes)")
    parser.add_argument("--checkpoints", help="checkpoint directory (default: CKPT or WORK/checkpoints)")
    parser.add_argument("--runs", help="rollout root (default: RUNS or WORK/runs)")
    parser.add_argument("--run", default=os.environ.get("RUN", "cars_released"), help="run directory name")
    parser.add_argument(
        "--selector",
        default="released",
        help="'released' (default), 'trained', or a classifier path; trained uses WORK/selector/trees.npz",
    )
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("NSHARDS", "24")))
    parser.add_argument("--shard", type=int, help="run and evaluate one one-based shard instead of all shards")
    parser.add_argument("--num-scenes", type=int, help="limit each selected shard to its first N scenes")
    parser.add_argument("--device", help="PyTorch device for rollout, such as cuda, mps, or cpu")
    parser.add_argument("--seed", type=int, help="rollout seed (default: value in the rollout config)")
    parser.add_argument("--save-all", action="store_true", help="save non-collision episodes too")
    parser.add_argument(
        "--total-attempts",
        type=int,
        help="evaluation denominator (default: sum of completed pairs in the selected shards)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000, help="H_crit bootstrap samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = set(args.steps)
    work = Path(args.work or os.environ.get("WORK", REPO / "work")).expanduser().resolve()
    nusc = Path(args.nusc or os.environ.get("NUSC", work / "nuscenes")).expanduser().resolve()
    checkpoints = Path(args.checkpoints or os.environ.get("CKPT", work / "checkpoints")).expanduser().resolve()
    runs = Path(args.runs or os.environ.get("RUNS", work / "runs")).expanduser().resolve()
    run_dir = runs / args.run
    shards = selected_shards(args.shard, args.num_shards)

    existing_path(nusc / "v1.0-trainval", "nuScenes trainval metadata")
    index = work / "nusc_index.npz"

    if "selector" in steps:
        existing_path(
            REPO / "CARS_original" / "checkpoints" / "decision_tree_model.pkl",
            "released selector",
        )
        run_module(
            "cars.run.train_selector",
            "--nusc",
            nusc,
            "--index",
            index,
            "--out",
            work / "selector",
        )

    if "rollout" in steps:
        existing_path(checkpoints / "g8.ckpt", "released prior checkpoint")
        existing_path(checkpoints / "cars_adversary.ckpt", "released adversary checkpoint")
        selected_model = existing_path(selector_path(args.selector, work), f"{args.selector} selector")
        for shard in shards:
            output = run_dir / f"shard{shard}"
            done = output / "done"
            if done.exists():
                print(f"Skipping completed shard {shard}: {output}", flush=True)
                continue
            command: list[object] = [
                "--nusc", nusc,
                "--index", index,
                "--ckpt", checkpoints,
                "--selector", selected_model,
                "--out", output,
                "--shard", shard,
                "--num_shards", args.num_shards,
            ]
            if args.num_scenes is not None:
                command += ["--num_scenes", args.num_scenes]
            if args.device:
                command += ["--device", args.device]
            if args.seed is not None:
                command += ["--seed", args.seed]
            if args.save_all:
                command.append("--save_all")
            run_module("cars.run.rollout", *command)
            done.touch()

    if "evaluate" in steps:
        hdf5_files = [run_dir / f"shard{shard}" / "data.hdf5" for shard in shards]
        hdf5_files = [path for path in hdf5_files if path.exists()]
        if not hdf5_files:
            raise SystemExit(f"No rollout files found under {run_dir}")
        total = args.total_attempts if args.total_attempts is not None else attempted_pairs(run_dir, shards)
        command = [
            "--hdf5", *hdf5_files,
            "--nusc", nusc,
            "--out", run_dir / "eval",
            "--bootstrap_b", args.bootstrap_samples,
        ]
        if total:
            command += ["--total_attempts", total]
        run_module("cars.run.evaluate", *command)

    print(f"\nPipeline complete: {run_dir}")


if __name__ == "__main__":
    main()
