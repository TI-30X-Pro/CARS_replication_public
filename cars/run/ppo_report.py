"""Summarise PPO training history and checkpoint weight changes.

    python -m cars.run.ppo_report --run $RUNS/ppo_g8 [--prior $CKPT/g8.ckpt]

Prints per-epoch metrics and compares the best and last checkpoints with the prior. Only the
denoiser, role embedding, and BatchNorm running statistics should change.
"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser(description="Summarise a PPO training run and its checkpoints.")
    ap.add_argument("--run", required=True, help="PPO run directory")
    ap.add_argument("--prior", default=os.path.join(os.environ.get("CKPT", "work/checkpoints"), "g8.ckpt"),
                    help="prior checkpoint used to measure weight changes")
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(os.path.join(a.run, "history.jsonl"))]
    keys = ["epoch", "train_reward", "collision_rate", "test_reward", "loss", "episodes", "steps", "seconds", "gpu_peak_gb"]
    print("  ".join(f"{k:>14s}" for k in keys))
    for r in rows:
        print("  ".join(f"{r[k]:14.4g}" if isinstance(r.get(k), float) else f"{str(r.get(k, '')):>14s}" for k in keys))
    secs = [r["seconds"] for r in rows if r.get("epoch", 0) > 0 and "seconds" in r]
    if secs:
        print(f"\n{sum(secs) / len(secs):.0f} s per epoch -> 500 epochs about {500 * sum(secs) / len(secs) / 3600:.0f} h")

    prior = torch.load(a.prior, map_location="cpu", weights_only=False)["state_dict"]
    for name in ("adversary_best.ckpt", "adversary_last.ckpt"):
        path = os.path.join(a.run, name)
        if not os.path.exists(path):
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        moved, passes = {}, None
        for k, v in sd.items():
            if ".encoder_heads." in k or k not in prior or k.count(".") < 3:   # skip the noise schedule
                continue
            if k.endswith("num_batches_tracked"):
                passes = int(v) - int(prior[k])
                continue
            part = k.split(".")[2]
            kind = "BN stats" if "running_" in k else "weights"
            d = (v.float() - prior[k].float()).abs().max().item()
            moved[(part, kind)] = max(moved.get((part, kind), 0.0), d)
        meta = {k: ck[k] for k in ("epoch", "test_reward") if k in ck}
        print(f"\n{name} {meta}; BatchNorm passes in train mode: {passes}")
        for (part, kind), d in sorted(moved.items()):
            print(f"  {part:18s} {kind:9s} max change {d:.2e}")


if __name__ == "__main__":
    main()
