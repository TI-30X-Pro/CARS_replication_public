"""Train the diffusion prior on ego trajectories.

    python -m cars.run.train_prior --nusc $NUSC --index $WORK/nusc_index.npz --out $RUNS/prior \
        [--max_steps 15000] [--workers 8] [--resume $RUNS/prior/latest.ckpt]

Uses the released architecture, batch size 64, AdamW, and mixed precision on GPU. The default
15,000-step run validates periodically and writes `history.jsonl`, `latest.ckpt`, and the three
best 5,000-step checkpoints.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from cars.data.maps import LOCATIONS, load_nusc_map, load_rasters, rasterize
from cars.data.nuscenes import NuscIndex, build_index
from cars.generation.data import PriorDataset
from cars.generation.gmdm import GMDM, GMDMConfig
from cars.generation.policy import to_device

REPO = Path(__file__).resolve().parents[2]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Checkpoints:
    """The best `keep` checkpoints by validation loss, plus `latest.ckpt`."""

    def __init__(self, out, keep=3):
        self.out, self.keep, self.saved = out, keep, []

    def save(self, net, step, epoch, val_loss, resume_state):
        """resume_state (optimizer, scaler) goes into latest.ckpt only."""
        ckpt = net.checkpoint(global_step=step, epoch=epoch, val_loss=val_loss)
        torch.save({**ckpt, **resume_state}, os.path.join(self.out, "latest.ckpt"))
        if val_loss is None:
            return
        path = os.path.join(self.out, f"iter{step}-val_loss{val_loss:.4f}.ckpt")
        torch.save(ckpt, path)
        self.saved.append((val_loss, path))
        self.saved.sort()
        for _, p in self.saved[self.keep:]:
            os.remove(p)
        self.saved = self.saved[:self.keep]


@torch.no_grad()
def validate(net, loader, batches, device, amp):
    net.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            losses.append(float(net.loss(to_device(batch, device))[0]))
    net.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser(description="Train the diffusion prior on nuScenes ego trajectories.")
    ap.add_argument("--nusc", required=True, help="unpacked nuScenes root")
    ap.add_argument("--index", help="cached nuScenes index (.npz)")
    ap.add_argument("--rasters", help="optional cached map rasters (.npz)")
    ap.add_argument("--out", required=True, help="training output directory")
    ap.add_argument("--config", default=str(REPO / "CARS_original" / "checkpoints" / "config.json"),
                    help="model architecture JSON")
    ap.add_argument("--max_steps", type=int, default=15000, help="optimizer steps")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5, help="AdamW learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--val_every", type=int, default=1500, help="training batches into each epoch")
    ap.add_argument("--val_batches", type=int, default=50, help="validation batches per check")
    ap.add_argument("--save_every", type=int, default=5000, help="checkpoint interval in steps")
    ap.add_argument("--workers", type=int, default=8, help="data-loader workers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="torch device")
    ap.add_argument("--resume", help="a latest.ckpt written by this script (continues with a fresh epoch)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seed_everything(a.seed)
    device = torch.device(a.device)
    amp = device.type == "cuda"

    idx = NuscIndex.load(a.index) if a.index and os.path.exists(a.index) else build_index(a.nusc)
    rasters = load_rasters(a.rasters) if a.rasters else \
        {loc: rasterize(load_nusc_map(a.nusc, loc), 1.0, loc) for loc in LOCATIONS}
    train_ds, valid_ds = PriorDataset(idx, rasters, "train"), PriorDataset(idx, rasters, "valid")
    loader = lambda ds, shuffle, workers: torch.utils.data.DataLoader(  # noqa: E731
        ds, batch_size=a.batch_size, shuffle=shuffle, drop_last=True, num_workers=workers,
        collate_fn=ds.collate_parsed, persistent_workers=workers > 0)
    train_loader = loader(train_ds, True, a.workers)
    valid_loader = loader(valid_ds, False, min(a.workers, 4))

    net = GMDM(GMDMConfig.from_json(a.config)).to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    step, epoch, last_val = 0, 0, None
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        prefix = "nets.policy."
        net.load_state_dict({k[len(prefix):]: v for k, v in ck["state_dict"].items()
                             if not k.startswith(prefix + "map_encoder.encoder_heads.")})
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, epoch, last_val = ck["global_step"], ck["epoch"], ck["val_loss"]
    ckpts = Checkpoints(a.out)
    log = open(os.path.join(a.out, "history.jsonl"), "a")
    print(f"train {len(train_ds)} samples, {len(train_loader)} batches per epoch; valid {len(valid_ds)}", flush=True)

    while step < a.max_steps:
        run = []
        for i, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss, h_pi, repel, eff_k = net.loss(batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            run.append([float(loss), float(h_pi), float(repel), float(eff_k)])
            if step % 10 == 0:
                m = np.mean(run, axis=0)
                row = {"step": step, "epoch": epoch, "loss": m[0], "H_pi": m[1], "repel": m[2], "effective_K": m[3]}
                log.write(json.dumps(row) + "\n")
                run = []
            if step % a.save_every == 0:
                ckpts.save(net, step, epoch, last_val, {"optimizer": opt.state_dict(), "scaler": scaler.state_dict()})
            if (i + 1) % a.val_every == 0:
                last_val = validate(net, valid_loader, a.val_batches, device, amp)
                row = {"step": step, "epoch": epoch, "val_loss": last_val}
                log.write(json.dumps(row) + "\n")
                print(row, flush=True)
            if step >= a.max_steps:
                break
        log.flush()
        epoch += 1


if __name__ == "__main__":
    main()
