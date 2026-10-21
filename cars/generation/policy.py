"""Wrap diffusion models as target, adversary, and background-agent policies.

`Planner` wraps one model for one role:
- every call draws 8 plans per agent and executes the one closest to the logged future
  (mean distance over the available steps, `choose_action_from_gt`); if an agent has no
  logged future step at all, its first plan;
- the stationary mask (agents whose logged future never exceeds 2 m/s) is computed at the
  episode's first call and kept by position for the rest of the episode. If the adversary
  switches, the "others" batch changes but the mask does not.

`TargetAdvOthers` is the rollout policy for run_episode: the selector picks the adversary, the
target and the other agents follow the prior (g8, two separate copies with their own masks), the
adversary follows the adversarial model (cars_adversary) and is told which neighbour is the
target.
"""
import numpy as np
import torch

from cars.generation.gmdm import Noise


def stationary_mask(batch, speed_th=2.0):
    """The original's "any_speed" rule on the logged full future (extras)."""
    speed = batch["extras"]["full_fut_traj"][..., 2]
    valid = batch["extras"]["full_fut_valid"]
    return ~(((speed > speed_th).float() * valid).sum(dim=-1) > 0)


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch


def closest_to_log(positions, batch):
    """Index of the plan (B, N, T, 2) with the smallest mean distance to the logged future."""
    B, N, T, _ = positions.shape
    idx = torch.zeros(B, dtype=torch.long, device=positions.device)
    end = min(T, batch["target_positions"].size(1))
    err = torch.norm(positions[:, :, :end] - batch["target_positions"][:, :end].unsqueeze(1), dim=-1)
    err[~batch["target_availabilities"][:, :end].unsqueeze(1).expand(B, N, end)] = torch.nan
    ade = torch.nanmean(err, dim=-1)
    ok = torch.isnan(ade).sum(dim=-1) == 0
    if ok.any():
        idx[ok] = torch.argmin(ade, dim=-1)[ok]
    return idx


class Planner:
    """Runs on the model's device; takes and returns CPU tensors."""

    def __init__(self, net, num_samples=8, noise=None):
        self.net, self.num_samples = net, num_samples
        self.noise = Noise() if noise is None else noise
        self.mask = None

    def __call__(self, batch, step):
        batch = to_device(batch, next(self.net.parameters()).device)
        if step == 0:
            self.mask = stationary_mask(batch)
        out = self.net.sample(batch, self.num_samples, self.noise, stationary=self.mask)
        best = closest_to_log(out["positions"], batch)
        rows = torch.arange(len(best), device=best.device)
        out = {k: v.cpu() for k, v in out.items()}
        return out["positions"][rows.cpu(), best.cpu()], out["yaws"][rows.cpu(), best.cpu()], out


def take(batch, idx):
    """Rows idx of every tensor / list in a (nested) batch."""
    if isinstance(batch, torch.Tensor):
        return batch.index_select(0, idx)
    if isinstance(batch, dict):
        return {k: take(v, idx) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [batch[i] for i in idx.tolist()]
    return batch


class TargetAdvOthers:
    """policy(obs, step) -> plans of all agents, for cars.sim.rollout.run_episode."""

    def __init__(self, target, adversary, others, selector, target_name):
        self.target, self.adversary, self.others = target, adversary, others
        self.selector, self.target_name = selector, target_name

    def __call__(self, obs, step):
        if step == 0:
            self.selector.reset()
        names = list(obs["agent_name"])
        t = names.index(self.target_name)
        adv, sel_info = self.selector(obs, t)
        M = len(names)
        i_t, i_a = torch.tensor([t]), torch.tensor([adv])
        keep = torch.ones(M, dtype=torch.bool)
        keep[[t, adv]] = False
        i_o = torch.arange(M)[keep]

        adv_batch = take(obs, i_a)
        adv_batch["target_idx"] = i_t
        parts = [(i_t, self.target(take(obs, i_t), step)), (i_a, self.adversary(adv_batch, step))]
        if len(i_o):
            parts.append((i_o, self.others(take(obs, i_o), step)))

        T = min(p[1][0].shape[1] for p in parts)
        xy = torch.empty(M, T, 2)
        yaw = torch.empty(M, T, 1)
        samples = torch.zeros(M, parts[0][1][2]["positions"].shape[1], T, 2)
        sample_yaws = torch.zeros(M, samples.shape[1], T, 1)
        for idx, (pos, yw, out) in parts:
            xy[idx], yaw[idx] = pos[:, :T], yw[:, :T]
            samples[idx], sample_yaws[idx] = out["positions"][:, :, :T], out["yaws"][:, :, :T]
        info = {"action_samples": {"positions": samples.numpy(), "yaws": sample_yaws.numpy()},
                "adv_idx": np.full(M, adv, dtype=np.int32)}
        return dict(xy=xy.numpy(), yaw=yaw.numpy(), adv_idx=adv, info=info, selection=sel_info)
