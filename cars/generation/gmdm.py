"""Gaussian-mixture diffusion model for multi-plan trajectory generation.

Actions are 20 steps of (acceleration, yaw rate), normalised by the action statistics, and
turned into states with the unicycle model (cars.sim.dynamics). With R = 100 diffusion steps
and a cosine schedule:

- Denoiser: for noisy actions x_r, a K-component Gaussian mixture over the clean actions x_0,
  per step: weights softmax(logits), means mu_k, scales (softplus + 0.1) * sigma_gmm_scale.
  Its time input is log sigma_r scaled to [0, 1], with sigma_r^2 = (1 - abar_r) / abar_r.
- Training loss: draw log sigma uniformly between its smallest and largest value, noise x_0,
  and take NLL(x_0) - 0.15 H(pi) + 0.03 repulsion between component means (scale 0.5),
  averaged over the available future steps.
- Sampling: 20 DDIM steps (eta 0.3) from r = 99 down to 0. Each step maps every component
  through the DDIM update, so x_{r'} follows a mixture with means A mu_k + C x_r and variances
  A^2 sigma_k^2 + s^2. Agents whose logged future never exceeds
  2 m/s get zero actions after every step (the stationary mask).

Random draws are isolated behind a `Noise` object for reproducible sampling.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cars.generation.nets import (MLP, AgentsMHA, Denoiser, EgoEncoder, HistoryEncoder, MapEncoder,
                                  NeighborHistoryEncoder, query_feature_grid)
from cars.sim.dynamics import UnicycleBounds, state_to_action, unicycle_forward
from cars.sim.geometry import transform_points

CONFIG = Path(__file__).resolve().parents[2] / "CARS_original" / "checkpoints" / "config.json"


@dataclass
class GMDMConfig:
    grid_feature_dim: int = 32
    context_encoder_out_dim: int = 128
    time_emb_dim: int = 32
    diffusion_hidden_dim: int = 64
    dilations: tuple = (1, 2, 4, 8)
    num_heads: int = 4
    num_Gaussian: int = 8
    n_diffusion_steps: int = 100
    ddim_steps: int = 20
    dt: float = 0.1
    sigma_gmm_scale: float = 3.0
    action_norm_info_center: tuple = ((-0.006135, 0.000198), (2.16333, 0.056355))
    norm_info_center: tuple = ()
    norm_info_neighbor: tuple = ()
    norm_info_ego: tuple = ()
    dynamics: UnicycleBounds = field(default_factory=UnicycleBounds)
    image_shape: tuple = (3, 224, 224)

    @classmethod
    def from_json(cls, path=CONFIG):
        algo = json.load(open(path))["algo"]
        names = {f for f in cls.__dataclass_fields__} - {"dynamics", "image_shape"}
        kw = {k: algo[k] for k in names if k in algo}
        d = algo["Dynamics"]
        kw["dynamics"] = UnicycleBounds(d["max_steer"], d["max_yawvel"], tuple(d["acce_bound"]), tuple(d["vbound"]))
        return cls(**kw)


def cosine_betas(n, s=0.008):
    x = np.linspace(0, n + 1, n + 1)
    abar = np.cos(((x / (n + 1)) + s) / (1 + s) * np.pi * 0.5) ** 2
    abar = abar / abar[0]
    return torch.tensor(np.clip(1 - abar[1:] / abar[:-1], 0, 0.999), dtype=torch.float32)


def ddim_timesteps(ddim_steps, n):
    t = torch.linspace(n - 1, 0, ddim_steps).long().tolist()
    return t, t[1:] + [0]


class Noise:
    """Fresh random draws: initial noise and, per DDIM step, a component index per action
    step and standard normal noise for every component (like MixtureSameFamily.sample)."""

    def __init__(self, generator=None):
        self.g = generator

    def initial(self, shape, device):
        return torch.randn(shape, generator=self.g, device=device)

    def step(self, logits, shape, device):
        k = torch.multinomial(torch.softmax(logits, -1).reshape(-1, logits.shape[-1]), 1, generator=self.g)
        return k.view(logits.shape[:-1]), torch.randn(shape, generator=self.g, device=device)


class GMDM(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg = GMDMConfig.from_json() if cfg is None else cfg
        D, G = cfg.context_encoder_out_dim, cfg.grid_feature_dim
        self.map_encoder = MapEncoder(G, cfg.image_shape[0])
        self.center_hist = HistoryEncoder(D, cfg.norm_info_center)
        self.ego_encoder = EgoEncoder(D, cfg.norm_info_ego)
        self.neighbor_hist = NeighborHistoryEncoder(D, cfg.norm_info_neighbor)
        self.process_cond_mlp = MLP(D + G + 2, D, [D, D])
        self.neigh_role_emb = nn.Embedding(2, D)
        nn.init.zeros_(self.neigh_role_emb.weight)
        self.agents_mha = AgentsMHA(D, cfg.num_heads)
        self.model = Denoiser(2, D, cfg.time_emb_dim, cfg.diffusion_hidden_dim, 2, cfg.dilations, cfg.num_heads,
                              grid_map_traj_dim=G, mix_gauss=cfg.num_Gaussian)
        self.add = torch.tensor(cfg.action_norm_info_center[0], dtype=torch.float32)
        self.div = torch.tensor(cfg.action_norm_info_center[1], dtype=torch.float32)

        betas = cosine_betas(cfg.n_diffusion_steps)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        abar_prev = torch.cat([torch.ones(1), abar[:-1]])
        post_var = betas * (1.0 - abar_prev) / (1.0 - abar)
        log_sigma = 0.5 * torch.log((1.0 - abar) / abar)
        for name, v in (("betas", betas), ("alphas_cumprod", abar), ("alphas_cumprod_prev", abar_prev),
                        ("sqrt_alphas_cumprod", torch.sqrt(abar)), ("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - abar)),
                        ("log_one_minus_alphas_cumprod", torch.log(1.0 - abar)),
                        ("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / abar)),
                        ("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / abar - 1)),
                        ("posterior_variance", post_var),
                        ("posterior_log_variance_clipped", torch.log(torch.clamp(post_var, min=1e-20))),
                        ("posterior_mean_coef1", betas * torch.sqrt(abar_prev) / (1.0 - abar)),
                        ("posterior_mean_coef2", (1.0 - abar_prev) * torch.sqrt(alphas) / (1.0 - abar)),
                        ("log_sigma_table", log_sigma), ("log_sigma_min", torch.min(log_sigma)),
                        ("log_sigma_max", torch.max(log_sigma))):
            self.register_buffer(name, v)

    # Checkpoint I/O
    @classmethod
    def from_checkpoint(cls, path, config=CONFIG, device="cpu"):
        """A released or trained Lightning checkpoint (keys nets.policy.*), in eval mode."""
        net = cls(GMDMConfig.from_json(config))
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        prefix = "nets.policy."
        sd = {k[len(prefix):]: v for k, v in sd.items()
              if k.startswith(prefix) and not k.startswith(prefix + "map_encoder.encoder_heads.")}
        net.load_state_dict(sd, strict=True)
        return net.to(device).eval()

    def checkpoint(self, **meta):
        """Return a Lightning-style checkpoint compatible with released-model tooling."""
        return {"state_dict": self.lightning_state_dict(), "pytorch-lightning_version": "2.0.0", **meta}

    def lightning_state_dict(self):
        """Return weights in the released `nets.policy.*` checkpoint layout."""
        sd = {f"nets.policy.{k}": v.detach().cpu() for k, v in self.state_dict().items()}
        for k in list(sd):
            if ".map_encoder.backbone." in k:
                sd[k.replace(".map_encoder.backbone.", ".map_encoder.encoder_heads.")] = sd[k]
        return sd

    # Model components
    def scale(self, a):
        return (a - self.add.to(a.device)) / self.div.to(a.device)

    def descale(self, x):
        return x * self.div.to(x.device) + self.add.to(x.device)

    @staticmethod
    def current_states(batch):
        """x, y, signed speed, yaw at the current step, in the agent frame."""
        s = torch.zeros(*batch["curr_speed"].shape, 4, device=batch["curr_speed"].device)
        s[..., :2] = batch["history_positions"][..., -1, :]
        s[..., 2] = batch["curr_speed_signed"] if "curr_speed_signed" in batch else batch["curr_speed"]
        s[..., 3] = batch["history_yaws"][..., -1, 0]
        return s

    def states(self, x, curr):
        """Normalised actions (B, T, 2) -> (B, T, 6): x, y, v, yaw, a, yaw rate."""
        a = self.descale(x)
        return torch.cat([unicycle_forward(curr, a, self.cfg.dt, self.cfg.dynamics), a], dim=-1)

    def map_along(self, pos, grid, raster_from_agent):
        """Map features at agent-frame positions (B, T, 2), on the grid's coarser pixels."""
        xy = transform_points(pos, raster_from_agent)
        _, H, W = self.cfg.image_shape
        xy[..., 0] = xy[..., 0] * (grid.shape[-1] / W)
        xy[..., 1] = xy[..., 1] * (grid.shape[-2] / H)
        return query_feature_grid(xy, grid).view(*pos.shape[:2], -1)

    def context(self, batch):
        global_feat, grid = self.map_encoder(batch["maps"])
        B, device = global_feat.size(0), global_feat.device
        center = self.center_hist(batch["history_positions"], batch["history_speeds"],
                                  batch["history_yaws"].squeeze(-1), batch["extent"][..., :2],
                                  batch["history_availabilities"])
        neigh, neigh_mask = self.neighbor_hist(
            batch["all_other_agents_history_positions"], batch["all_other_agents_history_speeds"],
            batch["all_other_agents_history_yaws"].squeeze(-1), batch["all_other_agents_history_extents"][:, :, 0, :2],
            batch["all_other_agents_history_availability"])
        role = torch.zeros(neigh.shape[:2], dtype=torch.long, device=device)
        if "target_idx" in batch:   # the adversary sees which neighbour is its target
            is_target = batch["neigh_indices"].long() == batch["target_idx"].long().unsqueeze(1)
            valid = is_target.sum(dim=1) == 1
            role[valid] = is_target[valid].long()
        center = self.agents_mha(center, neigh + self.neigh_role_emb(role), neigh_mask)
        sg = batch["map_is_singapore"].to(device).long()
        city = torch.stack([1 - sg, sg], dim=1).float()
        return self.process_cond_mlp(torch.cat([center, global_feat, city], dim=-1)), grid

    def time_input(self, log_sigma):
        return (log_sigma - self.log_sigma_min) / (self.log_sigma_max - self.log_sigma_min)

    def denoise(self, x, cond, grid, t_float, curr, raster_from_agent):
        """Mixture over clean actions given noisy x: logits, means, scales (with sigma_gmm_scale)."""
        traj = self.map_along(self.states(x, curr)[..., :2], grid, raster_from_agent)
        logits, mu, log_sigma = self.model(x, cond, t_float, grid, traj)
        return logits, mu, (0.1 + F.softplus(log_sigma)) * self.cfg.sigma_gmm_scale

    # Training objective
    def target_actions(self, batch):
        traj = torch.cat((batch["target_positions"][:, :20], batch["target_yaws"][:, :20]), dim=2)
        return state_to_action(traj, batch["curr_speed"], self.cfg.dt)[..., -2:]

    def loss(self, batch, u=None, noise=None):
        """Returns loss, mixture entropy H(pi), repulsion, effective number of components."""
        cond, grid = self.context(batch)
        x0 = self.scale(self.target_actions(batch))
        B = x0.shape[0]
        lo, hi = self.log_sigma_min, self.log_sigma_max
        u = torch.rand(B, device=x0.device) if u is None else u
        log_sig = lo + u * (hi - lo)
        abar = 1.0 / (1.0 + torch.exp(log_sig) ** 2)
        noise = torch.randn_like(x0) if noise is None else noise
        x_t = abar.sqrt().view(B, 1, 1) * x0 + (1.0 - abar).clamp_min(1e-8).sqrt().view(B, 1, 1) * noise

        logits, mu, sigma = self.denoise(x_t, cond, grid, (log_sig - lo) / (hi - lo),
                                         self.current_states(batch), batch["raster_from_agent"])
        gmm = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logits),
                                                    torch.distributions.Independent(torch.distributions.Normal(mu, sigma), 1))
        avail = batch["target_availabilities"].to(x0.device).float()
        denom = avail.sum().clamp_min(1.0)
        nll = -(gmm.log_prob(x0) * avail).sum() / denom

        pi = F.softmax(logits, dim=-1)
        with torch.no_grad():
            pi_mean = pi.mean(dim=(0, 1))
            eff_k = torch.exp(-(pi_mean * pi_mean.clamp_min(1e-12).log()).sum())
        h_pi = (-(pi * pi.clamp_min(1e-12).log()).sum(-1) * avail).sum() / denom
        K = mu.size(2)
        d2 = ((mu.unsqueeze(2) - mu.unsqueeze(3)) ** 2).sum(-1)
        offdiag = 1.0 - torch.eye(K, device=mu.device).view(1, 1, K, K)
        repel = ((torch.exp(-d2 / (0.5 ** 2 + 1e-8)) * offdiag).mean(dim=(-1, -2)) * avail).sum() / denom
        return nll - 0.15 * h_pi + 0.03 * repel, h_pi, repel, eff_k

    # DDIM sampling
    def ddim_mixture(self, x_t, mu, sigma, t, t_next, eta=0.3):
        """Per-component mean and std of x at t_next (B, T, K, 2), as mix_dist_continuous."""
        n, dev = self.cfg.n_diffusion_steps, self.alphas_cumprod.device
        a_t = self.alphas_cumprod[torch.as_tensor(t, device=dev).clamp(0, n - 1)].clamp(1e-4, 0.9999)
        a_n = self.alphas_cumprod[torch.as_tensor(t_next, device=dev).clamp(0, n - 1)].clamp(1e-4, 0.9999)
        B, T, _ = x_t.shape
        a_t = a_t.view(-1, 1, 1).expand(B, T, 1).repeat_interleave(2, dim=-1)
        a_n = a_n.view(-1, 1, 1).expand(B, T, 1).repeat_interleave(2, dim=-1)
        ratio = (a_t / a_n).clamp(1e-4, 1e4)
        s = eta * torch.sqrt(((1 - ratio) * (1 - a_n) / (1 - a_t).clamp_min(1e-4)).clamp_min(1e-8))
        s1 = torch.sqrt(a_n)
        s2 = torch.sqrt((1 - a_n - s ** 2).clamp_min(1e-8))
        s3 = torch.sqrt(a_t)
        s4 = torch.sqrt((1 - a_t).clamp_min(1e-4))
        A = s1 - s2 * (s3 / s4)
        C = (s2 / s4) * x_t
        mean = A.unsqueeze(2) * mu + C.unsqueeze(2)
        std = torch.sqrt(((A.unsqueeze(2) ** 2) * sigma ** 2 + s.unsqueeze(2) ** 2).clamp_min(1e-4))
        return mean, std

    @torch.no_grad()
    def sample(self, batch, num_samples=1, noise=None, stationary=None, eta=0.3):
        """Plans (B, num_samples, T, ...): positions (x, y) and yaws in the agent frame, the
        final normalised actions, and the whole chain x_T .. x_0 (B, num_samples, 21, T, 2).
        stationary: (B,) bool, agents whose actions are set to zero after every step."""
        noise = Noise() if noise is None else noise
        cond, grid = self.context(batch)
        B, T, _ = batch["target_positions"].shape
        curr = self.current_states(batch)
        rep = lambda v: v.repeat_interleave(num_samples, dim=0)  # noqa: E731
        cond, grid, curr, rfa = rep(cond), rep(grid), rep(curr), rep(batch["raster_from_agent"])
        BN = B * num_samples
        x = noise.initial((B, num_samples, T, 2), curr.device).reshape(BN, T, 2)
        still = rep(stationary) if stationary is not None else None
        chain = [x]

        steps, nexts = ddim_timesteps(self.cfg.ddim_steps, self.cfg.n_diffusion_steps)
        for t, t_next in zip(steps, nexts):
            t_vec = torch.full((BN,), t, dtype=torch.long, device=x.device)
            t_float = self.time_input(self.log_sigma_table[t_vec])
            logits, mu, sigma = self.denoise(x, cond, grid, t_float, curr, rfa)
            mean, std = self.ddim_mixture(x, mu, sigma, t_vec, torch.full_like(t_vec, t_next), eta)
            k, eps = noise.step(logits, mean.shape, x.device)
            idx = k.long()[..., None, None].expand(BN, T, 1, 2)
            x = (mean + std * eps).gather(2, idx).squeeze(2)
            if still is not None and still.any():
                x[still] = self.scale(torch.zeros_like(x[still]))
            chain.append(x)

        traj = self.states(x, curr).view(B, num_samples, T, -1)
        return {"positions": traj[..., :2], "yaws": traj[..., 3:4], "actions": x.view(B, num_samples, T, 2),
                "chain": torch.stack(chain, dim=1).view(B, num_samples, len(chain), T, 2)}
