"""Fine-tune the adversarial diffusion policy with PPO (paper Eq. 9-10).

The action of one 0.5 s step is the adversary's whole DDIM chain x_T, ..., x_0 (20 steps);
only x_0 is executed. Its log-probability is per diffusion step, log p(x_{k+1} | x_k, context),
under the same mixture the sampler draws from. Settings (simulation_config.json "ppo" and the
script):

- trainable: the denoiser and the neighbour-role embedding; everything else frozen. The model
  runs in train mode, so the map encoder's BatchNorm uses batch statistics and updates its
  running statistics on every forward pass. sigma_gmm_scale is 10 (3 at rollout).
- one update per collect of 5 episodes: values of obs and next obs (chunks of 256, no grad),
  GAE (gamma 0.99, lambda 0.95; bootstrap cut on collisions, not on timeouts), old
  log-probabilities on the whole batch at once, then 2 passes over shuffled minibatches of 16
  (the last short one merged): advantages normalised per minibatch, ratio clipped per
  diffusion step (0.2), summed over steps, plus 0.5 x value MSE; gradient norm 0.5; Adam 2e-5.
- the optimiser gets the denoiser's parameters twice (the critic holds the model too), so
  Adam updates them twice per step. Kept on purpose.
- the critic: frozen context (cond, grid mean) and the current state, 512-wide MLP with two
  residual blocks.
"""
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cars.generation.gmdm import ddim_timesteps
from cars.generation.policy import to_device


@dataclass
class PPOConfig:
    lr: float = 2e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    batch_size: int = 16
    repeat: int = 2
    max_batchsize: int = 256
    eta: float = 0.3
    sigma_gmm_scale: float = 10.0
    detach_context: bool = False
    logp_chunk: int = 640          # rows (sample x diffusion step) per denoiser pass for the old log-probs


class ValueCritic(nn.Module):
    """Estimate V(obs) from frozen diffusion context with checkpoint-compatible names."""

    def __init__(self, net, hidden=512):
        super().__init__()
        self.ppo_diffuser = net
        self.map_pool = nn.AdaptiveAvgPool2d(1)
        self.input_proj = nn.LazyLinear(hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.res_block1 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True),
                                        nn.Linear(hidden, hidden))
        self.res_block2 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True),
                                        nn.Linear(hidden, hidden))
        self.value_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs):
        net = self.ppo_diffuser
        with torch.no_grad():
            cond, grid = net.context(obs)
        x = torch.cat([cond, net.current_states(obs), self.map_pool(grid).flatten(1)], dim=-1)
        x = F.relu(self.input_norm(self.input_proj(x)))
        x = x + F.relu(self.res_block1(x))
        x = x + F.relu(self.res_block2(x))
        return self.value_head(x).squeeze(-1)


def chain_log_prob(net, obs, chains, eta=0.3, detach_context=False, chunk=None):
    """log p of each diffusion step of chains (B, K+1, T, 2) -> (B, K).

    The context (with the map encoder's BatchNorm) is computed on the whole batch, as in the
    original. The denoiser treats every (sample, step) row on its own, so with `chunk` it runs
    on at most that many rows at a time: same result, bounded memory."""
    B, K1, T, _ = chains.shape
    K = K1 - 1
    if detach_context:
        with torch.no_grad():
            cond, grid = net.context(obs)
    else:
        cond, grid = net.context(obs)
    curr, rfa = net.current_states(obs), obs["raster_from_agent"]
    x_t = chains[:, :K].reshape(B * K, T, 2)
    x_next = chains[:, 1:].reshape(B * K, T, 2)
    t, t_next = ddim_timesteps(K, net.cfg.n_diffusion_steps)
    t = torch.tensor(t, device=chains.device).view(1, K).expand(B, K).reshape(-1)
    t_next = torch.tensor(t_next, device=chains.device).view(1, K).expand(B, K).reshape(-1)
    rows = B * K
    step = rows if chunk is None else chunk
    out = []
    for i in range(0, rows, step):
        r = torch.arange(i, min(i + step, rows), device=chains.device)
        o = r // K                     # row r belongs to sample r // K (repeat_interleave order)
        logits, mu, sigma = net.denoise(x_t[r], cond[o], grid[o], net.time_input(net.log_sigma_table[t[r]]),
                                        curr[o], rfa[o])
        mean, std = net.ddim_mixture(x_t[r], mu, sigma, t[r], t_next[r], eta)
        dist = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=logits),
            torch.distributions.Independent(torch.distributions.Normal(mean, std), 1))
        out.append(dist.log_prob(x_next[r]).sum(dim=1))
    return torch.cat(out).reshape(B, K)


def gae(rew, v_s, v_s_, terminated, truncated, gamma=0.99, lam=0.95):
    """Returns and advantages, as tianshou's compute_episodic_return over one buffer in time
    order. The last transition always ends an episode (it was the last one collected)."""
    rew = np.asarray(rew, dtype=float)
    v_s_ = np.asarray(v_s_) * ~np.asarray(terminated, dtype=bool)
    end = np.logical_or(terminated, truncated)
    end[-1] = True
    delta = rew + v_s_ * gamma - np.asarray(v_s)
    discount = (1.0 - end) * (gamma * lam)
    adv = np.zeros(rew.shape)
    g = 0.0
    for i in range(len(rew) - 1, -1, -1):
        g = delta[i] + discount[i] * g
        adv[i] = g
    return adv + v_s, adv


def splits(n, size, rng=np.random):
    """Minibatch indices, shuffled unless rng is None; a short last batch is merged into the
    one before (tianshou's Batch.split with merge_last)."""
    idx = np.arange(n) if rng is None else rng.permutation(n)
    merge = n % size > 0
    for i in range(0, n, size):
        if merge and i + size + size >= n:
            yield idx[i:]
            break
        yield idx[i:i + size]


def stack(views):
    """Stack per-step observations (dicts of arrays, nested extras) into one torch batch."""
    out = {}
    for k, v in views[0].items():
        if isinstance(v, dict):
            out[k] = stack([x[k] for x in views])
        elif isinstance(v, (str, bytes)) or (isinstance(v, np.ndarray) and v.dtype.kind in "USO"):
            out[k] = [x[k] for x in views]
        else:
            out[k] = torch.as_tensor(np.stack([np.asarray(x[k]) for x in views]))
    return out


def take_rows(batch, idx):
    idx = torch.as_tensor(idx, dtype=torch.long)
    if isinstance(batch, dict):
        return {k: take_rows(v, idx) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.index_select(0, idx)
    return [batch[i] for i in idx.tolist()]


class DiffusionPPO:
    def __init__(self, net, cfg=PPOConfig()):
        self.cfg = cfg
        net.requires_grad_(False)
        net.neigh_role_emb.requires_grad_(True)
        net.model.requires_grad_(True)
        net.cfg.sigma_gmm_scale = cfg.sigma_gmm_scale
        net.train()
        self.net = net
        self.device = next(net.parameters()).device
        self.critic = ValueCritic(net).to(self.device)
        params = [p for p in chain(net.parameters(), self.critic.parameters()) if p.requires_grad]
        self.optim = torch.optim.Adam(params, lr=cfg.lr)

    def modules(self):
        return [self.net, self.critic]

    def train(self, mode=True):
        for m in self.modules():
            m.train(mode)

    @torch.no_grad()
    def act(self, obs):
        """Sample one chain for a batch of observations -> (B, K+1, T, 2), on CPU."""
        return self.net.sample(to_device(obs, self.device), 1, eta=self.cfg.eta)["chain"][:, 0].cpu()

    def update(self, data, rng=np.random):
        """data: obs, obs_next (stacked batches), act (B, K+1, T, 2), rew, terminated,
        truncated, in collection order. Returns the per-minibatch losses."""
        cfg, dev = self.cfg, self.device
        n = len(data["rew"])
        rows_on = lambda part, rows: to_device(take_rows(data[part], rows), dev)  # noqa: E731
        with torch.no_grad():
            v_s, v_s_ = [], []
            for rows in splits(n, cfg.max_batchsize, rng=None):
                v_s.append(self.critic(rows_on("obs", rows)))
                v_s_.append(self.critic(rows_on("obs_next", rows)))
            v_s, v_s_ = torch.cat(v_s).flatten(), torch.cat(v_s_).flatten()
        ret, adv = gae(data["rew"], v_s.cpu().numpy(), v_s_.cpu().numpy(), data["terminated"], data["truncated"],
                       cfg.gamma, cfg.gae_lambda)
        ret = torch.as_tensor(ret, dtype=v_s.dtype, device=dev)
        adv = torch.as_tensor(adv, dtype=v_s.dtype, device=dev)
        act = torch.as_tensor(data["act"], dtype=v_s.dtype, device=dev)
        with torch.no_grad():
            logp_old = chain_log_prob(self.net, to_device(data["obs"], dev), act, cfg.eta, cfg.detach_context,
                                      chunk=cfg.logp_chunk)
        self.last = {"v_s": v_s, "returns": ret, "adv": adv, "logp_old": logp_old}

        actor_critic = nn.ModuleList([self.net, self.critic])
        losses = {"loss": [], "loss/clip": [], "loss/vf": []}
        for _ in range(cfg.repeat):
            for rows in splits(n, cfg.batch_size, rng):
                obs = rows_on("obs", rows)
                rows = torch.as_tensor(rows, device=dev)
                logp = chain_log_prob(self.net, obs, act[rows], cfg.eta, cfg.detach_context)
                a = adv[rows]
                a = (a - a.mean()) / (a.std() + 1e-8)
                delta = logp - logp_old[rows]
                if not torch.isfinite(delta).all():
                    continue
                ratio = delta.clamp(-20.0, 20.0).exp().float()
                a = a.unsqueeze(1).expand_as(ratio)
                clip_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).sum(dim=1).mean()
                vf_loss = (ret[rows] - self.critic(obs).flatten()).pow(2).mean()
                loss = clip_loss + cfg.vf_coef * vf_loss
                self.optim.zero_grad()
                loss.backward()
                if cfg.max_grad_norm:
                    nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=cfg.max_grad_norm)
                self.optim.step()
                losses["loss"].append(loss.item())
                losses["loss/clip"].append(clip_loss.item())
                losses["loss/vf"].append(vf_loss.item())
        return losses
