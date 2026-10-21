from tianshou.policy import PPOPolicy
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.batch_utils import batch_utils

class DiffusionActor(nn.Module):
    def __init__(self, ppo_diffuser: nn.Module, eta: float = 0.3):
        super().__init__()
        self.ppo_diffuser = ppo_diffuser
        self.eta = float(eta)

    def forward(self, obs, state=None, info=None):
        device = next(self.ppo_diffuser.parameters()).device

        # obs_t = to_torch({k: v for k, v in obs.items() if k != "map_names"}, device=device)
        obs_torch = to_torch(obs, device=device)
        # obs_torch.update({k: obs[k] for k in ("map_names", "scene_ids")})
      
        out = self.ppo_diffuser.sample_ddim(obs_torch, eta=self.eta, num_samples=1, return_diffusion=True)

     
        x_chain = out["diffusion"][:, 0].permute(0, 3, 1, 2).contiguous()   # (B, K+1, T, 2)
        return x_chain, state

class ValueCritic(nn.Module):
    """PPO value critic.

    Design notes:
    - context_encoder is always detached so that value loss never back-propagates
      into the pretrained diffusion encoder.
    - LayerNorm + residual connections stabilise training.
    - 3-layer MLP at width 512 gives enough capacity for the value target.
    - Final head is small-initialised to keep early value estimates bounded.
    """

    def __init__(self, ppo_diffuser: nn.Module, hidden: int = 512):
        super().__init__()
        self.ppo_diffuser = ppo_diffuser

        self.map_pool = nn.AdaptiveAvgPool2d(1)

        self.input_proj = nn.LazyLinear(hidden)
        self.input_norm = nn.LayerNorm(hidden)

        self.res_block1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )

        # Small-init value head to keep initial value estimates bounded.
        self.value_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, state=None, info=None):
        device = next(self.parameters()).device
        obs = dict(obs)
        obs_t = to_torch(obs, device=device)

        # Encoder always detached: value loss must not perturb the pretrained encoder.
        with torch.no_grad():
            cond_feat, map_grid_feat = self.ppo_diffuser.context_encoder(obs_t)

        curr_state = batch_utils().get_current_states(obs_t, self.ppo_diffuser.dyn.type())
        map_global = self.map_pool(map_grid_feat).flatten(1)

        x = torch.cat([cond_feat, curr_state, map_global], dim=-1)
        x = F.relu(self.input_norm(self.input_proj(x)))

        x = x + F.relu(self.res_block1(x))
        x = x + F.relu(self.res_block2(x))

        v = self.value_head(x).squeeze(-1)
        return v

class DiffusionChainDist:
    def __init__(
        self,
        ppo_diffuser: nn.Module,
        obs: Any,                 
        eta: float,
        detach_context: bool = True,
        sigma_gmm_scale: float = 1.0,
    ):
        self.ppo_diffuser = ppo_diffuser
        self.obs = obs
        self.eta = float(eta)
        self.detach_context = bool(detach_context)
        self.sigma_gmm_scale = float(sigma_gmm_scale)
        self._prepared = False
   
        self._cond_feat: Optional[torch.Tensor] = None
        self._map_grid_feat: Optional[torch.Tensor] = None
        self._curr_state: Optional[torch.Tensor] = None
        self._raster_from_agent: Optional[torch.Tensor] = None

    def _prepare(self, device: torch.device):
        if self._prepared:
            return

        obs = to_torch(self.obs, device=device)

 

        if self.detach_context:
            with torch.no_grad():
                cond_feat, map_grid_feat = self.ppo_diffuser.context_encoder(obs)
        else:
            cond_feat, map_grid_feat = self.ppo_diffuser.context_encoder(obs)

        self._cond_feat = cond_feat
        self._map_grid_feat = map_grid_feat

        self._curr_state = batch_utils().get_current_states(obs, self.ppo_diffuser.dyn.type())
        self._raster_from_agent = obs["raster_from_agent"]

        self._prepared = True

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        device = act.device
        self._prepare(device)

        B, K1, T, D = act.shape
        assert D == 2
        K = K1 - 1

        # (B,K,T,2)
        x_t_all = act[:, :K]
        x_next_all = act[:, 1:]

        # flatten -> (B*K,T,2)
        x_t = x_t_all.reshape(B * K, T, 2)
        x_next = x_next_all.reshape(B * K, T, 2)

        # timesteps (length K)
        timesteps, next_timesteps = self.ppo_diffuser.make_ddim_timesteps(
            K, self.ppo_diffuser.n_timesteps
        )
        t = torch.tensor(timesteps, device=device, dtype=torch.long)          # (K,)
        t_next = torch.tensor(next_timesteps, device=device, dtype=torch.long) # (K,)

        # expand to (B,K) then flatten to (B*K,)
        t_tensor = t.view(1, K).expand(B, K).reshape(-1)
        t_next_tensor = t_next.view(1, K).expand(B, K).reshape(-1)

        # t_float: (B*K,)
        log_sigma_t = self.ppo_diffuser.log_sigma_table[t_tensor]
        t_float = (log_sigma_t - self.ppo_diffuser.log_sigma_min) / (
                self.ppo_diffuser.log_sigma_max - self.ppo_diffuser.log_sigma_min
            )

        # repeat context to (B*K, ...)
        cond_feat = self._cond_feat.repeat_interleave(K, dim=0)          # (B*K, Cc)
        map_grid_feat = self._map_grid_feat.repeat_interleave(K, dim=0)  # (B*K, Cm, H, W)
        curr_state = self._curr_state.repeat_interleave(K, dim=0)        # (B*K, state_dim)
        raster = self._raster_from_agent.repeat_interleave(K, dim=0)     # (B*K, ...)

        # decode x_t -> positions
        pred_positions = self.ppo_diffuser.convert_action_to_state_and_action(
            x_t, curr_state
        )[..., :2]  # (B*K, T, 2)

        map_grid_traj = self.ppo_diffuser.query_map_feats(
            pred_positions, map_grid_feat, raster
        )

        raw_logits, mu, log_sigma = self.ppo_diffuser.model(
            x_t, cond_feat, t_float, map_grid_feat, map_grid_traj
            )

        sigma_gmm = (0.1 + F.softplus(log_sigma)) * self.sigma_gmm_scale

        dist = self.ppo_diffuser.mix_dist_continuous(
                x_t, raw_logits, mu, sigma_gmm, t_tensor, t_next_tensor, self.eta
            )

        # (B*K, T)
        step_logp = dist.log_prob(x_next)

        # sum over T -> (B*K,)
        step_logp_sum = step_logp.sum(dim=1)

        # Reshape to (B, K): per-diffusion-step log prob.
        per_step_logp = step_logp_sum.reshape(B, K)
        return per_step_logp


    def entropy(self) -> torch.Tensor:

        return torch.zeros((1,), device=next(self.ppo_diffuser.parameters()).device)



class DiffusionPPOPolicy(PPOPolicy):

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        eta: float = 0.3,
        detach_context: bool = False,
        **kwargs: Any,
    ):
        super().__init__(actor=actor, critic=critic, optim=optim, dist_fn=dist_fn, **kwargs)
        self.eta = float(eta)
        self.detach_context = bool(detach_context)

        assert hasattr(actor, "ppo_diffuser"), "actor must have attribute `ppo_diffuser`"
        self.ppo_diffuser = actor.ppo_diffuser

    def is_collect(self, batch: Batch) -> bool:
        if hasattr(batch, "returns") and batch.returns is not None:
            return False
        if hasattr(batch, "adv") and batch.adv is not None:
            return False
        return True

    def forward(self, batch: Batch, state=None, **kwargs) -> Batch:
        if self.is_collect(batch):
            act, hidden = self.actor(batch.obs, state=state, info=batch.info)
            return Batch(act=act, state=hidden, logits=None, dist=None)
        else:
            dist = DiffusionChainDist(
                ppo_diffuser=self.ppo_diffuser,
                obs=batch.obs,
                eta=self.eta,
                detach_context=self.detach_context,
                sigma_gmm_scale=self.ppo_diffuser.sigma_gmm_scale,
            )
            return Batch(act=batch.act, state=state, logits=None, dist=dist)


    def learn(
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:

        losses, clip_losses, vf_losses, ent_losses = [], [], [], []

        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)

            for minibatch in batch.split(batch_size, merge_last=True):
                # --- actor loss: per-step ratio & clip ---
                dist = self(minibatch).dist
                # per-step log prob: (B, K)
                logp_new = dist.log_prob(minibatch.act)
                # TODO(PPO-Clamp): add PPO clip objective (ratio clamp) and log clipfrac/approx_kl

                logp_old = minibatch.logp_old  # (B, K)

                # advantage normalization
                adv = minibatch.adv  # (B,)
                if self._norm_adv:
                    mean, std = adv.mean(), adv.std()
                    adv = (adv - mean) / (std + self._eps)

                # per-step ratio: (B, K)
                delta = logp_new - logp_old
                if not torch.isfinite(delta).all():
                    continue

                # Numerical safety only: clamp to keep exp() finite.
                delta = delta.clamp(-20.0, 20.0)
                ratio = delta.exp().float()

                # Broadcast adv from (B,) to (B, K) for per-step computation.
                adv_expanded = adv.unsqueeze(1).expand_as(ratio)

                surr1 = ratio * adv_expanded
                surr2 = ratio.clamp(
                    1.0 - self._eps_clip, 1.0 + self._eps_clip
                ) * adv_expanded

                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * adv_expanded)
                    per_step_loss = -torch.where(adv_expanded < 0, clip2, clip1)
                else:
                    per_step_loss = -torch.min(surr1, surr2)

                # Sum over the K diffusion steps, then average over the batch.
                clip_loss = per_step_loss.sum(dim=1).mean()

                # --- critic loss ---
                value = self.critic(minibatch.obs).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + \
                        (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()

                # --- entropy loss ---
                ent_loss = dist.entropy().mean()

                # --- total loss ---
                loss = clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss

                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()

                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
    