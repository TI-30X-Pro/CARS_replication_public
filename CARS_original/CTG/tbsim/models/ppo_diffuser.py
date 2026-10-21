import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.dynamics import Unicycle
from tbsim.models.context import (
    AgentHistoryEncoderTCN,
    AgentsMHA,
    EgoEncoder,
    MapEncoder,
    NeighborHistoryEncoder,
)
from tbsim.models.context_encoder import MLP
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.utils.guidance_loss import approach_cost
from .diffuser_helpers import (
    convert_state_to_state_and_action,
    cosine_beta_schedule,
    extract,
    query_feature_grid,
    unicyle_forward_dynamics,
)
from .ppo_diffusion_model import ConvCrossAttnDiffuser


class PPO_Diffuser(nn.Module):
    def __init__(
        self,
        map_encoder_model_arch,
        input_image_shape,
        grid_feature_dim,
        history_frames,
        action_norm_info_center,
        norm_info_center,
        norm_info_neighbor,
        norm_info_ego,
        context_encoder_out_dim,
        time_emb_dim,
        diffusion_hidden_dim,
        dilations,
        num_heads,
        num_Gaussian,
        dynamics_kwargs,
        n_timesteps,
        dt,
        ddim_steps,
        sigma_gmm_scale,
    ):
        super().__init__()

        self.map_encoder = MapEncoder(
            model_arch=map_encoder_model_arch,
            input_image_shape=input_image_shape,
            grid_feature_dim=grid_feature_dim,
        )

        self.center_hist = AgentHistoryEncoderTCN(
            num_steps=history_frames,
            out_dim=context_encoder_out_dim,
            norm_info=norm_info_center,
        )

        self.ego_encoder = EgoEncoder(
            out_dim=context_encoder_out_dim,
            norm_info=norm_info_ego,
        )

        self.neighbor_hist = NeighborHistoryEncoder(
            num_steps=history_frames,
            out_dim=context_encoder_out_dim,
            norm_info=norm_info_neighbor,
        )

        self.process_cond_mlp = MLP(
            in_dim=context_encoder_out_dim + grid_feature_dim + 2,
            out_dim=context_encoder_out_dim,
            hidden_dims=[context_encoder_out_dim, context_encoder_out_dim],
        )

        self.neigh_role_emb = nn.Embedding(2, context_encoder_out_dim)
        nn.init.zeros_(self.neigh_role_emb.weight)

        self.agents_mha = AgentsMHA(
            d_model=context_encoder_out_dim,
            n_heads=num_heads,
        )

        self.model = ConvCrossAttnDiffuser(
            in_dim=2,
            cond_dim=context_encoder_out_dim,
            time_emb_dim=time_emb_dim,
            hidden_dim=diffusion_hidden_dim,
            out_dim=2,
            dilations=dilations,
            n_heads=num_heads,
            grid_map_traj_dim=grid_feature_dim,
            mix_gauss=num_Gaussian,
        )

        self._dynamics_kwargs = dynamics_kwargs
        self._create_dynamics(dynamics_kwargs)

        self.dt = dt

        norm_add_coeffs = action_norm_info_center[0]
        norm_div_coeffs = action_norm_info_center[1]
        self.add_coeffs = np.array(norm_add_coeffs).astype("float32")
        self.div_coeffs = np.array(norm_div_coeffs).astype("float32")
        print("self.add_coeffs", self.add_coeffs)
        print("self.div_coeffs", self.div_coeffs)

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=alphas_cumprod.device), alphas_cumprod[:-1]]
        )

        self.n_timesteps = int(n_timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
        )

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        )
        self.register_buffer("posterior_variance", posterior_variance)

        self.sqrt_alphas_over_one_minus_alphas_cumprod = torch.sqrt(
            alphas_cumprod / (1.0 - alphas_cumprod)
        )
        self.sqrt_recip_one_minus_alphas_cumprod = 1.0 / torch.sqrt(
            1.0 - alphas_cumprod
        )

        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "log_sigma_table",
            0.5 * torch.log((1.0 - self.alphas_cumprod) / self.alphas_cumprod),
        )
        self.register_buffer("log_sigma_min", torch.min(self.log_sigma_table))
        self.register_buffer("log_sigma_max", torch.max(self.log_sigma_table))

        self.input_image_shape = input_image_shape
        self.ddim_steps = ddim_steps
        self.stationary_mask = None
        self.sigma_gmm_scale = sigma_gmm_scale

    def _create_dynamics(self, config):
        self.dyn = Unicycle(
            "dynamics",
            max_steer=config["max_steer"],
            max_yawvel=config["max_yawvel"],
            acce_bound=config["acce_bound"],
            vbound=config["vbound"],
        )

    def scale_traj(self, target_traj_orig):
        add_coeffs = self.add_coeffs[None, None]
        div_coeffs = self.div_coeffs[None, None]
        device = target_traj_orig.device
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)
        target_traj = (target_traj_orig - dx_add) / dx_div

        return target_traj

    def descale_traj(self, target_traj_orig):
        add_coeffs = self.add_coeffs[None, None]
        div_coeffs = self.div_coeffs[None, None]

        device = target_traj_orig.device
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)

        target_traj = target_traj_orig * dx_div + dx_add

        return target_traj

    def query_map_feats(self, pos_traj, map_grid_feat, raster_from_agent):
        """
        - x : (B, T, D)
        - map_grid_feat : (B, C, H, W)
        - raster_from_agent: (B, 3, 3)
        """
        B, T, _ = pos_traj.shape
        _, _, Hf, Wf = map_grid_feat.shape
        raster_xy = transform_points_tensor(pos_traj, raster_from_agent)

        _, H, W = self.input_image_shape
        raster_xy[..., 0] = raster_xy[..., 0] * (Wf / W)
        raster_xy[..., 1] = raster_xy[..., 1] * (Hf / H)

        feats = query_feature_grid(raster_xy, map_grid_feat)
        return feats.view(B, T, -1)

    def get_state_and_action_from_data_batch(self, data_batch, chosen_inds=[]):
        """
        Extract state and(or) action from the data_batch from data_batch
        Input:
            data_batch: dict
        Output:
            x: (batch_size, num_steps, len(chosen_inds)).
        """
        traj_state = torch.cat(
            (
                data_batch["target_positions"][:, :20, :],
                data_batch["target_yaws"][:, :20, :],
            ),
            dim=2,
        )

        traj_state_and_action = convert_state_to_state_and_action(
            traj_state,
            data_batch["curr_speed"],
            self.dt,
        )

        return traj_state_and_action[..., -2:]

    def context_encoder(self, batch):
        global_feat, grid_feat = self.map_encoder(batch["maps"])
        B, device = global_feat.size(0), global_feat.device

        center_token = self.center_hist(
            batch["history_positions"],
            batch["history_speeds"],
            batch["history_yaws"].squeeze(-1),
            batch["extent"][..., :2],
            batch["history_availabilities"],
        )

        neigh_tokens, neigh_mask = self.neighbor_hist(
            batch["all_other_agents_history_positions"],
            batch["all_other_agents_history_speeds"],
            batch["all_other_agents_history_yaws"].squeeze(-1),
            batch["all_other_agents_history_extents"][:, :, 0, :2],
            batch["all_other_agents_history_availability"],
        )

        N = neigh_tokens.size(1)
        role_id = torch.zeros((B, N), dtype=torch.long, device=device)

        if "target_idx" in batch:
            target_idx = batch["target_idx"].long()
            neigh_indices = batch["neigh_indices"].long()
            target_mask = neigh_indices == target_idx.unsqueeze(1)

            valid = (target_mask.sum(dim=1) == 1)
            if valid.any():
                role_id[valid] = target_mask[valid].long()

        neigh_tokens = neigh_tokens + self.neigh_role_emb(role_id)

        center_agg = self.agents_mha(center_token, neigh_tokens, neigh_mask)

        sg = batch["map_is_singapore"].to(center_agg.device).long()
        map_type_onehot = torch.stack([1 - sg, sg], dim=1).float()

        cond_in = torch.cat([center_agg, global_feat, map_type_onehot], dim=-1)
        cond_feat = self.process_cond_mlp(cond_in)

        return cond_feat, grid_feat

    def compute_loss(self, batch):
        cond_feat, map_grid_feat = self.context_encoder(batch)

        target_traj_action = self.get_state_and_action_from_data_batch(batch)

        x0 = self.scale_traj(target_traj_action)

        B = x0.shape[0]
        log_sigma_min, log_sigma_max = self.log_sigma_min, self.log_sigma_max

        u = torch.rand(B, device=x0.device)
        logsig_noise = log_sigma_min + u * (log_sigma_max - log_sigma_min)
        sigma = torch.exp(logsig_noise)
        alpha_bar = 1.0 / (1.0 + sigma**2)
        sqrt_ab = alpha_bar.sqrt().view(B, 1, 1)
        sqrt_1ab = (1.0 - alpha_bar).clamp_min(1e-8).sqrt().view(B, 1, 1)

        noise = torch.randn_like(x0)

        x_t = sqrt_ab * x0 + sqrt_1ab * noise

        curr_state = batch_utils().get_current_states(batch, self.dyn.type())
        pred_positions = self.convert_action_to_state_and_action(x_t, curr_state)[..., :2]

        map_grid_traj = self.query_map_feats(
            pred_positions,
            map_grid_feat,
            batch["raster_from_agent"],
        )

        t_float = (logsig_noise - log_sigma_min) / (
            log_sigma_max - log_sigma_min
        )

        raw_logits, mu, log_sigma_gmm = self.model(
            x_t,
            cond_feat,
            t_float,
            map_grid_feat,
            map_grid_traj,
        )

        sigma_gmm = (0.1 + F.softplus(log_sigma_gmm)) * self.sigma_gmm_scale

        gmm = MixtureSameFamily(
            Categorical(logits=raw_logits),
            Independent(Normal(mu, sigma_gmm), 1),
        )

        avail = batch["target_availabilities"].to(x0.device).float()
        logp = gmm.log_prob(x0)
        denom = avail.sum().clamp_min(1.0)
        nll = -(logp * avail).sum() / denom

        pi = F.softmax(raw_logits, dim=-1)
        with torch.no_grad():
            pi_mean = pi.mean(dim=(0, 1))
            H_mix = -(pi_mean * pi_mean.clamp_min(1e-12).log()).sum()
            effective_K = torch.exp(H_mix)

        H_pi = (-(pi * pi.clamp_min(1e-12).log()).sum(-1) * avail).sum() / denom
        lambda_repel = 0.03
        repel_scale = 0.5

        diff = mu.unsqueeze(2) - mu.unsqueeze(3)
        dist2 = (diff * diff).sum(-1)

        K = mu.size(2)
        eye = torch.eye(K, device=mu.device).view(1, 1, K, K)
        offdiag = 1.0 - eye

        repel = torch.exp(-dist2 / (repel_scale**2 + 1e-8)) * offdiag

        repel_loss = (repel.mean(dim=(-1, -2)) * avail).sum() / denom

        loss = nll - 0.15 * H_pi + lambda_repel * repel_loss
        return loss, H_pi, repel_loss, effective_K

    def make_ddim_timesteps(self, ddim_steps: int, n_timesteps: int):
        c = torch.linspace(
            n_timesteps - 1,
            0,
            ddim_steps,
            device=self.betas.device,
        ).long()
        next_c = list(c[1:].tolist()) + [0]
        return c.tolist(), next_c

    def convert_action_to_state_and_action(self, x_out, curr_states):
        dim = len(x_out.shape)
        if dim == 4:
            B, N, T, _ = x_out.shape
            x_out = TensorUtils.join_dimensions(x_out, 0, 2)

        x_out = self.descale_traj(x_out)
        x_out_state = unicyle_forward_dynamics(
            dyn_model=self.dyn,
            initial_states=curr_states,
            actions=x_out,
            step_time=self.dt,
            mode="parallel",
        )

        x_out_all = torch.cat([x_out_state, x_out], dim=-1)

        if dim == 4:
            x_out_all = x_out_all.reshape([B, N, T, -1])
        return x_out_all

    def forward(self, obs, stationary_mask, global_t=0, num_samples=1):
        if global_t == 0:
            self.stationary_mask = stationary_mask

        return self.sample_ddim(obs, num_samples=num_samples)

    @torch.no_grad()
    def sample_ddim(self, batch, eta=0.3, num_samples=1, return_diffusion=False):
        cond_feat, map_grid_feat = self.context_encoder(batch)
        B, T, _ = batch["target_positions"].shape
        curr_state = batch_utils().get_current_states(batch, self.dyn.type())

        ddim_steps = int(self.ddim_steps)
        timesteps, next_timesteps = self.make_ddim_timesteps(
            ddim_steps,
            self.n_timesteps,
        )

        x_t = torch.randn((B, num_samples, T, 2), device=curr_state.device)
        x_t = TensorUtils.join_dimensions(
            x_t,
            begin_axis=0,
            end_axis=2,
        )

        if return_diffusion:
            diffusion = [x_t.clone()]

        cond_feat_exp = TensorUtils.repeat_by_expand_at(
            cond_feat,
            repeats=num_samples,
            dim=0,
        )
        map_grid_feat_exp = TensorUtils.repeat_by_expand_at(
            map_grid_feat,
            repeats=num_samples,
            dim=0,
        )
        curr_state_exp = TensorUtils.repeat_by_expand_at(
            curr_state,
            repeats=num_samples,
            dim=0,
        )
        raster_from_agent_exp = TensorUtils.repeat_by_expand_at(
            batch["raster_from_agent"], repeats=num_samples, dim=0
        )

        BN = B * num_samples
        for t, t_next in zip(timesteps, next_timesteps):
            t_tensor = torch.full((BN,), t, dtype=torch.long, device=x_t.device)
            t_next_tensor = torch.full((BN,), t_next, dtype=torch.long, device=x_t.device)
            log_sigma_t = self.log_sigma_table[t_tensor]
            t_float = (log_sigma_t - self.log_sigma_min) / (
                self.log_sigma_max - self.log_sigma_min
            )

            pred_positions = self.convert_action_to_state_and_action(
                x_t,
                curr_state_exp,
            )[..., :2]

            map_grid_traj = self.query_map_feats(
                pred_positions,
                map_grid_feat_exp,
                raster_from_agent_exp,
            )

            raw_logits, mu, log_sigma = self.model(
                x_t, cond_feat_exp, t_float, map_grid_feat_exp, map_grid_traj
            )
            sigma_gmm = (0.1 + F.softplus(log_sigma)) * self.sigma_gmm_scale

            mix_dist = self.mix_dist_continuous(
                x_t,
                raw_logits,
                mu,
                sigma_gmm,
                t_tensor,
                t_next_tensor,
                eta,
            )
            x_t = mix_dist.sample()

            if self.stationary_mask is not None:
                x_stationary = x_t[self.stationary_mask]
                x_stationary = self.descale_traj(x_stationary)
                x_stationary[...] = 0
                x_stationary = self.scale_traj(x_stationary)
                x_t[self.stationary_mask] = x_stationary

            if return_diffusion:
                diffusion.append(x_t.clone())

        traj = self.convert_action_to_state_and_action(
            x_t,
            curr_state_exp,
        )
        traj = TensorUtils.reshape_dimensions(
            traj, begin_axis=0, end_axis=1, target_dims=(B, num_samples)
        )

        out_dict = {
            "positions": traj[..., :2],
            "yaws": traj[..., 3:4],
        }

        if return_diffusion:
            diffusion = [
                TensorUtils.reshape_dimensions(
                    cur_diff,
                    begin_axis=0,
                    end_axis=1,
                    target_dims=(B, num_samples),
                )
                for cur_diff in diffusion
            ]
            out_dict["diffusion"] = torch.stack(diffusion, dim=-1)

        return out_dict

    def sample_ddim_guided(
        self,
        batch,
        target_fut_pos,
        eta=0.3,
        num_samples=1,
        grad_scale=0.05,
        grad_steps=3,
        perturb_th=0.3,
    ):
        """DDIM sampling with inference-time gradient guidance (CTG-style).

        Following the CTG approach: guidance is applied to the predicted clean
        x0 in decoded state space (after unicycle dynamics clip), then the
        guided x0 is used to compute the DDIM posterior for x_{t-1}.
        This ensures perturbations respect physical bounds.

        target_fut_pos: (T, 2) tensor, target predicted future positions
                        (in adv agent-centric frame).
        """
        cond_feat, map_grid_feat = self.context_encoder(batch)
        B, T, _ = batch["target_positions"].shape
        curr_state = batch_utils().get_current_states(batch, self.dyn.type())

        ddim_steps = int(self.ddim_steps)
        timesteps, next_timesteps = self.make_ddim_timesteps(
            ddim_steps,
            self.n_timesteps,
        )

        x_t = torch.randn((B, num_samples, T, 2), device=curr_state.device)
        x_t = TensorUtils.join_dimensions(
            x_t,
            begin_axis=0,
            end_axis=2,
        )

        cond_feat_exp = TensorUtils.repeat_by_expand_at(
            cond_feat,
            repeats=num_samples,
            dim=0,
        )
        map_grid_feat_exp = TensorUtils.repeat_by_expand_at(
            map_grid_feat,
            repeats=num_samples,
            dim=0,
        )
        curr_state_exp = TensorUtils.repeat_by_expand_at(
            curr_state,
            repeats=num_samples,
            dim=0,
        )
        rfa_exp = TensorUtils.repeat_by_expand_at(
            batch["raster_from_agent"],
            repeats=num_samples,
            dim=0,
        )

        tgt = target_fut_pos.to(x_t.device)
        BN = B * num_samples

        for t, t_next in zip(timesteps, next_timesteps):
            t_tensor = torch.full((BN,), t, dtype=torch.long, device=x_t.device)
            t_next_tensor = torch.full((BN,), t_next, dtype=torch.long, device=x_t.device)
            log_sigma_t = self.log_sigma_table[t_tensor]
            t_float = (log_sigma_t - self.log_sigma_min) / (
                self.log_sigma_max - self.log_sigma_min
            )

            with torch.no_grad():
                pred_pos = self.convert_action_to_state_and_action(
                    x_t,
                    curr_state_exp,
                )[..., :2]
                map_traj = self.query_map_feats(pred_pos, map_grid_feat_exp, rfa_exp)
                raw_logits, mu, log_sigma = self.model(
                    x_t, cond_feat_exp, t_float, map_grid_feat_exp, map_traj
                )

            pi = torch.softmax(raw_logits, dim=-1)
            x0_pred = (pi.unsqueeze(-1) * mu).sum(dim=2)

            x0_init = x0_pred.detach()
            x0_guided = x0_init.clone().requires_grad_(True)

            for _ in range(grad_steps):
                x0_state = self.convert_action_to_state_and_action(
                    x0_guided, curr_state_exp
                )[..., :2]

                cost = approach_cost(x0_state, tgt).mean()

                grad = torch.autograd.grad(cost, x0_guided)[0]
                x0_guided = (x0_guided - grad_scale * grad).detach()

                with torch.no_grad():
                    pos_init = self.convert_action_to_state_and_action(
                        x0_init, curr_state_exp
                    )[..., :2]
                    pos_guided = self.convert_action_to_state_and_action(
                        x0_guided, curr_state_exp
                    )[..., :2]
                    pos_delta = pos_guided - pos_init
                    pos_delta_clipped = pos_delta.clamp(-perturb_th, perturb_th)
                    action_delta = x0_guided - x0_init
                    scale = torch.where(
                        pos_delta.abs() > 1e-6,
                        pos_delta_clipped / pos_delta,
                        torch.ones_like(pos_delta),
                    )
                    scale_mean = scale.mean(dim=-1, keepdim=True)
                    x0_guided = (x0_init + action_delta * scale_mean).detach()

                x0_guided = x0_guided.requires_grad_(True)

            with torch.no_grad():
                sigma_gmm = (0.1 + F.softplus(log_sigma)) * self.sigma_gmm_scale
                mu_guided = x0_guided.detach().unsqueeze(2)
                best_k = pi.argmax(dim=-1)
                sigma_best = sigma_gmm[
                    torch.arange(BN).unsqueeze(1).expand_as(best_k),
                    torch.arange(T).unsqueeze(0).expand_as(best_k),
                    best_k,
                ].unsqueeze(2)
                logits_single = torch.zeros(BN, T, 1, device=x_t.device)

                mix_dist = self.mix_dist_continuous(
                    x_t,
                    logits_single,
                    mu_guided,
                    sigma_best,
                    t_tensor,
                    t_next_tensor,
                    eta,
                )
                x_t = mix_dist.sample()

            if self.stationary_mask is not None:
                x_stationary = x_t[self.stationary_mask]
                x_stationary = self.descale_traj(x_stationary)
                x_stationary[...] = 0
                x_stationary = self.scale_traj(x_stationary)
                x_t[self.stationary_mask] = x_stationary

        traj = self.convert_action_to_state_and_action(x_t, curr_state_exp)
        traj = TensorUtils.reshape_dimensions(
            traj,
            begin_axis=0,
            end_axis=1,
            target_dims=(B, num_samples),
        )

        return {
            "positions": traj[..., :2],
            "yaws": traj[..., 3:4],
        }

    def mix_dist_continuous(
        self,
        x_t,
        raw_logits,
        mu,
        sigma_gmm,
        t_tensor,
        t_next_tensor,
        eta,
    ):
        if not torch.isfinite(x_t).all():
            raise RuntimeError("x_t has NaN/Inf before mix_dist_continuous")
        if not torch.isfinite(raw_logits).all():
            raise RuntimeError("raw_logits has NaN/Inf before mix_dist_continuous")
        if not torch.isfinite(mu).all():
            raise RuntimeError("mu has NaN/Inf before mix_dist_continuous")
        if not torch.isfinite(sigma_gmm).all():
            raise RuntimeError("sigma_gmm has NaN/Inf before mix_dist_continuous")
        t_tensor = t_tensor.long().clamp_(0, self.n_timesteps - 1)
        t_next_tensor = t_next_tensor.long().clamp_(0, self.n_timesteps - 1)

        B, T, _ = x_t.shape
        shape_scalar = (B, 1, 1)

        alpha_t = extract(self.alphas_cumprod, t_tensor, shape_scalar)
        alpha_next = extract(self.alphas_cumprod, t_next_tensor, shape_scalar)
        alpha_t = alpha_t.clamp(1e-4, 0.9999)
        alpha_next = alpha_next.clamp(1e-4, 0.9999)

        alpha_t = alpha_t.expand(B, T, 1).repeat_interleave(2, dim=-1)
        alpha_next = alpha_next.expand(B, T, 1).repeat_interleave(2, dim=-1)

        ratio = (alpha_t / alpha_next).clamp(1e-4, 1e4)
        sigma_t_inner = (
            (1 - ratio) * (1 - alpha_next) / (1 - alpha_t).clamp_min(1e-4)
        ).clamp_min(1e-8)
        sigma_t = eta * torch.sqrt(sigma_t_inner)
        sigma_t_k = sigma_t.unsqueeze(2)

        s1 = torch.sqrt(alpha_next)
        s2 = torch.sqrt((1 - alpha_next - sigma_t**2).clamp_min(1e-8))
        s3 = torch.sqrt(alpha_t)
        s4 = torch.sqrt((1 - alpha_t).clamp_min(1e-4))

        s3_over_s4 = s3 / s4
        s2_over_s4 = s2 / s4

        A = s1 - s2 * s3_over_s4
        C = s2_over_s4 * x_t

        A_k = A.unsqueeze(2)
        C_k = C.unsqueeze(2)

        mu_next_k = A_k * mu + C_k
        var_next_k = (A_k**2) * (sigma_gmm**2) + (sigma_t_k**2)
        std_next_k = torch.sqrt(var_next_k.clamp_min(1e-4))

        if not torch.isfinite(mu_next_k).all() or not torch.isfinite(std_next_k).all():
            raise RuntimeError("NaN/Inf in DDIM-transformed params")

        comp = torch.distributions.Independent(
            torch.distributions.Normal(mu_next_k, std_next_k), 1
        )
        mix = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=raw_logits), comp
        )
        return mix
