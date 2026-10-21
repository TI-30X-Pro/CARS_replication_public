import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.models.ppo_diffuser import PPO_Diffuser
from tbsim.policies.common import Action
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.guidance_loss import choose_action_from_gt
from tbsim.utils.trajdata_utils import get_stationary_mask


FILTERED_KEYS = [
    "maps",
    "center_hist_positions",
    "center_hist_speeds",
    "center_hist_yaws",
    "center_hist_acc_lons",
    "center_hist_yaw_rates",
    "center_hist_availabilities",
    "extent",
    "neigh_hist_positions",
    "neigh_hist_speeds",
    "neigh_hist_yaws",
    "neigh_hist_acc_lons",
    "neigh_hist_yaw_rates",
    "neigh_hist_availabilities",
    "neigh_extent",
    "center_curr_positions",
    "center_curr_speeds",
    "center_curr_yaws",
    "raster_from_center",
    "world_from_agent",
    "extras",
]


class PPO_Diffusion_Trainer(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, registered_name):
        super().__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.disable_control_on_stationary = algo_config.disable_control_on_stationary
        self.moving_speed_th = algo_config.moving_speed_th
        self.guided_mode = False
        self.guidance_kwargs = {}

        self.nets["policy"] = PPO_Diffuser(
            map_encoder_model_arch=algo_config.map_encoder_model_arch,
            input_image_shape=modality_shapes["image"],
            grid_feature_dim=algo_config.grid_feature_dim,
            history_frames=int(algo_config.history_num_frames + 1),
            action_norm_info_center=algo_config.action_norm_info_center,
            norm_info_center=algo_config.norm_info_center,
            norm_info_neighbor=algo_config.norm_info_neighbor,
            norm_info_ego=algo_config.norm_info_ego,
            context_encoder_out_dim=algo_config.context_encoder_out_dim,
            time_emb_dim=algo_config.time_emb_dim,
            diffusion_hidden_dim=algo_config.diffusion_hidden_dim,
            dilations=algo_config.dilations,
            num_heads=algo_config.num_heads,
            num_Gaussian=algo_config.num_Gaussian,
            dynamics_kwargs=algo_config.Dynamics,
            n_timesteps=algo_config.n_diffusion_steps,
            dt=algo_config.dt,
            ddim_steps=algo_config.ddim_steps,
            sigma_gmm_scale=algo_config.sigma_gmm_scale,
        )

        self.cur_train_step = 0

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.cur_train_step += 1

    def training_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        batch_size = batch["target_positions"].shape[0]
        loss, H_pi, repel_loss, effective_K = self.nets["policy"].compute_loss(batch)
        K = self.algo_config.num_Gaussian
        H_norm = (
            H_pi / math.log(K)
            if K > 1
            else torch.ones(1, device=loss.device).squeeze()
        )

        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size, on_step=True)
        self.log("train_H_pi", H_pi, prog_bar=True, batch_size=batch_size, on_step=True)
        self.log("train_H_norm", H_norm, prog_bar=True, batch_size=batch_size, on_step=True)
        self.log(
            "train_repel_loss",
            repel_loss,
            prog_bar=True,
            batch_size=batch_size,
            on_step=True,
        )
        self.log(
            "train_effective_K",
            effective_K,
            prog_bar=True,
            batch_size=batch_size,
            on_step=True,
        )
        self.log("K", float(K), prog_bar=False, batch_size=batch_size, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        batch_size = batch["target_positions"].shape[0]
        loss, H_pi, _, effective_K = self.nets["policy"].compute_loss(batch)
        K = self.algo_config.num_Gaussian
        H_norm = (
            H_pi / math.log(K)
            if K > 1
            else torch.ones(1, device=loss.device).squeeze()
        )

        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size, on_epoch=True)
        self.log("val_H_pi", H_pi, prog_bar=True, batch_size=batch_size, on_epoch=True)
        self.log("val_H_norm", H_norm, prog_bar=True, batch_size=batch_size, on_epoch=True)
        self.log(
            "val_effective_K",
            effective_K,
            prog_bar=True,
            batch_size=batch_size,
            on_epoch=True,
        )
        self.log("val_K", float(K), prog_bar=False, batch_size=batch_size, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params
        return optim.AdamW(
            params=self.nets["policy"].parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["learning_rate"]["weight_decay"],
        )

    def on_train_epoch_end(self):
        """Log the current learning rate."""
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr)

    def forward(self, obs, global_t, num_samples):
        if global_t == 0:
            self.stationary_mask = get_stationary_mask(
                obs, self.disable_control_on_stationary, self.moving_speed_th
            )
            batch_size = self.stationary_mask.shape[0]
            stationary_mask_expand = (
                self.stationary_mask.unsqueeze(1)
                .expand(batch_size, num_samples)
                .reshape(batch_size * num_samples)
            )
        else:
            stationary_mask_expand = None

        return self.nets["policy"](
            obs,
            stationary_mask=stationary_mask_expand,
            global_t=global_t,
            num_samples=num_samples,
        )

    def _select_action_idx(self, rollout_all: dict, obs_torch: dict, **kwargs) -> torch.Tensor:
        return choose_action_from_gt(rollout_all, obs_torch)

    def get_action(self, obs_torch, **kwargs):
        num_action_samples = kwargs.get("num_action_samples", 1)

        rollout_all = self(
            obs_torch,
            global_t=kwargs["step_index"],
            num_samples=num_action_samples,
        )
        batch_size, _, _, _ = rollout_all["positions"].size()

        if num_action_samples == 1:
            act_idx = torch.zeros(
                (batch_size,),
                dtype=torch.long,
                device=rollout_all["positions"].device,
            )
        else:
            act_idx = self._select_action_idx(rollout_all, obs_torch, **kwargs)

        rollout_best = TensorUtils.map_tensor(
            rollout_all,
            lambda x: x[torch.arange(batch_size), act_idx],
        )

        action = Action(
            positions=rollout_best["positions"],
            yaws=rollout_best["yaws"],
        )

        info = dict(
            action_samples=Action(
                positions=rollout_all["positions"],
                yaws=rollout_all["yaws"],
            ).to_dict()
        )
        return action, info

    def get_action_guided(self, obs_torch, target_fut_pos, **kwargs):
        """Gradient-guided inference action for the CTG+reward baseline.

        target_fut_pos: (T, 2) target future positions in the adversary
        agent-centric frame.
        """
        num_action_samples = kwargs.get("num_action_samples", 1)
        step_index = kwargs["step_index"]

        if step_index == 0:
            self.stationary_mask_guided = get_stationary_mask(
                obs_torch,
                self.disable_control_on_stationary,
                self.moving_speed_th,
            )
        sm = getattr(self, "stationary_mask_guided", None)
        self.nets["policy"].stationary_mask = (
            sm.unsqueeze(1).expand(sm.shape[0], num_action_samples).reshape(-1)
            if sm is not None
            else None
        )

        rollout_all = self.nets["policy"].sample_ddim_guided(
            obs_torch,
            target_fut_pos=target_fut_pos,
            num_samples=num_action_samples,
            **self.guidance_kwargs,
        )

        batch_size, _, _, _ = rollout_all["positions"].size()
        act_idx = torch.zeros(
            (batch_size,),
            dtype=torch.long,
            device=rollout_all["positions"].device,
        )
        rollout_best = TensorUtils.map_tensor(
            rollout_all,
            lambda x: x[torch.arange(batch_size), act_idx],
        )

        action = Action(
            positions=rollout_best["positions"],
            yaws=rollout_best["yaws"],
        )
        info = dict(
            action_samples=Action(
                positions=rollout_all["positions"],
                yaws=rollout_all["yaws"],
            ).to_dict()
        )
        return action, info
