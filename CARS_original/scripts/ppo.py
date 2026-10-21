"""PPO training script for adversarial scenario generation."""

import argparse
import importlib
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter
import wandb
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import LazyLogger, WandbLogger

from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.datasets.factory import datamodule_factory
from tbsim.evaluation.env_builders import EnvCLDBuilder
from tbsim.models.diffusionppo import DiffusionActor, DiffusionPPOPolicy, ValueCritic
from tbsim.models.ppo_actor_critic import build_save_best_fn, build_save_checkpoint_fn
from tbsim.models.ppo_env import PPOEnv
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.safety_critical_batch_utils import set_global_trajdata_batch_raster_cfg
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env


@dataclass
class PPOTrainingConfig:
    """PPO training configuration."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    num_train_envs: int = 1
    num_test_envs: int = 1

    ppo_epochs: int = 100
    scenes_per_collect: int = 2
    update_per_collect: int = 10
    mini_batch_size: int = 64
    test_episodes: int = 3
    train_scenes_slice: Optional[list] = None

    wandb_project: str = "ppo-adversarial"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "ppo_run"
    wandb_id: Optional[str] = None

    @classmethod
    def from_eval_cfg(cls, eval_cfg) -> "PPOTrainingConfig":
        """Create the training configuration from eval_cfg.ppo."""
        ppo = eval_cfg.ppo
        return cls(
            learning_rate=ppo.get("learning_rate", cls.learning_rate),
            gamma=ppo.get("gamma", cls.gamma),
            gae_lambda=ppo.get("gae_lambda", cls.gae_lambda),
            clip_ratio=ppo.get("clip_ratio", cls.clip_ratio),
            vf_coef=ppo.get("vf_coef", cls.vf_coef),
            ent_coef=ppo.get("ent_coef", cls.ent_coef),
            max_grad_norm=ppo.get("max_grad_norm", cls.max_grad_norm),
            num_train_envs=ppo.get("num_train_envs", cls.num_train_envs),
            num_test_envs=ppo.get("num_test_envs", cls.num_test_envs),
            ppo_epochs=ppo.get("ppo_epochs", cls.ppo_epochs),
            scenes_per_collect=ppo.get("scenes_per_collect", cls.scenes_per_collect),
            update_per_collect=ppo.get("update_per_collect", cls.update_per_collect),
            mini_batch_size=ppo.get("mini_batch_size", cls.mini_batch_size),
            test_episodes=ppo.get("test_episodes", cls.test_episodes),
            train_scenes_slice=ppo.get("train_scenes_slice", cls.train_scenes_slice),
            wandb_project=ppo.get("wandb_project", cls.wandb_project),
            wandb_entity=ppo.get("wandb_entity", cls.wandb_entity),
            wandb_run_name=ppo.get("wandb_run_name", cls.wandb_run_name),
            wandb_id=ppo.get("wandb_id", cls.wandb_id),
        )


class ModelBuilder:
    """Build the policies and PPO modules."""

    def __init__(self, eval_cfg, device: torch.device):
        self.eval_cfg = eval_cfg
        self.device = device

    def build_policies(self) -> Tuple:
        policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
        composer_class = getattr(policy_composers, self.eval_cfg.eval_class)

        target_composer = composer_class(self.eval_cfg, self.device)
        target_policy, exp_config = target_composer.get_policy()

        adv_composer = composer_class(self.eval_cfg, self.device)
        adv_policy, _ = adv_composer.get_planner()

        others_composer = composer_class(self.eval_cfg, self.device)
        others_policy, _ = others_composer.get_predictor()

        print("=" * 60)
        print("EXP CONFIG")
        print("=" * 60)
        exclude = {
            "norm_info_center",
            "norm_info_ego",
            "action_norm_info_center",
            "norm_info_neighbor",
            "norm_info",
        }

        for k in sorted(exp_config["algo"]):
            if k not in exclude:
                print(f"{k:<25} : {exp_config['algo'][k]}")

        print("=" * 60)

        return target_policy, adv_policy, others_policy, exp_config

    def build_adv_diffuser(
        self,
        target_policy,
        adv_policy,
        others_policy,
    ) -> torch.nn.Module:
        for policy in [target_policy, others_policy]:
            policy.model.nets["policy"].eval()
            policy.model.nets["policy"].requires_grad_(False)
            policy.model.nets["policy"].sigma_gmm_scale = 3

        adv_diffuser = adv_policy.model.nets["policy"]

        adv_diffuser.requires_grad_(False)
        adv_diffuser.neigh_role_emb.requires_grad_(True)
        adv_diffuser.model.requires_grad_(True)
        adv_diffuser.sigma_gmm_scale = 10
        adv_diffuser.train()

        return adv_diffuser

    def build_actor_critic(
        self,
        adv_diffuser: torch.nn.Module,
    ) -> Tuple[DiffusionActor, ValueCritic]:
        actor = DiffusionActor(adv_diffuser).to(self.device)

        ckpt = self.eval_cfg.critic
        critic = ValueCritic(adv_diffuser).to(self.device)
        if ckpt is not None:
            critic.load_state_dict(torch.load(ckpt)["critic"])
        return actor, critic

    def build_ppo_policy(
        self,
        actor: DiffusionActor,
        critic: ValueCritic,
        config: PPOTrainingConfig,
    ) -> DiffusionPPOPolicy:
        trainable_params = filter(
            lambda p: p.requires_grad,
            chain(actor.parameters(), critic.parameters()),
        )
        optim = torch.optim.Adam(trainable_params, lr=config.learning_rate)

        return DiffusionPPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=D.Normal,
            eta=0.3,
            detach_context=False,
            discount_factor=config.gamma,
            gae_lambda=config.gae_lambda,
            max_grad_norm=config.max_grad_norm,
            eps_clip=config.clip_ratio,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            action_scaling=False,
            action_bound_method="",
        )


class EnvironmentFactory:
    """Create training and evaluation environments."""

    def __init__(
        self,
        eval_cfg,
        exp_config,
        target_policy,
        others_policy,
        adv_diffuser: torch.nn.Module,
        device: torch.device,
        n_steps_per_call: int = 10,
        train_scenes_slice: Optional[list] = None,
    ):
        self.eval_cfg = eval_cfg
        self.exp_config = exp_config
        self.target_policy = target_policy
        self.others_policy = others_policy
        self.adv_diffuser = adv_diffuser
        self.device = device
        self.n_steps_per_call = n_steps_per_call

        all_scenes = eval_cfg.eval_scenes
        if train_scenes_slice is not None:
            s, e = int(train_scenes_slice[0]), int(train_scenes_slice[1])
            self.valid_scenes = all_scenes[s:e]
        else:
            self.valid_scenes = all_scenes

        self.env_builder = EnvCLDBuilder(
            eval_config=eval_cfg,
            exp_config=exp_config,
            device=device,
            log_data=False,
        )
        self.target_adv_pairs = eval_cfg.target_adv_pairs

    def make_env(self, seed: int) -> PPOEnv:
        return PPOEnv(
            cfg=self.exp_config.algo,
            sim_env=self.env_builder.get_env(),
            target_policy=self.target_policy,
            others_policy=self.others_policy,
            adv_diffuser=self.adv_diffuser,
            target_adv_pairs=self.target_adv_pairs,
            valid_scenes=self.valid_scenes,
            scene_offset=self.eval_cfg.scene_offset,
            seed=seed,
            n_steps_per_call=self.n_steps_per_call,
            horizon=self.eval_cfg.num_simulation_steps,
        )

    def create_vector_envs(
        self,
        num_envs: int,
        base_seed: int,
        is_test: bool = False,
    ) -> DummyVectorEnv:
        seed_offset = 10000 if is_test else 0
        env_fns = [
            lambda s=base_seed + seed_offset + i: self.make_env(s)
            for i in range(num_envs)
        ]
        return DummyVectorEnv(env_fns)


class CustomWandbLogger(WandbLogger):
    """WandB logger that also records environment metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._episode_metrics = []

    def add_episode_info(self, info: dict):
        if info:
            self._episode_metrics.append(info)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        super().log_train_data(collect_result, step)

        if self._episode_metrics and collect_result.get("n/ep", 0) > 0:
            custom_metrics = {}

            d_final_list = [m["d_end"] for m in self._episode_metrics if "d_end" in m]
            l_final_list = [m["abs_l"] for m in self._episode_metrics if "abs_l" in m]
            lateral_list = [
                m["lateral_dist_world"]
                for m in self._episode_metrics
                if "lateral_dist_world" in m
            ]
            collision_list = [
                m["collision"] for m in self._episode_metrics if "collision" in m
            ]

            if d_final_list:
                custom_metrics["train/d_final"] = np.mean(d_final_list)
            if l_final_list:
                custom_metrics["train/l_final"] = np.mean(l_final_list)
            if lateral_list:
                custom_metrics["train/lateral_dist_world"] = np.mean(lateral_list)
            if collision_list:
                custom_metrics["train/collision_rate"] = np.mean(collision_list)

            if custom_metrics:
                wandb.log(custom_metrics, step=step)

            self._episode_metrics = []


def make_preprocess_fn(logger: CustomWandbLogger):
    def preprocess_fn(obs_next=None, rew=None, done=None, info=None, **kwargs):
        if info is not None and done is not None:
            for i, d in enumerate(done):
                if d:
                    info_dict = {}
                    for key in info.keys():
                        if key == "env_id":
                            continue
                        try:
                            val = info[key][i]
                            info_dict[key] = float(val)
                        except (TypeError, IndexError, KeyError):
                            pass

                    if info_dict:
                        logger.add_episode_info(info_dict)
        return {}

    return preprocess_fn


class LoggerSetup:
    """Configure experiment logging."""

    def __init__(self, config: PPOTrainingConfig, results_dir: str):
        self.config = config
        self.results_dir = Path(results_dir)

    def setup_logger(self, exp_config=None) -> CustomWandbLogger:
        apikey = os.environ.get("WANDB_APIKEY") or os.environ.get("WANDB_API_KEY")
        if not apikey:
            raise SystemExit(
                "WANDB logging requested (cfg.use_logger=True) but neither "
                "WANDB_APIKEY nor WANDB_API_KEY env var is set.\n"
                "Either:\n"
                "  - export WANDB_APIKEY=<your-key>\n"
                "  - set use_logger: false in your --config_file"
            )
        wandb.login(key=apikey)

        logger = CustomWandbLogger(
            train_interval=1,
            update_interval=1,
            test_interval=1,
            save_interval=1,
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_run_name,
            run_id=self.config.wandb_id,
        )

        writer = SummaryWriter(str(self.results_dir))
        if exp_config is not None:
            writer.add_text("config", str(exp_config))
        logger.load(writer)

        return logger

    def get_save_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = self.config.wandb_run_name or "PPO_Run"
        save_dir = self.results_dir / "ppo_runs" / f"{run_name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir


class PPOTrainer:
    """PPO training pipeline."""

    def __init__(self, eval_cfg):
        self.eval_cfg = eval_cfg
        self.config = PPOTrainingConfig.from_eval_cfg(eval_cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _setup_global_config(self):
        set_global_batch_type("trajdata")
        set_global_trajdata_batch_env(self.eval_cfg.trajdata_source_test[0])

    def _set_seeds(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _setup_directories(self):
        results_dir = Path(self.eval_cfg.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to {results_dir}")

    def train(self):
        self._setup_global_config()
        self._set_seeds(self.eval_cfg.seed)
        self._setup_directories()

        model_builder = ModelBuilder(self.eval_cfg, self.device)
        target_policy, adv_policy, others_policy, exp_config = (
            model_builder.build_policies()
        )

        datamodule = datamodule_factory(
            cls_name=exp_config.train.datamodule_class,
            config=exp_config,
        )
        datamodule.setup("fit")
        set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)

        adv_diffuser = model_builder.build_adv_diffuser(
            target_policy,
            adv_policy,
            others_policy,
        )

        actor, critic = model_builder.build_actor_critic(adv_diffuser)
        policy = model_builder.build_ppo_policy(actor, critic, self.config)

        env_factory = EnvironmentFactory(
            self.eval_cfg,
            exp_config,
            target_policy,
            others_policy,
            adv_diffuser,
            self.device,
            n_steps_per_call=5,
            train_scenes_slice=self.config.train_scenes_slice,
        )
        train_envs = env_factory.create_vector_envs(
            num_envs=self.config.num_train_envs,
            base_seed=self.eval_cfg.seed,
            is_test=False,
        )
        test_envs = env_factory.create_vector_envs(
            num_envs=1,
            base_seed=self.eval_cfg.seed,
            is_test=True,
        )

        step_per_collect = (
            self.config.scenes_per_collect
            * self.config.num_train_envs
            * self.eval_cfg.num_simulation_steps
            // self.eval_cfg.n_step_action
        )

        train_buffer = VectorReplayBuffer(
            total_size=step_per_collect,
            buffer_num=self.config.num_train_envs,
        )

        if self.eval_cfg.use_logger:
            logger_setup = LoggerSetup(self.config, self.eval_cfg.results_dir)
            logger = logger_setup.setup_logger(exp_config)
            save_dir = str(logger_setup.get_save_dir())
            save_best_fn = build_save_best_fn(save_dir, self.eval_cfg)
            save_checkpoint_fn = build_save_checkpoint_fn(save_dir, policy, self.eval_cfg)
            preprocess_fn = make_preprocess_fn(logger)
        else:
            logger = LazyLogger()
            save_best_fn = None
            save_checkpoint_fn = None
            preprocess_fn = None

        train_collector = Collector(
            policy,
            train_envs,
            buffer=train_buffer,
            preprocess_fn=preprocess_fn,
        )
        test_collector = Collector(policy, test_envs)

        result = onpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=self.config.ppo_epochs,
            repeat_per_collect=self.config.update_per_collect,
            step_per_epoch=step_per_collect,
            episode_per_collect=self.config.scenes_per_collect,
            episode_per_test=self.config.test_episodes,
            batch_size=self.config.mini_batch_size,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            logger=logger,
            show_progress=True,
        )

        print(f"Finished training! Best reward: {result['best_reward']}")
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO Training for Adversarial Agent Generation"
    )

    group = parser.add_argument_group("Training Options")

    for name, kwargs in {
        "--config_file": dict(
            type=str,
            default=None,
            help="JSON configuration file path",
        ),
        "--seed": dict(type=int, default=None, help="Random seed"),
        "--registered_name": dict(
            type=str,
            default="trajdata_nusc_diff",
            help="Registered config name",
        ),
        "--debug": dict(action="store_true", help="Enable debug mode"),
    }.items():
        group.add_argument(name, **kwargs)

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> SceneEditingConfig:
    cfg = SceneEditingConfig(registered_name=args.registered_name)

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    if args.seed is not None:
        cfg.seed = args.seed
    cfg.debug = args.debug

    for k in cfg[cfg.env]:
        cfg[k] = cfg[cfg.env][k]

    cfg.pop("nusc")
    cfg.pop("trajdata")
    cfg.lock()

    return cfg


def main():
    args = parse_args()
    cfg = load_config(args)
    # PPO trains the adversary as a single sample distribution (no multi-modal
    # sampling at training time). Rollout-time multi-sampling is configured
    # separately via simulation_config.json.
    cfg.policy.num_action_samples = 1
    trainer = PPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
