"""Stage 1: GMM diffusion policy training (nuScenes)."""
import argparse
import os
import sys

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

import tbsim.utils.sc_train_utils as TrainUtils
from tbsim.algos.factory import algo_factory
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.utils.log_utils import PrintLogger
from tbsim.utils.safety_critical_batch_utils import (
    set_global_trajdata_batch_raster_cfg,
)
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and validate mutually-required combinations."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help=(
            "(optional) path to a config JSON that overrides the default "
            "settings. This is the preferred way to run experiments."
        ),
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="(optional) create experiment config from a preregistered name (see configs/registry.py)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help="(optional) if provided, override the wandb project name defined in the config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory of training output (checkpoints, visualization, tensorboard log, etc.)",
    )
    parser.add_argument(
        "--remove_exp_dir",
        action="store_true",
        help=(
            "Whether to automatically remove an existing experiment directory "
            "with the same name."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode, suppress wandb logging, etc.",
    )
    parser.add_argument(
        "--source_train",
        type=str,
        default=None,
        help="Trajdata source train",
    )
    parser.add_argument(
        "--source_valid",
        type=str,
        default=None,
        help="Trajdata source valid",
    )
    parser.add_argument(
        "--datamodule_class",
        type=str,
        default="CLDDataModule",
        help="Trajdata datamodule class",
    )
    parser.add_argument(
        "--trajdata_cache_location",
        type=str,
        default=None,
        help="Trajdata cache location",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="Wandb run ID to resume logging",
    )
    parser.add_argument(
        "--num_Gaussian",
        type=int,
        default=None,
        help="Number of Gaussian mixture components",
    )
    parser.add_argument(
        "--training_num_steps",
        type=int,
        default=None,
        help="Number of training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    args = parser.parse_args()
    if args.config_name is None and args.config_file is None:
        parser.error("Need either --config_name or --config_file to build the training config")
    return args


def build_config(args: argparse.Namespace):
    """Load a base experiment config and apply CLI overrides; returns locked config."""
    if args.config_name is not None:
        cfg = get_registered_experiment_config(args.config_name)
    else:
        cfg = get_experiment_config_from_file(args.config_file, locked=False)

    if args.name is not None:
        cfg.name = args.name
    if args.output_dir is not None:
        cfg.root_dir = os.path.abspath(args.output_dir)
    if args.wandb_project_name is not None:
        cfg.train.logging.wandb_project_name = args.wandb_project_name
    if args.source_train is not None:
        cfg.train.trajdata_source_train = [args.source_train]
    if args.source_valid is not None:
        cfg.train.trajdata_source_valid = [args.source_valid]
    if args.training_num_steps is not None:
        cfg.train.training.num_steps = args.training_num_steps
    if args.trajdata_cache_location is not None:
        cfg.train.trajdata_cache_location = args.trajdata_cache_location
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_Gaussian is not None:
        cfg.algo.num_Gaussian = args.num_Gaussian
    if args.datamodule_class is not None:
        cfg.train.datamodule_class = args.datamodule_class

    if args.debug:
        cfg.train.validation.every_n_steps = 5
        cfg.train.save.every_n_steps = 10
        cfg.train.rollout.every_n_steps = 10
        cfg.train.rollout.num_episodes = 1
        cfg.train.validation.num_steps_per_epoch = 5

    cfg.lock()
    return cfg


def main(cfg, args, auto_remove_exp_dir=False, debug=False):
    """GMM diffusion policy training body."""
    # Process-level configuration. Kept inside main() so importing this module
    # is side-effect-free; the env-var must be set before wandb.init() but not
    # before `import wandb`, so this position is safe.
    os.environ["WANDB_DISABLE_CODE"] = "True"
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    pl.seed_everything(cfg.seed)
    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(cfg.train.trajdata_source_train[0])
    set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer)

    print("\n============= New Training Run with Config =============")
    root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
        exp_name=cfg.name,
        output_dir=cfg.root_dir,
        save_checkpoints=cfg.train.save.enabled,
        auto_remove_exp_dir=auto_remove_exp_dir,
    )

    cfg.dump(os.path.join(root_dir, version_key, "config.json"))

    if cfg.train.logging.terminal_output_to_txt and not debug:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    train_callbacks = []

    assert cfg.train.parallel_strategy in [
        "dp",
        "ddp_spawn",
        None,
    ]
    if cfg.devices.num_gpus <= 1:
        with cfg.train.unlocked():
            cfg.train.parallel_strategy = None

    datamodule = datamodule_factory(cls_name=cfg.train.datamodule_class, config=cfg)
    model = algo_factory(config=cfg, modality_shapes=datamodule.modality_shapes)

    ckpt_fixed_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="iter{step}-val_loss{val_loss:.4f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        every_n_train_steps=cfg.train.save.every_n_steps,
        verbose=True,
    )
    train_callbacks.append(ckpt_fixed_callback)

    ckpt_latest_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="latest",
        save_top_k=1,
        every_n_train_steps=cfg.train.save.every_n_steps,
        verbose=False,
    )
    train_callbacks.append(ckpt_latest_callback)

    logger = None
    if debug:
        print("Debugging mode, suppress logging.")
    elif cfg.train.logging.log_wandb:
        apikey = os.environ.get("WANDB_APIKEY") or os.environ.get("WANDB_API_KEY")
        if not apikey:
            raise SystemExit(
                "WANDB logging requested (cfg.train.logging.log_wandb=True) but "
                "neither WANDB_APIKEY nor WANDB_API_KEY env var is set.\n"
                "Either:\n"
                "  - export WANDB_APIKEY=<your-key>\n"
                "  - set log_wandb: false in your config\n"
                "  - pass --debug to suppress logging"
            )
        wandb.login(key=apikey)
        wandb_kwargs = dict(
            name=cfg.name,
            project=cfg.train.logging.wandb_project_name,
            log_model=True,
        )
        if args.wandb_run_id is not None:
            wandb_kwargs["id"] = args.wandb_run_id
            wandb_kwargs["resume"] = "must"
        logger = WandbLogger(**wandb_kwargs)
        logger.experiment.config.update(cfg.to_dict(), allow_val_change=True)
        logger.watch(model=model)
    else:
        print("WARNING: not logging training stats")

    trainer = pl.Trainer(
        default_root_dir=root_dir,
        enable_checkpointing=cfg.train.save.enabled,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        log_every_n_steps=cfg.train.logging.log_every_n_steps,
        max_steps=cfg.train.training.num_steps,
        val_check_interval=cfg.train.validation.every_n_steps,
        limit_val_batches=cfg.train.validation.num_steps_per_epoch,
        callbacks=train_callbacks,
        precision="16-mixed",
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=args.resume_ckpt)


if __name__ == "__main__":
    args = parse_args()
    cfg = build_config(args)
    main(cfg, args=args, auto_remove_exp_dir=args.remove_exp_dir, debug=args.debug)
