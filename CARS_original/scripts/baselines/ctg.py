"""CTG (DiffuserTrafficModel) target + GMD adversary baseline rollout.

Key differences from scripts.rollout.nusc:
  1. CTG target uses DiffuserTrafficModel (state-space diffusion, include_hist=True,
     history_num_frames=30).
  2. GMD adversary/predictor use PPO_Diffusion_Trainer (action-space, include_hist=False,
     history_num_frames=20).
  3. The env is built with history_sec=3.0 (30 frames) + include_hist=True so CTG
     gets real rasterized history (34ch image).
  4. GMDObsAdapter wraps GMD agents to:
       - strip image to last num_sem_layers channels (3ch sem map)
       - truncate history sequences to last (history_num_frames+1) frames (21 frames)
  5. sigma_gmm_scale is set only on models that have this GMD-specific attribute.

Usage:
    python -m scripts.baselines ctg --config_file configs/simulation_config.json
"""
import argparse
import numpy as np
import json
import random
import importlib
import os
import torch

from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env
from tbsim.utils.safety_critical_batch_utils import set_global_trajdata_batch_raster_cfg
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.evaluation.env_builders import EnvCLDBuilder

from scripts.rollout.nusc import run_rollout_for_scenes


# History keys with time dim at axis=1
_CENTER_HIST_KEYS = [
    "history_positions", "history_speeds", "history_yaws",
    "extent", "history_availabilities"
]
# History keys with time dim at axis=2 (neighbor dimension at axis=1)
_NEIGH_HIST_KEYS = [
   "all_other_agents_history_positions", "all_other_agents_history_speeds",
   "all_other_agents_history_yaws", "all_other_agents_history_extents",
   "all_other_agents_history_availability",
  
]


class GMDObsAdapter:
    """Adapts observations for GMD (PPO_Diffusion_Trainer) when the env is built
    with CTG's raster config (include_hist=True, history_sec=3.0).

    Image layout with include_hist=True:
        [rasterized_history (31 ch) | semantic_map (3 ch)]  →  34 channels total
    GMD was trained with include_hist=False → expects only 3 semantic channels.

    History sequences from the env have 31 frames (CTG's history_num_frames=30+1).
    GMD was trained with history_num_frames=20 → expects 21 frames.

    This adapter:
      1. Strips the image to the last num_sem_layers channels (semantic map only).
      2. Truncates history sequences to the last hist_frames frames.
    """
    def __init__(self, policy, num_sem_layers: int, hist_frames: int, future_frames: int):
        self.policy = policy
        self.num_sem_layers = num_sem_layers
        self.hist_frames = hist_frames      # GMD's expected history frames (21)
        self.model = policy.model
        self.future_frames = future_frames
    def get_action(self, obs, **kwargs):
        obs = dict(obs)
        # 1. Strip image: keep only semantic map channels (last num_sem_layers)
        obs['image'] = obs['image'][:, -self.num_sem_layers:, :, :]
        # 2. Truncate center-agent history sequences (time dim = 1)
        for k in _CENTER_HIST_KEYS:
            if k in obs and obs[k].shape[1] > self.hist_frames:
                obs[k] = obs[k][:, -self.hist_frames:, ...]
        # 3. Truncate neighbor history sequences (time dim = 2)
        for k in _NEIGH_HIST_KEYS:
            if k in obs and obs[k].shape[2] > self.hist_frames:
                obs[k] = obs[k][:, :, -self.hist_frames:, ...]
        if 'target_positions' in obs:
            obs['target_positions'] = obs['target_positions'][:, :self.future_frames,...]
     
        return self.policy.get_action(obs, **kwargs)

    def __getattr__(self, name):
        return getattr(self.policy, name)


def run_scene_editor_ctg(eval_cfg, save_cfg, data_to_disk, render_to_video, render_to_img,
                         render_cfg, save_all=False):
    assert eval_cfg.env in ["nusc", "trajdata"]

    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])

    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)

    print('saving results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)
    if render_to_video or render_to_img:
        os.makedirs(os.path.join(eval_cfg.results_dir, "viz/"), exist_ok=True)
    if save_cfg:
        json.dump(eval_cfg, open(os.path.join(eval_cfg.results_dir, "config.json"), "w+"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")

    # CTG is target, GMD is adversary + predictor.
    adv_eval_class    = getattr(eval_cfg, "eval_class", "PPO_Diffuser")
    target_eval_class = "CTG_Diffuser"

    target_composer_class = getattr(policy_composers, target_eval_class)
    adv_composer_class    = getattr(policy_composers, adv_eval_class)

    print(f"[rollout_ctg] target   composer: {target_eval_class}")
    print(f"[rollout_ctg] adversary composer: {adv_eval_class}")

    ego_composer                      = target_composer_class(eval_cfg, device)
    target_policy, target_exp_config  = ego_composer.get_policy()

    agents_composer                   = adv_composer_class(eval_cfg, device)
    agents_planner, adv_exp_config    = agents_composer.get_planner()

    others_composer                   = target_composer_class(eval_cfg, device)
    others_predictor, _               = others_composer.get_predictor()

    ctg_exp_config = target_exp_config
    gmd_exp_config = adv_exp_config

    # Use CTG's raster config: include_hist=True produces [history | sem_map] images.
    # Also use CTG's exp_config for env builder so history_num_frames=30 → 34ch raster.
    set_global_trajdata_batch_raster_cfg(ctg_exp_config.env.rasterizer)
    exp_config = ctg_exp_config

    ctg_hist_frames = ctg_exp_config.algo.history_num_frames + 1   # 31
    ctg_sem_layers  = ctg_exp_config.env.rasterizer.num_sem_layers  # 3
    print(f"[rollout_ctg] CTG image: {ctg_hist_frames + ctg_sem_layers}ch "
          f"(hist={ctg_hist_frames}, sem={ctg_sem_layers})")

    # GMD: strip image to 3ch sem map + truncate history to 21 frames
    gmd_sem_layers  = gmd_exp_config.env.rasterizer.num_sem_layers       # 3
    gmd_hist_frames = 20 + 1         # 21
    gmd_future_frames = 20
    print(f"[rollout_ctg] GMD adapted: image→{gmd_sem_layers}ch, "
          f"history→{gmd_hist_frames} frames")

    # Wrap GMD policies (planner + predictor) with the obs adapter.
    agents_planner   = GMDObsAdapter(agents_planner,   gmd_sem_layers, gmd_hist_frames, gmd_future_frames)
    others_predictor = GMDObsAdapter(others_predictor, gmd_sem_layers, gmd_hist_frames, gmd_future_frames)

    policy_model = None
    if hasattr(target_policy, 'model'):
        policy_model = target_policy.model
    print('target policy:', target_policy)

    # sigma_gmm_scale is GMD-specific; CTG's DiffuserModel does not have it
    for pol in [target_policy, others_predictor, agents_planner]:
        pol.model.nets['policy'].eval()
        pol.model.nets['policy'].requires_grad_(False)
        if hasattr(pol.model.nets['policy'], 'sigma_gmm_scale'):
            pol.model.nets['policy'].sigma_gmm_scale = 3
        # Prevent backward driving: override vbound lower bound to 0
        # (models may be trained with vbound=[-10,30] but real cars don't reverse)
        # CTG uses EMA at inference, so set vbound on BOTH nets['policy'] AND ema_policy
        targets_to_patch = [pol.model.nets['policy']]
        if hasattr(pol.model, 'ema_policy'):
            targets_to_patch.append(pol.model.ema_policy)
        for tgt in targets_to_patch:
            dyn = getattr(tgt, 'dyn', None)
            if dyn is not None and hasattr(dyn, 'vbound'):
                print(f"  [vbound] {type(tgt).__name__} ({type(pol.model).__name__}): {dyn.vbound} -> ", end="")
                dyn.vbound[0] = 0.0
                print(f"{dyn.vbound}")

    env_builder = EnvCLDBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device, log_data=True)
    env = env_builder.get_env()

    render_rasterizer = None
    if render_to_video or render_to_img:
        from tbsim.utils.scene_edit_utils import get_trajdata_renderer
        render_rasterizer = get_trajdata_renderer(
            eval_cfg.trajdata_source_train,
            eval_cfg.trajdata_data_dirs,
            future_sec=eval_cfg.future_sec,
            history_sec=eval_cfg.history_sec,
            raster_size=render_cfg['size'],
            px_per_m=render_cfg['px_per_m'],
            rebuild_maps=False,
            cache_location=eval_cfg.trajdata_cache_location,
        )

    target_adv_pairs  = eval_cfg.target_adv_pairs
    valid_scene_names = set(target_adv_pairs.keys())
    print(f"[Mode: target_adv_pairs] {len(valid_scene_names)} scenes")

    for scene_i in eval_cfg.eval_scenes:
        start_frame_index = [exp_config.algo.history_num_frames + 1]
        env.reset(scene_indices=[scene_i], start_frame_index=start_frame_index)

        scene_name = env.current_scene_names[0]
        if scene_name not in valid_scene_names:
            print(f"  Skipping {scene_name} (not in target_adv_pairs)")
            continue

        pairs = target_adv_pairs[scene_name]
        print(f"Scene {scene_name} is {env.current_scene_index[0]} has {len(pairs)} target-adv pair(s)")

        for pair_idx, pair in enumerate(pairs):
            target_name = pair[0]
            print(f"  pair {pair_idx + 1}/{len(pairs)}: target={target_name}")
            env.reset(scene_indices=[scene_i], start_frame_index=start_frame_index)

            # Verify the target agent is present in the scene at start_frame.
            # Agents in target_adv_pairs may have been extracted from later frames
            # or from a different rollout configuration; skip gracefully if absent.
            scene_agent_names = env.current_agent_names
            if target_name not in scene_agent_names:
                print(f"  [SKIP] target '{target_name}' not in scene at start_frame={start_frame_index[0]}. "
                      f"Available: {scene_agent_names}")
                continue

            run_rollout_for_scenes(
                scene_i, env, target_policy, agents_planner, others_predictor, policy_model,
                eval_cfg, exp_config, render_rasterizer, render_to_video, render_to_img,
                render_cfg, data_to_disk,
                target_name=target_name, pair_idx=pair_idx, save_all=save_all,
            )


def add_args(parser: argparse.ArgumentParser) -> None:
    for name, kwargs in {
        "--config_file":         dict(type=str, default=None),
        "--env":                 dict(type=str, choices=["nusc", "trajdata"], default="trajdata"),
        "--results_root_dir":    dict(type=str, default=None),
        "--seed":                dict(type=int, default=None),
        "--render":              dict(action="store_true", default=True),
        "--no-render":           dict(action="store_false", dest="render"),
        "--render_img":          dict(action="store_true", default=False),
        "--render_size":         dict(type=int, default=400),
        "--render_px_per_m":     dict(type=float, default=2.0),
        "--save_every_n_frames": dict(type=int, default=1),
        "--draw_mode":           dict(type=str, default="action"),
        "--registered_name":     dict(type=str, default="trajdata_nusc_diff"),
        "--save_all":            dict(action="store_true", default=False,
                                     help="Save all episodes including non-collision ones"),
        "--num_scenes":          dict(type=int, default=None, help="Limit number of scenes (debug)"),
        "--skip_scenes":         dict(type=int, default=0, help="Skip the first N scenes"),
    }.items():
        parser.add_argument(name, **kwargs)


def run(args) -> None:
    cfg = SceneEditingConfig(registered_name=args.registered_name)

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    for attr in ("seed", "env"):
        if getattr(args, attr) is not None:
            setattr(cfg, attr, getattr(args, attr))

    if cfg.name is None:
        cfg.name = "ctg_target"
    if args.results_root_dir is not None:
        cfg.results_dir = args.results_root_dir
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)
    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    for k in cfg[cfg.env]:
        cfg[k] = cfg[cfg.env][k]
    cfg.pop("nusc")
    cfg.pop("trajdata")

    # Override history_sec to match CTG's history_num_frames=30 requirement.
    # This makes the env collect 30 frames of real history for CTG.
    # GMDObsAdapter will truncate to the last 21 frames for GMD agents.
    ctg_history_sec = 3.0   # CTG: history_num_frames=30, dt=0.1s
    print(f"[rollout_ctg] Overriding history_sec: "
          f"{cfg.history_sec} → {ctg_history_sec} (CTG requirement)")
    cfg.history_sec = ctg_history_sec

    render_cfg = {
        'size': args.render_size,
        'px_per_m': args.render_px_per_m,
        'save_every_n_frames': args.save_every_n_frames,
        'draw_mode': args.draw_mode,
    }

    if args.skip_scenes > 0:
        cfg.eval_scenes = cfg.eval_scenes[args.skip_scenes:]
    if args.num_scenes is not None:
        cfg.eval_scenes = cfg.eval_scenes[:args.num_scenes]

    cfg.lock()
    run_scene_editor_ctg(
        cfg,
        save_cfg=False,
        data_to_disk=True,
        render_to_video=args.render,
        render_to_img=args.render_img,
        render_cfg=render_cfg,
        save_all=args.save_all,
    )


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="CTG baseline rollout")
    add_args(_parser)
    run(_parser.parse_args())
