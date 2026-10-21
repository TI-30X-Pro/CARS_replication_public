"""Main nuScenes closed-loop CARS rollout."""
import argparse

import numpy as np
import json
import random
import yaml
import importlib


import os
import pickle
import torch

from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env
from tbsim.utils.safety_critical_batch_utils import set_global_trajdata_batch_raster_cfg
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.utils.scene_edit_utils import safety_critical_guided_rollout
from tbsim.policies.decision_tree_wrapper import safety_critical_guided_rollout_with_custom_target
from tbsim.evaluation.env_builders import EnvCLDBuilder



from tbsim.policies.decision_tree_wrapper import TargetAdvOthersWrapper
from tbsim.policies.wrappers import RolloutWrapper
import tbsim.utils.tensor_utils as TensorUtils
# import tbsim.utils.tensor_utils as TensorUtils

def run_scene_editor(eval_cfg, save_cfg, data_to_disk, render_to_video, render_to_img, render_cfg, save_all=False):
    assert eval_cfg.env in ["nusc", "trajdata"], "Currently only nusc and trajdata environments are supported"
        
    set_global_batch_type("trajdata")

    set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])
    

    # print(eval_cfg)

    # for reproducibility
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)
    # basic setup
    print('saving results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)

    if render_to_video or render_to_img:
        os.makedirs(os.path.join(eval_cfg.results_dir, "viz/"), exist_ok=True)
    if save_cfg:
        json.dump(eval_cfg, open(os.path.join(eval_cfg.results_dir, "config.json"), "w+"))
    # if data_to_disk and os.path.exists(eval_cfg.experience_hdf5_path):
    #     os.remove(eval_cfg.experience_hdf5_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create policy and rollout wrapper
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    # composer = composer_class(eval_cfg, device)
    # policy, exp_config = composer.get_policy()

    ego_composer = composer_class(eval_cfg, device)
    target_policy, exp_config = ego_composer.get_policy()

    # ego_policy = get_ego_policy()#NOTE: use tuplan policy

    agents_composer = composer_class(eval_cfg, device)
    agents_planner, exp_config = agents_composer.get_planner()

    others_composer = composer_class(eval_cfg, device)
    others_predictor, exp_config = others_composer.get_predictor()
    
    # determines cfg for rasterizing agents
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)
    
    # print(exp_config.algo)
    # ----------------------------------------------------------------------------------
    policy_model = None
    print('policy', target_policy)
    if hasattr(target_policy, 'model'):
        policy_model = target_policy.model

    for policy in [target_policy, others_predictor, agents_planner]:
            policy.model.nets['policy'].eval()
            policy.model.nets['policy'].requires_grad_(False)
            # sigma_gmm_scale is PPO_Diffuser-specific (not present in CTG Diffuser)
            if hasattr(policy.model.nets['policy'], 'sigma_gmm_scale'):
                policy.model.nets['policy'].sigma_gmm_scale = 3
            # Prevent backward driving at inference time
            dyn = getattr(policy.model.nets['policy'], 'dyn', None)
            if dyn is not None and hasattr(dyn, 'vbound'):
                dyn.vbound[0] = 0.0

    env_builder = EnvCLDBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device, log_data=True)
    env = env_builder.get_env()





    render_rasterizer = None
    if render_to_video or render_to_img:
        from tbsim.utils.scene_edit_utils import get_trajdata_renderer
        # initialize rasterizer once for all scenes
        render_rasterizer = get_trajdata_renderer(eval_cfg.trajdata_source_train,
                                                  eval_cfg.trajdata_data_dirs,
                                                  future_sec=eval_cfg.future_sec,
                                                  history_sec=eval_cfg.history_sec,
                                                  raster_size=render_cfg['size'],
                                                  px_per_m=render_cfg['px_per_m'],
                                                  rebuild_maps=False,
                                                  cache_location=eval_cfg.trajdata_cache_location)

    use_target_adv_pairs = True

    if use_target_adv_pairs:
        target_adv_pairs = eval_cfg.target_adv_pairs
        valid_scene_names = set(target_adv_pairs.keys())
        print(f"[Mode: target_adv_pairs] Loaded {len(valid_scene_names)} valid scene names")

        for scene_i in eval_cfg.eval_scenes:
            start_frame_index = [exp_config.algo.history_num_frames + 1]
            env.reset(scene_indices=[scene_i], start_frame_index=start_frame_index)

            scene_name = env.current_scene_names[0]
            pairs = target_adv_pairs[scene_name]
            num_pairs = len(pairs)
            print(f"Scene {scene_name} is {env.current_scene_index[0]} has {num_pairs} target-adv pair(s)")

            for pair_idx, pair in enumerate(pairs):
                target_name = pair[0]
                print(f"  Processing pair {pair_idx + 1}/{num_pairs}: target={target_name}")
                env.reset(scene_indices=[scene_i], start_frame_index=start_frame_index)
                run_rollout_for_scenes(
                    scene_i, env, target_policy, agents_planner, others_predictor, policy_model,
                    eval_cfg, exp_config, render_rasterizer, render_to_video, render_to_img,
                    render_cfg, data_to_disk,
                    target_name=target_name, pair_idx=pair_idx, save_all=save_all,
                )
    else:
        print(f"[Mode: direct] Running rollout for {len(eval_cfg.eval_scenes)} scenes without target_adv_pairs")

        for scene_i in eval_cfg.eval_scenes:
            start_frame_index = [exp_config.algo.history_num_frames + 1]
            env.reset(scene_indices=[scene_i], start_frame_index=start_frame_index)

            scene_name = env.current_scene_names[0]
            print(f"Processing scene {scene_i}: {scene_name}")

            run_rollout_for_scenes(
                scene_i, env, target_policy, agents_planner, others_predictor, policy_model,
                eval_cfg, exp_config, render_rasterizer, render_to_video, render_to_img,
                render_cfg, data_to_disk,
                target_name=None, pair_idx=0, save_all=save_all,
            )

def run_rollout_for_scenes(scene_i, env, target_policy, agents_planner, others_predictor, policy_model,
                           eval_cfg, exp_config, render_rasterizer, render_to_video,
                           render_to_img, render_cfg, data_to_disk,
                           target_name=None, pair_idx=0, save_all=False):
    """Run a single closed-loop rollout for one scene and dump videos / HDF5."""
    start_frame_index = [exp_config.algo.history_num_frames + 1]
    

    
    model_path = "checkpoints/decision_tree_model.pkl"
    with open(model_path, 'rb') as f:
        decision_tree_model = pickle.load(f)
    

    if target_name is not None:
        target_adv_others_policy = TargetAdvOthersWrapper(
            target_policy=target_policy,
            adv_policy=agents_planner,
            others_predictor=others_predictor,
            decision_tree_model=decision_tree_model,
            target_name=target_name,
            enable_tree_viz=eval_cfg.decisioin_tree_visualization,
        )
        
       

        info = safety_critical_guided_rollout_with_custom_target(
            env,
            target_adv_others_policy,        
            n_step_action=eval_cfg.n_step_action,
            scene_indices=[scene_i],
            horizon=eval_cfg.num_simulation_steps,
            start_frames=start_frame_index,
        )
   
    else:
        rollout_policy = RolloutWrapper(ego_policy=target_policy, agents_policy=agents_planner, pass_agent_obs=False)
        
        stats, info, renderings, adv_name_ret = safety_critical_guided_rollout(
            env,
            rollout_policy,
            policy_model,
            n_step_action=eval_cfg.n_step_action,
            render=False,
            scene_indices=[scene_i],
            obs_to_torch=True,
            horizon=eval_cfg.num_simulation_steps,
            start_frames=start_frame_index,
        )
    had_collision = info.get("had_collision", [True])[0]

    agent_names = env.current_agent_names
    target_idx = None
    if target_name is not None and agent_names is not None:
        if target_name in agent_names:
            target_idx = agent_names.index(target_name)

    if (render_to_video or render_to_img) and had_collision:
        from tbsim.utils.scene_edit_utils import visualize_guided_rollout
        scene_cnt = 0
        for si, scene_buffer in zip(info["scene_index"], info["buffer"]):
            viz_dir = os.path.join(eval_cfg.results_dir, "viz/")
            sim_num = start_frame_index[scene_cnt]
            visualize_guided_rollout(viz_dir, render_rasterizer, si, scene_buffer,
                                        guidance_config=None,
                                        constraint_config=None,
                                        fps=(1.0 / exp_config.algo.step_time),
                                        n_step_action=eval_cfg.n_step_action,
                                        viz_diffusion_steps=False,
                                        first_frame_only=render_to_img,
                                        sim_num=sim_num,
                                        save_every_n_frames=render_cfg['save_every_n_frames'],
                                        draw_mode=render_cfg['draw_mode'],
                                        target_idx=target_idx,
                                        agent_names=agent_names,
                                        plot_decision_tree=eval_cfg.decisioin_tree_visualization,
                                        )
            scene_cnt += 1

    if data_to_disk and "buffer" in info and (had_collision or save_all):
        # Save target_name into each scene buffer for FSM replay
        if target_name is not None:
            for buf in info["buffer"]:
                buf["target_name"] = np.array([target_name])
        dump_episode_buffer(
            info["buffer"],
            info["scene_index"],
            start_frame_index,
            h5_path=eval_cfg.experience_hdf5_path,
            pair_idx=pair_idx,
            target_idx=target_idx,
        )
    elif data_to_disk and not had_collision and not save_all:
        print(f"  [SKIP] no collision, save_all=False")
    torch.cuda.empty_cache()

def dump_episode_buffer(buffer, scene_index, start_frames, h5_path, pair_idx=0, target_idx=None):
    import h5py, json
    import numpy as np

    def _as_utf8_array(x):
        """Convert a string-like container to a variable-length UTF-8 numpy array.

        Returns (arr, dtype) on success, or (None, None) if `x` is not a string
        type. Uses `dtype=object` rather than fixed-length 'S' so h5py writes
        proper variable-length UTF-8.
        """
        dt = h5py.string_dtype(encoding='utf-8')

        if isinstance(x, str):
            return np.array([x], dtype=object), dt

        if isinstance(x, np.ndarray):
            if x.dtype.kind in ('U', 'S'):
                return np.asarray(x.astype(str), dtype=object), dt
            if x.dtype == object:
                flat = []
                for v in x.ravel():
                    if isinstance(v, np.ndarray) and v.dtype.kind in ('U', 'S') and v.size == 1:
                        flat.append(str(v.reshape(-1)[0]))
                    elif isinstance(v, (str, bytes)):
                        flat.append(v.decode() if isinstance(v, bytes) else v)
                    else:
                        return None, None
                return np.array(flat, dtype=object), dt
            return None, None

        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return np.array([], dtype=object), dt
            flat = []
            for v in x:
                if isinstance(v, np.ndarray) and v.dtype.kind in ('U', 'S') and v.size == 1:
                    flat.append(str(v.reshape(-1)[0]))
                elif isinstance(v, (str, bytes)):
                    flat.append(v.decode() if isinstance(v, bytes) else v)
                else:
                    try:
                        flat.append(str(v))
                    except Exception:
                        return None, None
            return np.array(flat, dtype=object), dt

        return None, None

    with h5py.File(h5_path, "a") as h5f:
        for ei, si, scene_buf in zip(start_frames, scene_index, buffer):
            # Match video naming: {scene_name}_{sim_num:04d}_t_{target_idx}
            t_suffix = f"_t_{target_idx}" if target_idx is not None else f"_{pair_idx}"
            ep_key = f"{si}_{ei:04d}{t_suffix}"
            for mk, val in scene_buf.items():
                h5key = f"/{ep_key}/{mk}"

                if h5key in h5f:
                    del h5f[h5key]

                try:
                    h5f.create_dataset(h5key, data=val)
                    continue
                except TypeError:
                    pass

                arr, dt = _as_utf8_array(val)
                if arr is not None:
                    h5f.create_dataset(h5key, data=arr, dtype=dt)
                else:
                    # Last-resort JSON fallback so no field is silently dropped.
                    txt = json.dumps(val, default=str, ensure_ascii=False)
                    h5f.create_dataset(
                        h5key, data=np.array([txt], dtype=object),
                        dtype=h5py.string_dtype('utf-8'),
                    )

    print(f"  [SAVED] {ep_key} written to {h5_path}")



def add_args(parser: argparse.ArgumentParser) -> None:
    for name, kwargs in {
        "--config_file":        dict(type=str, default=None, help="A json file containing evaluation configs"),
        "--env":                dict(type=str, choices=["nusc", "trajdata"], default="trajdata", help="Which env to run editing in"),
        "--ckpt_yaml":          dict(type=str, default=None, help="Yaml file specifying checkpoint and config location of each model"),
        "--metric_ckpt_yaml":   dict(type=str, default=None, help="Yaml file specifying checkpoint and config location for the learned metric"),
        "--eval_class":         dict(type=str, default=None, help="Optionally specify the evaluation class through argparse"),
        "--policy_ckpt_dir":    dict(type=str, default=None, help="Directory to look for saved checkpoints"),
        "--policy_ckpt_key":    dict(type=str, default=None, help="Uniquely identifies a checkpoint file, e.g., iter50000"),
        "--planner_ckpt_dir":   dict(type=str, default=None, help="Directory to look for saved checkpoints"),
        "--planner_ckpt_key":   dict(type=str, default=None, help="Uniquely identifies a checkpoint file, e.g., iter50000"),
        "--predictor_ckpt_dir": dict(type=str, default=None, help="Directory to look for saved checkpoints"),
        "--predictor_ckpt_key": dict(type=str, default=None, help="Uniquely identifies a checkpoint file, e.g., iter50000"),
        "--results_root_dir":   dict(type=str, default=None, help="Directory to save results and videos"),
        "--dataset_path":       dict(type=str, default=None, help="Root directory of the dataset (default: from config)"),
        "--num_scenes_per_batch": dict(type=int, default=None, help="Number of scenes to run concurrently (to accelerate eval)"),
        "--render":             dict(action="store_true", default=True, help="Whether to render videos"),
        "--no-render":          dict(action="store_false", dest="render", help="Disable video rendering"),
        "--seed":               dict(type=int, default=None),
        "--prefix":             dict(type=str, default=None),
        "--registered_name":    dict(type=str, default='trajdata_nusc_diff'),
        "--render_img":         dict(action="store_true", default=False, help="Whether to only render the first frame of rollout"),
        "--render_size":        dict(type=int, default=400, help="Width and height of the rendered image size in pixels"),
        "--render_px_per_m":    dict(type=float, default=2.0, help="Resolution of rendering"),
        "--save_every_n_frames": dict(type=int, default=1, help="Saving videos while skipping every n frames"),
        "--draw_mode":          dict(type=str, default='action', help="['action', 'entire_traj', 'map']"),
        "--editing_source":     dict(type=str, choices=["config", "heuristic", "ui", "none"], nargs="+",
                                     help="Which edits to use. config/heuristic/ui/none. config and heuristic may be used together."),
        "--save_all":           dict(action="store_true", default=False,
                                     help="Save all episodes to HDF5 including non-collision ones (for baseline CR calculation)"),
        "--num_scenes":         dict(type=int, default=None, help="Limit number of scenes to run (for debugging)"),
        "--shard":              dict(type=int, default=None, help="Shard index (1-indexed). Splits eval_scenes across --num_shards shards."),
        "--num_shards":         dict(type=int, default=None, help="Total number of shards. Required together with --shard."),
    }.items():
        parser.add_argument(name, **kwargs)


def run(args) -> None:
    cfg = SceneEditingConfig(registered_name=args.registered_name)

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    for attr in ("eval_class", "num_scenes_per_batch", "dataset_path", "seed"):
        if getattr(args, attr) is not None:
            setattr(cfg, attr, getattr(args, attr))

    for role in ("policy", "planner", "predictor"):
        ckpt_dir = getattr(args, f"{role}_ckpt_dir")
        if ckpt_dir is not None:
            ckpt_key = getattr(args, f"{role}_ckpt_key")
            if role == "policy":
                assert ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
            getattr(cfg.ckpt, role).ckpt_dir = ckpt_dir
            getattr(cfg.ckpt, role).ckpt_key = ckpt_key

    if cfg.name is None:
        cfg.name = cfg.eval_class
    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name

    if args.results_root_dir is not None:
        cfg.results_dir = args.results_root_dir
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)

    assert args.env is not None or cfg.env is not None
    if args.env is not None:
        cfg.env = args.env

    if args.editing_source is not None:
        cfg.edits.editing_source = args.editing_source
    if not isinstance(cfg.edits.editing_source, list):
        cfg.edits.editing_source = [cfg.edits.editing_source]
    if "ui" in cfg.edits.editing_source:
        cfg.num_scenes_per_batch = 1

    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    for k in cfg[cfg.env]:  # copy env-specific config to the global-level
        cfg[k] = cfg[cfg.env][k]

    cfg.pop("nusc")
    cfg.pop("trajdata")

    if args.ckpt_yaml is not None:
        with open(args.ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    if args.metric_ckpt_yaml is not None:
        with open(args.metric_ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    
    render_cfg = {
        'size' : args.render_size,
        'px_per_m' : args.render_px_per_m,
        'save_every_n_frames': args.save_every_n_frames,
        'draw_mode': args.draw_mode,
    }

    if (args.shard is None) != (args.num_shards is None):
        raise ValueError("--shard and --num_shards must be specified together")
    if args.shard is not None:
        if not (1 <= args.shard <= args.num_shards):
            raise ValueError(f"--shard must be in 1..{args.num_shards} (got {args.shard})")
        n = len(cfg.eval_scenes)
        start = (n * (args.shard - 1)) // args.num_shards
        end = (n * args.shard) // args.num_shards
        print(f"[shard {args.shard}/{args.num_shards}] eval_scenes[{start}:{end}] = {end - start} of {n}")
        cfg.eval_scenes = cfg.eval_scenes[start:end]

    if args.num_scenes is not None:
        cfg.eval_scenes = cfg.eval_scenes[:args.num_scenes]

    cfg.lock()
    run_scene_editor(
        cfg,
        save_cfg=False,
        data_to_disk=True,
        render_to_video=args.render,
        render_to_img=args.render_img,
        render_cfg=render_cfg,
        save_all=args.save_all,
    )


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Main nuScenes closed-loop CARS rollout")
    add_args(_parser)
    run(_parser.parse_args())
