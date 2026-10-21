"""Reactive cubic-Bezier adversary baseline (closed-loop).

The adversary curve is re-fit every outer `get_action` call
(= every `n_step_action` sim steps) using the target's freshly-predicted
50-step plan as interception aim. Only collision episodes are saved.
Supports `--shard` / `--num_shards` for trivial multi-process sharding.
"""
import argparse
import importlib
import json
import os
import random

import numpy as np
import torch

from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env
from tbsim.utils.safety_critical_batch_utils import set_global_trajdata_batch_raster_cfg
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.policies.decision_tree_wrapper import safety_critical_guided_rollout_with_custom_target
from tbsim.evaluation.env_builders import EnvCLDBuilder

from ._bezier_wrapper import BezierReactiveAdvWrapper


def run_scene_editor(eval_cfg, shard_idx, num_shards, stop_after_collisions=None):
    assert eval_cfg.env in ["nusc", "trajdata"], \
        "Currently only nusc and trajdata environments are supported"

    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])

    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)

    os.makedirs(eval_cfg.results_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)

    ego_composer = composer_class(eval_cfg, device)
    target_policy, exp_config = ego_composer.get_policy()

    others_composer = composer_class(eval_cfg, device)
    others_predictor, _ = others_composer.get_predictor()

    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)

    for policy in (target_policy, others_predictor):
        policy.model.nets['policy'].eval()
        policy.model.nets['policy'].requires_grad_(False)
        if hasattr(policy.model.nets['policy'], 'sigma_gmm_scale'):
            policy.model.nets['policy'].sigma_gmm_scale = 3
        dyn = getattr(policy.model.nets['policy'], 'dyn', None)
        if dyn is not None and hasattr(dyn, 'vbound'):
            dyn.vbound[0] = 0.0

    env_builder = EnvCLDBuilder(eval_config=eval_cfg, exp_config=exp_config,
                                device=device, log_data=True)
    env = env_builder.get_env()

    # ------------------------------------------------------------------
    # Shard the scene list
    # ------------------------------------------------------------------
    all_scenes = list(eval_cfg.eval_scenes)
    if num_shards > 1:
        n = len(all_scenes)
        start = (n * (shard_idx - 1)) // num_shards
        end   = (n * shard_idx) // num_shards
        my_scenes = all_scenes[start:end]
        print(f"[Bezier-reactive shard {shard_idx}/{num_shards}] "
              f"eval_scenes[{start}:{end}] = {len(my_scenes)} of {n}")
    else:
        my_scenes = all_scenes
        print(f"[Bezier-reactive] {len(my_scenes)} scenes")

    target_adv_pairs = eval_cfg.target_adv_pairs
    num_sim_steps = int(eval_cfg.num_simulation_steps)

    collision_count = 0
    pair_total = 0
    pair_done = 0

    for scene_i in my_scenes:
        start_frame_index = [exp_config.algo.history_num_frames + 1]
        env.reset(scene_indices=[scene_i], start_frame_index=start_frame_index)
        scene_name = env.current_scene_names[0]

        if scene_name not in target_adv_pairs:
            continue
        pairs = target_adv_pairs[scene_name]
        print(f"Scene {scene_name} ({env.current_scene_index[0]}) "
              f"has {len(pairs)} target-adv pair(s)")

        for pair_idx, pair in enumerate(pairs):
            target_name, adv_name = pair[0], pair[1]
            pair_total += 1
            print(f"  pair {pair_idx + 1}/{len(pairs)}: "
                  f"target={target_name[:8]}... adv={adv_name[:8]}...")

            env.reset(scene_indices=[scene_i], start_frame_index=start_frame_index)

            # Quick existence check so we fail fast if either agent is not in obs
            obs0 = env.get_observation(split_ego=False)
            agent_names_now = list(obs0["agents"]["agent_name"])
            if target_name not in agent_names_now or adv_name not in agent_names_now:
                print(f"    [SKIP] target or adv not in current obs")
                continue

            wrapper = BezierReactiveAdvWrapper(
                target_policy=target_policy,
                others_predictor=others_predictor,
                target_name=target_name,
                adv_name=adv_name,
                debug=bool(os.environ.get("BEZIER_DEBUG")),
            )

            info = safety_critical_guided_rollout_with_custom_target(
                env,
                wrapper,
                n_step_action=eval_cfg.n_step_action,
                scene_indices=[scene_i],
                horizon=num_sim_steps,
                start_frames=start_frame_index,
            )

            had_collision = info.get("had_collision", [False])[0]
            agent_names_after = env.current_agent_names
            target_idx = (agent_names_after.index(target_name)
                          if agent_names_after is not None
                          and target_name in agent_names_after else None)
            pair_done += 1

            if "buffer" in info and had_collision:
                if target_name is not None:
                    for buf in info["buffer"]:
                        buf["target_name"] = np.array([target_name])
                dump_episode_buffer(
                    info["buffer"], info["scene_index"], start_frame_index,
                    h5_path=eval_cfg.experience_hdf5_path,
                    pair_idx=pair_idx, target_idx=target_idx,
                )
                collision_count += 1

            torch.cuda.empty_cache()

            if stop_after_collisions is not None and collision_count >= stop_after_collisions:
                print(f"[Bezier-reactive] reached stop_after_collisions, early stop")
                print(f"[Bezier-reactive] summary: {pair_done} pairs, {collision_count} collisions")
                return

    print(f"[Bezier-reactive] done: {pair_done}/{pair_total} pairs, "
          f"{collision_count} collisions")


def dump_episode_buffer(buffer, scene_index, start_frames, h5_path,
                         pair_idx=0, target_idx=None):
    """HDF5 dumper (string-safe). Copy of rollout_bezier.py's dumper."""
    import h5py

    def _as_utf8_array(x):
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
                    txt = json.dumps(val, default=str, ensure_ascii=False)
                    h5f.create_dataset(
                        h5key, data=np.array([txt], dtype=object),
                        dtype=h5py.string_dtype('utf-8'),
                    )
    print(f"  [SAVED] {ep_key} written to {h5_path}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config_file",      type=str, default=None)
    parser.add_argument("--env",              type=str, choices=["nusc", "trajdata"], default="trajdata")
    parser.add_argument("--eval_class",       type=str, default=None)
    parser.add_argument("--results_root_dir", type=str, default=None,
                        help="Results dir. Defaults to cfg.results_dir from --config_file.")
    parser.add_argument("--num_scenes_per_batch", type=int, default=None)
    parser.add_argument("--seed",             type=int, default=None)
    parser.add_argument("--prefix",           type=str, default=None)
    parser.add_argument("--registered_name",  type=str, default="trajdata_nusc_diff")
    parser.add_argument("--num_scenes",       type=int, default=None)
    parser.add_argument("--stop_after_collisions", type=int, default=None)
    parser.add_argument("--shard",       type=int, default=1,
                        help="Shard index (1-based); picks eval_scenes[(n*(shard-1))//N : (n*shard)//N]")
    parser.add_argument("--num_shards",  type=int, default=1)


def run(args) -> None:
    if args.num_shards > 1 and not (1 <= args.shard <= args.num_shards):
        raise ValueError(f"--shard must be in 1..{args.num_shards} (got {args.shard})")

    cfg = SceneEditingConfig(registered_name=args.registered_name)
    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    for attr in ("eval_class", "num_scenes_per_batch", "seed"):
        if getattr(args, attr) is not None:
            setattr(cfg, attr, getattr(args, attr))

    if args.results_root_dir is not None:
        cfg.results_dir = args.results_root_dir
    # Otherwise: keep cfg.results_dir as provided by --config_file (matches
    # rollout/nusc.py and rollout/ad4che.py behaviour).

    if cfg.name is None:
        cfg.name = cfg.eval_class
    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name
    cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)

    if args.env is not None:
        cfg.env = args.env

    cfg.edits.editing_source = ["config"]
    if not isinstance(cfg.edits.editing_source, list):
        cfg.edits.editing_source = [cfg.edits.editing_source]

    # Per-shard HDF5 so parallel processes don't clobber each other
    if args.num_shards > 1:
        cfg.experience_hdf5_path = os.path.join(
            cfg.results_dir, f"data_shard{args.shard:02d}.hdf5")
    else:
        cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    for k in cfg[cfg.env]:
        cfg[k] = cfg[cfg.env][k]
    cfg.pop("nusc")
    cfg.pop("trajdata")

    if args.num_scenes is not None:
        cfg.eval_scenes = cfg.eval_scenes[:args.num_scenes]

    cfg.lock()
    run_scene_editor(
        cfg,
        shard_idx=args.shard,
        num_shards=args.num_shards,
        stop_after_collisions=args.stop_after_collisions,
    )


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Reactive cubic-Bezier adversary baseline")
    add_args(_parser)
    run(_parser.parse_args())
