"""RounD generalization rollout — test nuScenes-trained adversary on roundabout data."""
import argparse
import json
import os
import pickle
import dill
import random
import importlib

import numpy as np
import torch
import h5py

from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env
from tbsim.utils.safety_critical_batch_utils import set_global_trajdata_batch_raster_cfg
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.evaluation.env_builders import EnvCLDBuilder
from scripts.decision_tree import extract_features_round
from tbsim.policies.decision_tree_wrapper import (
    TargetAdvOthersWrapper,
    safety_critical_guided_rollout_with_custom_target,
)
from trajdata.data_structures.agent import AgentType
from tbsim.policies.common import Action


class RounDPolicyWrapper:
    """Post-process rollout actions for RounD:
    - Zero actions for stationary agents (speed < threshold)
    - Replace 'others' agents with GT trajectories + EB braking

    Others follow GT path but slow down when too close to any other agent
    (roller-coaster model: same path, adjustable speed).
    """

    # EB braking parameters
    EB_SAFE_DIST = 8.0    # start braking below this distance (m)
    EB_STOP_DIST = 3.0    # full stop below this distance (m)
    EB_MAX_DEFICIT = 0.8  # max speed reduction per n_step_action call (fraction of GT speed)

    def __init__(self, inner, speed_th=0.5, gt_others=False):
        self.inner = inner
        self.speed_th = speed_th
        self.gt_others = gt_others
        self.stationary_mask = None
        # Per-agent accumulated speed deficit (roller-coaster model)
        self.speed_deficit = {}  # {agent_idx: cumulative_deficit_fraction}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _get_world_positions(self, obs):
        """Get current world positions of all agents from observation."""
        agents = obs["agents"]
        centroid = agents.get("centroid")  # (M, 2) world positions
        if centroid is None:
            return None
        if isinstance(centroid, torch.Tensor):
            return centroid.cpu().numpy()
        return np.array(centroid)

    def _compute_eb_scale(self, agent_idx, world_pos, world_yaw, t_idx, a_idx):
        """Compute speed scale factor for an 'other' agent based on EB.

        Only brakes for agents AHEAD (within ±60° of heading, positive longitudinal distance).
        Returns scale in [0, 1]: 1 = full GT speed, 0 = stop.
        """
        if world_pos is None or world_yaw is None:
            return 1.0

        my_pos = world_pos[agent_idx]
        my_yaw = float(world_yaw[agent_idx])
        heading = np.array([np.cos(my_yaw), np.sin(my_yaw)])
        M = world_pos.shape[0]

        d_lon_min = float('inf')
        for j in range(M):
            if j == agent_idx:
                continue
            rel = world_pos[j] - my_pos
            # Longitudinal distance (along my heading)
            d_lon = np.dot(rel, heading)
            if d_lon <= 0:
                continue  # behind me, ignore
            # Lateral distance (perpendicular to heading)
            d_lat = abs(np.dot(rel, np.array([-heading[1], heading[0]])))
            if d_lat > 4.0:
                continue  # not in my lane (>4m lateral), ignore
            if d_lon < d_lon_min:
                d_lon_min = d_lon

        if d_lon_min >= self.EB_SAFE_DIST:
            return 1.0
        elif d_lon_min <= self.EB_STOP_DIST:
            return 0.0
        else:
            return (d_lon_min - self.EB_STOP_DIST) / (self.EB_SAFE_DIST - self.EB_STOP_DIST)

    def get_action(self, obs, step_index):
        agents = obs["agents"]
        names = list(agents["agent_name"])
        M = len(names)

        # --- Stationary mask (step 0) ---
        if step_index == 0:
            tgt_name = self.inner.target_name
            if "target_positions" in agents:
                gt_pos = agents["target_positions"]
                if isinstance(gt_pos, torch.Tensor):
                    gt_pos = gt_pos.cpu().numpy()
                total_disp = np.linalg.norm(gt_pos[:, -1, :] - gt_pos[:, 0, :], axis=-1)
                self.stationary_mask = total_disp < self.speed_th
            elif "curr_speed" in agents:
                spd = agents["curr_speed"]
                spd_np = spd.squeeze().cpu().numpy() if isinstance(spd, torch.Tensor) else np.squeeze(spd)
                self.stationary_mask = spd_np < 0.5
            if tgt_name in names:
                self.stationary_mask[names.index(tgt_name)] = False
            stat_names = [names[i] for i in range(M) if self.stationary_mask[i]]
            print(f"  Stationary: {len(stat_names)}/{M} {stat_names} (disp_th={self.speed_th}m)")
            # Reset speed deficits
            self.speed_deficit = {i: 0.0 for i in range(M)}

        result = self.inner.get_action(obs, step_index)
        pos = result.agents.positions   # (M, T, 2) agent-centric relative
        yaw = result.agents.yaws        # (M, T, 1)
        dev = pos.device if isinstance(pos, torch.Tensor) else None

        # --- Apply stationary mask (exclude target + adv) ---
        if self.stationary_mask is not None:
            mask = self.stationary_mask.copy()
            a_idx = self.inner.current_adv_idx
            if a_idx is not None and 0 <= a_idx < len(mask):
                mask[a_idx] = False
            if dev is not None:
                keep = torch.tensor(~mask, dtype=torch.float32,
                                    device=dev).unsqueeze(-1).unsqueeze(-1)
            else:
                keep = (~mask).astype(np.float32)[:, None, None]
            pos = pos * keep
            yaw = yaw * keep

        # --- GT replay + EB braking for others ---
        if self.gt_others and "target_positions" in agents:
            gt_pos = agents["target_positions"]  # (M, T_gt, 2)
            gt_yaw = agents["target_yaws"]       # (M, T_gt, 1)
            if isinstance(gt_pos, np.ndarray):
                gt_pos = torch.tensor(gt_pos, dtype=pos.dtype, device=dev)
                gt_yaw = torch.tensor(gt_yaw, dtype=yaw.dtype, device=dev)

            T = pos.shape[1]
            T_gt = gt_pos.shape[1]
            T_use = min(T, T_gt)

            t_idx = names.index(self.inner.target_name) if self.inner.target_name in names else -1
            a_idx = self.inner.current_adv_idx if self.inner.current_adv_idx is not None else -1

            # Get world positions and yaws for EB distance check
            world_pos = self._get_world_positions(obs)
            world_yaw = agents.get("yaw")
            if isinstance(world_yaw, torch.Tensor):
                world_yaw = world_yaw.cpu().numpy()
            world_yaw = np.array(world_yaw).flatten() if world_yaw is not None else None

            eb_count = 0
            for i in range(M):
                if i == t_idx or i == a_idx:
                    continue

                # Start with GT trajectory
                pos[i, :T_use] = gt_pos[i, :T_use]
                yaw[i, :T_use] = gt_yaw[i, :T_use]

                # EB braking: only for agents AHEAD in driving direction
                eb_scale = self._compute_eb_scale(i, world_pos, world_yaw, t_idx, a_idx)

                # Roller-coaster: accumulate deficit, don't recover instantly
                if eb_scale < 1.0:
                    # Increase deficit (braking)
                    self.speed_deficit[i] = min(
                        self.speed_deficit.get(i, 0.0) + (1.0 - eb_scale) * 0.3,
                        self.EB_MAX_DEFICIT
                    )
                else:
                    # Slowly recover (release brake)
                    self.speed_deficit[i] = max(0.0, self.speed_deficit.get(i, 0.0) - 0.1)

                final_scale = max(0.0, 1.0 - self.speed_deficit.get(i, 0.0))

                if final_scale < 0.99:
                    # Scale GT displacement (keep direction, reduce magnitude)
                    if dev is not None:
                        pos[i, :T_use] = gt_pos[i, :T_use] * final_scale
                    else:
                        pos[i, :T_use] = gt_pos[i, :T_use] * final_scale
                    eb_count += 1

            if step_index == 0:
                n_gt = M - (1 if t_idx >= 0 else 0) - (1 if a_idx >= 0 else 0)
                print(f"  GT+EB replay: {n_gt}/{M} others (GT path + emergency braking)")

        result.agents.positions = pos
        result.agents.yaws = yaw
        return result


def find_valid_targets(scene_paths, history_frames, num_sim_steps, max_per_scene,
                       cache_dir="~/cld_cache/round", min_speed=2.0):
    """Auto-select targets: pick moving agents with longest duration."""
    import pandas as pd
    result = {}
    for si, path in enumerate(scene_paths):
        with open(path, "rb") as f:
            scene = dill.load(f)

        # Load speed data from cache to filter slow agents
        feather = os.path.join(os.path.expanduser(cache_dir),
                               scene.name, "agent_data_dt0.10.feather")
        df = pd.read_feather(feather) if os.path.exists(feather) else None

        valid = []
        for a in scene.agents:
            if a.type != AgentType.VEHICLE:
                continue
            dur = a.last_timestep - a.first_timestep
            if dur < history_frames + num_sim_steps:
                continue
            start = a.first_timestep + history_frames + 1
            # Check speed at start frame
            if df is not None:
                row = df[(df["agent_id"] == a.name) & (df["scene_ts"] == start)]
                if len(row) > 0:
                    spd = float(np.sqrt(row.iloc[0]["vx"]**2 + row.iloc[0]["vy"]**2))
                    if spd < min_speed:
                        continue  # skip slow/stationary agents
            valid.append((a.name, start, dur))
        valid.sort(key=lambda x: x[2], reverse=True)
        result[si] = valid[:max_per_scene]
        print(f"  Scene {si} ({scene.name}): {len(valid)} valid moving targets, selected {len(result[si])}")
    return result


def save_episode(h5_path, ep_key, scene_buf):
    """Minimal HDF5 save."""
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "a") as h5f:
        for mk, val in scene_buf.items():
            h5key = f"/{ep_key}/{mk}"
            if h5key in h5f:
                del h5f[h5key]
            try:
                h5f.create_dataset(h5key, data=val)
            except TypeError:
                if isinstance(val, (str, list)):
                    arr = np.array([val] if isinstance(val, str) else val, dtype=object)
                    h5f.create_dataset(h5key, data=arr, dtype=dt)
    print(f"  [SAVED] {ep_key}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config_file", type=str, default="configs/simulation_config_round.json")
    parser.add_argument("--num_scenes", type=int, default=3)
    parser.add_argument("--targets_per_scene", type=int, default=2)
    parser.add_argument("--render", action="store_true", default=False,
                        help="Save per-episode videos (slow; off by default)")
    parser.add_argument("--results_name", type=str, default=None,
                        help="Override cfg.name (e.g. 'RounD-shard1of6' for sharded runs)")
    parser.add_argument("--gt_others", action="store_true", default=False,
                        help="Others agents replay GT trajectory (only tgt+adv use model)")
    parser.add_argument("--save_all", action="store_true", default=False)
    parser.add_argument("--round_norm", type=str, default=None,
                        help="Path to RounD norm_info .pt file (override model normalization)")
    parser.add_argument("--labels", type=str, default=None,
                        help="RounD labels JSON (e.g. labels/round_labels_rec07.json). "
                             "Uses labeled TGTs + decision tree for ADV selection.")
    parser.add_argument("--auto", action="store_true", default=False,
                        help="Auto mode: every moving agent in every slice becomes TGT")
    parser.add_argument("--dt_model", type=str, default="checkpoints/round_decision_tree.pkl",
                        help="Decision tree model for ADV selection")
    parser.add_argument("--rec", type=str, default="07",
                        help="Recording ID")
    parser.add_argument("--slice_sec", type=float, default=5.0,
                        help="Slice length in seconds (for --auto mode)")
    parser.add_argument("--min_speed", type=float, default=1.0,
                        help="Min speed (m/s) to be considered as TGT in --auto mode")
    parser.add_argument("--seed", type=int, default=42)


def run(args) -> None:
    # --- Config ---
    cfg = SceneEditingConfig(registered_name="visier")
    cfg.update(**json.load(open(args.config_file)))
    cfg.seed = args.seed
    if args.results_name:
        cfg.name = args.results_name
    cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)
    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")
    for k in cfg[cfg.env]:
        cfg[k] = cfg[cfg.env][k]
    cfg.pop("nusc")
    cfg.pop("trajdata")
    cfg.eval_scenes = cfg.eval_scenes[:args.num_scenes]
    cfg.lock()

    # --- Seed ---
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # --- Global state ---
    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(cfg.trajdata_source_test[0])
    os.makedirs(cfg.results_dir, exist_ok=True)
    if args.render:
        os.makedirs(os.path.join(cfg.results_dir, "viz/"), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Policies (3 roles) ---
    composers = importlib.import_module("tbsim.evaluation.policy_composers")
    Composer = getattr(composers, cfg.eval_class)

    target_policy, exp_config = Composer(cfg, device).get_policy()
    agents_planner, _ = Composer(cfg, device).get_planner()
    others_predictor, _ = Composer(cfg, device).get_predictor()

    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)

    for p in [target_policy, agents_planner, others_predictor]:
        p.model.nets["policy"].eval()
        p.model.nets["policy"].requires_grad_(False)
        if hasattr(p.model.nets["policy"], "sigma_gmm_scale"):
            p.model.nets["policy"].sigma_gmm_scale = 3

    # --- Override dynamics for roundabout (relax yaw_rate clipping) ---
    for p in [target_policy, agents_planner, others_predictor]:
        dyn = p.model.nets["policy"].dyn
        dyn.max_yawvel = 2.5    # R≈35m: v=8→limit=0.31→R=25m, v=10→limit=0.25→R=40m
        dyn.max_steer = 0.30

    # --- Best-of-K: generate K samples, select closest to GT ---
    K = 16
    for p in [target_policy, agents_planner, others_predictor]:
        orig_get_action = p.model.get_action
        def make_patched(orig):
            def patched(obs_torch, **kwargs):
                kwargs.setdefault('num_action_samples', K)
                return orig(obs_torch, **kwargs)
            return patched
        p.model.get_action = make_patched(orig_get_action)
    print(f"  Best-of-K: num_action_samples={K} (select closest to GT)")

    # --- Override norm_info with RounD statistics ---
    if args.round_norm:
        print(f"Overriding norm_info from {args.round_norm}")
        rn = torch.load(args.round_norm)
        for p in [target_policy, agents_planner, others_predictor]:
            m = p.model.nets["policy"]
            # action norm
            m.add_coeffs = np.array(rn["action_norm_info_center"][0], dtype="float32")
            m.div_coeffs = np.array(rn["action_norm_info_center"][1], dtype="float32")
            # center history encoder
            m.center_hist.mean.copy_(torch.tensor(rn["norm_info_center"][0], dtype=torch.float32))
            m.center_hist.std.copy_(torch.tensor(rn["norm_info_center"][1], dtype=torch.float32))
            # ego encoder
            m.ego_encoder.mean_pos.copy_(torch.tensor(rn["norm_info_ego"][0][:2], dtype=torch.float32))
            m.ego_encoder.std_pos.copy_(torch.tensor(rn["norm_info_ego"][1][:2], dtype=torch.float32))
            m.ego_encoder.mean_speed.copy_(torch.tensor([rn["norm_info_ego"][0][2]], dtype=torch.float32))
            m.ego_encoder.std_speed.copy_(torch.tensor([rn["norm_info_ego"][1][2]], dtype=torch.float32))
            # neighbor history encoder
            m.neighbor_hist.neigh_encode.mean.copy_(torch.tensor(rn["norm_info_neighbor"][0], dtype=torch.float32))
            m.neighbor_hist.neigh_encode.std.copy_(torch.tensor(rn["norm_info_neighbor"][1], dtype=torch.float32))
        print("  Norm_info overridden for all 3 policies")

    # --- Environment ---
    env = EnvCLDBuilder(
        eval_config=cfg, exp_config=exp_config, device=device, log_data=True
    ).get_env()

    # --- Decision tree ---
    dt_path = args.dt_model if (args.labels or args.auto) else "checkpoints/decision_tree_model.pkl"
    with open(dt_path, "rb") as f:
        dt_data = pickle.load(f)
    dt_model = dt_data if not isinstance(dt_data, dict) else dt_data.get('model', dt_data)
    print(f"Decision tree loaded from {dt_path}")

    # --- Find valid targets ---
    history_frames = exp_config.algo.history_num_frames

    def _find_scene_idx(rec_id):
        """Find trajdata scene index for a recording."""
        scene_name = f"round_{rec_id.zfill(2)}"
        for si, sp in enumerate(env.dataset._scene_index):
            with open(sp, "rb") as f:
                sc = dill.load(f)
            if sc.name == scene_name:
                return si, scene_name
        return None, scene_name

    if args.labels or args.auto:
        fps_raw = 25
        dt_ratio = int(fps_raw * 0.2)  # 5
        scene_idx, scene_name = _find_scene_idx(args.rec)
        if scene_idx is None:
            print(f"ERROR: scene '{scene_name}' not found")
            return
        cfg.eval_scenes = [scene_idx]

    if args.auto:
        # Auto mode: every moving agent in every slice becomes TGT.
        # CSV tracks live under <dataset_path>/data/ (cfg.dataset_path is
        # provided by simulation_config_round.json).
        import pandas as pd
        rec_id = args.rec.zfill(2)
        round_data_dir = os.path.join(os.path.expanduser(cfg.dataset_path), "data")
        df = pd.read_csv(os.path.join(round_data_dir, f"{rec_id}_tracks.csv"))
        meta = pd.read_csv(os.path.join(round_data_dir, f"{rec_id}_tracksMeta.csv"))
        veh_ids = set(meta[meta['class'].isin({'car','truck','van','trailer','bus'})]['trackId'])
        df = df[df['trackId'].isin(veh_ids)]

        slice_frames = int(args.slice_sec * fps_raw)
        all_frames = sorted(df['frame'].unique())
        f_min, f_max = all_frames[0], all_frames[-1]

        # Build slice start_ts list — agents will be discovered from env after reset
        slice_ts_list = []
        start = f_min + int(2.0 * fps_raw)
        while start + slice_frames <= f_max:
            start_ts = max(start // dt_ratio, history_frames + 1)
            if not slice_ts_list or slice_ts_list[-1] != start_ts:
                slice_ts_list.append(start_ts)
            start += slice_frames

        # Use a special marker: targets will be discovered per-slice from env
        targets_per_scene = {scene_idx: "__auto__"}
        print(f"\nAuto mode: rec_{rec_id}, {len(slice_ts_list)} unique slices to scan")

    elif args.labels:
        # Use labelled TGTs from round_labels JSON. The file is now a
        # ``{scene_key: [[tgt, adv], ...]}`` list-of-pair-arrays (matching
        # nuScenes' simulation_config.json target_adv_pairs convention).
        with open(args.labels) as f:
            round_labels = json.load(f)

        targets_per_scene = {scene_idx: []}
        for scene_key, pairs in round_labels.items():
            if not isinstance(pairs, list) or not pairs:
                continue
            start_frame_raw = int(scene_key.split('_')[-1])
            label_ts = start_frame_raw // dt_ratio
            start_ts = max(label_ts, history_frames + 1)
            tgt_ids = {tgt for tgt, _adv in pairs}
            for tgt_id in tgt_ids:
                targets_per_scene[scene_idx].append((str(tgt_id), start_ts, 50))
        print(f"\nLabels mode: scene={scene_name} (idx={scene_idx}), "
              f"{len(targets_per_scene[scene_idx])} target entries from {len(round_labels)} slices")
    else:
        print(f"\nFinding valid targets (history={history_frames}, sim_steps={cfg.num_simulation_steps})...")
        targets_per_scene = find_valid_targets(
            env.dataset._scene_index, history_frames, cfg.num_simulation_steps,
            max_per_scene=args.targets_per_scene,
        )

    # --- Renderer (optional) ---
    render_rasterizer = None
    if args.render:
        from tbsim.utils.scene_edit_utils import get_trajdata_renderer
        render_rasterizer = get_trajdata_renderer(
            cfg.trajdata_source_train, cfg.trajdata_data_dirs,
            future_sec=cfg.future_sec, history_sec=cfg.history_sec,
            raster_size=400, px_per_m=2.0,
            rebuild_maps=False, cache_location=cfg.trajdata_cache_location,
        )

    # --- Rollout ---
    completed_pairs = set()
    skipped_dup = 0
    total_rollouts = 0
    total_collisions = 0

    # --- Resume: scan existing HDF5 for completed (scene, start_frame, tgt) tuples ---
    # Ep key format: f"{scene_name}_{start_frame:04d}_tgt{tgt}_adv{adv}"
    # We skip at the (scene, start_frame, tgt) level so DT-selected adv can differ
    # across runs without double-counting.
    resumed_prefixes = set()
    if os.path.exists(cfg.experience_hdf5_path):
        try:
            with h5py.File(cfg.experience_hdf5_path, "r") as rf:
                for ek in rf.keys():
                    # scene name is "round_NN", start_frame is 4 digits, then "_tgtX_advY"
                    idx = ek.find("_tgt")
                    if idx > 0:
                        resumed_prefixes.add(ek[:idx])  # "round_NN_XXXX"
                    idx2 = ek.find("_adv")
                    if idx2 > 0:
                        # also record full (scene_ts_tgt) for exact skip
                        resumed_prefixes.add(ek[:idx2])  # "round_NN_XXXX_tgtY"
            print(f"[resume] Found {len(resumed_prefixes)} existing ep_key prefixes "
                  f"in {cfg.experience_hdf5_path}")
        except Exception as e:
            print(f"[resume] WARN: failed to read existing HDF5 ({e}); starting fresh")

    for scene_i in cfg.eval_scenes:
        is_auto = targets_per_scene.get(scene_i) == "__auto__"

        if is_auto:
            # Auto mode: iterate slices, discover agents from env after each reset
            iter_list = [(None, ts, 50) for ts in slice_ts_list]
        else:
            iter_list = targets_per_scene.get(scene_i, [])

        if not iter_list:
            print(f"\nScene {scene_i}: nothing to rollout")
            continue

        ti_global = 0
        for target_name, start_frame, dur in iter_list:
            try:
                env.reset(scene_indices=[scene_i], start_frame_index=[start_frame])
            except Exception as e:
                print(f"  RESET ERROR ts={start_frame}: {e}")
                continue

            # Empty-scene guard: trajdata prints "Initial timestep X contains no agents
            # after filtering" and leaves current_scene_names empty. Must not crash.
            if not env.current_scene_names or env.is_done():
                print(f"  SKIP ts={start_frame}: empty scene after reset")
                continue

            scene_name = env.current_scene_names[0]
            agent_names = env.current_agent_names

            # In auto mode, iterate ALL moving agents at this timestep
            if is_auto:
                obs = env.get_observation(split_ego=False)
                agents_obs = obs["agents"]
                spd = agents_obs.get("curr_speed")
                if isinstance(spd, torch.Tensor):
                    spd = spd.cpu().numpy()
                tgt_candidates = [agent_names[i] for i in range(len(agent_names))
                                  if float(spd[i]) >= args.min_speed]
            else:
                tgt_candidates = [target_name] if target_name in agent_names else []

            for target_name in tgt_candidates:
                ti_global += 1
                target_idx = agent_names.index(target_name)

                # Resume skip: if this (scene, start_frame, tgt) was already run
                # (any adv), skip it. Matches any adv for the same tgt/slice.
                resume_key = f"{scene_name}_{start_frame:04d}_tgt{target_name}"
                if resume_key in resumed_prefixes:
                    skipped_dup += 1
                    continue

                # Need fresh reset for each TGT (env state changes after rollout)
                env.reset(scene_indices=[scene_i], start_frame_index=[start_frame])

                inner_wrapper = TargetAdvOthersWrapper(
                    decision_tree_model=dt_model,
                    target_policy=target_policy,
                    adv_policy=agents_planner,
                    others_predictor=others_predictor,
                    target_name=target_name,
                    feature_fn=extract_features_round,
                )
                wrapper = RounDPolicyWrapper(inner_wrapper, speed_th=7.0,
                                             gt_others=args.gt_others)

                info = safety_critical_guided_rollout_with_custom_target(
                    env, wrapper,
                    n_step_action=cfg.n_step_action,
                    scene_indices=[scene_i],
                    horizon=cfg.num_simulation_steps,
                    start_frames=[start_frame],
                )

                had_collision = info.get("had_collision", [False])[0]
                adv_idx_final = inner_wrapper.current_adv_idx
                adv_name_final = agent_names[adv_idx_final] if adv_idx_final is not None else "?"

                # Dedup
                pair_key = (target_name, adv_name_final)
                if pair_key in completed_pairs:
                    skipped_dup += 1
                    torch.cuda.empty_cache()
                    continue
                completed_pairs.add(pair_key)

                total_rollouts += 1
                status = "COLLISION" if had_collision else "no collision"
                if had_collision:
                    total_collisions += 1
                print(f"  [{ti_global}] ts={start_frame} {status}  TGT={target_name} ADV={adv_name_final}  "
                      f"({total_collisions} collisions / {total_rollouts} rollouts)")

                ep_key = f"{scene_name}_{start_frame:04d}_tgt{target_name}_adv{adv_name_final}"
                if "buffer" in info and (had_collision or args.save_all):
                    for buf in info["buffer"]:
                        buf["target_name"] = np.array([target_name])
                    save_episode(cfg.experience_hdf5_path, ep_key, info["buffer"][0])

                if args.render and "buffer" in info and (had_collision or args.save_all):
                    from tbsim.utils.scene_edit_utils import visualize_guided_rollout
                    viz_dir = os.path.join(cfg.results_dir, "viz/")
                    for si_buf, scene_buf in zip(info["scene_index"], info["buffer"]):
                        if "action_traj_positions" in scene_buf:
                            N = scene_buf["action_traj_positions"].shape[0]
                            for i in range(N):
                                if i != target_idx and i != adv_idx_final:
                                    scene_buf["action_traj_positions"][i] = 0
                        visualize_guided_rollout(
                            viz_dir, render_rasterizer, si_buf, scene_buf,
                            guidance_config=None, constraint_config=None,
                            fps=10.0, n_step_action=cfg.n_step_action,
                            viz_diffusion_steps=False, first_frame_only=False,
                        sim_num=start_frame, save_every_n_frames=1,
                        draw_mode="action", target_idx=target_idx,
                        agent_names=agent_names,
                    )

            torch.cuda.empty_cache()

    print(f"\nDone. Results in: {cfg.results_dir}")
    print(f"  Rollouts: {total_rollouts}, Collisions: {total_collisions}, "
          f"Duplicates/resumed skipped: {skipped_dup}")
    # Explicit completion marker so log tail tells finish-vs-crash at a glance
    print(f"=== RUN COMPLETE: rec={args.rec} ===", flush=True)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="RounD cross-dataset rollout")
    add_args(_parser)
    run(_parser.parse_args())
