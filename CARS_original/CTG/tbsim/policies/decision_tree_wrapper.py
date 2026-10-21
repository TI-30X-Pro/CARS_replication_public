from tbsim.policies.common import Action, Plan, RolloutAction
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, Dict
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.timer import Timers
from tbsim.envs.base import BatchedEnv
from typing import Optional, List, Tuple
from tbsim.utils.geometry_utils import transform_points_np, transform_points_tensor, VEH_VEH_collision


from scripts.decision_tree import extract_features as _default_extract_features

class TargetAdvOthersWrapper(object):
    def __init__(
        self,
        decision_tree_model,
        target_policy=None,
        adv_policy=None,
        others_predictor=None,
        target_name=None,
        confirm_k=5,
        initial_adv_idx=None,
        enable_tree_viz=False,
        topk=5,
        feature_fn=None,
    ):
        self.decision_tree_model = decision_tree_model
        self.target_policy = target_policy
        self.adv_policy = adv_policy
        self.others_predictor = others_predictor
        self.adversary_identifier_hook = None
        self.target_name = target_name
        self.device = target_policy.device if target_policy else (adv_policy.device if adv_policy else None)

        self.confirm_k = confirm_k
        self.current_adv_idx = initial_adv_idx
        self.cand_adv_idx = None
        self.cand_count = 0

        self.adv_name = None

        self.enable_tree_viz = enable_tree_viz
        self.topk = topk

        # Dataset-specific feature function. Default is nuScenes-compatible
        # (scripts/decision_tree.py::extract_features). RounD / AD4CHE that
        # train with their own feature set should pass in their matching
        # obs-based feature function so training and runtime stay aligned.
        self.feature_fn = (feature_fn if feature_fn is not None
                           else _default_extract_features)

    def eval(self):
        if self.target_policy is not None:
            self.target_policy.eval()
        if self.adv_policy is not None:
            self.adv_policy.eval()
        if self.others_predictor is not None:
            self.others_predictor.eval()

    def _infer_adv_name(self, obs_agents):
        names = list(obs_agents["agent_name"])
        t_idx = names.index(self.target_name)

        feats, _ = self.feature_fn(obs_agents, t_idx)                # (M, F)
        p = self.decision_tree_model.predict_proba(feats)[:, 1]      # (M,)

        p[t_idx] = -1.0
        top5_idx = np.argsort(p)[::-1][:5].astype(np.int32)
        pred_idx = int(top5_idx[0])

        top5_p = p[top5_idx].astype(np.float32)                     # (5,)
        feat_pred = feats[pred_idx].astype(np.float32)              # (9,)

        return t_idx, pred_idx, top5_idx, top5_p, feat_pred



    def update_adv_gate(self, pred_adv_idx):
        """K-step confirmation gate for adv switching.

        Switches to a new predicted adv only after `confirm_k` consecutive steps
        predict the same new candidate; this debounces noisy per-step decision-tree
        predictions and avoids high-frequency adv flipping during rollout.
        """
        if self.current_adv_idx is None:
            self.current_adv_idx = pred_adv_idx
            self.cand_adv_idx = None
            self.cand_count = 0
            return self.current_adv_idx

        if pred_adv_idx == self.current_adv_idx:
            self.cand_adv_idx = None
            self.cand_count = 0
            return self.current_adv_idx

        if pred_adv_idx != self.cand_adv_idx:
            self.cand_adv_idx = pred_adv_idx
            self.cand_count = 1
        else:
            self.cand_count += 1

        if self.cand_count >= self.confirm_k:
            self.current_adv_idx = self.cand_adv_idx
            self.cand_adv_idx = None
            self.cand_count = 0

        return self.current_adv_idx


    def get_action(self, obs, step_index) -> RolloutAction:
        target_idx, pred_adv_idx, top5_idx, top5_p, feat_pred = self._infer_adv_name(obs["agents"])
        adv_idx = self.update_adv_gate(pred_adv_idx)

        agent_names = list(obs["agents"]["agent_name"])
        self.adv_name = agent_names[adv_idx] if adv_idx is not None else None

        target_batch, adv_batch, others_batch, idx_target, idx_adv, idx_others, M = self._split_agents(
            obs["agents"], target_idx, adv_idx
        )
    
        target_action, target_info = self.target_policy.get_action(target_batch, step_index=step_index)

        if getattr(self.adv_policy, 'guided_mode', False):
            # Transform target's predicted positions from target-centric → adv-centric frame
            target_pos_tc = target_action.positions  # (1, T, 2) target-centric
            w_from_t = target_batch["world_from_agent"]   # (1, 3, 3)
            w_from_a = adv_batch["world_from_agent"]      # (1, 3, 3)
            target_pos_world = transform_points_tensor(target_pos_tc, w_from_t)  # (1, T, 2)
            a_from_w = torch.linalg.inv(w_from_a)  # (1, 3, 3)
            target_pos_adv = transform_points_tensor(target_pos_world, a_from_w)  # (1, T, 2)
            target_fut_pos = target_pos_adv[0]  # (T, 2)
            adv_action, adv_info = self.adv_policy.get_action_guided(
                adv_batch, target_fut_pos=target_fut_pos, step_index=step_index
            )
        else:
            adv_action, adv_info = self.adv_policy.get_action(adv_batch, step_index=step_index)
        if len(idx_others) > 0:
            others_action, others_info = self.others_predictor.get_action(others_batch, step_index=step_index)
        else:
            _dev = target_action.positions.device
            _dtp = target_action.positions.dtype
            others_action = Action(
                positions=torch.zeros((0,) + target_action.positions.shape[1:], device=_dev, dtype=_dtp),
                yaws=torch.zeros((0,) + target_action.yaws.shape[1:], device=_dev, dtype=_dtp),
            )
            others_info = {}

        pos, yaw = self._merge_actions(
            target_action.positions, target_action.yaws,
            adv_action.positions, adv_action.yaws,
            others_action.positions, others_action.yaws,
            idx_target, idx_adv, idx_others, M
        )

        agents_action = Action(positions=pos, yaws=yaw)

        info = {}

        action_samples = self._merge_action_samples(
            target_info, adv_info, others_info,
            idx_target, idx_adv, idx_others, M
        )
        if action_samples is not None:
            info["action_samples"] = action_samples

        info["adv_idx"] = np.full(M, adv_idx, dtype=np.int32)

        if self.enable_tree_viz:
            info["pred_adv_idx"] = np.full(M, pred_adv_idx, dtype=np.int32)
            info["cand_adv_idx"] = np.full(M, -1 if self.cand_adv_idx is None else self.cand_adv_idx, dtype=np.int32)
            info["cand_count"] = np.full(M, self.cand_count, dtype=np.int32)
            info["top5_idx"] = np.tile(top5_idx[None, :], (M, 1)).astype(np.int32)
            info["top5_p"] = np.tile(top5_p[None, :], (M, 1)).astype(np.float32)
            info["feat_pred"] = np.tile(feat_pred[None, :], (M, 1)).astype(np.float32)

        return RolloutAction(agents=agents_action, agents_info=info)


    def _split_agents(self, obs_agents: dict, target_idx: int, adv_idx: int):
        obs_agents = TensorUtils.to_torch(obs_agents, device=self.device, ignore_if_unspecified=True)
        M = len(obs_agents["agent_name"])

        idx_target = torch.tensor([target_idx], device=self.device, dtype=torch.long)
        idx_adv = torch.tensor([adv_idx], device=self.device, dtype=torch.long)

        mask = torch.ones(M, device=self.device, dtype=torch.bool)
        mask[idx_target] = False
        mask[idx_adv] = False
        idx_others = torch.arange(M, device=self.device, dtype=torch.long)[mask]

        def take(x, idx):
            if isinstance(x, torch.Tensor):
                return x.index_select(0, idx)
            if isinstance(x, dict):
                return {k: take(v, idx) for k, v in x.items()}
            if isinstance(x, list):
                il = idx.tolist()
                return [x[i] for i in il]
            return x

        target_batch = {k: take(v, idx_target) for k, v in obs_agents.items()}
        adv_batch = {k: take(v, idx_adv) for k, v in obs_agents.items()}
        others_batch = {k: take(v, idx_others) for k, v in obs_agents.items()}

        adv_batch["target_idx"] = idx_target

        return target_batch, adv_batch, others_batch, idx_target, idx_adv, idx_others, M

    def _merge_actions(self, target_pos, target_yaw, adv_pos, adv_yaw, others_pos, others_yaw,
                       idx_target: torch.Tensor, idx_adv: torch.Tensor, idx_others: torch.Tensor, M: int):
        # Truncate every branch to the shortest action horizon.
        T_pos = min(target_pos.shape[1], adv_pos.shape[1])
        T_yaw = min(target_yaw.shape[1], adv_yaw.shape[1])
        target_pos = target_pos[:, :T_pos, ...]
        target_yaw = target_yaw[:, :T_yaw, ...]
        adv_pos = adv_pos[:, :T_pos, ...]
        adv_yaw = adv_yaw[:, :T_yaw, ...]
        others_pos = others_pos[:, :T_pos, ...]
        others_yaw = others_yaw[:, :T_yaw, ...]

        pos = torch.empty((M,) + target_pos.shape[1:], device=target_pos.device, dtype=target_pos.dtype)
        yaw = torch.empty((M,) + target_yaw.shape[1:], device=target_yaw.device, dtype=target_yaw.dtype)

        pos[idx_target] = target_pos
        yaw[idx_target] = target_yaw
        pos[idx_adv] = adv_pos
        yaw[idx_adv] = adv_yaw
        pos[idx_others] = others_pos
        yaw[idx_others] = others_yaw
        return pos, yaw

    def _merge_action_samples(self, target_info, adv_info, others_info,
                              idx_target: torch.Tensor, idx_adv: torch.Tensor, idx_others: torch.Tensor, M: int):
        """Merge per-branch ``action_samples`` dicts back into agent order.

        Each branch's ``action_samples`` is a dict ``{positions, yaws}`` with
        either shape ``(N, num_samples, T, D)`` (multi-sample) or ``(N, T, D)``
        (single sample). Returns the merged dict, or ``None`` if no branch
        produced samples.
        """
        target_samples = target_info.get("action_samples") if target_info else None
        adv_samples = adv_info.get("action_samples") if adv_info else None
        others_samples = others_info.get("action_samples") if others_info else None

        if target_samples is None and adv_samples is None and others_samples is None:
            return None

        ref_samples = target_samples or adv_samples or others_samples
        ref_pos = ref_samples["positions"]
        ref_yaw = ref_samples["yaws"]

        has_sample_dim = len(ref_pos.shape) == 4

        def _get_T(samples):
            if samples is None:
                return None
            p = samples["positions"]
            return p.shape[2] if has_sample_dim else p.shape[1]
        T = min(t for t in (_get_T(target_samples), _get_T(adv_samples), _get_T(others_samples)) if t is not None)

        def _truncate(samples, T):
            if samples is None:
                return None
            if has_sample_dim:
                return {
                    "positions": samples["positions"][:, :, :T, :],
                    "yaws": samples["yaws"][:, :, :T, :],
                }
            return {
                "positions": samples["positions"][:, :T, :],
                "yaws": samples["yaws"][:, :T, :],
            }
        target_samples = _truncate(target_samples, T)
        adv_samples = _truncate(adv_samples, T)
        others_samples = _truncate(others_samples, T)

        if has_sample_dim:
            num_samples = ref_pos.shape[1]
            pos_dim = ref_pos.shape[3]
            yaw_dim = ref_yaw.shape[3]
            merged_pos = torch.zeros((M, num_samples, T, pos_dim), device=ref_pos.device, dtype=ref_pos.dtype)
            merged_yaw = torch.zeros((M, num_samples, T, yaw_dim), device=ref_yaw.device, dtype=ref_yaw.dtype)
        else:
            pos_dim = ref_pos.shape[2]
            yaw_dim = ref_yaw.shape[2]
            merged_pos = torch.zeros((M, T, pos_dim), device=ref_pos.device, dtype=ref_pos.dtype)
            merged_yaw = torch.zeros((M, T, yaw_dim), device=ref_yaw.device, dtype=ref_yaw.dtype)

        if target_samples is not None:
            merged_pos[idx_target] = target_samples["positions"]
            merged_yaw[idx_target] = target_samples["yaws"]

        if adv_samples is not None:
            merged_pos[idx_adv] = adv_samples["positions"]
            merged_yaw[idx_adv] = adv_samples["yaws"]

        if others_samples is not None:
            merged_pos[idx_others] = others_samples["positions"]
            merged_yaw[idx_others] = others_samples["yaws"]

        return {
            "positions": merged_pos,
            "yaws": merged_yaw,
        }


def _check_collision_rollout(obs, policy, adv_idx):
    """Return True if target collided with the currently-active adv this step.

    Used by the rollout loop as an early-stop signal.
    """
    if adv_idx is None:
        return False

    agents = obs["agents"]
    agent_names = agents["agent_name"]
    target_idx = agent_names.index(policy.target_name)

    hist = agents.get("history_positions")
    extent = agents.get("extent")
    wfa = np.asarray(agents["world_from_agent"], dtype=np.float32)

    def _get_world_pose(idx):
        p_agent = np.asarray(hist[idx, -1], dtype=np.float32).reshape(1, 2)
        p_world = transform_points_np(p_agent, wfa[idx])[0]
        yaw = np.arctan2(wfa[idx, 1, 0], wfa[idx, 0, 0])
        return np.array([p_world[0], p_world[1], yaw], dtype=np.float32)

    p_target = _get_world_pose(target_idx)
    target_extent = np.array([[float(extent[target_idx, 0]), float(extent[target_idx, 1])]], dtype=np.float32)

    p_adv = _get_world_pose(adv_idx)
    adv_extent = np.array([[float(extent[adv_idx, 0]), float(extent[adv_idx, 1])]], dtype=np.float32)
    min_dis = VEH_VEH_collision(
        p_adv[np.newaxis], p_target[np.newaxis],
        adv_extent, target_extent
    )
    return float(min_dis[0]) < 0


def safety_critical_guided_rollout_with_custom_target(
    env,
    policy,
    n_step_action=1,
    render=False,
    scene_indices=None,
    device=None,
    obs_to_torch=False,
    horizon=None,
    start_frames=None,

):

    stats = {}
    info = {}
    renderings = []
    is_batched_env = isinstance(env, BatchedEnv)
    timers = Timers()

  

    # env.reset(scene_indices=scene_indices, start_frame_index=start_frames)

    done = env.is_done()
    counter = 0
    step_since_last_update = 0
    frames = []
    obs = env.get_observation(split_ego=False)
    had_collision = False
    while not done:
        timers.tic("step")
        with timers.timed("to_torch"):
            if obs_to_torch:
                device = policy.device if device is None else device
                obs_torch = TensorUtils.to_torch(obs, device=device, ignore_if_unspecified=True)
            else:
                obs_torch = obs
        with timers.timed("network"):
            action = policy.get_action(obs_torch, step_index=counter)
        adv_idx_used = policy.current_adv_idx

        with timers.timed("env_step"):
            ims = env.step(
                action, num_steps_to_take=n_step_action, render=render
            )  # List of [num_scene, h, w, 3]
        if render:
            frames.extend(ims)
        counter += n_step_action
        step_since_last_update += n_step_action
        timers.toc("step")
        print(timers)
        print('counter', counter)

        obs = env.get_observation(split_ego=False)
        if _check_collision_rollout(obs, policy, adv_idx=adv_idx_used):
            had_collision = True
            # Log the collision frame before breaking so HDF5 contains the
            # full trajectory including the frame where the OBB overlap occurs.
            env.log_current_obs_as_dummy_step(adv_idx=adv_idx_used)
            break

        done = env.is_done()

        if horizon is not None and counter >= horizon:
            break


    env_info = env.get_info()
    for k, v in env_info.items():
        if k not in info:
            info[k] = []
        if is_batched_env:
            info[k].extend(v)
        else:
            info[k].append(v)

    info["had_collision"] = [had_collision]

    return info