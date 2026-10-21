from copy import deepcopy
import numpy as np
from collections import defaultdict
import os
from pathlib import Path

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer
from trajdata import AgentType, UnifiedDataset
from tbsim.datasets.cld_datamodules import FilteredUnifiedDataset
from tbsim.l5kit.vectorizer import build_vectorizer


from tbsim.configs.eval_config import EvaluationConfig
from tbsim.configs.base import ExperimentConfig
from tbsim.utils.metrics import OrnsteinUhlenbeckPerturbation
from tbsim.envs.env_l5kit import EnvL5KitSimulation
from tbsim.envs.env_trajdata import EnvUnifiedSimulation, EnvSplitUnifiedSimulation, PPOEnvSplitUnifiedSimulation
from tbsim.utils.config_utils import translate_l5kit_cfg, translate_trajdata_cfg, translate_pass_trajdata_cfg,translate_cld_cfg
from tbsim.utils.trajdata_utils import (
    TRAJDATA_AGENT_TYPE_MAP,
    get_closest_lane_point_wrapper,
    get_full_fut_traj,
    get_full_fut_valid,
    get_current_lane_point_wrapper_visier
)
import tbsim.envs.env_metrics as EnvMetrics
from tbsim.evaluation.metric_composers import CVAEMetrics, OccupancyMetrics

from tbsim.l5kit.l5_ego_dataset import EgoDatasetMixed
from tbsim.utils.vis_utils import build_visualization_rasterizer_l5kit
from trajdata import AgentBatch
from trajdata.data_structures.state import StateArray
from tbsim.utils.trajdata_utils import parse_trajdata_batch
from tbsim.utils.safety_critical_batch_utils import parse_batch
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.policies.common import Action, RolloutAction
from tbsim.envs.base import SimulationException


def _patch_env_with_custom_observation():
    """Monkey-patch EnvSplitUnifiedSimulation so observations and step() can
    be reorganized around an arbitrary `target_agent_name` (the attacked agent),
    instead of the dataset's default ego.
    """
    if hasattr(EnvSplitUnifiedSimulation, 'get_observation_with_custom_target'):
        return

    def get_observation_with_custom_target(self, target_agent_name: str, split_ego=None, return_raw=False):
        """Return the env observation reorganized so `target_agent_name`
        plays the role of ego (the attacked agent).
        """
        if split_ego is None:
            split_ego = self.split_ego

        # The cached default-ego observation is stale once we change role.
        self._cached_observation = None
        self._cached_raw_observation = None

        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False))

        if return_raw:
            return raw_obs

        target_idx = np.array([
            i for i, name in enumerate(self.current_agent_names)
            if name == target_agent_name
        ])

        if len(target_idx) == 0:
            raise ValueError(
                f"Target agent '{target_agent_name}' not found in scene. "
                f"Available agents: {self.current_agent_names}"
            )

        other_idx = np.array([
            i for i, name in enumerate(self.current_agent_names)
            if name != target_agent_name
        ])

        if split_ego:
            target_obs_raw = [raw_obs[idx] for idx in target_idx]
            target_obs_collated = self.dataset.get_collate_fn(return_dict=True)(target_obs_raw)

            other_obs_raw = [raw_obs[idx] for idx in other_idx] if len(other_idx) > 0 else []
            if len(other_obs_raw) > 0:
                other_obs_collated = self.dataset.get_collate_fn(return_dict=True)(other_obs_raw)
            else:
                other_obs_collated = None

            if self.parse_obs == True:
                parse_plan = dict(ego=True, agent=True)
            elif self.parse_obs == False:
                parse_plan = dict(ego=False, agent=False)
            elif isinstance(self.parse_obs, dict):
                parse_plan = self.parse_obs

            if parse_plan["ego"]:
                target_obs = parse_trajdata_batch(target_obs_collated)
                target_obs = TensorUtils.to_numpy(target_obs, ignore_if_unspecified=True)
                target_obs["scene_index"] = self.current_agent_scene_index[target_idx]
                target_obs["track_id"] = self.current_agent_track_id[target_idx]
            else:
                target_obs = AgentBatch(**target_obs_collated)

            if other_obs_collated is not None:
                if parse_plan["agent"]:
                    other_obs = parse_trajdata_batch(other_obs_collated)
                    other_obs = TensorUtils.to_numpy(other_obs, ignore_if_unspecified=True)
                    other_obs["scene_index"] = self.current_agent_scene_index[other_idx]
                    other_obs["track_id"] = self.current_agent_track_id[other_idx]
                else:
                    other_obs = AgentBatch(**other_obs_collated)
            else:
                other_obs = {}

            # Keep the same "ego" / "agents" keys so downstream policies don't
            # need to know that target_agent_name has been swapped in.
            return dict(ego=target_obs, agents=other_obs)
        else:
            agent_obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
            if self.parse_obs:
                agent_obs = parse_trajdata_batch(agent_obs)
                agent_obs = TensorUtils.to_numpy(agent_obs, ignore_if_unspecified=True)
                agent_obs["scene_index"] = self.current_agent_scene_index
                agent_obs["track_id"] = self.current_agent_track_id
            else:
                agent_obs = AgentBatch(**agent_obs)
            return dict(agents=agent_obs)

    def combine_action_with_custom_target(self, step_actions, target_agent_name: str):
        """Merge target / adv / others actions back into one tensor in the
        original agent order, with `target_agent_name` slotted at its
        pre-existing index (rather than the default ego index).

        `step_actions.ego` holds the target action and `step_actions.agents`
        holds adv + others, in the same order as `other_idx` below.
        """
        target_action = step_actions.ego.to_dict()
        agents_action = step_actions.agents.to_dict()

        target_idx = np.array([
            i for i, name in enumerate(self.current_agent_names)
            if name == target_agent_name
        ])
        other_idx = np.array([
            i for i, name in enumerate(self.current_agent_names)
            if name != target_agent_name
        ])

        if len(target_idx) != target_action["positions"].shape[0]:
            raise ValueError(
                f"Target action count mismatch: expected {len(target_idx)}, "
                f"got {target_action['positions'].shape[0]}"
            )
        if len(other_idx) != agents_action["positions"].shape[0]:
            raise ValueError(
                f"Agents action count mismatch: expected {len(other_idx)}, "
                f"got {agents_action['positions'].shape[0]}"
            )

        min_length = min(target_action["positions"].shape[1], agents_action["positions"].shape[1])
        combined_positions = np.zeros([len(self.current_agent_names), min_length, 2])
        combined_yaws = np.zeros([len(self.current_agent_names), min_length, 1])

        combined_positions[target_idx] = target_action["positions"][:, :min_length]
        combined_yaws[target_idx] = target_action["yaws"][:, :min_length]

        # agents_action rows must already be ordered to match other_idx.
        combined_positions[other_idx] = agents_action["positions"][:, :min_length]
        combined_yaws[other_idx] = agents_action["yaws"][:, :min_length]

        return RolloutAction(
            agents=Action(positions=combined_positions, yaws=combined_yaws),
            agents_info=step_actions.agents_info,
        )

    _original_step = EnvSplitUnifiedSimulation._step

    def _step_patched(self, step_actions, num_steps_to_take):
        """Patched _step that routes through `combine_action_with_custom_target`
        when env.custom_target_agent_name is set.
        """
        if self.is_done():
            raise SimulationException("Cannot step in a finished episode")
        self.timers.tic("_step")
        # to bypass all the ego split, collation and parsing, directly get raw obs
        raw_obs = self.get_observation(split_ego=False,return_raw=True)
        obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        # always parse when stepping
        obs = parse_trajdata_batch(obs)
        obs = TensorUtils.to_numpy(obs,ignore_if_unspecified=True)
        obs["scene_index"] = self.current_agent_scene_index
        obs["track_id"] = self.current_agent_track_id
        obs = {k:v for k,v in obs.items() if not isinstance(v,list)}

        # record metrics
        #TODO: fix the bugs in metrics when using diffstack
        self._add_per_step_metrics(obs)
        if step_actions.has_ego:
            if hasattr(self, '_custom_target_agent_name') and self._custom_target_agent_name is not None:
                if hasattr(self, 'combine_action_with_custom_target'):
                    combined_step_actions = self.combine_action_with_custom_target(
                        step_actions, self._custom_target_agent_name
                    )
                else:
                    combined_step_actions = self.combine_action(step_actions)
            else:
                combined_step_actions = self.combine_action(step_actions)
            action = combined_step_actions.agents.to_dict()
        else:
            action = step_actions.agents.to_dict()
        
        assert action["positions"].shape[0] == obs["centroid"].shape[0]
        for action_index in range(num_steps_to_take):
            if action_index >= action["positions"].shape[1]:  # GT actions may be shorter
                self._done = True
                self._frame_index += action_index
                self._cached_observation = None
                self._cached_raw_observation = None
                return
            # # log state and action
            obs_skimp = self.get_observation_skimp(split_ego=True)
            # self._add_per_step_metrics(obs_skimp["agents"])
            if self._log_data:
                if step_actions.has_ego:
                    if hasattr(self, '_custom_target_agent_name') and self._custom_target_agent_name is not None:
                        target_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name==self._custom_target_agent_name])
                        agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!=self._custom_target_agent_name])
                    else:
                        target_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
                        agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
                    action_t = TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])
                    action_to_log = RolloutAction(
                        agents=Action.from_dict(dict(positions=action_t["positions"][agent_idx],yaws=action_t["yaws"][agent_idx])),
                        agents_info=step_actions.agents_info,
                        ego = Action.from_dict(dict(positions=action_t["positions"][target_idx],yaws=action_t["yaws"][target_idx])),
                        ego_info=step_actions.ego_info,
                    )
                else:
                    action_to_log = RolloutAction(
                        agents=Action.from_dict(TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])),
                        agents_info=step_actions.agents_info
                    )

                self.logger.log_step(obs_skimp, action_to_log)

            idx = 0
            for scene in self._current_scenes:
                scene_action = dict()
                for agent in scene.agents:
                    curr_yaw = obs["curr_agent_state"][idx, -1]
                    curr_pos = obs["curr_agent_state"][idx, :2]
                    world_from_agent = np.array(
                        [
                            [np.cos(curr_yaw), np.sin(curr_yaw)],
                            [-np.sin(curr_yaw), np.cos(curr_yaw)],
                        ]
                    )
                    next_state = np.ones(4, dtype=obs["agent_fut"].dtype) * np.nan
                    if not np.any(np.isnan(action["positions"][idx, action_index])):  # ground truth action may be NaN
                        next_state[:2] = action["positions"][idx, action_index] @ world_from_agent + curr_pos
                        next_state[-1] = curr_yaw + action["yaws"][idx, action_index, 0]
                    else:
                        pass
                    scene_action[agent.name] = StateArray.from_array(next_state, "x,y,z,h")
                    idx += 1
                scene.step(scene_action, return_obs=False)

        self._cached_observation = None
        self._cached_raw_observation = None
        self.timers.toc("_step")
        if self._frame_index + num_steps_to_take >= self.horizon:
            self._done = True
        else:
            self._frame_index += num_steps_to_take
    
    EnvSplitUnifiedSimulation.get_observation_with_custom_target = get_observation_with_custom_target
    EnvSplitUnifiedSimulation.combine_action_with_custom_target = combine_action_with_custom_target
    EnvSplitUnifiedSimulation._step = _step_patched

    from tbsim.envs.env_trajdata import PPOEnvSplitUnifiedSimulation
    if hasattr(PPOEnvSplitUnifiedSimulation, '_step'):
        def _step_ppo_patched(self, step_actions, num_steps_to_take):
            """Parallel patch for PPOEnvSplitUnifiedSimulation."""
            if self.is_done():
                raise SimulationException("Cannot step in a finished episode")
            self.timers.tic("_step")
            # to bypass all the ego split, collation and parsing, directly get raw obs
            raw_obs = self.get_observation(split_ego=False,return_raw=True)
            obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
            # always parse when stepping
            obs = parse_batch(obs)
            obs = TensorUtils.to_numpy(obs,ignore_if_unspecified=True)
            obs["scene_index"] = self.current_agent_scene_index
            obs["track_id"] = self.current_agent_track_id
            obs = {k:v for k,v in obs.items() if not isinstance(v,list)}

            # record metrics
            #TODO: fix the bugs in metrics when using diffstack
            # self._add_per_step_metrics(obs)
            if step_actions.has_ego:
                if hasattr(self, '_custom_target_agent_name') and self._custom_target_agent_name is not None:
                    if hasattr(self, 'combine_action_with_custom_target'):
                        combined_step_actions = self.combine_action_with_custom_target(
                            step_actions, self._custom_target_agent_name
                        )
                    else:
                        combined_step_actions = self.combine_action(step_actions)
                else:
                    combined_step_actions = self.combine_action(step_actions)
                action = combined_step_actions.agents.to_dict()
            else:
                action = step_actions.agents.to_dict()
            
            assert action["positions"].shape[0] == obs["centroid"].shape[0]
            for action_index in range(num_steps_to_take):
                if action_index >= action["positions"].shape[1]:  # GT actions may be shorter
                    self._done = True
                    self._frame_index += action_index
                    self._cached_observation = None
                    self._cached_raw_observation = None
                    return
                # # log state and action
                obs_skimp = self.get_observation_skimp(split_ego=True)
                # self._add_per_step_metrics(obs_skimp["agents"])
                if self._log_data:
                    if step_actions.has_ego:
                        if hasattr(self, '_custom_target_agent_name') and self._custom_target_agent_name is not None:
                            target_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name==self._custom_target_agent_name])
                            agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!=self._custom_target_agent_name])
                        else:
                            target_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
                            agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
                        action_t = TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])
                        action_to_log = RolloutAction(
                            agents=Action.from_dict(dict(positions=action_t["positions"][agent_idx],yaws=action_t["yaws"][agent_idx])),
                            agents_info=step_actions.agents_info,
                            ego = Action.from_dict(dict(positions=action_t["positions"][target_idx],yaws=action_t["yaws"][target_idx])),
                            ego_info=step_actions.ego_info,
                        )
                   
                    else:
                        action_to_log = RolloutAction(
                            agents=Action.from_dict(TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])),
                            agents_info=step_actions.agents_info
                        )

                    self.logger.log_step(obs_skimp, action_to_log)

                idx = 0
                for scene in self._current_scenes:
                    scene_action = dict()
                    for agent in scene.agents:
                        curr_yaw = obs["curr_agent_state"][idx, -1]
                        curr_pos = obs["curr_agent_state"][idx, :2]
                        world_from_agent = np.array(
                            [
                                [np.cos(curr_yaw), np.sin(curr_yaw)],
                                [-np.sin(curr_yaw), np.cos(curr_yaw)],
                            ]
                        )
                        next_state = np.ones(4, dtype=obs["agent_fut"].dtype) * np.nan
                        if not np.any(np.isnan(action["positions"][idx, action_index])):  # ground truth action may be NaN
                            next_state[:2] = action["positions"][idx, action_index] @ world_from_agent + curr_pos
                            next_state[-1] = curr_yaw + action["yaws"][idx, action_index, 0]
                        else:
                            pass
                        scene_action[agent.name] = StateArray.from_array(next_state, "x,y,z,h")
                        idx += 1
                    scene.step(scene_action, return_obs=False)

            self._cached_observation = None
            self._cached_raw_observation = None
            self.timers.toc("_step")
            if self._frame_index + num_steps_to_take >= self.horizon:
                self._done = True
            else:
                self._frame_index += num_steps_to_take
            print(self.timers)
        
        PPOEnvSplitUnifiedSimulation._step = _step_ppo_patched


class EnvironmentBuilder(object):
    """Builds an simulation environment for evaluation."""
    def __init__(self, eval_config: EvaluationConfig, exp_config: ExperimentConfig, device):
        self.eval_cfg = eval_config
        self.exp_cfg = exp_config
        self.device = device

    def _get_analytical_metrics(self):
        metrics = dict(
            all_off_road_rate=EnvMetrics.OffRoadRate(),
            all_disk_off_road_rate=EnvMetrics.DiskOffRoadRate(),
            # all_sem_layer_rate=EnvMetrics.SemLayerRate(),
            all_collision_rate=EnvMetrics.CollisionRate(),
            all_disk_collision_rate=EnvMetrics.DiskCollisionRate(),
            # agents_collision_rate=EnvMetrics.CollisionRate(),
            all_coverage=EnvMetrics.OccupancyCoverage(
                gridinfo={"offset": np.zeros(2), "step": 2.0*np.ones(2)},
            ),
            all_diversity=EnvMetrics.OccupancyDiversity(
                gridinfo={"offset": np.zeros(2), "step": 4.0*np.ones(2)},
            ),
            all_failure=EnvMetrics.CriticalFailure(num_offroad_frames=2),
            all_comfort=EnvMetrics.Comfort(sim_dt=self.exp_cfg.algo.step_time, stat_dt=0.5),
        )
        return metrics

    def _get_learned_metrics(self):
        perturbations = dict()
        if self.eval_cfg.perturb.enabled:
            for sigma in self.eval_cfg.perturb.OU.sigma:
                perturbations["OU_sigma_{}".format(sigma)] = OrnsteinUhlenbeckPerturbation(
                    theta=self.eval_cfg.perturb.OU.theta*np.ones(3),
                    sigma=sigma*np.array(self.eval_cfg.perturb.OU.scale))

        cvae_metrics = CVAEMetrics(
            eval_config=self.eval_cfg,
            device=self.device,
            ckpt_root_dir=self.eval_cfg.ckpt_root_dir,
        )

        learned_occu_metric = OccupancyMetrics(
            eval_config=self.eval_cfg,
            device=self.device,
            ckpt_root_dir=self.eval_cfg.ckpt_root_dir,
        )

        metrics = dict(
            all_cvae_metrics=cvae_metrics.get_metrics(self.eval_cfg,perturbations=perturbations,rolling = self.eval_cfg.cvae.rolling,
            rolling_horizon = self.eval_cfg.cvae.rolling_horizon,env=self.eval_cfg.env,),
            all_occu_likelihood=learned_occu_metric.get_metrics(self.eval_cfg,perturbations=perturbations,rolling = self.eval_cfg.occupancy.rolling,
            rolling_horizon = self.eval_cfg.occupancy.rolling_horizon,env=self.eval_cfg.env,)
        )
        return metrics

    def get_env(self):
        raise NotImplementedError


class EnvL5Builder(EnvironmentBuilder):
    def get_env(self):
        exp_cfg = self.exp_cfg.clone()
        os.environ["L5KIT_DATA_FOLDER"] = self.eval_cfg.dataset_path
        exp_cfg.unlock()

        dm = LocalDataManager(None)
        l5_config = translate_l5kit_cfg(exp_cfg)

        rasterizer = build_rasterizer(l5_config, dm)
        vectorizer = build_vectorizer(l5_config, dm)
        eval_zarr = ChunkedDataset(dm.require(exp_cfg.train.dataset_valid_key)).open()
        env_dataset = EgoDatasetMixed(l5_config, eval_zarr, vectorizer, rasterizer)
        l5_config = deepcopy(l5_config)
        l5_config["raster_params"]["raster_size"] = (500, 500)
        l5_config["raster_params"]["pixel_size"] = (0.3, 0.3)
        # l5_config["raster_params"]["ego_center"] = (0.5, 0.5)
        render_rasterizer = build_visualization_rasterizer_l5kit(l5_config, LocalDataManager(None))
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.distance_th_far = 1e+5  # keep controlling everything
        exp_cfg.env.simulation.disable_new_agents = True
        exp_cfg.env.generate_agent_obs = True
        exp_cfg.env.simulation.distance_th_close = 30  # control everything within this bound
        exp_cfg.env.rasterizer.filter_agents_threshold = 0.8  # control everything that's above this confidence threshold

        exp_cfg.lock()

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
        if self.eval_cfg.metrics.compute_learned_metrics:
            metrics.update(self._get_learned_metrics())

        env = EnvL5KitSimulation(
            exp_cfg.env,
            dataset=env_dataset,
            num_scenes=self.eval_cfg.num_scenes_per_batch,
            prediction_only=False,
            renderer=render_rasterizer,
            metrics=metrics,
            seed=self.eval_cfg.seed,
            skimp_rollout=self.eval_cfg.skimp_rollout,
        )

        return env


class EnvNuscBuilder(EnvironmentBuilder):
    def get_env(self,split_ego=False,parse_obs=True):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.train.dataset_path = self.eval_cfg.dataset_path
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        exp_cfg.lock()

        data_cfg = translate_trajdata_cfg(exp_cfg)

        # This needs to be done for every environment that wants
        # to have access to future trajectories during evaluation.
        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance

        kwargs = dict(
            desired_data=["val"],
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs={
                "nusc_trainval": data_cfg.dataset_path,
                "nusc_mini": data_cfg.dataset_path,
            },
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=True,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center
            },
            incl_vector_map=True,
            num_workers=os.cpu_count(),
            desired_dt=data_cfg.step_time,
            standardize_data=data_cfg.standardize_data,
            # A dictionary that contains functions that generate our custom data.
            # Can be any function and has access to the batch element.
            extras={
            },
        )
        
        env_dataset = UnifiedDataset(**kwargs)

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
        # if self.eval_cfg.metrics.compute_learned_metrics:
        #     metrics.update(self._get_learned_metrics())
        if split_ego:
            env = EnvSplitUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
                split_ego=split_ego,
                parse_obs = parse_obs,
            )
        else:
            env = EnvUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
            )

        return env

class EnvUnifiedBuilder(EnvironmentBuilder):
    def get_env(self):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        exp_cfg.lock()

        # the config used at training time
        # print('exp_cfg', exp_cfg)
        data_cfg = translate_pass_trajdata_cfg(exp_cfg)

        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance
        agent_only_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_only_types]
        agent_predict_types = None
        if data_cfg.trajdata_predict_types is not None:
            agent_predict_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_predict_types]
        # TBD: hack to accomodate drivesim rollout data when not enough future data is available
        if 'drivesim' in self.eval_cfg.trajdata_data_dirs.keys():
            future_sec_min = 0.1
        else:
            future_sec_min = future_sec
        kwargs = dict(
            cache_location=Path(self.eval_cfg.trajdata_cache_location).expanduser(),
            desired_data=self.eval_cfg.trajdata_source_test,
            desired_dt=data_cfg.step_time,
            future_sec=(future_sec_min, future_sec),#(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs=self.eval_cfg.trajdata_data_dirs,
            only_types=agent_only_types,
            only_predict=agent_predict_types,
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=data_cfg.trajdata_incl_map,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "no_map_fill_value": data_cfg.no_map_fill_value,
            },
            incl_vector_map=True,
            vector_map_params = {
                "incl_road_lanes": True,                
                "incl_road_areas": False,                 
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,                 
                # Collation can be quite slow if vector maps are included,
                "no_collate": True,
            },
            # centric=data_cfg.trajdata_centric,
            # hardcode to agent-centric and then use a wrapper to support scene-centric
            # TBD: support scene-centric simulation
            centric='agent',
            scene_description_contains=data_cfg.trajdata_scene_desc_contains,
            standardize_data=data_cfg.trajdata_standardize_data,
            verbose=True,
            num_workers=os.cpu_count(),
            rebuild_cache=self.eval_cfg.trajdata_rebuild_cache,
            rebuild_maps=self.eval_cfg.trajdata_rebuild_cache,
            # A dictionary that contains functions that generate our custom data.
            # Can be any function and has access to the batch element.
            extras={
                "closest_lane_point": get_closest_lane_point_wrapper(self.eval_cfg.evaluation_vec_map_params),
                "full_fut_traj": get_full_fut_traj,
                "full_fut_valid": get_full_fut_valid,
            },
        )

        env_dataset = UnifiedDataset(**kwargs)

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
        if self.eval_cfg.metrics.compute_learned_metrics:
            metrics.update(self._get_learned_metrics())

        # if we don't have a map, can't compute map-based metrics
        if not data_cfg.trajdata_incl_map:
            metrics.pop("all_off_road_rate", None)
            metrics.pop("all_sem_layer_rate", None)
            metrics.pop("all_coverage", None)
            metrics.pop("all_diversity", None)
            metrics.pop("all_failure", None)

        env = EnvUnifiedSimulation(
            exp_cfg.env,
            dataset=env_dataset,
            seed=self.eval_cfg.seed,
            num_scenes=self.eval_cfg.num_scenes_per_batch,
            prediction_only=False,
            metrics=metrics,
            save_action_samples=self.eval_cfg.save_action_samples,
        )

        return env

class EnvCLDBuilder(EnvironmentBuilder):
    def __init__(self, eval_config, exp_config, device, log_data=False):
        super().__init__(eval_config, exp_config, device)
        self.log_data = log_data
    
    def get_env(self):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        exp_cfg.lock()

        # the config used at training time
        # print('exp_cfg', exp_cfg)
        data_cfg = translate_cld_cfg(exp_cfg)

        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        # neighbor_distance = data_cfg.max_agents_distance
        agent_only_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_only_types]
        agent_predict_types = None
        if data_cfg.trajdata_predict_types is not None:
            agent_predict_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_predict_types]
        # TBD: hack to accomodate drivesim rollout data when not enough future data is available
        if 'drivesim' in self.eval_cfg.trajdata_data_dirs.keys():
            future_sec_min = 0.1
        else:
            future_sec_min = future_sec
        kwargs = dict(
            cache_location=data_cfg.trajdata_cache_location,
            desired_data=self.eval_cfg.trajdata_source_train,#data_cfg.trajdata_source_train,
            data_dirs={**data_cfg.trajdata_data_dirs, **self.eval_cfg.trajdata_data_dirs},
            desired_dt=data_cfg.step_time,
            only_types=agent_only_types,
            
            only_predict=agent_predict_types,
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            incl_robot_future=   data_cfg.incl_robot_future,
            ego_only         =getattr(self.eval_cfg, 'ego_only', None) if getattr(self.eval_cfg, 'ego_only', None) is not None else not data_cfg.incl_robot_future,
            max_neighbor_num = data_cfg.max_neighbor_num,
            agent_interaction_distances=defaultdict(lambda: data_cfg.max_agents_distance),
            incl_raster_map=data_cfg.trajdata_incl_map,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": True,
                "offset_frac_xy": data_cfg.raster_center,
                "no_map_fill_value": data_cfg.no_map_fill_value,
            },
            centric=data_cfg.trajdata_centric,

            standardize_data=data_cfg.trajdata_standardize_data,
            verbose=True,
            num_workers=os.cpu_count(),
            rebuild_cache=data_cfg.trajdata_rebuild_cache,
            rebuild_maps=data_cfg.trajdata_rebuild_cache,
            
            incl_vector_map=True,
            vector_map_params = {
                "incl_road_lanes": True,                
                "incl_road_areas": False,                 
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,                 
                # Collation can be quite slow if vector maps are included,
                "no_collate": True,
            },
            extras={
                "full_fut_traj": get_full_fut_traj,
                "full_fut_valid": get_full_fut_valid,
                "closest_lane_point": get_closest_lane_point_wrapper(self.eval_cfg.evaluation_vec_map_params),
                # "current_lane_point": get_current_lane_point_wrapper_visier(self.eval_cfg.evaluation_vec_map_params),
            },
        )

        env_dataset = UnifiedDataset(**kwargs)
        
        # _patch_env_with_custom_observation()

        env = EnvSplitUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                log_data=self.log_data,
                split_ego=True,
                parse_obs = True,
                save_action_samples=self.eval_cfg.save_action_samples,
            )

        return env