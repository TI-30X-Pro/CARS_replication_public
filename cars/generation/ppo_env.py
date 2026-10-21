"""Collect adversarial episodes and run the PPO training loop.

Episodes: the labelled pairs of the training scenes (rollout scenes 0-20), scene after scene,
pair after pair, cycling. The adversary is the labelled one for the whole episode (no
selector). Every call: the adversary's chain is sampled by the PPO policy, target and others
draw one plan each from the prior (stationary mask from the first call), all agents follow
their plans for 5 steps, then the reward is computed. An episode ends
- on a collision (reward with the hit bonus; terminated),
- when the target leaves the adversary's neighbours (reward -0.1 (remaining calls + 5);
  terminated),
- after 120 steps (truncated).

Loop: per epoch, collect 5 episodes and update until 120 transitions were collected, then
test 3 episodes in eval mode (BatchNorm on running statistics) on a separate copy of the
episode list; the best mean test reward so far is saved. As in tianshou, a test starts by
resetting its environment (which skips a pair), the collectors reset their environment once
when created, and the initial policy is tested and saved before training.
"""
import itertools
import os
import time

import numpy as np
import torch

from cars.generation.policy import Planner, to_device
from cars.generation.ppo import stack
from cars.generation.reward import ApproachReward, adversary_view

VIEW_KEYS = ("maps", "history_positions", "history_speeds", "history_yaws", "extent", "history_availabilities",
             "all_other_agents_history_positions", "all_other_agents_history_speeds",
             "all_other_agents_history_yaws", "all_other_agents_history_extents",
             "all_other_agents_history_availability", "extras", "curr_speed", "curr_speed_signed",
             "raster_from_agent", "world_from_agent", "target_positions", "target_yaws", "target_availabilities",
             "neigh_indices", "map_is_singapore")


def row(obs, i):
    """Agent i's observation as numpy, without the batch dimension."""
    out = {}
    for k in VIEW_KEYS:
        v = obs[k]
        out[k] = {kk: vv[i].numpy() for kk, vv in v.items()} if isinstance(v, dict) else v[i].numpy()
    return out


def take(batch, idx):
    if isinstance(batch, torch.Tensor):
        return batch.index_select(0, idx)
    if isinstance(batch, dict):
        return {k: take(v, idx) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [batch[i] for i in idx.tolist()]
    return batch


class AdversaryEnv:
    def __init__(self, scenes, pairs, make_scene, adversary, prior_target, prior_others, n_steps=5, horizon=120):
        """scenes: names in order; pairs: {scene: [[target, adversary], ...]};
        make_scene(name) -> SimScene at the rollout start; adversary: the GMDM being trained
        (for decoding actions); prior_*: GMDMs for the target and the other agents."""
        self.scene_iter = itertools.cycle(scenes)
        self.pairs, self.make_scene, self.adv_net = pairs, make_scene, adversary
        self.target_policy = Planner(prior_target, 1)
        self.others_policy = Planner(prior_others, 1)
        self.n_steps, self.horizon = n_steps, horizon
        self.reward_fn = ApproachReward()
        self.scene, self.pairs_left, self.view = None, 0, None

    def reset(self):
        if self.scene is None or self.pairs_left <= 0:
            self.scene = next(self.scene_iter)
            self.pairs_left = len(self.pairs[self.scene])
        pair = len(self.pairs[self.scene]) - self.pairs_left
        self.pairs_left -= 1
        self.sim = self.make_scene(self.scene)
        target, adv = self.pairs[self.scene][pair]
        self.t_idx, self.a_idx = self.sim.agents.index(target), self.sim.agents.index(adv)
        self._refresh()
        self.frame, self.done = 0, False
        return self.view

    def _refresh(self):
        self.obs = self.sim.observe(get_map=True)
        view = adversary_view(row(self.obs, self.a_idx), self.t_idx)
        if view is None:                  # target no longer a neighbour: keep the last view
            self.done = True
            return
        self.view = view
        self.adv_extent = self.obs["extent"][self.a_idx, :2]
        self.target_extent = self.obs["extent"][self.t_idx, :2]

    def step(self, chain):
        """chain: (K+1, T, 2) normalised actions; executes x_0."""
        n = self.n_steps
        with torch.no_grad():
            dev = next(self.adv_net.parameters()).device
            v = to_device(stack([self.view]), dev)
            st = self.adv_net.states(torch.as_tensor(chain[-1], device=dev)[None], self.adv_net.current_states(v)).cpu()
        M = len(self.sim.agents)
        xy = np.zeros((M, st.shape[1], 2), np.float32)
        yaw = np.zeros((M, st.shape[1], 1), np.float32)
        xy[self.a_idx], yaw[self.a_idx] = st[0, :, :2].numpy(), st[0, :, 3:4].numpy()
        i_t = torch.tensor([self.t_idx])
        pos, yw, _ = self.target_policy(take(self.obs, i_t), self.frame)
        xy[self.t_idx], yaw[self.t_idx] = pos[0].numpy(), yw[0].numpy()
        others = torch.tensor([i for i in range(M) if i not in (self.a_idx, self.t_idx)], dtype=torch.long)
        if len(others):
            pos, yw, _ = self.others_policy(take(self.obs, others), self.frame)
            xy[others.numpy()], yaw[others.numpy()] = pos.numpy(), yw.numpy()

        obs = self.obs
        for k in range(n):
            p, h = self.sim.plan_to_world(obs, xy, yaw, k)
            self.sim.place(p, h)
        self._refresh()
        self.frame += n

        info = {}
        if self.done:                     # target lost
            reward = -self.reward_fn.time_penalty * (max((self.horizon - self.frame) // n, 0) + 5)
        else:
            reward, info = self.reward_fn(self.view, self.t_idx, self.target_extent, self.adv_extent, n)
            if info.get("collision", 0.0) > 0.5:
                self.done = True
        terminated = self.done
        truncated = not terminated and self.frame >= self.horizon
        return self.view, float(reward), terminated, truncated, info


class Collector:
    """Episodes from one environment; the environment is reset at creation and after every
    episode, as tianshou's Collector does."""

    def __init__(self, env, ppo):
        self.env, self.ppo = env, ppo
        self.reset_env()

    def reset_env(self):
        self.obs = self.env.reset()

    def collect(self, n_episode):
        data = {"obs": [], "act": [], "rew": [], "terminated": [], "truncated": [], "obs_next": [], "extents": []}
        episodes, ep_rew, infos = [], 0.0, []
        while len(episodes) < n_episode:
            chain = self.ppo.act(stack([self.obs]))[0].numpy()
            obs_next, rew, term, trunc, info = self.env.step(chain)
            extents = np.stack([np.asarray(self.env.target_extent), np.asarray(self.env.adv_extent)])
            for k, v in (("obs", self.obs), ("act", chain), ("rew", rew), ("terminated", term),
                         ("truncated", trunc), ("obs_next", obs_next), ("extents", extents)):
                data[k].append(v)
            ep_rew += rew
            self.obs = obs_next
            if term or trunc:
                episodes.append(ep_rew)
                infos.append(info)
                ep_rew = 0.0
                self.reset_env()
        return data, np.array(episodes), infos


def batch_data(data):
    return {"obs": stack(data["obs"]), "obs_next": stack(data["obs_next"]), "act": np.stack(data["act"]),
            "rew": np.array(data["rew"]), "terminated": np.array(data["terminated"]),
            "truncated": np.array(data["truncated"]), "extents": np.stack(data["extents"])}


def save_checkpoint(net, path, **meta):
    """The adversary in the released checkpoints' format (keys nets.policy.*)."""
    torch.save(net.checkpoint(**meta), path)


def train(ppo, train_env, test_env, out_dir, epochs=500, episodes_per_collect=5, steps_per_epoch=120,
          test_episodes=3, log=print):
    os.makedirs(out_dir, exist_ok=True)
    train_c, test_c = Collector(train_env, ppo), Collector(test_env, ppo)

    def test():
        test_c.reset_env()
        ppo.train(False)
        _, rews, _ = test_c.collect(test_episodes)
        return float(rews.mean())

    def gpu_gb():
        if ppo.device.type != "cuda":
            return None
        gb = torch.cuda.max_memory_allocated(ppo.device) / 1e9
        torch.cuda.reset_peak_memory_stats(ppo.device)
        return round(gb, 2)

    t0 = time.time()
    best = test()
    save_checkpoint(ppo.net, os.path.join(out_dir, "adversary_best.ckpt"), epoch=0, test_reward=best)
    log({"epoch": 0, "test_reward": best, "seconds": round(time.time() - t0, 1), "gpu_peak_gb": gpu_gb()})
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        ppo.train(True)
        steps, rews, collisions = 0, [], []
        while steps < steps_per_epoch:
            data, ep_rews, infos = train_c.collect(episodes_per_collect)
            steps += len(data["rew"])
            rews += ep_rews.tolist()
            collisions += [i.get("collision", 0.0) for i in infos]
            losses = ppo.update(batch_data(data))
        r_test = test()
        row = {"epoch": epoch, "train_reward": float(np.mean(rews)), "collision_rate": float(np.mean(collisions)),
               "episodes": len(rews), "steps": steps, "test_reward": r_test,
               "loss": float(np.mean(losses["loss"])) if losses["loss"] else float("nan"),
               "seconds": round(time.time() - t0, 1), "gpu_peak_gb": gpu_gb()}
        history.append(row)
        log(row)
        if r_test > best:
            best = r_test
            save_checkpoint(ppo.net, os.path.join(out_dir, "adversary_best.ckpt"), epoch=epoch, test_reward=best)
        save_checkpoint(ppo.net, os.path.join(out_dir, "adversary_last.ckpt"), epoch=epoch)
    return history
