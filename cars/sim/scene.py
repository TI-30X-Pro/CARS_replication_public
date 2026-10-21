"""Maintain one closed-loop scene and expose model-ready observations.

- Agents are frozen at the start step: every vehicle present then, in agent order (ego
  first). Agents whose own map crop shows no drivable area under them are dropped, except
  the ego.
- Nothing is integrated here. Each policy returns planned positions and yaws in its own
  frame at the last re-plan. Per step, every agent is placed at
  `plan_xy @ R(yaw0) + xy0` with yaw `yaw0 + plan_yaw`, in float32.
- The state table starts as the logged tracks of the frozen agents. A step overwrites the
  next row. Velocity and acceleration become backward differences over 0.1 s of the new
  positions. Logged rows further ahead stay, so "future" fields keep showing the log.
- The logged end of each track (`last`) is kept: futures stop there, as in trajdata.
"""
import numpy as np
import torch

from cars.data.batch import EncodingConfig, collate, element, parse
from cars.sim.geometry import affine_f32


class SimScene:
    def __init__(self, tracks, raster, start=21, horizon=120, cfg=EncodingConfig(), drop_offroad=True):
        self.cfg = cfg
        self.raster = raster
        self.name, self.location, self.dt = tracks.name, tracks.location, tracks.dt
        self.start = self.ts = start

        allowed = [int(t) for t in cfg.agent_types]
        frozen = [j for j in tracks.present(start) if tracks.types[j] in allowed]
        if not frozen:
            raise ValueError(f"{tracks.name}: no agents at step {start}")
        end = start + horizon + 1
        self.agents = [tracks.agents[j] for j in frozen]
        self.types = tracks.types[frozen]
        self.first = tracks.first[frozen]
        self.last = tracks.last[frozen]          # logged end, never moved
        self.extent = tracks.extent[frozen]
        self.states = []
        for j, f in zip(frozen, self.first):
            n = max(int(tracks.last[j]), end) - int(f) + 1
            arr = np.full((n, 8), np.nan)
            arr[: len(tracks.states[j])] = tracks.states[j]
            self.states.append(arr)

        if drop_offroad:
            obs = self.observe()
            keep = self._on_road(obs)
            self._select(keep)

    # Interface consumed by cars.data.batch.element
    def index(self, name):
        return self.agents.index(name)

    def state(self, i, ts):
        return self.states[i][ts - self.first[i]]

    def present(self, ts):
        return np.arange(len(self.agents))

    def _select(self, keep):
        self.agents = [self.agents[i] for i in keep]
        self.types, self.first, self.last, self.extent = (a[keep] for a in (self.types, self.first, self.last, self.extent))
        self.states = [self.states[i] for i in keep]

    def _on_road(self, obs):
        """Indices of agents whose own crop has drivable area under them (ego always kept)."""
        drivable = obs["drivable_map"]
        centroid, tf = obs["centroid"].numpy(), obs["raster_from_world"].numpy()
        keep = []
        for i, name in enumerate(self.agents):
            r = affine_f32(centroid[i], tf[i])   # about (56, 112), the agent's own pixel
            if name == "ego" or drivable[i, int(r[1]), int(r[0])].item() > 0:
                keep.append(i)
        return keep

    def observe(self, get_map=True):
        """Parsed agent-centric observation of every agent at the current step."""
        elems = [element(self, i, self.ts, self.raster, self.cfg, get_map=get_map)
                 for i in range(len(self.agents))]
        return parse(collate(elems, self.cfg), self.cfg)

    def place(self, xy, yaw):
        """Move every agent to world position xy (N, 2) and heading yaw (N,) at the next step."""
        ts = self.ts
        for i in range(len(self.agents)):
            prev = self.state(i, ts)
            new_x, new_y, h = xy[i, 0], xy[i, 1], yaw[i]
            vx = (new_x - prev[0]) / self.dt
            vy = (new_y - prev[1]) / self.dt
            ax = (vx - prev[3]) / self.dt
            ay = (vy - prev[4]) / self.dt
            self.states[i][ts + 1 - self.first[i]] = [new_x, new_y, np.nan, vx, vy, ax, ay, h]
        self.ts = ts + 1

    @staticmethod
    def plan_to_world(obs, plan_xy, plan_yaw, k):
        """World poses at plan step k (float32): plan_xy (N, T, 2), plan_yaw (N, T, 1)
        in each agent's frame of `obs`."""
        curr = obs["curr_agent_state"].numpy() if isinstance(obs["curr_agent_state"], torch.Tensor) \
            else obs["curr_agent_state"]
        n = curr.shape[0]
        xy = np.full((n, 2), np.nan, dtype=np.float32)
        yaw = np.full((n,), np.nan, dtype=np.float32)
        for i in range(n):
            h, pos = curr[i, -1], curr[i, :2]
            c, s = np.cos(h), np.sin(h)
            p = plan_xy[i, k]
            if not np.any(np.isnan(p)):
                # p @ [[c, s], [-s, c]] + pos, product by product in float32
                xy[i] = affine_f32(p, np.array([[c, -s, pos[0]], [s, c, pos[1]], [0, 0, 1]], dtype=np.float32))
                yaw[i] = h + plan_yaw[i, k, 0]
        return xy, yaw
