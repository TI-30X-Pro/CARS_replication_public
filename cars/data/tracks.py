"""Build 10 Hz per-scene agent tracks from nuScenes keyframes.

Per scene, at the 2 Hz key frames:
  - every vehicle / human annotated in more than one key frame becomes an agent;
    missing key frames inside its track are filled linearly (heading unwrapped);
  - the ego vehicle comes from the CAM_FRONT ego pose, with a fixed 4.084 x 1.730 m box;
  - velocity and acceleration are backward differences over 0.5 s, the first value
    duplicated from the second.
The table is then upsampled to 10 Hz by linear interpolation of every column, including
velocity and acceleration (so a and v are piecewise linear, not derivatives of x).
Columns: x, y, z, vx, vy, ax, ay, heading.
"""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from pyquaternion import Quaternion

NUSC_DT = 0.5
COLUMNS = ("x", "y", "z", "vx", "vy", "ax", "ay", "heading")
EGO_EXTENT = (4.084, 1.730, 1.562)   # length, width, height


class AgentType(IntEnum):            # same values as trajdata
    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    BICYCLE = 3
    MOTORCYCLE = 4


def agent_type(category):
    if category.startswith("human"):
        return AgentType.PEDESTRIAN
    if category == "vehicle.bicycle":
        return AgentType.BICYCLE
    if category == "vehicle.motorcycle":
        return AgentType.MOTORCYCLE
    if category.startswith("vehicle"):
        return AgentType.VEHICLE
    return AgentType.UNKNOWN


def angle_wrap(a):
    """Wrap to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw(q):
    return Quaternion(q).yaw_pitch_roll[0]


@dataclass
class SceneTracks:
    name: str
    location: str
    dt: float
    length: int                 # number of time steps at dt
    agents: list                # agent names; agents[0] == "ego"
    types: np.ndarray           # (N,) AgentType
    first: np.ndarray           # (N,) first time step
    last: np.ndarray            # (N,) last time step
    extent: np.ndarray          # (N, 3) length, width, height
    states: list                # per agent, (last - first + 1, 8) array in COLUMNS order

    def index(self, name):
        return self.agents.index(name)

    def state(self, i, ts):
        """State of agent i at time step ts (None if absent)."""
        if ts < self.first[i] or ts > self.last[i]:
            return None
        return self.states[i][ts - self.first[i]]

    def present(self, ts):
        """Indices of the agents present at time step ts, in agent order."""
        return np.nonzero((self.first <= ts) & (self.last >= ts))[0]


def _differences(pos):
    """Backward differences with the first value duplicated (trajdata's convention)."""
    prepend_pos = pos[0, :2] - (pos[1, :2] - pos[0, :2])
    vel = np.diff(pos[:, :2], axis=0, prepend=np.expand_dims(prepend_pos, axis=0)) / NUSC_DT
    prepend_vel = vel[0] - (vel[1] - vel[0])
    acc = np.diff(vel, axis=0, prepend=np.expand_dims(prepend_vel, axis=0)) / NUSC_DT
    return vel, acc


def _agent_keyframes(idx, a, frame_of_sample):
    """2 Hz table of one agent, starting at annotation a. Returns (first_frame, table)."""
    first = frame_of_sample[idx.ann_sample[a]]
    trans = [idx.ann_xyz[a][np.newaxis]]
    yaws = [yaw(idx.ann_rot[a])]
    prev = first
    a = idx.ann_next[a]
    while a >= 0:
        t = idx.ann_xyz[a]
        h = yaw(idx.ann_rot[a])
        cur = frame_of_sample[idx.ann_sample[a]]
        if cur > prev + 1:  # fill missing key frames
            fill = np.arange(prev + 1, cur)
            xs = np.interp(x=fill, xp=[prev, cur], fp=[trans[-1][0, 0], t[0]])
            ys = np.interp(x=fill, xp=[prev, cur], fp=[trans[-1][0, 1], t[1]])
            zs = np.interp(x=fill, xp=[prev, cur], fp=[trans[-1][0, 2], t[2]])
            hs = angle_wrap(np.interp(x=fill, xp=[prev, cur], fp=np.unwrap([yaws[-1], h])))
            trans.append(np.stack([xs, ys, zs], axis=1))
            yaws.extend(hs.tolist())
        trans.append(t[np.newaxis])
        yaws.append(h)
        prev = cur
        a = idx.ann_next[a]
    pos = np.concatenate(trans, axis=0)
    vel, acc = _differences(pos)
    heading = np.expand_dims(np.stack(yaws, axis=0), axis=1)
    return first, np.concatenate([pos, vel, acc, heading], axis=1)


def scene_keyframes(idx, s):
    """All agents of scene s at 2 Hz, in trajdata's order.

    Returns (names, types, first, extent, tables, n_frames). The order of the
    returned agents is ego first, then by first appearance; the value tables
    are stored by trajdata in a different order (agents first, ego last), which
    `upsample` needs to reproduce exactly.
    """
    frames = idx.scene_samples(s)
    frame_of_sample = {k: i for i, k in enumerate(frames)}
    anns_of = idx.sample_annotations()

    names, types, first, extent, tables = [], [], [], [], []
    seen = set()
    for k in frames:
        for a in anns_of[k]:
            inst = str(idx.ann_instance[a])
            if inst in seen or idx.ann_next[a] < 0:
                continue
            f0, table = _agent_keyframes(idx, a, frame_of_sample)
            w, l, h = idx.ann_size[a]
            names.append(inst)
            types.append(int(agent_type(str(idx.ann_category[a]))))
            first.append(f0)
            extent.append((l, w, h))
            tables.append(table)
            seen.add(inst)

    ego_pos = idx.ego_xyz[frames]
    vel, acc = _differences(ego_pos)
    ego_h = np.array([yaw(q) for q in idx.ego_rot[frames]])[:, None]
    ego_table = np.concatenate([ego_pos, vel, acc, ego_h], axis=1)

    names = ["ego"] + names
    types = [int(AgentType.VEHICLE)] + types
    first = [0] + first
    extent = [EGO_EXTENT] + extent
    tables = [ego_table] + tables
    return names, np.array(types), np.array(first), np.array(extent, dtype=np.float64), tables, len(frames)


def upsample(first, tables, factor):
    """Linear upsampling of 2 Hz tables, mirroring trajdata's DataFrameCache.

    trajdata interpolates one long table holding all agents (non-ego agents first,
    ego last), with the heading unwrapped over that whole column. Doing the same
    gives bit-identical numbers.
    """
    order = list(range(1, len(tables))) + [0]
    lengths = [len(tables[i]) for i in order]
    keyframes = np.concatenate([tables[i] for i in order], axis=0)
    keyframes[:, 7] = np.unwrap(keyframes[:, 7])

    n_up = [(n - 1) * factor + 1 for n in lengths]
    total = sum(n_up)
    rows = np.full((total, keyframes.shape[1]), np.nan)
    known = np.zeros(total, dtype=bool)
    start, k = 0, 0
    for n, m in zip(lengths, n_up):
        pos = start + np.arange(n) * factor
        rows[pos] = keyframes[k:k + n]
        known[pos] = True
        start += m
        k += n
    x = np.arange(total)
    for c in range(rows.shape[1]):
        rows[~known, c] = np.interp(x[~known], x[known], rows[known, c])
    rows[:, 7] = angle_wrap(rows[:, 7])

    out = [None] * len(tables)
    start = 0
    for i, m in zip(order, n_up):
        out[i] = rows[start:start + m]
        start += m
    return out


def scene_tracks(idx, s, dt=0.1):
    names, types, first, extent, tables, n_frames = scene_keyframes(idx, s)
    factor = int(round(NUSC_DT / dt))
    states = upsample(first, tables, factor)
    first = first * factor
    last = first + np.array([len(t) for t in states]) - 1
    return SceneTracks(
        name=str(idx.scene_name[s]),
        location=str(idx.scene_location[s]),
        dt=dt,
        length=(n_frames - 1) * factor + 1,
        agents=names,
        types=types,
        first=first,
        last=last,
        extent=extent,
        states=states,
    )


def scene_splits():
    """nuScenes scene names per split, as trajdata's nusc_trainval uses them."""
    from nuscenes.eval.prediction.splits import NUM_IN_TRAIN_VAL
    from nuscenes.utils.splits import create_splits_scenes

    splits = create_splits_scenes()
    train = list(splits["train"])
    return {"train": train[NUM_IN_TRAIN_VAL:], "train_val": train[:NUM_IN_TRAIN_VAL],
            "val": list(splits["val"])}
