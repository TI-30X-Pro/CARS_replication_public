"""Build agent-centric history, future, neighbour, and map tensors.

One element = one agent at one time step ts, everything in that agent's frame:
  - history: the agent's own states at ts-20 .. ts (2 s + now), padded with NaN in front;
  - future: ts+1 .. ts+20, padded with NaN at the end;
  - neighbours: every agent of an allowed type present at ts and within 50 m, nearest first,
    with their own histories and futures (a neighbour whose track ends at ts gets its
    current state as a one-step "future", as in trajdata);
  - map: a 224 x 224 crop of the map raster at 1 px/m, rotated so the agent faces +x, with
    the agent 56 px from the left edge.
States are x, y, vx, vy, ax, ay, sin h, cos h. Positions, velocities and accelerations are
rotated into the agent frame; headings are relative to the agent's heading.
`parse` then turns the collated batch into the fields the model reads (tbsim's
parse_trajdata_batch): positions, yaws, speeds, availabilities, extents, transforms.
"""
from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
import torch
import torch.nn.functional as F

from cars.data.tracks import AgentType


@dataclass
class EncodingConfig:
    history_steps: int = 20           # plus the current step
    future_steps: int = 20
    neighbor_radius: float = 50.0
    agent_types: tuple = (AgentType.VEHICLE,)
    raster_size: int = 224
    px_per_m: float = 1.0
    ego_center: tuple = (-0.5, 0.0)   # agent position in the crop, as a fraction of half its size
    full_future_steps: int = 200      # tbsim's extra "full future" fields
    env_name: str = "nusc_trainval"


def inverse_f32(m):
    """Invert `(B, 3, 3)` matrices with deterministic legacy float32 arithmetic.

    Uses LAPACK-style getrf/getri operations instead of the torch 2.x solve path so transform
    values remain stable at boundary pixels.
    """
    is_torch = isinstance(m, torch.Tensor)
    a = (m.numpy() if is_torch else np.asarray(m)).astype(np.float32, copy=True)
    rows = np.arange(len(a))
    piv = np.zeros((len(a), 3), dtype=np.int64)
    for j in range(3):                                           # getf2
        p = j + np.argmax(np.abs(a[:, j:, j]), axis=1)
        piv[:, j] = p
        tmp = a[rows, j].copy()
        a[rows, j] = a[rows, p]
        a[rows, p] = tmp
        if j < 2:
            a[:, j + 1:, j] *= (np.float32(1) / a[:, j, j])[:, None]
            a[:, j + 1:, j + 1:] -= a[:, j + 1:, j, None] * a[:, j, None, j + 1:]
    for j in range(3):                                           # trti2: inv(U)
        a[:, j, j] = np.float32(1) / a[:, j, j]
        x = a[:, :j, j].copy()
        for jj in range(j):
            t = x[:, jj].copy()
            for i in range(jj):
                x[:, i] = x[:, i] + t * a[:, i, jj]
            x[:, jj] = x[:, jj] * a[:, jj, jj]
        a[:, :j, j] = x * (-a[:, j, j])[:, None]
    for j in (2, 1, 0):                                          # getri: inv(A) L = inv(U)
        work = a[:, :, j].copy()
        a[:, j + 1:, j] = 0
        for jj in range(j + 1, 3):
            a[:, :, j] = a[:, :, j] + (-work[:, jj])[:, None] * a[:, :, jj]
    for j in (1, 0):                                             # undo the row swaps
        jp = piv[:, j]
        tmp = a[rows, :, j].copy()
        a[rows, :, j] = a[rows, :, jp]
        a[rows, :, jp] = tmp
    return torch.from_numpy(a) if is_torch else a


# Build one agent-time element

def to_agent_frame(states, frame_state):
    """World states (n, 8: x y z vx vy ax ay h) -> agent-frame obs (n, 8: x y vx vy ax ay s c)."""
    frame = frame_state.copy()
    frame[2:7] = 0.0                  # the frame does not move and sits on the ground
    obs = states - frame
    h = -frame_state[7]
    rot = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    obs[..., (0, 1, 3, 4, 5, 6)] = (obs[..., (0, 1, 3, 4, 5, 6)].reshape(-1, 3, 2) @ rot.T[None, :, :]).reshape(-1, 6)
    return np.concatenate([obs[:, [0, 1, 3, 4, 5, 6]], np.sin(obs[:, 7:8]), np.cos(obs[:, 7:8])], axis=1)


def _rows(tracks, i, t0, t1):
    """States of agent i for time steps t0..t1 (inclusive)."""
    return tracks.states[i][t0 - tracks.first[i]: t1 - tracks.first[i] + 1]


def map_patch(raster, x, y, heading, cfg):
    """The square around (x, y) that still covers the final crop after rotation.

    Returns the patch (C, 318, 318) and its raster_from_world transform.
    """
    tf = raster.map_from_world
    map_x, map_y = (tf @ np.array([x, y, 1.0]))[:2]
    map_x, map_y = map_x.item(), map_y.item()
    raster_from_world = np.array([[1.0, 0.0, -map_x], [0.0, 1.0, -map_y], [0.0, 0.0, 1.0]]) @ tf

    size = ceil(cfg.raster_size * raster.px_per_m / cfg.px_per_m)
    if tuple(cfg.ego_center) != (0.0, 0.0):
        offset = (-cfg.ego_center[0] * size // 2, cfg.ego_center[1] * size // 2)
        rot = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
        off_x, off_y = rot @ offset
        map_x += off_x
        map_y += off_y
        raster_from_world = np.array([[1.0, 0.0, -off_x], [0.0, 1.0, -off_y], [0.0, 0.0, 1.0]]) @ raster_from_world

    padded = ceil((sqrt(2) * size) / 2) * 2
    half = padded // 2
    map_x, map_y = round(map_x), round(map_y)
    top, bot, left, right = map_y - half, map_y + half, map_x - half, map_x + half
    data = raster.data
    _, h, w = data.shape
    patch = data[:, max(top, 0):min(bot, h), max(left, 0):min(right, w)]
    if patch.shape[-2:] != (padded, padded):
        if bot <= 0 or top >= h or right <= 0 or left >= w:
            patch = np.zeros((data.shape[0], padded, padded))
        else:
            patch = np.pad(patch, [(0, 0), (max(-top, 0), max(bot - h, 0) if bot >= h else 0),
                                   (max(-left, 0), max(right - w, 0) if right >= w else 0)])
    return patch, raster_from_world


def element(tracks, agent, ts, raster, cfg=EncodingConfig(), get_map=True):
    """Everything about one agent at one time step, before batching (numpy, float64).

    `tracks` is a SceneTracks, or a SimScene during closed-loop simulation.
    """
    i = tracks.index(agent) if isinstance(agent, str) else agent
    raw = tracks.state(i, ts)
    x, y, h = raw[0], raw[1], raw[7]

    hist_t0 = max(ts - cfg.history_steps, tracks.first[i])
    hist = to_agent_frame(_rows(tracks, i, hist_t0, ts), raw)
    if ts >= tracks.last[i]:
        fut = np.zeros((0, 8))
    else:
        fut = to_agent_frame(_rows(tracks, i, ts + 1, min(ts + cfg.future_steps, tracks.last[i])), raw)
    full_t1 = min(ts + cfg.full_future_steps, tracks.last[i])
    full_fut = (to_agent_frame(_rows(tracks, i, ts + 1, full_t1), raw)
                if ts < tracks.last[i] else np.zeros((0, 8)))

    # neighbours: agents of the allowed types present at ts, nearest first
    present = [j for j in tracks.present(ts) if tracks.types[j] in [int(t) for t in cfg.agent_types]]
    cur = np.stack([tracks.state(j, ts) for j in present])
    pos = to_agent_frame(cur, raw)[:, :2]
    own = to_agent_frame(raw[None], raw)[:, :2]
    dist = np.linalg.norm(pos - own, axis=1)
    me = present.index(i)
    near = dist <= cfg.neighbor_radius
    near[me] = False
    order = [k for k in dist.argsort() if near[k]]
    neighbors = [present[k] for k in order]

    n_hist, n_fut = [], []
    all_ended = bool(neighbors) and all(tracks.last[j] < ts for j in neighbors)
    for j in neighbors:
        t0 = max(ts - cfg.history_steps, tracks.first[j])
        n_hist.append(to_agent_frame(_rows(tracks, j, t0, ts), raw))
        if all_ended:   # only possible in simulation, past the end of every log
            n_fut.append(np.zeros((0, 8)))
            continue
        t1 = min(ts + cfg.future_steps, tracks.last[j])
        n_fut.append(to_agent_frame(_rows(tracks, j, min(ts + 1, tracks.last[j]), t1), raw))

    world_from_agent = np.array([[np.cos(h), -np.sin(h), x], [np.sin(h), np.cos(h), y], [0.0, 0.0, 1.0]])
    patch, raster_from_world = map_patch(raster, x, y, h, cfg) if get_map else (None, None)

    ext = tracks.extent
    return dict(
        scene=tracks.name, agent=tracks.agents[i], ts=ts, dt=tracks.dt,
        agent_type=int(tracks.types[i]),
        curr_state=raw[[0, 1, 3, 4, 5, 6, 7]],
        hist=hist, hist_extent=np.repeat(ext[i][None], len(hist), axis=0),
        fut=fut, fut_extent=np.repeat(ext[i][None], len(fut), axis=0),
        full_fut=full_fut, full_fut_extent=np.repeat(ext[i][None], len(full_fut), axis=0),
        neigh_indices=np.array(order), neigh_types=np.array([int(tracks.types[j]) for j in neighbors]),
        neigh_hist=n_hist, neigh_hist_extent=[np.repeat(ext[j][None], len(a), axis=0) for j, a in zip(neighbors, n_hist)],
        neigh_fut=n_fut, neigh_fut_extent=[np.repeat(ext[j][None], len(a), axis=0) for j, a in zip(neighbors, n_fut)],
        agent_from_world=np.linalg.inv(world_from_agent),
        map_name=f"{cfg.env_name}:{tracks.location}",
        map_patch=patch, map_heading=h, raster_from_world=raster_from_world,
    )


# Collate elements

def _pad_front(arrs, length, dim):
    out = torch.full((len(arrs), length, dim), float("nan"))
    for k, a in enumerate(arrs):
        if len(a):
            out[k, length - len(a):] = torch.as_tensor(a, dtype=torch.float)
    return out


def _pad_back(arrs, length, dim):
    out = torch.full((len(arrs), length, dim), float("nan"))
    for k, a in enumerate(arrs):
        if len(a):
            out[k, :len(a)] = torch.as_tensor(a, dtype=torch.float)
    return out


_PI = torch.tensor(3.14159265358979323846)


def rotate(img, angle_deg):
    """Rotate images (B, C, H, W) anti-clockwise about their centre, bilinear, zero padding.

    The same arithmetic as kornia 0.6.4's `rotate`, which the original uses.
    """
    b, _, h, w = img.shape
    dt = img.dtype
    center = torch.tensor([float(w - 1) / 2, float(h - 1) / 2], dtype=dt).expand(b, -1)
    eye = torch.eye(3, dtype=dt)[None].repeat(b, 1, 1)
    shift, shift_inv, scale, rot = eye.clone(), eye.clone(), eye.clone(), eye.clone()
    shift[:, :2, 2] = center
    shift_inv[:, :2, 2] = -center
    rad = angle_deg * _PI.type(dt) / 180.0
    cos_a, sin_a = torch.cos(rad), torch.sin(rad)
    rot[:, :2, :2] = torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(b, 2, 2)
    affine = (shift @ rot @ scale @ shift_inv)[:, :2, :]

    m = F.pad(affine, [0, 0, 0, 1], "constant", value=0.0)
    m[..., -1, -1] += 1.0
    norm = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], dtype=dt)
    norm[0, 0] = norm[0, 0] * 2.0 / (w - 1.0)
    norm[1, 1] = norm[1, 1] * 2.0 / (h - 1.0)
    norm = norm[None]
    dst_from_src = norm @ (m @ inverse_f32(norm))
    src_from_dst = inverse_f32(dst_from_src)
    grid = F.affine_grid(src_from_dst[:, :2, :], [b, img.shape[1], h, w], align_corners=True)
    return F.grid_sample(img, grid, align_corners=True, mode="bilinear", padding_mode="zeros")


def _collate_maps(elems, cfg):
    """Rotate the map patches so each agent faces +x, then crop the centre."""
    b = len(elems)
    patches = torch.as_tensor(np.stack([e["map_patch"] for e in elems]), dtype=torch.float)
    angles = torch.as_tensor([e["map_heading"] for e in elems], dtype=torch.float)
    rasters_from_world = torch.as_tensor(np.stack([e["raster_from_world"] for e in elems]), dtype=torch.float)
    size, half = cfg.raster_size, cfg.raster_size // 2
    cy, cx = patches.shape[-2] // 2, patches.shape[-1] // 2
    if torch.count_nonzero(angles) == 0 and size == patches.shape[-1] == patches.shape[-2]:
        shift = torch.tensor([[[1.0, 0.0, half], [0.0, 1.0, half], [0.0, 0.0, 1.0]]]).expand(b, -1, -1)
        maps = patches
    else:
        maps = rotate(patches, torch.rad2deg(angles))[..., cy - half:cy + half, cx - half:cx + half]
        cos_v, sin_v = torch.cos(-angles), torch.sin(-angles)
        t = torch.tensor([[half, half]]).expand(b, -1)
        shift = torch.stack([torch.stack([cos_v, -sin_v, t[:, 0]], -1),
                             torch.stack([sin_v, cos_v, t[:, 1]], -1),
                             torch.tensor([[0.0, 0.0, 1.0]]).expand(b, -1)], dim=-2)
    rasters_from_world = torch.bmm(shift, rasters_from_world)
    return maps, rasters_from_world


def collate(elems, cfg=EncodingConfig()):
    """Batch elements like trajdata's agent_collate_fn (NaN where nothing is known)."""
    b = len(elems)
    hist_len = cfg.history_steps + 1
    n_neigh = torch.tensor([len(e["neigh_indices"]) for e in elems], dtype=torch.long)
    max_n = int(n_neigh.max()) if b else 0
    nh_lens = [[len(a) for a in e["neigh_hist"]] for e in elems]
    nf_lens = [[len(a) for a in e["neigh_fut"]] for e in elems]
    max_nh = max([max(l) for l in nh_lens if l], default=0)
    max_nf = max([max(l) for l in nf_lens if l], default=0)

    def per_neighbor(key, length, dim, front):
        out = torch.full((b, max_n, length, dim), float("nan"))
        for k, e in enumerate(elems):
            if e[key]:
                out[k, :len(e[key])] = (_pad_front if front else _pad_back)(e[key], length, dim)
        return out

    def lens(ls):
        out = torch.zeros((b, max_n), dtype=torch.long)
        for k, l in enumerate(ls):
            out[k, :len(l)] = torch.tensor(l, dtype=torch.long)
        return out

    neigh_idx = torch.full((b, max_n), -1.0)
    neigh_types = torch.full((b, max_n), -1.0)
    for k, e in enumerate(elems):
        neigh_idx[k, :len(e["neigh_indices"])] = torch.as_tensor(e["neigh_indices"], dtype=torch.float)
        neigh_types[k, :len(e["neigh_types"])] = torch.as_tensor(e["neigh_types"], dtype=torch.float)

    if elems[0]["map_patch"] is None:
        maps = map_names = maps_resolution = rasters_from_world = None
    else:
        maps, rasters_from_world = _collate_maps(elems, cfg)
        map_names = [e["map_name"] for e in elems]
        maps_resolution = torch.full((b,), float(cfg.px_per_m))

    full = cfg.full_future_steps

    def pad_full(a, dim):
        out = torch.zeros((full, dim))
        n = min(len(a), full)
        if n:
            out[:n] = torch.as_tensor(np.asarray(a[:n]), dtype=torch.float)
        return out

    return dict(
        scene_ts=torch.tensor([e["ts"] for e in elems], dtype=torch.int),
        dt=torch.tensor([e["dt"] for e in elems], dtype=torch.float),
        agent_name=[e["agent"] for e in elems],
        agent_type=torch.tensor([e["agent_type"] for e in elems], dtype=torch.int),
        curr_agent_state=torch.as_tensor(np.stack([e["curr_state"] for e in elems]), dtype=torch.float),
        agent_hist=_pad_front([e["hist"] for e in elems], hist_len, 8),
        agent_hist_extent=_pad_front([e["hist_extent"] for e in elems], hist_len, 3),
        agent_hist_len=torch.tensor([len(e["hist"]) for e in elems], dtype=torch.long),
        agent_fut=_pad_back([e["fut"] for e in elems], cfg.future_steps, 8),
        agent_fut_extent=_pad_back([e["fut_extent"] for e in elems], cfg.future_steps, 3),
        agent_fut_len=torch.tensor([len(e["fut"]) for e in elems], dtype=torch.long),
        num_neigh=n_neigh,
        neigh_indices=neigh_idx,
        neigh_types=neigh_types,
        neigh_hist=per_neighbor("neigh_hist", max_nh, 8, True),
        neigh_hist_extents=per_neighbor("neigh_hist_extent", max_nh, 3, True),
        neigh_hist_len=lens(nh_lens),
        neigh_fut=per_neighbor("neigh_fut", max_nf, 8, False),
        neigh_fut_extents=per_neighbor("neigh_fut_extent", max_nf, 3, False),
        neigh_fut_len=lens(nf_lens),
        robot_fut_len=torch.zeros(b, dtype=torch.long),
        map_names=map_names,
        maps=maps,
        maps_resolution=maps_resolution,
        rasters_from_world_tf=rasters_from_world,
        agents_from_world_tf=torch.as_tensor(np.stack([e["agent_from_world"] for e in elems]), dtype=torch.float),
        scene_ids=[e["scene"] for e in elems],
        extras={
            "full_fut_traj": torch.stack([pad_full(e["full_fut"], 8) for e in elems]),
            "full_fut_valid": torch.stack([pad_full(e["full_fut_extent"], 3)[:, 0] for e in elems]),
        },
    )


# Convert to model inputs

def _angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def pos_yaw_speed(state):
    """(…, 8) states -> positions, yaws, speeds, availability; unknown entries set to 0."""
    pos = state[..., :2].clone()
    yaw = _angle_wrap(torch.atan2(state[..., [-2]], state[..., [-1]]))
    speed = torch.norm(state[..., 2:4], dim=-1)
    mask = torch.bitwise_not(torch.max(torch.isnan(state), dim=-1)[0])
    pos[~mask] = 0.0
    yaw[~mask] = 0.0
    speed[~mask] = 0.0
    return pos, yaw, speed, mask


def _lyft_types(t):
    """tbsim converts nuScenes type ids to l5kit ones (in place, in this order)."""
    t[t < 0] = 0
    t[t == 3] = 10
    t[t == 1] = 3
    t[t == 2] = 14
    t[t == 4] = 11
    return t


def pad_neighbors(batch):
    """tbsim's maybe_pad_neighbor, in place.

    No agent in the batch has a neighbour (all > 50 m apart): add one fully masked slot, so
    the model's neighbour tensors are never empty. Otherwise front-pad neighbour histories
    to the agent's history length. neigh_indices gets -1 in the dummy slot (tbsim leaves it
    empty) so every neighbour tensor has the same width.
    """
    b, n, t, d = batch["neigh_hist"].shape
    hist_len, fut_len = batch["agent_hist"].shape[1], batch["agent_fut"].shape[1]
    e = batch["neigh_hist_extents"].shape[-1]
    if n == 0:
        batch["neigh_hist"] = torch.full((b, 1, hist_len, d), float("nan"))
        batch["neigh_fut"] = torch.full((b, 1, fut_len, d), float("nan"))
        batch["neigh_types"] = torch.zeros(b, 1)
        batch["neigh_indices"] = torch.full((b, 1), -1.0)
        batch["neigh_hist_extents"] = torch.zeros(b, 1, hist_len, e)
        batch["neigh_fut_extents"] = torch.zeros(b, 1, fut_len, e)
    elif t < hist_len:
        batch["neigh_hist"] = torch.cat([torch.full((b, n, hist_len - t, d), float("nan")),
                                         batch["neigh_hist"]], dim=2)
        batch["neigh_hist_extents"] = torch.cat([torch.zeros(b, n, hist_len - t, e),
                                                 batch["neigh_hist_extents"]], dim=2)


def parse(batch, cfg=EncodingConfig()):
    """tbsim's parse_trajdata_batch for agent-centric batches."""
    batch = dict(batch)
    pad_neighbors(batch)
    fut_pos, fut_yaw, _, fut_mask = pos_yaw_speed(batch["agent_fut"])
    hist_pos, hist_yaw, hist_speed, hist_mask = pos_yaw_speed(batch["agent_hist"])
    last = batch["agent_hist"][..., -1, :]
    curr_speed_signed = last[..., 2] * last[..., -1] + last[..., 3] * last[..., -2]
    curr = batch["curr_agent_state"]

    agent_type = _lyft_types(batch["agent_type"])
    hist_extent = batch["agent_hist_extent"]
    hist_extent[torch.isnan(hist_extent)] = 0.0
    n_pos, n_yaw, n_speed, n_mask = pos_yaw_speed(batch["neigh_hist"])
    nf_pos, nf_yaw, _, nf_mask = pos_yaw_speed(batch["neigh_fut"])
    n_types = _lyft_types(batch["neigh_types"])
    n_extents = batch["neigh_hist_extents"]
    n_extents[torch.isnan(n_extents)] = 0.0

    agent_from_world = batch["agents_from_world_tf"]
    res = 1.0 / (1.0 / cfg.px_per_m)
    size = cfg.raster_size
    raster_from_agent = torch.Tensor([[res, 0, ((1.0 + cfg.ego_center[0]) / 2.0) * size],
                                      [0, res, ((1.0 + cfg.ego_center[1]) / 2.0) * size],
                                      [0, 0, 1]])
    b = agent_from_world.shape[0]
    agent_from_raster = inverse_f32(raster_from_agent[None])[0]
    raster_from_agent = raster_from_agent[None].expand(b, -1, -1)
    agent_from_raster = agent_from_raster[None].expand(b, -1, -1)

    maps = batch["maps"]
    drivable = None
    if maps is not None:
        drivable = torch.amax(maps[..., [0], :, :], dim=-3)
        no_map = torch.sum(torch.isclose(drivable[:, None], torch.tensor([-1.0])), dim=[1, 2, 3]) == drivable[0].numel()
        drivable[no_map] = 1.0
        drivable = drivable.bool()

    d = dict(
        image=maps,
        drivable_map=drivable,
        target_positions=fut_pos, target_yaws=fut_yaw, target_availabilities=fut_mask,
        history_positions=hist_pos, history_yaws=hist_yaw, history_speeds=hist_speed,
        history_availabilities=hist_mask,
        curr_speed=hist_speed[..., -1], curr_speed_signed=curr_speed_signed,
        centroid=curr[:, :2], yaw=curr[:, -1], type=agent_type,
        extent=hist_extent.max(dim=-2)[0],
        raster_from_agent=raster_from_agent, agent_from_raster=agent_from_raster,
        raster_from_world=torch.bmm(raster_from_agent, agent_from_world),
        agent_from_world=agent_from_world, world_from_agent=inverse_f32(agent_from_world),
        all_other_agents_indices=batch["neigh_indices"],
        all_other_agents_history_positions=n_pos, all_other_agents_history_yaws=n_yaw,
        all_other_agents_history_speeds=n_speed, all_other_agents_history_availability=n_mask,
        all_other_agents_curr_speed=n_speed[..., -1],
        all_other_agents_future_positions=nf_pos, all_other_agents_future_yaws=nf_yaw,
        all_other_agents_future_availability=nf_mask,
        all_other_agents_types=n_types,
        all_other_agents_extents=n_extents.max(dim=-2)[0],
        all_other_agents_history_extents=n_extents,
    )
    if maps is not None:
        d["map_is_singapore"] = torch.as_tensor([int("singapore" in n.lower()) for n in batch["map_names"]],
                                                dtype=torch.long)
    batch.update(d)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.nan_to_num(0)
    return batch
