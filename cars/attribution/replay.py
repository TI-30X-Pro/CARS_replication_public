"""Replay collision episodes with the target constrained to its logged path.

For each episode and reference driver, the adversary's logged trajectory is kept and the
target is re-driven along its own logged path:
    v(t) = max(0, v_logged(t) - dv(t)),  dv(t+1) = dv(t) + a_driver(t) dt,  s(t+1) = s(t) + v(t) dt
where s is arc length along the logged path. The episode is attributed to the system under
test (I_attr = 1) if the re-driven target no longer collides.

Replay rules (paper Eq. 2 and S23):
- The adversary is whichever agent was the adversary at each step (adv_idx), so its
  trajectory can jump when the selector switched; its box size is the last adversary's.
- Adversary velocity is a forward difference of its logged positions (backward at the end),
  projected on the target's heading (u) and, signed toward the target, sideways.
- The gap is bumper to bumper along the target's heading: d_lon - (L_target + L_adv) / 2.
- Collision: the replay box test (both boxes enlarged by 1 m / 0.3 m), checked on the next
  step after moving the target. The first collision ends the replay.
- Tiers from FSM's maxima over the replay: Hard if max CFS >= 0.9, else Medium if
  max PFS > 0.85, else Easy.
- Braking deficit BD = v^2 / (2 * 6) - max(gap, 0); TTC = gap / closing speed when closing
  faster than 0.1 m/s with a positive gap.
"""
import re

import h5py
import numpy as np

from cars.attribution.drivers import CCJP, FSM, RSS, compute_dist_to_react
from cars.sim.geometry import obb_collision

COLUMNS = ["episode", "scene_id", "controller", "target_name", "adv_name", "preventable",
           "collision_step", "min_distance", "max_pfs", "max_cfs", "criticality",
           "speed_at_collision", "collision_angle_rad", "min_ttc", "max_braking_deficit",
           "bd_at_trigger", "trigger_step", "unpreventable_reason", "raw_collision",
           "raw_collision_step", "raw_min_distance", "lane_width_m", "dist_to_react_m", "map_location"]


# Target path

def lon(vec, yaw):
    return vec[0] * np.cos(yaw) + vec[1] * np.sin(yaw)


def lat(vec, yaw):
    return -vec[0] * np.sin(yaw) + vec[1] * np.cos(yaw)


def arc_length(pos):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])


def _interp_angle(s, arc, yaw):
    i = int(np.clip(np.searchsorted(arc, s, side="right") - 1, 0, len(arc) - 2))
    t0, t1 = arc[i], arc[i + 1]
    if t1 == t0:
        return float(yaw[i])
    alpha = (s - t0) / (t1 - t0)
    y0, y1 = float(yaw[i]), float(yaw[i + 1])
    return y0 + alpha * ((y1 - y0 + np.pi) % (2 * np.pi) - np.pi)


def along_path(s, arc, pos, yaw):
    s = float(np.clip(s, arc[0], arc[-1]))
    return np.array([np.interp(s, arc, pos[:, 0]), np.interp(s, arc, pos[:, 1])]), _interp_angle(s, arc, yaw)


# Replay one driver

def replay_episode(target_pos, target_yaw, adv_pos, adv_yaw, target_ext, adv_ext, driver, dt=0.1):
    T = min(len(target_pos), len(adv_pos))
    if T < 2:
        return {"preventable": False, "collision_step": None, "min_distance": np.inf, "max_pfs": 0.0,
                "max_cfs": 0.0, "criticality": "Easy", "speed_at_collision": np.nan,
                "collision_angle": np.nan, "min_ttc": np.inf, "max_braking_deficit": np.nan,
                "unpreventable_reason": None, "timeseries": {}}
    driver.reset()
    arc = arc_length(target_pos)
    v_log = np.diff(arc) / dt
    deficit = deficit_prev = s = 0.0
    hl_t, hl_a = float(target_ext[0]) / 2.0, float(adv_ext[0]) / 2.0
    hw_t, hw_a = float(target_ext[1]) / 2.0, float(adv_ext[1]) / 2.0
    ts = {k: [] for k in ("pfs", "cfs", "a_ctrl", "v_target", "distance", "ttc", "closing_speed",
                          "required_braking_dist", "braking_deficit")}
    collision_step, trigger_step, min_distance = None, None, np.inf

    for t in range(T - 1):
        pos, yaw = along_path(s, arc, target_pos, target_yaw)
        gap_vec = adv_pos[t] - pos
        d_lon, d_lat = lon(gap_vec, yaw), lat(gap_vec, yaw)
        d_gap = d_lon - (hl_t + hl_a)
        adv_vel = (adv_pos[t + 1] - adv_pos[t]) / dt if t < T - 2 else (adv_pos[t] - adv_pos[t - 1]) / dt
        u1 = lon(adv_vel, yaw)
        v_lat_toward = -np.sign(d_lat) * lat(adv_vel, yaw) if d_lat != 0 else 0.0
        v_log_t = float(v_log[t])
        v = max(0.0, v_log_t - deficit)
        state = {"distance": max(d_gap, 0.0), "d_lon_signed": d_gap, "target_speed": v, "adv_speed": u1,
                 "target_accel": -(deficit - deficit_prev) / dt, "lateral_offset": d_lat,
                 "half_width_target": hw_t, "half_width_adv": hw_a, "half_length_target": hl_t,
                 "half_length_adv": hl_a, "adv_v_lat": v_lat_toward}
        a = driver(state)
        if trigger_step is None and driver.triggered:
            trigger_step = t

        closing = v - u1
        req = v ** 2 / (2 * 6.0)
        ts["a_ctrl"].append(a)
        ts["v_target"].append(v)
        ts["distance"].append(d_gap)
        ts["pfs"].append(state.get("_pfs", 0.0))
        ts["cfs"].append(state.get("_cfs", 0.0))
        ts["ttc"].append(d_gap / closing if (closing > 0.1 and d_gap > 0) else np.inf)
        ts["closing_speed"].append(closing)
        ts["required_braking_dist"].append(req)
        ts["braking_deficit"].append(req - max(d_gap, 0.0))

        deficit_prev = deficit
        deficit += a * dt
        s += max(0.0, v_log_t - deficit) * dt
        next_pos, next_yaw = along_path(s, arc, target_pos, target_yaw)
        hit, dist = obb_collision(next_pos, next_yaw, target_ext, adv_pos[t + 1], adv_yaw[t + 1], adv_ext)
        min_distance = min(min_distance, dist)
        if hit:
            collision_step = t + 1
            break

    ts = {k: np.array(v) for k, v in ts.items()}
    max_pfs = float(np.max(ts["pfs"])) if len(ts["pfs"]) else 0.0
    max_cfs = float(np.max(ts["cfs"])) if len(ts["cfs"]) else 0.0
    tier = "Hard" if max_cfs >= 0.9 else "Medium" if max_pfs > 0.85 else "Easy"
    col = collision_step - 1 if (collision_step is not None and collision_step > 0) else None
    speed_col = float(ts["v_target"][col]) if col is not None else np.nan
    if col is not None and col < len(target_yaw) and col < len(adv_yaw):
        angle = float((float(adv_yaw[col]) - float(target_yaw[col]) + np.pi) % (2 * np.pi) - np.pi)
    else:
        angle = np.nan
    finite = ts["ttc"][np.isfinite(ts["ttc"])]
    min_ttc = float(np.min(finite)) if len(finite) else np.inf
    max_bd = float(np.max(ts["braking_deficit"])) if len(ts["braking_deficit"]) else np.nan
    bd_trig = (float(ts["braking_deficit"][trigger_step])
               if trigger_step is not None and trigger_step < len(ts["braking_deficit"]) else np.nan)

    reason = None
    if collision_step is not None:
        r = []
        if max_cfs >= 0.9:
            r.append(f"CFS={max_cfs:.2f}>=0.9 (emergency zone)")
        elif max_pfs >= 0.85:
            r.append(f"PFS={max_pfs:.2f}>=0.85 (comfort zone)")
        if max_bd > 0:
            r.append(f"max_braking_deficit={max_bd:.2f}m>0")
        if np.isfinite(min_ttc) and min_ttc < 1.0:
            r.append(f"min_TTC={min_ttc:.2f}s<1.0s")
        if not np.isnan(speed_col) and speed_col > 5.0:
            r.append(f"v_col={speed_col:.1f}m/s")
        reason = "; ".join(r) if r else "late approach"

    return {"preventable": collision_step is None, "collision_step": collision_step,
            "min_distance": min_distance, "max_pfs": max_pfs, "max_cfs": max_cfs, "criticality": tier,
            "speed_at_collision": speed_col, "collision_angle": angle, "min_ttc": min_ttc,
            "max_braking_deficit": max_bd, "bd_at_trigger": bd_trig,
            "trigger_step": trigger_step if trigger_step is not None else -1,
            "unpreventable_reason": reason, "timeseries": ts}


# Read episode data

def _str(v):
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.ndarray):
        v = v.flat[0]
        return v.decode("utf-8") if isinstance(v, bytes) else str(v)
    return str(v)


def episode_agents(g, target_fallback=None):
    """(target index, adversary index per step, agent names) of one episode group."""
    names_raw = g["agent_name"]
    n = names_raw.shape[0]
    names = [_str(names_raw[i] if names_raw.ndim == 1 else names_raw[i, 0]) for i in range(n)]
    target = _str(g["target_name"][0]) if "target_name" in g else target_fallback
    if target is None:
        return None, None, names
    ti = next((i for i, x in enumerate(names) if x == target), None)
    if ti is None:
        ti = next((i for i, x in enumerate(names) if target in x or x in target), None)
    if ti is None or "adv_idx" not in g:
        return ti, None, names
    adv = g["adv_idx"][:]
    adv = (adv[0] if adv.ndim >= 2 else adv).astype(np.int32)
    last = int(adv[-1])
    if not (0 <= last < n) or last == ti:
        return ti, None, names
    return ti, adv, names


def episode_arrays(g, ti, adv):
    centroid, yaw, extent = g["centroid"][:], g["yaw"][:], g["extent"][:]
    t_pos, t_yaw = centroid[ti], yaw[ti].flatten()
    t_ext = extent[ti][0, :2].astype(np.float64)
    a_pos = np.zeros((len(adv), 2))
    a_yaw = np.zeros(len(adv))
    for t, i in enumerate(adv):
        a_pos[t] = centroid[i, t]
        a_yaw[t] = float(np.asarray(yaw[i, t]).flat[0])
    a_ext = extent[int(adv[-1]), 0, :2].astype(np.float64)
    return t_pos, t_yaw, t_ext, a_pos, a_yaw, a_ext


# CC-JP lane width

def lane_width(nusc_map, x, y, fallback=3.2):
    """Width of the lane closest to (x, y): twice the distance from its polygon's centroid
    to its boundary, if within 3 m and between 2 and 6 m wide."""
    token = nusc_map.get_closest_lane(x, y, radius=3.0)
    if not token:
        return fallback
    try:
        poly = nusc_map.extract_polygon(nusc_map.get("lane", token)["polygon_token"])
        if poly is None or poly.is_empty:
            return fallback
        w = 2.0 * float(poly.boundary.distance(poly.centroid))
        return fallback if (w < 2.0 or w > 6.0) else w
    except Exception:          # lane connectors are not in the "lane" table
        return fallback


def dist_to_react_for(g, location, nusc_maps, fallback=1.20):
    """(dist_to_react, lane width or None, adversary width or None, location or None)."""
    try:
        ti, adv, _ = episode_agents(g)
        if ti is None or adv is None:
            return fallback, None, None, None
        p0 = np.asarray(g["centroid"][ti, 0], dtype=np.float64)
        if not np.all(np.isfinite(p0)):
            return fallback, None, None, None
        adv_w = float(g["extent"][int(adv[-1]), 0, 1])
        if adv_w < 0.5 or adv_w > 4.0:
            adv_w = 1.85
    except Exception:
        return fallback, None, None, None
    if location is None or location not in nusc_maps:
        return fallback, None, adv_w, None
    w = lane_width(nusc_maps[location], float(p0[0]), float(p0[1]))
    return compute_dist_to_react(w, adv_w), w, adv_w, location


# Replay one HDF5 file

def replay_file(hdf5_path, scene_location, nusc_maps, dt=0.1, dtr_fallback=1.20, trust_collision=True):
    """Return one attribution row per episode and safety driver."""
    drivers = [FSM(dt=dt), CCJP(dt=dt), RSS(dt=dt)]
    rows = []
    with h5py.File(hdf5_path, "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            m = re.search(r"_tgt([^_]+)_adv", key)
            try:
                ti, adv, names = episode_agents(g, m.group(1) if m else None)
            except Exception:
                continue
            if ti is None or adv is None:
                continue
            try:
                t_pos, t_yaw, t_ext, a_pos, a_yaw, a_ext = episode_arrays(g, ti, adv)
            except Exception:
                continue
            scene = key.split("_")[0]
            dtr, w, _, loc = dist_to_react_for(g, scene_location.get(scene), nusc_maps, dtr_fallback)
            drivers[1].dist_to_react = float(dtr)
            if trust_collision:
                raw = (True, None, float("nan"))
            else:
                raw = _raw_collision(t_pos, t_yaw, a_pos, a_yaw, t_ext, a_ext)
            last = int(adv[-1])
            for drv in drivers:
                r = replay_episode(t_pos, t_yaw, a_pos, a_yaw, t_ext, a_ext, drv, dt=dt)
                rows.append({
                    "episode": key, "scene_id": scene, "controller": drv.name,
                    "target_name": names[ti] if ti < len(names) else "",
                    "adv_name": names[last] if last < len(names) else "",
                    "preventable": r["preventable"], "collision_step": r["collision_step"],
                    "min_distance": r["min_distance"], "max_pfs": r["max_pfs"], "max_cfs": r["max_cfs"],
                    "criticality": r["criticality"], "speed_at_collision": r["speed_at_collision"],
                    "collision_angle_rad": r["collision_angle"], "min_ttc": r["min_ttc"],
                    "max_braking_deficit": r["max_braking_deficit"], "bd_at_trigger": r["bd_at_trigger"],
                    "trigger_step": r["trigger_step"], "unpreventable_reason": r["unpreventable_reason"],
                    "raw_collision": raw[0], "raw_collision_step": raw[1], "raw_min_distance": raw[2],
                    "lane_width_m": w if w is not None else -1, "dist_to_react_m": dtr,
                    "map_location": loc or "fallback",
                })
    return rows


def _raw_collision(t_pos, t_yaw, a_pos, a_yaw, t_ext, a_ext):
    """Does the logged trajectory itself collide (no driver)? For files that keep every episode."""
    min_d, step = np.inf, None
    for t in range(min(len(t_pos), len(a_pos))):
        hit, d = obb_collision(t_pos[t], t_yaw[t], t_ext, a_pos[t], a_yaw[t], a_ext)
        min_d = min(min_d, d)
        if hit:
            step = t
            break
    return step is not None, step, min_d
