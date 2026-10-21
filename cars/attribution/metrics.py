"""Compute responsibility and kinematic metrics from replay results.

- Attribution rate of a driver = share of collision episodes it avoids (FSM %, CC-JP %, RSS %).
- Tiers on FSM: Hard if max CFS >= 0.9, Medium if max PFS > 0.85, else Easy.
- H_crit = -sum p ln p / ln 3 over the tiers of the FSM-avoidable episodes; its 95 % lower
  bound is the 5th percentile of 10,000 bootstrap resamples, median over seeds 1..10.
- BD+ = share of episodes whose braking deficit v^2 / 12 - gap was positive at some step
  (FSM replay).
- IP % = mean over episodes of the share of steps where the final adversary's trajectory
  (Savitzky-Golay smoothed, window 7, cubic) exceeds |a| 7 m/s^2, |jerk| 12.65 m/s^3 or
  |a_lat| 3 m/s^2. Acceleration and jerk are vector magnitudes.
"""
import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

TIERS = ("Easy", "Medium", "Hard")
DRIVERS = ("FSM", "CC-JP-LaneAware", "RSS")


def h_crit(tiers):
    if len(tiers) == 0:
        return 0.0
    counts = pd.Series(tiers).value_counts()
    p = np.array([counts.get(t, 0) for t in TIERS], dtype=float)
    if p.sum() == 0:
        return 0.0
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(3)) if len(p) else 0.0


def h_crit_lcb(tiers, B=10000, n_seeds=10, alpha=0.05):
    if len(tiers) == 0:
        return float("nan")
    tiers = np.asarray(tiers)
    lcbs = []
    for seed in range(1, n_seeds + 1):
        rng = np.random.default_rng(seed)
        hs = np.empty(B)
        for b in range(B):
            hs[b] = h_crit(tiers[rng.integers(0, len(tiers), size=len(tiers))])
        lcbs.append(float(np.percentile(hs, alpha * 100)))
    return float(np.median(lcbs))


def summary(df, n_collisions, total_attempts, bootstrap_b=10000, bootstrap_seeds=10):
    """The original's per-run metrics (columns of paper_metrics.csv, minus per_tier)."""
    fsm = df[df["controller"] == "FSM"]
    out = {"N_total_attempts": total_attempts, "N_collisions": n_collisions,
           "CR": n_collisions / total_attempts if total_attempts > 0 else float("nan"),
           "N_replay_eps": len(fsm)}
    out["Hard%"] = int((fsm["criticality"] == "Hard").sum()) / len(fsm)
    for t, k in (("Hard", "Hard_n"), ("Medium", "Med_n"), ("Easy", "Easy_n")):
        out[k] = int((fsm["criticality"] == t).sum())
    prev = fsm[fsm["preventable"] == True]   # noqa: E712
    out["N_FSM_prev"] = len(prev)
    for t, k in (("Hard", "Hard_n_prev"), ("Medium", "Med_n_prev"), ("Easy", "Easy_n_prev")):
        out[k] = int((prev["criticality"] == t).sum())
    out["H_crit_prev"] = h_crit(prev["criticality"].values)
    out["H_crit_LCB"] = h_crit_lcb(prev["criticality"].values, bootstrap_b, bootstrap_seeds)
    ttc = fsm["min_ttc"].replace([np.inf, -np.inf], np.nan).dropna()
    out["TTC_mean"], out["TTC_median"] = ttc.mean(), ttc.median()
    for name, key in (("FSM", "FSM"), ("CC-JP-LaneAware", "CCJP"), ("RSS", "RSS")):
        sub = df[df["controller"] == name]
        n_un = int((sub["preventable"] == False).sum())   # noqa: E712
        out[f"{key}_unprev%"] = n_un / len(sub) if len(sub) else float("nan")
        out[f"{key}_unprev_n"] = n_un
        if key != "FSM":
            out[f"{key}_n"] = len(sub)
    angles = np.degrees(fsm["collision_angle_rad"].dropna().abs())
    out["Angle_std"] = float(angles.std()) if len(angles) else float("nan")
    out["Speed_col"] = float(fsm["speed_at_collision"].dropna().mean())
    bd = fsm["max_braking_deficit"].dropna()
    out["BD_pos%"] = float((bd > 0).mean()) if len(bd) else float("nan")
    return out


def per_tier(df):
    fsm = df[df["controller"] == "FSM"]
    out = {}
    for tier in ("Easy", "Medium", "Hard", "Overall"):
        sub = fsm if tier == "Overall" else fsm[fsm["criticality"] == tier]
        if len(sub) == 0:
            out[tier] = {"n": 0, "TTC_med": float("nan"), "v_col": float("nan"),
                         "BD_pos%": float("nan"), "d_min": float("nan")}
            continue
        ttc = sub["min_ttc"].replace([np.inf, -np.inf], np.nan).dropna()
        bd = sub["max_braking_deficit"].dropna()
        out[tier] = {"n": int(len(sub)),
                     "TTC_med": float(ttc.median()) if len(ttc) else float("nan"),
                     "v_col": float(sub["speed_at_collision"].dropna().mean()),
                     "BD_pos%": float((bd > 0).mean()) if len(bd) else float("nan"),
                     "d_min": float(sub["min_distance"].dropna().mean())}
    return out


def triggers(df):
    """Per driver: how many of its unavoidable episodes show BD+, CFS >= 0.9, TTC < 1 s."""
    def counts(sub, cfs):
        return {"BD+": int((sub["max_braking_deficit"] > 0).fillna(False).sum()),
                "CFS": int((sub["max_cfs"] >= 0.9).fillna(False).sum()) if cfs else None,
                "TTC": int((sub["min_ttc"] < 1.0).fillna(False).sum()), "n": int(len(sub))}
    unprev = {c: df[(df.controller == c) & (~df.preventable.astype(bool))] for c in DRIVERS}
    rows = {c: counts(unprev[c], c == "FSM") for c in DRIVERS}
    keys = set().union(*(set(u.episode) for u in unprev.values()))
    union = df[df.episode.isin(keys) & (~df.preventable.astype(bool))]
    fsm_cfs = df[df.controller == "FSM"].drop_duplicates("episode").set_index("episode")["max_cfs"]
    bd = cfs = ttc = 0
    for ep, g in union.groupby("episode"):
        bd += bool((g["max_braking_deficit"] > 0).fillna(False).any())
        cfs += float(fsm_cfs.get(ep, 0.0)) >= 0.9
        ttc += bool((g["min_ttc"] < 1.0).fillna(False).any())
    rows["Any"] = {"BD+": bd, "CFS": cfs, "TTC": ttc, "n": len(keys)}
    return rows


# Adversary kinematics and IP%

A_MAX, JERK_MAX, ALAT_MAX = 7.0, 12.65, 3.0


def savgol(x, window=7, poly=3):
    """Savitzky-Golay smoothing along axis 0, as SciPy <= 1.13 does it (mode "interp").

    The ends (window // 2 samples each) come from a polynomial fitted to the first / last
    window with np.polyfit, in float64. Newer SciPy fits them in the input's precision,
    which for float32 positions moves the ends by up to 6e-4 m and the jerk by ~0.6 m/s^3.
    """
    y = savgol_filter(x, window_length=window, polyorder=poly, axis=0)
    n, half = x.shape[0], window // 2
    for w0, i0, i1 in ((0, 0, half), (n - window, n - half, n)):
        edge = x[w0:w0 + window].reshape(window, -1)
        coeffs = np.polyfit(np.arange(0, window), edge, poly)
        vals = np.polyval(coeffs, np.arange(i0 - w0, i1 - w0).reshape(-1, 1))
        y[i0:i1] = vals.reshape(i1 - i0, *x.shape[1:])
    return y


def kinematics(pos, yaw, dt=0.1, smooth=True, window=7, poly=3, roundabout_center=None):
    if pos.shape[0] < max(4, window):
        return None
    if smooth:
        pos = savgol(pos, window, poly)
        yaw = savgol(np.unwrap(yaw), window, poly)
    vel = np.diff(pos, axis=0) / dt
    speed = np.linalg.norm(vel, axis=1)
    acc = np.diff(vel, axis=0) / dt
    acc_mag = np.linalg.norm(acc, axis=1)
    jerk_mag = np.linalg.norm(np.diff(acc, axis=0) / dt, axis=1)
    dyaw = (np.diff(yaw) + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = dyaw / dt
    a_lat = speed * np.abs(yaw_rate)
    if roundabout_center is not None:
        cx, cy = roundabout_center
        r = np.maximum(np.sqrt((pos[:-1, 0] - cx) ** 2 + (pos[:-1, 1] - cy) ** 2), 1.0)
        a_lat_check = np.maximum(a_lat - speed ** 2 / r, 0.0)
    else:
        a_lat_check = a_lat
    n = min(len(acc_mag), len(a_lat_check), len(jerk_mag))
    if n == 0:
        return None
    def p95(v):   # NumPy < 2 interpolated percentiles of float32 data in float64
        return float(np.percentile(np.asarray(v, dtype=np.float64), 95))

    return {"a_max": float(np.max(acc_mag)), "a_95": p95(acc_mag),
            "jerk_max": float(np.max(jerk_mag)), "jerk_95": p95(jerk_mag),
            "a_lat_max": float(np.max(a_lat)), "a_lat_95": p95(a_lat),
            "yawrate_max": float(np.max(np.abs(yaw_rate))), "speed_max": float(np.max(speed)),
            "speed_mean": float(np.mean(speed)),
            "acc_viol": float((acc_mag > A_MAX).mean()), "jerk_viol": float((jerk_mag > JERK_MAX).mean()),
            "alat_viol": float((a_lat_check > ALAT_MAX).mean()),
            "any_viol": float(((acc_mag[:n] > A_MAX) | (a_lat_check[:n] > ALAT_MAX) | (jerk_mag[:n] > JERK_MAX)).mean())}


def kinematics_file(hdf5_path, dt=0.1, smooth=True, roundabout_center=None):
    """One row per episode: the kinematics of the final adversary's whole logged trajectory."""
    rows = []
    with h5py.File(hdf5_path, "r") as f:
        for ep in f.keys():
            g = f[ep]
            centroid, yaw, adv = g["centroid"][:], g["yaw"][:], g["adv_idx"][:]
            i = int(adv[0, -1])
            if i < 0 or i >= centroid.shape[0]:
                continue
            k = kinematics(centroid[i], yaw[i], dt, smooth=smooth, roundabout_center=roundabout_center)
            if k is not None:
                k["episode"] = ep
                rows.append(k)
    return pd.DataFrame(rows)
