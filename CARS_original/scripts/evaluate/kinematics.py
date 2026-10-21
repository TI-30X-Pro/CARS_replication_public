#!/usr/bin/env python3
"""Compute adversary trajectory kinematic metrics for a single rollout HDF5.

Metrics (per-episode, adversary agent only):
  a_max, a_95   — acceleration magnitude max / 95th percentile (m/s²)
  jerk_max, jerk_95 — jerk magnitude max / 95th percentile (m/s³)
  a_lat_max, a_lat_95 — lateral acceleration max / 95th (m/s²)
  yawrate_max   — max yaw rate (rad/s)
  speed_max     — max speed (m/s)
  any_viol      — fraction of timesteps exceeding ANY regulation limit

Regulation-derived limits (UN R157 ECE/TRANS/WP.29/2022/59/Rev.1):
  |a|     ≤ 7      m/s²    (Table 3: b_cut-in,max — adversary's max decel
                            assumed by FSM)
  |jerk|  ≤ 12.65  m/s³    (Table 3: jerking rate)
  |a_lat| ≤ 3.0    m/s²    (§5.6.2.2 + JRC RSS adv max_a_lat)

Usage:
  python -m scripts.evaluate kinematics \\
      --hdf5 <run_dir>/data.hdf5 \\
      --output_dir <run_dir>/replay_results/

For the Neuweiler roundabout, pass `--roundabout_center 79.85 -51.34`.
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def compute_kinematics(pos, yaw, dt=0.1, roundabout_center=None,
                       smooth=True, sg_window=7, sg_poly=3):
    """Compute kinematic metrics for a single agent trajectory.

    `smooth=True` applies a Savitzky-Golay filter (window=7 frames at 10 Hz =
    0.7 s, cubic polynomial) to the raw position trace BEFORE finite differencing.
    Without smoothing, triple finite differencing of cm-precision position data
    at dt=0.1 s amplifies noise by ~1000x (sigma_jerk floor ~20 m/s^3),
    making the resulting jerk metric noise-dominated rather than reflecting
    the true adversary kinematics. Verified on CARS data: raw jerk_max median
    23.6 m/s^3 vs SG-smoothed 8.3 m/s^3 (raw 100% > 12.65, SG 3.3% > 12.65).

    `roundabout_center=(cx, cy)` enables curvature-aware lateral check for the
    Neuweiler RounD dataset: the centripetal floor v^2 / r_local (where
    r_local is the distance from the local position to the roundabout centre)
    is subtracted from |a_lat| before comparing to ALAT_MAX.
    """
    T = pos.shape[0]
    if T < max(4, sg_window):
        return None
    if smooth:
        pos = savgol_filter(pos, window_length=sg_window, polyorder=sg_poly, axis=0)
        yaw_unwrapped = np.unwrap(yaw)
        yaw = savgol_filter(yaw_unwrapped, window_length=sg_window, polyorder=sg_poly)

    # Velocity
    vel = np.diff(pos, axis=0) / dt                    # (T-1, 2)
    speed = np.linalg.norm(vel, axis=1)                # (T-1,)

    # Acceleration
    acc_vec = np.diff(vel, axis=0) / dt                # (T-2, 2)
    acc_mag = np.linalg.norm(acc_vec, axis=1)          # (T-2,)

    # Jerk
    jerk_vec = np.diff(acc_vec, axis=0) / dt           # (T-3, 2)
    jerk_mag = np.linalg.norm(jerk_vec, axis=1)        # (T-3,)

    # Yaw rate and lateral acceleration
    dyaw = np.diff(yaw)                                # (T-1,)
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi       # unwrap
    yaw_rate = dyaw / dt                               # (T-1,)
    a_lat = speed * np.abs(yaw_rate)                   # (T-1,)

    # Curvature-aware lateral residual for fixed-geometry roundabouts:
    #   a_lat_excess = max(a_lat - v^2/r_local, 0)
    if roundabout_center is not None:
        cx, cy = roundabout_center
        r_local = np.sqrt((pos[:-1,0]-cx)**2 + (pos[:-1,1]-cy)**2)  # (T-1,)
        r_local = np.maximum(r_local, 1.0)             # avoid div-by-zero
        a_lat_geom = speed**2 / r_local
        a_lat_check = np.maximum(a_lat - a_lat_geom, 0.0)
    else:
        a_lat_check = a_lat

    # Regulation-derived limits (UN R157 ECE/TRANS/WP.29/2022/59/Rev.1)
    A_MAX, JERK_MAX, ALAT_MAX = 7.0, 12.65, 3.0
    n = min(len(acc_mag), len(a_lat_check), len(jerk_mag))
    if n == 0:
        return None
    any_viol = float(((acc_mag[:n] > A_MAX) |
                       (a_lat_check[:n] > ALAT_MAX) |
                       (jerk_mag[:n] > JERK_MAX)).mean())

    return {
        "a_max":       float(np.max(acc_mag)),
        "a_95":        float(np.percentile(acc_mag, 95)),
        "jerk_max":    float(np.max(jerk_mag)),
        "jerk_95":     float(np.percentile(jerk_mag, 95)),
        "a_lat_max":   float(np.max(a_lat)),
        "a_lat_95":    float(np.percentile(a_lat, 95)),
        "yawrate_max": float(np.max(np.abs(yaw_rate))),
        "speed_max":   float(np.max(speed)),
        "speed_mean":  float(np.mean(speed)),
        "acc_viol":    float((acc_mag > A_MAX).mean()),
        "jerk_viol":   float((jerk_mag > JERK_MAX).mean()),
        "alat_viol":   float((a_lat_check > ALAT_MAX).mean()),
        "any_viol":    any_viol,
    }


def process_hdf5(path, dt=0.1, roundabout_center=None, smooth=True):
    """Extract adversary kinematics from all episodes.

    `smooth` and `roundabout_center` are forwarded to compute_kinematics.
    """
    rows = []
    with h5py.File(path, "r") as f:
        for ep in f.keys():
            g = f[ep]
            centroid = g["centroid"][:]     # (N, T, 2)
            yaw = g["yaw"][:]              # (N, T)
            adv_idx = g["adv_idx"][:]      # (N, T)
            adv_i = int(adv_idx[0, -1])
            if adv_i < 0 or adv_i >= centroid.shape[0]:
                continue
            kin = compute_kinematics(centroid[adv_i], yaw[adv_i], dt,
                                     roundabout_center=roundabout_center,
                                     smooth=smooth)
            if kin is None:
                continue
            kin["episode"] = ep
            rows.append(kin)
    return pd.DataFrame(rows)


def _print_summary(df: pd.DataFrame, label: str) -> None:
    print("\n" + "=" * 95)
    print(f"Adversary kinematic realism — {label} ({len(df)} eps)")
    print("Limits (UN R157 Table 3 + §5.6.2.2): |a|>7 m/s^2, |jerk|>12.65 m/s^3, |a_lat|>3.0 m/s^2")
    print("=" * 95)

    hdr = (f'{"a_max":>8} {"a_95":>8} {"jerk_95":>9} {"a_lat_95":>9} '
           f'{"w_max":>8} {"v_max":>7} {"Viol%":>7}')
    print(hdr)
    print("-" * len(hdr))
    print(f'{df["a_max"].median():>7.2f} '
          f'{df["a_95"].median():>7.2f} '
          f'{df["jerk_95"].median():>8.2f} '
          f'{df["a_lat_95"].median():>8.2f} '
          f'{df["yawrate_max"].median():>7.2f} '
          f'{df["speed_max"].median():>6.1f} '
          f'{df["any_viol"].mean()*100:>6.1f}%')
    print(f'{"m/s2":>8} {"m/s2":>8} {"m/s3":>9} {"m/s2":>9} '
          f'{"rad/s":>8} {"m/s":>7}')

    print("\nDistribution: median -> p95 -> max")
    for col, lab, u in [("a_max",       "|a|_max",    "m/s2"),
                        ("jerk_95",     "jerk_p95",   "m/s3"),
                        ("a_lat_max",   "|a_lat|_max","m/s2"),
                        ("yawrate_max", "w_max",      "rad/s")]:
        print(f'  {lab:<12}: '
              f'{df[col].median():>7.2f} -> '
              f'{df[col].quantile(0.95):>8.2f} -> '
              f'{df[col].max():>9.2f} {u}')
    print(f'  Violations  : '
          f'acc={df["acc_viol"].mean()*100:.1f}%  '
          f'jerk={df["jerk_viol"].mean()*100:.1f}%  '
          f'a_lat={df["alat_viol"].mean()*100:.1f}%  '
          f'total={df["any_viol"].mean()*100:.1f}%')


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hdf5", required=True,
                        help="Rollout HDF5 to process")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write kinematics.csv into")
    parser.add_argument("--label", default=None,
                        help="Display label for the summary header (default: HDF5 basename)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Simulation timestep (default 0.1 s)")
    parser.add_argument("--roundabout_center", nargs=2, type=float, default=None,
                        metavar=("X", "Y"),
                        help="Optional roundabout centre for curvature-aware "
                             "lateral check (e.g. Neuweiler RounD: 79.85 -51.34)")
    parser.add_argument("--no_smooth", action="store_true",
                        help="Disable Savitzky-Golay smoothing before finite differencing "
                             "(default: smoothing on; raw differencing is noise-dominated)")


def run(args) -> None:
    ra_centre = tuple(args.roundabout_center) if args.roundabout_center is not None else None
    label = args.label or os.path.splitext(os.path.basename(args.hdf5))[0]

    print(f"Processing {label}{' (curvature-aware)' if ra_centre else ''} from {args.hdf5} ...")
    df = process_hdf5(args.hdf5, dt=args.dt, roundabout_center=ra_centre,
                      smooth=not args.no_smooth)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "kinematics.csv")
    df.to_csv(csv_path, index=False)
    print(f"  {len(df)} episodes -> {csv_path}")

    if len(df):
        _print_summary(df, label)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Adversary kinematic-realism metrics for a single rollout HDF5",
    )
    add_args(_parser)
    run(_parser.parse_args())
