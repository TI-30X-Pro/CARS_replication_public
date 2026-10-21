"""Train decision tree for RounD tgt-adv selection.

Reads labels from labels/round_labels_recXX.json, extracts features directly from
RounD CSV tracks, trains a classifier, and saves to a separate model file
(does NOT overwrite decision_tree_model.pkl used by nuScenes PPO).

Usage:
    python scripts/decision_tree_round.py --rec 07
    python scripts/decision_tree_round.py --rec 07 --output round_decision_tree.pkl
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ── Data loading ────────────────────────────────────────────────────

def load_recording(rec_id, data_dir):
    df = pd.read_csv(os.path.join(data_dir, f"{rec_id}_tracks.csv"))
    meta = pd.read_csv(os.path.join(data_dir, f"{rec_id}_tracksMeta.csv"))
    rmeta = pd.read_csv(os.path.join(data_dir, f"{rec_id}_recordingMeta.csv"))

    veh_classes = {'car', 'truck', 'van', 'trailer', 'bus'}
    veh_ids = set(meta[meta['class'].isin(veh_classes)]['trackId'].values)
    df_veh = df[df['trackId'].isin(veh_ids)].copy()

    fps = int(rmeta['frameRate'].iloc[0])
    return df_veh, meta, fps


# ── Feature extraction ──────────────────────────────────────────────

FEATURE_NAMES = [
    'rel_dist',         # Euclidean distance
    's',                # longitudinal offset (along tgt heading)
    'l',                # lateral offset
    'abs_l',            # |lateral offset|
    'closing',          # longitudinal closing speed
    'lateral_closing',  # lateral closing speed
    'rel_speed',        # speed difference (adv - tgt)
    'adv_speed',        # adv absolute speed
    'cos_yaw_diff',     # heading alignment
    'ttc',              # time-to-collision
    'cos_bearing',      # adv facing tgt?
    'dist_rate_5',      # distance change rate (last 1s)
    'abs_l_delta_5',    # lateral cut-in trend (last 1s)
]


def extract_pair_features(df, frame, tgt_id, adv_id, fps, hist_sec=1.0):
    """Extract features for ONE (tgt, adv) pair at a given frame.

    Returns feature vector (13,) or None if data insufficient.
    """
    dt = 1.0 / fps
    hist_frames = int(hist_sec * fps)

    # Get tgt and adv rows at current frame
    tgt_row = df[(df['frame'] == frame) & (df['trackId'] == tgt_id)]
    adv_row = df[(df['frame'] == frame) & (df['trackId'] == adv_id)]
    if tgt_row.empty or adv_row.empty:
        return None

    tgt = tgt_row.iloc[0]
    adv = adv_row.iloc[0]

    # Positions
    tgt_pos = np.array([tgt['xCenter'], tgt['yCenter']])
    adv_pos = np.array([adv['xCenter'], adv['yCenter']])

    # Velocities
    tgt_vel = np.array([tgt['xVelocity'], tgt['yVelocity']])
    adv_vel = np.array([adv['xVelocity'], adv['yVelocity']])
    tgt_speed = np.linalg.norm(tgt_vel)
    adv_speed = np.linalg.norm(adv_vel)

    # Headings (RounD heading is in degrees)
    tgt_yaw = np.radians(tgt['heading'])
    adv_yaw = np.radians(adv['heading'])

    # Relative position
    rel_pos = adv_pos - tgt_pos
    rel_dist = np.linalg.norm(rel_pos)

    # Tgt heading unit vector
    u = tgt_vel / (tgt_speed + 1e-6)
    u_perp = np.array([-u[1], u[0]])

    # Longitudinal / lateral offsets
    s = np.dot(rel_pos, u)
    l = np.dot(rel_pos, u_perp)
    abs_l = abs(l)

    # Closing speeds
    dv = tgt_vel - adv_vel
    closing = np.dot(dv, u)
    lateral_closing = np.dot(dv, u_perp)

    # Relative speed and heading
    rel_speed = adv_speed - tgt_speed
    yaw_diff = adv_yaw - tgt_yaw
    cos_yaw_diff = np.cos(yaw_diff)

    # TTC
    bumper_margin = 1.0
    gap = s - 0.5 * (tgt['length'] + adv['length']) - bumper_margin
    ttc = gap / (closing + 1e-6) if closing > 0.1 else 999.0
    ttc = np.clip(ttc, 0, 999.0)

    # Bearing: is adv facing tgt?
    bearing = np.arctan2(tgt_pos[1] - adv_pos[1], tgt_pos[0] - adv_pos[0])
    cos_bearing = np.cos(bearing - adv_yaw)

    # History-based features (last hist_sec)
    hist_start = frame - hist_frames
    tgt_hist = df[(df['trackId'] == tgt_id) & (df['frame'] >= hist_start) & (df['frame'] <= frame)]
    adv_hist = df[(df['trackId'] == adv_id) & (df['frame'] >= hist_start) & (df['frame'] <= frame)]

    if len(tgt_hist) < 3 or len(adv_hist) < 3:
        dist_rate_5 = 0.0
        abs_l_delta_5 = 0.0
    else:
        tgt_hist = tgt_hist.sort_values('frame')
        adv_hist = adv_hist.sort_values('frame')
        # Merge on frame
        merged = pd.merge(tgt_hist[['frame', 'xCenter', 'yCenter']],
                         adv_hist[['frame', 'xCenter', 'yCenter']],
                         on='frame', suffixes=('_t', '_a'))
        if len(merged) < 2:
            dist_rate_5 = 0.0
            abs_l_delta_5 = 0.0
        else:
            dists = np.sqrt((merged['xCenter_a'] - merged['xCenter_t'])**2 +
                           (merged['yCenter_a'] - merged['yCenter_t'])**2).values
            dt_hist = len(merged) * dt
            dist_rate_5 = (dists[-1] - dists[0]) / (dt_hist + 1e-6)

            # Lateral change
            rel_hist = np.stack([merged['xCenter_a'].values - merged['xCenter_t'].values,
                                merged['yCenter_a'].values - merged['yCenter_t'].values], axis=1)
            l_hist = np.abs(rel_hist @ u_perp)
            abs_l_delta_5 = l_hist[-1] - l_hist[0]  # negative = cutting in

    return np.array([
        rel_dist, s, l, abs_l, closing, lateral_closing,
        rel_speed, adv_speed, cos_yaw_diff, ttc,
        cos_bearing, dist_rate_5, abs_l_delta_5
    ], dtype=np.float32)


# ── Dataset building ────────────────────────────────────────────────

def build_dataset(df, labels, fps, slice_sec=5.0, hist_sec=1.0):
    """Build training dataset from labels.

    For each labeled slice:
      - positive samples: labeled (tgt, adv) pairs
      - negative samples: all other (tgt, agent) pairs in the same frame
    """
    X_list, y_list = [], []
    hist_frames = int(hist_sec * fps)
    slice_frames = int(slice_sec * fps)

    for scene_key, pairs in labels.items():
        if not isinstance(pairs, dict) or not pairs:
            continue

        # Parse scene_key: "round_07_50" -> start_frame=50
        parts = scene_key.split('_')
        start_frame = int(parts[-1])
        mid_frame = start_frame + slice_frames // 2  # use middle of slice

        # All agents present at mid_frame
        frame_df = df[df['frame'] == mid_frame]
        all_agents = set(frame_df['trackId'].values)

        for tgt_str, adv_str in pairs:
            tgt_id = int(tgt_str)
            adv_id = int(adv_str)

            if tgt_id not in all_agents or adv_id not in all_agents:
                continue

            # Positive: the labeled adv
            feat = extract_pair_features(df, mid_frame, tgt_id, adv_id, fps, hist_sec)
            if feat is not None:
                X_list.append(feat)
                y_list.append(1)

            # Negatives: all other agents paired with this tgt
            for other_id in all_agents:
                if other_id == tgt_id or other_id == adv_id:
                    continue
                feat = extract_pair_features(df, mid_frame, tgt_id, int(other_id), fps, hist_sec)
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Dataset: {len(X)} samples, {(y==1).sum()} positive, {(y==0).sum()} negative")
    return X, y


# ── Training ────────────────────────────────────────────────────────

def train(X, y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    w = np.ones_like(y, dtype=float)
    w[y == 1] = np.sqrt(neg / pos)

    model = HistGradientBoostingClassifier(
        max_depth=5, max_iter=800, learning_rate=0.03,
        min_samples_leaf=10, random_state=42
    )
    model.fit(X, y, sample_weight=w)
    return model


def evaluate(model, X, y, split_name="Test"):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n{'='*60}")
    print(f"{split_name} accuracy: {acc:.4f}")
    print(classification_report(y, y_pred, target_names=['Safe', 'Dangerous']))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    if hasattr(model, 'feature_importances_'):
        print("\nFeature importances:")
        indices = np.argsort(model.feature_importances_)[::-1]
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {FEATURE_NAMES[idx]}: {model.feature_importances_[idx]:.4f}")

    return acc


# ── Main ────────────────────────────────────────────────────────────

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--rec', type=str, default='07', help='Recording ID')
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/rounD/rounD'))
    parser.add_argument('--labels', type=str, default=None,
                        help='Labels JSON (default: labels/round_labels_rec{rec}.json)')
    parser.add_argument('--output', type=str, default='checkpoints/round_decision_tree.pkl',
                        help='Output model file (default: checkpoints/round_decision_tree.pkl)')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--slice_sec', type=float, default=5.0)


def run(args) -> None:
    rec_id = args.rec.zfill(2)
    labels_file = args.labels or f'labels/round_labels_rec{rec_id}.json'

    print(f"Loading rec_{rec_id}...")
    df, meta, fps = load_recording(rec_id, args.data_dir)
    print(f"  {len(df)} rows, {df['trackId'].nunique()} vehicles, {fps} fps")

    print(f"Loading labels from {labels_file}...")
    with open(labels_file) as f:
        labels = json.load(f)
    n_slices = len(labels)
    n_pairs = sum(len(v) for v in labels.values() if isinstance(v, dict))
    print(f"  {n_slices} slices, {n_pairs} pairs")

    print("\nBuilding dataset...")
    X, y = build_dataset(df, labels, fps, slice_sec=args.slice_sec)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} ({(y_train==1).sum()} pos)  "
          f"Test: {len(X_test)} ({(y_test==1).sum()} pos)")

    print("\nTraining...")
    model = train(X_train, y_train)

    evaluate(model, X_train, y_train, "Train")
    evaluate(model, X_test, y_test, "Test")

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': FEATURE_NAMES,
            'rec_id': rec_id,
            'n_train': len(X_train),
            'n_test': len(X_test),
        }, f)
    print(f"\nModel saved to {args.output}")
    print(f"  (decision_tree_model.pkl is NOT modified)")


# ── Runtime obs-based feature function ─────────────────────────────
#
# Mirrors extract_pair_features above but consumes an `agents_obs` dict in
# the shape used by TargetAdvOthersWrapper (centroid / curr_speed / yaw /
# history_positions / world_from_agent / extent) so that rollout_round.py
# feeds the RounD decision tree the SAME 13 features that were used during
# training. Each feature below must stay numerically equivalent to its
# counterpart in extract_pair_features.

def extract_features_round(agents_obs, target_idx):
    """Obs-based RounD feature extractor. Returns (features, feature_names).

    Shape of `features` is (N, 13). Feature order matches FEATURE_NAMES at
    the top of this file (`rel_dist, s, l, abs_l, closing, lateral_closing,
    rel_speed, adv_speed, cos_yaw_diff, ttc, cos_bearing, dist_rate_5,
    abs_l_delta_5`).
    """
    from tbsim.utils.geometry_utils import transform_points_np

    centroid = np.asarray(agents_obs["centroid"])             # (N, 2) world
    curr_speed = np.asarray(agents_obs["curr_speed"])          # (N,)
    yaw = np.asarray(agents_obs["yaw"])                        # (N,)
    history_positions = np.asarray(agents_obs["history_positions"])  # (N, T, 2) local
    world_from_agent = np.asarray(agents_obs["world_from_agent"])    # (N, 3, 3)
    extent = np.asarray(agents_obs["extent"])                  # (N, 3) [L, W, H]

    N = centroid.shape[0]
    history_world = transform_points_np(history_positions, world_from_agent)  # (N, T, 2)

    dt = 0.1
    T_hist = history_world.shape[1]

    tgt_pos = centroid[target_idx]           # (2,)
    tgt_yaw = float(yaw[target_idx])

    # Velocity vectors. Training uses (xVelocity, yVelocity) directly from
    # the RounD CSV (world-frame). Reconstruct the same vector from speed +
    # yaw when available; fall back to world history diff if speed is 0.
    def _vel(i):
        sp = float(curr_speed[i])
        if sp > 0.05:
            return np.array([sp * np.cos(float(yaw[i])),
                             sp * np.sin(float(yaw[i]))], dtype=np.float64)
        if T_hist >= 2:
            return (history_world[i, -1] - history_world[i, -2]) / dt
        return np.zeros(2, dtype=np.float64)

    tgt_vel = _vel(target_idx)
    tgt_speed_v = float(np.linalg.norm(tgt_vel))
    if tgt_speed_v > 1e-6:
        u = tgt_vel / tgt_speed_v
    else:
        u = np.array([np.cos(tgt_yaw), np.sin(tgt_yaw)], dtype=np.float64)
    u_perp = np.array([-u[1], u[0]], dtype=np.float64)

    features = np.zeros((N, 13), dtype=np.float32)

    for i in range(N):
        adv_pos = centroid[i]
        adv_yaw = float(yaw[i])
        adv_vel = _vel(i)
        adv_speed = float(np.linalg.norm(adv_vel))
        tgt_speed_i = float(np.linalg.norm(tgt_vel))  # scalar shared

        rel_pos = adv_pos - tgt_pos
        rel_dist = float(np.linalg.norm(rel_pos))

        s = float(np.dot(rel_pos, u))
        l = float(np.dot(rel_pos, u_perp))
        abs_l = abs(l)

        dv = tgt_vel - adv_vel
        closing = float(np.dot(dv, u))
        lateral_closing = float(np.dot(dv, u_perp))

        rel_speed = adv_speed - tgt_speed_i

        raw = adv_yaw - tgt_yaw
        yaw_diff = np.arctan2(np.sin(raw), np.cos(raw))
        cos_yaw_diff = float(np.cos(yaw_diff))

        tgt_len = float(extent[target_idx, 0])
        adv_len = float(extent[i, 0])
        bumper_margin = 1.0
        gap = s - 0.5 * (tgt_len + adv_len) - bumper_margin
        if closing > 0.1:
            ttc = float(np.clip(gap / (closing + 1e-6), 0.0, 999.0))
        else:
            ttc = 999.0

        bearing = float(np.arctan2(tgt_pos[1] - adv_pos[1],
                                    tgt_pos[0] - adv_pos[0]))
        cos_bearing = float(np.cos(bearing - adv_yaw))

        # History-based features: match the training-time 1 s window.
        # Training uses `dt_hist = len(merged) * dt` where `merged` is a
        # ~26-frame window. Runtime obs history is usually shorter
        # (CLDAlgoConfig history_num_frames+1 at 10 Hz ≈ 11 frames ≈ 1.0 s),
        # so we use the full history window the simulator gives us.
        if T_hist < 2:
            dist_rate_5 = 0.0
            abs_l_delta_5 = 0.0
        else:
            rel_hist = history_world[i] - history_world[target_idx]   # (T, 2)
            dists_hist = np.linalg.norm(rel_hist, axis=-1)
            dt_hist = (T_hist - 1) * dt
            dist_rate_5 = float((dists_hist[-1] - dists_hist[0]) /
                                 (dt_hist + 1e-6))
            l_hist = np.abs(rel_hist @ u_perp)
            abs_l_delta_5 = float(l_hist[-1] - l_hist[0])

        features[i] = (
            rel_dist, s, l, abs_l,
            closing, lateral_closing,
            rel_speed, adv_speed,
            cos_yaw_diff, ttc,
            cos_bearing, dist_rate_5, abs_l_delta_5,
        )

    return features, list(FEATURE_NAMES)


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description="Train the RounD adversary-selection decision tree")
    add_args(_parser)
    run(_parser.parse_args())
