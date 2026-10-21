"""Train decision tree for AD4CHE tgt-adv selection on highway data.

Reads labels from labels/ad4che_labels_all.json (multi-recording format), loads
the matching recording CSVs lazily, extracts TGT-centric features for
positive (labeled adv) and negative (other 8-neighbors) candidate pairs,
and trains a HistGradientBoostingClassifier. Saves to
ad4che_decision_tree.pkl — does NOT overwrite decision_tree_model.pkl or
round_decision_tree.pkl.

Features are computed at the slice's FIRST frame (matches the labeling
tool and the DT inference point at rollout time).

Usage:
    python scripts/decision_tree_ad4che.py
    python -m scripts.decision_tree ad4che --labels labels/ad4che_labels_all.json
"""
import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


NEIGHBOR_COLS = [
    "precedingId", "followingId",
    "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
    "rightPrecedingId", "rightAlongsideId", "rightFollowingId",
]

RAW_FPS = 30


FEATURE_NAMES_FULL = [
    "rel_dist",        # Euclidean distance (m)
    "s",               # longitudinal offset in TGT frame (+ = ahead)
    "l",               # lateral offset in TGT frame (+ = left)
    "abs_l",           # |l|
    "closing",         # closing speed along TGT heading (+ = approaching)
    "lateral_closing", # closing speed perpendicular
    "rel_speed",       # |v_adv| - |v_tgt|
    "adv_speed",       # |v_adv|
    "cos_yaw_diff",    # cos of heading difference (+ = same direction)
    "ttc",             # time-to-collision (capped at 999)
    "cos_bearing",     # does adv face tgt? (+ = yes)
    "same_lane",       # 1 if same laneId, else 0
    "lane_diff",       # |laneId(adv) - laneId(tgt)|
]

# --- Runtime-compatible 13 features (matches scripts/decision_tree.py used by
#     TargetAdvOthersWrapper for nuScenes). Pure kinematic, no laneId.
#     Trained on AD4CHE so distribution matches; consumable at rollout time
#     inside the simulator without CSV lookups.
FEATURE_NAMES_RUNTIME = [
    "rel_dist", "s", "l", "gap",
    "closing", "lateral_closing", "rel_speed", "curr_speed",
    "cos_yaw_diff", "ttc", "cos_bearing",
    "dist_rate_5", "abs_l_delta_5",
]

# Module-level toggle; set from CLI before build_dataset runs.
FEATURE_NAMES = FEATURE_NAMES_FULL
_FEATURE_MODE = "full"   # 'full' | 'runtime'


def load_recording_csv(rec_dir: str):
    """Return tracks_pixel DataFrame for one DJI recording.

    Strips whitespace from column names — 55/68 recordings ship with
    a trailing space in `yaw_rate`.
    """
    import glob as g
    matches = g.glob(os.path.join(rec_dir, "*_tracks_pixel.csv"))
    if not matches:
        raise FileNotFoundError(f"no *_tracks_pixel.csv in {rec_dir}")
    df = pd.read_csv(matches[0])
    df.columns = [c.strip() for c in df.columns]
    return df


def find_recording_dir(root: str, loc: str, dji: str) -> str:
    return os.path.join(root, loc, dji)


def _row_at(df: pd.DataFrame, frame: int, agent_id: int):
    r = df[(df["frame"] == frame) & (df["id"] == agent_id)]
    return r.iloc[0] if not r.empty else None


def extract_pair_features(df: pd.DataFrame, frame: int, tgt_id: int, adv_id: int):
    """Extract TGT-centric features for one (tgt, adv) pair at `frame`.

    Returns a feature vector (len(FEATURE_NAMES),) or None if data missing.
    Two modes (set via module-level `_FEATURE_MODE`):
      - 'full'    : 13 features incl. same_lane / lane_diff (original AD4CHE DT)
      - 'runtime' : 13 features matching nuScenes scripts/decision_tree.py,
                    pure kinematic so rollout's TargetAdvOthersWrapper can
                    compute them from obs without CSV lookups.
    """
    tgt_row = df[(df["frame"] == frame) & (df["id"] == tgt_id)]
    adv_row = df[(df["frame"] == frame) & (df["id"] == adv_id)]
    if tgt_row.empty or adv_row.empty:
        return None

    tgt = tgt_row.iloc[0]
    adv = adv_row.iloc[0]

    tgt_pos = np.array([tgt["x"], tgt["y"]], dtype=np.float64)
    adv_pos = np.array([adv["x"], adv["y"]], dtype=np.float64)
    tgt_vel = np.array([tgt["xVelocity"], tgt["yVelocity"]], dtype=np.float64)
    adv_vel = np.array([adv["xVelocity"], adv["yVelocity"]], dtype=np.float64)
    tgt_speed = float(np.linalg.norm(tgt_vel))
    adv_speed = float(np.linalg.norm(adv_vel))

    # Yaw from velocity (highway: direction-invariant if we use
    # heading-aligned frame for the TGT)
    if tgt_speed > 0.2:
        tgt_yaw = float(np.arctan2(tgt_vel[1], tgt_vel[0]))
    else:
        tgt_yaw = 0.0
    if adv_speed > 0.2:
        adv_yaw = float(np.arctan2(adv_vel[1], adv_vel[0]))
    else:
        adv_yaw = tgt_yaw

    rel_pos = adv_pos - tgt_pos
    rel_dist = float(np.linalg.norm(rel_pos))

    cos_t = np.cos(tgt_yaw)
    sin_t = np.sin(tgt_yaw)
    u = np.array([cos_t, sin_t])              # TGT forward unit vector
    u_perp = np.array([-sin_t, cos_t])        # TGT left unit vector

    s = float(np.dot(rel_pos, u))
    l = float(np.dot(rel_pos, u_perp))
    abs_l = abs(l)

    dv = tgt_vel - adv_vel
    closing = float(np.dot(dv, u))            # positive = TGT catching up
    lateral_closing = float(np.dot(dv, u_perp))

    rel_speed = adv_speed - tgt_speed

    yaw_diff = adv_yaw - tgt_yaw
    yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
    cos_yaw_diff = float(np.cos(yaw_diff))

    # TTC using longitudinal gap
    bumper = 1.0
    tgt_len = float(tgt["width"])             # AD4CHE convention: width=length
    adv_len = float(adv["width"])
    gap = s - 0.5 * (tgt_len + adv_len) - bumper
    if closing > 0.1:
        ttc = float(np.clip(gap / closing, 0.0, 999.0))
    else:
        ttc = 999.0

    # Bearing from adv to tgt
    bearing = float(np.arctan2(tgt_pos[1] - adv_pos[1], tgt_pos[0] - adv_pos[0]))
    cos_bearing = float(np.cos(bearing - adv_yaw))

    if _FEATURE_MODE == "runtime":
        # Runtime-compatible 13 features (matches scripts/decision_tree.py).
        # Compute dist_rate_5 and abs_l_delta_5 from a 15-raw-frame history
        # window: raw 30 Hz × 15 frames = 0.5 s, matching nuScenes sim 5 frames
        # × 0.1 s.
        hist_frame = frame - 15
        if hist_frame < int(df["frame"].min()):
            return None
        tgt_hist = _row_at(df, hist_frame, tgt_id)
        adv_hist = _row_at(df, hist_frame, adv_id)
        if tgt_hist is None or adv_hist is None:
            return None
        tgt_hist_pos = np.array([tgt_hist["x"], tgt_hist["y"]], dtype=np.float64)
        adv_hist_pos = np.array([adv_hist["x"], adv_hist["y"]], dtype=np.float64)
        rel_pos_hist = adv_hist_pos - tgt_hist_pos
        rel_dist_hist = float(np.linalg.norm(rel_pos_hist))
        dist_rate_5 = (rel_dist - rel_dist_hist) / 0.5  # m/s; negative = closing
        # Lateral offset of adv in TGT frame at the historical frame. Use TGT's
        # instantaneous heading at `hist_frame` for the basis.
        tgt_vel_hist = np.array([tgt_hist["xVelocity"], tgt_hist["yVelocity"]], dtype=np.float64)
        tgt_speed_hist = float(np.linalg.norm(tgt_vel_hist))
        if tgt_speed_hist > 0.2:
            yaw_h = float(np.arctan2(tgt_vel_hist[1], tgt_vel_hist[0]))
        else:
            yaw_h = tgt_yaw
        u_perp_h = np.array([-np.sin(yaw_h), np.cos(yaw_h)], dtype=np.float64)
        l_hist = float(np.dot(rel_pos_hist, u_perp_h))
        abs_l_delta_5 = abs(l) - abs(l_hist)

        curr_speed = adv_speed  # same as 'speed_agents' in nuScenes version
        return np.array([
            rel_dist, s, l, gap,
            closing, lateral_closing, rel_speed, curr_speed,
            cos_yaw_diff, ttc, cos_bearing,
            dist_rate_5, abs_l_delta_5,
        ], dtype=np.float32)

    # --- 'full' mode (legacy) ---
    same_lane = 1.0 if int(tgt["laneId"]) == int(adv["laneId"]) else 0.0
    lane_diff = float(abs(int(tgt["laneId"]) - int(adv["laneId"])))
    return np.array([
        rel_dist, s, l, abs_l, closing, lateral_closing,
        rel_speed, adv_speed, cos_yaw_diff, ttc, cos_bearing,
        same_lane, lane_diff,
    ], dtype=np.float32)


def get_tgt_neighbors_at_frame(df: pd.DataFrame, frame: int, tgt_id: int):
    """Return the 8 pre-computed neighbor ids of TGT at `frame` (drops 0s)."""
    row = df[(df["frame"] == frame) & (df["id"] == tgt_id)]
    if row.empty:
        return []
    r = row.iloc[0]
    out = []
    seen = set()
    for col in NEIGHBOR_COLS:
        nid = int(r[col])
        if nid == 0 or nid in seen:
            continue
        seen.add(nid)
        out.append(nid)
    return out


def build_dataset(labels: dict, root: str, verbose: bool = True):
    """Build (X, y, meta) training set from label dict.

    For each labeled pair (tgt, adv) at scene_key:
      - 1 positive sample: features(tgt, adv) at slice start frame
      - up to 7 negative samples: features(tgt, other_neighbor)
        where other_neighbor is one of the remaining 8 neighbors
        present at the start frame but != adv.
    """
    # Group label entries by (loc, DJI) so we only load each CSV once
    by_rec = defaultdict(list)
    for scene_key, pairs in labels.items():
        if not pairs:
            continue
        # scene_key format: ad4che_{loc}_{dji}_frame_{start}
        # parse: split on '_frame_' first
        head, _, tail = scene_key.partition("_frame_")
        try:
            start_frame = int(tail)
        except ValueError:
            continue
        # head = "ad4che_{loc}_{DJI_xxxx}" — split by underscore but keep DJI_xxxx together
        parts = head.split("_", 2)   # ["ad4che", "{loc}", "DJI_xxxx"]
        if len(parts) != 3 or parts[0] != "ad4che":
            continue
        loc, dji = parts[1], parts[2]
        # `pairs` is a [[tgt_str, adv_str], ...] list (nuScenes convention).
        pair_dict = {int(t): int(a) for t, a in pairs}
        by_rec[(loc, dji)].append((start_frame, pair_dict))

    X_list, y_list, meta_list = [], [], []
    n_recordings = len(by_rec)
    for ri, ((loc, dji), slices) in enumerate(by_rec.items()):
        rec_dir = find_recording_dir(root, loc, dji)
        try:
            df = load_recording_csv(rec_dir)
        except FileNotFoundError as e:
            print(f"[skip] {loc}/{dji}: {e}")
            continue
        if verbose:
            print(f"[{ri+1}/{n_recordings}] {loc}/{dji}: {len(slices)} slices")

        for start_frame, pair_dict in slices:
            # Positive + negatives come from the 8-neighbor candidate list
            for tgt_id, adv_id in pair_dict.items():
                neighbors = get_tgt_neighbors_at_frame(df, start_frame, tgt_id)
                if adv_id not in neighbors:
                    # Labeled adv is outside the 8-neighbor set (rare edge case).
                    # Still add as positive so DT sees it but without negatives
                    # from the candidate list.
                    feat = extract_pair_features(df, start_frame, tgt_id, adv_id)
                    if feat is not None:
                        X_list.append(feat)
                        y_list.append(1)
                        meta_list.append((loc, dji, start_frame, tgt_id, adv_id, "pos_outside"))
                    continue

                # Positive
                feat = extract_pair_features(df, start_frame, tgt_id, adv_id)
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(1)
                    meta_list.append((loc, dji, start_frame, tgt_id, adv_id, "pos"))

                # Negatives = other 8 neighbors
                for nid in neighbors:
                    if nid == adv_id:
                        continue
                    feat = extract_pair_features(df, start_frame, tgt_id, nid)
                    if feat is not None:
                        X_list.append(feat)
                        y_list.append(0)
                        meta_list.append((loc, dji, start_frame, tgt_id, nid, "neg"))

    X = np.array(X_list, dtype=np.float32) if X_list else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array(y_list, dtype=np.int32) if y_list else np.zeros(0, dtype=np.int32)
    print(f"\nDataset: {len(X)} samples, {(y == 1).sum()} positive, {(y == 0).sum()} negative")
    return X, y, meta_list


def train(X, y):
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    w = np.ones_like(y, dtype=float)
    if pos > 0:
        w[y == 1] = float(np.sqrt(max(neg, 1) / max(pos, 1)))
    model = HistGradientBoostingClassifier(
        max_depth=5, max_iter=600, learning_rate=0.03,
        min_samples_leaf=5, random_state=42,
    )
    model.fit(X, y, sample_weight=w)
    return model


def evaluate(model, X, y, split_name: str):
    if len(X) == 0:
        print(f"{split_name}: empty split, skipping")
        return 0.0
    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    print(f"\n{'=' * 60}")
    print(f"{split_name} accuracy: {acc:.4f}")
    print(classification_report(y, y_pred, target_names=["Neighbor", "Adv"]))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))
    return acc


def top1_scene_accuracy(model, labels: dict, root: str):
    """Per-slice top-1 accuracy: for each labeled pair, score all 8 neighbors
    with the trained model and check whether the labeled adv gets highest
    probability. This measures the metric we actually care about at rollout.
    """
    correct = 0
    total = 0
    by_rec = defaultdict(list)
    for scene_key, pairs in labels.items():
        if not pairs:
            continue
        head, _, tail = scene_key.partition("_frame_")
        try:
            start_frame = int(tail)
        except ValueError:
            continue
        parts = head.split("_", 2)
        if len(parts) != 3:
            continue
        loc, dji = parts[1], parts[2]
        pair_dict = {int(t): int(a) for t, a in pairs}
        by_rec[(loc, dji)].append((start_frame, pair_dict))

    for (loc, dji), slices in by_rec.items():
        try:
            df = load_recording_csv(find_recording_dir(root, loc, dji))
        except FileNotFoundError:
            continue
        for start_frame, pair_dict in slices:
            for tgt_id, adv_id in pair_dict.items():
                cands = get_tgt_neighbors_at_frame(df, start_frame, tgt_id)
                if not cands or adv_id not in cands:
                    continue
                feats = []
                ids = []
                for nid in cands:
                    f = extract_pair_features(df, start_frame, tgt_id, nid)
                    if f is None:
                        continue
                    feats.append(f)
                    ids.append(nid)
                if not feats:
                    continue
                arr = np.stack(feats)
                probs = model.predict_proba(arr)[:, 1]
                pred = ids[int(np.argmax(probs))]
                if pred == adv_id:
                    correct += 1
                total += 1
    return correct, total


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--labels", type=str, default="labels/ad4che_labels_all.json")
    parser.add_argument("--root", type=str, default="~/AD4CHE")
    parser.add_argument("--output", type=str, default="checkpoints/ad4che_decision_tree.pkl")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--runtime_compat", action="store_true",
                        help="Train with the 13 nuScenes-compatible features "
                             "(scripts/decision_tree/nusc.py) so the model can "
                             "run inside TargetAdvOthersWrapper at rollout time "
                             "without any CSV lookups.")


def run(args) -> None:
    global _FEATURE_MODE, FEATURE_NAMES
    _FEATURE_MODE = "runtime" if args.runtime_compat else "full"
    FEATURE_NAMES = FEATURE_NAMES_RUNTIME if args.runtime_compat else FEATURE_NAMES_FULL
    print(f"Feature mode: {_FEATURE_MODE}  ({len(FEATURE_NAMES)} features)")

    root = os.path.expanduser(args.root)
    with open(args.labels) as f:
        labels = json.load(f)
    n_scenes = len(labels)
    n_pairs = sum(len(v) for v in labels.values())
    print(f"Loaded {n_scenes} scenes, {n_pairs} pairs from {args.labels}")

    print("\nBuilding dataset...")
    X, y, meta = build_dataset(labels, root)

    if len(X) == 0:
        raise SystemExit("empty dataset — check that CSV paths resolve under --root")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    print(f"Train: {len(X_train)} ({(y_train == 1).sum()} pos)  "
          f"Test: {len(X_test)} ({(y_test == 1).sum()} pos)")

    print("\nTraining HistGradientBoostingClassifier...")
    model = train(X_train, y_train)

    evaluate(model, X_train, y_train, "Train")
    evaluate(model, X_test, y_test, "Test")

    # Per-pair Top-1 metric (the one that matters for rollout)
    print("\nComputing per-pair Top-1 accuracy over 8-neighbor candidates...")
    correct, total = top1_scene_accuracy(model, labels, root)
    if total > 0:
        print(f"Top-1 accuracy: {correct}/{total} = {correct/total:.4f}")

    # Feature importances (permutation importance is expensive;
    # HistGradientBoostingClassifier exposes them directly only via a permutation
    # importance tool, so skip for speed and just show the built-in if present)
    if hasattr(model, "feature_importances_"):
        print("\nFeature importances:")
        order = np.argsort(model.feature_importances_)[::-1]
        for i in order:
            print(f"  {FEATURE_NAMES[i]:20s} {model.feature_importances_[i]:.4f}")

    with open(args.output, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": FEATURE_NAMES,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "top1_correct": correct if total > 0 else None,
            "top1_total": total if total > 0 else None,
        }, f)
    print(f"\nModel saved to {args.output}")
    print("  (decision_tree_model.pkl and round_decision_tree.pkl are NOT modified)")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Train the AD4CHE adversary-selection decision tree")
    add_args(_parser)
    run(_parser.parse_args())
