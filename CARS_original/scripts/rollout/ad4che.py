"""AD4CHE cross-dataset rollout.

Runs the nuScenes-trained CARS adversary on AD4CHE highway recordings as
a zero-shot (re-normalized) generalization test. Two-tier TGT sampling:

  Tier 1 — labeled recordings (7 recordings, 261 pairs):
    Uses the exact (TGT, ADV) pairs from labels/ad4che_labels_all.json.
    Slice start frames come directly from the label file.
    Tier 1 does NOT use the decision tree (ADV is from ground-truth labels).

  Tier 2 — unlabeled recordings (61 recordings, 4 TGT each):
    For each unlabeled recording, pick the MIDDLE slice and sample up to
    N=4 moving agents at that slice's first frame as TGTs. Query
    ad4che_decision_tree.pkl to pick ADV for each TGT.

All pairs run in closed-loop closed-loop CARS rollout. Target / others use
frozen g8.ckpt, adversary uses Mar-11-best.ckpt. Norm_info is overridden
with AD4CHE statistics from ad4che_norm_info_dt01.pt.

The first invocation triggers a one-off trajdata-cache build (~1-2 min,
~361 MB) under ``cfg.trajdata_cache_location/ad4che/``; subsequent runs
reuse it. Pass ``--rebuild_cache`` to force a rebuild.

Usage:
    python -m scripts.rollout ad4che \
        --config_file configs/simulation_config_ad4che.json \
        --ad4che_norm ad4che_norm_info_dt01.pt \
        --dt_model checkpoints/ad4che_decision_tree.pkl \
        --labels labels/ad4che_labels_all.json \
        --stop_after_collisions 10
"""
import argparse
import importlib
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dill
import h5py
import numpy as np
import pandas as pd
import torch

from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env
from tbsim.utils.safety_critical_batch_utils import set_global_trajdata_batch_raster_cfg
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.evaluation.env_builders import EnvCLDBuilder
from tbsim.policies.decision_tree_wrapper import (
    TargetAdvOthersWrapper,
    safety_critical_guided_rollout_with_custom_target,
)
from trajdata.data_structures.agent import AgentType


# ---------------------------------------------------------------------------
# One-off trajdata cache builder (inlined from the old build_ad4che_cache.py).
# ---------------------------------------------------------------------------

def _ensure_ad4che_cache(data_dir: Path, cache_location: Path,
                          force: bool = False) -> None:
    """Build the trajdata AD4CHE cache if missing (or if ``force=True``).

    First-time build reads the 68 raw AD4CHE recordings and writes the parsed
    trajectories + rasterised maps to ``cache_location/ad4che/``. Takes
    ~1-2 minutes, produces ~361 MB. Subsequent calls short-circuit when the
    cache is present.
    """
    cache_ad4che = cache_location / "ad4che"
    if cache_ad4che.exists() and not force:
        n_scenes = sum(1 for p in cache_ad4che.iterdir() if p.is_dir())
        print(f"[ad4che cache] present at {cache_ad4che} ({n_scenes} scene entries)")
        return

    if not data_dir.exists():
        raise FileNotFoundError(
            f"AD4CHE raw data not found at {data_dir} — pass --ad4che_root."
        )
    cache_location.mkdir(parents=True, exist_ok=True)
    print(f"[ad4che cache] building at {cache_ad4che} (~1-2 min, ~361 MB)...")

    from trajdata import UnifiedDataset as _UnifiedDataset

    ds = _UnifiedDataset(
        desired_data=["ad4che-train"],
        data_dirs={"ad4che": str(data_dir)},
        desired_dt=0.1,
        history_sec=(2.0, 2.0),
        future_sec=(2.0, 2.0),
        only_types=[AgentType.VEHICLE],
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "return_rgb": True,
            "offset_frac_xy": [-0.5, 0.0],
            "no_map_fill_value": -1.0,
        },
        incl_vector_map=True,
        vector_map_params={
            "incl_road_lanes": True,
            "incl_road_areas": False,
            "incl_ped_crosswalks": False,
            "incl_ped_walkways": False,
            "no_collate": True,
        },
        rebuild_cache=True,
        rebuild_maps=True,
        cache_location=str(cache_location),
        centric="agent",
        standardize_data=True,
        ego_only=False,
        verbose=True,
    )
    print(f"[ad4che cache] done — {len(ds)} samples")


# ---------------------------------------------------------------------------
# AD4CHE raw-CSV helpers for Tier 2 pair pre-selection.
#
# Runtime adversary switching is handled by TargetAdvOthersWrapper using the
# same 13 nuScenes-compatible features from scripts/decision_tree/nusc.py, so
# the DT file referenced here (ad4che_decision_tree_runtime.pkl) must be
# trained with the same feature set
# (see scripts/decision_tree/ad4che.py --runtime_compat).
# ---------------------------------------------------------------------------

NEIGHBOR_COLS = [
    "precedingId", "followingId",
    "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
    "rightPrecedingId", "rightAlongsideId", "rightFollowingId",
]

# 13 features, identical in meaning and order to scripts/decision_tree.py.
DT_FEATURE_NAMES = [
    "rel_dist", "s", "l", "gap",
    "closing", "lateral_closing", "rel_speed", "curr_speed",
    "cos_yaw_diff", "ttc", "cos_bearing",
    "dist_rate_5", "abs_l_delta_5",
]

# Raw AD4CHE CSV is 30 Hz. nuScenes DT uses 5 sim frames (0.5 s) of history
# for dist_rate_5 / abs_l_delta_5; at 30 Hz that is 15 raw frames.
DT_HIST_RAW_FRAMES = 15
DT_HIST_SECONDS = 0.5


def _find_rec_dir(ad4che_root: Path, loc: str, dji: str) -> Path:
    return ad4che_root / loc / dji


def _load_raw_tracks(rec_dir: Path) -> pd.DataFrame:
    csvs = list(rec_dir.glob("*_tracks_pixel.csv"))
    if not csvs:
        raise FileNotFoundError(f"no *_tracks_pixel.csv in {rec_dir}")
    df = pd.read_csv(csvs[0])
    df.columns = [c.strip() for c in df.columns]
    return df


def _row_at(df: pd.DataFrame, frame: int, agent_id: int):
    r = df[(df["frame"] == frame) & (df["id"] == agent_id)]
    return r.iloc[0] if not r.empty else None


def _extract_dt_features(df: pd.DataFrame, raw_frame: int,
                         tgt_id: int, adv_id: int) -> Optional[np.ndarray]:
    """Compute the 13 nuScenes-compatible features for (tgt, adv) at
    `raw_frame` from AD4CHE raw CSV. Feature order matches
    scripts/decision_tree.py::extract_features so the same DT can be used
    here (Tier 2 pre-selection) and inside TargetAdvOthersWrapper (runtime).
    """
    tgt = _row_at(df, raw_frame, tgt_id)
    adv = _row_at(df, raw_frame, adv_id)
    if tgt is None or adv is None:
        return None

    tgt_pos = np.array([tgt["x"], tgt["y"]], dtype=np.float64)
    adv_pos = np.array([adv["x"], adv["y"]], dtype=np.float64)
    tgt_vel = np.array([tgt["xVelocity"], tgt["yVelocity"]], dtype=np.float64)
    adv_vel = np.array([adv["xVelocity"], adv["yVelocity"]], dtype=np.float64)
    tgt_speed = float(np.linalg.norm(tgt_vel))
    adv_speed = float(np.linalg.norm(adv_vel))

    tgt_yaw = float(np.arctan2(tgt_vel[1], tgt_vel[0])) if tgt_speed > 0.2 else 0.0
    adv_yaw = float(np.arctan2(adv_vel[1], adv_vel[0])) if adv_speed > 0.2 else tgt_yaw

    rel_pos = adv_pos - tgt_pos
    rel_dist = float(np.linalg.norm(rel_pos))
    cos_t, sin_t = np.cos(tgt_yaw), np.sin(tgt_yaw)
    u = np.array([cos_t, sin_t])
    u_perp = np.array([-sin_t, cos_t])
    s = float(np.dot(rel_pos, u))
    l = float(np.dot(rel_pos, u_perp))

    dv = tgt_vel - adv_vel
    closing = float(np.dot(dv, u))
    lateral_closing = float(np.dot(dv, u_perp))
    rel_speed = adv_speed - tgt_speed

    yaw_diff = adv_yaw - tgt_yaw
    yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
    cos_yaw_diff = float(np.cos(yaw_diff))

    # AD4CHE reports length in the `width` column, width in `height`.
    bumper = 1.0
    tgt_len = float(tgt["width"])
    adv_len = float(adv["width"])
    gap = s - 0.5 * (tgt_len + adv_len) - bumper
    ttc = float(np.clip(gap / closing, 0.0, 999.0)) if closing > 0.1 else 999.0

    bearing = float(np.arctan2(tgt_pos[1] - adv_pos[1], tgt_pos[0] - adv_pos[0]))
    cos_bearing = float(np.cos(bearing - adv_yaw))

    # History window for dist_rate_5 / abs_l_delta_5: 15 raw frames = 0.5 s.
    hist_frame = raw_frame - DT_HIST_RAW_FRAMES
    if hist_frame < int(df["frame"].min()):
        return None
    tgt_h = _row_at(df, hist_frame, tgt_id)
    adv_h = _row_at(df, hist_frame, adv_id)
    if tgt_h is None or adv_h is None:
        return None
    tgt_pos_h = np.array([tgt_h["x"], tgt_h["y"]], dtype=np.float64)
    adv_pos_h = np.array([adv_h["x"], adv_h["y"]], dtype=np.float64)
    rel_pos_h = adv_pos_h - tgt_pos_h
    rel_dist_h = float(np.linalg.norm(rel_pos_h))
    dist_rate_5 = (rel_dist - rel_dist_h) / DT_HIST_SECONDS
    # Lateral offset at historical frame in that frame's TGT heading basis.
    tgt_vel_h = np.array([tgt_h["xVelocity"], tgt_h["yVelocity"]], dtype=np.float64)
    if float(np.linalg.norm(tgt_vel_h)) > 0.2:
        yaw_h = float(np.arctan2(tgt_vel_h[1], tgt_vel_h[0]))
    else:
        yaw_h = tgt_yaw
    u_perp_h = np.array([-np.sin(yaw_h), np.cos(yaw_h)], dtype=np.float64)
    l_h = float(np.dot(rel_pos_h, u_perp_h))
    abs_l_delta_5 = abs(l) - abs(l_h)

    return np.array([
        rel_dist, s, l, gap,
        closing, lateral_closing, rel_speed, adv_speed,   # adv_speed == curr_speed in nusc
        cos_yaw_diff, ttc, cos_bearing,
        dist_rate_5, abs_l_delta_5,
    ], dtype=np.float32)


def _neighbors_at_frame(df: pd.DataFrame, raw_frame: int, tgt_id: int) -> List[int]:
    row = df[(df["frame"] == raw_frame) & (df["id"] == tgt_id)]
    if row.empty:
        return []
    r = row.iloc[0]
    seen, out = set(), []
    for col in NEIGHBOR_COLS:
        nid = int(r[col])
        if nid == 0 or nid in seen:
            continue
        seen.add(nid)
        out.append(nid)
    return out


def _dt_pick_adv(df: pd.DataFrame, raw_frame: int, tgt_id: int, dt_model):
    """Return (adv_id, score) for the highest-scored neighbor of tgt_id."""
    cands = _neighbors_at_frame(df, raw_frame, tgt_id)
    if not cands:
        return None, 0.0
    feats = []
    keep = []
    for nid in cands:
        f = _extract_dt_features(df, raw_frame, tgt_id, nid)
        if f is not None:
            feats.append(f)
            keep.append(nid)
    if not feats:
        return None, 0.0
    X = np.stack(feats)
    scores = dt_model.predict_proba(X)[:, 1]
    best = int(np.argmax(scores))
    return keep[best], float(scores[best])


def _moving_agents_at_raw_frame(df: pd.DataFrame, raw_frame: int,
                                min_speed: float = 2.0) -> List[int]:
    sub = df[df["frame"] == raw_frame]
    spd = np.hypot(sub["xVelocity"], sub["yVelocity"])
    return sub.loc[spd >= min_speed, "id"].astype(int).tolist()


# ---------------------------------------------------------------------------
# Pair list construction
# ---------------------------------------------------------------------------

def build_tier1_pairs(labels: dict) -> Dict[str, List[Tuple[str, int, int]]]:
    """Parse labeled pairs. Returns {scene_name: [(tgt, adv, raw_frame)]}.

    scene_name format: "ad4che_14_DJI_0009"
    raw_frame: 30 Hz frame index (from label file's `_frame_N` suffix).
    """
    out: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    for scene_key, pairs in labels.items():
        if not pairs:
            continue
        head, _, tail = scene_key.partition("_frame_")
        try:
            raw_frame = int(tail)
        except ValueError:
            continue
        # head = "ad4che_{loc}_{DJI_xxxx}"
        parts = head.split("_", 2)
        if len(parts) != 3 or parts[0] != "ad4che":
            continue
        scene_name = head
        # `pairs` is a [[tgt_str, adv_str], ...] list (nuScenes convention).
        for tgt, adv in pairs:
            out[scene_name].append((str(tgt), str(adv), raw_frame))
    return out


def build_tier2_pairs(
    unlabeled_scenes: List[str],
    ad4che_root: Path,
    dt_model,
    n_per_rec: int,
    rng: random.Random,
    slices_per_rec: int = 6,
) -> Dict[str, List[Tuple[str, str, int]]]:
    """For each unlabeled recording, pick `slices_per_rec` slices evenly
    spaced across the recording and sample up to `n_per_rec` moving TGTs
    at each slice's first frame. DT picks the ADV per TGT.

    Returns {scene_name: [(tgt, adv, raw_frame), ...]}.
    """
    out: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    for scene_name in unlabeled_scenes:
        # scene_name = "ad4che_14_DJI_0010"
        parts = scene_name.split("_", 2)
        if len(parts) != 3:
            continue
        _, loc, dji = parts
        rec_dir = _find_rec_dir(ad4che_root, loc, dji)
        try:
            df = _load_raw_tracks(rec_dir)
        except FileNotFoundError:
            print(f"  [skip] {scene_name}: {rec_dir} not found")
            continue

        f_min = int(df["frame"].min())
        f_max = int(df["frame"].max())
        if f_max - f_min < 150:   # need at least ~5 s of data
            continue

        # `slices_per_rec` raw frames, uniformly spaced in the INNER 80% of the
        # recording (skip first/last 10% so there's always a full 5 s of
        # history available and agents are mid-stream).
        span = f_max - f_min
        inner_start = f_min + int(0.1 * span)
        inner_end = f_max - int(0.1 * span)
        raw_frames = [
            int(round(inner_start + (inner_end - inner_start) * i / max(slices_per_rec - 1, 1)))
            for i in range(slices_per_rec)
        ]
        # Align each raw frame to a multiple of 3 (matches trajdata 10 Hz).
        raw_frames = sorted(set((f // 3) * 3 for f in raw_frames))

        seen_pairs_this_rec = set()
        for raw_frame in raw_frames:
            moving = _moving_agents_at_raw_frame(df, raw_frame, min_speed=2.0)
            if not moving:
                continue
            rng.shuffle(moving)
            n_added_at_slice = 0
            for tgt_id in moving:
                if n_added_at_slice >= n_per_rec:
                    break
                adv_id, _score = _dt_pick_adv(df, raw_frame, tgt_id, dt_model)
                if adv_id is None:
                    continue
                key = (raw_frame, tgt_id, adv_id)
                if key in seen_pairs_this_rec:
                    continue
                seen_pairs_this_rec.add(key)
                out[scene_name].append((str(tgt_id), str(adv_id), raw_frame))
                n_added_at_slice += 1
    return out


# ---------------------------------------------------------------------------
# Scene index lookup
# ---------------------------------------------------------------------------

def find_scene_index_map(env) -> Dict[str, int]:
    """Map scene_name → trajdata scene_index (0-based)."""
    mapping: Dict[str, int] = {}
    for si, sp in enumerate(env.dataset._scene_index):
        with open(sp, "rb") as f:
            sc = dill.load(f)
        mapping[sc.name] = si
    return mapping


# ---------------------------------------------------------------------------
# HDF5 save helper
# ---------------------------------------------------------------------------

def save_episode(h5_path: str, ep_key: str, scene_buf: dict) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "a") as h5f:
        for mk, val in scene_buf.items():
            h5key = f"/{ep_key}/{mk}"
            if h5key in h5f:
                del h5f[h5key]
            try:
                h5f.create_dataset(h5key, data=val)
            except TypeError:
                if isinstance(val, (str, list)):
                    arr = np.array(
                        [val] if isinstance(val, str) else val, dtype=object
                    )
                    h5f.create_dataset(h5key, data=arr, dtype=dt)
    print(f"  [SAVED] {ep_key}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config_file", type=str, default="configs/simulation_config_ad4che.json")
    parser.add_argument("--labels", type=str, default="labels/ad4che_labels_all.json")
    parser.add_argument("--dt_model", type=str, default="checkpoints/ad4che_decision_tree_runtime.pkl",
                        help="Decision tree for BOTH Tier 2 pair pre-selection "
                             "(from raw CSV) AND runtime switching inside the "
                             "TargetAdvOthersWrapper. Must be trained with the "
                             "13 nuScenes-compatible features "
                             "(scripts/decision_tree_ad4che.py --runtime_compat).")
    parser.add_argument("--ad4che_norm", type=str, default="ad4che_norm_info_dt01.pt")
    parser.add_argument("--ad4che_root", type=str, default=os.path.expanduser("~/AD4CHE"))
    parser.add_argument("--n_per_rec", type=int, default=6,
                        help="Tier 2: TGTs sampled per slice per recording")
    parser.add_argument("--slices_per_rec", type=int, default=6,
                        help="Tier 2: slices sampled per recording (uniform in inner 80%%)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="0-indexed shard id for parallel runs (0..num_shards-1)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of parallel shards")
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--no-render", action="store_false", dest="render")
    parser.add_argument("--gt_others", action="store_true", default=True,
                        help="Other agents replay GT trajectory (tgt + adv use model)")
    parser.add_argument("--no-gt_others", action="store_false", dest="gt_others")
    parser.add_argument("--save_all", action="store_true", default=False)
    parser.add_argument("--stop_after_collisions", type=int, default=None,
                        help="Early stop after N collision episodes (smoke test)")
    parser.add_argument("--results_name", type=str, default=None,
                        help="Override cfg.name so the HDF5 is written to a "
                             "separate directory (useful for smoke tests that "
                             "must not clobber an existing rollout).")
    parser.add_argument("--only_scene", type=str, default=None,
                        help="Restrict pair list to a single scene "
                             "(e.g. 'ad4che_14_DJI_0009'); skips Tier 2 "
                             "scanning of unlabeled recordings.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuild of the trajdata AD4CHE cache before rollout")


def run(args) -> None:
    if args.num_shards < 1 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(
            f"invalid --shard_id {args.shard_id} for --num_shards {args.num_shards}"
        )

    ad4che_root = Path(args.ad4che_root)

    # --- Config ---
    cfg = SceneEditingConfig(registered_name="visier")
    cfg.update(**json.load(open(args.config_file)))
    cfg.seed = args.seed
    if args.results_name:
        cfg.name = args.results_name
    cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)
    # Shard-aware HDF5 path so parallel runs don't clobber each other
    if args.num_shards > 1:
        hdf5_name = f"data_shard{args.shard_id + 1}of{args.num_shards}.hdf5"
    else:
        hdf5_name = "data.hdf5"
    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, hdf5_name)
    for k in cfg[cfg.env]:
        cfg[k] = cfg[cfg.env][k]
    cfg.pop("nusc")
    cfg.pop("trajdata")
    cfg.lock()

    # Build / verify the AD4CHE trajdata cache before EnvCLDBuilder tries to use it.
    _ensure_ad4che_cache(
        data_dir=ad4che_root,
        cache_location=Path(cfg.trajdata_cache_location).expanduser(),
        force=args.rebuild_cache,
    )

    os.makedirs(cfg.results_dir, exist_ok=True)
    if args.render:
        os.makedirs(os.path.join(cfg.results_dir, "viz/"), exist_ok=True)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(cfg.trajdata_source_test[0])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Policies ---
    composers = importlib.import_module("tbsim.evaluation.policy_composers")
    Composer = getattr(composers, cfg.eval_class)

    target_policy, exp_config = Composer(cfg, device).get_policy()
    agents_planner, _ = Composer(cfg, device).get_planner()
    others_predictor, _ = Composer(cfg, device).get_predictor()

    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)

    for p in (target_policy, agents_planner, others_predictor):
        p.model.nets["policy"].eval()
        p.model.nets["policy"].requires_grad_(False)
        if hasattr(p.model.nets["policy"], "sigma_gmm_scale"):
            p.model.nets["policy"].sigma_gmm_scale = 3
        dyn = getattr(p.model.nets["policy"], "dyn", None)
        if dyn is not None and hasattr(dyn, "vbound"):
            dyn.vbound[0] = 0.0

    # --- Override norm_info with AD4CHE statistics ---
    if args.ad4che_norm and os.path.exists(args.ad4che_norm):
        print(f"Overriding norm_info from {args.ad4che_norm}")
        rn = torch.load(args.ad4che_norm)
        for p in (target_policy, agents_planner, others_predictor):
            m = p.model.nets["policy"]
            m.add_coeffs = np.array(rn["action_norm_info_center"][0], dtype="float32")
            m.div_coeffs = np.array(rn["action_norm_info_center"][1], dtype="float32")
            m.center_hist.mean.copy_(torch.tensor(rn["norm_info_center"][0], dtype=torch.float32))
            m.center_hist.std.copy_(torch.tensor(rn["norm_info_center"][1], dtype=torch.float32))
            m.ego_encoder.mean_pos.copy_(torch.tensor(rn["norm_info_ego"][0][:2], dtype=torch.float32))
            m.ego_encoder.std_pos.copy_(torch.tensor(rn["norm_info_ego"][1][:2], dtype=torch.float32))
            m.ego_encoder.mean_speed.copy_(torch.tensor([rn["norm_info_ego"][0][2]], dtype=torch.float32))
            m.ego_encoder.std_speed.copy_(torch.tensor([rn["norm_info_ego"][1][2]], dtype=torch.float32))
            m.neighbor_hist.neigh_encode.mean.copy_(torch.tensor(rn["norm_info_neighbor"][0], dtype=torch.float32))
            m.neighbor_hist.neigh_encode.std.copy_(torch.tensor(rn["norm_info_neighbor"][1], dtype=torch.float32))
        print("  Norm_info overridden for all 3 policies")

    # --- Environment (builds cache if missing) ---
    env = EnvCLDBuilder(
        eval_config=cfg, exp_config=exp_config, device=device, log_data=True
    ).get_env()

    scene_name_to_idx = find_scene_index_map(env)
    print(f"Env has {len(scene_name_to_idx)} scenes")

    # --- Decision tree: one model for BOTH Tier 2 pair pre-selection AND
    #     runtime adversary switching inside TargetAdvOthersWrapper. Training
    #     uses the 13 nuScenes-compatible kinematic features so the same model
    #     can be fed by raw-CSV features here or by sim-observation features
    #     inside the wrapper.
    with open(args.dt_model, "rb") as f:
        dt_data = pickle.load(f)
    dt_model = dt_data["model"] if isinstance(dt_data, dict) else dt_data
    runtime_dt = dt_model
    print(f"Decision tree loaded from {args.dt_model} "
          f"(used for both Tier 2 pre-selection and runtime switching)")

    # --- Labels ---
    with open(args.labels) as f:
        labels = json.load(f)
    tier1_pairs = build_tier1_pairs(labels)
    labeled_scenes = set(tier1_pairs.keys())
    print(f"Tier 1: {len(labeled_scenes)} labeled recordings, "
          f"{sum(len(v) for v in tier1_pairs.values())} pairs")

    # Smoke-test filter: restrict everything to a single scene and skip Tier 2.
    if args.only_scene:
        print(f"[only_scene] restricting to {args.only_scene}")
        tier1_pairs = {k: v for k, v in tier1_pairs.items() if k == args.only_scene}
        labeled_scenes = set(tier1_pairs.keys())

    # --- Tier 2: unlabeled recordings ---
    if args.only_scene:
        # Skip Tier 2 entirely when in single-scene mode.
        tier2_pairs = {}
        if not tier1_pairs:
            # The requested scene has no labels — fall back to Tier 2 on just
            # that one scene so the smoke test still has pairs to run.
            all_scene_names = list(scene_name_to_idx.keys())
            if args.only_scene in all_scene_names:
                tier2_pairs = build_tier2_pairs(
                    [args.only_scene], ad4che_root, dt_model,
                    n_per_rec=args.n_per_rec, rng=rng,
                    slices_per_rec=args.slices_per_rec,
                )
    else:
        all_scene_names = list(scene_name_to_idx.keys())
        unlabeled = sorted(set(all_scene_names) - labeled_scenes)
        tier2_pairs = build_tier2_pairs(
            unlabeled, ad4che_root, dt_model,
            n_per_rec=args.n_per_rec, rng=rng,
            slices_per_rec=args.slices_per_rec,
        )
    print(f"Tier 2: {len(tier2_pairs)} unlabeled recordings, "
          f"{sum(len(v) for v in tier2_pairs.values())} pairs")

    # --- Merge pair list ---
    all_pair_entries: List[Tuple[str, str, str, int, str]] = []
    # tier tag for logging
    for scene_name, pairs in tier1_pairs.items():
        for tgt, adv, raw_frame in pairs:
            all_pair_entries.append((scene_name, tgt, adv, raw_frame, "T1"))
    for scene_name, pairs in tier2_pairs.items():
        for tgt, adv, raw_frame in pairs:
            all_pair_entries.append((scene_name, tgt, adv, raw_frame, "T2"))

    global_total = len(all_pair_entries)
    print(f"\nTOTAL (full dataset): {global_total} pairs")

    # --- Shard filter ---
    # Stride split so each shard gets a mix of Tier 1 and Tier 2 pairs.
    if args.num_shards > 1:
        all_pair_entries = [
            p for i, p in enumerate(all_pair_entries)
            if i % args.num_shards == args.shard_id
        ]
        print(f"Shard {args.shard_id + 1}/{args.num_shards}: "
              f"{len(all_pair_entries)} pairs to roll out "
              f"(from {global_total} total)")
    else:
        print(f"TOTAL: {len(all_pair_entries)} pairs to roll out")

    # --- Renderer ---
    render_rasterizer = None
    if args.render:
        from tbsim.utils.scene_edit_utils import get_trajdata_renderer
        render_rasterizer = get_trajdata_renderer(
            cfg.trajdata_source_train, cfg.trajdata_data_dirs,
            future_sec=cfg.future_sec, history_sec=cfg.history_sec,
            raster_size=400, px_per_m=2.0,
            rebuild_maps=False, cache_location=cfg.trajdata_cache_location,
        )

    # --- Rollout loop ---
    history_frames = exp_config.algo.history_num_frames
    dt_ratio = 3   # 30 Hz raw → 10 Hz trajdata

    total_rollouts = 0
    total_collisions = 0
    completed_pairs = set()
    skipped_dup = 0

    for idx, (scene_name, target_name, adv_name, raw_frame, tier) in enumerate(all_pair_entries):
        scene_i = scene_name_to_idx.get(scene_name)
        if scene_i is None:
            print(f"  [skip] {scene_name}: not in trajdata scene index")
            continue

        # Convert raw frame (30 Hz) → trajdata frame (10 Hz)
        start_ts = max(raw_frame // dt_ratio, history_frames + 1)

        pair_key = (scene_name, target_name, adv_name, start_ts)
        if pair_key in completed_pairs:
            skipped_dup += 1
            continue
        completed_pairs.add(pair_key)

        try:
            env.reset(scene_indices=[scene_i], start_frame_index=[start_ts])
        except Exception as e:
            print(f"  [{idx+1}] RESET ERROR {scene_name}@{start_ts}: {e}")
            continue

        if env.is_done():
            continue

        agent_names = env.current_agent_names
        if target_name not in agent_names:
            print(f"  [{idx+1}] TGT {target_name} not present at {scene_name}@{start_ts}")
            continue

        target_idx = agent_names.index(target_name)

        env.reset(scene_indices=[scene_i], start_frame_index=[start_ts])

        # Runtime adversary selection via the shared TargetAdvOthersWrapper:
        #  - initial_adv_idx seeds the wrapper with the Tier 1 / Tier 2 pre-pick
        #  - confirm_k=5 means the DT must predict the same new adversary for
        #    5 consecutive decision steps before the wrapper actually switches
        # Same mechanism used by nuScenes / RounD rollouts.
        try:
            initial_adv_idx_val = agent_names.index(adv_name)
        except ValueError:
            initial_adv_idx_val = None   # adv not in scene; wrapper will let DT pick
        wrapper = TargetAdvOthersWrapper(
            decision_tree_model=runtime_dt,
            target_policy=target_policy,
            adv_policy=agents_planner,
            others_predictor=others_predictor,
            target_name=target_name,
            confirm_k=5,
            initial_adv_idx=initial_adv_idx_val,
        )

        try:
            info = safety_critical_guided_rollout_with_custom_target(
                env, wrapper,
                n_step_action=cfg.n_step_action,
                scene_indices=[scene_i],
                horizon=cfg.num_simulation_steps,
                start_frames=[start_ts],
            )
        except Exception as e:
            print(f"  [{idx+1}] ROLLOUT ERROR {scene_name}@{start_ts}: {e}")
            torch.cuda.empty_cache()
            continue

        had_collision = info.get("had_collision", [False])[0]
        total_rollouts += 1
        if had_collision:
            total_collisions += 1

        # Resolve the FINAL adversary from the wrapper's gated state so the
        # ep_key reflects what actually ended up being the active adv after
        # any runtime switches (matches rollout_round.py convention).
        final_adv_idx = wrapper.current_adv_idx
        if final_adv_idx is not None and 0 <= final_adv_idx < len(agent_names):
            final_adv_name = agent_names[final_adv_idx]
        else:
            final_adv_name = adv_name
        switched = final_adv_name != adv_name
        status = "COLLISION" if had_collision else "no collision"
        print(f"  [{idx+1}/{len(all_pair_entries)}] {tier} {scene_name}@ts{start_ts} "
              f"TGT={target_name} ADV0={adv_name} ADV_final={final_adv_name}"
              f"{' [SWITCHED]' if switched else ''} {status}  "
              f"({total_collisions} coll / {total_rollouts} rollouts)")

        ep_key = f"{scene_name}_{start_ts:04d}_tgt{target_name}_adv{final_adv_name}"
        if "buffer" in info and (had_collision or args.save_all):
            for buf in info["buffer"]:
                buf["target_name"] = np.array([target_name])
                buf["adv_name"] = np.array([final_adv_name])
                buf["adv_name_initial"] = np.array([adv_name])
                # Store full per-buffer-slot agent_name so post-hoc viz can
                # resolve TGT / ADV → buffer index (trajdata's `track_id` in
                # the buffer is a sequential 0-based index, not the AD4CHE id).
                buf["agent_name"] = np.array(agent_names, dtype=object)
            save_episode(cfg.experience_hdf5_path, ep_key, info["buffer"][0])

        if args.render and "buffer" in info and (had_collision or args.save_all):
            from tbsim.utils.scene_edit_utils import visualize_guided_rollout
            viz_dir = os.path.join(cfg.results_dir, "viz/")
            for si_buf, scene_buf in zip(info["scene_index"], info["buffer"]):
                if "action_traj_positions" in scene_buf:
                    N = scene_buf["action_traj_positions"].shape[0]
                    adv_idx = wrapper.current_adv_idx
                    for i in range(N):
                        if i != target_idx and i != adv_idx:
                            scene_buf["action_traj_positions"][i] = 0
                try:
                    visualize_guided_rollout(
                        viz_dir, render_rasterizer, si_buf, scene_buf,
                        guidance_config=None, constraint_config=None,
                        fps=10.0, n_step_action=cfg.n_step_action,
                        viz_diffusion_steps=False, first_frame_only=False,
                        sim_num=start_ts, save_every_n_frames=1,
                        draw_mode="action", target_idx=target_idx,
                        agent_names=agent_names,
                    )
                except Exception as e:
                    print(f"     [viz error] {e}")

        torch.cuda.empty_cache()

        if args.stop_after_collisions is not None and total_collisions >= args.stop_after_collisions:
            print(f"\n[stop] reached {args.stop_after_collisions} collisions, stopping early")
            break

    print(f"\nDone. Results in: {cfg.results_dir}")
    print(f"  Rollouts: {total_rollouts}, Collisions: {total_collisions}, "
          f"Skipped duplicates: {skipped_dup}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="AD4CHE cross-dataset rollout")
    add_args(_parser)
    run(_parser.parse_args())
