"""Train the nuScenes adversary-selection decision tree."""
import argparse
import numpy as np
import json
import random
import importlib
import os
import pickle
from pathlib import Path
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env
from tbsim.utils.safety_critical_batch_utils import set_global_trajdata_batch_raster_cfg
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.evaluation.env_builders import EnvCLDBuilder
from tbsim.utils.geometry_utils import transform_points_np
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier

def load_and_split_data(target_adv_pairs, scene_name_to_idx, test_size=0.2, random_state=42):
    """Train/test split by scene, balancing on number of (target, adv) pairs."""
    random.seed(random_state)

    valid_pairs = {k: v for k, v in target_adv_pairs.items() if k in scene_name_to_idx}

    scene_pair_counts = [(scene, len(pairs)) for scene, pairs in valid_pairs.items()]
    total_pairs = sum(count for _, count in scene_pair_counts)
    target_test_pairs = int(total_pairs * test_size)

    random.shuffle(scene_pair_counts)

    # Greedy fill: send scenes to test until the pair budget is met.
    test_scene_names = []
    train_scene_names = []
    test_pairs_count = 0

    for scene, count in scene_pair_counts:
        if test_pairs_count < target_test_pairs:
            test_scene_names.append(scene)
            test_pairs_count += count
        else:
            train_scene_names.append(scene)

    train_pairs_count = total_pairs - test_pairs_count

    train_scene_indices = [scene_name_to_idx[name] for name in train_scene_names]
    test_scene_indices = [scene_name_to_idx[name] for name in test_scene_names]
    
    print("="*60)
    print(f"Data split by target-adv pairs")
    print("="*60)
    print(f"  Total scenes: {len(valid_pairs)}")
    print(f"  Total pairs:  {total_pairs}")
    print(f"  Train: {len(train_scene_indices)} scenes, {train_pairs_count} pairs ({train_pairs_count/total_pairs*100:.1f}%)")
    print(f"  Test:  {len(test_scene_indices)} scenes, {test_pairs_count} pairs ({test_pairs_count/total_pairs*100:.1f}%)")
    
    return train_scene_indices, test_scene_indices


def extract_features(agents_obs, target_idx):
    centroid_ego = agents_obs.get('centroid')[target_idx]          # (2,)
    speed_ego = agents_obs.get('curr_speed')[target_idx]           # scalar
    yaw_ego = agents_obs.get('yaw')[target_idx]                    # scalar
    history_ego = agents_obs.get('history_positions')[target_idx:target_idx+1]  # (1,T,2)
    world_from_agent_ego = agents_obs.get('world_from_agent')[target_idx:target_idx+1]  # (1,3,3)

    centroid_agents = agents_obs.get('centroid')                # (N,2)
    speed_agents = agents_obs.get('curr_speed')                 # (N,)
    yaw_agents = agents_obs.get('yaw')                          # (N,)
    history_agents = agents_obs.get('history_positions')        # (N,T,2)
    world_from_agent_agents = agents_obs.get('world_from_agent')# (N,3,3)

    # --- Basic relative quantities ---
    rel_pos = centroid_agents - centroid_ego
    rel_dist = np.linalg.norm(rel_pos, axis=-1)
    rel_speed = speed_agents - speed_ego

    raw = yaw_agents - yaw_ego
    yaw_diff = np.arctan2(np.sin(raw), np.cos(raw))
    cos_yaw_diff = np.cos(yaw_diff)

    # --- Mirror the reward function's geometry: s / l / closing ---
    history_ego_world = transform_points_np(history_ego, world_from_agent_ego)
    history_agents_world = transform_points_np(history_agents, world_from_agent_agents)

    dt = 0.1
    v_ego = (history_ego_world[0, -1, :] - history_ego_world[0, -2, :]) / dt
    v_agents = (history_agents_world[:, -1, :] - history_agents_world[:, -2, :]) / dt

    # Unit vector along ego heading, plus its perpendicular.
    u = v_ego / (np.linalg.norm(v_ego) + 1e-6)
    u_perp = np.array([-u[1], u[0]], dtype=np.float32)

    # Use world-frame last-frame position differences for consistency with the
    # reward function (avoids agent-centric distortion under yaw changes).
    dp = history_agents_world[:, -1, :] - history_ego_world[0, -1, :]

    s = np.sum(dp * u[None, :], axis=1)                                             # (N,)
    l = np.sum(dp * u_perp[None, :], axis=1)                                        # (N,)
    closing = np.sum((v_ego[None, :] - v_agents) * u[None, :], axis=1)              # (N,)

    # --- Bumper gap and TTC using real extents ---
    extent_agents = agents_obs.get('extent')               # (N, 3) - [L, W, H]
    ego_L = extent_agents[target_idx, 0]
    adv_L = extent_agents[:, 0]
    bumper_margin = 1.0
    d_bumper = 0.5 * (ego_L + adv_L) + bumper_margin

    gap = s - d_bumper

    # Tree splits learn non-linear cuts directly from |l|.
    abs_l = np.abs(l)

    # ---------- TTC (time-to-collision) ----------
    ttc = np.where(closing > 0.1, gap / (closing + 1e-6), 999.0)
    ttc = np.clip(ttc, 0, 999.0)

    # --- Multi-scale history features ---
    T_hist = history_agents_world.shape[1]

    # Per-step agent-to-target distance trajectory: (N, T).
    hist_rel_pos = history_agents_world - history_ego_world
    hist_dist = np.linalg.norm(hist_rel_pos, axis=-1)

    # Distance deltas over 5 / 10 / full history. Negative => approaching.
    dist_delta_5 = hist_dist[:, -1] - hist_dist[:, max(-6, -T_hist)]
    dist_delta_10 = hist_dist[:, -1] - hist_dist[:, max(-11, -T_hist)]
    dist_delta_all = hist_dist[:, -1] - hist_dist[:, 0]

    # Normalised rate of change (m/s).
    dist_rate_5 = dist_delta_5 / (5 * dt + 1e-6)
    dist_rate_10 = dist_delta_10 / (min(10, T_hist - 1) * dt + 1e-6)

    # Closest historical distance to target.
    min_hist_dist = np.min(hist_dist, axis=-1)

    # Historical speeds (per-step magnitude of position delta).
    hist_speed_agents = np.linalg.norm(
        history_agents_world[:, 1:, :] - history_agents_world[:, :-1, :], axis=-1
    ) / dt
    hist_speed_ego = np.linalg.norm(
        history_ego_world[0, 1:, :] - history_ego_world[0, :-1, :], axis=-1
    ) / dt

    # Acceleration over the last ~5 steps.
    accel_agents = (hist_speed_agents[:, -1] - hist_speed_agents[:, max(-6, -(T_hist-1))]) / (min(5, T_hist - 2) * dt + 1e-6)
    accel_ego = (hist_speed_ego[-1] - hist_speed_ego[max(-6, -(T_hist-1))]) / (min(5, T_hist - 2) * dt + 1e-6)
    rel_accel = accel_agents - accel_ego

    # --- Longitudinal-distance history (in target frame) ---
    hist_s = np.sum(hist_rel_pos * u[None, None, :], axis=-1)
    hist_l = np.sum(hist_rel_pos * u_perp[None, None, :], axis=-1)

    s_delta_5 = hist_s[:, -1] - hist_s[:, max(-6, -T_hist)]
    l_delta_5 = hist_l[:, -1] - hist_l[:, max(-6, -T_hist)]

    # Is |l| shrinking (i.e. cutting in)? Negative => cutting in.
    abs_l_delta_5 = np.abs(hist_l[:, -1]) - np.abs(hist_l[:, max(-6, -T_hist)])

    # --- Heading / bearing features ---
    bearing_to_target = np.arctan2(
        centroid_ego[1] - centroid_agents[:, 1],
        centroid_ego[0] - centroid_agents[:, 0]
    )
    bearing_diff = np.arctan2(
        np.sin(bearing_to_target - yaw_agents),
        np.cos(bearing_to_target - yaw_agents)
    )  # 0 => agent points directly at target
    cos_bearing = np.cos(bearing_diff)  # 1 => facing target, -1 => facing away

    lateral_closing = np.sum((v_ego[None, :] - v_agents) * u_perp[None, :], axis=1)

    is_behind = (s < 0).astype(np.float32)

    # --- Combined threat features ---
    # The tree learns closing × distance × lateral interactions on its own;
    # no manual gates needed.

    # --- Multi-frame longitudinal-s trajectory features ---
    min_hist_s = np.min(np.abs(hist_s), axis=-1)  # Closest historical |s|.
    std_hist_s = np.std(hist_s, axis=-1)          # Trajectory stability proxy.

    # --- Multi-frame lateral-l trajectory features ---
    min_hist_abs_l = np.min(np.abs(hist_l), axis=-1)
    std_hist_l = np.std(hist_l, axis=-1)          # Lateral motion amplitude.

    # Current distance vs. historical minimum (< 1 => currently the closest seen).
    dist_vs_min = rel_dist / (min_hist_dist + 1e-6)

    # Fraction of history where the agent was in front of the target.
    time_in_front = np.mean((hist_s > 0).astype(np.float32), axis=-1)

    # Speed-match: closer speeds make a collision harder to avoid.
    speed_ratio = speed_agents / (speed_ego + 1e-6)  # > 1 => agent faster

    # --- Predicted-collision proxy: 2-second linear extrapolation ---
    future_pos_agents = history_agents_world[:, -1, :] + v_agents * 2.0
    future_pos_ego = history_ego_world[0, -1, :] + v_ego * 2.0
    future_dist = np.linalg.norm(future_pos_agents - future_pos_ego[None, :], axis=-1)
    dist_change_2s = future_dist - rel_dist  # Negative => closer in 2 s.

    features = np.stack([
        # --- Spatial (4) ---
        rel_dist,           # Euclidean distance
        s,                  # longitudinal offset
        l,                  # lateral offset
        gap,                # bumper gap
        # --- Dynamic (6) ---
        closing,            # longitudinal closing speed
        lateral_closing,    # lateral closing speed
        rel_speed,          # relative speed
        speed_agents,       # agent speed (absolute)
        cos_yaw_diff,       # heading alignment
        ttc,                # time-to-collision
        # --- Bearing (1) ---
        cos_bearing,        # agent-bearing-to-target cosine
        # --- History trend (2) ---
        dist_rate_5,        # 5-step distance rate-of-change
        abs_l_delta_5,      # 5-step lateral cut-in trend
    ], axis=-1)

    feature_names = [
        'rel_dist',
        's',
        'l',
        'gap',
        'closing',
        'lateral_closing',
        'rel_speed',
        'curr_speed',
        'cos_yaw_diff',
        'ttc',
        'cos_bearing',
        'dist_rate_5',
        'abs_l_delta_5',
    ]

    return features, feature_names







def build_dataset(env, scenes, target_adv_pairs, exp_config):

    X_list, y_list = [], []
    feature_names = None
    start_frame = exp_config.algo.history_num_frames + 1
    
    for scene_idx in scenes:
        env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame])
        obs = env.get_observation(split_ego=False)
        
        scene_name = env.current_scene_names[0]
        
        if scene_name not in target_adv_pairs:
            continue
        
        agents_obs = obs['agents']
        agent_names = list(agents_obs['agent_name'])
        
        for pair in target_adv_pairs[scene_name]:
            target_name, adv_name = pair[0], pair[1]

            if target_name not in agent_names:
                continue
            target_idx = agent_names.index(target_name)

            if adv_name not in agent_names:
                continue
            adv_idx = agent_names.index(adv_name)

            features, feat_names = extract_features(agents_obs, target_idx)

            if feature_names is None:
                feature_names = feat_names

            # Emit one sample per agent (excluding the target itself);
            # label = 1 for the labelled adv, 0 otherwise.
            for i, feat in enumerate(features):
                if i == target_idx:
                    continue
                X_list.append(feat)
                y_list.append(1 if i == adv_idx else 0)
    
    X, y = np.array(X_list), np.array(y_list)
    print(f"Dataset: {len(X)} samples ({X.shape[1]} features), Adv={np.sum(y==1)}, Non-Adv={np.sum(y==0)}")
    return X, y, feature_names


def train_decision_tree(X_train, y_train):

    print("\n" + "="*80)
    print("Training decision tree")
    print("="*80)

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    w = np.ones_like(y_train, dtype=float)
    w[y_train == 1] = np.sqrt(neg / pos)

    model = HistGradientBoostingClassifier(
        max_depth=5, max_iter=800, learning_rate=0.03,
        min_samples_leaf=20, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"Train accuracy: {train_acc:.4f}")
    
    return model


def evaluate_model(model, X_test, y_test, feature_names=None):
    """Standard classification evaluation"""
    print("\n" + "="*80)
    print("Standard classification evaluation")
    print("="*80)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Adv', 'Adversary']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(cm)
    
    feature_importance_dict = {}
    if feature_names:
        print("\nFeature importances:")
        if hasattr(model, 'feature_importances_'):
            # HistGradientBoostingClassifier, RandomForestClassifier, DecisionTreeClassifier
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for i in range(len(indices)):
                idx = indices[i]
                print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
                feature_importance_dict[feature_names[idx]] = float(importances[idx])
        else:
            print("  (Model does not have feature_importances_ attribute)")
    
    return {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importances': feature_importance_dict
    }


def sample_adversary_from_top_k(proba, k=3, temperature=0.5, min_prob=0.05):
    """Sample an adversary from the Top-K candidates with temperature shaping.

    Args:
        proba:       per-agent adversary probability array.
        k:           number of top candidates to keep.
        temperature: > 1 broadens the distribution, < 1 sharpens it.
        min_prob:    floor probability so under-represented candidates can
                     still be reached (exploration).

    Returns:
        sampled_idx:               sampled agent index (global).
        top_k_indices:             original indices of the Top-K candidates.
        top_k_proba_normalized:    final sampling distribution over those K.
    """
    sorted_indices = np.argsort(proba)[::-1]
    top_k_indices = sorted_indices[:min(k, len(proba))]
    top_k_proba = proba[top_k_indices].copy()

    if temperature != 1.0:
        top_k_proba = np.power(top_k_proba + 1e-8, 1.0 / temperature)

    top_k_proba = np.maximum(top_k_proba, min_prob)
    top_k_proba_normalized = top_k_proba / top_k_proba.sum()

    sampled_idx = np.random.choice(top_k_indices, p=top_k_proba_normalized)

    return sampled_idx, top_k_indices, top_k_proba_normalized


def evaluate_model_per_scene(model, env, test_scenes, ego_adv_pairs, exp_config, feature_names=None):
    """Argmax evaluation, scene by scene.

    Args:
        model:         trained decision-tree classifier.
        env:           simulation environment.
        test_scenes:   indices of scenes to evaluate.
        ego_adv_pairs: ``{scene_name: [[ego_name, adv_name], ...]}``.
        exp_config:    experiment config.
        feature_names: optional feature-name list for reporting.
    """
    print("\n" + "="*80)
    print("Per-Scene Argmax Evaluation")
    print("="*80)
    
    start_frame_index = exp_config.algo.history_num_frames + 1
    
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_scenes = 0
    
    predictions = []  
    
    print(f"\nPredicting {len(test_scenes)} test scenes...\n")
    
    for idx, scene_idx in enumerate(test_scenes):
        try:
            env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame_index])
            obs = env.get_observation(split_ego=False)
            
            scene_name = env.current_scene_names[0]
            if scene_name not in ego_adv_pairs:
                continue
            
            agents_obs = obs['agents']
            agent_names = list(agents_obs['agent_name'])
            
            for pair in ego_adv_pairs[scene_name]:
                target_name = pair[0]
                adv_name = pair[1]

                if target_name not in agent_names or adv_name not in agent_names:
                    continue

                target_idx = agent_names.index(target_name)
                adv_idx = agent_names.index(adv_name)

                total_scenes += 1

                features, _ = extract_features(agents_obs, target_idx)
                if features is None:
                    print(f"  Warning: scene_idx={scene_idx} feature extraction failed")
                    continue

                num_agents = features.shape[0]

                proba = model.predict_proba(features)[:, 1]

                # Rank only non-target agents (the target can never be its own adv).
                candidates = np.arange(len(proba))
                candidates = candidates[candidates != target_idx]
                sorted_candidates = candidates[np.argsort(proba[candidates])[::-1]]

                pred_adv_idx = sorted_candidates[0]
                num_candidates = len(sorted_candidates)
                
                is_top1_correct = (pred_adv_idx == adv_idx)
                if is_top1_correct:
                    top1_correct += 1
                
                top3_set = set(sorted_candidates[:min(3, num_candidates)].tolist())
                is_top3_correct = adv_idx in top3_set
                if is_top3_correct:
                    top3_correct += 1
                
                top5_set = set(sorted_candidates[:min(5, num_candidates)].tolist())
                is_top5_correct = adv_idx in top5_set
                if is_top5_correct:
                    top5_correct += 1
                
                status = "✓" if is_top1_correct else "✗"
                print(f"  {status} {scene_name} (idx={scene_idx}): target={target_name[:8]}..., true_adv={adv_idx}, pred={pred_adv_idx}, "
                      f"Top3={sorted_candidates[:min(3, num_candidates)].tolist()}, "
                      f"proba={proba[pred_adv_idx]:.3f}")
                
                predictions.append({
                    'scene_name': scene_name,
                    'scene_index': int(scene_idx),
                    'target_name': target_name,
                    'true_adv_idx': int(adv_idx),
                    'pred_adv_idx': int(pred_adv_idx),
                    'top3_pred': sorted_candidates[:min(3, num_candidates)].tolist(),
                    'top5_pred': sorted_candidates[:min(5, num_candidates)].tolist(),
                    'probabilities': proba.tolist(),
                    'correct_top1': bool(is_top1_correct),
                    'correct_top3': bool(is_top3_correct),
                    'correct_top5': bool(is_top5_correct)
                })
            
        except Exception as e:
            print(f"  Error: scene={scene_idx} failed: {e}")
            continue
    
    top1_acc = top1_correct / total_scenes if total_scenes > 0 else 0
    top3_acc = top3_correct / total_scenes if total_scenes > 0 else 0
    top5_acc = top5_correct / total_scenes if total_scenes > 0 else 0
    
    print(f"\n" + "="*80)
    print(f"Scene-level evaluation results:")
    print(f"  Top-1 accuracy: {top1_acc:.4f} ({top1_correct}/{total_scenes})")
    print(f"  Top-3 accuracy: {top3_acc:.4f} ({top3_correct}/{total_scenes})")
    print(f"  Top-5 accuracy: {top5_acc:.4f} ({top5_correct}/{total_scenes})")
    print("="*80)
    
    return {
        'top1_accuracy': float(top1_acc),
        'top3_accuracy': float(top3_acc),
        'top5_accuracy': float(top5_acc),
        'top1_correct': int(top1_correct),
        'top3_correct': int(top3_correct),
        'top5_correct': int(top5_correct),
        'total_scenes': int(total_scenes),
        'predictions': predictions
    }


def evaluate_model_with_sampling(model, env, test_scenes, ego_adv_pairs, exp_config,
                                  k=3, temperature=1.0, min_prob=0.05, num_trials=100):
    """Evaluate the model with Top-K probability sampling.

    Args:
        model:        trained decision-tree classifier.
        env:          simulation environment.
        test_scenes:  scene indices to evaluate.
        ego_adv_pairs:``{scene_name: [[ego_name, adv_name], ...]}``.
        exp_config:   experiment config.
        k:            Top-K candidate count.
        temperature:  > 1 broadens, < 1 sharpens the sampling distribution.
        min_prob:     floor probability for exploration.
        num_trials:   sampling trials per scene for hit-rate estimation.
    """
    print("\n" + "="*80)
    print(f"Per-Scene Top-{k} Sampling Evaluation (temperature={temperature})")
    print("="*80)

    start_frame_index = exp_config.algo.history_num_frames + 1

    total_scenes = 0
    sampling_correct_counts = []  # per-scene number of correct sampling trials.

    print(f"\nEvaluating {len(test_scenes)} test scenes with {num_trials} trials each...\n")

    for scene_idx in test_scenes:
        try:
            env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame_index])
            obs = env.get_observation(split_ego=False)

            scene_name = env.current_scene_names[0]
            if scene_name not in ego_adv_pairs:
                continue

            agents_obs = obs['agents']
            agent_names = list(agents_obs['agent_name'])

            for pair in ego_adv_pairs[scene_name]:
                target_name = pair[0]
                adv_name = pair[1]

                if target_name not in agent_names or adv_name not in agent_names:
                    continue

                target_idx = agent_names.index(target_name)
                adv_idx = agent_names.index(adv_name)

                total_scenes += 1

                features, _ = extract_features(agents_obs, target_idx)
                if features is None:
                    continue

                proba_all = model.predict_proba(features)[:, 1]

                # Restrict to non-target candidates and rebase the probability vector.
                candidates = np.arange(len(proba_all))
                candidates = candidates[candidates != target_idx]
                proba = proba_all[candidates]

                correct_count = 0
                for _ in range(num_trials):
                    sampled_j, topk_js, _ = sample_adversary_from_top_k(
                        proba, k=k, temperature=temperature, min_prob=min_prob
                    )
                    sampled_idx = int(candidates[sampled_j])
                    if sampled_idx == adv_idx:
                        correct_count += 1

                sampling_correct_counts.append(correct_count)

                # Display one extra sampling trial for human inspection.
                sampled_j, topk_js, top_k_proba_norm = sample_adversary_from_top_k(
                    proba, k=k, temperature=temperature, min_prob=min_prob
                )
                sampled_idx = int(candidates[sampled_j])
                top_k_indices = candidates[topk_js]

                in_top_k = adv_idx in top_k_indices.tolist()
                status = "[hit]" if in_top_k else "[miss]"

                print(f"  {status} {scene_name} (idx={scene_idx}): "
                      f"target={target_name[:8]}..., true_adv={adv_idx}, sampled={sampled_idx}, "
                      f"Top-{k}={top_k_indices.tolist()}, "
                      f"proba_norm={[f'{p:.2f}' for p in top_k_proba_norm]}, "
                      f"hit_rate={correct_count}/{num_trials}")

        except Exception as e:
            print(f"  Error: scene={scene_idx} failed: {e}")
            continue

    avg_hit_rate = np.mean(sampling_correct_counts) / num_trials if sampling_correct_counts else 0
    
    print(f"\n" + "="*80)
    print(f"Sampling evaluation results (k={k}, temperature={temperature}):")
    print(f"  Average sampling hit rate: {avg_hit_rate:.4f} ({np.mean(sampling_correct_counts):.1f}/{num_trials} per scene)")
    print(f"  Total scenes evaluated: {total_scenes}")
    print("="*80)
    
    return {
        'k': k,
        'temperature': temperature,
        'min_prob': min_prob,
        'num_trials': num_trials,
        'avg_hit_rate': float(avg_hit_rate),
        'total_scenes': int(total_scenes),
        'per_scene_correct_counts': [int(c) for c in sampling_correct_counts]
    }


def run_scene_editor(eval_cfg):
    """Main workflow for decision tree training"""
    print("\n" + "="*80)

    print("Decision tree training workflow")
    print("="*80)
    
    assert eval_cfg.env in ["nusc", "trajdata"]
    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])
    
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)
    
    print(f'Results will be saved to: {eval_cfg.results_dir}')
    os.makedirs(eval_cfg.results_dir, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    
    ego_composer = composer_class(eval_cfg, device)
    _, exp_config = ego_composer.get_policy()
    
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)
    
    env_builder = EnvCLDBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
    env = env_builder.get_env()
    
    # ego_adv_pairs from the config: {scene_name: [[ego_name, adv_name], ...]}
    target_adv_pairs = eval_cfg.target_adv_pairs
    print(f"Target-Adv pairs loaded: {len(target_adv_pairs)} scenes")

    total_pairs = sum(len(pairs) for pairs in target_adv_pairs.values())
    print(f"  Total target-adv pairs: {total_pairs}")

    # Pull scene names straight from dataset._scene_index — no env.reset needed.
    print("\nBuilding scene_name -> scene_idx mapping from dataset index...")
    scene_name_to_idx = {}
    for idx, scene_path in enumerate(env.dataset._scene_index):
        scene_name = Path(scene_path).parent.name  # e.g. "scene-0301"
        scene_name_to_idx[scene_name] = idx
    print(f"  Mapped {len(scene_name_to_idx)} scenes (no reset required)")
    
    train_scenes, test_scenes = load_and_split_data(
        target_adv_pairs=target_adv_pairs,
        scene_name_to_idx=scene_name_to_idx,
        test_size=0.2,
        random_state=42
    )
    
    print("\n" + "="*80)
    print("Building train dataset")
    print("="*80)
    X_train, y_train, feature_names = build_dataset(env, train_scenes, target_adv_pairs, exp_config)
    
    print("\n" + "="*80)
    print("Building test dataset")
    print("="*80)
    X_test, y_test, _ = build_dataset(env, test_scenes, target_adv_pairs, exp_config)
    
    model = train_decision_tree(X_train, y_train)
    
    results_binary = evaluate_model(model, X_test, y_test, feature_names)
    
    results_scene = evaluate_model_per_scene(model, env, test_scenes, target_adv_pairs, exp_config, feature_names)

    results_sampling = evaluate_model_with_sampling(
        model, env, test_scenes, target_adv_pairs, exp_config,
        k=3, temperature=1.0, min_prob=0.05, num_trials=100
    )
    
    output_dir = Path(eval_cfg.results_dir)
    model_path = output_dir / 'decision_tree_model_v2.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {model_path}")
    
    with open(output_dir / 'evaluation_binary_v2.json', 'w') as f:
        json.dump(results_binary, f, indent=2)
    print(f"Standard evaluation results saved: {output_dir / 'evaluation_binary_v2.json'}")

    with open(output_dir / 'evaluation_per_scene_v2.json', 'w') as f:
        json.dump(results_scene, f, indent=2)
    print(f"Per-scene evaluation results saved: {output_dir / 'evaluation_per_scene_v2.json'}")

    with open(output_dir / 'evaluation_sampling_v2.json', 'w') as f:
        json.dump(results_sampling, f, indent=2)
    print(f"Sampling evaluation results saved: {output_dir / 'evaluation_sampling_v2.json'}")

    if feature_names:
        with open(output_dir / 'feature_names_v2.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    
    return model, results_scene, results_sampling


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to JSON config file",
    )


def run(args) -> None:
    cfg = SceneEditingConfig(registered_name='trajdata_nusc_diff')

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)
    cfg.env = "trajdata"

    cfg.results_dir = os.path.join(cfg.results_dir, cfg.name + "_decision_tree")

    for k in cfg[cfg.env]:
        cfg[k] = cfg[cfg.env][k]

    cfg.pop("nusc")
    cfg.pop("trajdata")

    cfg.lock()
    run_scene_editor(cfg)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Train the nuScenes adversary-selection decision tree")
    add_args(_parser)
    run(_parser.parse_args())


  