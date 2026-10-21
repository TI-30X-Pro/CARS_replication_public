"""Build and evaluate the adversary-classifier dataset.

- Labels: target-adversary pairs per scene (`configs/labels_nusc.json`, copied from
  CARS_original/configs/simulation_config.json: 314 scenes, 679 pairs).
- Split by scene: shuffle the labelled scenes (random.seed(42)) and move them to the test set
  until it holds 20 % of the pairs. Only scenes of the simulation index (train + train_val)
  count.
- One sample per non-target agent and pair, from the observation at the rollout start
  (step 21, the simulator's agents after its off-road filter). Label 1 for the labelled
  adversary, 0 for the rest.
"""
import json
import random
from dataclasses import dataclass

import numpy as np

from cars.selection.features import features

START = 21


def load_pairs(path):
    """{scene: [[target, adversary], ...]} in file order."""
    with open(path) as f:
        return json.load(f)


def split_scenes(pairs, scene_index, test_size=0.2, seed=42):
    """Return deterministic train and test scene lists, split by labelled-pair count."""
    random.seed(seed)
    counts = [(s, len(p)) for s, p in pairs.items() if s in scene_index]
    budget = int(sum(c for _, c in counts) * test_size)
    random.shuffle(counts)
    train, test, n_test = [], [], 0
    for s, c in counts:
        if n_test < budget:
            test.append(s)
            n_test += c
        else:
            train.append(s)
    return train, test


@dataclass
class Samples:
    X: np.ndarray            # (n, 13) float32
    y: np.ndarray            # (n,) 0/1
    pair: np.ndarray         # (n,) which pair (row of `pairs`) the sample belongs to
    agent: np.ndarray        # (n,) agent index in the observation
    pairs: list              # (scene, target name, adversary name, target index, adversary index)


def build(scenes, pairs, observe, feature_fn=features):
    """Samples for the labelled pairs of `scenes`. observe(scene) -> observation at START.
    Pairs whose target or adversary is absent from the observation are skipped."""
    X, y, pair_of, agent_of, kept = [], [], [], [], []
    for scene in scenes:
        if scene not in pairs:
            continue
        obs = observe(scene)
        names = list(obs["agent_name"])
        for target, adv in pairs[scene]:
            if target not in names or adv not in names:
                continue
            t, a = names.index(target), names.index(adv)
            f = feature_fn(obs, t)
            for i in range(len(names)):
                if i == t:
                    continue
                X.append(f[i])
                y.append(int(i == a))
                pair_of.append(len(kept))
                agent_of.append(i)
            kept.append((scene, target, adv, t, a))
    return Samples(np.array(X, dtype=np.float32).reshape(-1, 13), np.array(y, dtype=np.int64),
                   np.array(pair_of, dtype=np.int64), np.array(agent_of, dtype=np.int64), kept)


def evaluate(trees, data):
    """Table S4: per-class precision / recall / F1 at threshold 0.5, and the rank of the
    labelled adversary among the non-target agents of its pair (top-1 / 3 / 5)."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    p = trees.predict_proba(data.X)
    pred = (p > 0.5).astype(int)
    prec, rec, f1, n = precision_recall_fscore_support(data.y, pred, labels=[0, 1], zero_division=0)
    out = {"n_samples": int(len(data.y)), "n_pairs": len(data.pairs),
           "n_scenes": len({s for s, *_ in data.pairs}),
           "accuracy": float((pred == data.y).mean()),
           "confusion_matrix": confusion_matrix(data.y, pred, labels=[0, 1]).tolist()}
    for c, name in ((1, "adversary"), (0, "non_adversary")):
        out[name] = {"precision": float(prec[c]), "recall": float(rec[c]), "f1": float(f1[c]), "n": int(n[c])}

    ranks, top1 = [], []
    for j, (_, _, _, t, a) in enumerate(data.pairs):
        rows = data.pair == j
        order = list(data.agent[rows][np.argsort(p[rows], kind="stable")[::-1]])
        ranks.append(order.index(a) + 1)
        top1.append(order[0])
    ranks = np.array(ranks)
    for k in (1, 3, 5):
        out[f"top{k}"] = float((ranks <= k).mean())
    out["ranks"] = ranks.tolist()
    out["top1_agent"] = [int(i) for i in top1]
    return out
