"""Train and run the histogram-gradient-boosting adversary classifier.

The released classifier is a scikit-learn 1.2.2 pickle that newer versions cannot load
directly. `Trees.from_pickle` extracts its arrays without importing scikit-learn internals.
`Trees` also stores fitted models as portable `.npz` files and predicts as follows:
- a sample goes left if its feature is <= the node threshold, NaN goes where the node says;
- raw score = baseline + leaf values, added tree by tree in float64;
- P(adversary) = sigmoid(raw score).
"""
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expit

SHIPPED = Path(__file__).resolve().parents[2] / "CARS_original" / "checkpoints" / "decision_tree_model.pkl"

PARAMS = dict(max_depth=5, max_iter=800, learning_rate=0.03, min_samples_leaf=20, random_state=42)
NODE_FIELDS = ("value", "feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf")


def sample_weights(y):
    """Adversaries weighted by sqrt(#non-adversaries / #adversaries), the rest by 1."""
    y = np.asarray(y)
    w = np.ones_like(y, dtype=float)
    w[y == 1] = np.sqrt((y == 0).sum() / (y == 1).sum())
    return w


def fit(X, y, **overrides):
    """Fit the released classifier recipe with the installed scikit-learn."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(**{**PARAMS, **overrides})
    model.fit(X, y, sample_weight=sample_weights(y))
    return Trees.from_sklearn(model)


class Trees:
    def __init__(self, baseline, nodes, offsets):
        self.baseline = float(baseline)
        self.nodes = {k: np.asarray(v) for k, v in nodes.items()}   # all trees' nodes, concatenated
        self.offsets = np.asarray(offsets, dtype=np.int64)          # first node of each tree

    @classmethod
    def from_sklearn(cls, model):
        if model.n_trees_per_iteration_ != 1:
            raise ValueError("binary classifier expected")
        preds = [p[0] for p in model._predictors]
        nodes = {k: np.concatenate([p.nodes[k] for p in preds]) for k in NODE_FIELDS}
        offsets = np.cumsum([0] + [len(p.nodes) for p in preds[:-1]])
        if np.any(np.concatenate([p.nodes["is_categorical"] for p in preds])):
            raise ValueError("categorical splits are not supported")
        return cls(np.ravel(model._baseline_prediction)[0], nodes, offsets)

    @classmethod
    def from_pickle(cls, path=SHIPPED):
        """The trees of a pickled HistGradientBoostingClassifier from any scikit-learn version.
        scikit-learn classes are replaced by plain attribute holders while unpickling."""
        with open(path, "rb") as f:
            return cls.from_sklearn(_Unpickler(f).load())

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["baseline"], {k: z[k] for k in NODE_FIELDS}, z["offsets"])

    def save(self, path):
        np.savez_compressed(path, baseline=self.baseline, offsets=self.offsets, **self.nodes)

    @property
    def n_trees(self):
        return len(self.offsets)

    def raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = self.nodes
        rows = np.arange(len(X))
        out = np.full(len(X), self.baseline)
        for start in self.offsets:
            node = np.full(len(X), start)
            while True:
                leaf = n["is_leaf"][node].astype(bool)
                if leaf.all():
                    break
                x = X[rows, n["feature_idx"][node]]
                left = np.where(np.isnan(x), n["missing_go_to_left"][node].astype(bool), x <= n["num_threshold"][node])
                nxt = start + np.where(left, n["left"][node], n["right"][node])
                node = np.where(leaf, node, nxt)
            out += n["value"][node]
        return out

    def predict_proba(self, X):
        """P(adversary) per row."""
        return expit(self.raw(X))


class _Stub(SimpleNamespace):
    def __setstate__(self, state):
        if isinstance(state, tuple):
            state = next((x for x in state if isinstance(x, dict)), {})
        self.__dict__.update(state or {})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "sklearn":
            if name.startswith("__pyx_unpickle"):     # Cython objects (the loss): not needed
                return lambda *args: _Stub()
            return type(name, (_Stub,), {})
        return super().find_class(module, name)
