"""Choose and confirm an adversary online (paper Eq. 3-4).

A selector is called once per re-plan with the observation of all agents and the target's
index, and returns the adversary's index for the next block plus some diagnostics:

    selector.reset()
    adv, info = selector(obs, target)

- `TreeSelector`: classifier scores of all agents, argmax over non-targets (Eq. 3), then the
  confirmation gate (Eq. 4).
- `FixedSelector`: always the same agent, by name (PPO training uses the labelled adversary).
"""
import numpy as np

from cars.selection.features import features


class ConfirmationGate:
    """Keeps the active adversary until another one has been predicted k calls in a row.

    The first prediction is taken as is. A run of a new candidate restarts whenever the
    prediction changes, including back to the active one. Calls are re-plans (0.5 s), so
    k = 5 means 2.5 s."""

    def __init__(self, k=5):
        self.k = k
        self.reset()

    def reset(self, active=None):
        self.active, self.candidate, self.count = active, None, 0

    def __call__(self, pred):
        if self.active is None:
            self.active, self.candidate, self.count = pred, None, 0
        elif pred == self.active:
            self.candidate, self.count = None, 0
        else:
            if pred != self.candidate:
                self.candidate, self.count = pred, 1
            else:
                self.count += 1
            if self.count >= self.k:
                self.active, self.candidate, self.count = self.candidate, None, 0
        return self.active


def rank(proba, target):
    """Rank agents by descending score, placing the target last.

    Equal scores use the higher agent index first for deterministic selection.
    """
    p = np.array(proba, dtype=np.float64)
    p[target] = -1.0
    return np.argsort(p, kind="stable")[::-1]


class TreeSelector:
    def __init__(self, trees, k=5, feature_fn=features):
        self.trees = trees
        self.gate = ConfirmationGate(k)
        self.feature_fn = feature_fn

    def reset(self):
        self.gate.reset()

    def __call__(self, obs, target):
        x = self.feature_fn(obs, target)
        p = self.trees.predict_proba(x)
        order = rank(p, target)
        adv = int(self.gate(int(order[0])))
        return adv, {"pred": int(order[0]), "top5": order[:5], "proba": p,
                     "candidate": self.gate.candidate, "count": self.gate.count}


class FixedSelector:
    def __init__(self, name):
        self.name = name

    def reset(self):
        pass

    def __call__(self, obs, target):
        names = list(obs["agent_name"])
        return (names.index(self.name) if self.name in names else None), {}
