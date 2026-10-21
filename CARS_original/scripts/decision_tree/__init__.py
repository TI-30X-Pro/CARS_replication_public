"""Adversary-selection decision-tree training.

Subcommands:
    nusc      Train the nuScenes decision tree.
    round     Train the RounD decision tree.
    ad4che    Train the AD4CHE decision tree.

This module also re-exports the runtime feature extractors so that the
decision-tree wrapper (in CTG) and figure scripts can import them via
``from scripts.decision_tree import extract_features, extract_features_round``.
"""

from .nusc import extract_features
from .round import extract_features_round

__all__ = ["extract_features", "extract_features_round"]
