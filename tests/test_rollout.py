"""Tests for rollout configuration and deterministic scene sharding."""
import json
from pathlib import Path

from cars.run.rollout import CONFIG, shard


def test_scenes_and_shards_cover_the_config():
    cfg = json.load(open(CONFIG))
    scenes = cfg["scenes"]
    pairs = json.load(open(Path(CONFIG).parent / "labels_nusc.json"))
    assert len(scenes) == 311 and sum(len(pairs[s]) for s in scenes) == 676
    shards = [shard(scenes, k, 24) for k in range(1, 25)]
    assert sum(shards, []) == scenes
    # Keep the configured ordering stable across runs.
    assert shards[0][:3] == ["scene-0306", "scene-0328", "scene-0388"]
