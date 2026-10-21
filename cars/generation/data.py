"""Build ego-vehicle samples for diffusion-prior training.

- agent-centric, vehicles only, and only the ego vehicle (`ego_only = not incl_robot_future`);
- every time step with a full 2 s of history and 2 s of future (20 steps each way at 10 Hz);
- train = nusc_trainval-train + train_val (109,850 samples), valid = nusc_trainval-val
  (23,495);
- samples in trajdata's order: scenes as in its scene index (split, then Boston before
  Singapore, then scene.json order), time steps ascending.

Elements and batches use `cars.data.batch`.
"""
import torch

from cars.data.batch import EncodingConfig, collate, element, parse
from cars.data.tracks import scene_splits, scene_tracks
from cars.sim.rollout import trajdata_scene_index

SPLITS = {"train": ("train", "train_val"), "valid": ("val",)}


def sample_index(idx, part, history=20, future=20):
    """[(scene name, ts)] of the ego samples of one part, in trajdata's order."""
    names = [str(n) for n in idx.scene_name]
    locations = [str(x) for x in idx.scene_location]
    split_of = {n: k for k, v in scene_splits().items() for n in v}
    order = trajdata_scene_index(names, locations, split_of, wanted=SPLITS[part])
    out = []
    for name in sorted(order, key=order.get):
        s = names.index(name)
        length = 1 + 5 * (len(idx.scene_samples(s)) - 1)       # 2 Hz key frames at 10 Hz
        out += [(name, ts) for ts in range(history, length - future)]
    return out


class PriorDataset(torch.utils.data.Dataset):
    """Item i: the element of sample i (numpy). Use `collate_parsed` as the DataLoader's collate.

    The tracks of all scenes are built up front (700 training scenes: ~30 s, ~150 MB), so
    DataLoader workers forked afterwards share them."""

    def __init__(self, idx, rasters, part="train", cfg=EncodingConfig()):
        self.rasters, self.cfg = rasters, cfg
        self.samples = sample_index(idx, part)
        names = [str(n) for n in idx.scene_name]
        self.tracks = {s: scene_tracks(idx, names.index(s)) for s in dict.fromkeys(s for s, _ in self.samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        name, ts = self.samples[i]
        tr = self.tracks[name]
        return element(tr, "ego", ts, self.rasters[tr.location], self.cfg)

    def collate_parsed(self, elems):
        return parse(collate(elems, self.cfg), self.cfg)
