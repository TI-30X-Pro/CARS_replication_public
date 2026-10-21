"""Build the selection dataset, train a classifier, and report selection metrics.

    python -m cars.run.train_selector --nusc $NUSC --out $WORK/selector \
        [--index nusc_index.npz] [--rasters rasters.npz]

Writes `trees.npz`, `samples.npz`, and `metrics.json` for the trained and released classifiers.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from cars.data.maps import LOCATIONS, load_nusc_map, load_rasters, rasterize
from cars.data.nuscenes import NuscIndex, build_index
from cars.data.tracks import scene_splits, scene_tracks
from cars.selection import dataset, model
from cars.sim.rollout import trajdata_scene_index
from cars.sim.scene import SimScene

REPO = Path(__file__).resolve().parents[2]


class Observer:
    """Observation of a scene at the rollout start, from our simulator."""

    def __init__(self, nusc_root, index=None, rasters=None):
        self.idx = NuscIndex.load(index) if index and Path(index).exists() else build_index(nusc_root)
        if index and not Path(index).exists():
            self.idx.save(index)
        if rasters and Path(rasters).exists():
            self.rasters = load_rasters(rasters)
        else:
            self.rasters = {loc: rasterize(load_nusc_map(nusc_root, loc), 1.0, loc) for loc in LOCATIONS}
        self.names = [str(n) for n in self.idx.scene_name]
        split_of = {n: k for k, names in scene_splits().items() for n in names}
        self.scene_index = trajdata_scene_index(self.names, [str(x) for x in self.idx.scene_location], split_of)

    def __call__(self, scene, start=dataset.START):
        tr = scene_tracks(self.idx, self.names.index(scene))
        return SimScene(tr, self.rasters[tr.location], start=start).observe(get_map=False)


def save_samples(path, **parts):
    arrays = {}
    for name, d in parts.items():
        arrays.update({f"{name}/X": d.X, f"{name}/y": d.y, f"{name}/pair": d.pair, f"{name}/agent": d.agent,
                       f"{name}/pairs": np.array([[s, t, a, str(ti), str(ai)] for s, t, a, ti, ai in d.pairs])})
    np.savez_compressed(path, **arrays)


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate the adversary selector.")
    ap.add_argument("--nusc", required=True, help="unpacked nuScenes root")
    ap.add_argument("--out", required=True, help="directory for model, samples, and metrics")
    ap.add_argument("--labels", default=str(REPO / "configs" / "labels_nusc.json"), help="label JSON")
    ap.add_argument("--index", help="NuscIndex .npz (built and saved here if missing)")
    ap.add_argument("--rasters", help="map rasters .npz from cars.data.maps")
    ap.add_argument("--shipped", default=str(model.SHIPPED), help="released pickled classifier")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    obs = Observer(a.nusc, a.index, a.rasters)
    pairs = dataset.load_pairs(a.labels)
    train, test = dataset.split_scenes(pairs, obs.scene_index)
    d_train = dataset.build(train, pairs, obs)
    d_test = dataset.build(test, pairs, obs)
    save_samples(os.path.join(a.out, "samples.npz"), train=d_train, test=d_test)
    print(f"train: {len(train)} scenes, {len(d_train.pairs)} pairs, {len(d_train.y)} samples ({d_train.y.sum()} adv)")
    print(f"test:  {len(test)} scenes, {len(d_test.pairs)} pairs, {len(d_test.y)} samples ({d_test.y.sum()} adv)")

    trees = model.fit(d_train.X, d_train.y)
    trees.save(os.path.join(a.out, "trees.npz"))
    metrics = {"ours": dataset.evaluate(trees, d_test),
               "shipped": dataset.evaluate(model.Trees.from_pickle(a.shipped), d_test)}
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)

    for name, m in metrics.items():
        adv = m["adversary"]
        print(f"{name:8s} adversary P/R/F1 {adv['precision']:.2f} / {adv['recall']:.2f} / {adv['f1']:.2f}   "
              f"top-1/3/5 {100 * m['top1']:.1f} / {100 * m['top3']:.1f} / {100 * m['top5']:.1f} %")


if __name__ == "__main__":
    main()
