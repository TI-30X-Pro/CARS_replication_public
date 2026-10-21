"""Stream nuScenes v1.0 tables into a compact scene and agent index.

sample_data.json (1.3 GB), ego_pose.json and sample_annotation.json are too big to load
whole on a laptop, so they are streamed record by record and only what CARS needs is kept:
scenes, key frames, the CAM_FRONT ego pose of every key frame, and the vehicle / human
annotations. The result fits in a small .npz (see `build_index` / `NuscIndex.load`).
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KEEP_CATEGORIES = ("vehicle", "human")   # same filter as trajdata's agent_iterator


def iter_records(path):
    """Yield the records of a nuScenes table one at a time.

    The official files are pretty-printed with one key per line and no nested objects,
    so a record runs from a line "{" to a line "}" or "},".
    """
    buf = []
    n = 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not buf:
                if s == "{":
                    buf = [line]
                continue
            buf.append(line)
            if s in ("}", "},"):
                yield json.loads("".join(buf).rstrip().rstrip(","))
                buf = []
                n += 1
    if n == 0:  # not pretty-printed: fall back to a plain load
        with open(path) as f:
            yield from json.load(f)


@dataclass
class NuscIndex:
    """The parts of nuScenes v1.0 that CARS uses, as flat arrays.

    Scenes keep the order of scene.json and annotations the order of
    sample_annotation.json; both orders matter for matching trajdata.
    """
    scene_name: np.ndarray        # (S,) str
    scene_location: np.ndarray    # (S,) str, e.g. "boston-seaport"
    scene_first: np.ndarray       # (S,) index of the first sample
    sample_token: np.ndarray      # (K,) str
    sample_next: np.ndarray       # (K,) index of the next sample, -1 at the end
    sample_time: np.ndarray       # (K,) int64 timestamp [us]
    ego_xyz: np.ndarray           # (K, 3) CAM_FRONT ego pose of the key frame
    ego_rot: np.ndarray           # (K, 4) quaternion w, x, y, z
    ann_sample: np.ndarray        # (A,) sample index
    ann_instance: np.ndarray      # (A,) instance token
    ann_category: np.ndarray      # (A,) category name
    ann_xyz: np.ndarray           # (A, 3)
    ann_size: np.ndarray          # (A, 3) width, length, height
    ann_rot: np.ndarray           # (A, 4)
    ann_next: np.ndarray          # (A,) index of the next annotation of the instance, -1 if none
    ann_prev: np.ndarray          # (A,)

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(**{k: z[k] for k in z.files})

    def scene_samples(self, s):
        """Sample indices of scene s, in time order."""
        out = []
        k = int(self.scene_first[s])
        while k >= 0:
            out.append(k)
            k = int(self.sample_next[k])
        return out

    def sample_annotations(self):
        """For every sample, the indices of its annotations in file order."""
        order = np.argsort(self.ann_sample, kind="stable")
        bounds = np.searchsorted(self.ann_sample[order], np.arange(len(self.sample_token) + 1))
        return [order[bounds[k]:bounds[k + 1]] for k in range(len(self.sample_token))]


def build_index(root, version="v1.0-trainval", scenes=None):
    """Stream the nuScenes tables under root/version into a NuscIndex.

    scenes: optional list of scene names to keep (all if None).
    """
    d = Path(root) / version
    load = lambda name: list(iter_records(d / f"{name}.json"))  # noqa: E731

    logs = {r["token"]: r["location"] for r in load("log")}
    scene_recs = [r for r in load("scene") if scenes is None or r["name"] in scenes]
    keep_scene = {r["token"] for r in scene_recs}

    samples = [r for r in load("sample") if r["scene_token"] in keep_scene]
    s_idx = {r["token"]: i for i, r in enumerate(samples)}

    channel = {}
    sensors = {r["token"]: r["channel"] for r in load("sensor")}
    for r in load("calibrated_sensor"):
        channel[r["token"]] = sensors[r["sensor_token"]]

    # CAM_FRONT key frame of every sample -> its ego pose
    pose_of_sample = {}
    for r in iter_records(d / "sample_data.json"):
        if r["is_key_frame"] and r["sample_token"] in s_idx \
                and channel[r["calibrated_sensor_token"]] == "CAM_FRONT":
            pose_of_sample[r["sample_token"]] = r["ego_pose_token"]
    wanted = set(pose_of_sample.values())
    poses = {r["token"]: r for r in iter_records(d / "ego_pose.json") if r["token"] in wanted}

    cats = {r["token"]: r["name"] for r in load("category")}
    inst_cat = {r["token"]: cats[r["category_token"]] for r in load("instance")}

    anns = []   # only the fields we need, to keep memory down on the full dataset
    for r in iter_records(d / "sample_annotation.json"):
        if r["sample_token"] not in s_idx:
            continue
        cat = inst_cat[r["instance_token"]]
        if cat.startswith(KEEP_CATEGORIES):
            anns.append((r["token"], r["sample_token"], r["instance_token"], cat, r["translation"],
                         r["size"], r["rotation"], r["next"], r["prev"]))
    a_idx = {a[0]: i for i, a in enumerate(anns)}

    ego = [poses[pose_of_sample[r["token"]]] for r in samples]
    return NuscIndex(
        scene_name=np.array([r["name"] for r in scene_recs]),
        scene_location=np.array([logs[r["log_token"]] for r in scene_recs]),
        scene_first=np.array([s_idx[r["first_sample_token"]] for r in scene_recs]),
        sample_token=np.array([r["token"] for r in samples]),
        sample_next=np.array([s_idx.get(r["next"], -1) for r in samples]),
        sample_time=np.array([r["timestamp"] for r in samples], dtype=np.int64),
        ego_xyz=np.array([p["translation"] for p in ego], dtype=np.float64),
        ego_rot=np.array([p["rotation"] for p in ego], dtype=np.float64),
        ann_sample=np.array([s_idx[a[1]] for a in anns], dtype=np.int64),
        ann_instance=np.array([a[2] for a in anns]),
        ann_category=np.array([a[3] for a in anns]),
        ann_xyz=np.array([a[4] for a in anns], dtype=np.float64).reshape(-1, 3),
        ann_size=np.array([a[5] for a in anns], dtype=np.float64).reshape(-1, 3),
        ann_rot=np.array([a[6] for a in anns], dtype=np.float64).reshape(-1, 4),
        ann_next=np.array([a_idx.get(a[7], -1) for a in anns], dtype=np.int64),
        ann_prev=np.array([a_idx.get(a[8], -1) for a in anns], dtype=np.int64),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stream nuScenes into a compact index file.")
    ap.add_argument("--root", required=True, help="nuScenes root (holds v1.0-trainval/)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()
    index = build_index(args.root, scenes=args.scenes)
    index.save(args.out)
    print(f"{len(index.scene_name)} scenes, {len(index.sample_token)} samples, "
          f"{len(index.ann_sample)} annotations -> {args.out}")
