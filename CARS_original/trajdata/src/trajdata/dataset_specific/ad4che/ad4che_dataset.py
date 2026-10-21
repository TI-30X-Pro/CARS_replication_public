"""AD4CHE (Aerial Dataset for Chinese Highway Environment) adapter for trajdata.

AD4CHE is a drone-based highway dataset with 68 recordings across 11
physical locations (folders 14-24 under the dataset root). Each recording
ships as three CSVs + background/lane PNGs:

  {loc}/DJI_xxxx/{xx}_tracks_pixel.csv      per-frame trajectories (30 Hz)
  {loc}/DJI_xxxx/{xx}_tracksMeta.csv        per-track metadata (class, extent)
  {loc}/DJI_xxxx/{xx}_recordingMeta.csv     recording-level metadata

We downsample 30 Hz → 10 Hz (SUBSAMPLE_FACTOR=3) to match the CARS nuScenes
convention (dt=0.1 s, history=20 frames, future=20 frames).

Known data-quality issue: 55 of 68 recordings ship with a trailing space in
the `yaw_rate` column name. We strip all column names after reading.

Map: AD4CHE has per-recording PNG lane images but no vector map. For the
cross-dataset rollout we only need "something" that the rasterizer can
render — a single 1 km straight lane at y=0 is a reasonable highway stub.
"""
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import AD4CHERecord


# ---- constants ----
AD4CHE_RAW_DT: Final[float] = 1.0 / 30.0          # 30 Hz native
SUBSAMPLE_FACTOR: Final[int] = 3                   # 30 → 10 Hz
AD4CHE_DT: Final[float] = AD4CHE_RAW_DT * SUBSAMPLE_FACTOR   # 0.1 s

# AD4CHE pixel → world conversion: each recording has origin at image center.
SCALE_M_PER_PX: Final[float] = 0.0375
# Minimum pixels per lane mask below which we skip that lane (noise / fragments)
LANE_MIN_PIXELS: Final[int] = 500
# Number of sample points along each lane centerline
LANE_POLYLINE_N_PTS: Final[int] = 32

CLASS_TO_AGENT_TYPE: Final[Dict[str, AgentType]] = {
    "car": AgentType.VEHICLE,
    "truck": AgentType.VEHICLE,
    "bus": AgentType.VEHICLE,
    "van": AgentType.VEHICLE,
}


# ---- helpers ----

def _discover_recordings(data_dir: Path):
    """Yield (loc, dji, rec_dir, stem) for every AD4CHE recording.

    If `data_dir` exists, iterate the filesystem to discover recordings.
    Otherwise fall back to the hardcoded manifest (used for inference-only
    deployments where only the pre-built trajdata cache is present).
    `stem` is the filename stem (e.g. "09" for DJI_0009); for the manifest
    fallback we derive it from the DJI number.
    """
    if data_dir.exists() and data_dir.is_dir():
        for loc_dir in sorted(data_dir.iterdir(), key=lambda p: p.name):
            if not loc_dir.is_dir() or not loc_dir.name.isdigit():
                continue
            for dji_dir in sorted(loc_dir.iterdir(), key=lambda p: p.name):
                if not dji_dir.is_dir() or not dji_dir.name.startswith("DJI_"):
                    continue
                matches = list(dji_dir.glob("*_tracks_pixel.csv"))
                if not matches:
                    continue
                stem = matches[0].name.replace("_tracks_pixel.csv", "")
                yield loc_dir.name, dji_dir.name, dji_dir, stem
        return

    # Fallback: no raw data on disk. Use the hardcoded manifest. The
    # returned rec_dir still points to the (non-existent) expected path
    # so that any code path needing raw files will fail loudly.
    from ._manifest import AD4CHE_RECORDINGS
    print(
        f"[ad4che] data_dir {data_dir} missing; using hardcoded manifest "
        f"({len(AD4CHE_RECORDINGS)} recordings)",
        flush=True,
    )
    for loc, dji in AD4CHE_RECORDINGS:
        rec_dir = data_dir / loc / dji
        # Derive stem from DJI number: "DJI_0009" -> "09"
        # The last 2 digits of the DJI id form the filename prefix.
        digits = "".join(ch for ch in dji if ch.isdigit())
        stem = digits[-2:] if len(digits) >= 2 else digits
        yield loc, dji, rec_dir, stem


def _build_vectormap_from_lanepicture(png_path: Path, map_id: str):
    """Build a trajdata VectorMap from an AD4CHE `*_lanePicture.png`.

    The PNG is a single-channel image where each unique non-zero integer
    encodes a distinct lane. Highway lanes are essentially straight, so
    each lane becomes a RoadLane with:
      - center line: polyline sampled along the lane's x extent,
                     y(x) = mean y of the mask inside that x bucket
      - left / right edge: center ± half_width (from y extent of mask)
    """
    from trajdata.maps.vec_map import VectorMap
    from trajdata.maps.vec_map_elements import Polyline, RoadLane
    from PIL import Image as PImage

    arr = np.array(PImage.open(str(png_path)))
    if arr.ndim != 2:
        # Unexpected, but fall back gracefully
        arr = arr[..., 0]
    H, W = arr.shape

    vector_map = VectorMap(map_id=map_id)

    all_x, all_y = [], []
    lane_ids = [int(v) for v in np.unique(arr) if int(v) != 0]

    for lane_id_int in lane_ids:
        mask = arr == lane_id_int
        if mask.sum() < LANE_MIN_PIXELS:
            continue
        ys_px, xs_px = np.where(mask)

        # Pixel → world meters. Image origin is at image center;
        # image y is flipped relative to world y.
        xs_m = (xs_px.astype(np.float64) - W / 2.0) * SCALE_M_PER_PX
        ys_m = -(ys_px.astype(np.float64) - H / 2.0) * SCALE_M_PER_PX

        x_min = float(xs_m.min())
        x_max = float(xs_m.max())
        if x_max - x_min < 1e-3:
            continue

        n_pts = LANE_POLYLINE_N_PTS
        x_sample = np.linspace(x_min, x_max, n_pts)
        # Mean y per x bucket — handles slight lane curvature
        y_sample = np.zeros(n_pts, dtype=np.float64)
        bin_edges = np.linspace(x_min, x_max, n_pts + 1)
        # Clip rightmost edge so the last bin is inclusive
        bin_edges[-1] += 1e-6
        for i in range(n_pts):
            in_bin = (xs_m >= bin_edges[i]) & (xs_m < bin_edges[i + 1])
            if in_bin.any():
                y_sample[i] = ys_m[in_bin].mean()
            else:
                # Fill with nearest neighbor by interpolation later; for now
                # use the global mean as a fallback
                y_sample[i] = float(ys_m.mean())

        # Lane half-width from overall y extent of the mask
        half_w = float((ys_m.max() - ys_m.min()) / 2.0)
        half_w = max(half_w, 0.5)   # sanity floor

        z = np.zeros(n_pts, dtype=np.float64)
        center = np.stack([x_sample, y_sample, z], axis=1).astype(np.float32)
        left = np.stack([x_sample, y_sample + half_w, z], axis=1).astype(np.float32)
        right = np.stack([x_sample, y_sample - half_w, z], axis=1).astype(np.float32)

        vector_map.add_map_element(
            RoadLane(
                id=f"lane_{lane_id_int}",
                center=Polyline(points=center),
                left_edge=Polyline(points=left),
                right_edge=Polyline(points=right),
            )
        )
        all_x += [x_min, x_max]
        all_y += [float(ys_m.min()), float(ys_m.max())]

    if all_x:
        vector_map.extent = np.array(
            [min(all_x), min(all_y), 0.0, max(all_x), max(all_y), 0.0],
            dtype=np.float32,
        )
    else:
        vector_map.extent = np.array(
            [-100.0, -50.0, 0.0, 100.0, 50.0, 0.0], dtype=np.float32
        )

    return vector_map


def _find_lane_picture(data_dir: Path, loc: str, dji: str) -> Optional[Path]:
    rec_dir = data_dir / loc / dji
    if not rec_dir.is_dir():
        return None
    matches = list(rec_dir.glob("*_lanePicture.png"))
    return matches[0] if matches else None


def _parse_scene_name(scene_name: str):
    """'ad4che_14_DJI_0009' -> ('14', 'DJI_0009'); returns (None,None) on failure."""
    parts = scene_name.split("_", 2)
    if len(parts) != 3 or parts[0] != "ad4che":
        return None, None
    return parts[1], parts[2]


def _load_ad4che_csvs(rec_dir: Path, stem: str):
    """Load tracks_pixel + tracksMeta, strip column whitespace,
    normalize frames to start from 0, subsample 30 → 10 Hz.
    """
    tracks = pd.read_csv(rec_dir / f"{stem}_tracks_pixel.csv")
    tracks.columns = [c.strip() for c in tracks.columns]

    meta = pd.read_csv(rec_dir / f"{stem}_tracksMeta.csv")
    meta.columns = [c.strip() for c in meta.columns]

    min_frame = int(tracks["frame"].min())
    tracks["frame"] = tracks["frame"] - min_frame
    tracks = tracks[tracks["frame"] % SUBSAMPLE_FACTOR == 0].copy()
    tracks["frame"] = tracks["frame"] // SUBSAMPLE_FACTOR
    return tracks, meta, min_frame


# ---- dataset class ----

class AD4CHEDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        data_path = Path(data_dir)

        scene_split_map: Dict[str, str] = {}
        scene_names: List[str] = []
        for loc, dji, _rec_dir, _stem in _discover_recordings(data_path):
            scene_name = f"ad4che_{loc}_{dji}"
            # All 68 recordings serve the cross-dataset inference case.
            scene_split_map[scene_name] = "train"
            scene_names.append(scene_name)

        # Each recording has its OWN location so we can cache a real
        # per-recording VectorMap from its `lanePicture.png`.
        dataset_parts: List[Tuple[str, ...]] = [
            ("train",),
            tuple(scene_names),
        ]

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=AD4CHE_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
            map_locations=tuple(scene_names),
        )

    # ------------------------------------------------------------------

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        data_path = Path(self.metadata.data_dir)
        self.dataset_obj: Dict[str, Dict[str, Any]] = {}

        n = 0
        for loc, dji, rec_dir, stem in _discover_recordings(data_path):
            scene_name = f"ad4che_{loc}_{dji}"
            tracks, meta, min_frame = _load_ad4che_csvs(rec_dir, stem)
            self.dataset_obj[scene_name] = {
                "tracks": tracks,
                "meta": meta,
                "min_frame": min_frame,
                "loc": loc,
                "dji": dji,
            }
            n += 1

        if verbose:
            total = sum(d["tracks"].shape[0] for d in self.dataset_obj.values())
            print(f"  Loaded {n} recordings ({total} rows @ 10 Hz)", flush=True)

    # ------------------------------------------------------------------

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[AD4CHERecord] = []
        scenes_list: List[SceneMetadata] = []

        for idx, (scene_name, data) in enumerate(self.dataset_obj.items()):
            scene_split = self.metadata.scene_split_map[scene_name]
            scene_length = int(data["tracks"]["frame"].max()) + 1

            # Each scene is its own "location" so we can cache per-recording maps.
            all_scenes_list.append(
                AD4CHERecord(scene_name, scene_name, scene_length, scene_split, idx)
            )

            if (
                scene_name in scene_tag
                and scene_split in scene_tag
                and scene_desc_contains is None
            ):
                scenes_list.append(
                    SceneMetadata(
                        env_name=self.metadata.name,
                        name=scene_name,
                        dt=self.metadata.dt,
                        raw_data_idx=idx,
                    )
                )

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[AD4CHERecord] = env_cache.load_env_scenes_list(self.name)
        scenes_list: List[Scene] = []

        for rec in all_scenes_list:
            scene_name, scene_location, scene_length, scene_split, data_idx = rec
            if (
                scene_location in scene_tag
                and scene_split in scene_tag
                and scene_desc_contains is None
            ):
                scenes_list.append(
                    Scene(
                        self.metadata,
                        scene_name,
                        scene_location,
                        scene_split,
                        scene_length,
                        data_idx,
                        None,
                    )
                )
        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, scene_name, _, data_idx = scene_info
        data = self.dataset_obj[scene_name]
        scene_split = self.metadata.scene_split_map[scene_name]
        scene_length = int(data["tracks"]["frame"].max()) + 1
        return Scene(
            self.metadata,
            scene_name,
            scene_name,   # location = scene name (per-recording map)
            scene_split,
            scene_length,
            data_idx,
            None,
        )

    # ------------------------------------------------------------------

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        data = self.dataset_obj[scene.name]
        tracks: pd.DataFrame = data["tracks"].copy()
        meta: pd.DataFrame = data["meta"]

        # tracksMeta.csv column convention:
        #   id, width (= vehicle length in m), height (= vehicle width in m),
        #   initialFrame, finalFrame, class, drivingDirection, ...
        class_map: Dict[int, str] = dict(zip(meta["id"], meta["class"]))
        ext_map: Dict[int, Tuple[float, float]] = dict(
            zip(meta["id"], zip(meta["width"], meta["height"]))
        )

        # World heading from velocity (AD4CHE has no absolute-yaw column;
        # the `angle` column is deviation from lane direction, not world yaw)
        speed = np.hypot(tracks["xVelocity"], tracks["yVelocity"])
        tracks["heading"] = np.where(
            speed > 0.1,
            np.arctan2(tracks["yVelocity"], tracks["xVelocity"]),
            0.0,
        )

        tracks.rename(
            columns={
                "frame": "scene_ts",
                "id": "agent_id",
                "xVelocity": "vx",
                "yVelocity": "vy",
                "xAcceleration": "ax",
                "yAcceleration": "ay",
            },
            inplace=True,
        )
        tracks["z"] = 0.0

        tracks = tracks[
            ["agent_id", "scene_ts", "x", "y", "z", "vx", "vy", "ax", "ay", "heading"]
        ]

        tracks["agent_id"] = tracks["agent_id"].astype(int)
        tracks.set_index(["agent_id", "scene_ts"], inplace=True)
        tracks.sort_index(inplace=True)

        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        to_remove: List[int] = []

        for agent_id, grp in tracks.groupby(level="agent_id"):
            frames = grp.index.get_level_values("scene_ts")
            if len(frames) <= 1:
                to_remove.append(agent_id)
                continue

            start_f = int(frames.min())
            last_f = int(frames.max())

            cls = class_map.get(agent_id, "car")
            agent_type = CLASS_TO_AGENT_TYPE.get(cls, AgentType.VEHICLE)

            length_m, width_m = ext_map.get(agent_id, (4.5, 1.8))

            agent_metadata = AgentMetadata(
                name=str(agent_id),
                agent_type=agent_type,
                first_timestep=start_f,
                last_timestep=last_f,
                extent=FixedExtent(
                    length=float(length_m),
                    width=float(width_m),
                    height=1.5,
                ),
            )
            agent_list.append(agent_metadata)
            for f in frames:
                if f < scene.length_timesteps:
                    agent_presence[f].append(agent_metadata)

        if to_remove:
            tracks.drop(index=to_remove, level="agent_id", inplace=True)

        tracks.reset_index(inplace=True)
        tracks["agent_id"] = tracks["agent_id"].astype(str)
        tracks.set_index(["agent_id", "scene_ts"], inplace=True)

        cache_class.save_agent_data(tracks, cache_path, scene)
        return agent_list, agent_presence

    # ------------------------------------------------------------------

    def cache_map(
        self,
        map_name: str,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """Build a real VectorMap from the recording's `lanePicture.png`.

        `map_name` equals the scene name (e.g. "ad4che_14_DJI_0009"), which
        we parse back into (loc, DJI) to locate the PNG under `data_dir/`.
        Each non-zero lane ID in the PNG becomes a separate RoadLane.
        """
        loc, dji = _parse_scene_name(map_name)
        data_dir = Path(self.metadata.data_dir)

        png_path = _find_lane_picture(data_dir, loc, dji) if loc is not None else None
        map_id = f"{self.name}:{map_name}"

        if png_path is None:
            # Fallback: empty map so the rasterizer doesn't crash on a bad scene.
            from trajdata.maps.vec_map import VectorMap
            print(f"[ad4che] no lanePicture.png for {map_name}, writing empty map",
                  flush=True)
            vector_map = VectorMap(map_id=map_id)
            vector_map.extent = np.array(
                [-100.0, -50.0, 0.0, 100.0, 50.0, 0.0], dtype=np.float32
            )
        else:
            vector_map = _build_vectormap_from_lanepicture(png_path, map_id)

        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        # One map per recording.
        for map_name in self.metadata.map_locations:
            self.cache_map(map_name, cache_path, map_cache_class, map_params)
