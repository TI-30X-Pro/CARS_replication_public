"""
RounD (Roundabout Drone Dataset) adapter for trajdata.

RounD is a drone-based naturalistic driving dataset recorded at a roundabout
in Neuweiler, Germany.  Each recording is stored as a pair of CSV files:
  - XX_tracks.csv      (per-frame trajectory data at 25 Hz)
  - XX_tracksMeta.csv  (per-track metadata: class, extent, frame range)

The dataset ships with a single OpenDRIVE map (neuweiler.xodr).
Maps are handled by rasterising the background PNG that comes with each
recording, so no OpenDRIVE parser is needed at this stage.

Reference: Krajewski et al., "The rounD Dataset: A Drone Dataset of Road
User Trajectories at Roundabouts in Germany", IEEE ITSC 2020.
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
from trajdata.dataset_specific.scene_records import RounDRecord
from trajdata.dataset_specific.round.xodr_parser import xodr_to_vector_map
from trajdata.utils import arr_utils

# ---- constants ----
# RounD native frame rate is 25 Hz (0.04s).
# We subsample by 5x to get 5 Hz (0.2s) which is integer-divisible by
# common desired_dt values (0.1 is NOT divisible by 0.04, but 0.2 is).
ROUND_RAW_DT: Final[float] = 0.04  # 25 Hz
SUBSAMPLE_FACTOR: Final[int] = 5   # 25 Hz / 5 = 5 Hz
ROUND_DT: Final[float] = ROUND_RAW_DT * SUBSAMPLE_FACTOR  # 0.2s = 5 Hz
ROUND_LOCATION: Final[str] = "neuweiler"

# Map RounD class strings to trajdata AgentType
CLASS_TO_AGENT_TYPE: Final[Dict[str, AgentType]] = {
    "car": AgentType.VEHICLE,
    "van": AgentType.VEHICLE,
    "truck": AgentType.VEHICLE,
    "bus": AgentType.VEHICLE,
    "trailer": AgentType.VEHICLE,
    "motorcycle": AgentType.MOTORCYCLE,
    "bicycle": AgentType.BICYCLE,
    "pedestrian": AgentType.PEDESTRIAN,
}

# Train / val split: first 70% of recordings for train, rest for val
# Recording IDs in the dataset: 2-23 (22 recordings total)
TRAIN_REC_IDS: Final[List[int]] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
VAL_REC_IDS: Final[List[int]] = [17, 18, 19, 20, 21, 22, 23]


def _rec_id_to_split(rec_id: int) -> str:
    if rec_id in TRAIN_REC_IDS:
        return "train"
    return "val"


class RounDDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        # Discover all recording IDs from files on disk
        data_path = Path(data_dir) / "data"
        rec_ids = sorted(
            {
                int(p.stem.split("_")[0])
                for p in data_path.glob("*_tracks.csv")
            }
        )

        scene_split_map: Dict[str, str] = {}
        for rid in rec_ids:
            scene_name = f"round_{rid:02d}"
            scene_split_map[scene_name] = _rec_id_to_split(rid)

        dataset_parts: List[Tuple[str, ...]] = [
            ("train", "val"),
            (ROUND_LOCATION,),
        ]

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=ROUND_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
            map_locations=(ROUND_LOCATION,),
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        data_path = Path(self.metadata.data_dir) / "data"
        self.dataset_obj: Dict[str, Dict[str, pd.DataFrame]] = {}

        rec_ids = sorted(
            {
                int(p.stem.split("_")[0])
                for p in data_path.glob("*_tracks.csv")
            }
        )

        for rid in rec_ids:
            tracks_path = data_path / f"{rid:02d}_tracks.csv"
            meta_path = data_path / f"{rid:02d}_tracksMeta.csv"

            tracks = pd.read_csv(tracks_path)
            tracks_meta = pd.read_csv(meta_path)

            # Normalise frame to start from 0
            min_frame = tracks["frame"].min()
            tracks["frame"] = tracks["frame"] - min_frame

            # Subsample: keep every SUBSAMPLE_FACTOR-th frame, then re-index
            tracks = tracks[tracks["frame"] % SUBSAMPLE_FACTOR == 0].copy()
            tracks["frame"] = tracks["frame"] // SUBSAMPLE_FACTOR

            scene_name = f"round_{rid:02d}"
            self.dataset_obj[scene_name] = {
                "tracks": tracks,
                "tracks_meta": tracks_meta,
                "min_frame": min_frame,
            }

        if verbose:
            print(
                f"  Loaded {len(self.dataset_obj)} recordings "
                f"({sum(d['tracks'].shape[0] for d in self.dataset_obj.values())} rows)",
                flush=True,
            )

    # ------------------------------------------------------------------
    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[RounDRecord] = []
        scenes_list: List[SceneMetadata] = []

        for idx, (scene_name, data) in enumerate(self.dataset_obj.items()):
            scene_split = self.metadata.scene_split_map[scene_name]
            scene_length = int(data["tracks"]["frame"].max()) + 1

            all_scenes_list.append(
                RounDRecord(scene_name, ROUND_LOCATION, scene_length, scene_split, idx)
            )

            if (
                ROUND_LOCATION in scene_tag
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
        all_scenes_list: List[RounDRecord] = env_cache.load_env_scenes_list(self.name)
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
            ROUND_LOCATION,
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
        tracks_meta: pd.DataFrame = data["tracks_meta"]

        # Build class lookup: trackId -> class string
        class_map = dict(
            zip(tracks_meta["trackId"], tracks_meta["class"])
        )
        # Build extent lookup: trackId -> (length, width)
        extent_map = dict(
            zip(
                tracks_meta["trackId"],
                zip(tracks_meta["length"], tracks_meta["width"]),
            )
        )

        # Rename columns to trajdata convention
        tracks.rename(
            columns={
                "frame": "scene_ts",
                "trackId": "agent_id",
                "xCenter": "x",
                "yCenter": "y",
                "heading": "heading_deg",
            },
            inplace=True,
        )

        # Convert heading from degrees to radians
        tracks["heading"] = np.deg2rad(tracks["heading_deg"])

        # z = 0 (2D dataset)
        tracks["z"] = 0.0

        # Use dataset-provided velocities (m/s) and accelerations (m/s^2)
        tracks.rename(
            columns={
                "xVelocity": "vx",
                "yVelocity": "vy",
                "xAcceleration": "ax",
                "yAcceleration": "ay",
            },
            inplace=True,
        )

        # Keep only required columns
        tracks = tracks[
            ["agent_id", "scene_ts", "x", "y", "z", "vx", "vy", "ax", "ay", "heading"]
        ]

        # Set multi-index
        tracks["agent_id"] = tracks["agent_id"].astype(int)
        tracks.set_index(["agent_id", "scene_ts"], inplace=True)
        tracks.sort_index(inplace=True)

        # Build agent metadata and presence lists
        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        agents_to_remove: List[int] = []

        for agent_id, grp in tracks.groupby(level="agent_id"):
            frames = grp.index.get_level_values("scene_ts")
            if len(frames) <= 1:
                agents_to_remove.append(agent_id)
                continue

            start_frame = int(frames.min())
            last_frame = int(frames.max())

            # Agent type
            cls_str = class_map.get(agent_id, "car")
            agent_type = CLASS_TO_AGENT_TYPE.get(cls_str, AgentType.VEHICLE)

            # Agent extent (length, width from tracksMeta; height=1.5m default)
            length, width = extent_map.get(agent_id, (4.0, 1.8))

            agent_metadata = AgentMetadata(
                name=str(agent_id),
                agent_type=agent_type,
                first_timestep=start_frame,
                last_timestep=last_frame,
                extent=FixedExtent(length=float(length), width=float(width), height=1.5),
            )

            agent_list.append(agent_metadata)
            for f in frames:
                if f < scene.length_timesteps:
                    agent_presence[f].append(agent_metadata)

        # Remove single-detection agents
        if agents_to_remove:
            tracks.drop(index=agents_to_remove, level="agent_id", inplace=True)

        # Convert agent_id index to string (trajdata convention)
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
        xodr_path = Path(self.metadata.data_dir) / "neuweiler.xodr"
        map_id = f"{self.name}:{map_name}"
        vector_map = xodr_to_vector_map(str(xodr_path), map_id, ds=0.3)
        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        self.cache_map(ROUND_LOCATION, cache_path, map_cache_class, map_params)
