"""
Minimal OpenDRIVE (.xodr) parser that converts road/lane geometry into
trajdata VectorMap elements (RoadLane with center, left_edge, right_edge).

Supports geometry types: line, arc, spiral (Euler spiral / clothoid).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ #
# Reference-line geometry primitives
# ------------------------------------------------------------------ #

def _sample_line(x0, y0, hdg, length, ds=0.5):
    n = max(int(length / ds), 2)
    s = np.linspace(0, length, n)
    x = x0 + s * np.cos(hdg)
    y = y0 + s * np.sin(hdg)
    h = np.full_like(s, hdg)
    return np.column_stack([x, y, h])


def _sample_arc(x0, y0, hdg, length, curvature, ds=0.5):
    if abs(curvature) < 1e-12:
        return _sample_line(x0, y0, hdg, length, ds)
    n = max(int(length / ds), 2)
    s = np.linspace(0, length, n)
    r = 1.0 / curvature
    theta = hdg + s * curvature
    x = x0 + r * (np.sin(theta) - np.sin(hdg))
    y = y0 + r * (np.cos(hdg) - np.cos(theta))
    return np.column_stack([x, y, theta])


def _sample_spiral(x0, y0, hdg, length, curv_start, curv_end, ds=0.5):
    """Euler spiral (clothoid) via numerical integration."""
    n = max(int(length / ds), 2)
    s_arr = np.linspace(0, length, n)
    dt = s_arr[1] - s_arr[0] if n > 1 else length

    x = np.zeros(n)
    y = np.zeros(n)
    h = np.zeros(n)
    x[0], y[0], h[0] = x0, y0, hdg

    for i in range(1, n):
        s = s_arr[i]
        # Linear interpolation of curvature
        k = curv_start + (curv_end - curv_start) * s / length if length > 0 else 0
        h[i] = h[i - 1] + k * dt
        x[i] = x[i - 1] + dt * np.cos(h[i - 1])
        y[i] = y[i - 1] + dt * np.sin(h[i - 1])

    return np.column_stack([x, y, h])


def sample_geometry(geom_elem, ds=0.5):
    """Sample an OpenDRIVE <geometry> element into (N, 3) array [x, y, hdg]."""
    x0 = float(geom_elem.get("x"))
    y0 = float(geom_elem.get("y"))
    hdg = float(geom_elem.get("hdg"))
    length = float(geom_elem.get("length"))

    child = geom_elem[0]  # <line/>, <arc/>, or <spiral/>
    tag = child.tag

    if tag == "line":
        return _sample_line(x0, y0, hdg, length, ds)
    elif tag == "arc":
        curv = float(child.get("curvature"))
        return _sample_arc(x0, y0, hdg, length, curv, ds)
    elif tag == "spiral":
        curv_s = float(child.get("curvStart"))
        curv_e = float(child.get("curvEnd"))
        return _sample_spiral(x0, y0, hdg, length, curv_s, curv_e, ds)
    else:
        # Fallback: straight line
        return _sample_line(x0, y0, hdg, length, ds)


# ------------------------------------------------------------------ #
# Lane offset computation
# ------------------------------------------------------------------ #

def _lane_width_at_s(lane_elem, s_local):
    """Evaluate lane width polynomial at local s offset within laneSection."""
    width_elem = lane_elem.find("width")
    if width_elem is None:
        return 0.0
    a = float(width_elem.get("a", 0))
    b = float(width_elem.get("b", 0))
    c = float(width_elem.get("c", 0))
    d = float(width_elem.get("d", 0))
    return a + b * s_local + c * s_local ** 2 + d * s_local ** 3


@dataclass
class LaneGeometry:
    lane_id: int
    lane_type: str
    center: np.ndarray      # (N, 3) x, y, z=0
    left_edge: np.ndarray   # (N, 3)
    right_edge: np.ndarray  # (N, 3)
    road_id: str = "0"


def compute_lane_boundaries(
    ref_line: np.ndarray, lane_section_elem, ds=0.5
) -> List[LaneGeometry]:
    """
    Given a reference line (N, 3) [x, y, hdg] and a <laneSection>,
    compute left/right boundaries for each driving lane.
    """
    results = []
    n = ref_line.shape[0]

    # s values along the reference line
    diffs = np.sqrt(np.diff(ref_line[:, 0]) ** 2 + np.diff(ref_line[:, 1]) ** 2)
    s_values = np.concatenate([[0], np.cumsum(diffs)])

    # Process left lanes (positive IDs, offset to the left of reference)
    left_elem = lane_section_elem.find("left")
    if left_elem is not None:
        # Sort by ID ascending (1, 2, 3...) — lane 1 is closest to center
        lanes = sorted(left_elem.findall("lane"), key=lambda e: int(e.get("id")))
        cumulative_offset = np.zeros(n)
        for lane_elem in lanes:
            lane_id = int(lane_elem.get("id"))
            lane_type = lane_elem.get("type", "driving")
            widths = np.array([_lane_width_at_s(lane_elem, s) for s in s_values])

            inner_offset = cumulative_offset.copy()
            cumulative_offset = cumulative_offset + widths
            outer_offset = cumulative_offset.copy()
            center_offset = (inner_offset + outer_offset) / 2

            # Left = positive direction = perpendicular left of heading
            normals = np.column_stack([
                -np.sin(ref_line[:, 2]),
                np.cos(ref_line[:, 2]),
            ])

            center_pts = ref_line[:, :2] + normals * center_offset[:, None]
            left_pts = ref_line[:, :2] + normals * outer_offset[:, None]
            right_pts = ref_line[:, :2] + normals * inner_offset[:, None]

            z = np.zeros((n, 1))
            results.append(LaneGeometry(
                lane_id=lane_id,
                lane_type=lane_type,
                center=np.hstack([center_pts, z]),
                left_edge=np.hstack([left_pts, z]),
                right_edge=np.hstack([right_pts, z]),
            ))

    # Process right lanes (negative IDs, offset to the right)
    right_elem = lane_section_elem.find("right")
    if right_elem is not None:
        # Sort by ID descending (-1, -2, -3...) — -1 is closest to center
        lanes = sorted(right_elem.findall("lane"), key=lambda e: int(e.get("id")), reverse=True)
        cumulative_offset = np.zeros(n)
        for lane_elem in lanes:
            lane_id = int(lane_elem.get("id"))
            lane_type = lane_elem.get("type", "driving")
            widths = np.array([_lane_width_at_s(lane_elem, s) for s in s_values])

            inner_offset = cumulative_offset.copy()
            cumulative_offset = cumulative_offset + widths
            outer_offset = cumulative_offset.copy()
            center_offset = (inner_offset + outer_offset) / 2

            # Right = negative direction
            normals = np.column_stack([
                np.sin(ref_line[:, 2]),
                -np.cos(ref_line[:, 2]),
            ])

            center_pts = ref_line[:, :2] + normals * center_offset[:, None]
            left_pts = ref_line[:, :2] + normals * inner_offset[:, None]
            right_pts = ref_line[:, :2] + normals * outer_offset[:, None]

            z = np.zeros((n, 1))
            results.append(LaneGeometry(
                lane_id=lane_id,
                lane_type=lane_type,
                center=np.hstack([center_pts, z]),
                left_edge=np.hstack([left_pts, z]),
                right_edge=np.hstack([right_pts, z]),
            ))

    return results


# ------------------------------------------------------------------ #
# Main parser
# ------------------------------------------------------------------ #

def parse_xodr(xodr_path: str, ds: float = 0.5) -> List[LaneGeometry]:
    """Parse an OpenDRIVE file and return all lane geometries."""
    tree = ET.parse(xodr_path)
    root = tree.getroot()

    all_lanes: List[LaneGeometry] = []

    for road in root.findall("road"):
        road_id = road.get("id", "0")
        # Build reference line by concatenating all geometry segments
        ref_segments = []
        for geom in road.findall(".//planView/geometry"):
            pts = sample_geometry(geom, ds=ds)
            ref_segments.append(pts)

        if not ref_segments:
            continue

        ref_line = np.vstack(ref_segments)

        # Process each laneSection
        for lane_section in road.findall(".//lanes/laneSection"):
            lanes = compute_lane_boundaries(ref_line, lane_section, ds=ds)
            # Tag each lane with road_id for unique naming
            for lg in lanes:
                lg.road_id = road_id
            all_lanes.extend(lanes)

    return all_lanes


def xodr_to_vector_map(xodr_path: str, map_id: str, ds: float = 0.3):
    """Convert an OpenDRIVE file to a trajdata VectorMap."""
    from trajdata.maps.vec_map import VectorMap
    from trajdata.maps.vec_map_elements import Polyline, RoadLane

    lane_geoms = parse_xodr(xodr_path, ds=ds)
    vector_map = VectorMap(map_id=map_id)

    for i, lg in enumerate(lane_geoms):
        if lg.lane_type != "driving":
            continue

        lane_id = f"road{getattr(lg, 'road_id', '0')}_lane{lg.lane_id}_{i}"

        road_lane = RoadLane(
            id=lane_id,
            center=Polyline(points=lg.center),
            left_edge=Polyline(points=lg.left_edge),
            right_edge=Polyline(points=lg.right_edge),
        )
        vector_map.add_map_element(road_lane)

    # Compute extent from all lane points
    all_pts = []
    for lg in lane_geoms:
        all_pts.extend([lg.center, lg.left_edge, lg.right_edge])
    if all_pts:
        pts = np.vstack(all_pts)
        vector_map.extent = np.array([
            pts[:, 0].min(), pts[:, 1].min(), 0.0,
            pts[:, 0].max(), pts[:, 1].max(), 0.0,
        ])

    return vector_map
