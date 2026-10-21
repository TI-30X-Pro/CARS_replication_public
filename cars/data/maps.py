"""Rasterise nuScenes vector maps for agent-centric model inputs.

From the nuScenes map expansion, per location:
  - lanes: the lane polygon is split into a left and a right edge (by which side of the
    centre line each vertex lies on), both resampled to the same number of points;
  - lane connectors: centre line only, so they are not drawn;
  - drivable area: road polygons, with their holes cut out;
  - pedestrian crossings and walkways: polygons.
These are drawn with OpenCV (anti-aliased, 1/512 px sub-pixel precision) into three
layers at `px_per_m`: drivable area (lanes + road polygons), lane edges, pedestrian areas.
Values are in [0, 1]. Drawing order matters at the anti-aliased borders: lanes first,
then road polygons, then crosswalks and walkways, each in map-file order.
"""
from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np
from scipy.spatial.distance import cdist

LAYERS = ("drivable_area", "lane_divider", "ped_area")
SUBPIXEL_SHIFT = 9
LOCATIONS = ("boston-seaport", "singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown")


@dataclass
class MapRaster:
    name: str
    data: np.ndarray            # (3, H, W) float32 in [0, 1]
    map_from_world: np.ndarray  # (3, 3): world [m] -> raster [px]
    px_per_m: float


# Read vector-map geometry

def lane_center(nusc_map, lane_record):
    """Lane centre line at 1 m spacing, duplicate points removed."""
    from nuscenes.map_expansion import arcline_path_utils
    arc = nusc_map.arcline_path_3.get(lane_record["token"], [])
    mid = np.array(arcline_path_utils.discretize_lane(arc, resolution_meters=1.0))[:, :2]
    dup = np.where(np.linalg.norm(np.diff(mid, axis=0, prepend=0), axis=1) < 1e-10)[0]
    if dup.size > 0:
        mid = np.delete(mid, dup, axis=0)
    return mid


def _order_matches(pts, ref):
    return np.linalg.norm(pts[0] - ref[0]) <= np.linalg.norm(pts[-1] - ref[0])


def _endpoints_intersect(left, right):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    a, b = left[-1], right[-1]
    c, d = right[0], left[0]
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def _resample(pts, num_pts):
    """Resample a 2-D polyline to num_pts points, evenly spaced along its length."""
    if pts.shape[0] == num_pts:
        return pts
    cum = np.cumsum(np.linalg.norm(np.diff(pts[..., :2], axis=0), axis=-1))
    cum = np.insert(cum, 0, 0)
    steps = np.linspace(cum[0], cum[-1], num_pts)
    out = np.empty((num_pts, pts.shape[-1]), dtype=pts.dtype)
    for i in range(2):
        out[:, i] = np.interp(steps, xp=cum, fp=pts[:, i])
    return out


def lane_edges(nusc_map, lane_record):
    """Left and right edge of a lane polygon, with the same number of points."""
    poly = nusc_map.get("polygon", lane_record["polygon_token"])
    pts = np.array([(n["x"], n["y"]) for n in
                    (nusc_map.get("node", t) for t in poly["exterior_node_tokens"])])
    mid = lane_center(nusc_map, lane_record)

    closest = np.argmin(cdist(pts, mid), axis=1)
    direction = np.diff(mid, axis=0, prepend=mid[[0]] - (mid[[1]] - mid[[0]]))
    d, v = direction[closest], pts - mid[closest]
    on_right = (d[:, 0] * v[:, 1] - d[:, 1] * v[:, 0]) < 0

    change = np.where(np.roll(on_right, 1) < on_right)[0].item()
    if change > 0:
        pts = np.roll(pts, shift=-change, axis=0)
        on_right = np.roll(on_right, shift=-change)
    left, right = pts[~on_right], pts[on_right]

    if _endpoints_intersect(left, right):
        if not _order_matches(left, mid):
            left = left[::-1]
        else:
            right = right[::-1]
    if left.shape[0] < right.shape[0]:
        left = _resample(left, right.shape[0])
    elif right.shape[0] < left.shape[0]:
        right = _resample(right, left.shape[0])
    return left, right


def _area_points(nusc_map, record):
    key = "exterior_node_tokens" if "exterior_node_tokens" in record else "node_tokens"
    return np.array([(n["x"], n["y"]) for n in (nusc_map.get("node", t) for t in record[key])])


def vector_elements(nusc_map):
    """The drawable elements in trajdata's order: (kind, data) tuples.

    trajdata keeps elements in one dict per type, keyed by id, so a repeated id is drawn
    once, at the position where it first appeared.
    """
    lanes = {rec["token"]: lane_edges(nusc_map, rec) for rec in nusc_map.lane}
    # lane connectors only have a centre line and are never drawn
    roads = {}
    for area in nusc_map.drivable_area:
        for token in area["polygon_tokens"]:
            if token is None and str(None) in roads:
                continue   # nuScenes has two None tokens in a row; trajdata keeps one
            poly = nusc_map.get("polygon", token)
            holes = [_area_points(nusc_map, h) for h in poly["holes"]]
            roads[str(token)] = (_area_points(nusc_map, poly), holes)
    crossings = {rec["token"]: _area_points(nusc_map, rec) for rec in nusc_map.ped_crossing}
    walkways = {rec["token"]: _area_points(nusc_map, rec) for rec in nusc_map.walkway}
    return ([("lane", v) for v in lanes.values()] + [("road", v) for v in roads.values()]
            + [("ped", v) for v in crossings.values()] + [("ped", v) for v in walkways.values()])


# Rasterise map layers

def _subpixel(pts, map_from_world):
    xy = pts[..., :2] @ map_from_world[:2, :2] + map_from_world[:2, -1]
    return (xy * 2 ** SUBPIXEL_SHIFT).astype(int)


def _fill(img, pts, color):
    cv2.fillPoly(img=img, pts=[pts], color=color, shift=SUBPIXEL_SHIFT, lineType=cv2.LINE_AA)


def _lines(img, polylines, color):
    cv2.polylines(img=img, pts=polylines, isClosed=False, color=color,
                  shift=SUBPIXEL_SHIFT, lineType=cv2.LINE_AA)


def rasterize(nusc_map, px_per_m=1.0, name=""):
    e = nusc_map.explorer
    min_x, min_y, max_x, max_y = e.canvas_min_x, e.canvas_min_y, e.canvas_max_x, e.canvas_max_y
    w = ceil((max_x - min_x) * px_per_m)
    h = ceil((max_y - min_y) * px_per_m)
    raster_from_local = np.array([[px_per_m, 0, w / 2], [0, px_per_m, h / 2], [0, 0, 1]])
    local_from_world = np.array([[1, 0, -(min_x + max_x) / 2], [0, 1, -(min_y + max_y) / 2], [0, 0, 1]])
    map_from_world = raster_from_local @ local_from_world

    area = np.zeros((h, w, 3), dtype=np.uint8)
    line = np.zeros((h, w, 3), dtype=np.uint8)
    ped = np.zeros((h, w, 3), dtype=np.uint8)
    for kind, data in vector_elements(nusc_map):
        if kind == "lane":
            left, right = data
            _fill(area, _subpixel(np.concatenate([left, right[::-1]], axis=0), map_from_world), (255, 0, 0))
            _lines(line, [_subpixel(left, map_from_world), _subpixel(right[::-1], map_from_world)], (0, 255, 0))
        elif kind == "road":
            exterior, holes = data
            _fill(area, _subpixel(exterior, map_from_world), (255, 0, 0))
            for hole in holes:
                _fill(area, _subpixel(hole, map_from_world), (0, 0, 0))
        else:
            _fill(ped, _subpixel(data, map_from_world), (0, 0, 255))

    data = (area + line + ped).astype(np.float32).transpose(2, 0, 1) / 255
    return MapRaster(name=name, data=data, map_from_world=map_from_world, px_per_m=px_per_m)


def load_nusc_map(nusc_root, location):
    from nuscenes.map_expansion.map_api import NuScenesMap
    return NuScenesMap(dataroot=str(nusc_root), map_name=location)


def build_rasters(nusc_root, out, px_per_m=1.0, locations=LOCATIONS):
    """Rasterise every nuScenes map and save them to one .npz."""
    arrays = {}
    for loc in locations:
        r = rasterize(load_nusc_map(nusc_root, loc), px_per_m, loc)
        arrays[loc + "/data"] = r.data
        arrays[loc + "/map_from_world"] = r.map_from_world
        arrays[loc + "/px_per_m"] = np.array(px_per_m)
    np.savez_compressed(out, **arrays)


def load_rasters(path):
    with np.load(path) as z:
        names = sorted({k.split("/")[0] for k in z.files})
        return {n: MapRaster(n, z[n + "/data"], z[n + "/map_from_world"], float(z[n + "/px_per_m"]))
                for n in names}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Rasterise the nuScenes maps.")
    ap.add_argument("--root", required=True, help="nuScenes root (holds maps/expansion)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--px_per_m", type=float, default=1.0)
    a = ap.parse_args()
    build_rasters(a.root, a.out, a.px_per_m)
    print("wrote", a.out)
