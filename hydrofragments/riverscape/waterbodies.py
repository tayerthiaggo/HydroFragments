"""DEA Waterbodies fragment merging and riverine-role classification."""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

WATERBODY_RULESET_VERSION = "1.0.0"
MIN_RIVERINE_ELONGATION = 2.0
MIN_CENTRELINE_SPAN_FRACTION = 0.5
MIN_BOUNDARY_CONTACTS = 2


@dataclass(frozen=True)
class WaterbodyResult:
    polygons: gpd.GeoDataFrame
    riverine_mask: np.ndarray
    off_channel_mask: np.ndarray


def _polygon_parts(geometry):
    if isinstance(geometry, Polygon) and not geometry.is_empty:
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    return []


def _merged_components(polygons: gpd.GeoDataFrame) -> list[tuple[int, object]]:
    parts: list[tuple[int, object]] = []
    for position, geometry in enumerate(polygons.geometry):
        parts.extend((position, part) for part in _polygon_parts(geometry))
    parent = list(range(len(parts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(parts)):
        for right in range(left + 1, len(parts)):
            if parts[left][1].intersects(parts[right][1]):
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(len(parts)):
        groups.setdefault(find(index), []).append(index)
    merged = []
    for indices in groups.values():
        source_index = min(parts[index][0] for index in indices)
        merged.append(
            (source_index, unary_union([parts[index][1] for index in indices]))
        )
    return sorted(merged, key=lambda item: item[0])


def _rectangle_axes(
    geometry, pixel_m: float
) -> tuple[float, float, np.ndarray]:
    rectangle = geometry.minimum_rotated_rectangle
    coords = np.asarray(rectangle.exterior.coords, float)
    edges = np.diff(coords, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    major_index = int(np.argmax(lengths))
    major = float(lengths[major_index])
    minor = float(np.min(lengths))
    unit = edges[major_index] / max(major, pixel_m)
    return major, max(minor, pixel_m), unit


def _pixel_centres(mask: np.ndarray, transform: Affine) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.empty((0, 2), float)
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    return np.column_stack([xs, ys])


def _projected_span(points: np.ndarray, unit: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    projected = points @ unit
    return float(projected.max() - projected.min())


def classify_waterbodies(
    polygons: gpd.GeoDataFrame,
    centreline: np.ndarray,
    *,
    transform: Affine,
    pixel_m: float,
) -> WaterbodyResult:
    centreline = np.asarray(centreline, bool)
    if centreline.ndim != 2:
        raise ValueError("centreline must be 2-D")
    if pixel_m <= 0:
        raise ValueError("pixel_m must be positive")
    rows: list[dict[str, object]] = []
    riverine = np.zeros(centreline.shape, bool)
    off_channel = np.zeros(centreline.shape, bool)
    structure = np.ones((3, 3), bool)

    for waterbody_id, (_, geometry) in enumerate(
        _merged_components(polygons), start=1
    ):
        body = rasterize(
            [(geometry, 1)],
            out_shape=centreline.shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)
        inside_line = centreline & body
        exterior_ring = ndimage.binary_dilation(body, structure=structure) & ~body
        crossing_inside = inside_line & ndimage.binary_dilation(
            centreline & exterior_ring, structure=structure
        )
        _, contacts = ndimage.label(crossing_inside, structure=structure)
        major, minor, major_unit = _rectangle_axes(geometry, pixel_m)
        elongation = major / minor
        span_fraction = _projected_span(
            _pixel_centres(inside_line, transform), major_unit
        ) / max(major, pixel_m)
        is_riverine = (
            contacts >= MIN_BOUNDARY_CONTACTS
            and elongation >= MIN_RIVERINE_ELONGATION
            and span_fraction >= MIN_CENTRELINE_SPAN_FRACTION
        )
        role = "riverine" if is_riverine else "off_channel"
        rows.append(
            {
                "waterbody_id": waterbody_id,
                "role": role,
                "elongation": float(elongation),
                "centreline_span_fraction": float(span_fraction),
                "boundary_contacts": int(contacts),
                "geometry": geometry,
            }
        )
        if is_riverine:
            riverine |= body
        else:
            off_channel |= body

    frame = gpd.GeoDataFrame(
        rows,
        columns=[
            "waterbody_id",
            "role",
            "elongation",
            "centreline_span_fraction",
            "boundary_contacts",
            "geometry",
        ],
        geometry="geometry",
        crs=polygons.crs,
    )
    return WaterbodyResult(frame, riverine, off_channel)


__all__ = [
    "MIN_BOUNDARY_CONTACTS",
    "MIN_CENTRELINE_SPAN_FRACTION",
    "MIN_RIVERINE_ELONGATION",
    "WATERBODY_RULESET_VERSION",
    "WaterbodyResult",
    "classify_waterbodies",
]
