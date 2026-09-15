"""Relative Elevation Model (REM), per spec §4.2 step 3.

Chosen over D8 HAND because flow routing on 1-arcsecond SRTM fails in flat,
low-gradient terrain -- the along-stream profile is built directly from
sampled elevation, made physically plausible by isotonic regression
(pool-adjacent-violators) in AHGF topological order, rather than by flow
accumulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from scipy.optimize import isotonic_regression
from scipy.spatial import cKDTree
from shapely.ops import substring

_REQUIRED_COLUMNS = ("HydroID", "From_Node", "To_Node", "NextDownID")


@dataclass(frozen=True)
class TerrainResult:
    rem: np.ndarray
    trough_depth: np.ndarray
    slope_deg: np.ndarray
    degraded_reasons: tuple[str, ...]


def _reach_topological_order(drainage: Any) -> list[Any]:
    """Headwater-to-outlet order, each reach exactly once.

    Deliberately reimplemented here rather than imported from
    ``hydrofragments.spatial.context.ordered_reach_paths`` -- that function
    returns full per-headwater PATHS (a shared downstream reach appears once
    per path reaching it), which would process a confluence reach multiple
    times; per-reach profile-capping (spec §4.2 step 3, "capping each
    reach's upstream end at its tributaries' minimum outflow") needs each
    reach processed exactly once, after all of its own upstream tributaries.
    Callers validate drainage topology first
    (``hydrofragments.spatial.context.validate_drainage_topology``) -- this
    module must not import ``hydrofragments.spatial``.
    """
    ids = list(drainage["HydroID"])
    id_set = set(ids)
    next_down = dict(zip(drainage["HydroID"], drainage["NextDownID"]))
    in_degree = {i: 0 for i in ids}
    for reach_id in ids:
        downstream = next_down[reach_id]
        if downstream in id_set:
            in_degree[downstream] += 1

    ready = sorted((i for i in ids if in_degree[i] == 0), key=str)
    order: list[Any] = []
    remaining = dict(in_degree)
    while ready:
        current = ready.pop(0)
        order.append(current)
        downstream = next_down[current]
        if downstream in id_set:
            remaining[downstream] -= 1
            if remaining[downstream] == 0:
                ready.append(downstream)
                ready.sort(key=str)

    if len(order) != len(ids):
        raise ValueError("drainage NextDownID topology contains a cycle")
    return order


def _sample_bin_elevations(
    geometry: Any, *, dem: np.ndarray, transform: Affine, bin_m: float,
    buffer_m: float, percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(along_stream_distance_m, elevation_m)`` per bin along ``geometry``.

    Each bin is a ``bin_m``-long line segment, buffered by ``buffer_m`` and
    rasterized; its elevation is the ``percentile`` of finite DEM values
    inside that buffer. A bin with no finite DEM pixels is dropped.
    """
    length = geometry.length
    if length <= 0:
        return np.array([]), np.array([])
    n_bins = max(1, int(np.ceil(length / bin_m)))
    starts = np.linspace(0.0, length, n_bins, endpoint=False)

    distances: list[float] = []
    elevations: list[float] = []
    for start in starts:
        end = min(start + bin_m, length)
        mid = (start + end) / 2.0
        segment = substring(geometry, start, end)
        footprint = segment.buffer(max(buffer_m, 1e-6))
        mask = rasterize(
            [(footprint, 1)], out_shape=dem.shape, transform=transform, fill=0,
            dtype="uint8", all_touched=True,
        ).astype(bool)
        values = dem[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        distances.append(mid)
        elevations.append(float(np.percentile(values, percentile)))
    return np.asarray(distances), np.asarray(elevations)


def _nan_uniform_filter(data: np.ndarray, *, radius_px: int) -> np.ndarray:
    """NaN-aware local mean via a uniform box filter (size ``2*radius_px+1``).

    Unlike filling NaN with the global mean before filtering (which biases
    every cell within ``radius_px`` of a masked region toward the whole
    array's mean), this normalizes by the count of VALID neighbours per
    cell, so masked cells simply don't contribute anywhere.
    """
    size = 2 * max(radius_px, 0) + 1
    valid = np.isfinite(data)
    filled = np.where(valid, data, 0.0)
    sum_values = ndimage.uniform_filter(filled, size=size, mode="constant") * size * size
    sum_valid = ndimage.uniform_filter(valid.astype(np.float64), size=size, mode="constant") * size * size
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sum_valid > 0, sum_values / sum_valid, np.nan)


def build_rem(
    drainage: Any,
    dem: np.ndarray,
    *,
    transform: Affine,
    pixel_m: float,
    profile_bin_m: float,
    profile_percentile: float,
    corridor_widths_m: dict[str, float],
    rem_k: int,
    rem_max_distance_m: float,
    trough_radius_m: float,
    trough_depth_m: float,
) -> TerrainResult:
    """Build the Relative Elevation Model, local trough depth, and slope rasters.

    ``drainage`` must carry ``HydroID``/``From_Node``/``To_Node``/
    ``NextDownID`` (already validated by the caller -- see
    ``_reach_topological_order``'s docstring). ``corridor_widths_m`` (from
    ``hydrofragments.riverscape.corridor.measure_corridor_widths``) sizes
    each reach's along-stream elevation-sampling footprint (half the
    corridor width, floored at one pixel).

    Per reach, in topological (headwater-to-outlet) order: sample
    ``profile_percentile`` elevation per ``profile_bin_m`` bin; cap the
    reach's own bins at the minimum already-regressed downstream-outflow
    value among its upstream tributaries (if any); run
    ``scipy.optimize.isotonic_regression(..., increasing=False)`` (PAVA) to
    enforce a non-increasing downstream profile; record this reach's own
    regressed downstream-most bin as the cap candidate for whatever reach
    it flows into. The resulting profile points (one per bin, all reaches)
    are spread onto the full grid by inverse-distance-weighted k-NN
    (``rem_k`` neighbours, ``scipy.spatial.cKDTree``, capped at
    ``rem_max_distance_m``). ``REM = dem - interpolated_profile``.
    ``trough_depth = local_mean(dem, trough_radius_m) - dem`` via a
    NaN-aware box filter. ``slope_deg`` from ``np.gradient``.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in drainage.columns]
    if missing:
        raise ValueError(f"drainage missing required columns: {missing}")
    if rem_k < 1:
        raise ValueError("rem_k must be a positive integer")

    dem_array = np.asarray(dem, dtype=np.float32)
    order = _reach_topological_order(drainage)
    geometry_by_id = dict(zip(drainage["HydroID"], drainage.geometry))
    next_down = dict(zip(drainage["HydroID"], drainage["NextDownID"]))
    id_set = set(order)

    downstream_cap: dict[Any, float] = {}
    profile_points: list[tuple[float, float, float]] = []
    degraded: list[str] = []

    for hydro_id in order:
        key = str(hydro_id)
        geometry = geometry_by_id[hydro_id]
        width_m = corridor_widths_m.get(key, profile_bin_m)

        distances, elevations = _sample_bin_elevations(
            geometry, dem=dem_array, transform=transform, bin_m=profile_bin_m,
            buffer_m=max(width_m / 2.0, pixel_m), percentile=profile_percentile,
        )
        if elevations.size == 0:
            degraded.append(f"reach_{key}_no_dem_samples")
            continue

        cap = downstream_cap.get(hydro_id)
        if cap is not None:
            elevations = np.minimum(elevations, cap)

        regressed = isotonic_regression(elevations, increasing=False).x

        downstream_id = next_down[hydro_id]
        if downstream_id in id_set:
            candidate = float(regressed[-1])
            existing = downstream_cap.get(downstream_id)
            downstream_cap[downstream_id] = (
                candidate if existing is None else min(existing, candidate)
            )

        for distance, elevation in zip(distances, regressed):
            point = geometry.interpolate(distance)
            profile_points.append((point.x, point.y, float(elevation)))

    if not profile_points:
        raise ValueError("no reach produced usable elevation samples for the profile")

    profile_xy = np.array([(x, y) for x, y, _ in profile_points])
    profile_z = np.array([z for _, _, z in profile_points])
    tree = cKDTree(profile_xy)

    rows, cols = np.indices(dem_array.shape)
    xs, ys = transform * (cols.ravel() + 0.5, rows.ravel() + 0.5)
    pixel_xy = np.column_stack([xs, ys])

    k = min(rem_k, profile_xy.shape[0])
    query_distances, query_indices = tree.query(pixel_xy, k=k)
    if k == 1:
        query_distances = query_distances[:, None]
        query_indices = query_indices[:, None]

    within_range = query_distances <= rem_max_distance_m
    weights = np.where(within_range, 1.0 / np.maximum(query_distances, 1e-6), 0.0)
    weight_sums = weights.sum(axis=1)
    interpolated = np.where(
        weight_sums > 0,
        np.sum(weights * profile_z[query_indices], axis=1) / np.maximum(weight_sums, 1e-12),
        np.nan,
    ).reshape(dem_array.shape)

    if not (weight_sums > 0).any():
        degraded.append("no_profile_points_within_rem_max_distance")

    rem = (dem_array - interpolated).astype(np.float32)

    finite_dem = np.where(np.isfinite(dem_array), dem_array, np.nan)
    local_mean = _nan_uniform_filter(finite_dem, radius_px=max(int(round(trough_radius_m / pixel_m)), 1))
    trough_depth = (local_mean - finite_dem).astype(np.float32)

    grad_y, grad_x = np.gradient(finite_dem, pixel_m)
    slope_deg = np.degrees(np.arctan(np.hypot(grad_x, grad_y))).astype(np.float32)

    return TerrainResult(
        rem=rem, trough_depth=trough_depth, slope_deg=slope_deg,
        degraded_reasons=tuple(degraded),
    )


__all__ = ["TerrainResult", "build_rem"]
