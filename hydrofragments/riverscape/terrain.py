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
from shapely.ops import linemerge, substring

_REQUIRED_COLUMNS = ("HydroID", "From_Node", "To_Node", "NextDownID")


def _linear_parts(geometry: Any) -> list[Any]:
    """Return ``geometry`` as an ordered list of single-part ``LineString`` pieces.

    A ``LineString`` is returned as a one-element list unchanged. A
    ``MultiLineString`` (routinely produced by
    ``hydrofragments.spatial.context.create_channel_context`` clipping a
    reach against the AOI boundary via ``geometry.intersection``) is first
    passed through ``shapely.ops.linemerge``: when its parts share
    endpoints -- the common case for an AOI-boundary split -- linemerge
    stitches them back into one continuous ``LineString``, restoring a
    single along-stream distance axis. If linemerge still returns a
    ``MultiLineString`` (the parts are genuinely disjoint, no shared
    endpoints), each part is kept separately, in ``geometry.geoms`` order,
    for the caller to sample/measure independently.
    """
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        merged = linemerge(geometry)
        if merged.geom_type == "LineString":
            return [merged]
        return list(merged.geoms)
    raise ValueError(f"unsupported reach geometry type: {geometry.geom_type!r}")


def _line_endpoints(geometry: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``(first_coord, last_coord)`` for a Line/MultiLineString reach.

    Delegates to ``_linear_parts`` so a mergeable ``MultiLineString`` yields
    the same endpoints as its merged ``LineString`` would. For a genuinely
    disjoint ``MultiLineString``, this uses the first coordinate of the
    first part and the last coordinate of the last part (in geometry
    order) as the effective endpoints -- an approximation, since disjoint
    parts have no single well-defined direction, but sufficient for this
    diagnostic-only direction check (it must never crash, not be exact).
    """
    parts = _linear_parts(geometry)
    return parts[0].coords[0], parts[-1].coords[-1]


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
    if len(set(ids)) != len(ids):
        seen: set[Any] = set()
        duplicates: set[Any] = set()
        for reach_id in ids:
            if reach_id in seen:
                duplicates.add(reach_id)
            seen.add(reach_id)
        raise ValueError(
            f"drainage HydroID contains duplicate value(s): {sorted(duplicates, key=str)}"
        )
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(along_stream_distance_m, elevation_m, point_xy)`` per bin.

    Each bin is a ``bin_m``-long line segment, buffered by ``buffer_m`` and
    rasterized; its elevation is the ``percentile`` of finite DEM values
    inside that buffer. A bin with no finite DEM pixels is dropped.

    ``geometry`` may be a ``LineString`` or ``MultiLineString`` (see
    ``_linear_parts``); each returned part is sampled independently, with
    ``along_stream_distance_m`` running continuously across parts (each
    part's own distances offset by the running total length of the parts
    before it) so bins from different parts never collide. ``point_xy`` is
    each bin's midpoint, already resolved against the correct part -- the
    caller must use these coordinates directly rather than re-interpolating
    ``distance`` against the original (possibly multi-part) ``geometry``,
    since ``LineString.interpolate`` is not defined for a MultiLineString
    and, even after linemerge, its internal vertex order need not match
    this function's per-part distance axis.
    """
    parts = _linear_parts(geometry)

    distances: list[float] = []
    elevations: list[float] = []
    points: list[tuple[float, float]] = []
    running_offset = 0.0
    for part in parts:
        length = part.length
        if length <= 0:
            continue
        n_bins = max(1, int(np.ceil(length / bin_m)))
        starts = np.linspace(0.0, length, n_bins, endpoint=False)
        for start in starts:
            end = min(start + bin_m, length)
            mid = (start + end) / 2.0
            segment = substring(part, start, end)
            footprint = segment.buffer(max(buffer_m, 1e-6))
            mask = rasterize(
                [(footprint, 1)], out_shape=dem.shape, transform=transform, fill=0,
                dtype="uint8", all_touched=True,
            ).astype(bool)
            values = dem[mask]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            distances.append(running_offset + mid)
            elevations.append(float(np.percentile(values, percentile)))
            point = part.interpolate(mid)
            points.append((point.x, point.y))
        running_offset += length

    if not distances:
        return np.array([]), np.array([]), np.zeros((0, 2))
    return np.asarray(distances), np.asarray(elevations), np.asarray(points)


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
) -> TerrainResult:
    """Build the Relative Elevation Model, local trough depth, and slope rasters.

    ``drainage`` must carry ``HydroID``/``From_Node``/``To_Node``/
    ``NextDownID`` (already validated by the caller -- see
    ``_reach_topological_order``'s docstring). ``corridor_widths_m`` (from
    ``hydrofragments.riverscape.corridor.measure_corridor_widths``) sizes
    each reach's along-stream elevation-sampling footprint: the corridor
    width itself, used directly as a radius around the reach's line
    (floored at one pixel) -- consistent with how
    ``hydrofragments.riverscape.centreline.build_centreline`` uses the same
    value (``geometry.buffer(width_m)``). Every ``str(HydroID)`` key in
    ``drainage`` must be present in ``corridor_widths_m``; a missing key
    raises ``ValueError`` rather than silently substituting a default (the
    same contract ``build_centreline`` enforces).

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
    NaN-aware box filter -- this is the raw, continuous local-trough-depth
    signal; thresholding it into a boolean/evidence signal against a depth
    cutoff is a downstream (Phase 4 channel-rules) concern, not this
    module's, so no trough-depth threshold parameter is accepted here.
    ``slope_deg`` from ``np.gradient``.

    Flow direction (which end of each reach's ``LineString`` is
    "downstream") is assumed from coordinate digitisation order
    (``regressed[-1]`` = downstream outflow) since ``From_Node``/``To_Node``
    are not otherwise read. For every reach whose ``NextDownID`` names
    another reach in this same drainage set, the endpoint-proximity to that
    downstream reach is checked as a diagnostic: if the reach's first
    coordinate sits closer to the downstream reach than its last coordinate
    does, ``f"reach_{key}_geometry_direction_suspect"`` is appended to
    ``degraded_reasons`` (a flag, not an error -- some legitimate low
    sample-density topologies can trigger it too).
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in drainage.columns]
    if missing:
        raise ValueError(f"drainage missing required columns: {missing}")
    if rem_k < 1:
        raise ValueError("rem_k must be a positive integer")
    if not np.isfinite(pixel_m) or pixel_m <= 0:
        raise ValueError("pixel_m must be finite and positive")
    if not np.isfinite(profile_bin_m) or profile_bin_m <= 0:
        raise ValueError("profile_bin_m must be finite and positive")
    if not (0.0 <= profile_percentile <= 100.0):
        raise ValueError("profile_percentile must be in [0, 100]")
    if not np.isfinite(rem_max_distance_m) or rem_max_distance_m <= 0:
        raise ValueError("rem_max_distance_m must be finite and positive")
    if not np.isfinite(trough_radius_m) or trough_radius_m <= 0:
        raise ValueError("trough_radius_m must be finite and positive")

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
        if key not in corridor_widths_m:
            raise ValueError(f"corridor_widths_m missing entry for reach {key!r}")
        width_m = corridor_widths_m[key]

        distances, elevations, points_xy = _sample_bin_elevations(
            geometry, dem=dem_array, transform=transform, bin_m=profile_bin_m,
            buffer_m=max(width_m, pixel_m), percentile=profile_percentile,
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
            downstream_geometry = geometry_by_id[downstream_id]
            own_first, own_last = _line_endpoints(geometry)
            downstream_ends = _line_endpoints(downstream_geometry)
            dist_first = min(np.hypot(own_first[0] - dx, own_first[1] - dy) for dx, dy in downstream_ends)
            dist_last = min(np.hypot(own_last[0] - dx, own_last[1] - dy) for dx, dy in downstream_ends)
            if dist_first < dist_last:
                degraded.append(f"reach_{key}_geometry_direction_suspect")

            candidate = float(regressed[-1])
            existing = downstream_cap.get(downstream_id)
            downstream_cap[downstream_id] = (
                candidate if existing is None else min(existing, candidate)
            )

        for (x, y), elevation in zip(points_xy, regressed):
            profile_points.append((float(x), float(y), float(elevation)))

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
