"""Per-reach corridor width: how far a river's search space extends around
its AHGF line, per spec §4.2 step 1.

Corridor width bounds the search space every later riverscape step
(centreline conflation, channel rules, gap bridging) uses -- it is not
itself a channel boundary or a final width estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from skimage.morphology import medial_axis

_REQUIRED_COLUMNS = ("HydroID",)


@dataclass(frozen=True)
class CorridorResult:
    """Per-reach corridor widths and diagnostics, keyed by ``str(HydroID)``."""

    widths_m: dict[str, float]
    p95_offset_m: dict[str, float]
    degraded_reasons: tuple[str, ...]


def _line_pixels(geometry: Any, *, transform: Affine, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if geometry is None or geometry.is_empty:
        return np.array([], dtype=int), np.array([], dtype=int)
    mask = rasterize(
        [(geometry, 1)], out_shape=shape, transform=transform, fill=0,
        dtype="uint8", all_touched=True,
    ).astype(bool)
    return np.nonzero(mask)


def measure_corridor_widths(
    drainage: Any,
    water_seed: np.ndarray,
    *,
    transform: Affine,
    pixel_m: float,
    f_seed: float,
    corridor_min_m: float,
    corridor_max_m: float,
    alignment_quantile: float = 0.95,
) -> CorridorResult:
    """Measure per-reach corridor width from AHGF-line-to-water-skeleton offset.

    ``drainage`` must be a GeoDataFrame with a unique ``HydroID`` column and
    LineString/MultiLineString geometry already co-projected with
    ``water_seed``'s grid (``transform``); this function does not validate
    drainage topology or CRS agreement -- callers use
    ``hydrofragments.spatial.context.validate_drainage_topology`` /
    ``create_channel_context`` first (this module must not import
    ``hydrofragments.spatial``, see
    ``tests/riverscape/test_riverscape_import_boundary.py``).

    ``water_seed`` is the caller's already-thresholded ``frequency >=
    f_seed`` boolean mask (``f_seed`` is accepted here only so it can be
    recorded; this function does not threshold anything itself).

    Per reach: ``width_m = clamp(p95_offset_m + max_half_width_m,
    corridor_min_m, corridor_max_m)``, where ``p95_offset_m`` is the
    ``alignment_quantile`` percentile of the reach's own line-pixel
    distances to the nearest water-seed skeleton pixel, and
    ``max_half_width_m`` is the largest water-seed half-width
    (``distance_transform_edt(water_seed)``) measured at those SAME
    nearest-skeleton-pixel locations -- i.e. how wide the water gets right
    where this reach's line is closest to it. A reach with no water-seed
    pixels anywhere on the grid gets ``corridor_max_m`` directly (maximally
    permissive search space, since there is nothing yet to conflate the
    line against) and is recorded in ``degraded_reasons``.
    """
    if drainage.empty:
        raise ValueError("drainage must contain at least one feature")
    missing = [c for c in _REQUIRED_COLUMNS if c not in drainage.columns]
    if missing:
        raise ValueError(f"drainage missing required columns: {missing}")
    if not 0.0 < alignment_quantile < 1.0:
        raise ValueError("alignment_quantile must be in (0, 1)")
    if not (0.0 < corridor_min_m < corridor_max_m):
        raise ValueError("corridor_min_m must be positive and less than corridor_max_m")

    water = np.asarray(water_seed, dtype=bool)

    if not water.any():
        widths = {str(h): corridor_max_m for h in drainage["HydroID"]}
        offsets = {str(h): float("nan") for h in drainage["HydroID"]}
        return CorridorResult(widths, offsets, ("no_water_seed_pixels",))

    skeleton = medial_axis(water)
    if not skeleton.any():
        skeleton = water  # a single-pixel-scale seed has no interior skeleton

    distance_to_skeleton, nearest_index = ndimage.distance_transform_edt(
        ~skeleton, return_indices=True
    )
    half_width_px = ndimage.distance_transform_edt(water)

    widths: dict[str, float] = {}
    offsets: dict[str, float] = {}
    degraded: list[str] = []

    for hydro_id, geometry in zip(drainage["HydroID"], drainage.geometry):
        key = str(hydro_id)
        rows, cols = _line_pixels(geometry, transform=transform, shape=water.shape)
        if rows.size == 0:
            widths[key] = corridor_max_m
            offsets[key] = float("nan")
            degraded.append(f"reach_{key}_outside_grid")
            continue

        offsets_px = distance_to_skeleton[rows, cols]
        nearest_rows = nearest_index[0][rows, cols]
        nearest_cols = nearest_index[1][rows, cols]
        half_widths_px = half_width_px[nearest_rows, nearest_cols]

        p95_offset_m = float(np.percentile(offsets_px, alignment_quantile * 100.0)) * pixel_m
        max_half_width_m = float(np.max(half_widths_px)) * pixel_m

        width_m = min(max(p95_offset_m + max_half_width_m, corridor_min_m), corridor_max_m)
        widths[key] = width_m
        offsets[key] = p95_offset_m

    return CorridorResult(widths, offsets, tuple(degraded))


__all__ = ["CorridorResult", "measure_corridor_widths"]
