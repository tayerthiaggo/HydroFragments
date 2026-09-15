"""Centreline conflation: EO water skeleton restricted to each reach's
corridor, per spec §4.2 step 2.
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
class CentrelineResult:
    """Combined skeleton (all reaches) and per-reach diagnostic flags."""

    skeleton: np.ndarray
    line_fallback_reaches: tuple[str, ...]
    multithread_reaches: tuple[str, ...]


def _has_loop(mask: np.ndarray) -> bool:
    """True if ``mask`` encloses at least one island (a real anabranch loop).

    A genuine loop (water splits and rejoins around dry land) encloses a
    hole in the water body -- detected as more than one connected
    component in the mask's background (4-connectivity), after padding by
    one pixel so the grid edge itself never counts as an enclosure. This
    replaces an earlier pixel-graph cycle check that false-positived on
    ordinary channel corners (any 8-adjacent L-shaped triple is a 3-cycle
    in that graph despite representing no real loop -- verified to fire on
    a plain sinuous single-thread channel with zero enclosed holes).
    """
    if not mask.any():
        return False
    background = ~np.pad(mask, 1, mode="constant", constant_values=False)
    _, n_background_components = ndimage.label(background)
    return n_background_components > 1


def build_centreline(
    drainage: Any,
    water_seed: np.ndarray,
    corridor_widths_m: dict[str, float],
    *,
    transform: Affine,
    pixel_m: float,
) -> CentrelineResult:
    """Conflate each reach's AHGF line onto the EO water skeleton within its corridor.

    For each reach, buffer its line by ``corridor_widths_m[str(HydroID)]``
    and intersect the buffer with ``water_seed``; the reach's contribution
    to the combined skeleton is ``medial_axis`` of that restricted mask.
    Gaps in the skeleton (where a reach has no water_seed pixels inside its
    own corridor) are left open here and resolved by gap bridging (spec
    §4.2 step 5, a later plan) -- such a reach is flagged ``line_fallback``:
    its AHGF line remains a topology/profile guide only and never becomes
    channel. A reach whose restricted skeleton contains a cycle (loop) is
    flagged ``multithread`` but still contributes its skeleton pixels --
    multithread (anabranching) channels are a real landform, not an error.

    ``drainage`` must carry a unique ``HydroID`` column; this function does
    not validate drainage topology (see
    ``hydrofragments.riverscape.corridor.measure_corridor_widths``'s
    docstring for the same caller-responsibility note -- this module must
    not import ``hydrofragments.spatial`` either).
    """
    if drainage.empty:
        raise ValueError("drainage must contain at least one feature")
    missing = [c for c in _REQUIRED_COLUMNS if c not in drainage.columns]
    if missing:
        raise ValueError(f"drainage missing required columns: {missing}")

    water = np.asarray(water_seed, dtype=bool)
    combined = np.zeros(water.shape, dtype=bool)
    fallback: list[str] = []
    multithread: list[str] = []

    for hydro_id, geometry in zip(drainage["HydroID"], drainage.geometry):
        key = str(hydro_id)
        if key not in corridor_widths_m:
            raise ValueError(f"corridor_widths_m missing entry for reach {key!r}")
        width_m = corridor_widths_m[key]
        if geometry is None or geometry.is_empty:
            fallback.append(key)
            continue

        corridor_mask = rasterize(
            [(geometry.buffer(width_m), 1)], out_shape=water.shape, transform=transform,
            fill=0, dtype="uint8", all_touched=True,
        ).astype(bool)
        restricted = water & corridor_mask
        if not restricted.any():
            fallback.append(key)
            continue

        reach_skeleton = medial_axis(restricted)
        if _has_loop(restricted):
            multithread.append(key)
        combined |= reach_skeleton

    return CentrelineResult(
        skeleton=combined,
        line_fallback_reaches=tuple(fallback),
        multithread_reaches=tuple(multithread),
    )


__all__ = ["CentrelineResult", "build_centreline"]
