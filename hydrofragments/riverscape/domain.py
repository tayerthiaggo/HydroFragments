"""Observed-wet analysis domain shared by the landform and hydroperiod layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hydrofragments.io.dea import WoStatistics


def wet_domain(stats: "WoStatistics", *, min_valid_obs: int) -> np.ndarray:
    """Return the native-grid observed-wet domain as a 2-D bool array.

    ``domain = (count_wet > 0) & isfinite(frequency) & (count_clear >=
    min_valid_obs)`` -- the same extent ``build_zones`` zones. It is derived
    only from the DEA WO statistics themselves; the coarsened planning
    footprint / ``analysis_mask`` must never be used for zoning, so this
    function deliberately has no parameter that could accept one.
    """
    if min_valid_obs < 1:
        raise ValueError("min_valid_obs must be at least 1")
    frequency = np.asarray(stats.frequency, dtype=float)
    count_wet = np.asarray(stats.count_wet)
    count_clear = np.asarray(stats.count_clear)
    if frequency.ndim != 2:
        raise ValueError("frequency must be a 2-D array")
    if count_wet.shape != frequency.shape or count_clear.shape != frequency.shape:
        raise ValueError("frequency, count_wet and count_clear must share shape")
    return (count_wet > 0) & np.isfinite(frequency) & (count_clear >= min_valid_obs)


__all__ = ["wet_domain"]
