"""Hydroperiod classifier on multi-year DEA WO frequency.

Independent of the landform layer by contract: this module never imports
``hydrofragments.riverscape``. Later classifiers (DEA WO seasonal summaries,
hydroseason ``end_dry`` snapshots) replace this module without touching the
landform layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydrofragments.hydroperiod.codes import (
    HYDROPERIOD_MARGINAL,
    HYDROPERIOD_PERSISTENT,
    HYDROPERIOD_SEASONAL,
    HYDROPERIOD_UNOBSERVED,
)


@dataclass(frozen=True)
class HydroperiodResult:
    classes: np.ndarray
    method: str
    t_persist: float
    t_season: float


def classify_hydroperiod(
    frequency,
    domain,
    *,
    t_persist: float,
    t_season: float,
    unobserved_mask=None,
) -> HydroperiodResult:
    """Classify observed-wet pixels as persistent, seasonal or marginal.

    ``frequency`` is PERCENT (0-100); ``t_persist``/``t_season`` are FRACTIONS
    converted to percent once here. Boundaries match ``build_zones``:
    marginal ``< t_season``, seasonal ``t_season..t_persist`` inclusive,
    persistent ``> t_persist``. Pixels outside ``domain`` are 0 unless
    ``unobserved_mask`` supplies them (bridged channel pixels), which get
    ``HYDROPERIOD_UNOBSERVED``; that class is never inferred from frequency
    or neighbours.
    """
    values = np.asarray(frequency, dtype=float)
    inside = np.asarray(domain, dtype=bool)
    if values.ndim != 2:
        raise ValueError("frequency must be a 2-D array")
    if inside.shape != values.shape:
        raise ValueError("frequency and domain must share shape")
    if not 0.0 <= t_season < t_persist <= 1.0:
        raise ValueError("hydroperiod thresholds require 0 <= t_season < t_persist <= 1")
    if not np.all(np.isfinite(values[inside])):
        raise ValueError("frequency must be finite inside the domain")

    t_persist_pct = t_persist * 100.0
    t_season_pct = t_season * 100.0

    classes = np.zeros(values.shape, dtype=np.uint8)
    classes[inside & (values < t_season_pct)] = HYDROPERIOD_MARGINAL
    classes[inside & (values >= t_season_pct) & (values <= t_persist_pct)] = HYDROPERIOD_SEASONAL
    classes[inside & (values > t_persist_pct)] = HYDROPERIOD_PERSISTENT

    if unobserved_mask is not None:
        unobserved = np.asarray(unobserved_mask, dtype=bool)
        if unobserved.shape != values.shape:
            raise ValueError("frequency and unobserved_mask must share shape")
        if np.any(unobserved & inside):
            raise ValueError("unobserved_mask must not overlap the observed domain")
        classes[unobserved] = HYDROPERIOD_UNOBSERVED

    return HydroperiodResult(
        classes=classes,
        method="wofs_multiyear",
        t_persist=float(t_persist),
        t_season=float(t_season),
    )


__all__ = ["HydroperiodResult", "classify_hydroperiod"]
