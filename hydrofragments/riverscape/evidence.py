"""Per-pixel evidence for riverscape observed-channel rules."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class BareCalibration:
    threshold_pct: float
    candidate_median_pct: float | None
    background_median_pct: float | None
    candidate_pixels: int
    background_pixels: int
    degraded_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RiverscapeEvidence:
    water_high: np.ndarray
    waterbody_riverine: np.ndarray
    bare_stable: np.ndarray
    terrain: np.ndarray
    connected: np.ndarray
    line_fallback: np.ndarray
    bare_fraction: np.ndarray
    bare_valid_years: np.ndarray
    confidence: np.ndarray
    calibration: BareCalibration
    degraded_reasons: tuple[str, ...]


def _year_stack(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim != 3:
        raise ValueError("bare_yearly must be 2-D or 3-D (time, y, x)")
    return array


def calibrate_bare_threshold(
    bare_yearly: np.ndarray,
    candidate_mask: np.ndarray,
    background_mask: np.ndarray,
    *,
    floor_pct: float,
    min_pixels: int = 200,
) -> BareCalibration:
    yearly = _year_stack(bare_yearly)
    candidate = np.asarray(candidate_mask, bool)
    background = np.asarray(background_mask, bool)
    if candidate.shape != yearly.shape[1:] or background.shape != candidate.shape:
        raise ValueError("bare_yearly and calibration masks must share shape")
    if not 0.0 <= floor_pct <= 100.0:
        raise ValueError("floor_pct must be in [0, 100]")
    if min_pixels < 1:
        raise ValueError("min_pixels must be positive")
    with np.errstate(all="ignore"):
        per_pixel = np.nanmedian(yearly, axis=0)
    candidate_values = per_pixel[candidate & np.isfinite(per_pixel)]
    background_values = per_pixel[background & np.isfinite(per_pixel)]
    candidate_median = (
        float(np.median(candidate_values)) if candidate_values.size else None
    )
    background_median = (
        float(np.median(background_values)) if background_values.size else None
    )
    usable = (
        candidate_values.size >= min_pixels
        and background_values.size >= min_pixels
        and candidate_median is not None
        and background_median is not None
        and candidate_median > background_median
    )
    threshold = (
        max(floor_pct, (candidate_median + background_median) / 2.0)
        if usable
        else floor_pct
    )
    return BareCalibration(
        threshold_pct=float(threshold),
        candidate_median_pct=candidate_median,
        background_median_pct=background_median,
        candidate_pixels=int(candidate_values.size),
        background_pixels=int(background_values.size),
        degraded_reasons=() if usable else ("bare_threshold_fallback",),
    )


def build_evidence(
    frequency: np.ndarray,
    bare_yearly: np.ndarray,
    riverine_waterbody_mask: np.ndarray,
    rem: np.ndarray,
    trough_depth: np.ndarray,
    domain: np.ndarray,
    centreline: np.ndarray,
    corridor_mask: np.ndarray,
    *,
    f_chan_high: float,
    bare_threshold_floor_pct: float,
    bare_year_fraction: float,
    h_chan_m: float,
    trough_depth_m: float,
    min_calibration_pixels: int = 200,
    line_fallback_mask: np.ndarray | None = None,
) -> RiverscapeEvidence:
    frequency = np.asarray(frequency, float)
    yearly = _year_stack(bare_yearly)
    arrays = [
        np.asarray(riverine_waterbody_mask, bool),
        np.asarray(rem, float),
        np.asarray(trough_depth, float),
        np.asarray(domain, bool),
        np.asarray(centreline, bool),
        np.asarray(corridor_mask, bool),
    ]
    if frequency.ndim != 2 or any(a.shape != frequency.shape for a in arrays):
        raise ValueError("all 2-D evidence arrays must share shape")
    if yearly.shape[1:] != frequency.shape:
        raise ValueError("bare_yearly and 2-D evidence arrays must share shape")
    if not 0.0 <= f_chan_high <= 1.0:
        raise ValueError("f_chan_high must be in [0, 1]")
    if not 0.0 <= bare_year_fraction <= 1.0:
        raise ValueError("bare_year_fraction must be in [0, 1]")

    waterbody, rem, trough, domain, centreline, corridor = arrays
    fallback = (
        np.zeros(frequency.shape, bool)
        if line_fallback_mask is None
        else np.asarray(line_fallback_mask, bool)
    )
    if fallback.shape != frequency.shape:
        raise ValueError("line_fallback_mask must share shape")

    water_high = np.isfinite(frequency) & (frequency >= 100.0 * f_chan_high)
    terrain = (
        (np.isfinite(rem) & (rem <= h_chan_m))
        | (np.isfinite(trough) & (trough >= trough_depth_m))
    )
    support = domain & corridor
    labels, _ = ndimage.label(support, structure=np.ones((3, 3), int))
    touching = np.unique(labels[centreline & support])
    touching = touching[touching != 0]
    connected = np.isin(labels, touching)

    calibration = calibrate_bare_threshold(
        yearly,
        connected & (water_high | terrain),
        (~corridor) & np.isfinite(yearly).any(axis=0),
        floor_pct=bare_threshold_floor_pct,
        min_pixels=min_calibration_pixels,
    )
    valid_years = np.isfinite(yearly).sum(axis=0).astype(np.uint16)
    exceed = np.isfinite(yearly) & (yearly >= calibration.threshold_pct)
    bare_fraction = np.divide(
        exceed.sum(axis=0),
        valid_years,
        out=np.zeros(frequency.shape),
        where=valid_years > 0,
    )
    bare_stable = (valid_years >= 2) & (
        bare_fraction >= bare_year_fraction
    )
    topology_family = connected | fallback
    water_family = water_high | waterbody
    geomorphic_family = bare_stable | terrain
    confidence = (
        topology_family.astype(np.uint8)
        + water_family.astype(np.uint8)
        + geomorphic_family.astype(np.uint8)
    )
    return RiverscapeEvidence(
        water_high=water_high,
        waterbody_riverine=waterbody,
        bare_stable=bare_stable,
        terrain=terrain,
        connected=connected,
        line_fallback=fallback,
        bare_fraction=bare_fraction,
        bare_valid_years=valid_years,
        confidence=confidence,
        calibration=calibration,
        degraded_reasons=calibration.degraded_reasons,
    )


__all__ = [
    "BareCalibration",
    "RiverscapeEvidence",
    "build_evidence",
    "calibrate_bare_threshold",
]
