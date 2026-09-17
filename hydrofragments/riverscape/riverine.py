"""Floodplain envelope calibration and riverine vs non-riverine landform."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from hydrofragments.riverscape.codes import (
    LANDFORM_IN_CHANNEL,
    LANDFORM_NON_RIVERINE,
    LANDFORM_OFF_CHANNEL_RIVERINE,
    LANDFORM_OUTSIDE,
)

RIVERINE_RULESET_VERSION = "1.0.0"
LOG_BIN_WIDTH = 0.5
_STRUCTURE_8 = np.ones((3, 3), bool)


@dataclass(frozen=True)
class FloodplainEnvelope:
    a: float
    b: float
    n_bins: int
    bin_log_a_mid: tuple[float, ...]
    bin_rem_quantile: tuple[float, ...]
    bin_counts: tuple[int, ...]
    degraded_reasons: tuple[str, ...]


@dataclass(frozen=True)
class LandformLayerResult:
    landform: np.ndarray
    envelope: FloodplainEnvelope
    degraded_reasons: tuple[str, ...]


def _require_aligned(*arrays: np.ndarray) -> tuple[int, ...]:
    if not arrays:
        raise ValueError("expected at least one array")
    shape = np.asarray(arrays[0]).shape
    if len(shape) != 2:
        raise ValueError("arrays must be 2-D")
    for array in arrays[1:]:
        if np.asarray(array).shape != shape:
            raise ValueError("arrays must share shape")
    return shape


def _channel_connected_domain(
    domain: np.ndarray, channel: np.ndarray
) -> np.ndarray:
    seed = np.asarray(channel, bool) & np.asarray(domain, bool)
    return ndimage.binary_propagation(
        seed, structure=_STRUCTURE_8, mask=np.asarray(domain, bool)
    )


def _nearest_reach_labels(
    channel: np.ndarray, reach_labels: np.ndarray
) -> np.ndarray:
    channel = np.asarray(channel, bool)
    labels = np.asarray(reach_labels)
    if not channel.any():
        return np.zeros(channel.shape, labels.dtype)
    _, indices = ndimage.distance_transform_edt(~channel, return_indices=True)
    return labels[indices[0], indices[1]]


def _lookup_area(
    reach_id: np.ndarray, upstr_darea: Mapping[int, float]
) -> np.ndarray:
    area = np.full(reach_id.shape, np.nan, float)
    for key, value in upstr_darea.items():
        area[reach_id == int(key)] = float(value)
    return area


def _h_fp(
    area: np.ndarray,
    envelope: FloodplainEnvelope,
    envelope_h_max_m: float,
) -> np.ndarray:
    valid = np.isfinite(area) & (area > 0)
    height = np.full(area.shape, np.nan, float)
    height[valid] = envelope.a * np.power(area[valid], envelope.b)
    height[valid] = np.minimum(height[valid], envelope_h_max_m)
    return height


def fit_floodplain_envelope(
    rem: np.ndarray,
    domain: np.ndarray,
    channel: np.ndarray,
    reach_labels: np.ndarray,
    upstr_darea: Mapping[int, float],
    *,
    envelope_quantile: float,
    envelope_min_bin_pixels: int,
) -> FloodplainEnvelope:
    rem = np.asarray(rem, float)
    domain = np.asarray(domain, bool)
    channel = np.asarray(channel, bool)
    reach_labels = np.asarray(reach_labels)
    _require_aligned(rem, domain, channel, reach_labels)
    if not 0.0 < envelope_quantile < 1.0:
        raise ValueError("envelope_quantile must be in (0, 1)")
    if envelope_min_bin_pixels < 1:
        raise ValueError("envelope_min_bin_pixels must be a positive integer")

    connected = _channel_connected_domain(domain, channel)
    nearest = _nearest_reach_labels(channel, reach_labels)
    area = _lookup_area(nearest, upstr_darea)
    sample = (
        connected
        & np.isfinite(rem)
        & (nearest != 0)
        & np.isfinite(area)
        & (area > 0)
    )
    if not sample.any():
        return FloodplainEnvelope(
            a=0.0,
            b=0.0,
            n_bins=0,
            bin_log_a_mid=(),
            bin_rem_quantile=(),
            bin_counts=(),
            degraded_reasons=("envelope_no_bins",),
        )

    log_a = np.log10(area[sample])
    rem_sample = rem[sample]
    if log_a.min() == log_a.max():
        centre = float(log_a.min())
        edges = np.asarray([centre - LOG_BIN_WIDTH / 2, centre + LOG_BIN_WIDTH / 2])
    else:
        lo = np.floor(log_a.min() / LOG_BIN_WIDTH) * LOG_BIN_WIDTH
        hi = float(log_a.max())
        edges = np.arange(lo, hi + LOG_BIN_WIDTH, LOG_BIN_WIDTH)
        if edges[-1] < hi:
            edges = np.append(edges, edges[-1] + LOG_BIN_WIDTH)

    mids: list[float] = []
    quantiles: list[float] = []
    counts: list[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (log_a >= left) & (log_a < right)
        if right == edges[-1]:
            in_bin = (log_a >= left) & (log_a <= right)
        count = int(in_bin.sum())
        if count < envelope_min_bin_pixels:
            continue
        q = float(np.nanpercentile(rem_sample[in_bin], 100.0 * envelope_quantile))
        if not np.isfinite(q) or q <= 0.0:
            continue
        mids.append(float((left + right) / 2.0))
        quantiles.append(q)
        counts.append(count)

    if len(mids) == 0:
        return FloodplainEnvelope(
            a=0.0,
            b=0.0,
            n_bins=0,
            bin_log_a_mid=(),
            bin_rem_quantile=(),
            bin_counts=(),
            degraded_reasons=("envelope_no_bins",),
        )

    if len(mids) == 1:
        return FloodplainEnvelope(
            a=float(quantiles[0]),
            b=0.0,
            n_bins=1,
            bin_log_a_mid=tuple(mids),
            bin_rem_quantile=tuple(quantiles),
            bin_counts=tuple(counts),
            degraded_reasons=("envelope_single_bin",),
        )

    a_mid = np.power(10.0, np.asarray(mids, float))
    q = np.asarray(quantiles, float)
    slope, intercept = np.polyfit(np.log(a_mid), np.log(q), 1)
    return FloodplainEnvelope(
        a=float(np.exp(intercept)),
        b=float(slope),
        n_bins=len(mids),
        bin_log_a_mid=tuple(mids),
        bin_rem_quantile=tuple(quantiles),
        bin_counts=tuple(counts),
        degraded_reasons=(),
    )


def classify_off_channel(
    rem: np.ndarray,
    slope_deg: np.ndarray,
    domain: np.ndarray,
    channel: np.ndarray,
    reach_labels: np.ndarray,
    upstr_darea: Mapping[int, float],
    envelope: FloodplainEnvelope,
    *,
    slope_max_deg: float,
    envelope_h_max_m: float,
) -> np.ndarray:
    rem = np.asarray(rem, float)
    slope_deg = np.asarray(slope_deg, float)
    domain = np.asarray(domain, bool)
    channel = np.asarray(channel, bool)
    reach_labels = np.asarray(reach_labels)
    _require_aligned(rem, slope_deg, domain, channel, reach_labels)
    if not np.isfinite(slope_max_deg) or slope_max_deg <= 0:
        raise ValueError("slope_max_deg must be finite and positive")
    if not np.isfinite(envelope_h_max_m) or envelope_h_max_m <= 0:
        raise ValueError("envelope_h_max_m must be finite and positive")

    nearest = _nearest_reach_labels(channel, reach_labels)
    area = _lookup_area(nearest, upstr_darea)
    height = _h_fp(area, envelope, envelope_h_max_m)
    return (
        domain
        & ~channel
        & np.isfinite(rem)
        & np.isfinite(slope_deg)
        & np.isfinite(height)
        & (rem <= height)
        & (slope_deg <= slope_max_deg)
    )


def assemble_landform(
    domain: np.ndarray,
    channel: np.ndarray,
    off_channel_riverine: np.ndarray,
) -> np.ndarray:
    domain = np.asarray(domain, bool)
    channel = np.asarray(channel, bool)
    off_channel = np.asarray(off_channel_riverine, bool)
    _require_aligned(domain, channel, off_channel)
    landform = np.full(domain.shape, LANDFORM_OUTSIDE, np.uint8)
    off = domain & ~channel & off_channel
    non = domain & ~channel & ~off_channel
    landform[non] = LANDFORM_NON_RIVERINE
    landform[off] = LANDFORM_OFF_CHANNEL_RIVERINE
    landform[domain & channel] = LANDFORM_IN_CHANNEL
    return landform


def build_landform_layer(
    rem: np.ndarray,
    slope_deg: np.ndarray,
    domain: np.ndarray,
    channel: np.ndarray,
    reach_labels: np.ndarray,
    upstr_darea: Mapping[int, float],
    *,
    envelope_quantile: float,
    envelope_min_bin_pixels: int,
    envelope_h_max_m: float,
    slope_max_deg: float,
) -> LandformLayerResult:
    envelope = fit_floodplain_envelope(
        rem,
        domain,
        channel,
        reach_labels,
        upstr_darea,
        envelope_quantile=envelope_quantile,
        envelope_min_bin_pixels=envelope_min_bin_pixels,
    )
    off_channel = classify_off_channel(
        rem,
        slope_deg,
        domain,
        channel,
        reach_labels,
        upstr_darea,
        envelope,
        slope_max_deg=slope_max_deg,
        envelope_h_max_m=envelope_h_max_m,
    )
    landform = assemble_landform(domain, channel, off_channel)
    return LandformLayerResult(
        landform=landform,
        envelope=envelope,
        degraded_reasons=envelope.degraded_reasons,
    )


__all__ = [
    "FloodplainEnvelope",
    "LOG_BIN_WIDTH",
    "LandformLayerResult",
    "RIVERINE_RULESET_VERSION",
    "assemble_landform",
    "build_landform_layer",
    "classify_off_channel",
    "fit_floodplain_envelope",
]
