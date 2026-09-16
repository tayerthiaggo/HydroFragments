"""Ordered, named observed-channel rules for riverscape zoning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage

from hydrofragments.riverscape.evidence import RiverscapeEvidence

RULESET_VERSION = "1.0.0"
RULE_NONE = 0
RULE_CONNECTED_WATER = 1
RULE_CONNECTED_BARE_TERRAIN = 2
RULE_ADJACENT_GEOMORPHIC = 3
RULE_LINE_FALLBACK = 4


@dataclass(frozen=True)
class ChannelResult:
    channel: np.ndarray
    confidence: np.ndarray
    rule_id: np.ndarray
    seed_half_width_m: dict[str, float]
    degraded_reasons: tuple[str, ...]


def classify_channel(
    evidence: RiverscapeEvidence,
    domain: np.ndarray,
    centreline: np.ndarray,
    water_seed: np.ndarray,
    reach_labels: np.ndarray,
    reach_keys: Mapping[int, str],
    *,
    pixel_m: float,
    width_growth_factor: float,
    min_channel_confidence: int,
    include_line_fallback_in_channel: bool = False,
) -> ChannelResult:
    domain = np.asarray(domain, bool)
    centreline = np.asarray(centreline, bool)
    water_seed = np.asarray(water_seed, bool)
    reach_labels = np.asarray(reach_labels)
    shape = domain.shape
    evidence_arrays = (
        evidence.water_high,
        evidence.waterbody_riverine,
        evidence.bare_stable,
        evidence.terrain,
        evidence.connected,
        evidence.line_fallback,
        evidence.confidence,
    )
    if domain.ndim != 2 or any(np.shape(array) != shape for array in evidence_arrays):
        raise ValueError("domain and evidence arrays must share a 2-D shape")
    if centreline.shape != shape or water_seed.shape != shape or reach_labels.shape != shape:
        raise ValueError("centreline, water_seed, reach_labels and domain must share shape")
    if pixel_m <= 0:
        raise ValueError("pixel_m must be positive")
    if width_growth_factor < 1.0:
        raise ValueError("width_growth_factor must be at least 1")
    if not 1 <= min_channel_confidence <= 3:
        raise ValueError("min_channel_confidence must be in [1, 3]")

    positive_labels = set(int(value) for value in np.unique(reach_labels) if value > 0)
    missing_keys = positive_labels - set(reach_keys)
    if missing_keys:
        raise ValueError(f"reach_keys missing labels: {sorted(missing_keys)}")

    half_width = ndimage.distance_transform_edt(water_seed) * pixel_m
    distance_to_centreline = ndimage.distance_transform_edt(~centreline) * pixel_m
    width_eligible = np.zeros(shape, bool)
    seed_widths: dict[str, float] = {}
    degraded: list[str] = list(evidence.degraded_reasons)
    for label in sorted(positive_labels):
        key = str(reach_keys[label])
        samples = half_width[
            (reach_labels == label) & centreline & water_seed
        ]
        samples = samples[np.isfinite(samples) & (samples > 0)]
        if samples.size == 0:
            degraded.append(f"reach_{key}_missing_seed_width")
            continue
        median_width = float(np.median(samples))
        seed_widths[key] = median_width
        width_eligible |= (
            (reach_labels == label)
            & (distance_to_centreline <= width_growth_factor * median_width)
        )

    confidence_ok = evidence.confidence >= min_channel_confidence
    ordinary = domain & width_eligible & confidence_ok
    water_family = evidence.water_high | evidence.waterbody_riverine
    channel = np.zeros(shape, bool)
    rule_id = np.zeros(shape, np.uint8)

    first = ordinary & evidence.connected & water_family
    channel[first] = True
    rule_id[first] = RULE_CONNECTED_WATER

    second = (
        ordinary
        & ~channel
        & evidence.connected
        & evidence.bare_stable
        & evidence.terrain
    )
    channel[second] = True
    rule_id[second] = RULE_CONNECTED_BARE_TERRAIN

    growth_candidates = (
        ordinary
        & ~channel
        & evidence.connected
        & (evidence.bare_stable | evidence.terrain)
    )
    structure = np.ones((3, 3), bool)
    while True:
        grown = growth_candidates & ndimage.binary_dilation(
            channel, structure=structure
        )
        if not grown.any():
            break
        channel[grown] = True
        rule_id[grown] = RULE_ADJACENT_GEOMORPHIC
        growth_candidates[grown] = False

    if include_line_fallback_in_channel:
        fallback = (
            domain
            & ~channel
            & evidence.line_fallback
            & confidence_ok
        )
        channel[fallback] = True
        rule_id[fallback] = RULE_LINE_FALLBACK

    return ChannelResult(
        channel=channel,
        confidence=np.asarray(evidence.confidence, np.uint8).copy(),
        rule_id=rule_id,
        seed_half_width_m=seed_widths,
        degraded_reasons=tuple(dict.fromkeys(degraded)),
    )


__all__ = [
    "ChannelResult",
    "RULESET_VERSION",
    "RULE_ADJACENT_GEOMORPHIC",
    "RULE_CONNECTED_BARE_TERRAIN",
    "RULE_CONNECTED_WATER",
    "RULE_LINE_FALLBACK",
    "RULE_NONE",
    "classify_channel",
]
