from __future__ import annotations

import numpy as np
import pytest

from hydrofragments.riverscape.evidence import (
    build_evidence,
    calibrate_bare_threshold,
)


def test_calibration_uses_midpoint_when_channel_is_barer() -> None:
    yearly = np.array(
        [
            [[60, 60, 20, 20], [60, 60, 20, 20]],
            [[70, 70, 30, 30], [70, 70, 30, 30]],
        ],
        dtype=float,
    )
    candidate = np.zeros((2, 4), bool)
    candidate[:, :2] = True
    background = ~candidate

    result = calibrate_bare_threshold(
        yearly, candidate, background, floor_pct=30.0, min_pixels=4
    )

    assert result.candidate_median_pct == 65.0
    assert result.background_median_pct == 25.0
    assert result.threshold_pct == 45.0
    assert result.degraded_reasons == ()


def test_single_year_bare_emits_no_stability_degradation() -> None:
    shape = (2, 4)
    yearly = np.array([[60, 60, 20, 20], [60, 60, 20, 20]], float)
    corridor_mask = np.zeros(shape, bool)
    corridor_mask[:, :2] = True
    centreline = np.zeros(shape, bool)
    centreline[:, 0] = True
    result = build_evidence(
        frequency=np.full(shape, 20.0),
        bare_yearly=yearly,
        riverine_waterbody_mask=np.zeros(shape, bool),
        rem=np.zeros(shape),
        trough_depth=np.zeros(shape),
        domain=np.ones(shape, bool),
        centreline=centreline,
        corridor_mask=corridor_mask,
        f_chan_high=0.1,
        bare_threshold_floor_pct=30.0,
        bare_year_fraction=0.6,
        h_chan_m=2.0,
        trough_depth_m=0.5,
        min_calibration_pixels=4,
    )

    assert "bare_single_year_no_stability" in result.degraded_reasons
    assert result.calibration.degraded_reasons == ()
    assert result.calibration.candidate_median_pct == 60.0
    assert result.calibration.background_median_pct == 20.0
    assert not result.bare_stable.any()


def test_calibration_falls_back_when_contrast_reverses() -> None:
    yearly = np.array([[[10, 10, 50, 50]], [[20, 20, 60, 60]]], float)
    candidate = np.array([[True, True, False, False]])
    background = ~candidate

    result = calibrate_bare_threshold(
        yearly, candidate, background, floor_pct=30.0, min_pixels=2
    )

    assert result.threshold_pct == 30.0
    assert result.degraded_reasons == ("bare_threshold_fallback",)


def test_b_requires_fraction_boundary_and_two_valid_years() -> None:
    shape = (1, 3)
    bare = np.array(
        [
            [[30.0, 40.0, 30.0]],
            [[30.0, np.nan, 20.0]],
            [[20.0, np.nan, 30.0]],
            [[20.0, np.nan, 20.0]],
            [[30.0, np.nan, 30.0]],
        ]
    )
    result = build_evidence(
        frequency=np.full(shape, 20.0),
        bare_yearly=bare,
        riverine_waterbody_mask=np.zeros(shape, bool),
        rem=np.zeros(shape),
        trough_depth=np.zeros(shape),
        domain=np.ones(shape, bool),
        centreline=np.ones(shape, bool),
        corridor_mask=np.ones(shape, bool),
        f_chan_high=0.1,
        bare_threshold_floor_pct=30.0,
        bare_year_fraction=0.6,
        h_chan_m=2.0,
        trough_depth_m=0.5,
        min_calibration_pixels=99,
    )

    assert result.bare_fraction.tolist() == [[0.6, 1.0, 0.6]]
    assert result.bare_valid_years.tolist() == [[5, 1, 5]]
    assert result.bare_stable.tolist() == [[True, False, True]]


def test_confidence_counts_w_and_s_once() -> None:
    shape = (3, 3)
    result = build_evidence(
        frequency=np.full(shape, 20.0),
        bare_yearly=np.full((2, *shape), 10.0),
        riverine_waterbody_mask=np.ones(shape, bool),
        rem=np.full(shape, 10.0),
        trough_depth=np.zeros(shape),
        domain=np.ones(shape, bool),
        centreline=np.zeros(shape, bool),
        corridor_mask=np.ones(shape, bool),
        line_fallback_mask=np.ones(shape, bool),
        f_chan_high=0.1,
        bare_threshold_floor_pct=30.0,
        bare_year_fraction=0.6,
        h_chan_m=2.0,
        trough_depth_m=0.5,
        min_calibration_pixels=99,
    )

    assert np.all(result.confidence == 2)  # topology + one water family


def test_connected_requires_domain_component_touching_centreline() -> None:
    domain = np.zeros((5, 7), bool)
    domain[2, 0:3] = True
    domain[2, 4:7] = True
    centreline = np.zeros_like(domain)
    centreline[2, 0] = True

    result = build_evidence(
        frequency=np.full(domain.shape, 20.0),
        bare_yearly=np.full((2, *domain.shape), 10.0),
        riverine_waterbody_mask=np.zeros_like(domain),
        rem=np.full(domain.shape, 10.0),
        trough_depth=np.zeros(domain.shape),
        domain=domain,
        centreline=centreline,
        corridor_mask=np.ones_like(domain),
        f_chan_high=0.1,
        bare_threshold_floor_pct=30.0,
        bare_year_fraction=0.6,
        h_chan_m=2.0,
        trough_depth_m=0.5,
    )

    assert result.connected[2, 0:3].all()
    assert not result.connected[2, 4:7].any()


def test_water_high_converts_fraction_threshold_to_percent_once() -> None:
    shape = (1, 2)
    result = build_evidence(
        frequency=np.array([[9.9, 10.0]]),
        bare_yearly=np.full((2, *shape), 10.0),
        riverine_waterbody_mask=np.zeros(shape, bool),
        rem=np.full(shape, 10.0),
        trough_depth=np.zeros(shape),
        domain=np.ones(shape, bool),
        centreline=np.ones(shape, bool),
        corridor_mask=np.ones(shape, bool),
        f_chan_high=0.10,
        bare_threshold_floor_pct=30.0,
        bare_year_fraction=0.6,
        h_chan_m=2.0,
        trough_depth_m=0.5,
    )
    assert result.water_high.tolist() == [[False, True]]


def test_build_evidence_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share shape"):
        build_evidence(
            frequency=np.zeros((2, 2)),
            bare_yearly=np.zeros((2, 3, 3)),
            riverine_waterbody_mask=np.zeros((2, 2), bool),
            rem=np.zeros((2, 2)),
            trough_depth=np.zeros((2, 2)),
            domain=np.zeros((2, 2), bool),
            centreline=np.zeros((2, 2), bool),
            corridor_mask=np.zeros((2, 2), bool),
            f_chan_high=0.1,
            bare_threshold_floor_pct=30.0,
            bare_year_fraction=0.6,
            h_chan_m=2.0,
            trough_depth_m=0.5,
        )
