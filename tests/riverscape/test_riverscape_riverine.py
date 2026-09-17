from __future__ import annotations

import numpy as np
import pytest

from hydrofragments.riverscape.codes import (
    LANDFORM_IN_CHANNEL,
    LANDFORM_NON_RIVERINE,
    LANDFORM_OFF_CHANNEL_RIVERINE,
    LANDFORM_OUTSIDE,
)
from hydrofragments.riverscape.riverine import (
    assemble_landform,
    build_landform_layer,
    fit_floodplain_envelope,
)


def test_single_bin_sets_b_zero_and_degrades() -> None:
    shape = (20, 20)
    rem = np.full(shape, 1.0, np.float32)
    domain = np.ones(shape, bool)
    channel = np.zeros(shape, bool)
    channel[10, 5:15] = True
    reach_labels = np.zeros(shape, np.int32)
    reach_labels[channel] = 1
    envelope = fit_floodplain_envelope(
        rem,
        domain,
        channel,
        reach_labels,
        {1: 1.0e6},
        envelope_quantile=0.95,
        envelope_min_bin_pixels=10,
    )
    assert envelope.b == 0.0
    assert envelope.n_bins == 1
    assert envelope.a > 0.0
    assert "envelope_single_bin" in envelope.degraded_reasons


def test_zero_bins_fail_closed() -> None:
    shape = (8, 8)
    rem = np.full(shape, 1.0, np.float32)
    domain = np.ones(shape, bool)
    channel = np.zeros(shape, bool)
    channel[4, 2:6] = True
    reach_labels = np.zeros(shape, np.int32)
    reach_labels[channel] = 1
    envelope = fit_floodplain_envelope(
        rem,
        domain,
        channel,
        reach_labels,
        {1: 1.0e6},
        envelope_quantile=0.95,
        envelope_min_bin_pixels=10_000,
    )
    assert envelope.a == 0.0
    assert envelope.b == 0.0
    assert envelope.n_bins == 0
    assert "envelope_no_bins" in envelope.degraded_reasons


def _layer_fixture():
    """Channel with two UpstrDArea classes and distinct rem/slope pockets."""
    shape = (40, 40)
    rem = np.full(shape, 1.0, np.float32)
    slope = np.full(shape, 0.5, np.float32)
    domain = np.ones(shape, bool)
    channel = np.zeros(shape, bool)
    channel[20, 5:35] = True
    reach_labels = np.zeros(shape, np.int32)
    reach_labels[20, 5:20] = 1
    reach_labels[20, 20:35] = 2
    # Billabong: low rem, gentle slope, connected through domain to channel
    rem[10:15, 10:15] = 0.5
    # Hillslope dam: low rem would pass envelope, but steep slope rejects
    rem[2:6, 30:34] = 0.5
    slope[2:6, 30:34] = 5.0
    upstr = {1: 1.0e4, 2: 1.0e7}
    return rem, slope, domain, channel, reach_labels, upstr


def test_flat_plain_billabong_is_off_channel_riverine() -> None:
    rem, slope, domain, channel, labels, upstr = _layer_fixture()
    result = build_landform_layer(
        rem,
        slope,
        domain,
        channel,
        labels,
        upstr,
        envelope_quantile=0.95,
        envelope_min_bin_pixels=5,
        envelope_h_max_m=15.0,
        slope_max_deg=2.0,
    )
    assert result.landform[12, 12] == LANDFORM_OFF_CHANNEL_RIVERINE


def test_hillslope_dam_is_non_riverine() -> None:
    rem, slope, domain, channel, labels, upstr = _layer_fixture()
    result = build_landform_layer(
        rem,
        slope,
        domain,
        channel,
        labels,
        upstr,
        envelope_quantile=0.95,
        envelope_min_bin_pixels=5,
        envelope_h_max_m=15.0,
        slope_max_deg=2.0,
    )
    assert result.landform[4, 32] == LANDFORM_NON_RIVERINE


def test_channel_wins_over_floodplain_criteria() -> None:
    rem, slope, domain, channel, labels, upstr = _layer_fixture()
    result = build_landform_layer(
        rem,
        slope,
        domain,
        channel,
        labels,
        upstr,
        envelope_quantile=0.95,
        envelope_min_bin_pixels=5,
        envelope_h_max_m=15.0,
        slope_max_deg=2.0,
    )
    assert np.all(result.landform[channel] == LANDFORM_IN_CHANNEL)


def test_outside_domain_is_zero() -> None:
    rem, slope, domain, channel, labels, upstr = _layer_fixture()
    domain[0, 0] = False
    result = build_landform_layer(
        rem,
        slope,
        domain,
        channel,
        labels,
        upstr,
        envelope_quantile=0.95,
        envelope_min_bin_pixels=5,
        envelope_h_max_m=15.0,
        slope_max_deg=2.0,
    )
    assert result.landform[0, 0] == LANDFORM_OUTSIDE


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        assemble_landform(
            np.ones((4, 4), bool),
            np.zeros((3, 4), bool),
            np.zeros((4, 4), bool),
        )


def test_zero_bins_yield_no_landform_two_from_rem() -> None:
    shape = (10, 10)
    rem = np.full(shape, 0.1, np.float32)
    slope = np.full(shape, 0.1, np.float32)
    domain = np.ones(shape, bool)
    channel = np.zeros(shape, bool)
    channel[5, 2:8] = True
    labels = np.zeros(shape, np.int32)
    labels[channel] = 1
    result = build_landform_layer(
        rem,
        slope,
        domain,
        channel,
        labels,
        {1: 1e6},
        envelope_quantile=0.95,
        envelope_min_bin_pixels=10_000,
        envelope_h_max_m=15.0,
        slope_max_deg=2.0,
    )
    assert "envelope_no_bins" in result.degraded_reasons
    assert not np.any(result.landform == LANDFORM_OFF_CHANNEL_RIVERINE)


def test_diagonal_touch_enters_envelope_sample() -> None:
    shape = (5, 5)
    rem = np.full(shape, np.nan, np.float32)
    domain = np.zeros(shape, bool)
    channel = np.zeros(shape, bool)
    channel[2, 2] = True
    domain[2, 2] = True
    domain[1, 1] = True
    rem[1, 1] = 2.0
    labels = np.zeros(shape, np.int32)
    labels[2, 2] = 1
    envelope = fit_floodplain_envelope(
        rem,
        domain,
        channel,
        labels,
        {1: 1.0e6},
        envelope_quantile=0.95,
        envelope_min_bin_pixels=1,
    )
    assert envelope.n_bins == 1
    assert envelope.a > 0.0
    assert "envelope_no_bins" not in envelope.degraded_reasons
