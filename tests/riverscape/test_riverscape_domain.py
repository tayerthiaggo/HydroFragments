from __future__ import annotations

import inspect

import numpy as np
import pytest

from hydrofragments.io.dea import WoStatistics
from hydrofragments.riverscape import codes as landform_codes
from hydrofragments.riverscape.domain import wet_domain
from hydrofragments.hydroperiod import codes as hydroperiod_codes
from hydrofragments.spatial.zones import build_zones


def _stats(frequency, count_wet, count_clear) -> WoStatistics:
    return WoStatistics(
        frequency=frequency,
        count_wet=count_wet,
        count_clear=count_clear,
        product="ga_ls_wo_fq_myear_3",
        version="test",
        crs="EPSG:3577",
        time_span=None,
        provenance={},
    )


def test_landform_codes_are_the_spec_contract() -> None:
    assert landform_codes.LANDFORM_OUTSIDE == 0
    assert landform_codes.LANDFORM_IN_CHANNEL == 1
    assert landform_codes.LANDFORM_OFF_CHANNEL_RIVERINE == 2
    assert landform_codes.LANDFORM_NON_RIVERINE == 3
    assert landform_codes.LANDFORM_CODES == frozenset({0, 1, 2, 3})
    assert landform_codes.LANDFORM_NAMES == {
        1: "in_channel",
        2: "off_channel_riverine",
        3: "non_riverine",
    }


def test_hydroperiod_codes_are_the_spec_contract() -> None:
    assert hydroperiod_codes.HYDROPERIOD_OUTSIDE == 0
    assert hydroperiod_codes.HYDROPERIOD_PERSISTENT == 1
    assert hydroperiod_codes.HYDROPERIOD_SEASONAL == 2
    assert hydroperiod_codes.HYDROPERIOD_MARGINAL == 3
    assert hydroperiod_codes.HYDROPERIOD_UNOBSERVED == 4
    assert hydroperiod_codes.HYDROPERIOD_CODES == frozenset({0, 1, 2, 3, 4})
    assert hydroperiod_codes.HYDROPERIOD_NAMES == {
        1: "persistent",
        2: "seasonal",
        3: "marginal",
        4: "unobserved",
    }


def test_domain_requires_wet_finite_and_support() -> None:
    stats = _stats(
        frequency=np.array([[50.0, 50.0, np.nan, 50.0]]),
        count_wet=np.array([[5, 0, 5, 5]]),
        count_clear=np.array([[20, 20, 20, 19]]),
    )

    domain = wet_domain(stats, min_valid_obs=20)

    assert domain.dtype == bool
    assert domain.tolist() == [[True, False, False, False]]


def test_domain_matches_build_zones_valid_extent() -> None:
    rng = np.random.default_rng(20260914)
    frequency = rng.uniform(0.0, 100.0, size=(40, 30))
    frequency[rng.random((40, 30)) < 0.05] = np.nan
    count_wet = rng.integers(0, 4, size=(40, 30))
    count_clear = rng.integers(10, 40, size=(40, 30))
    stats = _stats(frequency, count_wet, count_clear)

    domain = wet_domain(stats, min_valid_obs=20)
    zoned = build_zones(
        frequency,
        max_wet_mask=count_wet > 0,
        valid_count=count_clear,
        min_valid_obs=20,
    ).mask > 0

    np.testing.assert_array_equal(domain, zoned)


def test_domain_rejects_shape_mismatch() -> None:
    stats = _stats(np.ones((2, 2)), np.ones((2, 3)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="share shape"):
        wet_domain(stats, min_valid_obs=1)


def test_domain_rejects_non_2d_frequency() -> None:
    stats = _stats(np.ones(4), np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="2-D"):
        wet_domain(stats, min_valid_obs=1)


def test_domain_rejects_min_valid_obs_below_one() -> None:
    stats = _stats(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError, match="min_valid_obs"):
        wet_domain(stats, min_valid_obs=0)


def test_domain_accepts_dask_backed_xarray() -> None:
    xr = pytest.importorskip("xarray")
    da = pytest.importorskip("dask.array")
    stats = _stats(
        frequency=xr.DataArray(da.from_array(np.array([[60.0, 5.0]]), chunks=(1, 1)), dims=("y", "x")),
        count_wet=xr.DataArray(da.from_array(np.array([[3, 1]]), chunks=(1, 1)), dims=("y", "x")),
        count_clear=xr.DataArray(da.from_array(np.array([[20, 20]]), chunks=(1, 1)), dims=("y", "x")),
    )

    assert wet_domain(stats, min_valid_obs=20).tolist() == [[True, True]]


def test_domain_has_no_planning_footprint_input() -> None:
    assert list(inspect.signature(wet_domain).parameters) == ["stats", "min_valid_obs"]
