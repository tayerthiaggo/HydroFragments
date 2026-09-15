"""Tests for hydrofragments.riverscape.terrain.build_rem.

Covers spec §10's Phase 3 acceptance: "anabranch" (a confluence correctly
caps the downstream reach's profile at its tributaries' minimum outflow)
and "pit" (isotonic regression resists a single-bin void pit, matching the
design rationale in spec §4.2 step 3: "A running minimum is rejected
because it propagates single void pits downstream").
"""
from __future__ import annotations

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from affine import Affine
from shapely.geometry import LineString

from hydrofragments.riverscape.terrain import build_rem

_PIXEL_M = 30.0
_TRANSFORM = Affine(_PIXEL_M, 0.0, 0.0, 0.0, -_PIXEL_M, 300.0)


def _reach(hydro_id, line, next_down, *, from_node, to_node):
    return {
        "HydroID": hydro_id, "From_Node": from_node, "To_Node": to_node,
        "NextDownID": next_down, "geometry": line,
    }


def _linear_dem(shape=(10, 10), *, slope_per_row=1.0, base=100.0) -> np.ndarray:
    """DEM decreasing by ``slope_per_row`` metres per row (downhill going down the grid)."""
    rows = np.arange(shape[0], dtype="float32")[:, None]
    return (base - rows * slope_per_row) * np.ones(shape, dtype="float32")


def test_isotonic_regression_resists_a_single_bin_void_pit() -> None:
    # 30-row DEM (bigger than the original 10-row fixture) so the pit at
    # row 10 has enough downstream length to show attenuation. profile_percentile=10.0
    # (RiverscapeConfig's production default, not this test's original 50.0) is required
    # for the pit to actually survive per-bin sampling and reach isotonic regression --
    # at percentile 50 the median over the sampling buffer already launders a single
    # corrupted row before PAVA ever sees it, which is why an earlier version of this
    # test could not distinguish PAVA from a naive running minimum (mutation-tested:
    # swapping isotonic_regression for np.minimum.accumulate left the old test green).
    dem = _linear_dem(shape=(30, 10), slope_per_row=1.0, base=100.0)
    dem[10, :] = -500.0  # a single corrupted row: a "void pit"

    line = LineString([(150.0, 300.0 - 15.0), (150.0, 300.0 - 30 * 30.0 + 15.0)])
    drainage = gpd.GeoDataFrame(
        [_reach(1, line, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )

    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=10.0, corridor_widths_m={"1": 90.0}, rem_k=4,
        rem_max_distance_m=500.0, trough_radius_m=60.0,
    )

    # PAVA pools the pit into one non-increasing block with its neighbors,
    # attenuating it to the pooled block's mean (a large but bounded value,
    # not -500 propagated forever); it cannot increase values once pooled
    # (monotonicity forbids it), so downstream bins do not "recover toward
    # their own measured values" -- they stay attenuated but bounded. A
    # running minimum instead propagates -500 to every downstream bin
    # forever. Rows strictly downstream of the pit (rows 14+, comfortably
    # past the pooled block) must stay well below the running-minimum's
    # ~586 REM, but the PAVA design keeps them near ~103-118.
    assert np.nanmax(result.rem[14:, :]) < 200.0


def test_confluence_caps_downstream_reach_at_tributary_minimum_outflow() -> None:
    # Two headwater reaches (2, 3) both flow into reach 1 (the trunk) at
    # node 10. The DEM is asymmetric between the two tributaries' columns
    # (not just row-varying, unlike the original fixture, whose mirror-image
    # tributary geometry over a row-only DEM made both tributaries produce
    # IDENTICAL outflow values -- unable to distinguish min from max from
    # mean; mutation-tested by a reviewer, confirmed all three left the old
    # test green) so their outflow elevations genuinely differ, letting this
    # test actually verify the cap uses the MINIMUM.
    dem = _linear_dem(shape=(10, 10), slope_per_row=1.0, base=100.0)
    dem[:, :5] -= 20.0  # lower the left half (trib_low's columns) by 20m

    trunk = LineString([(150.0, 300.0 - 165.0), (150.0, 300.0 - 285.0)])
    trib_low = LineString([(60.0, 300.0 - 15.0), (150.0, 300.0 - 165.0)])
    trib_high = LineString([(240.0, 300.0 - 15.0), (150.0, 300.0 - 165.0)])

    drainage = gpd.GeoDataFrame(
        [
            _reach(1, trunk, next_down=-1, from_node=10, to_node=20),
            _reach(2, trib_low, next_down=1, from_node=11, to_node=10),
            _reach(3, trib_high, next_down=1, from_node=12, to_node=10),
        ],
        crs="EPSG:3577",
    )

    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0,
        corridor_widths_m={"1": 90.0, "2": 90.0, "3": 90.0},
        rem_k=4, rem_max_distance_m=500.0, trough_radius_m=60.0,
    )

    # trib_low's outflow (~77) is lower than trib_high's (~94); the trunk's
    # upstream end must be capped at the MINIMUM of the two, not their
    # average or the higher one. A wrong cap rule (max or mean) produces a
    # visibly different trunk REM at the column nearest the confluence.
    assert np.nanmax(result.rem[6:, 5]) > 12.0

    # All three reaches here are digitised headwater-to-outlet (their last
    # coordinate is the downstream end), so none should be flagged as
    # having a suspect direction -- this guards against the direction
    # check false-positiving on normally-oriented topology.
    assert not any("geometry_direction_suspect" in reason for reason in result.degraded_reasons)


def test_reversed_reach_is_flagged_geometry_direction_suspect() -> None:
    # Same confluence fixture as test_confluence_caps_downstream_reach_at_-
    # tributary_minimum_outflow, but trib_low's endpoints are swapped so its
    # LAST coordinate (assumed downstream per the module's docstring) is the
    # upstream headwater end instead of the confluence node -- its first
    # coordinate now sits closer to the downstream reach (the trunk) than
    # its last coordinate does, which is exactly the condition build_rem
    # flags as a suspect digitisation direction.
    dem = _linear_dem(shape=(10, 10), slope_per_row=1.0, base=100.0)
    dem[:, :5] -= 20.0

    trunk = LineString([(150.0, 300.0 - 165.0), (150.0, 300.0 - 285.0)])
    trib_low_reversed = LineString([(150.0, 300.0 - 165.0), (60.0, 300.0 - 15.0)])
    trib_high = LineString([(240.0, 300.0 - 15.0), (150.0, 300.0 - 165.0)])

    drainage = gpd.GeoDataFrame(
        [
            _reach(1, trunk, next_down=-1, from_node=10, to_node=20),
            _reach(2, trib_low_reversed, next_down=1, from_node=11, to_node=10),
            _reach(3, trib_high, next_down=1, from_node=12, to_node=10),
        ],
        crs="EPSG:3577",
    )

    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0,
        corridor_widths_m={"1": 90.0, "2": 90.0, "3": 90.0},
        rem_k=4, rem_max_distance_m=500.0, trough_radius_m=60.0,
    )

    assert "reach_2_geometry_direction_suspect" in result.degraded_reasons


def test_rejects_missing_drainage_topology_columns() -> None:
    drainage = gpd.GeoDataFrame(
        {"HydroID": [1], "geometry": [LineString([(0.0, 0.0), (100.0, 100.0)])]}, crs="EPSG:3577"
    )
    with pytest.raises(ValueError, match="From_Node"):
        build_rem(
            drainage, _linear_dem(), transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
            profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=4,
            rem_max_distance_m=500.0, trough_radius_m=60.0,
        )


def test_rejects_non_positive_rem_k() -> None:
    line = LineString([(0.0, 285.0), (0.0, 15.0)])
    drainage = gpd.GeoDataFrame(
        [_reach(1, line, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )
    with pytest.raises(ValueError, match="rem_k"):
        build_rem(
            drainage, _linear_dem(), transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
            profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=0,
            rem_max_distance_m=500.0, trough_radius_m=60.0,
        )


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("pixel_m", 0.0, "pixel_m"),
        ("pixel_m", float("nan"), "pixel_m"),
        ("profile_bin_m", -5.0, "profile_bin_m"),
        ("profile_percentile", 150.0, "profile_percentile"),
        ("profile_percentile", -1.0, "profile_percentile"),
        ("rem_max_distance_m", 0.0, "rem_max_distance_m"),
        ("trough_radius_m", -1.0, "trough_radius_m"),
    ],
)
def test_rejects_invalid_numeric_parameters(field, value, match) -> None:
    line = LineString([(0.0, 285.0), (0.0, 15.0)])
    drainage = gpd.GeoDataFrame(
        [_reach(1, line, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )
    kwargs = dict(
        transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=4,
        rem_max_distance_m=500.0, trough_radius_m=60.0,
    )
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        build_rem(drainage, _linear_dem(), **kwargs)
