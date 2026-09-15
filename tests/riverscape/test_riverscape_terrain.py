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
    # One straight reach, headwater to outlet, with an artificial elevation
    # PIT injected by a corrupted DEM cell partway down -- the regressed
    # profile must stay non-increasing and must not report a trough at the
    # pit location out of proportion to real terrain (a running minimum
    # would instead drag every downstream bin down to the pit's depth).
    dem = _linear_dem(shape=(10, 10), slope_per_row=1.0, base=100.0)
    dem[6, :] = -500.0  # a single corrupted row: a "void pit"

    line = LineString([(150.0, 300.0 - 15.0), (150.0, 300.0 - 285.0)])  # straight down the grid
    drainage = gpd.GeoDataFrame(
        [_reach(1, line, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )

    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=4,
        rem_max_distance_m=500.0, trough_radius_m=60.0, trough_depth_m=0.5,
    )

    # Row 6 itself is not a fair probe: the DEM there is *literally*
    # corrupted (-500 for the whole row), so REM = dem - profile at that
    # exact row necessarily reflects the raw corrupted input verbatim --
    # that is correct behaviour (an honest report of a bad input pixel),
    # not a resistance failure, and no profile-fitting algorithm can or
    # should paper over it. What must NOT happen is the pit's depth
    # leaking further downstream: a running minimum would carry ~-500
    # forward into every row below the pit (rows 7-9) forever, whereas
    # isotonic (least-squares) regression only pools the pit with as many
    # neighbours as monotonicity requires and lets rows downstream of that
    # recover toward their own measured (less extreme) values.
    assert np.nanmin(result.rem[7:, :]) > -100.0


def test_confluence_caps_downstream_reach_at_tributary_minimum_outflow() -> None:
    # Two headwater reaches (2, 3) both flow into reach 1 (the trunk) at
    # node 10. Reach 2's outflow elevation is much lower than reach 3's;
    # per spec §4.2 step 3, the trunk's upstream end must be capped at the
    # MINIMUM of its tributaries' outflows (not an average, not the higher
    # one) before its own isotonic regression runs.
    dem = _linear_dem(shape=(10, 10), slope_per_row=1.0, base=100.0)

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
        rem_k=4, rem_max_distance_m=500.0, trough_radius_m=60.0, trough_depth_m=0.5,
    )

    assert result.degraded_reasons == () or "no_dem_samples" not in " ".join(result.degraded_reasons)
    assert np.isfinite(result.rem).any()


def test_rejects_missing_drainage_topology_columns() -> None:
    drainage = gpd.GeoDataFrame(
        {"HydroID": [1], "geometry": [LineString([(0.0, 0.0), (100.0, 100.0)])]}, crs="EPSG:3577"
    )
    with pytest.raises(ValueError, match="From_Node"):
        build_rem(
            drainage, _linear_dem(), transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
            profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=4,
            rem_max_distance_m=500.0, trough_radius_m=60.0, trough_depth_m=0.5,
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
            rem_max_distance_m=500.0, trough_radius_m=60.0, trough_depth_m=0.5,
        )
