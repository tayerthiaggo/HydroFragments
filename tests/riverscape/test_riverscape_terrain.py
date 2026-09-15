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
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point

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


def _y(row: int) -> float:
    """Pixel-center y for ``row`` under ``_TRANSFORM``."""
    return 300.0 - _PIXEL_M * (row + 0.5)


def test_build_rem_handles_multilinestring_reach_without_crashing() -> None:
    # hydrofragments.spatial.context.create_channel_context clips drainage
    # to the AOI via geometry.intersection(aoi_geometry), which routinely
    # turns a reach crossing the AOI boundary into a MultiLineString --
    # explicitly listed as valid drainage geometry
    # (hydrofragments/spatial/context.py's _LINE_TYPES). Before the fix,
    # shapely.ops.substring() in _sample_bin_elevations raised
    # GeometryTypeError on a MultiLineString; this fixture uses two
    # disjoint segments (no shared endpoint, so linemerge cannot stitch
    # them back into one LineString) to exercise the genuinely-disjoint
    # fallback path, not just the mergeable one.
    dem = _linear_dem(shape=(20, 10), slope_per_row=1.0, base=100.0)
    part_a = LineString([(150.0, _y(0)), (150.0, _y(3))])
    part_b = LineString([(150.0, _y(10)), (150.0, _y(13))])  # disjoint gap: rows 4-9
    geometry = MultiLineString([part_a, part_b])

    drainage = gpd.GeoDataFrame(
        [_reach(1, geometry, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )

    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=4,
        rem_max_distance_m=500.0, trough_radius_m=60.0,
    )

    # Must not have crashed, and must produce finite REM at the reach's
    # own location (both parts).
    assert np.isfinite(result.rem[2, 5])
    assert np.isfinite(result.rem[12, 5])


def test_multilinestring_disjoint_parts_processed_in_original_geometry_order() -> None:
    # Regression test for the reordering bug: _linear_parts used to fall
    # back to `list(merged.geoms)` for a genuinely-disjoint MultiLineString
    # (parts that don't share an endpoint, so linemerge can't stitch them
    # into one LineString -- the routine shape of an AOI-clipped reach with
    # a hole in the middle). GEOS's LineMerge does NOT preserve input part
    # order, so `merged.geoms` can silently REVERSE (or otherwise reorder)
    # the along-stream sequence versus the original `geometry.geoms` order
    # (which IS preserved, since this geometry -- like a real
    # `intersection()` clip result -- was built with part_a upstream,
    # part_b downstream).
    #
    # Confirmed empirically for this exact fixture: `list(linemerge(geom
    # ).geoms)` returns part_b (the downstream part) FIRST and part_a
    # (upstream) SECOND -- the reverse of `geometry.geoms` order.
    #
    # part_a (rows 0-3) sits upstream on a monotonically-declining DEM
    # (higher elevation); part_b (rows 10-13, disjoint gap at rows 4-9,
    # mimicking an AOI hole) sits downstream (lower elevation). Correctly
    # ordered, the sampled elevation sequence is already non-increasing
    # (~196 down to ~175) so PAVA leaves it essentially unchanged. If the
    # parts were fed to PAVA in the WRONG (reversed) order, the sequence
    # would look like it INCREASES downstream, which a non-increasing PAVA
    # constraint pools into a single flat block at the overall mean
    # (~186.17) -- destroying the real ~21 m upstream/downstream decline.
    dem = _linear_dem(shape=(20, 10), slope_per_row=2.0, base=200.0)
    part_a = LineString([(150.0, _y(0)), (150.0, _y(3))])
    part_b = LineString([(150.0, _y(10)), (150.0, _y(13))])  # disjoint gap: rows 4-9
    geometry = MultiLineString([part_a, part_b])

    drainage = gpd.GeoDataFrame(
        [_reach(1, geometry, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )

    # rem_k=1 (pure nearest-neighbor) makes the reconstructed profile at
    # each pixel exactly equal to its nearest profile point's regressed
    # elevation -- no blending across neighbors -- so the expected values
    # below are exact, not approximate.
    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=1,
        rem_max_distance_m=500.0, trough_radius_m=60.0,
    )

    profile_upstream = float(dem[0, 5] - result.rem[0, 5])
    profile_downstream = float(dem[13, 5] - result.rem[13, 5])

    # Correctly ordered: profile_upstream == 196.0, profile_downstream ==
    # 175.0 (measured exactly, matching the unperturbed sampled elevations
    # since the true along-stream sequence is already non-increasing).
    # Reordered (the bug): both collapse to the pooled mean 186.1667, so
    # profile_upstream == profile_downstream and this assertion fails.
    assert profile_upstream == pytest.approx(196.0)
    assert profile_downstream == pytest.approx(175.0)
    assert profile_upstream - profile_downstream > 15.0


def test_build_rem_handles_linemerge_mergeable_multilinestring() -> None:
    # A MultiLineString whose two parts DO share an endpoint -- the common
    # case from an AOI-boundary clip splitting one continuous reach into
    # two touching pieces. linemerge should stitch these back into a
    # single LineString, preserving one continuous along-stream axis.
    dem = _linear_dem(shape=(20, 10), slope_per_row=1.0, base=100.0)
    part_a = LineString([(150.0, _y(0)), (150.0, _y(6))])
    part_b = LineString([(150.0, _y(6)), (150.0, _y(13))])  # shares (150, _y(6)) with part_a
    geometry = MultiLineString([part_a, part_b])

    drainage = gpd.GeoDataFrame(
        [_reach(1, geometry, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )

    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=4,
        rem_max_distance_m=500.0, trough_radius_m=60.0,
    )

    assert np.isfinite(result.rem[2, 5])
    assert np.isfinite(result.rem[12, 5])


def test_sample_buffer_uses_full_corridor_width_not_half() -> None:
    # Regression test for the radius/diameter fix: build_rem samples each
    # reach's elevation bins with `buffer_m=max(width_m, pixel_m)` (the
    # corridor width used directly as a buffer radius), NOT
    # `max(width_m / 2.0, pixel_m)`. Every OTHER fixture in this file is
    # laterally uniform (same elevation across all columns), so the two
    # formulas sample identical pixel values and a mutant reverting to
    # `/ 2.0` survives unnoticed. This fixture varies elevation ACROSS
    # COLUMNS (a near band at cols 4-6 = 100.0, a far band everywhere else
    # = 900.0) so a buffer_m of 80.0 (correct, corridor_widths_m={"1": 80.0})
    # reaches the far band at row 1's bin, while a buffer_m of 40.0 (the
    # `/ 2.0` mutant) does not.
    #
    # The reach line runs from (165, 300) to (165, 0) -- x=165 is column
    # 5's pixel center, and the line's start at y=300 (the grid's top edge,
    # not row 0's pixel center) makes each 30 m bin's midpoint land exactly
    # on a pixel row center, so profile points coincide exactly with pixel
    # centers and rem_k=1 gives an exact (not blended) nearest-neighbor
    # reconstruction.
    dem = np.full((10, 10), 900.0, dtype="float32")
    dem[:, 4:7] = 100.0  # near band directly under/around the line

    line = LineString([(165.0, 300.0), (165.0, 0.0)])
    drainage = gpd.GeoDataFrame(
        [_reach(1, line, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )

    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0, corridor_widths_m={"1": 80.0}, rem_k=1,
        rem_max_distance_m=500.0, trough_radius_m=60.0,
    )

    # Measured with the real (unmutated) code: REM[1, 5] == -600.0 (a
    # buffer_m of 80.0 reaches column 3's far-band pixel at this bin,
    # pulling the regressed profile well above the local DEM value of
    # 100.0). Under the `/ 2.0` mutant (buffer_m == 40.0, verified by
    # calling build_rem with corridor_widths_m={"1": 40.0} -- which
    # reproduces max(80.0 / 2.0, pixel_m) exactly), REM[1, 5] == 0.0: the
    # far band is never reached, so the profile exactly matches the local
    # DEM and the lateral contamination this test is checking for vanishes.
    assert result.rem[1, 5] == pytest.approx(-600.0)


def test_build_rem_degrades_gracefully_on_empty_linestring_reach() -> None:
    # hydrofragments.spatial.context's AOI clip does
    # ``reach.intersection(aoi)``, which returns an empty LineString for a
    # reach wholly outside the AOI. Before the fix, _linear_parts' is_empty/
    # Point guard sat AFTER the LineString branch, so an empty LineString
    # fell into the LineString branch and returned [<LINESTRING EMPTY>] -- a
    # non-empty list containing a degenerate geometry. Constructed directly
    # here (no actual AOI clipping needed): a normal upstream reach (1)
    # flowing into a reach (2) with an empty LineString geometry.
    #
    # reach 2's own elevations.size == 0 short-circuits (via `continue`)
    # before build_rem's per-reach direction-check block ever calls
    # _line_endpoints on reach 2's OWN geometry -- so this crash needs
    # reach 2 to be the NextDownID target of reach 1: while processing
    # reach 1 (upstream, headwater, processed first in topological order),
    # build_rem calls ``_line_endpoints(downstream_geometry)`` where
    # downstream_geometry is reach 2's empty LineString. That call
    # delegates to _linear_parts, which (pre-fix) returned
    # [<LINESTRING EMPTY>] -- a non-empty list -- bypassing
    # _line_endpoints' "no parts" guard and crashing with IndexError on
    # ``parts[0].coords[0]`` (an empty LineString has no coordinates).
    dem = _linear_dem()
    normal_line = LineString([(0.0, 285.0), (0.0, 15.0)])
    drainage = gpd.GeoDataFrame(
        [
            _reach(1, normal_line, next_down=2, from_node=1, to_node=2),
            _reach(2, LineString(), next_down=-1, from_node=2, to_node=3),
        ],
        crs="EPSG:3577",
    )

    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0, corridor_widths_m={"1": 90.0, "2": 90.0}, rem_k=4,
        rem_max_distance_m=500.0, trough_radius_m=60.0,
    )

    # The empty-geometry reach must degrade via the same "no usable
    # samples" path as a reach with zero DEM coverage, not raise.
    assert "reach_2_no_dem_samples" in result.degraded_reasons
    # The normal upstream reach must still be processed and contribute a
    # finite REM.
    assert np.any(np.isfinite(result.rem))


def test_build_rem_handles_geometrycollection_reach_with_point_and_linestring() -> None:
    # A multi-part AOI (unary_union of AOI polygons often yields a
    # MultiPolygon) where a reach crosses one part and only touches another
    # at a single point produces geometry.intersection(aoi) as a
    # GeometryCollection containing both a LineString and a Point. Before
    # the fix, this hit _linear_parts' final
    # `raise ValueError(f"unsupported reach geometry type: ...")` branch --
    # a live crash (is_empty is False and .length is positive, so it passes
    # create_channel_context's existing filters and reaches build_rem).
    # The Point member must be dropped (via the Point branch when
    # _linear_parts recurses into it) and only the LineString member used,
    # producing output identical to a plain-LineString reach.
    dem = _linear_dem()
    line = LineString([(0.0, 285.0), (0.0, 15.0)])
    point = Point(9999.0, 9999.0)  # far outside the DEM/reach; must be dropped, not sampled

    drainage_plain = gpd.GeoDataFrame(
        [_reach(1, line, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )
    drainage_collection = gpd.GeoDataFrame(
        [_reach(1, GeometryCollection([line, point]), next_down=-1, from_node=1, to_node=2)],
        crs="EPSG:3577",
    )
    kwargs = dict(
        transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0, profile_percentile=50.0,
        corridor_widths_m={"1": 90.0}, rem_k=4, rem_max_distance_m=500.0, trough_radius_m=60.0,
    )

    result_plain = build_rem(drainage_plain, dem, **kwargs)
    result_collection = build_rem(drainage_collection, dem, **kwargs)

    assert np.allclose(result_plain.rem, result_collection.rem, equal_nan=True)
    assert result_plain.degraded_reasons == result_collection.degraded_reasons


def test_rejects_missing_corridor_widths_entry() -> None:
    # Mirrors build_centreline's identical contract
    # (test_rejects_missing_corridor_width_entry in
    # test_riverscape_centreline.py): a reach's HydroID missing from
    # corridor_widths_m must raise, not silently fall back to
    # profile_bin_m (a bin LENGTH, not a corridor WIDTH).
    line = LineString([(0.0, 285.0), (0.0, 15.0)])
    drainage = gpd.GeoDataFrame(
        [_reach(1, line, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )
    with pytest.raises(ValueError, match="corridor_widths_m"):
        build_rem(
            drainage, _linear_dem(), transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
            profile_percentile=50.0, corridor_widths_m={}, rem_k=4,
            rem_max_distance_m=500.0, trough_radius_m=60.0,
        )


def test_pava_enforces_non_increasing_profile_downstream() -> None:
    # Discrimination test for PAVA's monotone direction: build_rem calls
    # isotonic_regression(elevations, increasing=False). Flipping that to
    # increasing=True leaves every other test in this file green (they
    # only check bounds), because it silently collapses a genuinely
    # decreasing profile into one flat pooled block (the isotonic solution
    # under a non-decreasing constraint for a strictly decreasing input is
    # a single block equal to the overall mean) -- this test's assertion on
    # the SIZE of the downstream decrease is what catches that: a flat
    # (mutated) profile has ~0 decrease per step, while the correct
    # non-increasing regression tracks the DEM's real ~8 m/step decline.
    dem = _linear_dem(shape=(20, 10), slope_per_row=2.0, base=200.0)
    line = LineString([(150.0, _y(0)), (150.0, _y(19))])
    drainage = gpd.GeoDataFrame(
        [_reach(1, line, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )

    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=4,
        rem_max_distance_m=500.0, trough_radius_m=60.0,
    )

    # Reconstruct the regressed along-stream profile at the reach's column
    # (col 5) by subtracting REM back out of the DEM, sampled at rows well
    # inside the reach (away from the k-NN interpolation smoothing that can
    # occur right at the ends).
    col = 5
    rows_to_check = [2, 6, 10, 14, 18]
    profile = dem[rows_to_check, col] - result.rem[rows_to_check, col]
    diffs = np.diff(profile)

    # Correct (increasing=False): profile tracks the DEM's ~8 m/step
    # decline downstream. Mutated (increasing=True): profile collapses
    # to one flat pooled value, diffs ~0. A mean-per-step decrease of at
    # least 3 m clears that gap with margin.
    assert np.mean(diffs) < -3.0, f"profile not decreasing meaningfully downstream: {profile}"
    assert np.all(diffs <= 0.0), f"profile increased somewhere downstream: {profile}"


def test_trough_depth_and_slope_deg_are_finite_correct_shape_and_responsive() -> None:
    # Neither trough_depth nor slope_deg has any assertion elsewhere in
    # this file.
    line = LineString([(0.0, 285.0), (0.0, 15.0)])
    drainage = gpd.GeoDataFrame(
        [_reach(1, line, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )
    common_kwargs = dict(
        transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0, profile_percentile=50.0,
        corridor_widths_m={"1": 90.0}, rem_k=4, rem_max_distance_m=500.0, trough_radius_m=60.0,
    )

    # A perfectly flat DEM: trough_depth (local_mean - dem) and slope_deg
    # (from np.gradient) should both be ~0 everywhere.
    flat = np.full((10, 10), 100.0, dtype="float32")
    result_flat = build_rem(drainage, flat, **common_kwargs)

    assert result_flat.trough_depth.shape == flat.shape
    assert result_flat.slope_deg.shape == flat.shape
    assert np.all(np.isfinite(result_flat.trough_depth))
    assert np.all(np.isfinite(result_flat.slope_deg))
    assert np.allclose(result_flat.trough_depth, 0.0, atol=1e-2)
    assert np.allclose(result_flat.slope_deg, 0.0, atol=1e-2)

    # An artificial depression at the centre: trough_depth there must be
    # meaningfully nonzero (local mean pulled above the pit).
    depressed = flat.copy()
    depressed[5, 5] -= 20.0
    result_pit = build_rem(drainage, depressed, **common_kwargs)
    assert result_pit.trough_depth[5, 5] > 1.0

    # A DEM with a known constant gradient: slope_deg must match
    # atan(gradient) in degrees.
    gradient = 0.5  # metres of rise per metre of run
    rows = np.arange(10, dtype="float32")[:, None]
    graded = (100.0 - rows * gradient * _PIXEL_M) * np.ones((10, 10), dtype="float32")
    result_grad = build_rem(drainage, graded, **common_kwargs)
    expected_slope_deg = np.degrees(np.arctan(gradient))
    assert result_grad.slope_deg[2:-2, 2:-2] == pytest.approx(expected_slope_deg, abs=1e-2)


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
