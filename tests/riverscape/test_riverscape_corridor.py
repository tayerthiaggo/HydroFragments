"""Tests for hydrofragments.riverscape.corridor.measure_corridor_widths."""
from __future__ import annotations

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from affine import Affine
from shapely.geometry import LineString

from hydrofragments.riverscape.corridor import measure_corridor_widths

_TRANSFORM = Affine(30.0, 0.0, 0.0, 0.0, -30.0, 300.0)  # 30 m pixels, origin top-left
_PIXEL_M = 30.0


def _drainage(line: LineString, hydro_id: int = 1) -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame({"HydroID": [hydro_id]}, geometry=[line], crs="EPSG:3577")


def test_corridor_width_reflects_offset_and_water_half_width() -> None:
    # 10x10 grid, 30 m pixels: rows 0-9 map to y in [300, 0), cols 0-9 to x in [0, 300).
    water = np.zeros((10, 10), dtype=bool)
    water[5, 2:8] = True  # a wide (6-pixel) water band along row 5

    # AHGF line offset 3 pixels (90 m) north of the water band, same columns.
    # Pixel-center y for row r under _TRANSFORM is 300 - 30*(r + 0.5); row 2
    # centers at y=225 (row 5's water band centers at y=75).
    line = LineString([(75.0, 225.0), (225.0, 225.0)])
    drainage = _drainage(line)

    result = measure_corridor_widths(
        drainage, water, transform=_TRANSFORM, pixel_m=_PIXEL_M, f_seed=0.05,
        corridor_min_m=90.0, corridor_max_m=1200.0, alignment_quantile=0.95,
    )

    assert result.degraded_reasons == ()
    width = result.widths_m["1"]
    # Offset is exactly 3 px (90 m): the line sits directly above the
    # 1-pixel-thick water band, same columns, so every line pixel's
    # nearest-skeleton distance is exactly 3 px. The water band is only
    # 1 pixel thick (not 3), so its half-width is 1 px (30 m) -- giving a
    # deterministic width of 90 + 30 = 120 m for this fixture.
    assert width == pytest.approx(120.0)
    assert result.p95_offset_m["1"] == pytest.approx(90.0)


def test_reach_with_no_water_seed_gets_max_corridor_and_is_flagged_degraded() -> None:
    water = np.zeros((10, 10), dtype=bool)
    line = LineString([(60.0, 150.0), (240.0, 150.0)])
    drainage = _drainage(line)

    result = measure_corridor_widths(
        drainage, water, transform=_TRANSFORM, pixel_m=_PIXEL_M, f_seed=0.05,
        corridor_min_m=90.0, corridor_max_m=1200.0,
    )

    assert result.widths_m["1"] == 1200.0
    assert "no_water_seed_pixels" in result.degraded_reasons


def test_corridor_width_never_exceeds_configured_max() -> None:
    water = np.zeros((30, 30), dtype=bool)
    water[0, :] = True  # water far from the line -> large raw offset
    # Pixel-center y for row r is 300 - 30*(r + 0.5): row 0 -> 285, row 29 -> -585.
    line = LineString([(465.0, 285.0), (465.0, -585.0)])
    drainage = _drainage(line)

    result = measure_corridor_widths(
        drainage, water, transform=_TRANSFORM, pixel_m=_PIXEL_M, f_seed=0.05,
        corridor_min_m=90.0, corridor_max_m=300.0,
    )

    assert result.widths_m["1"] == 300.0
    # The raw p95_offset_m + max_half_width_m sum for this fixture is far
    # beyond 300 m (water is many pixels away from the line), so the clamp
    # actually reduced the value -- this must be recorded, distinguishing
    # "measured exactly 300 m" from "measured much more, clamped to 300 m".
    assert "reach_1_corridor_clamped_max" in result.degraded_reasons


def test_corridor_width_naturally_below_max_is_not_flagged_clamped() -> None:
    # Same fixture as test_corridor_width_reflects_offset_and_water_half_width
    # (deterministic width 120.0), but with a generous ceiling the raw sum
    # never approaches -- the clamp must NOT fire just because a ceiling
    # exists; only an actual reduction should be flagged.
    water = np.zeros((10, 10), dtype=bool)
    water[5, 2:8] = True
    line = LineString([(75.0, 225.0), (225.0, 225.0)])
    drainage = _drainage(line)

    result = measure_corridor_widths(
        drainage, water, transform=_TRANSFORM, pixel_m=_PIXEL_M, f_seed=0.05,
        corridor_min_m=90.0, corridor_max_m=1200.0, alignment_quantile=0.95,
    )

    assert result.widths_m["1"] == pytest.approx(120.0)
    assert result.degraded_reasons == ()


def test_corridor_width_exactly_at_ceiling_is_not_flagged_clamped() -> None:
    # Same deterministic-120.0 fixture as
    # test_corridor_width_reflects_offset_and_water_half_width, but with
    # corridor_max_m set to EXACTLY 120.0 -- the raw measured width sits
    # AT the ceiling, not beyond it, so this must NOT be flagged clamped.
    # A mutant that changes the clamp condition from `raw_width_m >
    # corridor_max_m` to `width_m >= corridor_max_m` would incorrectly flag
    # this case (width_m == corridor_max_m == 120.0 satisfies `>=`), so this
    # test discriminates the two conditions where the other clamp tests
    # (whose raw values are far from the ceiling) cannot.
    water = np.zeros((10, 10), dtype=bool)
    water[5, 2:8] = True
    line = LineString([(75.0, 225.0), (225.0, 225.0)])
    drainage = _drainage(line)

    result = measure_corridor_widths(
        drainage, water, transform=_TRANSFORM, pixel_m=_PIXEL_M, f_seed=0.05,
        corridor_min_m=90.0, corridor_max_m=120.0, alignment_quantile=0.95,
    )

    assert result.widths_m["1"] == pytest.approx(120.0)
    assert result.degraded_reasons == ()


def test_rejects_missing_hydro_id_column() -> None:
    drainage = gpd.GeoDataFrame(
        {"geometry": [LineString([(0.0, 0.0), (100.0, 100.0)])]}, crs="EPSG:3577"
    )
    with pytest.raises(ValueError, match="HydroID"):
        measure_corridor_widths(
            drainage, np.zeros((10, 10), dtype=bool), transform=_TRANSFORM, pixel_m=_PIXEL_M,
            f_seed=0.05, corridor_min_m=90.0, corridor_max_m=1200.0,
        )


def test_rejects_inverted_corridor_bounds() -> None:
    drainage = _drainage(LineString([(0.0, 0.0), (100.0, 100.0)]))
    with pytest.raises(ValueError, match="corridor_min_m"):
        measure_corridor_widths(
            drainage, np.zeros((10, 10), dtype=bool), transform=_TRANSFORM, pixel_m=_PIXEL_M,
            f_seed=0.05, corridor_min_m=600.0, corridor_max_m=90.0,
        )
