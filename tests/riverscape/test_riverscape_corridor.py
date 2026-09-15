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
    # Offset is ~3 px (90 m); water half-width is up to 3 px (90 m) at the
    # band's centre. Width should exceed the raw offset (accounts for water
    # half-width too) and stay within the configured ceiling.
    assert 90.0 < width <= 1200.0
    assert result.p95_offset_m["1"] > 0.0


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
