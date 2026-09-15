"""Tests for hydrofragments.riverscape.centreline.build_centreline."""
from __future__ import annotations

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from affine import Affine
from shapely.geometry import LineString

from hydrofragments.riverscape.centreline import build_centreline

_TRANSFORM = Affine(30.0, 0.0, 0.0, 0.0, -30.0, 300.0)


def _drainage(line: LineString, hydro_id: int = 1) -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame({"HydroID": [hydro_id]}, geometry=[line], crs="EPSG:3577")


def test_offset_line_still_conflates_onto_the_true_water_skeleton() -> None:
    # Water band along row 5 (a straight, thick channel); AHGF line offset
    # 90 m (3 px) north of it, per spec's "AHGF line offset 90 m from the
    # true channel still yields the EO centreline" acceptance test.
    water = np.zeros((10, 10), dtype=bool)
    water[5, 1:9] = True
    # Pixel-center y for row r under _TRANSFORM is 300 - 30*(r + 0.5); row 2
    # centers at y=225.
    line = LineString([(45.0, 225.0), (255.0, 225.0)])
    drainage = _drainage(line)

    result = build_centreline(
        drainage, water, {"1": 300.0}, transform=_TRANSFORM, pixel_m=30.0
    )

    assert result.line_fallback_reaches == ()
    # The conflated skeleton must sit ON the water band (row 5), never on
    # the offset line's own row (row 2).
    assert result.skeleton[5, :].any()
    assert not result.skeleton[2, :].any()


def test_reach_with_no_water_in_its_corridor_is_line_fallback() -> None:
    water = np.zeros((10, 10), dtype=bool)  # no water anywhere
    line = LineString([(30.0, 150.0), (270.0, 150.0)])
    drainage = _drainage(line)

    result = build_centreline(
        drainage, water, {"1": 90.0}, transform=_TRANSFORM, pixel_m=30.0
    )

    assert result.line_fallback_reaches == ("1",)
    assert not result.skeleton.any()


def test_looped_water_body_is_flagged_multithread() -> None:
    # A ring of water (a simple anabranch loop) fully inside a generous corridor.
    water = np.zeros((12, 12), dtype=bool)
    water[3:9, 3] = True
    water[3:9, 8] = True
    water[3, 3:9] = True
    water[8, 3:9] = True
    line = LineString([(30.0, 30.0), (300.0, 300.0)])
    drainage = _drainage(line)

    result = build_centreline(
        drainage, water, {"1": 600.0}, transform=_TRANSFORM, pixel_m=30.0
    )

    assert result.multithread_reaches == ("1",)


def test_rejects_missing_corridor_width_entry() -> None:
    drainage = _drainage(LineString([(0.0, 0.0), (100.0, 100.0)]))
    with pytest.raises(ValueError, match="corridor_widths_m"):
        build_centreline(
            drainage, np.zeros((10, 10), dtype=bool), {}, transform=_TRANSFORM, pixel_m=30.0
        )
