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
    assert result.multithread_reaches == ()
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
    # Flagged multithread but still contributes its pixels to the combined
    # skeleton, per the spec's "flagged but still contributes" contract.
    assert result.skeleton.any()


def test_corridor_restricts_which_water_contributes_to_the_skeleton() -> None:
    # Water in two separate bands: row 2 (near) and row 7 (far). The
    # reach's line runs along row 2 -- pixel-center y = 300 - 30*(2+0.5) =
    # 225 -- with a narrow 45 m corridor: wide enough to reach row 2's
    # water (the line sits inside that row's pixel extent, y in
    # [210, 240]) but far short of row 7's water (y in [60, 90], 135-165 m
    # away), so the corridor intersection actually excludes something.
    water = np.zeros((10, 10), dtype=bool)
    water[2, 1:9] = True
    water[7, 1:9] = True
    line = LineString([(45.0, 225.0), (255.0, 225.0)])
    drainage = _drainage(line)

    result = build_centreline(
        drainage, water, {"1": 45.0}, transform=_TRANSFORM, pixel_m=30.0
    )

    assert result.multithread_reaches == ()
    assert result.skeleton[2, :].any()
    assert not result.skeleton[7, :].any()


def test_sinuous_single_thread_channel_is_not_flagged_multithread() -> None:
    # A single 90-degree channel bend: a vertical arm (rows 1-7, cols 2-4)
    # meeting a horizontal arm (rows 5-7, cols 2-8) -- an ordinary channel
    # corner, not an enclosed loop. Its medial-axis skeleton contains an
    # L-tromino corner motif (three mutually 8-adjacent pixels), which is
    # exactly the shape the old Euler-characteristic cycle check misread
    # as a 3-cycle (verified: reverting _has_loop to that check flags this
    # reach multithread). The corridor-restricted water mask has no
    # enclosed hole, so the current implementation must not flag it.
    water = np.zeros((10, 10), dtype=bool)
    water[1:8, 2:5] = True
    water[5:8, 2:9] = True
    line = LineString([(30.0, 150.0), (270.0, 150.0)])
    drainage = _drainage(line)

    result = build_centreline(
        drainage, water, {"1": 300.0}, transform=_TRANSFORM, pixel_m=30.0
    )

    assert result.multithread_reaches == ()
    assert result.skeleton.any()


def test_reach_whose_own_corridor_excludes_all_nearby_water_is_line_fallback() -> None:
    # Water exists on the grid (row 2) -- unlike the "no water anywhere"
    # fallback test above -- but this reach's own line+corridor sits far
    # away (row 7, pixel-center y = 75) with a narrow 30 m corridor that
    # never reaches row 2's water 150 m away, so this reach's restricted
    # mask is empty even though the grid isn't.
    water = np.zeros((10, 10), dtype=bool)
    water[2, 1:9] = True
    line = LineString([(45.0, 75.0), (255.0, 75.0)])
    drainage = _drainage(line)

    result = build_centreline(
        drainage, water, {"1": 30.0}, transform=_TRANSFORM, pixel_m=30.0
    )

    assert result.line_fallback_reaches == ("1",)


def test_rejects_missing_corridor_width_entry() -> None:
    drainage = _drainage(LineString([(0.0, 0.0), (100.0, 100.0)]))
    with pytest.raises(ValueError, match="corridor_widths_m"):
        build_centreline(
            drainage, np.zeros((10, 10), dtype=bool), {}, transform=_TRANSFORM, pixel_m=30.0
        )
