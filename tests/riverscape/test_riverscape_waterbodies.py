from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

gpd = pytest.importorskip("geopandas")

from hydrofragments.riverscape.waterbodies import classify_waterbodies

TRANSFORM = Affine(30, 0, 0, 0, -30, 300)


def test_touching_tile_fragments_merge() -> None:
    polygons = gpd.GeoDataFrame(
        {"source": [0, 1]},
        geometry=[box(30, 120, 120, 180), box(120, 120, 240, 180)],
        crs="EPSG:3577",
    )
    centreline = np.zeros((10, 10), bool)
    centreline[4:6, 1:8] = True

    result = classify_waterbodies(
        polygons, centreline, transform=TRANSFORM, pixel_m=30.0
    )

    assert len(result.polygons) == 1
    assert int(result.polygons.iloc[0].waterbody_id) == 1


def test_two_contact_elongated_body_is_riverine() -> None:
    polygons = gpd.GeoDataFrame(
        geometry=[box(30, 120, 270, 180)], crs="EPSG:3577"
    )
    centreline = np.zeros((10, 10), bool)
    centreline[4, 0:10] = True

    result = classify_waterbodies(
        polygons, centreline, transform=TRANSFORM, pixel_m=30.0
    )

    row = result.polygons.iloc[0]
    assert row.role == "riverine"
    assert row.boundary_contacts >= 2
    assert 0.5 <= row.centreline_span_fraction <= 1.0
    assert result.riverine_mask.any()
    assert not result.off_channel_mask.any()


def test_one_ended_billabong_branch_is_off_channel() -> None:
    polygons = gpd.GeoDataFrame(
        geometry=[box(120, 30, 180, 210)], crs="EPSG:3577"
    )
    centreline = np.zeros((10, 10), bool)
    centreline[3:9, 5] = True
    centreline[3, 0:6] = True  # branch enters body once and terminates

    result = classify_waterbodies(
        polygons, centreline, transform=TRANSFORM, pixel_m=30.0
    )

    assert result.polygons.iloc[0].role == "off_channel"
    assert result.off_channel_mask.any()
    assert not np.any(result.riverine_mask & result.off_channel_mask)


def test_compact_body_fails_elongation_even_when_crossed() -> None:
    polygons = gpd.GeoDataFrame(
        geometry=[box(90, 90, 210, 210)], crs="EPSG:3577"
    )
    centreline = np.zeros((10, 10), bool)
    centreline[5, 0:10] = True

    result = classify_waterbodies(
        polygons, centreline, transform=TRANSFORM, pixel_m=30.0
    )

    assert result.polygons.iloc[0].elongation < 2.0
    assert result.polygons.iloc[0].role == "off_channel"


def test_empty_input_returns_typed_empty_result() -> None:
    polygons = gpd.GeoDataFrame(geometry=[], crs="EPSG:3577")
    result = classify_waterbodies(
        polygons, np.zeros((4, 4), bool), transform=TRANSFORM, pixel_m=30.0
    )
    assert result.polygons.empty
    assert result.riverine_mask.shape == (4, 4)
