"""Phase 6b export surfaces: zone names, GPKG columns, evidence rasters, bridges.

Everything here is unit-level and calls the finalize/raster helpers
directly. The end-to-end wiring is Task 4's
``tests/integration/test_riverscape_exports.py``.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
xr = pytest.importorskip("xarray")
pytest.importorskip("rasterio")
pytest.importorskip("rioxarray")
pyogrio = pytest.importorskip("pyogrio")

import rasterio
from shapely.geometry import LineString

from hydrofragments.output import finalize as finalize_module
from hydrofragments.output.finalize import (
    CHANNEL_BRIDGES_LAYER,
    SpatialProductUnavailable,
    _channel_bridges_geodataframe,
    _riverscape_evidence_arrays,
    _zone_names_for,
    _zones_geodataframe,
)
from hydrofragments.output.rasters import (
    RASTER_PRODUCT_CONTRACTS,
    RIVERSCAPE_EVIDENCE_BANDS,
    write_riverscape_evidence_geotiffs,
)
from hydrofragments.output.riverscape_export import RiverscapeExportBundle
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.riverscape.bridging import _empty_bridge_frame
from hydrofragments.riverscape.riverine import FloodplainEnvelope
from hydrofragments.riverscape.pipeline import LandformResult
from hydrofragments.spatial.zones import build_zones, combine_zones

SHAPE = (3, 4)
PIXEL_M = 30.0
CRS = "EPSG:3577"
_DEFAULT_BRIDGES = object()


def _grid() -> SpatialGrid:
    y = 120.0 - np.arange(SHAPE[0]) * PIXEL_M - PIXEL_M / 2.0
    x = np.arange(SHAPE[1]) * PIXEL_M + PIXEL_M / 2.0
    template = xr.DataArray(
        np.zeros(SHAPE, dtype=np.float32), dims=("y", "x"), coords={"y": y, "x": x}
    ).rio.write_crs(CRS)
    return SpatialGrid.from_dataarray(template, require_georeference=True)


_LANDFORM = np.array(
    [
        [1, 2, 2, 0],
        [2, 2, 3, 0],
        [1, 3, 1, 0],
    ],
    dtype=np.uint8,
)
_HYDROPERIOD = np.array(
    [
        [1, 1, 2, 0],
        [2, 3, 1, 0],
        [4, 2, 1, 0],
    ],
    dtype=np.uint8,
)


def _landform_result(*, channel_bridges, grid):
    bridged = np.zeros(SHAPE, dtype=bool)
    bridged[2, 0] = True
    return LandformResult(
        landform=_LANDFORM,
        channel_source=np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [2, 0, 1, 0]], dtype=np.uint8
        ),
        channel_confidence=np.array(
            [[3, 255, 255, 255], [255, 255, 255, 255], [2, 255, 3, 255]],
            dtype=np.uint8,
        ),
        rule_id=np.zeros(SHAPE, dtype=np.uint8),
        rem=np.array(
            [[0.0, 1.5, 2.0, np.nan], [1.0, 1.0, 3.0, np.nan], [0.0, 2.5, 0.0, np.nan]],
            dtype=np.float32,
        ),
        bridged_mask=bridged,
        channel_bridges=channel_bridges,
        unbridged=(),
        envelope=FloodplainEnvelope(
            a=2.0,
            b=0.0,
            n_bins=1,
            bin_log_a_mid=(3.7,),
            bin_rem_quantile=(2.0,),
            bin_counts=(9,),
            degraded_reasons=(),
        ),
        provenance={"years": (2019, 2021)},
        degraded_reasons=(),
        grid=grid,
    )


def _bridges(crs=CRS):
    return gpd.GeoDataFrame(
        {
            "gap_id": [0],
            "reach_ids": ["1,2"],
            "length_m": [120.0],
            "cumulative_cost": [1.0],
            "cost_per_m": [0.01],
            "upstream_width_m": [30.0],
            "downstream_width_m": [30.0],
            "bridge_confidence": [2],
            "gap_cause": ["vegetated"],
        },
        geometry=[LineString([(15.0, 15.0), (105.0, 15.0)])],
        crs=crs,
    )


def _bundle(*, channel_bridges=_DEFAULT_BRIDGES):
    grid = _grid()
    zone_result = replace(
        combine_zones(_LANDFORM, _HYDROPERIOD, source="ga_ls_wo_fq_myear_3"),
        grid=grid,
    )
    if channel_bridges is _DEFAULT_BRIDGES:
        bridges = _bridges()
    else:
        bridges = channel_bridges
    landform = _landform_result(
        channel_bridges=bridges,
        grid=grid,
    )
    return RiverscapeExportBundle.from_zoning(zone_result, landform)


def _occurrence_zone_result():
    frequency = np.array(
        [
            [80.0, 30.0, 5.0, 80.0],
            [80.0, 30.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    drainage = np.zeros(SHAPE, dtype=bool)
    drainage[0, 0] = True
    result = build_zones(
        frequency,
        max_wet_mask=frequency > 0,
        valid_count=np.full(SHAPE, 40),
        drainage_mask=drainage,
    )
    return replace(result, grid=_grid())


# --------------------------------------------------------------------------
# Mode-keyed zone names (spec 6b section 3.4)
# --------------------------------------------------------------------------


def test_zone_names_are_keyed_by_mode() -> None:
    """MUTANT: leaving ``_ZONE_NAMES`` mode-blind labels a riverscape Zone 2
    'persistent' instead of 'persistent_off_channel'."""
    assert _zone_names_for("occurrence") == {
        1: "channel_connected",
        2: "persistent",
        3: "seasonal",
        4: "ephemeral",
    }
    assert _zone_names_for("riverscape") == {
        1: "in_channel",
        2: "persistent_off_channel",
        3: "seasonal_floodplain",
        4: "marginal_floodplain",
    }


def test_unknown_mode_falls_back_to_occurrence_names() -> None:
    assert _zone_names_for("something_else")[1] == "channel_connected"


# --------------------------------------------------------------------------
# GPKG zones layer columns (spec 6b section 3.4)
# --------------------------------------------------------------------------


def test_riverscape_zones_gdf_uses_riverscape_names_and_null_science_columns() -> None:
    bundle = _bundle()

    frame = _zones_geodataframe(bundle.zone_result, pixel_size_m=PIXEL_M)

    assert list(frame.columns) == [
        "zone_id",
        "zone_name",
        "area_km2",
        "source",
        "mode",
        "landform",
        "hydroperiod",
        "geometry",
    ]
    assert set(frame["mode"]) == {"riverscape"}
    assert frame["landform"].isna().all()
    assert frame["hydroperiod"].isna().all()
    names = dict(zip(frame["zone_id"], frame["zone_name"]))
    assert names[1] == "in_channel"
    assert names[2] == "persistent_off_channel"


def test_occurrence_zones_gdf_keeps_legacy_names_and_still_has_the_columns() -> None:
    """Regression (spec 6b section 6): occurrence zone names are unchanged,
    but the schema is stable -- the new columns are present and null."""
    frame = _zones_geodataframe(_occurrence_zone_result(), pixel_size_m=PIXEL_M)

    assert set(frame["mode"]) == {"occurrence"}
    assert frame["landform"].isna().all()
    assert frame["hydroperiod"].isna().all()
    names = dict(zip(frame["zone_id"], frame["zone_name"]))
    assert names[1] == "channel_connected"
    assert names[2] == "persistent"


def test_empty_zone_mask_still_returns_the_full_column_set() -> None:
    empty = replace(
        _occurrence_zone_result(), mask=np.zeros(SHAPE, dtype=np.uint8)
    )

    frame = _zones_geodataframe(empty, pixel_size_m=PIXEL_M)

    assert len(frame) == 0
    assert set(["mode", "landform", "hydroperiod"]).issubset(frame.columns)


# --------------------------------------------------------------------------
# Evidence rasters (spec 6b section 3.5)
# --------------------------------------------------------------------------


def test_evidence_filenames_and_dtypes_match_the_locked_contract() -> None:
    expected = {
        "riverscape_landform": ("landform.tif", "uint8"),
        "riverscape_hydroperiod": ("hydroperiod.tif", "uint8"),
        "riverscape_zone_crosstab": ("zone_crosstab.tif", "uint16"),
        "riverscape_channel_source": ("channel_source.tif", "uint8"),
        "riverscape_channel_confidence": ("channel_confidence.tif", "uint8"),
        "riverscape_rem": ("rem.tif", "float32"),
    }
    assert [key for key, _band in RIVERSCAPE_EVIDENCE_BANDS] == list(expected)
    for key, (filename, dtype) in expected.items():
        contract = RASTER_PRODUCT_CONTRACTS[key]
        assert contract.filename == filename
        assert contract.dtype.name == dtype


def test_writing_evidence_produces_six_grid_aligned_rasters(tmp_path: Path) -> None:
    bundle = _bundle()
    grid = bundle.landform.grid

    written = write_riverscape_evidence_geotiffs(
        _riverscape_evidence_arrays(bundle),
        tmp_path,
        grid=grid,
        metadata={"algorithm_version": "1.0.0", "scientific_config_hash": "abc"},
    )

    assert set(written) == {key for key, _band in RIVERSCAPE_EVIDENCE_BANDS}
    for key, path in written.items():
        contract = RASTER_PRODUCT_CONTRACTS[key]
        assert path.name == contract.filename
        with rasterio.open(path) as dataset:
            assert dataset.dtypes[0] == contract.dtype.name
            assert (dataset.height, dataset.width) == SHAPE
            assert dataset.transform == grid.transform
            # Host PROJ_LIB conflicts (PostGIS vs pyproj) can make
            # to_epsg()/CRS.equals fail for a correctly written 3577 GeoTIFF.
            # Align against the same SpatialGrid CRS the writer received.
            assert dataset.crs is not None
            assert str(dataset.crs) == str(grid.crs) or "Australian Albers" in (
                dataset.crs.to_wkt() or ""
            )


def test_zone_crosstab_raster_is_uint16_and_equals_landform_times_ten_plus_hydroperiod(
    tmp_path: Path,
) -> None:
    """MUTANT: writing the crosstab as uint8, or writing ``landform`` into
    the crosstab band, fails here."""
    bundle = _bundle()

    written = write_riverscape_evidence_geotiffs(
        _riverscape_evidence_arrays(bundle),
        tmp_path,
        grid=bundle.landform.grid,
        metadata={"algorithm_version": "1.0.0", "scientific_config_hash": "abc"},
    )

    with rasterio.open(written["riverscape_zone_crosstab"]) as dataset:
        values = dataset.read(1)
        assert dataset.dtypes[0] == "uint16"
    np.testing.assert_array_equal(
        values, np.where(_LANDFORM == 0, 0, _LANDFORM * 10 + _HYDROPERIOD)
    )


def test_hydroperiod_raster_keeps_code_four_on_the_bridged_pixel(tmp_path: Path) -> None:
    bundle = _bundle()

    written = write_riverscape_evidence_geotiffs(
        _riverscape_evidence_arrays(bundle),
        tmp_path,
        grid=bundle.landform.grid,
        metadata={"algorithm_version": "1.0.0", "scientific_config_hash": "abc"},
    )

    with rasterio.open(written["riverscape_hydroperiod"]) as dataset:
        assert int(dataset.read(1)[2, 0]) == 4


def test_rem_raster_round_trips_nan_outside_the_domain(tmp_path: Path) -> None:
    bundle = _bundle()

    written = write_riverscape_evidence_geotiffs(
        _riverscape_evidence_arrays(bundle),
        tmp_path,
        grid=bundle.landform.grid,
        metadata={"algorithm_version": "1.0.0", "scientific_config_hash": "abc"},
    )

    with rasterio.open(written["riverscape_rem"]) as dataset:
        values = dataset.read(1)
    assert np.isnan(values[:, 3]).all()
    assert values[0, 0] == pytest.approx(0.0)


def test_evidence_write_refuses_an_array_off_the_grid(tmp_path: Path) -> None:
    bundle = _bundle()
    arrays = _riverscape_evidence_arrays(bundle)
    arrays["landform"] = np.zeros((2, 2), dtype=np.uint8)

    from hydrofragments.output.rasters import RasterExportError

    with pytest.raises(RasterExportError, match="landform"):
        write_riverscape_evidence_geotiffs(
            arrays,
            tmp_path,
            grid=bundle.landform.grid,
            metadata={"algorithm_version": "1.0.0", "scientific_config_hash": "abc"},
        )


# --------------------------------------------------------------------------
# Preflight gate (spec 6b sections 3.5, 5)
# --------------------------------------------------------------------------


def test_riverscape_evidence_without_a_bundle_fails_preflight(tmp_path: Path) -> None:
    """MUTANT: writing evidence without checking for the bundle would
    AttributeError deep inside the raster writer instead of failing the run
    up front (spec 6b section 5)."""
    from hydrofragments.config import HydroConfig
    from hydrofragments.models import AnalysisInputs, WaterCube
    from hydrofragments.output.finalize import preflight_spatial_outputs

    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {
                "output_dir": str(tmp_path / "run"),
                "spatial_products": ["riverscape_evidence"],
            },
        }
    )
    cube = _water_cube()

    with pytest.raises(SpatialProductUnavailable, match="riverscape_evidence requires"):
        preflight_spatial_outputs(
            config,
            cube=cube,
            inputs=AnalysisInputs(),
            hydroyear_result=None,
            zone_result=None,
            riverscape_bundle=None,
        )


def test_riverscape_evidence_with_a_gridless_bundle_fails_preflight(tmp_path: Path) -> None:
    from hydrofragments.config import HydroConfig
    from hydrofragments.models import AnalysisInputs
    from hydrofragments.output.finalize import preflight_spatial_outputs

    bundle = _bundle()
    gridless = RiverscapeExportBundle(
        landform=replace(bundle.landform, grid=None),
        hydroperiod_classes=bundle.hydroperiod_classes,
        zone_result=bundle.zone_result,
    )
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {
                "output_dir": str(tmp_path / "run"),
                "spatial_products": ["riverscape_evidence"],
            },
        }
    )

    with pytest.raises(SpatialProductUnavailable, match="georeferenced"):
        preflight_spatial_outputs(
            config,
            cube=_water_cube(),
            inputs=AnalysisInputs(),
            hydroyear_result=None,
            zone_result=bundle.zone_result,
            riverscape_bundle=gridless,
        )


def _water_cube():
    """Minimal georeferenced WaterCube for preflight only."""
    import pandas as pd

    from hydrofragments import open_water_cube

    times = pd.date_range("2020-01-01", periods=3, freq="MS")
    y = 120.0 - np.arange(SHAPE[0]) * PIXEL_M - PIXEL_M / 2.0
    x = np.arange(SHAPE[1]) * PIXEL_M + PIXEL_M / 2.0
    water = xr.DataArray(
        np.zeros((3, *SHAPE), dtype=np.uint8),
        dims=("time", "y", "x"),
        coords={"time": times, "y": y, "x": x},
    ).rio.write_crs(CRS)
    return open_water_cube(water, input_kind="generic_binary")


# --------------------------------------------------------------------------
# channel_bridges layer (spec 6b section 3.6)
# --------------------------------------------------------------------------


def test_bridges_frame_keeps_the_bridging_schema_and_order() -> None:
    bundle = _bundle()

    frame = _channel_bridges_geodataframe(bundle, crs=bundle.landform.grid.crs)

    assert list(frame.columns) == list(_empty_bridge_frame(CRS).columns)
    assert len(frame) == 1
    assert frame.iloc[0]["gap_cause"] == "vegetated"


def test_empty_bridges_produce_an_empty_but_typed_frame() -> None:
    bundle = _bundle(channel_bridges=_empty_bridge_frame(CRS))

    frame = _channel_bridges_geodataframe(bundle, crs=bundle.landform.grid.crs)

    assert len(frame) == 0
    assert list(frame.columns) == list(_empty_bridge_frame(CRS).columns)
    assert frame["length_m"].dtype == np.float64


def test_absent_bridges_frame_falls_back_to_the_empty_schema() -> None:
    bundle = _bundle(channel_bridges=None)

    frame = _channel_bridges_geodataframe(bundle, crs=bundle.landform.grid.crs)

    assert len(frame) == 0
    assert list(frame.columns) == list(_empty_bridge_frame(CRS).columns)


def test_bridges_in_a_foreign_crs_are_refused(tmp_path: Path) -> None:
    """Spec 6b section 5: prefer raising over silently reprojecting, since
    6a already projected bridges onto the analysis grid."""
    bundle = _bundle(channel_bridges=_bridges(crs="EPSG:4326"))

    with pytest.raises(SpatialProductUnavailable, match="CRS"):
        _channel_bridges_geodataframe(bundle, crs=bundle.landform.grid.crs)


def test_empty_bridges_write_an_empty_gpkg_layer(tmp_path: Path) -> None:
    bundle = _bundle(channel_bridges=_empty_bridge_frame(CRS))
    frame = _channel_bridges_geodataframe(bundle, crs=bundle.landform.grid.crs)
    path = tmp_path / "spatial.gpkg"

    pyogrio.write_dataframe(
        frame,
        path,
        layer=CHANNEL_BRIDGES_LAYER,
        driver="GPKG",
        encoding="UTF-8",
        geometry_type="LineString",
    )

    info = pyogrio.read_info(path, layer=CHANNEL_BRIDGES_LAYER)
    layer_names = pyogrio.list_layers(path)[:, 0]
    assert CHANNEL_BRIDGES_LAYER in layer_names
    assert info["features"] == 0
    assert "gap_cause" in info["fields"]
