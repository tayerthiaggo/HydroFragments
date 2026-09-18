"""End-to-end: a riverscape run writes zoning, evidence rasters, and bridges.

Reuses ``tests/integration/test_riverscape_modes.py``'s monkeypatch
convention -- every hydroseason entry point is patched as a module attribute
on the real ``hydroseason`` package, and the riverscape zoning call is
replaced with a stub -- so nothing here touches STAC/WFS.

Only ``riverscape_evidence`` is requested, deliberately: the ``zones``
raster branch in ``output/finalize.py`` additionally needs
``core.spatial_grid`` from ``SectionSpatialCollector``, which is an
orthogonal dependency. Zone names and the GPKG zone columns are unit-tested
in ``tests/output/test_riverscape_evidence_exports.py``.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
xr = pytest.importorskip("xarray")
pytest.importorskip("rasterio")
pytest.importorskip("rioxarray")
pyogrio = pytest.importorskip("pyogrio")

import hydroseason
import rasterio
from shapely import wkb
from shapely.geometry import LineString, box

from hydrofragments import workflow as workflow_module
from hydrofragments.config import HydroConfig
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.riverscape.pipeline import LandformResult, with_grid
from hydrofragments.riverscape.riverine import FloodplainEnvelope
from hydrofragments.spatial.zones import combine_zones
from hydrofragments.workflow import _default_config, analyze_from_dea

_SHAPE = (4, 4)
_TRANSFORM = (30.0, 0.0, 0.0, 0.0, -30.0, 120.0)
_CRS = "EPSG:3577"


def _aoi() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame({"geometry": [box(0.0, 0.0, 120.0, 120.0)]}, crs=_CRS)


def _drainage() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame(
        {
            "HydroID": [1],
            "From_Node": [1],
            "To_Node": [2],
            "NextDownID": [-1],
            "UpstrDArea": [5000.0],
        },
        geometry=[LineString([(10.0, 10.0), (100.0, 100.0)])],
        crs=_CRS,
    )


def _wo_statistics_dataset() -> "xr.Dataset":
    ny, nx = _SHAPE
    y = 120.0 - np.arange(ny) * 30.0 - 15.0
    x = np.arange(nx) * 30.0 + 15.0
    wet = np.full((ny, nx), 6, dtype=np.int16)
    clear = np.full((ny, nx), 30, dtype=np.int16)
    frequency = 100.0 * wet.astype("float32") / clear.astype("float32")
    ds = xr.Dataset(
        {
            "count_wet": (("y", "x"), wet),
            "count_clear": (("y", "x"), clear),
            "frequency": (("y", "x"), frequency),
        },
        coords={"y": y, "x": x},
    ).rio.write_crs(_CRS)
    ds.attrs["provenance"] = {
        "product": "ga_ls_wo_fq_myear_3",
        "stac_url": "https://example.test/stac",
        "item_ids": ["item-1"],
        "crs": _CRS,
        "resolution": 30.0,
        "time_span": "2020-01-01T00:00:00Z/2020-12-31T23:59:59Z",
        "frequency": {"derivation": "100 * count_wet / count_clear"},
    }
    return ds


def _stats():
    dataset = _wo_statistics_dataset()
    return SimpleNamespace(
        frequency=dataset["frequency"].rio.write_crs(_CRS),
        count_wet=dataset["count_wet"].values,
        count_clear=dataset["count_clear"].values,
        product="ga_ls_wo_fq_myear_3",
        version="0.1.0",
        crs=_CRS,
        time_span="2020-01-01T00:00:00Z/2020-12-31T23:59:59Z",
        provenance=dict(dataset.attrs["provenance"]),
    )


class _Recorder:
    def __init__(self) -> None:
        self.acquire_calls: list[dict] = []
        self.riverscape_calls: list[dict] = []


def _install_happy_path(monkeypatch, recorder: _Recorder, tmp_path: Path):
    def fake_open_wo_statistics(aoi, **kwargs):
        return _wo_statistics_dataset()

    def fake_build_wet_planning_footprint(stats, **kwargs):
        return SimpleNamespace(
            native_mask=None,
            coarse_mask=None,
            active_windows=(),
            factor=4,
            safety_cells=1,
            digest="footprint-digest",
            covered_years=list(kwargs.get("requested_years", [])),
            source_collection="ga_ls_wo_fq_myear_3",
            source_version="2.1.0",
            source_lineage="ga_ls_wo_3",
        )

    def fake_acquire_wofs_cache(*args, **kwargs):
        recorder.acquire_calls.append(kwargs)
        return SimpleNamespace(
            path=str(tmp_path / "cache" / "aoi.zarr"),
            identity="identity-digest",
            request_digest="request-digest",
        )

    def fake_open_completed_mask_cache(handle, start_date, end_date, **kwargs):
        time = pd.date_range("2020-01-01", periods=4, freq="MS")
        rng = np.random.default_rng(0)
        values = rng.integers(0, 2, size=(4, *_SHAPE)).astype(np.int16)
        return xr.DataArray(
            values,
            dims=("time", "y", "x"),
            coords={
                "time": time,
                "y": 120.0 - np.arange(_SHAPE[0]) * 30.0 - 15.0,
                "x": np.arange(_SHAPE[1]) * 30.0 + 15.0,
            },
        ).rio.write_crs(_CRS)

    def fake_verify_cache_footprints(handle):
        full = box(0.0, 0.0, 120.0, 120.0)
        return SimpleNamespace(
            aoi_geometry_wkb_hex=wkb.dumps(full).hex(),
            analysis_geometry_wkb_hex=wkb.dumps(full).hex(),
            crs=_CRS,
            shape=_SHAPE,
            transform=_TRANSFORM,
            aoi_pixel_count=16,
            analysis_pixel_count=16,
            aoi_digest="a" * 64,
            analysis_digest="b" * 64,
        )

    def fake_open_completed_dual_extent_counts(handle, start_date, end_date):
        time = pd.date_range("2020-01-01", periods=4, freq="MS")
        return pd.DataFrame(
            {
                "aoi_pixel_count": [16] * 4,
                "analysis_mask_pixel_count": [16] * 4,
                "n_max_water": [8] * 4,
                "n_median_water": [6] * 4,
                "n_valid_analysis": [16] * 4,
            },
            index=pd.DatetimeIndex(time),
        )

    monkeypatch.setattr(
        hydroseason, "open_wo_statistics", fake_open_wo_statistics, raising=False
    )
    monkeypatch.setattr(
        hydroseason._io_dea_stats,
        "build_wet_planning_footprint",
        fake_build_wet_planning_footprint,
        raising=False,
    )
    monkeypatch.setattr(
        hydroseason, "acquire_wofs_cache", fake_acquire_wofs_cache, raising=False
    )
    monkeypatch.setattr(
        hydroseason,
        "open_completed_mask_cache",
        fake_open_completed_mask_cache,
        raising=False,
    )
    monkeypatch.setattr(
        hydroseason, "verify_cache_footprints", fake_verify_cache_footprints, raising=False
    )
    monkeypatch.setattr(
        hydroseason,
        "open_completed_dual_extent_counts",
        fake_open_completed_dual_extent_counts,
        raising=False,
    )
    monkeypatch.setattr(
        workflow_module, "_frequency_geobox", lambda frequency: SimpleNamespace()
    )
    return recorder


_PROVENANCE = {
    "channel_ruleset_version": "c-1.0.0",
    "waterbody_ruleset_version": "w-1.0.0",
    "riverine_ruleset_version": "r-1.0.0",
    "dem_product": "ga_srtm_dem1sv1_0",
    "dem_band": "elevation",
    "fc_product": "ga_ls_fc_pc_cyear_3",
    "fc_bands": ("bs_pc_50", "pv_pc_50", "npv_pc_50"),
    "waterbodies_source": "dea_waterbodies",
    "years": (2020, 2020),
    "bridge_enabled": True,
    "bare_threshold_pct": 30.0,
    "envelope": {"a": 2.0, "b": 0.0, "n_bins": 1},
    "pixel_counts": {
        "domain": 16,
        "water_seed": 16,
        "observed_channel": 12,
        "bridged_channel": 1,
        "in_channel": 13,
        "off_channel_riverine": 2,
        "non_riverine": 1,
    },
    "bridge_counts_by_cause": {"vegetated": 1},
    "unbridged_counts_by_reason": {},
    "reach_counts": {"total": 1, "line_fallback": 0, "multithread": 0},
}


def _bridges():
    return gpd.GeoDataFrame(
        {
            "gap_id": [0],
            "reach_ids": ["1"],
            "length_m": [90.0],
            "cumulative_cost": [1.0],
            "cost_per_m": [0.011],
            "upstream_width_m": [30.0],
            "downstream_width_m": [30.0],
            "bridge_confidence": [2],
            "gap_cause": ["vegetated"],
        },
        geometry=[LineString([(15.0, 15.0), (105.0, 15.0)])],
        crs=_CRS,
    )


_LANDFORM = np.array(
    [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 2, 2, 3],
    ],
    dtype=np.uint8,
)
_HYDROPERIOD = np.array(
    [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 4],
        [1, 1, 2, 3],
    ],
    dtype=np.uint8,
)


def _landform_result():
    bridged = np.zeros(_SHAPE, dtype=bool)
    bridged[2, 3] = True
    return LandformResult(
        landform=_LANDFORM,
        channel_source=np.where(bridged, 2, 1).astype(np.uint8),
        channel_confidence=np.full(_SHAPE, 3, dtype=np.uint8),
        rule_id=np.ones(_SHAPE, dtype=np.uint8),
        rem=np.zeros(_SHAPE, dtype=np.float32),
        bridged_mask=bridged,
        channel_bridges=_bridges(),
        unbridged=(),
        envelope=FloodplainEnvelope(
            a=2.0,
            b=0.0,
            n_bins=1,
            bin_log_a_mid=(3.7,),
            bin_rem_quantile=(2.0,),
            bin_counts=(16,),
            degraded_reasons=(),
        ),
        provenance=dict(_PROVENANCE),
        degraded_reasons=("bare_threshold_fallback",),
    )


def _install_riverscape(monkeypatch, recorder):
    def fake_zones_from_riverscape_with_landform(stats, **kwargs):
        recorder.riverscape_calls.append(kwargs)
        grid = SpatialGrid.from_dataarray(stats.frequency, require_georeference=True)
        zone_result = replace(
            combine_zones(
                _LANDFORM,
                _HYDROPERIOD,
                source="ga_ls_wo_fq_myear_3",
                degraded_reasons=("bare_threshold_fallback",),
            ),
            grid=grid,
        )
        return zone_result, with_grid(_landform_result(), grid)

    monkeypatch.setattr(
        workflow_module,
        "zones_from_riverscape_with_landform",
        fake_zones_from_riverscape_with_landform,
    )
    monkeypatch.setattr(
        workflow_module, "_frequency_geobox", lambda frequency: SimpleNamespace()
    )


def _evidence_config(output_dir: Path, mode: str = "auto") -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "max_water",
                "composite_owner": "upstream",
            },
            "riverscape": {"mode": mode},
            "output": {
                "output_dir": str(output_dir),
                "spatial_products": ["riverscape_evidence"],
            },
        }
    )


def _run(monkeypatch, tmp_path, output_dir, *, config=None, drainage=None):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)
    _install_riverscape(monkeypatch, recorder)
    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        drainage=_drainage() if drainage is None else drainage,
        cache_dir=tmp_path / "wofs_cache",
        config=config or _evidence_config(output_dir),
    )
    return result, recorder


# --------------------------------------------------------------------------
# Acceptance (spec 6b section 10)
# --------------------------------------------------------------------------


def test_riverscape_run_writes_the_full_zoning_section(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "out_zoning"
    result, _recorder = _run(monkeypatch, tmp_path, output_dir)

    manifest = json.loads(
        (Path(result.output_dir) / "run_manifest.json").read_text(encoding="utf-8")
    )
    zoning = manifest["zoning"]
    assert zoning["mode"] == "riverscape"
    assert zoning["degraded_reasons"] == ["bare_threshold_fallback"]
    assert zoning["domain_pixel_count"] == 16
    assert zoning["sources"]["dem_product"] == "ga_srtm_dem1sv1_0"
    assert zoning["sources"]["zone_source"] == "ga_ls_wo_fq_myear_3"
    assert zoning["ruleset_versions"]["channel"] == "c-1.0.0"
    assert zoning["bridge_counts_by_cause"] == {"vegetated": 1}
    assert zoning["bridged_length_m"] == pytest.approx(90.0)
    assert zoning["bridged_area_m2"] == pytest.approx(900.0)
    assert zoning["non_riverine_area_m2"] == pytest.approx(900.0)


def test_riverscape_run_writes_six_evidence_rasters(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "out_rasters"
    result, _recorder = _run(monkeypatch, tmp_path, output_dir)
    raster_dir = Path(result.output_dir) / "rasters"

    for filename in (
        "landform.tif",
        "hydroperiod.tif",
        "zone_crosstab.tif",
        "channel_source.tif",
        "channel_confidence.tif",
        "rem.tif",
    ):
        assert (raster_dir / filename).exists(), filename

    with rasterio.open(raster_dir / "hydroperiod.tif") as dataset:
        assert int(dataset.read(1)[2, 3]) == 4
    with rasterio.open(raster_dir / "zone_crosstab.tif") as dataset:
        assert dataset.dtypes[0] == "uint16"
        assert int(dataset.read(1)[2, 3]) == 14

    inventory = {
        item["relative_path"]
        for item in result.manifest["artifact_inventory"]
    }
    assert "rasters/landform.tif" in inventory
    assert "rasters/rem.tif" in inventory


def test_riverscape_run_writes_the_channel_bridges_layer(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "out_bridges"
    result, _recorder = _run(monkeypatch, tmp_path, output_dir)
    gpkg = Path(result.output_dir) / "vectors" / "spatial.gpkg"

    assert gpkg.exists()
    assert "channel_bridges" in pyogrio.list_layers(gpkg)[:, 0]
    frame = gpd.read_file(gpkg, layer="channel_bridges")
    assert len(frame) == 1
    assert frame.iloc[0]["gap_cause"] == "vegetated"
    assert set(
        [
            "gap_id",
            "reach_ids",
            "length_m",
            "cumulative_cost",
            "cost_per_m",
            "upstream_width_m",
            "downstream_width_m",
            "bridge_confidence",
            "gap_cause",
        ]
    ).issubset(frame.columns)


# --------------------------------------------------------------------------
# Hermetic occurrence defaults (spec 6b sections 2 row 9, 6, 10)
# --------------------------------------------------------------------------


def test_default_config_does_not_opt_into_riverscape_evidence(tmp_path) -> None:
    """MUTANT: adding ``riverscape_evidence`` to ``_default_config`` would
    make every ``config=None`` run try to write evidence it cannot produce."""
    config = _default_config(output_dir=tmp_path)

    assert "riverscape_evidence" not in config.output.spatial_products


def test_mode_off_run_writes_no_evidence_and_a_thin_zoning_section(
    monkeypatch, tmp_path
) -> None:
    output_dir = tmp_path / "out_off"
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "max_water",
                "composite_owner": "upstream",
            },
            "riverscape": {"mode": "off"},
            "output": {"output_dir": str(output_dir)},
        }
    )
    result, _recorder = _run(monkeypatch, tmp_path, output_dir, config=config)

    manifest = json.loads(
        (Path(result.output_dir) / "run_manifest.json").read_text(encoding="utf-8")
    )
    zoning = manifest["zoning"]
    assert zoning["mode"] == "occurrence"
    assert "sources" not in zoning
    assert "bridged_length_m" not in zoning
    assert not (Path(result.output_dir) / "rasters" / "landform.tif").exists()


def test_evidence_requested_on_an_occurrence_run_fails_preflight(
    monkeypatch, tmp_path
) -> None:
    """Spec 6b section 5: requested-without-bundle is a preflight failure,
    not a silently skipped product."""
    from hydrofragments.output.finalize import SpatialProductUnavailable

    output_dir = tmp_path / "out_no_bundle"
    with pytest.raises(SpatialProductUnavailable, match="riverscape_evidence requires"):
        _run(
            monkeypatch,
            tmp_path,
            output_dir,
            config=_evidence_config(output_dir, mode="off"),
        )
