"""``analyze_from_dea``'s riverscape.mode table (Phase 6a spec section 6).

Two layers of test. The mode table itself is exercised directly against
``_resolve_zone_result``, because that is where every row of the table lives
and a direct call needs no WOfS acquisition at all. Two end-to-end
``analyze_from_dea`` runs then prove the wiring: that the resolved
``ZoneResult`` really is the one the run uses, that ``required`` fails
before any acquisition happens, and that the riverscape branch is timed.

Following ``tests/integration/test_dea_workflow.py``'s convention, every
hydroseason entry point is monkeypatched as a module attribute on the real
``hydroseason`` package. ``zones_from_riverscape`` and ``_frequency_geobox``
are monkeypatched on ``hydrofragments.workflow`` so no test touches
STAC/WFS or needs a real odc-geo GeoBox.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
xr = pytest.importorskip("xarray")
pytest.importorskip("rasterio")
pytest.importorskip("rioxarray")

import hydroseason
from shapely import wkb
from shapely.geometry import LineString, box

from hydrofragments import workflow as workflow_module
from hydrofragments.config import HydroConfig
from hydrofragments.io.riverscape_sources import RiverscapeSourceUnavailable
from hydrofragments.models import HydroResult
from hydrofragments.spatial.zones import ZoneResult, combine_zones
from hydrofragments.workflow import (
    _default_config,
    _resolve_zone_result,
    _riverscape_reach_context,
    analyze_from_dea,
)

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


def _drainage_two_reach() -> "gpd.GeoDataFrame":
    """Two reaches sharing the endpoint (60, 60) so 60 m buffers overlap."""
    return gpd.GeoDataFrame(
        {
            "HydroID": [1, 2],
            "From_Node": [1, 2],
            "To_Node": [2, 3],
            "NextDownID": [2, -1],
            "UpstrDArea": [3000.0, 5000.0],
        },
        geometry=[
            LineString([(10.0, 10.0), (60.0, 60.0)]),
            LineString([(60.0, 60.0), (110.0, 110.0)]),
        ],
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


def _riverscape_zone_result() -> ZoneResult:
    landform = np.full(_SHAPE, 1, dtype=np.uint8)
    hydroperiod = np.full(_SHAPE, 1, dtype=np.uint8)
    return combine_zones(landform, hydroperiod, source="ga_ls_wo_fq_myear_3")


def _config(mode: str) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "max_water",
                "composite_owner": "upstream",
            },
            "riverscape": {"mode": mode},
        }
    )


def _output_config(mode: str, output_dir: Path) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "max_water",
                "composite_owner": "upstream",
            },
            "riverscape": {"mode": mode},
            "output": {"output_dir": str(output_dir)},
        }
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
        )

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


def _install_riverscape(monkeypatch, recorder: _Recorder, *, raises=None):
    def fake_zones_from_riverscape(stats, **kwargs):
        recorder.riverscape_calls.append(kwargs)
        if raises is not None:
            raise raises
        return _riverscape_zone_result()

    monkeypatch.setattr(
        workflow_module, "zones_from_riverscape", fake_zones_from_riverscape
    )


# --------------------------------------------------------------------------
# Decision D1: the built-in default config opts out of riverscape zoning
# --------------------------------------------------------------------------


def test_default_config_pins_riverscape_mode_off(tmp_path) -> None:
    assert _default_config(output_dir=tmp_path).riverscape.mode == "off"


# --------------------------------------------------------------------------
# Mode table (spec section 6), exercised directly
# --------------------------------------------------------------------------


def test_mode_off_uses_occurrence_zoning(monkeypatch) -> None:
    recorder = _Recorder()
    _install_riverscape(monkeypatch, recorder)
    timings: dict[str, float] = {}

    zone_result, reasons = _resolve_zone_result(
        _stats(),
        _drainage(),
        config=_config("off"),
        years=(2020, 2020),
        pixel_m=30.0,
        timings=timings,
    )

    assert zone_result is not None
    assert zone_result.mode == "occurrence"
    assert reasons == ()
    assert recorder.riverscape_calls == []
    assert "riverscape" not in timings


def test_mode_off_without_stats_returns_no_zones(monkeypatch) -> None:
    zone_result, reasons = _resolve_zone_result(
        None,
        _drainage(),
        config=_config("off"),
        years=(2020, 2020),
        pixel_m=30.0,
        timings={},
    )

    assert zone_result is None
    assert reasons == ()


def test_auto_without_drainage_degrades_to_occurrence_with_a_reason(monkeypatch) -> None:
    """MUTANT 1: skipping the degrade reason must fail here."""
    zone_result, reasons = _resolve_zone_result(
        _stats(),
        None,
        config=_config("auto"),
        years=(2020, 2020),
        pixel_m=30.0,
        timings={},
    )

    assert zone_result is not None
    assert zone_result.mode == "occurrence"
    assert "riverscape_no_drainage" in zone_result.degraded_reasons
    assert reasons == ("riverscape_no_drainage",)


def test_auto_with_drainage_uses_riverscape_zoning(monkeypatch) -> None:
    recorder = _Recorder()
    _install_riverscape(monkeypatch, recorder)
    monkeypatch.setattr(
        workflow_module, "_frequency_geobox", lambda frequency: SimpleNamespace()
    )
    timings: dict[str, float] = {}

    zone_result, reasons = _resolve_zone_result(
        _stats(),
        _drainage(),
        config=_config("auto"),
        years=(2020, 2020),
        pixel_m=30.0,
        timings=timings,
    )

    assert zone_result is not None
    assert zone_result.mode == "riverscape"
    assert reasons == ()
    assert timings["riverscape"] >= 0.0
    call = recorder.riverscape_calls[0]
    assert call["years"] == (2020, 2020)
    assert call["pixel_m"] == 30.0
    assert call["reach_keys"] == {1: "1"}
    assert call["upstr_darea"] == {1: 5000.0}
    assert call["reach_labels"].shape == _SHAPE


def test_auto_with_source_failure_degrades_to_occurrence_with_a_reason(monkeypatch) -> None:
    recorder = _Recorder()
    _install_riverscape(
        monkeypatch, recorder, raises=RiverscapeSourceUnavailable("DEM unavailable")
    )
    monkeypatch.setattr(
        workflow_module, "_frequency_geobox", lambda frequency: SimpleNamespace()
    )
    timings: dict[str, float] = {}

    zone_result, reasons = _resolve_zone_result(
        _stats(),
        _drainage(),
        config=_config("auto"),
        years=(2020, 2020),
        pixel_m=30.0,
        timings=timings,
    )

    assert zone_result is not None
    assert zone_result.mode == "occurrence"
    assert "riverscape_source_unavailable" in zone_result.degraded_reasons
    assert reasons == ("riverscape_source_unavailable",)
    assert "riverscape" in timings


def test_auto_without_stats_reports_no_stats_and_no_zones() -> None:
    zone_result, reasons = _resolve_zone_result(
        None,
        _drainage(),
        config=_config("auto"),
        years=(2020, 2020),
        pixel_m=30.0,
        timings={},
    )

    assert zone_result is None
    assert reasons == ("riverscape_no_stats",)


def test_required_without_drainage_raises() -> None:
    """MUTANT 2: falling back instead of raising must fail here."""
    with pytest.raises(RiverscapeSourceUnavailable, match="needs drainage"):
        _resolve_zone_result(
            _stats(),
            None,
            config=_config("required"),
            years=(2020, 2020),
            pixel_m=30.0,
            timings={},
        )


def test_required_without_stats_raises() -> None:
    with pytest.raises(RiverscapeSourceUnavailable, match="WO statistics"):
        _resolve_zone_result(
            None,
            _drainage(),
            config=_config("required"),
            years=(2020, 2020),
            pixel_m=30.0,
            timings={},
        )


def test_required_propagates_a_source_failure(monkeypatch) -> None:
    recorder = _Recorder()
    _install_riverscape(
        monkeypatch, recorder, raises=RiverscapeSourceUnavailable("FC unavailable")
    )
    monkeypatch.setattr(
        workflow_module, "_frequency_geobox", lambda frequency: SimpleNamespace()
    )

    with pytest.raises(RiverscapeSourceUnavailable, match="FC unavailable"):
        _resolve_zone_result(
            _stats(),
            _drainage(),
            config=_config("required"),
            years=(2020, 2020),
            pixel_m=30.0,
            timings={},
        )


def test_unexpected_exceptions_are_never_swallowed_as_a_degrade(monkeypatch) -> None:
    def exploding_zones_from_riverscape(stats, **kwargs):
        raise KeyError("bug in the pipeline, not a missing source")

    monkeypatch.setattr(
        workflow_module, "zones_from_riverscape", exploding_zones_from_riverscape
    )
    monkeypatch.setattr(
        workflow_module, "_frequency_geobox", lambda frequency: SimpleNamespace()
    )

    with pytest.raises(KeyError):
        _resolve_zone_result(
            _stats(),
            _drainage(),
            config=_config("auto"),
            years=(2020, 2020),
            pixel_m=30.0,
            timings={},
        )


# --------------------------------------------------------------------------
# Reach-label context (built on the spatial side of the import boundary)
# --------------------------------------------------------------------------


def test_reach_context_labels_every_pixel_and_keys_by_label() -> None:
    stats = _stats()
    labels, reach_keys, upstr_darea = _riverscape_reach_context(
        _drainage(), stats.frequency, buffer_m=60.0
    )

    assert labels.shape == _SHAPE
    assert labels.dtype == np.int32
    assert (labels == 1).all()
    assert not (labels == -1).any()
    assert reach_keys == {1: "1"}
    assert upstr_darea == {1: 5000.0}


def test_reach_context_reprojects_drainage_crs_to_frequency_grid() -> None:
    """Drainage in EPSG:4326 must label the EPSG:3577 frequency grid."""
    stats = _stats()
    drainage_wgs84 = _drainage().to_crs("EPSG:4326")

    labels, reach_keys, upstr_darea = _riverscape_reach_context(
        drainage_wgs84, stats.frequency, buffer_m=60.0
    )

    assert labels.max() > 0
    assert (labels == 1).all()
    assert not (labels == -1).any()
    assert reach_keys == {1: "1"}
    assert upstr_darea == {1: 5000.0}


def test_reach_context_raises_when_drainage_misses_grid() -> None:
    stats = _stats()
    far_away = gpd.GeoDataFrame(
        {
            "HydroID": [1],
            "From_Node": [1],
            "To_Node": [2],
            "NextDownID": [-1],
            "UpstrDArea": [5000.0],
        },
        geometry=[LineString([(10000.0, 10000.0), (20000.0, 20000.0)])],
        crs=_CRS,
    )

    with pytest.raises(ValueError, match="reach label raster is empty"):
        _riverscape_reach_context(far_away, stats.frequency, buffer_m=60.0)


def test_reach_context_resolves_overlap_and_fills_unlabeled_pixels() -> None:
    """Two-reach fixture exercises overlap resolution and nearest-fill."""
    stats = _stats()
    labels, reach_keys, upstr_darea = _riverscape_reach_context(
        _drainage_two_reach(), stats.frequency, buffer_m=60.0
    )

    assert labels.shape == _SHAPE
    assert labels.dtype == np.int32
    assert not (labels == -1).any()
    assert labels.max() > 0
    assert (labels > 0).all()
    assert set(np.unique(labels)) == {1, 2}
    assert reach_keys == {1: "1", 2: "2"}
    assert upstr_darea == {1: 3000.0, 2: 5000.0}


# --------------------------------------------------------------------------
# End-to-end wiring
# --------------------------------------------------------------------------


def test_analyze_from_dea_auto_with_drainage_runs_the_riverscape_branch(
    monkeypatch, tmp_path
) -> None:
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)
    _install_riverscape(monkeypatch, recorder)

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        drainage=_drainage(),
        cache_dir=tmp_path / "wofs_cache",
        config=_output_config("auto", tmp_path / "output_auto"),
    )

    assert len(recorder.riverscape_calls) == 1
    assert recorder.riverscape_calls[0]["years"] == (2020, 2020)
    timings = result.manifest["timings_seconds"]
    assert "riverscape" in timings
    assert timings["dea_planning"] >= 0.0
    assert (Path(result.output_dir) / "run_manifest.json").exists()


def test_dea_planning_carves_out_riverscape_timing(monkeypatch, tmp_path) -> None:
    """MUTANT D2: removing ``- timings.get("riverscape", 0.0)`` from the
    ``dea_planning`` assignment in ``analyze_from_dea`` must fail here."""
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)
    _install_riverscape(monkeypatch, recorder)

    clock = {"t": 0.0}

    def fake_perf_counter() -> float:
        return clock["t"]

    monkeypatch.setattr(workflow_module.time, "perf_counter", fake_perf_counter)

    riverscape_seconds = 2.0
    original_resolve = workflow_module._resolve_zone_result

    def wrapped_resolve_zone_result(
        stats, drainage_gdf, *, config, years, pixel_m, timings
    ):
        clock["t"] += riverscape_seconds
        result = original_resolve(
            stats,
            drainage_gdf,
            config=config,
            years=years,
            pixel_m=pixel_m,
            timings=timings,
        )
        timings["riverscape"] = riverscape_seconds
        return result

    monkeypatch.setattr(
        workflow_module, "_resolve_zone_result", wrapped_resolve_zone_result
    )

    def fake_finalize(_config, _core, **kwargs):
        timings = dict(kwargs["timings_seconds"])
        timings["total"] = sum(value for key, value in timings.items() if key != "total")
        return HydroResult(
            metrics_table=pd.DataFrame(),
            manifest={"timings_seconds": timings},
            output_dir=Path(_config.output.output_dir),
            run_id="carveout-test",
        )

    monkeypatch.setattr(workflow_module, "finalize_analysis_bundle", fake_finalize)

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        drainage=_drainage(),
        cache_dir=tmp_path / "wofs_cache",
        config=_output_config("auto", tmp_path / "output_carveout"),
    )

    timings = result.manifest["timings_seconds"]
    assert timings["riverscape"] == riverscape_seconds
    assert timings["dea_planning"] == pytest.approx(0.0, abs=1e-9)


def test_analyze_from_dea_required_without_drainage_fails_before_acquisition(
    monkeypatch, tmp_path
) -> None:
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)
    _install_riverscape(monkeypatch, recorder)

    with pytest.raises(RiverscapeSourceUnavailable, match="needs drainage"):
        analyze_from_dea(
            _aoi(),
            "2020-01-01",
            "2020-04-30",
            aoi_id="test_aoi",
            cache_dir=tmp_path / "wofs_cache",
            config=_output_config("required", tmp_path / "output_required"),
        )

    assert recorder.acquire_calls == []


def test_bad_drainage_schema_fails_before_acquisition(monkeypatch, tmp_path) -> None:
    """Spec section 6: drainage validation runs before the riverscape loaders."""
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)
    _install_riverscape(monkeypatch, recorder)

    with pytest.raises(ValueError, match="UpstrDArea"):
        analyze_from_dea(
            _aoi(),
            "2020-01-01",
            "2020-04-30",
            aoi_id="test_aoi",
            drainage=_drainage().drop(columns=["UpstrDArea"]),
            cache_dir=tmp_path / "wofs_cache",
            config=_output_config("auto", tmp_path / "output_bad_schema"),
        )

    assert recorder.acquire_calls == []
    assert recorder.riverscape_calls == []


def test_mode_off_still_validates_drainage_topology_early(monkeypatch, tmp_path) -> None:
    """Parent spec section 7: validate CRS/columns right after AOI load, even
    when mode is off -- channel metrics already need a valid topology."""
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)
    _install_riverscape(monkeypatch, recorder)

    broken = _drainage().drop(columns=["To_Node"])
    with pytest.raises(ValueError, match="To_Node"):
        analyze_from_dea(
            _aoi(),
            "2020-01-01",
            "2020-04-30",
            aoi_id="test_aoi",
            drainage=broken,
            cache_dir=tmp_path / "wofs_cache",
            config=_output_config("off", tmp_path / "output_off_topology"),
        )

    assert recorder.acquire_calls == []
