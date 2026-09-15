"""Unit tests for the DEA STAC/WFS loaders behind riverscape zoning's
terrain, cover, and waterbody evidence.

Every DEA-touching function is exercised by monkeypatching the real
`pystac_client`/`odc.stac`/`geopandas` module attributes as fakes -- the
same convention `tests/integration/test_dea_workflow.py` uses for
hydroseason. No test here makes a real network call.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from shapely.geometry import box

gpd = pytest.importorskip("geopandas")
xr = pytest.importorskip("xarray")
odc_stac = pytest.importorskip("odc.stac")
pystac_client = pytest.importorskip("pystac_client")
pyogrio = pytest.importorskip("pyogrio")

from hydrofragments.io.riverscape_sources import (
    RiverscapeSourceUnavailable,
    load_dem,
    load_fc_percentiles,
    load_waterbodies,
)


class _FakeGeobox:
    """Minimal odc.geo.GeoBox stand-in: only `.extent.to_crs(...).boundingbox` is used."""

    def __init__(self, bbox_ll=(120.0, -20.0, 121.0, -19.0)):
        self._bbox_ll = bbox_ll

    @property
    def extent(self):
        return SimpleNamespace(to_crs=lambda _crs: SimpleNamespace(boundingbox=self._bbox_ll))


class _FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class _FakeClient:
    def __init__(self, items):
        self._items = items

    def search(self, **kwargs):
        return _FakeSearch(self._items)


def _dem_dataset(*, band: str) -> "xr.Dataset":
    data = np.array([[10.0, 12.0], [9.0, 11.0]], dtype="float32")
    return xr.Dataset({band: (("y", "x"), data)}, coords={"y": [1.0, 0.0], "x": [0.0, 1.0]})


def test_load_dem_returns_grid_aligned_dataarray(monkeypatch) -> None:
    monkeypatch.setattr(
        pystac_client.Client, "open", staticmethod(lambda url, **kw: _FakeClient(["item-1"]))
    )
    monkeypatch.setattr(
        odc_stac, "load", lambda items, **kw: _dem_dataset(band=kw["bands"][0])
    )
    monkeypatch.setattr(odc_stac, "configure_rio", lambda **kw: None)

    result = load_dem(_FakeGeobox(), product="ga_srtm_dem1sv1_0", band="dem_s")

    assert result.shape == (2, 2)
    assert float(result.isel(y=0, x=0)) == 10.0


class _CapturingClient:
    """Like _FakeClient, but records the kwargs passed to .search()."""

    def __init__(self, items, captured: dict[str, Any]):
        self._items = items
        self._captured = captured

    def search(self, **kwargs):
        self._captured["search_kwargs"] = kwargs
        return _FakeSearch(self._items)


def test_load_dem_forwards_geobox_and_derived_bbox_to_odc_stac_load(monkeypatch) -> None:
    # Nothing previously asserted that `geobox` actually reaches
    # odc.stac.load, or that the STAC search bbox is derived from the
    # geobox's own extent -- the fake loaders monkeypatched elsewhere
    # ignore kwargs entirely.
    captured: dict[str, Any] = {}
    search_captured: dict[str, Any] = {}

    monkeypatch.setattr(
        pystac_client.Client, "open",
        staticmethod(lambda url, **kw: _CapturingClient(["item-1"], search_captured)),
    )
    monkeypatch.setattr(odc_stac, "configure_rio", lambda **kw: None)

    def _fake_load(items, **kw):
        captured["load_kwargs"] = kw
        return _dem_dataset(band=kw["bands"][0])

    monkeypatch.setattr(odc_stac, "load", _fake_load)

    bbox_ll = (110.0, -30.0, 111.0, -29.0)
    geobox = _FakeGeobox(bbox_ll)
    load_dem(geobox, product="ga_srtm_dem1sv1_0", band="dem_s")

    assert captured["load_kwargs"]["geobox"] is geobox
    assert search_captured["search_kwargs"]["bbox"] == list(bbox_ll)


def test_load_fc_percentiles_forwards_years_as_datetime_filter_to_search(monkeypatch) -> None:
    # Nothing previously asserted that `years` actually becomes the
    # `datetime=` filter string passed to Client.search.
    search_captured: dict[str, Any] = {}

    monkeypatch.setattr(
        pystac_client.Client, "open",
        staticmethod(lambda url, **kw: _CapturingClient(["item-1"], search_captured)),
    )
    monkeypatch.setattr(odc_stac, "configure_rio", lambda **kw: None)
    monkeypatch.setattr(
        odc_stac, "load", lambda items, **kw: _dem_dataset(band=kw["bands"][0])
    )

    load_fc_percentiles(
        _FakeGeobox(), product="ga_ls_fc_pc_cyear_3", bands=["bs_pc_50"], years=(2019, 2021)
    )

    datetime_filter = search_captured["search_kwargs"]["datetime"]
    assert "2019" in datetime_filter
    assert "2021" in datetime_filter


def test_load_dem_raises_riverscape_source_unavailable_when_all_stac_urls_fail(
    monkeypatch,
) -> None:
    def _raise_open(url, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(pystac_client.Client, "open", staticmethod(_raise_open))
    monkeypatch.setattr(odc_stac, "configure_rio", lambda **kw: None)

    with pytest.raises(RiverscapeSourceUnavailable, match="ga_srtm_dem1sv1_0"):
        load_dem(_FakeGeobox(), product="ga_srtm_dem1sv1_0", band="dem_s")


def test_load_dem_raises_riverscape_source_unavailable_when_read_fails(monkeypatch) -> None:
    """The STAC search can succeed while the odc.stac.load read itself fails
    (a bad COG fetch, a rasterio IO error, an S3 timeout) -- that read half
    must be wrapped in the same RiverscapeSourceUnavailable contract as the
    search half, or a mode="auto" caller catching only that one exception
    type would crash instead of falling back to occurrence zoning.
    """
    monkeypatch.setattr(
        pystac_client.Client, "open", staticmethod(lambda url, **kw: _FakeClient(["item-1"]))
    )
    monkeypatch.setattr(odc_stac, "configure_rio", lambda **kw: None)

    def _raise_load(items, **kw):
        raise RuntimeError("rasterio IO error: failed to fetch COG")

    monkeypatch.setattr(odc_stac, "load", _raise_load)

    with pytest.raises(RiverscapeSourceUnavailable, match="ga_srtm_dem1sv1_0"):
        load_dem(_FakeGeobox(), product="ga_srtm_dem1sv1_0", band="dem_s")


def test_load_fc_percentiles_raises_riverscape_source_unavailable_when_read_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pystac_client.Client, "open", staticmethod(lambda url, **kw: _FakeClient(["item-1"]))
    )
    monkeypatch.setattr(odc_stac, "configure_rio", lambda **kw: None)

    def _raise_load(items, **kw):
        raise RuntimeError("rasterio IO error: failed to fetch COG")

    monkeypatch.setattr(odc_stac, "load", _raise_load)

    with pytest.raises(RiverscapeSourceUnavailable, match="ga_ls_fc_pc_cyear_3"):
        load_fc_percentiles(
            _FakeGeobox(), product="ga_ls_fc_pc_cyear_3", bands=["bs_pc_50"], years=(2022, 2023)
        )


def test_load_fc_percentiles_returns_one_array_per_band(monkeypatch) -> None:
    monkeypatch.setattr(
        pystac_client.Client, "open", staticmethod(lambda url, **kw: _FakeClient(["item-1"]))
    )
    monkeypatch.setattr(
        odc_stac, "load", lambda items, **kw: _dem_dataset(band=kw["bands"][0])
    )
    monkeypatch.setattr(odc_stac, "configure_rio", lambda **kw: None)

    result = load_fc_percentiles(
        _FakeGeobox(),
        product="ga_ls_fc_pc_cyear_3",
        bands=["bs_pc_50", "pv_pc_50"],
        years=(2022, 2023),
    )

    assert set(result) == {"bs_pc_50", "pv_pc_50"}
    assert result["bs_pc_50"].shape == (2, 2)


def _fc_dataset_with_time_and_nodata(*, band: str, nodata: int = 255) -> "xr.Dataset":
    """A 2-step time series for one pixel: a valid observation (50) and a
    nodata sentinel (255, DEA FC's uint8 nodata convention). An unmasked
    ``median("time")`` over [50, 255] would return 152.5 -- a value pulled
    toward the sentinel and nowhere near either real observation.
    """
    data = np.array([[[50]], [[nodata]]], dtype="uint8")
    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": [0, 1], "y": [0.0], "x": [0.0]},
        attrs={"nodata": nodata},
    )
    return xr.Dataset({band: da})


def test_load_fc_percentiles_masks_nodata_before_median(monkeypatch) -> None:
    monkeypatch.setattr(
        pystac_client.Client, "open", staticmethod(lambda url, **kw: _FakeClient(["item-1"]))
    )
    monkeypatch.setattr(odc_stac, "configure_rio", lambda **kw: None)
    monkeypatch.setattr(
        odc_stac,
        "load",
        lambda items, **kw: _fc_dataset_with_time_and_nodata(band=kw["bands"][0]),
    )

    result = load_fc_percentiles(
        _FakeGeobox(), product="ga_ls_fc_pc_cyear_3", bands=["bs_pc_50"], years=(2022, 2023)
    )

    # Masked median of [50, <nodata>] must reflect only the valid
    # observation (50), not a blend that includes the 255 sentinel.
    assert float(result["bs_pc_50"].isel(y=0, x=0)) == 50.0


def test_load_fc_percentiles_rejects_end_year_before_start_year() -> None:
    with pytest.raises(ValueError, match="years end"):
        load_fc_percentiles(
            _FakeGeobox(), product="ga_ls_fc_pc_cyear_3", bands=["bs_pc_50"], years=(2023, 2020)
        )


def test_load_waterbodies_returns_empty_geodataframe_when_no_features(monkeypatch) -> None:
    empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    monkeypatch.setattr(pyogrio, "read_info", lambda *a, **kw: {"crs": "EPSG:4326"})
    monkeypatch.setattr(gpd, "read_file", lambda *a, **kw: empty)

    result = load_waterbodies((0.0, 0.0, 100.0, 100.0), "EPSG:3577")

    assert result.empty
    assert result.crs == "EPSG:3577"


def test_load_waterbodies_raises_riverscape_source_unavailable_on_read_failure(
    monkeypatch,
) -> None:
    def _raise_read(*a, **kw):
        raise RuntimeError("WFS endpoint unreachable")

    monkeypatch.setattr(pyogrio, "read_info", lambda *a, **kw: {"crs": "EPSG:4326"})
    monkeypatch.setattr(gpd, "read_file", _raise_read)

    with pytest.raises(RiverscapeSourceUnavailable, match="Waterbodies"):
        load_waterbodies((0.0, 0.0, 100.0, 100.0), "EPSG:3577")


def test_load_waterbodies_raises_riverscape_source_unavailable_on_read_info_failure(
    monkeypatch,
) -> None:
    def _raise_read_info(*a, **kw):
        raise RuntimeError("cannot reach WFS endpoint to inspect it")

    monkeypatch.setattr(pyogrio, "read_info", _raise_read_info)

    with pytest.raises(RiverscapeSourceUnavailable, match="Waterbodies"):
        load_waterbodies((0.0, 0.0, 100.0, 100.0), "EPSG:3577")


def test_load_waterbodies_bbox_is_reprojected_into_layer_native_crs(monkeypatch) -> None:
    """pyogrio's bbox filter runs in the layer's native CRS, not EPSG:4326 --
    if the layer isn't 4326 (here EPSG:3577), the bbox passed to
    gpd.read_file must be reprojected into 3577, not left in the input CRS
    (EPSG:3577 too, coincidentally, but via a totally different value here)
    or hardcoded to 4326.
    """
    captured: dict[str, Any] = {}

    def _fake_read_file(*a, **kw):
        captured["bbox"] = kw["bbox"]
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:3577")

    monkeypatch.setattr(pyogrio, "read_info", lambda *a, **kw: {"crs": "EPSG:3577"})
    monkeypatch.setattr(gpd, "read_file", _fake_read_file)

    # Input bounds in EPSG:4326 (lon/lat-shaped numbers); the layer is 3577
    # (metres), so a correctly reprojected bbox must NOT look like the input
    # (lon/lat magnitudes) and must NOT equal a naive EPSG:4326 bbox either.
    bounds_4326 = (130.0, -20.0, 131.0, -19.0)
    load_waterbodies(bounds_4326, "EPSG:4326")

    bbox = captured["bbox"]
    expected = tuple(
        gpd.GeoSeries([box(*bounds_4326)], crs="EPSG:4326").to_crs("EPSG:3577").total_bounds
    )
    assert bbox == pytest.approx(expected)
    # Reprojected bbox (metres, EPSG:3577) is nowhere near the raw lon/lat
    # input values -- confirms it was NOT left in the input CRS.
    assert all(abs(v) > 1000 for v in bbox)
