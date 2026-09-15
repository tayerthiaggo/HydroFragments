"""Unit tests for the DEA STAC/WFS loaders behind riverscape zoning's
terrain, cover, and waterbody evidence.

Every DEA-touching function is exercised by monkeypatching the real
`pystac_client`/`odc.stac`/`geopandas` module attributes as fakes -- the
same convention `tests/integration/test_dea_workflow.py` uses for
hydroseason. No test here makes a real network call.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
xr = pytest.importorskip("xarray")
odc_stac = pytest.importorskip("odc.stac")
pystac_client = pytest.importorskip("pystac_client")

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


def test_load_dem_raises_riverscape_source_unavailable_when_all_stac_urls_fail(
    monkeypatch,
) -> None:
    def _raise_open(url, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(pystac_client.Client, "open", staticmethod(_raise_open))
    monkeypatch.setattr(odc_stac, "configure_rio", lambda **kw: None)

    with pytest.raises(RiverscapeSourceUnavailable, match="ga_srtm_dem1sv1_0"):
        load_dem(_FakeGeobox(), product="ga_srtm_dem1sv1_0", band="dem_s")


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


def test_load_fc_percentiles_rejects_end_year_before_start_year() -> None:
    with pytest.raises(ValueError, match="years end"):
        load_fc_percentiles(
            _FakeGeobox(), product="ga_ls_fc_pc_cyear_3", bands=["bs_pc_50"], years=(2023, 2020)
        )


def test_load_waterbodies_returns_empty_geodataframe_when_no_features(monkeypatch) -> None:
    empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    monkeypatch.setattr(gpd, "read_file", lambda *a, **kw: empty)

    result = load_waterbodies((0.0, 0.0, 100.0, 100.0), "EPSG:3577")

    assert result.empty


def test_load_waterbodies_raises_riverscape_source_unavailable_on_read_failure(
    monkeypatch,
) -> None:
    def _raise_read(*a, **kw):
        raise RuntimeError("WFS endpoint unreachable")

    monkeypatch.setattr(gpd, "read_file", _raise_read)

    with pytest.raises(RiverscapeSourceUnavailable, match="Waterbodies"):
        load_waterbodies((0.0, 0.0, 100.0, 100.0), "EPSG:3577")
