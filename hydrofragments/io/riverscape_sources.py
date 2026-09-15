"""DEA STAC/WFS loaders for riverscape zoning's terrain, cover, and waterbody
evidence.

Deliberate exception to "hydroseason owns STAC access" (see the module
docstrings of ``io/dea.py`` and ``workflow.py``, both updated alongside this
module): riverscape zoning's DEM, Fractional Cover, and DEA Waterbodies
inputs have no hydroseason equivalent, so this module owns
``pystac-client``/``odc.stac`` access for them directly. hydroseason's own
DEA WO Statistics loader (``io/dea.py``) is unaffected and remains the only
acquisition path for zoning's water-frequency input.

Every loader raises :class:`RiverscapeSourceUnavailable` on a failed or
empty search/read -- never a silent empty result standing in for a real
failure -- so a caller in ``riverscape.mode="auto"`` can catch this one
exception type to fall back to occurrence zoning (spec §4.3), and a caller
in ``mode="required"`` can let it propagate unchanged.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import geopandas as gpd
import odc.stac
import pyogrio
import pystac_client
import xarray as xr
from shapely.geometry import box

STAC_URLS: tuple[str, ...] = (
    "https://explorer.sandbox.dea.ga.gov.au/stac",
    "https://explorer.dea.ga.gov.au/stac",
)
WATERBODIES_WFS_URL = "https://geoserver.dea.ga.gov.au/geoserver/wfs"
WATERBODIES_TYPENAME = "dea:DigitalEarthAustraliaWaterbodies_v3"


class RiverscapeSourceUnavailable(RuntimeError):
    """Raised when a riverscape evidence source cannot be loaded."""


def _search(collection: str, bbox: Sequence[float], *, time_range: str | None = None) -> list[Any]:
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    errors: dict[str, str] = {}
    for url in STAC_URLS:
        try:
            client = pystac_client.Client.open(url, timeout=(10, 120))
            kwargs: dict[str, Any] = {"collections": [collection], "bbox": list(bbox), "limit": 1000}
            if time_range is not None:
                kwargs["datetime"] = time_range
            items = list(client.search(**kwargs).items())
            if items:
                return items
            errors[url] = "no items"
        except Exception as exc:  # noqa: BLE001 -- fold every backend failure into one contract
            errors[url] = f"{type(exc).__name__}: {exc}"
    raise RiverscapeSourceUnavailable(f"{collection} unavailable from {STAC_URLS}: {errors}")


def _mask_nodata(data: xr.DataArray) -> xr.DataArray:
    """Convert a declared nodata sentinel to NaN without reducing time."""
    nodata = data.attrs.get("nodata")
    if nodata is None:
        nodata = data.odc.nodata
    if nodata is None:
        return data
    return data.where(data != nodata)


def _reduce_dem_time(data: xr.DataArray) -> xr.DataArray:
    """Reduce a repeated DEM time dimension after masking nodata."""
    masked = _mask_nodata(data)
    return masked.median("time", skipna=True) if "time" in masked.dims else masked


def load_dem(geobox: Any, *, product: str, band: str) -> xr.DataArray:
    """Load one DEM band onto ``geobox``.

    ``geobox`` is typically ``WoStatistics.frequency.odc.geobox`` (or
    anything ``odc.stac.load``'s ``geobox=`` argument accepts), so the
    returned array shares the zoning grid contract by construction --
    callers still validate this explicitly at the point of use via
    ``hydrofragments.output.spatial.SpatialGrid.validate_dataarray``, this
    function does not assert grid equality itself.
    """
    bbox_ll = list(geobox.extent.to_crs("EPSG:4326").boundingbox)
    items = _search(product, bbox_ll)
    try:
        dataset = odc.stac.load(items, bands=[band], geobox=geobox, resampling="bilinear")
        if band not in dataset:
            raise RiverscapeSourceUnavailable(f"{product} item(s) missing band {band!r}")
    except RiverscapeSourceUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 -- fold every read failure into one contract
        raise RiverscapeSourceUnavailable(
            f"{product} band {band!r} read failed: {type(exc).__name__}: {exc}"
        ) from exc
    return _reduce_dem_time(dataset[band])


def load_fc_percentiles(
    geobox: Any, *, product: str, bands: Sequence[str], years: tuple[int, int]
) -> dict[str, xr.DataArray]:
    """Load Fractional Cover percentile bands onto ``geobox`` for ``years``.

    ``years`` is an inclusive ``(start_year, end_year)`` pair. The time
    dimension is preserved (one step per year); nodata sentinels are
    converted to NaN. Scientific reductions (e.g. bare-year fractions)
    belong in ``hydrofragments.riverscape.evidence``. Returns one
    DataArray per requested band name.
    """
    start_year, end_year = years
    if end_year < start_year:
        raise ValueError(f"years end ({end_year}) must not precede start ({start_year})")
    time_range = f"{start_year}-01-01/{end_year}-12-31"
    bbox_ll = list(geobox.extent.to_crs("EPSG:4326").boundingbox)
    items = _search(product, bbox_ll, time_range=time_range)

    result: dict[str, xr.DataArray] = {}
    for band in bands:
        try:
            dataset = odc.stac.load(items, bands=[band], geobox=geobox, resampling="nearest")
            if band not in dataset:
                raise RiverscapeSourceUnavailable(f"{product} item(s) missing band {band!r}")
        except RiverscapeSourceUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 -- fold every read failure into one contract
            raise RiverscapeSourceUnavailable(
                f"{product} band {band!r} read failed: {type(exc).__name__}: {exc}"
            ) from exc
        result[band] = _mask_nodata(dataset[band])
    return result


def load_waterbodies(
    bounds: Sequence[float], crs: str, *, source: str | None = None
) -> "gpd.GeoDataFrame":
    """Load DEA Waterbodies polygons intersecting ``bounds`` (in ``crs``).

    ``source`` overrides the default WFS route (Phase 0's decision: WFS
    ``geoserver.dea.ga.gov.au``, typename
    ``dea:DigitalEarthAustraliaWaterbodies_v3``) with a different WFS base
    URL. Uses GDAL's ``WFS:`` driver prefix with a native ``bbox=`` filter
    (the same bbox-pushdown idiom ``scripts/spikes/widen_fitzroy_aoi.py``
    uses for a local geodatabase), so both routes share one code path.
    ``pyogrio``'s ``bbox=`` filter is applied in the dataset's own native
    CRS, not necessarily EPSG:4326 -- the layer's actual CRS is looked up
    first via ``pyogrio.read_info`` and ``bounds`` is reprojected into
    *that* CRS before being used as the filter, so this is correct even if
    the WFS layer's native CRS isn't 4326 (a bbox left in the wrong CRS
    would silently return zero features instead of raising). An empty
    result (no waterbodies in the AOI) is a valid, non-error outcome and is
    returned as an empty GeoDataFrame -- always reprojected to ``crs``, the
    same as a non-empty result, so callers never observe the CRS varying
    with feature count.
    """
    wfs_url = source or WATERBODIES_WFS_URL
    try:
        layer_info = pyogrio.read_info(f"WFS:{wfs_url}", layer=WATERBODIES_TYPENAME)
    except Exception as exc:  # noqa: BLE001
        raise RiverscapeSourceUnavailable(
            f"DEA Waterbodies unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    layer_crs = layer_info.get("crs") or "EPSG:4326"
    bounds_layer_crs = tuple(
        gpd.GeoSeries([box(*bounds)], crs=crs).to_crs(layer_crs).total_bounds
    )
    try:
        polygons = gpd.read_file(
            f"WFS:{wfs_url}", layer=WATERBODIES_TYPENAME, bbox=bounds_layer_crs, engine="pyogrio"
        )
    except Exception as exc:  # noqa: BLE001
        raise RiverscapeSourceUnavailable(
            f"DEA Waterbodies unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    if polygons.crs is None:
        polygons = polygons.set_crs(layer_crs, allow_override=True)
    return polygons.to_crs(crs)


__all__ = [
    "RiverscapeSourceUnavailable",
    "load_dem",
    "load_fc_percentiles",
    "load_waterbodies",
]
