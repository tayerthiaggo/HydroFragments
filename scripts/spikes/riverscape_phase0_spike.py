"""Phase 0 data-access spike for riverscape zoning (Fitzroy, Kimberley).

Not part of the package and not run by pytest. Requires network access to
DEA STAC, DEA S3 COGs and DEA GeoServer. Every section records its own
failure verbatim in the findings JSON; the script exits non-zero if any
section failed, so a partial result is never mistaken for a complete one.

Run:  python scripts/spikes/riverscape_phase0_spike.py
Out:  output/spikes/riverscape_phase0.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import odc.geo.xr  # noqa: F401  (registers .odc accessor)
import odc.stac
import pystac_client
from rasterio.features import rasterize
from scipy import ndimage
from skimage.morphology import medial_axis

from hydrofragments.io.dea import open_wo_statistics_for_zoning

REPO = Path(__file__).resolve().parents[2]
AOI_PATH = REPO / "data" / "fitzroy_kimberley_aoi.geojson"
DRAINAGE_PATH = REPO / "data" / "fitzroy_kimberley_drainage.gpkg"
OUT_PATH = REPO / "output" / "spikes" / "riverscape_phase0.json"

STAC_URLS = (
    "https://explorer.sandbox.dea.ga.gov.au/stac",
    "https://explorer.dea.ga.gov.au/stac",
)
WFS_URL = "https://geoserver.dea.ga.gov.au/geoserver/wfs"
PIXEL_M = 30.0
SEED_FREQUENCY_PCT = 5.0
TROUGH_RADIUS_PX = 5  # 150 m
TROUGH_DEPTH_M = 0.5
MAX_LINE_OFFSET_PX = 33  # 1 km: ignore AHGF lines with no nearby EO water
FC_TIME_RANGE = "2022-01-01/2023-12-31"
FC_BAND_PATTERN = re.compile(r"^(bs|pv|npv)_pc_(10|50|90)$")


def _configure_unsigned_s3_env() -> None:
    """Persist unsigned-S3 GDAL/PROJ env vars for the whole script run.

    Adjustment made during the spike (not in the original brief text):
    ``open_wo_statistics_for_zoning`` (via hydroseason) scopes its own
    ``AWS_NO_SIGN_REQUEST``/GDAL env to the STAC search + ``odc.stac.load``
    call and restores the caller's prior (unset) environment in a
    ``finally`` before returning -- but the returned dataset is Dask-backed
    and lazy, so the actual S3 COG reads happen later, when this script
    first materializes ``stats.frequency``. Without this, that first
    ``np.asarray(stats.frequency, ...)`` call fails with
    ``CPLE_AWSInvalidCredentialsError`` because unsigned access has already
    been un-set by the time the read actually happens. Setting these here
    (via ``setdefault``, before calling ``open_wo_statistics_for_zoning``)
    means hydroseason's own snapshot/restore cycle restores back to *this*
    script's values instead of removing them, so unsigned access stays in
    effect for every S3 read this script triggers afterwards (the WO stats
    compute, and the later DEM/FC ``odc.stac.load`` calls in ``_search``).
    """
    os.environ.pop("PROJ_LIB", None)
    os.environ.pop("PROJ_DATA", None)
    defaults = {
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MAX_RETRY": "10",
        "GDAL_HTTP_RETRY_DELAY": "3",
        "GDAL_HTTP_RETRY_CODES": "429,500,502,503,504,520,522,524",
        "GDAL_HTTP_TIMEOUT": "30",
        "GDAL_HTTP_CONNECTTIMEOUT": "15",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    try:
        import rasterio

        proj_data = Path(rasterio.__file__).parent / "proj_data"
        if proj_data.exists():
            os.environ["PROJ_DATA"] = str(proj_data)
    except Exception:
        pass


def _search(collection: str, bbox: list[float], time_range: str | None = None):
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    errors: dict[str, str] = {}
    for url in STAC_URLS:
        try:
            client = pystac_client.Client.open(url, timeout=(10, 120))
            kwargs: dict[str, Any] = {"collections": [collection], "bbox": bbox, "limit": 100}
            if time_range is not None:
                kwargs["datetime"] = time_range
            items = list(client.search(**kwargs).items())
            if items:
                return url, items
            errors[url] = "no items"
        except Exception as exc:  # spike: record every failure verbatim
            errors[url] = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"{collection} unavailable: {errors}")


def _percentiles(distances_px: np.ndarray) -> dict[str, float] | None:
    if distances_px.size == 0:
        return None
    metres = distances_px.astype(float) * PIXEL_M
    return {
        "p50_m": float(np.percentile(metres, 50)),
        "p95_m": float(np.percentile(metres, 95)),
        "n": int(metres.size),
    }


def _distance_to(target: np.ndarray, sample: np.ndarray) -> np.ndarray:
    if not target.any():
        return np.array([], dtype=float)
    return ndimage.distance_transform_edt(~target)[sample]


def _trough(dem: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    size = 2 * TROUGH_RADIUS_PX + 1
    local_mean = ndimage.uniform_filter(filled, size=size)
    return np.isfinite(dem) & ((local_mean - dem) >= TROUGH_DEPTH_M)


def _band_array(dataset, band: str) -> np.ndarray:
    data = dataset[band]
    if "time" in data.dims:
        data = data.median("time")
    return np.asarray(data, dtype=float)


def _waterbodies(bbox_3577: tuple[float, float, float, float]) -> dict[str, Any]:
    result: dict[str, Any] = {"wfs_url": WFS_URL}
    caps_url = f"{WFS_URL}?service=WFS&version=2.0.0&request=GetCapabilities"
    caps = urllib.request.urlopen(caps_url, timeout=60).read().decode("utf-8", "replace")
    names = sorted(set(re.findall(r"<(?:wfs:)?Name>([^<]*[Ww]aterbod[^<]*)</(?:wfs:)?Name>", caps)))
    result["wfs_typenames"] = names
    if not names:
        raise RuntimeError("no Waterbodies typename in WFS capabilities")
    minx, miny, maxx, maxy = bbox_3577
    feature_url = (
        f"{WFS_URL}?service=WFS&version=2.0.0&request=GetFeature"
        f"&typeNames={names[0]}&bbox={minx},{miny},{maxx},{maxy},EPSG:3577"
        "&outputFormat=application/json"
    )
    gdf = gpd.read_file(feature_url)
    result.update(
        typename_used=names[0],
        feature_count=int(len(gdf)),
        columns=sorted(map(str, gdf.columns)),
        crs=str(gdf.crs),
    )
    return result


def main() -> int:
    findings: dict[str, Any] = {"errors": {}}

    _configure_unsigned_s3_env()

    aoi = gpd.read_file(AOI_PATH)
    stats = open_wo_statistics_for_zoning(aoi)
    geobox = stats.frequency.odc.geobox
    frequency = np.asarray(stats.frequency, dtype=float)
    bbox_ll = list(geobox.extent.to_crs("EPSG:4326").boundingbox)
    findings["grid"] = {
        "crs": str(geobox.crs),
        "shape": list(geobox.shape),
        "resolution": [float(v) for v in geobox.resolution.xy],
        "wo_product": stats.product,
        "wo_time_span": stats.time_span,
    }

    water = np.isfinite(frequency) & (frequency >= SEED_FREQUENCY_PCT)
    skeleton = medial_axis(water)
    findings["seed"] = {"water_pixels": int(water.sum()), "skeleton_pixels": int(skeleton.sum())}

    drainage = gpd.read_file(DRAINAGE_PATH).to_crs(str(geobox.crs))
    lines = rasterize(
        ((geom, 1) for geom in drainage.geometry if geom is not None and not geom.is_empty),
        out_shape=geobox.shape,
        transform=geobox.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    line_to_skeleton = _distance_to(skeleton, lines)
    findings["ahgf_offset"] = {
        "line_to_skeleton_all": _percentiles(line_to_skeleton),
        "line_to_skeleton_within_1km": _percentiles(
            line_to_skeleton[line_to_skeleton <= MAX_LINE_OFFSET_PX]
        ),
        "reach_count": int(len(drainage)),
        "upstr_darea_m2_quantiles": [
            float(q) for q in np.nanpercentile(drainage["UpstrDArea"].astype(float), [5, 50, 95])
        ],
    }

    try:
        dem_url, dem_items = _search("ga_srtm_dem1sv1_0", bbox_ll)
        dem_ds = odc.stac.load(dem_items, bands=["dem_s", "dem_h"], geobox=geobox, resampling="bilinear")
        findings["dem"] = {
            "stac_url": dem_url,
            "assets": sorted(dem_items[0].assets),
            "grid_equal": bool(dem_ds.odc.geobox == geobox),
            "bands": {},
        }
        for band in ("dem_s", "dem_h"):
            dem = _band_array(dem_ds, band)
            trough = _trough(dem)
            findings["dem"]["bands"][band] = {
                "finite_fraction": float(np.isfinite(dem).mean()),
                "trough_pixels": int(trough.sum()),
                "skeleton_to_trough": _percentiles(_distance_to(trough, skeleton)),
                "line_to_trough": _percentiles(_distance_to(trough, lines)),
            }
    except Exception as exc:
        findings["errors"]["dem"] = f"{type(exc).__name__}: {exc}"

    try:
        fc_url, fc_items = _search("ga_ls_fc_pc_cyear_3", bbox_ll, FC_TIME_RANGE)
        assets = sorted(fc_items[0].assets)
        fc_bands = [name for name in assets if FC_BAND_PATTERN.match(name)]
        fc_ds = odc.stac.load(fc_items, bands=fc_bands, geobox=geobox, resampling="nearest")
        trough_for_fc = _trough(_band_array(dem_ds, "dem_s")) if "dem" in findings else np.zeros(geobox.shape, bool)
        band_stats: dict[str, Any] = {}
        for band in fc_bands:
            values = _band_array(fc_ds, band)
            band_stats[band] = {
                "finite_fraction": float(np.isfinite(values).mean()),
                "median_on_seed_water": float(np.nanmedian(values[water])) if water.any() else None,
                "median_on_trough": float(np.nanmedian(values[trough_for_fc])) if trough_for_fc.any() else None,
                "median_aoi": float(np.nanmedian(values)),
            }
        findings["fc"] = {
            "stac_url": fc_url,
            "item_count": len(fc_items),
            "assets": assets,
            "percentile_bands": fc_bands,
            "grid_equal": bool(fc_ds.odc.geobox == geobox),
            "band_stats": band_stats,
        }
    except Exception as exc:
        findings["errors"]["fc"] = f"{type(exc).__name__}: {exc}"

    try:
        bounds = geobox.extent.boundingbox
        findings["waterbodies"] = _waterbodies((bounds.left, bounds.bottom, bounds.right, bounds.top))
    except Exception as exc:
        findings["errors"]["waterbodies"] = f"{type(exc).__name__}: {exc}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
    print(json.dumps(findings, indent=2, default=str))
    print(f"wrote {OUT_PATH}")
    return 1 if findings["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
