"""Phase 0 rerun at basin scale: AHGF-offset, UpstrDArea and Fractional
Cover distributions over the full Fitzroy River basin (data/fitzroy_basin_*
from scripts/spikes/widen_fitzroy_aoi.py).

Does NOT re-decide the DEM band, FC band names, or the DEA Waterbodies route --
those came from Plan 1's Phase 0 spike and don't depend on AOI size. This
script only re-measures the *distributions* that were degenerate on the
narrow AOI.

Loads one Fractional Cover band at a time (float32, freed before the next)
to bound peak memory at basin scale -- Phase 0's spike loaded all 9 bands
simultaneously as float64, which would be ~15 GB at this AOI's size.

Not part of the package and not run by pytest.

Run:  python scripts/spikes/riverscape_phase0_basin_spike.py
Out:  output/spikes/riverscape_phase0_basin.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import odc.geo.xr  # noqa: F401
import odc.stac
import pystac_client
from rasterio.features import rasterize
from scipy import ndimage
from skimage.morphology import medial_axis

from hydrofragments.io.dea import open_wo_statistics_for_zoning

REPO = Path(__file__).resolve().parents[2]
AOI_PATH = REPO / "data" / "fitzroy_basin_aoi.geojson"
DRAINAGE_PATH = REPO / "data" / "fitzroy_basin_drainage.gpkg"
OUT_PATH = REPO / "output" / "spikes" / "riverscape_phase0_basin.json"

STAC_URLS = (
    "https://explorer.sandbox.dea.ga.gov.au/stac",
    "https://explorer.dea.ga.gov.au/stac",
)
PIXEL_M = 30.0
SEED_FREQUENCY_PCT = 5.0
TROUGH_RADIUS_PX = 5
TROUGH_DEPTH_M = 0.5
MAX_LINE_OFFSET_PX = 33  # ~990 m
FC_TIME_RANGE = "2022-01-01/2023-12-31"
DEM_BAND = "dem_s"  # decided in Plan 1 Phase 0; not re-decided here
FC_BANDS = ("bs_pc_50", "pv_pc_50", "npv_pc_50")


def _search(collection: str, bbox: list[float], time_range: str | None = None):
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    errors: dict[str, str] = {}
    for url in STAC_URLS:
        try:
            client = pystac_client.Client.open(url, timeout=(10, 120))
            kwargs: dict[str, Any] = {"collections": [collection], "bbox": bbox, "limit": 1000}
            if time_range is not None:
                kwargs["datetime"] = time_range
            items = list(client.search(**kwargs).items())
            if items:
                return url, items
            errors[url] = "no items"
        except Exception as exc:
            errors[url] = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"{collection} unavailable: {errors}")


def _percentiles_px(distances_px: np.ndarray) -> dict[str, float] | None:
    if distances_px.size == 0:
        return None
    metres = distances_px.astype(float) * PIXEL_M
    return {
        "p50_m": float(np.percentile(metres, 50)),
        "p95_m": float(np.percentile(metres, 95)),
        "n": int(metres.size),
    }


def _band_array(dataset, band: str) -> tuple[np.ndarray, float]:
    """Load one band eagerly as float32 (~800MB at this AOI's pixel count).

    Masks the band's declared nodata sentinel to NaN *before* any time
    reduction. Diagnostic finding during this run (not in the original brief
    text, but the exact gap the Phase 0 findings doc already warned Plan 2
    about -- "loaders must mask against the declared nodata value ... rather
    than trusting np.isfinite on a raw cast"): at basin scale this bites in
    two concrete ways if skipped --
    (1) FC bands declare nodata=255 (uint8). ~84% of this basin's seed-water
        pixels are FC-nodata (FC has no land-cover estimate over persistent
        water), so an unmasked ``nanmedian`` over the water mask returns
        exactly 255.0 -- the sentinel, not a cover fraction. Masking after
        ``.median("time")`` is also insufficient: a pixel with a mix of
        valid/nodata observations across time would already have 255 values
        pulling its per-pixel temporal median, producing implausible
        results (observed: an on-water max of 176.5 on a ~0-100 band).
    (2) The DEM's nodata sentinel is a large-magnitude finite float, not
        NaN, so ``np.isfinite`` alone does not catch it (finite fraction
        read 99.87%). Left unmasked, ~264k such cells feed
        ``ndimage.uniform_filter``'s box-average and blow up ``local_mean``
        to NaN/extreme values across virtually the whole raster, so
        ``depth >= TROUGH_DEPTH_M`` is never true and trough_pixels reads 0
        even though real terrain relief exists.

    Raises ``RuntimeError`` if the band carries no declared ``nodata``
    attribute. A missing attribute must not silently skip masking --
    letting the band flow through unmasked would silently reproduce the
    exact degenerate-output bug fixed here (see "Nodata-masking bug found
    and fixed during this task" in
    docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md), with
    nothing raised and nothing recorded. Per the project rule, invalid
    inputs raise errors naming the violated contract rather than falling
    back silently.

    Returns the masked/reduced array and the declared nodata value used to
    mask it, so callers can record both in ``findings`` and make masking
    self-evidencing in the JSON output.
    """
    data = dataset[band]
    nodata = data.attrs.get("nodata")
    if nodata is None:
        raise RuntimeError(
            f"{band}: no declared nodata attribute -- contract requires masking "
            f"against a declared sentinel before reduction (see basin-scale rerun bug)"
        )
    print(f"  {band}: declared nodata = {nodata!r}")
    values = data.astype("float32")
    values = values.where(values != float(nodata))
    if "time" in values.dims:
        values = values.median("time", skipna=True)
    return np.asarray(values, dtype=np.float32), float(nodata)


def _masked_median(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return float(np.nanmedian(values[mask]))


def _fix_proj_data_env() -> None:
    """Clear an inherited PROJ_LIB/PROJ_DATA and point pyproj at rasterio's own.

    Adjustment made during this run (not in the original brief text, but the
    same class of env-var-scoping gotcha the brief warned about): this
    machine has a system-wide ``PROJ_LIB`` pointing at a PostGIS install's
    ``proj.db`` (under PostgreSQL's ``contrib\\postgis-3.4\\proj``), whose
    DATABASE.LAYOUT.VERSION.MINOR is older than what this repo's
    pyproj/PROJ expects. Left as inherited, every CRS-transform call (e.g.
    building the WO-statistics geobox, later STAC loads) fails with
    ``pyproj.exceptions.ProjError: Error creating Transformer from CRS.``
    Unsetting it and pointing PROJ_DATA at rasterio's bundled proj database
    (the same fix used by ``riverscape_phase0_spike.py``) resolves it.
    """
    os.environ.pop("PROJ_LIB", None)
    os.environ.pop("PROJ_DATA", None)
    try:
        import rasterio

        proj_data = Path(rasterio.__file__).parent / "proj_data"
        if proj_data.exists():
            os.environ["PROJ_DATA"] = str(proj_data)
    except Exception:
        pass


def main() -> int:
    _fix_proj_data_env()
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    findings: dict[str, Any] = {"errors": {}}

    aoi = gpd.read_file(AOI_PATH)
    stats = open_wo_statistics_for_zoning(aoi)
    geobox = stats.frequency.odc.geobox
    bbox_ll = list(geobox.extent.to_crs("EPSG:4326").boundingbox)
    findings["grid"] = {
        "crs": str(geobox.crs),
        "shape": list(geobox.shape),
        "pixel_count": int(geobox.shape[0] * geobox.shape[1]),
    }
    print(f"grid: {geobox.shape}, {findings['grid']['pixel_count']:,} pixels")

    frequency = np.asarray(stats.frequency, dtype=np.float32)
    water = np.isfinite(frequency) & (frequency >= SEED_FREQUENCY_PCT)
    del frequency
    print(f"water-seed mask: {water.sum():,} / {water.size:,} px")
    skeleton = medial_axis(water)

    drainage = gpd.read_file(DRAINAGE_PATH).to_crs(str(geobox.crs))
    lines = rasterize(
        ((geom, 1) for geom in drainage.geometry if geom is not None and not geom.is_empty),
        out_shape=geobox.shape,
        transform=geobox.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    line_to_skeleton = ndimage.distance_transform_edt(~skeleton)[lines]
    findings["ahgf_offset"] = {
        "line_to_skeleton_all": _percentiles_px(line_to_skeleton),
        "line_to_skeleton_within_1km": _percentiles_px(
            line_to_skeleton[line_to_skeleton <= MAX_LINE_OFFSET_PX]
        ),
        "reach_count": int(len(drainage)),
    }
    upstr = drainage["UpstrDArea"].astype(float)
    quantile_probs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    findings["ahgf_offset"]["upstr_darea_m2_quantiles"] = {
        str(q): float(v) for q, v in zip(quantile_probs, upstr.quantile(quantile_probs))
    }
    print("AHGF offset within 1km:", findings["ahgf_offset"]["line_to_skeleton_within_1km"])

    trough = np.zeros(geobox.shape, dtype=bool)
    try:
        dem_url, dem_items = _search("ga_srtm_dem1sv1_0", bbox_ll)
        dem_ds = odc.stac.load(
            dem_items, bands=[DEM_BAND], geobox=geobox, resampling="bilinear"
        )
        dem, dem_nodata = _band_array(dem_ds, DEM_BAND)
        dem_finite_fraction = float(np.isfinite(dem).mean())
        local_mean = ndimage.uniform_filter(
            np.where(np.isfinite(dem), dem, np.nanmean(dem)), size=2 * TROUGH_RADIUS_PX + 1
        )
        trough = np.isfinite(dem) & ((local_mean - dem) >= TROUGH_DEPTH_M)
        findings["dem"] = {
            "stac_url": dem_url,
            "band": DEM_BAND,
            "nodata": dem_nodata,
            "masked": True,
            "finite_fraction": dem_finite_fraction,
            "trough_pixels": int(trough.sum()),
        }
        print("DEM finite fraction (post nodata-mask):", dem_finite_fraction)
        print("DEM trough pixels:", findings["dem"]["trough_pixels"])
        del dem, local_mean, dem_ds
        gc.collect()
    except Exception as exc:
        findings["errors"]["dem"] = f"{type(exc).__name__}: {exc}"

    try:
        fc_url, fc_items = _search("ga_ls_fc_pc_cyear_3", bbox_ll, FC_TIME_RANGE)
        band_stats: dict[str, Any] = {}
        for band in FC_BANDS:
            fc_ds = odc.stac.load(
                fc_items, bands=[band], geobox=geobox, resampling="nearest"
            )
            values, band_nodata = _band_array(fc_ds, band)
            band_stats[band] = {
                "nodata": band_nodata,
                "masked": True,
                "median_on_seed_water": _masked_median(values, water),
                "median_on_trough": _masked_median(values, trough),
                "median_aoi": float(np.nanmedian(values)),
            }
            print(band, band_stats[band])
            del values, fc_ds
            gc.collect()
        findings["fc"] = {
            "stac_url": fc_url,
            "item_count": len(fc_items),
            "bands": list(FC_BANDS),
            "band_stats": band_stats,
        }
    except Exception as exc:
        findings["errors"]["fc"] = f"{type(exc).__name__}: {exc}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
    print(json.dumps(findings, indent=2, default=str))
    print(f"wrote {OUT_PATH}")
    return 1 if findings["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
