# Riverscape Zoning — Plan 1: Data Spike and Layer Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confirm the external evidence data on Fitzroy (Phase 0) and land the source-independent foundation of riverscape zoning: the shared observed-wet domain, the independent hydroperiod classifier, and the landform × hydroperiod zone combination.

**Architecture:** Two independent layers feed zones. This plan builds the code-value contracts for both layers (`hydrofragments/riverscape/codes.py`, `hydrofragments/hydroperiod/codes.py`), the shared `wet_domain`, `classify_hydroperiod`, and `combine_zones` in `spatial/zones.py`. It does not build the landform evidence pipeline (Plans 2–4) and does not change `build_zones`, `analyze_from_dea`, config, or exports.

**Tech Stack:** Python 3.10+, numpy, xarray/rioxarray, scipy, scikit-image, odc-stac/odc-geo, pystac-client, geopandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-riverscape-zoning-design.md`

## Global Constraints

- Analysis domain: `(count_wet > 0) & isfinite(frequency) & (count_clear >= min_valid_obs)` on the native 30 m DEA WO statistics grid. Never the coarsened planning footprint / `analysis_mask`.
- Frequency is PERCENT (0–100); `t_persist`/`t_season` are FRACTIONS in `[0, 1]`, converted to percent once at the function boundary.
- Hydroperiod boundaries match `build_zones` exactly: marginal `freq < 100·t_season`; seasonal `100·t_season <= freq <= 100·t_persist`; persistent `freq > 100·t_persist`.
- Landform codes: `0 outside`, `1 in_channel`, `2 off_channel_riverine`, `3 non_riverine`.
- Hydroperiod codes: `0 outside`, `1 persistent`, `2 seasonal`, `3 marginal`, `4 unobserved` (bridged channel only; never inferred).
- Cross-tab: `landform*10 + hydroperiod`, `0` where landform is `0`.
- Legacy zones: Z1 = landform 1 (any hydroperiod); Z2 = landform 2 & persistent; Z3 = landform 2 & seasonal; Z4 = landform 2 & marginal; else 0.
- `hydrofragments/hydroperiod/` must not import `hydrofragments.riverscape`.
- `hydrofragments/riverscape/__init__.py` and both `codes.py` modules import only numpy/stdlib and each other, because `spatial/zones.py` imports the codes (avoids import cycles when later plans add heavier riverscape modules).
- Config is not changed in this plan: hydroperiod reuses `ZonesConfig.t_persist` / `t_season`; `RiverscapeConfig` and the `SCIENTIFIC_HASH_SCHEMA_VERSION` bump move to Plan 2 because their defaults depend on the Phase 0 findings.
- `build_zones` behaviour and all existing tests in `tests/spatial/test_zones.py` stay unchanged.
- No silent fallbacks: invalid inputs raise `ValueError` with a message naming the violated contract.
- Test file basenames must be unique across `tests/` (no `__init__.py` in test dirs).

## Roadmap (later plans, written after this one lands)

| Plan | Scope | Depends on |
|---|---|---|
| 1 (this) | Phase 0 spike + domain, hydroperiod, zone combination | — |
| 2 | Evidence loaders (`io/riverscape_sources.py`), `RiverscapeConfig` + hash bump, corridor, centreline, REM | Plan 1 findings doc |
| 3 | Channel rules, DEA Waterbodies roles, gap bridging (FC bare/green/npv costs) | Plan 2 |
| 4 | Riverine vs non-riverine, `build_landform` pipeline, workflow/exports/manifest integration, gating | Plan 3 |
| 5 | Windowed execution, Fitzroy validation harness | Plan 4 |

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `scripts/spikes/riverscape_phase0_spike.py` | Create | One-off network spike: DEM/FC/Waterbodies access, grid alignment, AHGF and trough offsets |
| `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md` | Create | Recorded spike results and the `dem_s` vs `dem_h` decision |
| `hydrofragments/riverscape/__init__.py` | Create | Package marker; re-exports codes and `wet_domain` |
| `hydrofragments/riverscape/codes.py` | Create | Landform code constants |
| `hydrofragments/riverscape/domain.py` | Create | `wet_domain` |
| `hydrofragments/hydroperiod/__init__.py` | Create | Package marker; re-exports codes and classifier |
| `hydrofragments/hydroperiod/codes.py` | Create | Hydroperiod code constants |
| `hydrofragments/hydroperiod/classify.py` | Create | `HydroperiodResult`, `classify_hydroperiod` |
| `hydrofragments/spatial/zones.py` | Modify | Extend `ZoneResult`; add `combine_zones` |
| `tests/riverscape/test_riverscape_domain.py` | Create | Domain tests |
| `tests/hydroperiod/test_hydroperiod_classify.py` | Create | Classifier + import-boundary tests |
| `tests/spatial/test_zone_combination.py` | Create | `ZoneResult` extension + `combine_zones` tests |

---

### Task 1: Phase 0 data-access spike and findings

**Files:**
- Create: `scripts/spikes/riverscape_phase0_spike.py`
- Create: `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md`

**Interfaces:**
- Consumes: `hydrofragments.io.dea.open_wo_statistics_for_zoning(aoi) -> WoStatistics` (existing, `hydrofragments/io/dea.py:55`).
- Produces: `output/spikes/riverscape_phase0.json` and the findings doc. Plan 2 reads the findings doc for: DEM STAC URL + default band, FC band names, Waterbodies access route/typename, AHGF→skeleton offset percentiles.

- [ ] **Step 1: Write the spike script**

```python
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
```

- [ ] **Step 2: Run the spike**

Run: `python scripts/spikes/riverscape_phase0_spike.py`
Expected: JSON printed, `wrote .../output/spikes/riverscape_phase0.json`, exit code 0. If exit code is 1, read `errors` in the JSON, fix the access route (e.g. collection only on one explorer, different WFS typename, band naming) inside the script, and rerun until `errors` is `{}`. Do not proceed with any section still in `errors`.

- [ ] **Step 3: Record findings and the DEM band decision**

Create `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md` with the values copied from `output/spikes/riverscape_phase0.json`, in exactly this structure:

```markdown
# Riverscape Zoning — Phase 0 Findings (Fitzroy, Kimberley)

**Date:** <run date>
**Source JSON:** `output/spikes/riverscape_phase0.json` (not committed)
**Script:** `scripts/spikes/riverscape_phase0_spike.py`

## Grid
- CRS / shape / resolution: <grid.crs> / <grid.shape> / <grid.resolution>
- WO product / time span: <grid.wo_product> / <grid.wo_time_span>

## DEM (`ga_srtm_dem1sv1_0`)
- STAC URL: <dem.stac_url>
- Assets: <dem.assets>
- Grid equal after `odc.stac.load(geobox=...)`: <dem.grid_equal>

| Band | finite fraction | skeleton→trough p50 / p95 (m) | line→trough p50 / p95 (m) |
|---|---|---|---|
| dem_s | ... | ... | ... |
| dem_h | ... | ... | ... |

**Decision:** default `dem_band = <dem_s|dem_h>`.
Rule applied: choose the band with the lower skeleton→trough p50. If the two
p50 values differ by 30 m or less, choose `dem_s`, because `dem_h` troughs are
stream-burned at AHGF line positions (record both line→trough p50 values as
evidence of that bias).

## Fractional Cover percentiles (`ga_ls_fc_pc_cyear_3`)
- STAC URL / items: <fc.stac_url> / <fc.item_count>
- Percentile bands present: <fc.percentile_bands>
- Grid equal: <fc.grid_equal>
- Band medians (seed water / trough / AOI): table from `fc.band_stats`
- Chosen names: `bare_band = <bs_pc_50 or closest present>`,
  `green_band = <pv_pc_50 or closest present>`,
  `npv_band = <npv_pc_50 or closest present>`

## DEA Waterbodies
- Route: WFS `<waterbodies.wfs_url>`, typename `<waterbodies.typename_used>`
- Features in AOI bbox: <waterbodies.feature_count>
- Columns: <waterbodies.columns>

## AHGF alignment
- line→EO skeleton (lines within 1 km): p50 <..> m, p95 <..> m, n <..>
- Implied corridor seed: `p95 + max half-width`, clamped to 90–600 m
- `UpstrDArea` 5/50/95 %: <...> m²

## Implications for Plan 2
- <one bullet per deviation from the spec defaults, or "none">
```

- [ ] **Step 4: Commit**

```bash
git add scripts/spikes/riverscape_phase0_spike.py docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md
git commit -m "docs: record riverscape phase 0 data-access findings for Fitzroy"
```

---

### Task 2: Landform/hydroperiod codes and the shared wet domain

**Files:**
- Create: `hydrofragments/riverscape/__init__.py`
- Create: `hydrofragments/riverscape/codes.py`
- Create: `hydrofragments/riverscape/domain.py`
- Create: `hydrofragments/hydroperiod/__init__.py`
- Create: `hydrofragments/hydroperiod/codes.py`
- Test: `tests/riverscape/test_riverscape_domain.py`

**Interfaces:**
- Consumes: `hydrofragments.io.dea.WoStatistics` (fields `frequency`, `count_wet`, `count_clear`; numpy, dask or xarray-backed); `hydrofragments.spatial.zones.build_zones` (parity only).
- Produces:
  - `hydrofragments.riverscape.codes`: `LANDFORM_OUTSIDE = 0`, `LANDFORM_IN_CHANNEL = 1`, `LANDFORM_OFF_CHANNEL_RIVERINE = 2`, `LANDFORM_NON_RIVERINE = 3`, `LANDFORM_CODES: frozenset[int]`, `LANDFORM_NAMES: dict[int, str]`.
  - `hydrofragments.hydroperiod.codes`: `HYDROPERIOD_OUTSIDE = 0`, `HYDROPERIOD_PERSISTENT = 1`, `HYDROPERIOD_SEASONAL = 2`, `HYDROPERIOD_MARGINAL = 3`, `HYDROPERIOD_UNOBSERVED = 4`, `HYDROPERIOD_CODES: frozenset[int]`, `HYDROPERIOD_NAMES: dict[int, str]`.
  - `hydrofragments.riverscape.domain.wet_domain(stats: WoStatistics, *, min_valid_obs: int) -> np.ndarray` (2-D bool).

- [ ] **Step 1: Write the failing tests**

`tests/riverscape/test_riverscape_domain.py`:

```python
from __future__ import annotations

import inspect

import numpy as np
import pytest

from hydrofragments.io.dea import WoStatistics
from hydrofragments.riverscape import codes as landform_codes
from hydrofragments.riverscape.domain import wet_domain
from hydrofragments.hydroperiod import codes as hydroperiod_codes
from hydrofragments.spatial.zones import build_zones


def _stats(frequency, count_wet, count_clear) -> WoStatistics:
    return WoStatistics(
        frequency=frequency,
        count_wet=count_wet,
        count_clear=count_clear,
        product="ga_ls_wo_fq_myear_3",
        version="test",
        crs="EPSG:3577",
        time_span=None,
        provenance={},
    )


def test_landform_codes_are_the_spec_contract() -> None:
    assert landform_codes.LANDFORM_OUTSIDE == 0
    assert landform_codes.LANDFORM_IN_CHANNEL == 1
    assert landform_codes.LANDFORM_OFF_CHANNEL_RIVERINE == 2
    assert landform_codes.LANDFORM_NON_RIVERINE == 3
    assert landform_codes.LANDFORM_CODES == frozenset({0, 1, 2, 3})
    assert landform_codes.LANDFORM_NAMES == {
        1: "in_channel",
        2: "off_channel_riverine",
        3: "non_riverine",
    }


def test_hydroperiod_codes_are_the_spec_contract() -> None:
    assert hydroperiod_codes.HYDROPERIOD_OUTSIDE == 0
    assert hydroperiod_codes.HYDROPERIOD_PERSISTENT == 1
    assert hydroperiod_codes.HYDROPERIOD_SEASONAL == 2
    assert hydroperiod_codes.HYDROPERIOD_MARGINAL == 3
    assert hydroperiod_codes.HYDROPERIOD_UNOBSERVED == 4
    assert hydroperiod_codes.HYDROPERIOD_CODES == frozenset({0, 1, 2, 3, 4})
    assert hydroperiod_codes.HYDROPERIOD_NAMES == {
        1: "persistent",
        2: "seasonal",
        3: "marginal",
        4: "unobserved",
    }


def test_domain_requires_wet_finite_and_support() -> None:
    stats = _stats(
        frequency=np.array([[50.0, 50.0, np.nan, 50.0]]),
        count_wet=np.array([[5, 0, 5, 5]]),
        count_clear=np.array([[20, 20, 20, 19]]),
    )

    domain = wet_domain(stats, min_valid_obs=20)

    assert domain.dtype == bool
    assert domain.tolist() == [[True, False, False, False]]


def test_domain_matches_build_zones_valid_extent() -> None:
    rng = np.random.default_rng(20260914)
    frequency = rng.uniform(0.0, 100.0, size=(40, 30))
    frequency[rng.random((40, 30)) < 0.05] = np.nan
    count_wet = rng.integers(0, 4, size=(40, 30))
    count_clear = rng.integers(10, 40, size=(40, 30))
    stats = _stats(frequency, count_wet, count_clear)

    domain = wet_domain(stats, min_valid_obs=20)
    zoned = build_zones(
        frequency,
        max_wet_mask=count_wet > 0,
        valid_count=count_clear,
        min_valid_obs=20,
    ).mask > 0

    np.testing.assert_array_equal(domain, zoned)


def test_domain_rejects_shape_mismatch() -> None:
    stats = _stats(np.ones((2, 2)), np.ones((2, 3)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="share shape"):
        wet_domain(stats, min_valid_obs=1)


def test_domain_rejects_non_2d_frequency() -> None:
    stats = _stats(np.ones(4), np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="2-D"):
        wet_domain(stats, min_valid_obs=1)


def test_domain_rejects_min_valid_obs_below_one() -> None:
    stats = _stats(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError, match="min_valid_obs"):
        wet_domain(stats, min_valid_obs=0)


def test_domain_accepts_dask_backed_xarray() -> None:
    xr = pytest.importorskip("xarray")
    da = pytest.importorskip("dask.array")
    stats = _stats(
        frequency=xr.DataArray(da.from_array(np.array([[60.0, 5.0]]), chunks=(1, 1)), dims=("y", "x")),
        count_wet=xr.DataArray(da.from_array(np.array([[3, 1]]), chunks=(1, 1)), dims=("y", "x")),
        count_clear=xr.DataArray(da.from_array(np.array([[20, 20]]), chunks=(1, 1)), dims=("y", "x")),
    )

    assert wet_domain(stats, min_valid_obs=20).tolist() == [[True, True]]


def test_domain_has_no_planning_footprint_input() -> None:
    assert list(inspect.signature(wet_domain).parameters) == ["stats", "min_valid_obs"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/riverscape/test_riverscape_domain.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'hydrofragments.riverscape'`

- [ ] **Step 3: Implement the code modules and domain**

`hydrofragments/riverscape/codes.py`:

```python
"""Landform layer code contract (spec 2026-09-14, section 3.2)."""

from __future__ import annotations

LANDFORM_OUTSIDE = 0
LANDFORM_IN_CHANNEL = 1
LANDFORM_OFF_CHANNEL_RIVERINE = 2
LANDFORM_NON_RIVERINE = 3

LANDFORM_CODES = frozenset(
    {
        LANDFORM_OUTSIDE,
        LANDFORM_IN_CHANNEL,
        LANDFORM_OFF_CHANNEL_RIVERINE,
        LANDFORM_NON_RIVERINE,
    }
)

LANDFORM_NAMES = {
    LANDFORM_IN_CHANNEL: "in_channel",
    LANDFORM_OFF_CHANNEL_RIVERINE: "off_channel_riverine",
    LANDFORM_NON_RIVERINE: "non_riverine",
}

__all__ = [
    "LANDFORM_CODES",
    "LANDFORM_IN_CHANNEL",
    "LANDFORM_NAMES",
    "LANDFORM_NON_RIVERINE",
    "LANDFORM_OFF_CHANNEL_RIVERINE",
    "LANDFORM_OUTSIDE",
]
```

`hydrofragments/riverscape/domain.py`:

```python
"""Observed-wet analysis domain shared by the landform and hydroperiod layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hydrofragments.io.dea import WoStatistics


def wet_domain(stats: "WoStatistics", *, min_valid_obs: int) -> np.ndarray:
    """Return the native-grid observed-wet domain as a 2-D bool array.

    ``domain = (count_wet > 0) & isfinite(frequency) & (count_clear >=
    min_valid_obs)`` -- the same extent ``build_zones`` zones. It is derived
    only from the DEA WO statistics themselves; the coarsened planning
    footprint / ``analysis_mask`` must never be used for zoning, so this
    function deliberately has no parameter that could accept one.
    """
    if min_valid_obs < 1:
        raise ValueError("min_valid_obs must be at least 1")
    frequency = np.asarray(stats.frequency, dtype=float)
    count_wet = np.asarray(stats.count_wet)
    count_clear = np.asarray(stats.count_clear)
    if frequency.ndim != 2:
        raise ValueError("frequency must be a 2-D array")
    if count_wet.shape != frequency.shape or count_clear.shape != frequency.shape:
        raise ValueError("frequency, count_wet and count_clear must share shape")
    return (count_wet > 0) & np.isfinite(frequency) & (count_clear >= min_valid_obs)


__all__ = ["wet_domain"]
```

`hydrofragments/riverscape/__init__.py`:

```python
"""Riverscape landform layer: in-channel, off-channel riverine, non-riverine."""

from hydrofragments.riverscape.codes import (
    LANDFORM_CODES,
    LANDFORM_IN_CHANNEL,
    LANDFORM_NAMES,
    LANDFORM_NON_RIVERINE,
    LANDFORM_OFF_CHANNEL_RIVERINE,
    LANDFORM_OUTSIDE,
)
from hydrofragments.riverscape.domain import wet_domain

__all__ = [
    "LANDFORM_CODES",
    "LANDFORM_IN_CHANNEL",
    "LANDFORM_NAMES",
    "LANDFORM_NON_RIVERINE",
    "LANDFORM_OFF_CHANNEL_RIVERINE",
    "LANDFORM_OUTSIDE",
    "wet_domain",
]
```

`hydrofragments/hydroperiod/codes.py`:

```python
"""Hydroperiod layer code contract (spec 2026-09-14, section 3.3)."""

from __future__ import annotations

HYDROPERIOD_OUTSIDE = 0
HYDROPERIOD_PERSISTENT = 1
HYDROPERIOD_SEASONAL = 2
HYDROPERIOD_MARGINAL = 3
HYDROPERIOD_UNOBSERVED = 4

HYDROPERIOD_CODES = frozenset(
    {
        HYDROPERIOD_OUTSIDE,
        HYDROPERIOD_PERSISTENT,
        HYDROPERIOD_SEASONAL,
        HYDROPERIOD_MARGINAL,
        HYDROPERIOD_UNOBSERVED,
    }
)

HYDROPERIOD_NAMES = {
    HYDROPERIOD_PERSISTENT: "persistent",
    HYDROPERIOD_SEASONAL: "seasonal",
    HYDROPERIOD_MARGINAL: "marginal",
    HYDROPERIOD_UNOBSERVED: "unobserved",
}

__all__ = [
    "HYDROPERIOD_CODES",
    "HYDROPERIOD_MARGINAL",
    "HYDROPERIOD_NAMES",
    "HYDROPERIOD_OUTSIDE",
    "HYDROPERIOD_PERSISTENT",
    "HYDROPERIOD_SEASONAL",
    "HYDROPERIOD_UNOBSERVED",
]
```

`hydrofragments/hydroperiod/__init__.py` (classifier export is added in Task 3):

```python
"""Hydroperiod layer: how often observed-wet pixels are wet."""

from hydrofragments.hydroperiod.codes import (
    HYDROPERIOD_CODES,
    HYDROPERIOD_MARGINAL,
    HYDROPERIOD_NAMES,
    HYDROPERIOD_OUTSIDE,
    HYDROPERIOD_PERSISTENT,
    HYDROPERIOD_SEASONAL,
    HYDROPERIOD_UNOBSERVED,
)

__all__ = [
    "HYDROPERIOD_CODES",
    "HYDROPERIOD_MARGINAL",
    "HYDROPERIOD_NAMES",
    "HYDROPERIOD_OUTSIDE",
    "HYDROPERIOD_PERSISTENT",
    "HYDROPERIOD_SEASONAL",
    "HYDROPERIOD_UNOBSERVED",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/riverscape/test_riverscape_domain.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add hydrofragments/riverscape hydrofragments/hydroperiod tests/riverscape/test_riverscape_domain.py
git commit -m "feat: add landform/hydroperiod code contracts and shared wet domain"
```

---

### Task 3: Independent hydroperiod classifier

**Files:**
- Create: `hydrofragments/hydroperiod/classify.py`
- Modify: `hydrofragments/hydroperiod/__init__.py`
- Test: `tests/hydroperiod/test_hydroperiod_classify.py`

**Interfaces:**
- Consumes: `hydrofragments.hydroperiod.codes` (Task 2); `hydrofragments.spatial.zones.build_zones` (parity only, test side).
- Produces:
  - `HydroperiodResult` frozen dataclass: `classes: np.ndarray` (uint8), `method: str`, `t_persist: float`, `t_season: float`.
  - `classify_hydroperiod(frequency, domain, *, t_persist: float, t_season: float, unobserved_mask=None) -> HydroperiodResult`.

- [ ] **Step 1: Write the failing tests**

`tests/hydroperiod/test_hydroperiod_classify.py`:

```python
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from hydrofragments.hydroperiod import (
    HYDROPERIOD_MARGINAL,
    HYDROPERIOD_OUTSIDE,
    HYDROPERIOD_PERSISTENT,
    HYDROPERIOD_SEASONAL,
    HYDROPERIOD_UNOBSERVED,
    HydroperiodResult,
    classify_hydroperiod,
)
from hydrofragments.spatial.zones import build_zones

HYDROPERIOD_PACKAGE = Path(__file__).resolve().parents[2] / "hydrofragments" / "hydroperiod"


def _single(value: float, *, t_persist: float = 0.50, t_season: float = 0.10) -> int:
    result = classify_hydroperiod(
        np.array([[value]]),
        np.array([[True]]),
        t_persist=t_persist,
        t_season=t_season,
    )
    return int(result.classes[0, 0])


@pytest.mark.parametrize(
    ("frequency", "expected"),
    [
        (0.5, HYDROPERIOD_MARGINAL),
        (9.9, HYDROPERIOD_MARGINAL),
        (10.0, HYDROPERIOD_SEASONAL),
        (45.0, HYDROPERIOD_SEASONAL),
        (50.0, HYDROPERIOD_SEASONAL),
        (50.1, HYDROPERIOD_PERSISTENT),
        (100.0, HYDROPERIOD_PERSISTENT),
    ],
)
def test_percent_boundaries_match_spec(frequency: float, expected: int) -> None:
    assert _single(frequency) == expected


def test_result_carries_method_and_thresholds() -> None:
    result = classify_hydroperiod(
        np.array([[60.0]]), np.array([[True]]), t_persist=0.6, t_season=0.2
    )
    assert isinstance(result, HydroperiodResult)
    assert result.classes.dtype == np.uint8
    assert result.method == "wofs_multiyear"
    assert (result.t_persist, result.t_season) == (0.6, 0.2)


def test_pixels_outside_domain_are_outside_even_if_frequency_is_high() -> None:
    result = classify_hydroperiod(
        np.array([[90.0, 90.0]]),
        np.array([[True, False]]),
        t_persist=0.5,
        t_season=0.1,
    )
    assert result.classes.tolist() == [[HYDROPERIOD_PERSISTENT, HYDROPERIOD_OUTSIDE]]


def test_unobserved_mask_marks_bridged_pixels_outside_domain() -> None:
    result = classify_hydroperiod(
        np.array([[30.0, np.nan, 0.0]]),
        np.array([[True, False, False]]),
        t_persist=0.5,
        t_season=0.1,
        unobserved_mask=np.array([[False, True, False]]),
    )
    assert result.classes.tolist() == [
        [HYDROPERIOD_SEASONAL, HYDROPERIOD_UNOBSERVED, HYDROPERIOD_OUTSIDE]
    ]


def test_unobserved_mask_must_not_overlap_domain() -> None:
    with pytest.raises(ValueError, match="unobserved_mask must not overlap"):
        classify_hydroperiod(
            np.array([[30.0]]),
            np.array([[True]]),
            t_persist=0.5,
            t_season=0.1,
            unobserved_mask=np.array([[True]]),
        )


def test_non_finite_frequency_inside_domain_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite inside the domain"):
        classify_hydroperiod(
            np.array([[np.nan]]), np.array([[True]]), t_persist=0.5, t_season=0.1
        )


@pytest.mark.parametrize(("t_persist", "t_season"), [(0.5, 0.5), (0.4, 0.5), (1.1, 0.1), (0.5, -0.1)])
def test_invalid_thresholds_are_rejected(t_persist: float, t_season: float) -> None:
    with pytest.raises(ValueError, match="t_season < t_persist"):
        classify_hydroperiod(
            np.array([[30.0]]), np.array([[True]]), t_persist=t_persist, t_season=t_season
        )


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="share shape"):
        classify_hydroperiod(
            np.ones((2, 2)), np.ones((2, 3), dtype=bool), t_persist=0.5, t_season=0.1
        )
    with pytest.raises(ValueError, match="share shape"):
        classify_hydroperiod(
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
            t_persist=0.5,
            t_season=0.1,
            unobserved_mask=np.zeros((1, 2), dtype=bool),
        )


def test_matches_build_zones_classes_on_the_same_domain() -> None:
    rng = np.random.default_rng(7)
    frequency = rng.uniform(0.0, 100.0, size=(50, 50))
    frequency[::7, ::5] = 10.0
    frequency[::9, ::4] = 50.0
    count_wet = rng.integers(0, 3, size=(50, 50))
    count_clear = rng.integers(15, 30, size=(50, 50))
    domain = (count_wet > 0) & np.isfinite(frequency) & (count_clear >= 20)

    zones = build_zones(
        frequency, max_wet_mask=count_wet > 0, valid_count=count_clear, min_valid_obs=20
    ).mask
    classes = classify_hydroperiod(frequency, domain, t_persist=0.5, t_season=0.1).classes

    expected = np.select(
        [zones == 2, zones == 3, zones == 4],
        [HYDROPERIOD_PERSISTENT, HYDROPERIOD_SEASONAL, HYDROPERIOD_MARGINAL],
        default=HYDROPERIOD_OUTSIDE,
    )
    np.testing.assert_array_equal(classes, expected)


def test_hydroperiod_package_never_imports_riverscape() -> None:
    offenders: list[str] = []
    for path in sorted(HYDROPERIOD_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("hydrofragments.riverscape"):
                offenders.append(f"{path.name}: from {node.module}")
            if isinstance(node, ast.Import):
                offenders.extend(
                    f"{path.name}: import {alias.name}"
                    for alias in node.names
                    if alias.name.startswith("hydrofragments.riverscape")
                )
    assert offenders == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/hydroperiod/test_hydroperiod_classify.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'HydroperiodResult' from 'hydrofragments.hydroperiod'`

- [ ] **Step 3: Implement the classifier**

`hydrofragments/hydroperiod/classify.py`:

```python
"""Hydroperiod classifier on multi-year DEA WO frequency.

Independent of the landform layer by contract: this module never imports
``hydrofragments.riverscape``. Later classifiers (DEA WO seasonal summaries,
hydroseason ``end_dry`` snapshots) replace this module without touching the
landform layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydrofragments.hydroperiod.codes import (
    HYDROPERIOD_MARGINAL,
    HYDROPERIOD_PERSISTENT,
    HYDROPERIOD_SEASONAL,
    HYDROPERIOD_UNOBSERVED,
)


@dataclass(frozen=True)
class HydroperiodResult:
    classes: np.ndarray
    method: str
    t_persist: float
    t_season: float


def classify_hydroperiod(
    frequency,
    domain,
    *,
    t_persist: float,
    t_season: float,
    unobserved_mask=None,
) -> HydroperiodResult:
    """Classify observed-wet pixels as persistent, seasonal or marginal.

    ``frequency`` is PERCENT (0-100); ``t_persist``/``t_season`` are FRACTIONS
    converted to percent once here. Boundaries match ``build_zones``:
    marginal ``< t_season``, seasonal ``t_season..t_persist`` inclusive,
    persistent ``> t_persist``. Pixels outside ``domain`` are 0 unless
    ``unobserved_mask`` supplies them (bridged channel pixels), which get
    ``HYDROPERIOD_UNOBSERVED``; that class is never inferred from frequency
    or neighbours.
    """
    values = np.asarray(frequency, dtype=float)
    inside = np.asarray(domain, dtype=bool)
    if values.ndim != 2:
        raise ValueError("frequency must be a 2-D array")
    if inside.shape != values.shape:
        raise ValueError("frequency and domain must share shape")
    if not 0.0 <= t_season < t_persist <= 1.0:
        raise ValueError("hydroperiod thresholds require 0 <= t_season < t_persist <= 1")
    if not np.all(np.isfinite(values[inside])):
        raise ValueError("frequency must be finite inside the domain")

    t_persist_pct = t_persist * 100.0
    t_season_pct = t_season * 100.0

    classes = np.zeros(values.shape, dtype=np.uint8)
    classes[inside & (values < t_season_pct)] = HYDROPERIOD_MARGINAL
    classes[inside & (values >= t_season_pct) & (values <= t_persist_pct)] = HYDROPERIOD_SEASONAL
    classes[inside & (values > t_persist_pct)] = HYDROPERIOD_PERSISTENT

    if unobserved_mask is not None:
        unobserved = np.asarray(unobserved_mask, dtype=bool)
        if unobserved.shape != values.shape:
            raise ValueError("frequency and unobserved_mask must share shape")
        if np.any(unobserved & inside):
            raise ValueError("unobserved_mask must not overlap the observed domain")
        classes[unobserved] = HYDROPERIOD_UNOBSERVED

    return HydroperiodResult(
        classes=classes,
        method="wofs_multiyear",
        t_persist=float(t_persist),
        t_season=float(t_season),
    )


__all__ = ["HydroperiodResult", "classify_hydroperiod"]
```

Replace `hydrofragments/hydroperiod/__init__.py` with:

```python
"""Hydroperiod layer: how often observed-wet pixels are wet."""

from hydrofragments.hydroperiod.classify import HydroperiodResult, classify_hydroperiod
from hydrofragments.hydroperiod.codes import (
    HYDROPERIOD_CODES,
    HYDROPERIOD_MARGINAL,
    HYDROPERIOD_NAMES,
    HYDROPERIOD_OUTSIDE,
    HYDROPERIOD_PERSISTENT,
    HYDROPERIOD_SEASONAL,
    HYDROPERIOD_UNOBSERVED,
)

__all__ = [
    "HYDROPERIOD_CODES",
    "HYDROPERIOD_MARGINAL",
    "HYDROPERIOD_NAMES",
    "HYDROPERIOD_OUTSIDE",
    "HYDROPERIOD_PERSISTENT",
    "HYDROPERIOD_SEASONAL",
    "HYDROPERIOD_UNOBSERVED",
    "HydroperiodResult",
    "classify_hydroperiod",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/hydroperiod/test_hydroperiod_classify.py tests/riverscape/test_riverscape_domain.py -v`
Expected: all passed (19 in the hydroperiod file including parametrized cases, 9 in the domain file)

- [ ] **Step 5: Commit**

```bash
git add hydrofragments/hydroperiod tests/hydroperiod/test_hydroperiod_classify.py
git commit -m "feat: add independent WOfS multi-year hydroperiod classifier"
```

---

### Task 4: Zone combination (landform × hydroperiod)

**Files:**
- Modify: `hydrofragments/spatial/zones.py:1-38` (imports, `ZoneResult`) and append `combine_zones` before `__all__` (line 160)
- Test: `tests/spatial/test_zone_combination.py`

**Interfaces:**
- Consumes: `hydrofragments.riverscape.codes` and `hydrofragments.hydroperiod.codes` (Task 2).
- Produces:
  - `ZoneResult` gains `mode: str = "occurrence"`, `crosstab: np.ndarray | None = None`, `degraded_reasons: tuple[str, ...] = ()`; `__post_init__` validates `mode in {"occurrence", "riverscape"}`, riverscape requires `crosstab`, `crosstab.shape == mask.shape`.
  - `combine_zones(landform, hydroperiod, *, source: str = "riverscape", degraded_reasons: tuple[str, ...] = ()) -> ZoneResult` with `emitted_zones == (1, 2, 3, 4)`, `has_zone_1 is True`, `mode == "riverscape"`.
  - Plan 4 will add `zones_from_riverscape` on top of this.

- [ ] **Step 1: Write the failing tests**

`tests/spatial/test_zone_combination.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from hydrofragments.spatial.zones import ZoneResult, build_zones, combine_zones

VALID_PAIRS = [
    # (landform, hydroperiod, legacy_zone, crosstab)
    (0, 0, 0, 0),
    (1, 1, 1, 11),
    (1, 2, 1, 12),
    (1, 3, 1, 13),
    (1, 4, 1, 14),
    (2, 1, 2, 21),
    (2, 2, 3, 22),
    (2, 3, 4, 23),
    (3, 1, 0, 31),
    (3, 2, 0, 32),
    (3, 3, 0, 33),
]


@pytest.mark.parametrize(("landform", "hydroperiod", "zone", "crosstab"), VALID_PAIRS)
def test_mapping_table_matches_spec(landform: int, hydroperiod: int, zone: int, crosstab: int) -> None:
    result = combine_zones(np.array([[landform]]), np.array([[hydroperiod]]))

    assert int(result.mask[0, 0]) == zone
    assert int(result.crosstab[0, 0]) == crosstab


def test_combined_result_contract() -> None:
    landform = np.array([[1, 2], [3, 0]], dtype=np.uint8)
    hydroperiod = np.array([[4, 1], [2, 0]], dtype=np.uint8)

    result = combine_zones(landform, hydroperiod, source="riverscape", degraded_reasons=("envelope_single_bin",))

    assert result.mask.dtype == np.uint8
    assert result.crosstab.dtype == np.uint8
    assert result.mask.tolist() == [[1, 2], [0, 0]]
    assert result.crosstab.tolist() == [[14, 21], [32, 0]]
    assert result.emitted_zones == (1, 2, 3, 4)
    assert result.has_zone_1 is True
    assert result.mode == "riverscape"
    assert result.source == "riverscape"
    assert result.degraded_reasons == ("envelope_single_bin",)


@pytest.mark.parametrize(
    ("landform", "hydroperiod", "message"),
    [
        (0, 1, "disagree on the zoned extent"),
        (1, 0, "disagree on the zoned extent"),
        (2, 0, "disagree on the zoned extent"),
        (2, 4, "unobserved hydroperiod is only valid for in-channel"),
        (3, 4, "unobserved hydroperiod is only valid for in-channel"),
    ],
)
def test_inconsistent_layers_are_rejected(landform: int, hydroperiod: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        combine_zones(np.array([[landform]]), np.array([[hydroperiod]]))


def test_invalid_codes_are_rejected() -> None:
    with pytest.raises(ValueError, match="landform has invalid codes"):
        combine_zones(np.array([[5]]), np.array([[1]]))
    with pytest.raises(ValueError, match="hydroperiod has invalid codes"):
        combine_zones(np.array([[1]]), np.array([[7]]))


def test_shape_mismatch_and_dimensionality_are_rejected() -> None:
    with pytest.raises(ValueError, match="share shape"):
        combine_zones(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="2-D"):
        combine_zones(np.zeros(4), np.zeros(4))


def test_zone_result_defaults_keep_occurrence_contract() -> None:
    result = build_zones(
        np.array([[90.0]]),
        max_wet_mask=np.array([[True]]),
        valid_count=np.array([[20]]),
    )
    assert result.mode == "occurrence"
    assert result.crosstab is None
    assert result.degraded_reasons == ()


def test_zone_result_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        ZoneResult(mask=np.zeros((1, 1), dtype=np.uint8), emitted_zones=(), has_zone_1=False, mode="hybrid")


def test_riverscape_zone_result_requires_matching_crosstab() -> None:
    mask = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="crosstab"):
        ZoneResult(mask=mask, emitted_zones=(1, 2, 3, 4), has_zone_1=True, mode="riverscape")
    with pytest.raises(ValueError, match="crosstab"):
        ZoneResult(
            mask=mask,
            emitted_zones=(1, 2, 3, 4),
            has_zone_1=True,
            mode="riverscape",
            crosstab=np.zeros((1, 2), dtype=np.uint8),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/spatial/test_zone_combination.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'combine_zones' from 'hydrofragments.spatial.zones'`

- [ ] **Step 3: Extend `ZoneResult` and add `combine_zones`**

In `hydrofragments/spatial/zones.py`, replace the module docstring and imports block (lines 1-17) with:

```python
"""Zone masks: occurrence-only zoning and landform x hydroperiod combination."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from scipy import ndimage

import rioxarray  # noqa: F401 — registers the .rio accessor for DataArray

from hydrofragments.hydroperiod.codes import (
    HYDROPERIOD_CODES,
    HYDROPERIOD_MARGINAL,
    HYDROPERIOD_OUTSIDE,
    HYDROPERIOD_PERSISTENT,
    HYDROPERIOD_SEASONAL,
    HYDROPERIOD_UNOBSERVED,
)
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.riverscape.codes import (
    LANDFORM_CODES,
    LANDFORM_IN_CHANNEL,
    LANDFORM_OFF_CHANNEL_RIVERINE,
    LANDFORM_OUTSIDE,
)

if TYPE_CHECKING:
    from hydrofragments.io.dea import WoStatistics

_ZONE_MODES = frozenset({"occurrence", "riverscape"})
```

Replace the `ZoneResult` class header and fields (lines 20-26, keeping `as_dataarray` unchanged) with:

```python
@dataclass(frozen=True)
class ZoneResult:
    mask: np.ndarray
    emitted_zones: tuple[int, ...]
    has_zone_1: bool
    source: str = "occurrence"
    grid: SpatialGrid | None = None
    mode: str = "occurrence"
    crosstab: np.ndarray | None = None
    degraded_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in _ZONE_MODES:
            raise ValueError(f"ZoneResult mode must be one of {sorted(_ZONE_MODES)}")
        if self.mode == "riverscape" and self.crosstab is None:
            raise ValueError("riverscape ZoneResult requires a crosstab")
        if self.crosstab is not None and np.shape(self.crosstab) != np.shape(self.mask):
            raise ValueError("ZoneResult crosstab must share the mask shape")
```

Insert before `__all__` (current line 160):

```python
def _validated_codes(values: np.ndarray, allowed: frozenset[int], name: str) -> np.ndarray:
    invalid = sorted(set(np.unique(values).tolist()) - set(allowed))
    if invalid:
        raise ValueError(f"{name} has invalid codes: {invalid}")
    return values.astype(np.uint8)


def combine_zones(
    landform,
    hydroperiod,
    *,
    source: str = "riverscape",
    degraded_reasons: tuple[str, ...] = (),
) -> ZoneResult:
    """Derive zones from the landform and hydroperiod layers; no new logic.

    ``crosstab = landform * 10 + hydroperiod`` (0 outside). Legacy view:
    Z1 = in-channel (any hydroperiod, including unobserved bridges);
    Z2/Z3/Z4 = off-channel riverine x persistent/seasonal/marginal;
    non-riverine water is 0. The two layers must agree on the zoned extent,
    and ``unobserved`` is only valid on in-channel pixels.
    """
    landform_values = np.asarray(landform)
    hydroperiod_values = np.asarray(hydroperiod)
    if landform_values.ndim != 2:
        raise ValueError("landform must be a 2-D array")
    if hydroperiod_values.shape != landform_values.shape:
        raise ValueError("landform and hydroperiod must share shape")
    lf = _validated_codes(landform_values, LANDFORM_CODES, "landform")
    hp = _validated_codes(hydroperiod_values, HYDROPERIOD_CODES, "hydroperiod")

    if np.any((lf == LANDFORM_OUTSIDE) != (hp == HYDROPERIOD_OUTSIDE)):
        raise ValueError("landform and hydroperiod disagree on the zoned extent")
    if np.any((hp == HYDROPERIOD_UNOBSERVED) & (lf != LANDFORM_IN_CHANNEL)):
        raise ValueError("unobserved hydroperiod is only valid for in-channel landform")

    crosstab = np.where(lf == LANDFORM_OUTSIDE, 0, lf * 10 + hp).astype(np.uint8)
    off_channel = lf == LANDFORM_OFF_CHANNEL_RIVERINE
    mask = np.zeros(lf.shape, dtype=np.uint8)
    mask[lf == LANDFORM_IN_CHANNEL] = 1
    mask[off_channel & (hp == HYDROPERIOD_PERSISTENT)] = 2
    mask[off_channel & (hp == HYDROPERIOD_SEASONAL)] = 3
    mask[off_channel & (hp == HYDROPERIOD_MARGINAL)] = 4

    return ZoneResult(
        mask=mask,
        emitted_zones=(1, 2, 3, 4),
        has_zone_1=True,
        source=source,
        mode="riverscape",
        crosstab=crosstab,
        degraded_reasons=tuple(degraded_reasons),
    )
```

Replace the final `__all__` line with:

```python
__all__ = ["ZoneResult", "build_zones", "combine_zones", "zones_from_wo_statistics"]
```

- [ ] **Step 4: Run new and existing zone tests**

Run: `python -m pytest tests/spatial/test_zone_combination.py tests/spatial/test_zones.py tests/guards/test_scientific_guards.py tests/gating/test_zones_do_not_multiply_metrics.py -v`
Expected: all passed; `tests/spatial/test_zones.py` count unchanged from before this task.

- [ ] **Step 5: Check for circular imports from package entry points**

Run: `python -c "import hydrofragments, hydrofragments.api, hydrofragments.workflow, hydrofragments.spatial.zones; print('ok')"`
Expected: `ok` (a CuPy CUDA-path warning on stderr is pre-existing and acceptable).

- [ ] **Step 6: Commit**

```bash
git add hydrofragments/spatial/zones.py tests/spatial/test_zone_combination.py
git commit -m "feat: derive zones from landform and hydroperiod layers"
```

---

### Task 5: Full-suite regression gate

**Files:** none modified.

**Interfaces:**
- Consumes: everything above.
- Produces: confirmation that Plan 1 introduced no regressions; baseline count for Plan 2.

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q`
Expected: only the previously accepted failure (stale-bundle test) fails; every other test passes, including the new files (9 domain + 19 hydroperiod + 22 combination tests).

- [ ] **Step 2: If anything else fails**

Stop. Do not edit unrelated tests. Diagnose with `python -m pytest <failing test> -vv` and fix within the files this plan touches, then rerun Step 1.

- [ ] **Step 3: Record the baseline**

Append to `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md`:

```markdown
## Plan 1 completion
- Full suite: <N passed>, <M failed> (<name of accepted stale-bundle failure>)
- Commit: <git rev-parse --short HEAD>
```

```bash
git add docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md
git commit -m "docs: record plan 1 regression baseline for riverscape zoning"
```
