# Riverscape Zoning — Plan 2: AOI Widen, Zone-Semantics Fix, Config Skeleton

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the three items Plan 1's final review flagged as needing a decision before heavier riverscape code is written: widen the Fitzroy test AOI so its drainage-area distribution can actually calibrate a scaling law, fix `combine_zones`'s zone-reporting fields to reflect what was actually produced, and land a validated `RiverscapeConfig` skeleton whose defaults come from real, basin-wide numbers rather than the narrow AOI's degenerate ones.

**Architecture:** Two independent, small fixes (Tasks 1, 3) and one data-preparation task (Task 2) that produces the numbers Task 4 (`RiverscapeConfig`) needs. Task 2's raster work is written Dask-native throughout — no full-array `np.asarray` materialization — because the widened AOI is ~360× the pixel count of Plan 1's Phase 0 spike (~205M vs ~600K pixels) and the eager pattern from that spike would not scale.

**Tech Stack:** Python 3.10+, geopandas, pyogrio, numpy, dask, dask-image, scipy, scikit-image, odc-stac/odc-geo, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-riverscape-zoning-design.md` (§3.4, §4.2 step 4, §6, §10, §12)
**Prior findings:** `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md` (narrow-AOI Phase 0 — DEM band, FC band names, and DEA Waterbodies route decided there still hold; only the *distributions* need re-measuring at basin scale)

## Global Constraints

- `combine_zones` and `ZoneResult` live in `hydrofragments/spatial/zones.py`. `build_zones`'s behaviour and `tests/spatial/test_zones.py` (24 tests) must stay unchanged — every task in this plan that touches `zones.py` touches only the `combine_zones` function.
- Landform codes: `0 outside`, `1 in_channel`, `2 off_channel_riverine`, `3 non_riverine`. Hydroperiod codes: `0 outside`, `1 persistent`, `2 seasonal`, `3 marginal`, `4 unobserved`.
- No silent fallbacks: invalid inputs raise `ValueError` naming the violated contract; every calibration fallback is recorded, not silently applied.
- `hydrofragments/riverscape/` must never import `hydrofragments.spatial` (guarded by `tests/riverscape/test_riverscape_import_boundary.py`); `hydrofragments/hydroperiod/` must never import `hydrofragments.riverscape` (guarded by `tests/hydroperiod/test_hydroperiod_classify.py`). Neither guard is touched by this plan.
- `data/fitzroy_kimberley_aoi.geojson` and `data/fitzroy_kimberley_drainage.gpkg` are **not modified or replaced** — other tests (`tests/spatial/test_channel_context.py:161` pins a SHA-256 digest of the current drainage file) depend on them unchanged. The widened basin gets its own, new files.
- Task 2's one-off script depends on a local file geodatabase (`SH_Network.gdb`, national AHGF Surface Network, ~6.5 GB) that exists only on this development machine, outside the repo, at `D:/RLH/5.6/data_local/raw/extracted/SH_Network_GDB/SH_Network.gdb`. That path is a script argument with this path as its default, never hardcoded assuming portability, and the 6.5 GB source is never copied into the repo — only its filtered, basin-clipped extract (tens of MB) is committed.
- Test file basenames must be unique across `tests/`.

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `scripts/spikes/widen_fitzroy_aoi.py` | Create | One-off: clip national AHGF network/catchment layers to the Fitzroy River basin boundary, write `data/fitzroy_basin_*` |
| `data/fitzroy_basin_aoi.geojson` | Create | Fitzroy River basin boundary polygon, EPSG:3577 |
| `data/fitzroy_basin_drainage.gpkg` | Create | AHGF network lines clipped to the basin, EPSG:3577, same schema as `fitzroy_kimberley_drainage.gpkg` |
| `scripts/spikes/riverscape_phase0_basin_spike.py` | Create | Dask-native rerun of the Phase 0 distributional checks (AHGF offset, `UpstrDArea`, FC medians) over the widened basin |
| `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md` | Modify | Append a "Basin-scale rerun" section with the real numbers |
| `hydrofragments/spatial/zones.py` | Modify | `combine_zones` derives `emitted_zones`/`has_zone_1` from the mask |
| `tests/spatial/test_zone_combination.py` | Modify | Update the two tests pinned to the old hardcoded values |
| `hydrofragments/config.py` | Modify | Add `RiverscapeConfig`, parsing, `scientific_config()` entry, hash-schema bump |
| `tests/contracts/test_config.py`, `tests/contracts/test_hashing.py` | Modify | Cover the new config section; update the golden hash |

---

### Task 1: `combine_zones` reports what it actually produced

**Files:**
- Modify: `hydrofragments/spatial/zones.py:194-239` (`combine_zones` only)
- Modify: `tests/spatial/test_zone_combination.py` (two assertions)

**Interfaces:**
- Consumes: `LANDFORM_IN_CHANNEL`, `LANDFORM_OFF_CHANNEL_RIVERINE`, `LANDFORM_OUTSIDE` (already imported in `zones.py`).
- Produces: `combine_zones(...)`'s `ZoneResult.emitted_zones` and `.has_zone_1` now vary with input, matching `build_zones`'s convention. No signature change.

This task is fully independent of Tasks 2 and 4 — do it first, it unblocks nothing but has no dependency either.

- [ ] **Step 1: Write the failing test**

Open `tests/spatial/test_zone_combination.py` and replace the `test_combined_result_contract` function (it currently asserts hardcoded `emitted_zones == (1, 2, 3, 4)` and `has_zone_1 is True` for an input whose mask is actually `[[1, 2], [0, 0]]`) with:

```python
def test_combined_result_contract() -> None:
    landform = np.array([[1, 2], [3, 0]], dtype=np.uint8)
    hydroperiod = np.array([[4, 1], [2, 0]], dtype=np.uint8)

    result = combine_zones(landform, hydroperiod, source="riverscape", degraded_reasons=("envelope_single_bin",))

    assert result.mask.dtype == np.uint8
    assert result.crosstab.dtype == np.uint8
    assert result.mask.tolist() == [[1, 2], [0, 0]]
    assert result.crosstab.tolist() == [[14, 21], [32, 0]]
    assert result.emitted_zones == (1, 2)
    assert result.has_zone_1 is True
    assert result.mode == "riverscape"
    assert result.source == "riverscape"
    assert result.degraded_reasons == ("envelope_single_bin",)
```

(Only `emitted_zones` and `has_zone_1`'s assertions changed — `(1, 2, 3, 4)`/`True` → `(1, 2)`/`True` derived from what the mask actually contains: zones 1 and 2 only.)

Then find `test_mapping_table_matches_spec` in the same file — it doesn't check `emitted_zones`/`has_zone_1` (only `mask`/`crosstab` per-pixel), so it needs no change. Add one new test after `test_combined_result_contract`:

```python
def test_emitted_zones_and_has_zone_1_reflect_the_actual_mask() -> None:
    off_channel_only = combine_zones(
        np.array([[2, 2]], dtype=np.uint8), np.array([[1, 2]], dtype=np.uint8)
    )
    assert off_channel_only.emitted_zones == (2, 3)
    assert off_channel_only.has_zone_1 is False

    outside_only = combine_zones(np.array([[0]], dtype=np.uint8), np.array([[0]], dtype=np.uint8))
    assert outside_only.emitted_zones == ()
    assert outside_only.has_zone_1 is False

    all_four = combine_zones(
        np.array([[1, 2, 2, 2]], dtype=np.uint8),
        np.array([[1, 1, 2, 3]], dtype=np.uint8),
    )
    assert all_four.emitted_zones == (1, 2, 3, 4)
    assert all_four.has_zone_1 is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/spatial/test_zone_combination.py -v`
Expected: `test_combined_result_contract` FAILS on `assert result.emitted_zones == (1, 2)` (currently hardcoded to `(1, 2, 3, 4)`); `test_emitted_zones_and_has_zone_1_reflect_the_actual_mask` FAILS (new test, current code always returns `(1,2,3,4)`/`True`).

- [ ] **Step 3: Fix `combine_zones`**

In `hydrofragments/spatial/zones.py`, replace the `return ZoneResult(...)` block at the end of `combine_zones` (currently):

```python
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

with:

```python
    emitted_zones = tuple(sorted(int(z) for z in np.unique(mask) if z != 0))

    return ZoneResult(
        mask=mask,
        emitted_zones=emitted_zones,
        has_zone_1=bool(np.any(mask == 1)),
        source=source,
        mode="riverscape",
        crosstab=crosstab,
        degraded_reasons=tuple(degraded_reasons),
    )
```

Also update the function's docstring: after the existing paragraph, add:

```python
    ``emitted_zones``/``has_zone_1`` are derived from ``mask`` (the zones
    actually present), matching ``build_zones``'s convention -- not declared
    by mode. A degraded run with zero in-channel pixels reports
    ``has_zone_1=False``.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/spatial/test_zone_combination.py tests/spatial/test_zones.py tests/guards/test_scientific_guards.py tests/gating/test_zones_do_not_multiply_metrics.py -v`
Expected: all pass (24 in `test_zone_combination.py` including the new test, 24 unchanged in `test_zones.py`, 15 + 1 in guards/gating).

- [ ] **Step 5: Commit**

```bash
git add hydrofragments/spatial/zones.py tests/spatial/test_zone_combination.py
git commit -m "fix: combine_zones reports emitted_zones/has_zone_1 from the actual mask"
```

---

### Task 2: Widen the Fitzroy test AOI to the full river basin

**Files:**
- Create: `scripts/spikes/widen_fitzroy_aoi.py`
- Create: `data/fitzroy_basin_aoi.geojson` (script output)
- Create: `data/fitzroy_basin_drainage.gpkg` (script output)

**Interfaces:**
- Consumes: local file `D:/RLH/5.6/data_local/raw/extracted/SH_Network_GDB/SH_Network.gdb` (layers `AHGFNetworkStream`, `AHGFCatchment`; developer-machine-only, not in the repo), and one BOM ArcGIS FeatureServer query for the basin boundary polygon (`https://hosting.wsapi.cloud.bom.gov.au/arcgis/rest/services/ahgf/Geofabric_V3x_All_Products/FeatureServer/13`, layer 13 = `RiverRegion`, field `rivregname = 'FITZROY RIVER (WA)'`).
- Produces: `data/fitzroy_basin_aoi.geojson` (single polygon, EPSG:3577), `data/fitzroy_basin_drainage.gpkg` (line layer, EPSG:3577, column names matching `fitzroy_kimberley_drainage.gpkg`'s existing PascalCase schema: `HydroID`, `AHGFFType`, `Name`, `Hierarchy`, `Perennial`, `From_Node`, `To_Node`, `NextDownID`, `SrcFCName`, `UpstrDArea`, `ConCatID`, `geometry`, etc. — the GDB layer already uses this exact schema, so no renaming is needed). Task 3 reads both.

This has already been prototyped interactively and verified to work: a single BOM query returns the Fitzroy River basin polygon (`albersarea = 97,200,597,573 m²` ≈ 97,201 km²), and a `pyogrio`-backed bbox read of the local GDB's `AHGFNetworkStream` layer, clipped to that polygon, returns 31,318 reaches spanning `UpstrDArea` from ~900 m² to ~5.6×10¹⁰ m² (p1=2.79×10⁵, p25=1.70×10⁶, p50=5.25×10⁶, p75=3.32×10⁷, p95=3.06×10⁹, p99=5.36×10¹⁰) — five-plus orders of magnitude, unlike the narrow AOI's near-constant distribution. The read completed in under two seconds.

- [ ] **Step 1: Write the script**

```python
"""Widen the Fitzroy test data from the narrow lower-mainstem AOI (data/
fitzroy_kimberley_*) to the full Fitzroy River basin.

Not part of the package and not run by pytest -- a one-off data-preparation
script, like scripts/spikes/riverscape_phase0_spike.py. Requires:
  1. Network access to one BOM ArcGIS FeatureServer query (the basin
     boundary polygon; a few MB).
  2. A local copy of the national AHGF Surface Network file geodatabase
     (SH_Network.gdb, ~6.5 GB), which exists only on this development
     machine and is never added to the repo. Pass its path with --gdb if it
     is not at the default location.

Run:  python scripts/spikes/widen_fitzroy_aoi.py
Out:  data/fitzroy_basin_aoi.geojson, data/fitzroy_basin_drainage.gpkg
"""
from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GDB = Path("D:/RLH/5.6/data_local/raw/extracted/SH_Network_GDB/SH_Network.gdb")
BASIN_QUERY_URL = (
    "https://hosting.wsapi.cloud.bom.gov.au/arcgis/rest/services/ahgf/"
    "Geofabric_V3x_All_Products/FeatureServer/13/query"
)
TARGET_CRS = "EPSG:3577"
AOI_OUT = REPO / "data" / "fitzroy_basin_aoi.geojson"
DRAINAGE_OUT = REPO / "data" / "fitzroy_basin_drainage.gpkg"


def fetch_basin_boundary(*, river_region_name: str) -> gpd.GeoDataFrame:
    """Query the BOM RiverRegion layer for one named basin polygon."""
    where = urllib.parse.quote(f"rivregname='{river_region_name}'")
    url = (
        f"{BASIN_QUERY_URL}?where={where}&outFields=*&f=geojson"
        f"&outSR={TARGET_CRS.split(':')[1]}"
    )
    raw = urllib.request.urlopen(url, timeout=120).read()
    gdf = gpd.read_file(raw.decode("utf-8"))
    if len(gdf) != 1:
        raise RuntimeError(
            f"expected exactly one RiverRegion feature for {river_region_name!r}, "
            f"got {len(gdf)}"
        )
    if gdf.crs is None or gdf.crs.to_string() != TARGET_CRS:
        gdf = gdf.set_crs(TARGET_CRS, allow_override=True)
    return gdf


def clip_network_to_basin(gdb_path: Path, basin: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Bbox-read AHGFNetworkStream from the local GDB, then exact-clip to the basin."""
    basin_native = basin.to_crs("EPSG:4283")
    bbox = tuple(basin_native.total_bounds)
    streams = gpd.read_file(str(gdb_path), layer="AHGFNetworkStream", bbox=bbox, engine="pyogrio")
    if streams.crs is None:
        streams = streams.set_crs("EPSG:4283", allow_override=True)
    streams = streams.to_crs(TARGET_CRS)
    basin_geom = basin.geometry.iloc[0]
    clipped = streams[streams.intersects(basin_geom)].copy()
    if clipped.empty:
        raise RuntimeError("basin clip produced zero reaches -- check the GDB path and basin polygon")
    return clipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdb", type=Path, default=DEFAULT_GDB, help="path to SH_Network.gdb")
    parser.add_argument(
        "--river-region", default="FITZROY RIVER (WA)", help="BOM RiverRegion.rivregname value"
    )
    args = parser.parse_args()

    if not args.gdb.exists():
        print(f"GDB not found at {args.gdb} -- pass --gdb", file=sys.stderr)
        return 1

    basin = fetch_basin_boundary(river_region_name=args.river_region)
    area_km2 = float(basin.geometry.area.sum()) / 1e6
    print(f"basin polygon: {area_km2:,.1f} km2, bounds {tuple(basin.total_bounds)}")

    drainage = clip_network_to_basin(args.gdb, basin)
    print(f"clipped drainage: {len(drainage)} reaches")
    upstr = drainage["UpstrDArea"].astype(float)
    print(
        "UpstrDArea (m2) p1/p5/p25/p50/p75/p95/p99:",
        [round(v, 1) for v in upstr.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
    )

    AOI_OUT.parent.mkdir(parents=True, exist_ok=True)
    basin[["geometry"]].to_file(AOI_OUT, driver="GeoJSON")
    drainage.to_file(DRAINAGE_OUT, driver="GPKG")
    print(f"wrote {AOI_OUT}")
    print(f"wrote {DRAINAGE_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

Run: `python scripts/spikes/widen_fitzroy_aoi.py`
Expected: basin area printed (~97,200 km²), reach count printed (order of magnitude ~30,000), a `UpstrDArea` quantile list spanning from roughly 10⁵ to 10¹⁰–10¹¹ m² (five-plus orders of magnitude — this is the property being verified; if the printed quantiles are all within one or two orders of magnitude of each other, STOP, do not proceed to Step 3, and report the actual numbers rather than guessing why), both output files written.

If the GDB is not present at the default path on the machine running this task, pass `--gdb <path>`; if the BOM query returns zero or more than one feature, the script raises `RuntimeError` naming the count — do not relax that check to "pick the first one".

- [ ] **Step 3: Sanity-check the output files**

```bash
python -c "
import geopandas as gpd
aoi = gpd.read_file('data/fitzroy_basin_aoi.geojson')
drainage = gpd.read_file('data/fitzroy_basin_drainage.gpkg')
assert aoi.crs.to_string() == 'EPSG:3577', aoi.crs
assert drainage.crs.to_string() == 'EPSG:3577', drainage.crs
assert len(aoi) == 1
assert len(drainage) > 1000
required = {'HydroID', 'From_Node', 'To_Node', 'NextDownID', 'UpstrDArea', 'Hierarchy', 'Perennial', 'SrcFCName', 'ConCatID'}
missing = required - set(drainage.columns)
assert not missing, missing
print('ok:', len(drainage), 'reaches,', aoi.geometry.area.sum() / 1e6, 'km2')
"
```

Expected: `ok: <n> reaches, <area> km2` with no assertion error.

- [ ] **Step 4: Confirm the existing narrow-AOI test data and its dependent tests are untouched**

```bash
git status --short data/
python -m pytest tests/spatial/test_channel_context.py tests/spatial/test_reach_wet_parity.py -v
```

Expected: `git status --short data/` shows only the two new untracked files (`fitzroy_basin_aoi.geojson`, `fitzroy_basin_drainage.gpkg`); `data/fitzroy_kimberley_aoi.geojson` and `data/fitzroy_kimberley_drainage.gpkg` do not appear as modified. Both test files pass unchanged (`test_channel_context.py`'s digest-pinned test must still pass — if it doesn't, something touched the narrow-AOI drainage file; stop and investigate rather than updating the pinned digest).

- [ ] **Step 5: Commit**

```bash
git add scripts/spikes/widen_fitzroy_aoi.py data/fitzroy_basin_aoi.geojson data/fitzroy_basin_drainage.gpkg
git commit -m "data: widen Fitzroy test AOI to the full river basin

The narrow lower-mainstem AOI (data/fitzroy_kimberley_*, kept unchanged for
existing tests) has a near-constant UpstrDArea distribution and cannot
calibrate a drainage-area scaling law. Clip the national AHGF Surface
Network (local SH_Network.gdb) to the Fitzroy River basin boundary (BOM
RiverRegion, ~97,200 km2) instead, giving ~31,000 reaches spanning five-plus
orders of magnitude of UpstrDArea."
```

---

### Task 3: Rerun Phase 0's distributional checks at basin scale

**Files:**
- Create: `scripts/spikes/riverscape_phase0_basin_spike.py`
- Modify: `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md`

**Interfaces:**
- Consumes: `hydrofragments.io.dea.open_wo_statistics_for_zoning`, `data/fitzroy_basin_aoi.geojson` / `data/fitzroy_basin_drainage.gpkg` (Task 2).
- Produces: printed/appended distributional numbers (AHGF-offset percentiles, FC bare/green medians on seed/trough/background) that Task 4 reads to set `RiverscapeConfig` defaults. No package code.

Depends on Task 2 (needs its output files). Independent of Task 1.

**Why not a straight copy of the Phase 0 script:** the widened AOI is ~481 km × 384 km at 30 m ≈ 16,000 × 12,800 ≈ 205 million pixels — about 360× Plan 1's Phase 0 AOI (539 × 1117 ≈ 602,000 pixels). Phase 0's spike loaded all 9 Fractional Cover bands into memory *simultaneously* as `float64`; at this scale that would be ~1.6 GB per band × 9 ≈ 15 GB held at once — too much to assume is available. This script keeps that pattern (eager `numpy`/`.compute()`, same as Phase 0 — proven to work and simple to reason about) but bounds peak memory by loading **one FC band at a time as `float32`** (≈800 MB each), computing its stats against the already-materialized `water`/`trough` boolean masks, then freeing it before the next band. The DEM band load and the AHGF-offset distance transform stay eager too (same as Phase 0), since each is a single ≈800 MB–1 GB array, not nine of them.

- [ ] **Step 1: Write the script**

```python
"""Phase 0 rerun at basin scale: AHGF-offset, UpstrDArea and Fractional
Cover distributions over the full Fitzroy River basin (data/fitzroy_basin_*
from scripts/spikes/widen_fitzroy_aoi.py).

Does NOT re-decide the DEM band, FC band names, or DEA Waterbodies route --
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


def _band_array(dataset, band: str) -> np.ndarray:
    """Load one band eagerly as float32 (~800MB at this AOI's pixel count)."""
    data = dataset[band]
    if "time" in data.dims:
        data = data.median("time")
    return np.asarray(data, dtype=np.float32)


def _masked_median(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return float(np.nanmedian(values[mask]))


def main() -> int:
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
        dem = _band_array(dem_ds, DEM_BAND)
        local_mean = ndimage.uniform_filter(
            np.where(np.isfinite(dem), dem, np.nanmean(dem)), size=2 * TROUGH_RADIUS_PX + 1
        )
        trough = np.isfinite(dem) & ((local_mean - dem) >= TROUGH_DEPTH_M)
        findings["dem"] = {"stac_url": dem_url, "band": DEM_BAND, "trough_pixels": int(trough.sum())}
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
            values = _band_array(fc_ds, band)
            band_stats[band] = {
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
```

Peak memory is bounded to roughly: the water mask (~200 MB bool) plus skeleton (~200 MB bool, both held for the whole run) plus, at any one time, either the DEM band (~800 MB float32) or one FC band (~800 MB float32) plus the trough mask (~200 MB bool, held after the DEM step). Report actual observed peak memory (e.g. via `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` on Linux/macOS, or Windows Task Manager / `Get-Process -Id $PID` peak working set on this Windows machine) and wall-clock time in the task report — if either is impractically large or the process is killed, report BLOCKED with the specifics rather than letting it run unbounded.

- [ ] **Step 2: Run it**

Run: `python scripts/spikes/riverscape_phase0_basin_spike.py`

Task 1 of Plan 1 found that `hydroseason.open_wo_statistics`'s env-var scoping requires `AWS_NO_SIGN_REQUEST` to be set via `os.environ.setdefault(...)` *before* calling `open_wo_statistics_for_zoning`, because the returned dataset is Dask-lazy and any later compute happens after hydroseason's own snapshot/restore has already reverted the environment. This script does that at the top of `main()`; if you still see `RasterioIOError: InvalidCredentials`, re-read `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md`'s "Implications for Plan 2" note on this before debugging from scratch.

Expected: exits 0 with `errors: {}`, printed `UpstrDArea` quantiles spanning at least four orders of magnitude, and populated FC `band_stats`.

Note total wall-clock time and peak memory in the report — if either is impractically large (e.g. many hours, or the process is killed by the OS), report BLOCKED with the specifics rather than letting the task run unbounded; a fallback (a stratified spatial sample of the basin instead of the full raster) can be designed in a follow-up if the full-basin read proves infeasible on this machine.

- [ ] **Step 3: Append the basin-scale findings**

Append this section to `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md` (after the existing "## Plan 1 completion" section), filling in real numbers from the script's JSON output:

```markdown
## Basin-scale rerun (Plan 2, Task 3)

**Source JSON:** `output/spikes/riverscape_phase0_basin.json` (not committed)
**Script:** `scripts/spikes/riverscape_phase0_basin_spike.py`
**AOI:** `data/fitzroy_basin_aoi.geojson` (full Fitzroy River basin, ~97,200 km²), replacing the narrow AOI for distributional (not band/route) decisions.

This does not re-decide `dem_band`, FC band names, or the DEA Waterbodies
route — those came from the narrow-AOI Phase 0 spike above and don't depend
on AOI size. It re-measures the distributions that were degenerate there.

### Grid
- Shape / pixel count: <grid.shape> / <grid.pixel_count>

### AHGF alignment (basin-wide)
- line→EO skeleton (within ~1 km): p50 <..> m, p95 <..> m, n <..>
- `UpstrDArea` quantiles (m²): p1 <..>, p5 <..>, p25 <..>, p50 <..>, p75 <..>, p95 <..>, p99 <..>
- Reach count: <ahgf_offset.reach_count>

**Comparison with the narrow AOI:** the narrow AOI's `UpstrDArea` p50≈p95
(≈53,000–54,000 km²) is now the basin's <state where it falls, e.g. "p95–p99
range"> — confirming it sampled only the largest reaches. The envelope fit
(spec §4.2 step 7) now has real per-log-A bins to fit against.

### Fractional Cover medians (basin-wide, `dem_s` trough)
| Band | median on seed water | median on trough | median AOI |
|---|---|---|---|
| bs_pc_50 | <..> | <..> | <..> |
| pv_pc_50 | <..> | <..> | <..> |
| npv_pc_50 | <..> | <..> | <..> |

**Comparison with the narrow AOI:** narrow-AOI `bs_pc_50` was 31.5 (seed) /
20.5 (AOI) — <state whether the basin-wide numbers are similar, confirming
the per-run-calibration decision in §4.2 step 4 rather than a fixed
threshold, or materially different, which would itself be evidence for
per-run calibration>.

### Run cost
- Wall-clock time: <..>
- Peak memory: <..> (method: <resource.getrusage | Task Manager | other>)
- FC-band computation approach used: <dask-aligned boolean indexing | one-band-at-a-time eager materialization>

### Implications for Plan 3
- `corridor_min_m`/`corridor_max_m` (spec §6, currently 90/600): <state
  whether the basin-wide line→skeleton p95 still saturates the 600 m
  ceiling the way the narrow AOI's did, or whether it now falls within
  range>.
- `bare_threshold` calibration inputs: <one line on whether basin-wide bare
  medians support a single per-run calibration or vary enough by reach size
  to need per-reach calibration instead>.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/spikes/riverscape_phase0_basin_spike.py docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md
git commit -m "docs: rerun Phase 0 distributional checks at Fitzroy basin scale"
```

---

### Task 4: `RiverscapeConfig` skeleton and hash-schema bump

**Files:**
- Modify: `hydrofragments/config.py` (add `RiverscapeConfig`, parsing, `scientific_config()` entry, `_TOP_LEVEL_KEYS`, `SCIENTIFIC_HASH_SCHEMA_VERSION`)
- Modify: `tests/contracts/test_config.py` (round-trip + unknown-key tests)
- Modify: `tests/contracts/test_hashing.py` (golden hash)

**Interfaces:**
- Consumes: `HydroConfig.from_mapping`, `_section`, `_fraction`, `_TOP_LEVEL_KEYS` (all existing in `config.py`); the basin-scale numbers from Task 3 (for choosing defaults, not for any code dependency).
- Produces: `hydrofragments.config.RiverscapeConfig` (new frozen dataclass), `HydroConfig.riverscape: RiverscapeConfig` field, `config.riverscape.*` readable the same way `config.zones.*` is today. Plan 3's evidence loaders and corridor/terrain modules will read this.

Depends on Task 3's findings for two default values (`corridor_min_m`/`corridor_max_m`, `bare_threshold_floor_pct`) — do not start this task until Task 3's findings section is complete and committed.

- [ ] **Step 1: Write the failing tests**

Add to `tests/contracts/test_config.py`:

```python
def test_riverscape_config_has_documented_defaults() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())

    assert config.riverscape.mode == "auto"
    assert config.riverscape.dem_product == "ga_srtm_dem1sv1_0"
    assert config.riverscape.dem_band == "dem_s"
    assert config.riverscape.fc_product == "ga_ls_fc_pc_cyear_3"
    assert config.riverscape.bare_band == "bs_pc_50"
    assert config.riverscape.green_band == "pv_pc_50"
    assert 0.0 < config.riverscape.f_seed < config.riverscape.f_chan_high < 1.0
    assert config.riverscape.corridor_min_m < config.riverscape.corridor_max_m
    assert config.riverscape.min_channel_confidence >= 1


def test_riverscape_config_rejects_bad_mode() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="riverscape.mode"):
        HydroConfig.from_mapping(minimal_config(riverscape={"mode": "sometimes"}))


def test_riverscape_config_rejects_f_seed_above_f_chan_high() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="f_seed"):
        HydroConfig.from_mapping(
            minimal_config(riverscape={"f_seed": 0.5, "f_chan_high": 0.1})
        )


def test_riverscape_config_rejects_inverted_corridor_bounds() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="corridor_min_m"):
        HydroConfig.from_mapping(
            minimal_config(riverscape={"corridor_min_m": 600, "corridor_max_m": 90})
        )


def test_unknown_riverscape_key_is_rejected() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match=r"unknown config key.*riverscape\.mystery"):
        HydroConfig.from_mapping(minimal_config(riverscape={"mystery": True}))
```

Update `tests/contracts/test_hashing.py`: the golden hash constant will change (adding a new required section to `scientific_config()` changes its canonical JSON). Do not guess the new value — Step 4 below computes and records it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/contracts/test_config.py -k riverscape -v`
Expected: FAIL — `AttributeError: 'HydroConfig' object has no attribute 'riverscape'` (the new tests) or `ConfigError: unknown config key: config.riverscape` (if a mapping already includes a `riverscape` key before this task's config-schema change lands — expected, since `_TOP_LEVEL_KEYS` doesn't include it yet).

- [ ] **Step 3: Implement `RiverscapeConfig`**

In `hydrofragments/config.py`, add after `ZonesConfig` (currently ends at line 98):

```python
@dataclass(frozen=True)
class RiverscapeConfig:
    mode: str = "auto"
    dem_product: str = "ga_srtm_dem1sv1_0"
    dem_band: str = "dem_s"
    fc_product: str = "ga_ls_fc_pc_cyear_3"
    bare_band: str = "bs_pc_50"
    green_band: str = "pv_pc_50"
    npv_band: str = "npv_pc_50"
    waterbodies_source: str | None = None
    f_seed: float = 0.05
    f_chan_high: float = 0.10
    bare_threshold_floor_pct: float = 50.0
    bare_year_fraction: float = 0.6
    trough_radius_m: float = 150.0
    trough_depth_m: float = 0.5
    h_chan_m: float = 2.0
    corridor_min_m: float = 90.0
    corridor_max_m: float = 600.0
    alignment_quantile: float = 0.95
    width_growth_factor: float = 3.0
    profile_bin_m: float = 300.0
    profile_percentile: float = 10.0
    rem_k: int = 8
    rem_max_distance_m: float = 5000.0
    envelope_quantile: float = 0.95
    envelope_h_max_m: float = 15.0
    envelope_min_bin_pixels: int = 200
    slope_max_deg: float = 2.0
    min_channel_confidence: int = 2
    include_line_fallback_in_channel: bool = False
    bridge_enabled: bool = True
    bridge_max_length_m: float = 2000.0
    bridge_max_cost_per_m: float = 1.0
    bridge_rem_max_m: float = 5.0
    riparian_green_pct: float = 40.0
    narrow_width_px: int = 2
```

(Defaults above match spec §6's table; `bare_threshold_pct` is renamed to `bare_threshold_floor_pct` per §4.2 step 4's per-run-calibration decision — it is now a fallback floor, not the primary threshold. **Update the numeric defaults for `corridor_min_m`/`corridor_max_m` and `bare_threshold_floor_pct` from Task 3's basin-scale findings before writing this dataclass** — the values above are Plan 1's narrow-AOI-derived placeholders; Task 3's findings doc's "Implications for Plan 3" bullets state whether they should change.)

Add `RiverscapeConfig` to the imports/exports and to `HydroConfig`'s fields (after `zones: ZonesConfig = field(default_factory=ZonesConfig)`, currently followed by `dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)`):

```python
    riverscape: RiverscapeConfig = field(default_factory=RiverscapeConfig)
```

Add `"riverscape"` to `_TOP_LEVEL_KEYS` (the set literal starting at line 183).

After the existing `zones_raw = _section(...)` / `if zones.t_season >= zones.t_persist:` block (ends at line 527), add:

```python
        riverscape_raw = _section(
            source,
            "riverscape",
            {
                "mode", "dem_product", "dem_band", "fc_product", "bare_band",
                "green_band", "npv_band", "waterbodies_source", "f_seed",
                "f_chan_high", "bare_threshold_floor_pct", "bare_year_fraction",
                "trough_radius_m", "trough_depth_m", "h_chan_m",
                "corridor_min_m", "corridor_max_m", "alignment_quantile",
                "width_growth_factor", "profile_bin_m", "profile_percentile",
                "rem_k", "rem_max_distance_m", "envelope_quantile",
                "envelope_h_max_m", "envelope_min_bin_pixels", "slope_max_deg",
                "min_channel_confidence", "include_line_fallback_in_channel",
                "bridge_enabled", "bridge_max_length_m", "bridge_max_cost_per_m",
                "bridge_rem_max_m", "riparian_green_pct", "narrow_width_px",
            },
        )
        defaults = RiverscapeConfig()
        riverscape_mode = str(riverscape_raw.get("mode", defaults.mode))
        if riverscape_mode not in {"off", "auto", "required"}:
            raise ConfigError(
                f"riverscape.mode has unsupported value: {riverscape_mode}"
            )
        riverscape_f_seed = _fraction(
            riverscape_raw.get("f_seed", defaults.f_seed), "riverscape.f_seed"
        )
        riverscape_f_chan_high = _fraction(
            riverscape_raw.get("f_chan_high", defaults.f_chan_high),
            "riverscape.f_chan_high",
        )
        if riverscape_f_seed > riverscape_f_chan_high:
            raise ConfigError("riverscape.f_seed must not exceed riverscape.f_chan_high")
        riverscape_corridor_min = float(
            riverscape_raw.get("corridor_min_m", defaults.corridor_min_m)
        )
        riverscape_corridor_max = float(
            riverscape_raw.get("corridor_max_m", defaults.corridor_max_m)
        )
        if not (
            math.isfinite(riverscape_corridor_min)
            and math.isfinite(riverscape_corridor_max)
            and 0.0 < riverscape_corridor_min < riverscape_corridor_max
        ):
            raise ConfigError(
                "riverscape.corridor_min_m must be positive and less than "
                "riverscape.corridor_max_m"
            )
        riverscape = RiverscapeConfig(
            mode=riverscape_mode,
            dem_product=str(riverscape_raw.get("dem_product", defaults.dem_product)),
            dem_band=str(riverscape_raw.get("dem_band", defaults.dem_band)),
            fc_product=str(riverscape_raw.get("fc_product", defaults.fc_product)),
            bare_band=str(riverscape_raw.get("bare_band", defaults.bare_band)),
            green_band=str(riverscape_raw.get("green_band", defaults.green_band)),
            npv_band=str(riverscape_raw.get("npv_band", defaults.npv_band)),
            waterbodies_source=riverscape_raw.get("waterbodies_source"),
            f_seed=riverscape_f_seed,
            f_chan_high=riverscape_f_chan_high,
            bare_threshold_floor_pct=_percentage(
                riverscape_raw.get(
                    "bare_threshold_floor_pct", defaults.bare_threshold_floor_pct
                ),
                "riverscape.bare_threshold_floor_pct",
            ),
            bare_year_fraction=_fraction(
                riverscape_raw.get("bare_year_fraction", defaults.bare_year_fraction),
                "riverscape.bare_year_fraction",
            ),
            trough_radius_m=float(riverscape_raw.get("trough_radius_m", defaults.trough_radius_m)),
            trough_depth_m=float(riverscape_raw.get("trough_depth_m", defaults.trough_depth_m)),
            h_chan_m=float(riverscape_raw.get("h_chan_m", defaults.h_chan_m)),
            corridor_min_m=riverscape_corridor_min,
            corridor_max_m=riverscape_corridor_max,
            alignment_quantile=_fraction(
                riverscape_raw.get("alignment_quantile", defaults.alignment_quantile),
                "riverscape.alignment_quantile",
            ),
            width_growth_factor=float(
                riverscape_raw.get("width_growth_factor", defaults.width_growth_factor)
            ),
            profile_bin_m=float(riverscape_raw.get("profile_bin_m", defaults.profile_bin_m)),
            profile_percentile=_percentage(
                riverscape_raw.get("profile_percentile", defaults.profile_percentile),
                "riverscape.profile_percentile",
            ),
            rem_k=int(riverscape_raw.get("rem_k", defaults.rem_k)),
            rem_max_distance_m=float(
                riverscape_raw.get("rem_max_distance_m", defaults.rem_max_distance_m)
            ),
            envelope_quantile=_fraction(
                riverscape_raw.get("envelope_quantile", defaults.envelope_quantile),
                "riverscape.envelope_quantile",
            ),
            envelope_h_max_m=float(
                riverscape_raw.get("envelope_h_max_m", defaults.envelope_h_max_m)
            ),
            envelope_min_bin_pixels=int(
                riverscape_raw.get("envelope_min_bin_pixels", defaults.envelope_min_bin_pixels)
            ),
            slope_max_deg=float(riverscape_raw.get("slope_max_deg", defaults.slope_max_deg)),
            min_channel_confidence=int(
                riverscape_raw.get("min_channel_confidence", defaults.min_channel_confidence)
            ),
            include_line_fallback_in_channel=bool(
                riverscape_raw.get(
                    "include_line_fallback_in_channel",
                    defaults.include_line_fallback_in_channel,
                )
            ),
            bridge_enabled=bool(riverscape_raw.get("bridge_enabled", defaults.bridge_enabled)),
            bridge_max_length_m=float(
                riverscape_raw.get("bridge_max_length_m", defaults.bridge_max_length_m)
            ),
            bridge_max_cost_per_m=float(
                riverscape_raw.get("bridge_max_cost_per_m", defaults.bridge_max_cost_per_m)
            ),
            bridge_rem_max_m=float(
                riverscape_raw.get("bridge_rem_max_m", defaults.bridge_rem_max_m)
            ),
            riparian_green_pct=_percentage(
                riverscape_raw.get("riparian_green_pct", defaults.riparian_green_pct),
                "riverscape.riparian_green_pct",
            ),
            narrow_width_px=int(
                riverscape_raw.get("narrow_width_px", defaults.narrow_width_px)
            ),
        )
        if riverscape.min_channel_confidence < 1:
            raise ConfigError("riverscape.min_channel_confidence must be at least 1")
```

Pass `riverscape=riverscape` in the `return cls(...)` call in `from_mapping` (alongside the existing `zones=zones,`).

In `scientific_config()`, the top-level dict keys are alphabetically sorted (`channel`, `connectivity`, `dynamics`, `hash_algorithm_version`, `hydroyear`, `input`, `metric_overrides`, `metric_profiles`, `patches`, `persistence`, `spatial`, `state`, `temporal`, `validity`, `zones`). Insert the new `"riverscape"` entry between `"persistence"` and `"spatial"` to preserve that ordering:

```python
            "riverscape": {
                "alignment_quantile": self.riverscape.alignment_quantile,
                "bare_band": self.riverscape.bare_band,
                "bare_threshold_floor_pct": self.riverscape.bare_threshold_floor_pct,
                "bare_year_fraction": self.riverscape.bare_year_fraction,
                "bridge_enabled": self.riverscape.bridge_enabled,
                "bridge_max_cost_per_m": self.riverscape.bridge_max_cost_per_m,
                "bridge_max_length_m": self.riverscape.bridge_max_length_m,
                "bridge_rem_max_m": self.riverscape.bridge_rem_max_m,
                "corridor_max_m": self.riverscape.corridor_max_m,
                "corridor_min_m": self.riverscape.corridor_min_m,
                "dem_band": self.riverscape.dem_band,
                "dem_product": self.riverscape.dem_product,
                "envelope_h_max_m": self.riverscape.envelope_h_max_m,
                "envelope_min_bin_pixels": self.riverscape.envelope_min_bin_pixels,
                "envelope_quantile": self.riverscape.envelope_quantile,
                "f_chan_high": self.riverscape.f_chan_high,
                "f_seed": self.riverscape.f_seed,
                "fc_product": self.riverscape.fc_product,
                "green_band": self.riverscape.green_band,
                "h_chan_m": self.riverscape.h_chan_m,
                "include_line_fallback_in_channel": (
                    self.riverscape.include_line_fallback_in_channel
                ),
                "min_channel_confidence": self.riverscape.min_channel_confidence,
                "mode": self.riverscape.mode,
                "narrow_width_px": self.riverscape.narrow_width_px,
                "npv_band": self.riverscape.npv_band,
                "profile_bin_m": self.riverscape.profile_bin_m,
                "profile_percentile": self.riverscape.profile_percentile,
                "rem_k": self.riverscape.rem_k,
                "rem_max_distance_m": self.riverscape.rem_max_distance_m,
                "riparian_green_pct": self.riverscape.riparian_green_pct,
                "slope_max_deg": self.riverscape.slope_max_deg,
                "trough_depth_m": self.riverscape.trough_depth_m,
                "trough_radius_m": self.riverscape.trough_radius_m,
                "waterbodies_source": self.riverscape.waterbodies_source,
                "width_growth_factor": self.riverscape.width_growth_factor,
            },
```

Bump `SCIENTIFIC_HASH_SCHEMA_VERSION` from `"1.1.0"` to `"1.2.0"` and add `"1.2.0"` to `ACCEPTED_CONFIG_SCHEMA_VERSIONS`... **do not** add it there — `ACCEPTED_CONFIG_SCHEMA_VERSIONS` gates `config.config_schema_version` (the input mapping's declared schema version), which is a different, unrelated version number from `SCIENTIFIC_HASH_SCHEMA_VERSION` (the hash's own schema tag, emitted inside `scientific_config()`, never read from input). Only change `SCIENTIFIC_HASH_SCHEMA_VERSION`'s value.

Export `RiverscapeConfig` from `hydrofragments/config.py`'s `__all__` list.

- [ ] **Step 4: Compute and record the new golden hash**

Run:

```bash
python -c "
from hydrofragments.config import HydroConfig
config = HydroConfig.from_mapping({
    'config_schema_version': '1.0.0',
    'input': {'kind': 'generic_binary'},
    'temporal': {'input_cadence': 'monthly', 'monthly_composite': 'supplied', 'composite_owner': 'caller'},
})
print(config.config_hash)
"
```

Copy the printed 64-character hex string. In `tests/contracts/test_hashing.py`, replace the `GOLDEN_MINIMAL_CONFIG_HASH` constant's value with it. Do not hand-compute or guess this value — it must come from running the code, because `scientific_config()`'s canonical JSON ordering and the exact float representations of every new default matter to the hash.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/contracts/ -v`
Expected: all pass, including every pre-existing hashing/config test (the golden hash test now passes with the updated constant; `test_scientific_threshold_changes_config_hash` and similar tests are unaffected since they don't touch `riverscape` config).

- [ ] **Step 6: Full regression check**

Run: `python -m pytest -q`
Expected: same two pre-existing, unrelated failure categories as Plan 1's Task 5 baseline (`tests/release/test_branding.py::test_tracked_text_uses_only_readme_lineage_mention`, deterministic; the intermittent Windows bundle-rename race, rare) — no new failures.

- [ ] **Step 7: Commit**

```bash
git add hydrofragments/config.py tests/contracts/test_config.py tests/contracts/test_hashing.py
git commit -m "feat: add RiverscapeConfig with basin-calibrated defaults, bump hash schema to 1.2.0"
```

---

## Verification

- `pytest tests/spatial/test_zone_combination.py tests/spatial/test_zones.py tests/contracts/ tests/guards tests/gating -q`
- `pytest -q` (full suite; expect only the two known pre-existing failures)
- `python -c "import hydrofragments, hydrofragments.api, hydrofragments.workflow, hydrofragments.spatial.zones; print('ok')"`
- Inspect `data/fitzroy_basin_aoi.geojson` / `data/fitzroy_basin_drainage.gpkg` in QGIS or similar against a basemap: the polygon should visibly match the Fitzroy River catchment in the Kimberley, and the drainage lines should follow it, not spill into neighbouring basins.
