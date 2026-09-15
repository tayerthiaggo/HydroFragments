# Riverscape Zoning — Plan 3: Loaders, Corridor, Centreline, REM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three per-reach conflation steps riverscape zoning's channel rules (Plan 4) will consume — corridor width, EO centreline, and a Relative Elevation Model — plus the DEA STAC loaders (DEM, Fractional Cover, Waterbodies) that feed them, per the design spec's Phase 3 (`docs/superpowers/specs/2026-09-14-riverscape-zoning-design.md` §4.2 steps 1-3, §5).

**Architecture:** Four independent-to-mostly-independent modules, built bottom-up: `io/riverscape_sources.py` (Task 1, no dependency on the others) loads raw evidence rasters/polygons from DEA STAC/WFS; `riverscape/corridor.py` (Task 2) measures each reach's search-space width from a caller-supplied water-seed mask and drainage lines; `riverscape/centreline.py` (Task 3) and `riverscape/terrain.py` (Task 4) both consume Task 2's per-reach corridor widths (Task 3 to restrict its skeleton search, Task 4 to size its along-stream elevation sampling footprint) but not each other. None of `hydrofragments/riverscape/*` may import `hydrofragments.spatial` (existing guard, `tests/riverscape/test_riverscape_import_boundary.py`) — every function here takes already-validated drainage as a plain `GeoDataFrame` and does its own minimal column checks; callers are responsible for calling `hydrofragments.spatial.context.validate_drainage_topology`/`create_channel_context` first.

**Tech Stack:** Python 3.10+, numpy, scipy (`ndimage`, `spatial.cKDTree`, `optimize.isotonic_regression` — needs scipy >= 1.12, see Global Constraints), scikit-image (`medial_axis`), rasterio (`features.rasterize`), shapely 2.x (`ops.substring`), geopandas, odc-stac, pystac-client (new direct dependency), pytest.

## Global Constraints

- `hydrofragments/riverscape/` must never import `hydrofragments.spatial` (guarded by `tests/riverscape/test_riverscape_import_boundary.py`); `hydrofragments/hydroperiod/` must never import `hydrofragments.riverscape` (guarded by `tests/hydroperiod/test_hydroperiod_classify.py`). Neither guard is touched by this plan.
- Landform codes: `0 outside`, `1 in_channel`, `2 off_channel_riverine`, `3 non_riverine` (`hydrofragments/riverscape/codes.py`). Hydroperiod codes: `0 outside`, `1 persistent`, `2 seasonal`, `3 marginal`, `4 unobserved` (`hydrofragments/hydroperiod/codes.py`). This plan does not touch either codes module.
- No silent fallbacks: invalid inputs raise `ValueError` (or `RiverscapeSourceUnavailable` for source-acquisition failures) naming the violated contract; every degraded calibration is recorded in a `degraded_reasons: tuple[str, ...]` field, never silently applied.
- `RiverscapeConfig` already exists (`hydrofragments/config.py`) with every default this plan needs: `dem_product="ga_srtm_dem1sv1_0"`, `dem_band="dem_s"`, `fc_product="ga_ls_fc_pc_cyear_3"`, `bare_band="bs_pc_50"`, `green_band="pv_pc_50"`, `npv_band="npv_pc_50"`, `waterbodies_source=None`, `f_seed=0.05`, `corridor_min_m=90.0`, `corridor_max_m=1200.0`, `alignment_quantile=0.95`, `profile_bin_m=300.0`, `profile_percentile=10.0`, `rem_k=8`, `rem_max_distance_m=5000.0`, `trough_radius_m=150.0`, `trough_depth_m=0.5`. This plan does not add or change any `RiverscapeConfig` field — every new function takes these as plain keyword arguments, and wiring them to `RiverscapeConfig` is a later (Phase 4/6) integration task.
- DEA Waterbodies route (decided in Phase 0, `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md` "DEA Waterbodies" section): WFS `https://geoserver.dea.ga.gov.au/geoserver/wfs`, typename `dea:DigitalEarthAustraliaWaterbodies_v3`. Do not re-decide this route.
- DEM band (`dem_s`) and FC band names (`bs_pc_50`/`pv_pc_50`/`npv_pc_50`) were decided in Phase 0 and confirmed present on real STAC items (`docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md`). Do not re-decide them.
- `scipy>=1.11` (current `pyproject.toml` floor) does not have `scipy.optimize.isotonic_regression` (added in scipy 1.12.0). Task 4 bumps the floor to `scipy>=1.12` — document this in the same commit that first imports it, do not silently rely on a newer scipy already being installed in the dev environment.
- `pystac-client` is not currently a declared dependency (only used transitively via `odc-stac`/`hydroseason`). Task 1 adds it as a direct dependency in `pyproject.toml`, per spec §5 ("`pystac-client` is declared as a direct dependency").
- Test file basenames must be unique across `tests/`.
- Every new network-touching function (`load_dem`, `load_fc_percentiles`, `load_waterbodies`) is unit-tested by monkeypatching the real `pystac_client`/`odc.stac`/`geopandas` module attributes as module-level fakes — the established convention in `tests/integration/test_dea_workflow.py` and `tests/io/test_dea.py` (monkeypatch the real package's public entry points, `raising=False`, rather than mocking an internal abstraction). No test in this plan makes a real network call.

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `hydrofragments/io/riverscape_sources.py` | Create | `load_dem`, `load_fc_percentiles`, `load_waterbodies`, `RiverscapeSourceUnavailable` |
| `hydrofragments/io/dea.py` | Modify (docstring only) | Note the `io/riverscape_sources.py` exception to "hydroseason owns STAC access" |
| `hydrofragments/workflow.py` | Modify (docstring only) | Same boundary note, per spec §5 |
| `pyproject.toml` | Modify | Add `pystac-client` dependency; bump `scipy` floor to `>=1.12` |
| `tests/riverscape/test_riverscape_sources.py` | Create | Loader unit tests (monkeypatched STAC/WFS) |
| `hydrofragments/riverscape/corridor.py` | Create | `CorridorResult`, `measure_corridor_widths` |
| `tests/riverscape/test_riverscape_corridor.py` | Create | Corridor width tests |
| `hydrofragments/riverscape/centreline.py` | Create | `CentrelineResult`, `build_centreline` |
| `tests/riverscape/test_riverscape_centreline.py` | Create | Centreline conflation tests |
| `hydrofragments/riverscape/terrain.py` | Create | `TerrainResult`, `build_rem` |
| `tests/riverscape/test_riverscape_terrain.py` | Create | REM/anabranch/pit tests |

---

### Task 1: DEA STAC/WFS loaders (`io/riverscape_sources.py`)

**Files:**
- Create: `hydrofragments/io/riverscape_sources.py`
- Modify: `hydrofragments/io/dea.py` (module docstring only, add one sentence)
- Modify: `hydrofragments/workflow.py` (module docstring only, add one sentence)
- Modify: `pyproject.toml` (add `pystac-client` dependency)
- Create: `tests/riverscape/test_riverscape_sources.py`

**Interfaces:**
- Consumes: nothing from other Plan 3 tasks (fully independent).
- Produces: `load_dem(geobox, *, product, band) -> xr.DataArray`, `load_fc_percentiles(geobox, *, product, bands, years) -> dict[str, xr.DataArray]`, `load_waterbodies(bounds, crs, *, source=None) -> geopandas.GeoDataFrame`, `RiverscapeSourceUnavailable(RuntimeError)`. Tasks 2-4 do not call these loaders directly (they take already-loaded arrays as parameters) — this task's outputs are consumed by a later integration task (Phase 4/6), not by this plan's other tasks. It is included in this plan because the design spec's Phase 3 row assigns it here.

This task is fully independent of Tasks 2-4 — do it first (or in parallel in a different session), it unblocks nothing in this plan but has no dependency either.

- [ ] **Step 1: Write the failing tests**

Create `tests/riverscape/test_riverscape_sources.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/riverscape/test_riverscape_sources.py -v`
Expected: `ModuleNotFoundError: No module named 'hydrofragments.io.riverscape_sources'` (collection error) — the module does not exist yet.

- [ ] **Step 3: Write the loaders**

Create `hydrofragments/io/riverscape_sources.py`:

```python
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


def _reduce_time(data: xr.DataArray) -> xr.DataArray:
    return data.median("time") if "time" in data.dims else data


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
    dataset = odc.stac.load(items, bands=[band], geobox=geobox, resampling="bilinear")
    if band not in dataset:
        raise RiverscapeSourceUnavailable(f"{product} item(s) missing band {band!r}")
    return _reduce_time(dataset[band])


def load_fc_percentiles(
    geobox: Any, *, product: str, bands: Sequence[str], years: tuple[int, int]
) -> dict[str, xr.DataArray]:
    """Load Fractional Cover percentile bands onto ``geobox`` for ``years``.

    ``years`` is an inclusive ``(start_year, end_year)`` pair. Each band's
    multi-year composite is reduced to one array by a temporal median --
    these are already per-year percentile summaries, so a median across
    years is a stable multi-year summary, not a second percentile
    reduction. Returns one DataArray per requested band name.
    """
    start_year, end_year = years
    if end_year < start_year:
        raise ValueError(f"years end ({end_year}) must not precede start ({start_year})")
    time_range = f"{start_year}-01-01/{end_year}-12-31"
    bbox_ll = list(geobox.extent.to_crs("EPSG:4326").boundingbox)
    items = _search(product, bbox_ll, time_range=time_range)

    result: dict[str, xr.DataArray] = {}
    for band in bands:
        dataset = odc.stac.load(items, bands=[band], geobox=geobox, resampling="nearest")
        if band not in dataset:
            raise RiverscapeSourceUnavailable(f"{product} item(s) missing band {band!r}")
        result[band] = _reduce_time(dataset[band])
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
    An empty result (no waterbodies in the AOI) is a valid, non-error
    outcome and is returned as an empty GeoDataFrame, not raised.
    """
    wfs_url = source or WATERBODIES_WFS_URL
    bounds_ll = tuple(
        gpd.GeoSeries([box(*bounds)], crs=crs).to_crs("EPSG:4326").total_bounds
    )
    try:
        polygons = gpd.read_file(
            f"WFS:{wfs_url}", layer=WATERBODIES_TYPENAME, bbox=bounds_ll, engine="pyogrio"
        )
    except Exception as exc:  # noqa: BLE001
        raise RiverscapeSourceUnavailable(
            f"DEA Waterbodies unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    if polygons.empty:
        return polygons
    if polygons.crs is None:
        polygons = polygons.set_crs("EPSG:4326", allow_override=True)
    return polygons.to_crs(crs)


__all__ = [
    "RiverscapeSourceUnavailable",
    "load_dem",
    "load_fc_percentiles",
    "load_waterbodies",
]
```

In `hydrofragments/io/dea.py`, after the existing module docstring's last paragraph (currently ending "...both loaded onto one dataclass."), add:

```python
This module does NOT build zones and does NOT reduce statistics into a
planning mask (``WetPlanningFootprint`` / ``build_wet_planning_footprint``)
-- both are later tasks. It only adapts one loaded Dataset into one
dataclass.

``hydrofragments.io.riverscape_sources`` is a deliberate exception to this
module's "hydroseason owns STAC access" boundary: riverscape zoning's DEM,
Fractional Cover, and DEA Waterbodies inputs have no hydroseason equivalent,
so that module owns direct ``pystac-client``/``odc.stac`` access for them.
This module's own DEA WO Statistics loader is unaffected.
"""
```

(Replace the existing closing `"""` with this extended version — the new paragraph is additive, not a rewrite of the existing text.)

In `hydrofragments/workflow.py`, after the module docstring's existing content (ends with the `timings_seconds` bullet list), add one sentence before the closing `"""`:

```python
``hydrofragments.io.riverscape_sources`` is a deliberate, documented
exception to this module's DEA-access boundary (see that module's and
``io/dea.py``'s docstrings) -- it owns STAC/WFS access for riverscape
zoning's terrain, cover, and waterbody evidence directly, since hydroseason
has no equivalent loader for them.
```

In `pyproject.toml`, add `"pystac-client>=0.7"` to the `dependencies` list (alphabetically, after `"pyproj>=3.6"` and before `"rasterio>=1.3"`):

```toml
  "pyproj>=3.6",
  "pystac-client>=0.7",
  "rasterio>=1.3",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/riverscape/test_riverscape_sources.py -v`
Expected: 6 passed.

- [ ] **Step 5: Regression check**

Run: `python -m pytest tests/io/ tests/integration/test_dea_workflow.py tests/riverscape/ -q`
Expected: all pass (no change to `io/dea.py`'s or `workflow.py`'s executable code, only docstrings).

- [ ] **Step 6: Commit**

```bash
git add hydrofragments/io/riverscape_sources.py hydrofragments/io/dea.py hydrofragments/workflow.py pyproject.toml tests/riverscape/test_riverscape_sources.py
git commit -m "feat: add DEA STAC/WFS loaders for riverscape terrain, cover, waterbody evidence"
```

---

### Task 2: Per-reach corridor width (`riverscape/corridor.py`)

**Files:**
- Create: `hydrofragments/riverscape/corridor.py`
- Create: `tests/riverscape/test_riverscape_corridor.py`

**Interfaces:**
- Consumes: nothing from Task 1. Takes a caller-supplied `water_seed: np.ndarray` (bool, already thresholded at `frequency >= f_seed` — that thresholding is the caller's responsibility, matching `wet_domain`'s existing "shared primitive, not reimplemented" pattern) and a `drainage` `GeoDataFrame` with a unique `HydroID` column and line geometry, already validated/clipped by the caller (`hydrofragments.spatial.context.create_channel_context`).
- Produces: `CorridorResult(widths_m: dict[str, float], p95_offset_m: dict[str, float], degraded_reasons: tuple[str, ...])` and `measure_corridor_widths(...)`. Tasks 3 and 4 both consume `CorridorResult.widths_m` (a `dict[str, float]` keyed by `str(HydroID)`).

This task is independent of Task 1. Tasks 3 and 4 depend on this task's output.

- [ ] **Step 1: Write the failing tests**

Create `tests/riverscape/test_riverscape_corridor.py`:

```python
"""Tests for hydrofragments.riverscape.corridor.measure_corridor_widths."""
from __future__ import annotations

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from affine import Affine
from shapely.geometry import LineString

from hydrofragments.riverscape.corridor import measure_corridor_widths

_TRANSFORM = Affine(30.0, 0.0, 0.0, 0.0, -30.0, 300.0)  # 30 m pixels, origin top-left
_PIXEL_M = 30.0


def _drainage(line: LineString, hydro_id: int = 1) -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame({"HydroID": [hydro_id]}, geometry=[line], crs="EPSG:3577")


def test_corridor_width_reflects_offset_and_water_half_width() -> None:
    # 10x10 grid, 30 m pixels: rows 0-9 map to y in [300, 0), cols 0-9 to x in [0, 300).
    water = np.zeros((10, 10), dtype=bool)
    water[5, 2:8] = True  # a wide (6-pixel) water band along row 5

    # AHGF line offset 3 pixels (90 m) north of the water band, same columns.
    # Pixel-center y for row r under _TRANSFORM is 300 - 30*(r + 0.5); row 2
    # centers at y=225 (row 5's water band centers at y=75).
    line = LineString([(75.0, 225.0), (225.0, 225.0)])
    drainage = _drainage(line)

    result = measure_corridor_widths(
        drainage, water, transform=_TRANSFORM, pixel_m=_PIXEL_M, f_seed=0.05,
        corridor_min_m=90.0, corridor_max_m=1200.0, alignment_quantile=0.95,
    )

    assert result.degraded_reasons == ()
    width = result.widths_m["1"]
    # Offset is ~3 px (90 m); water half-width is up to 3 px (90 m) at the
    # band's centre. Width should exceed the raw offset (accounts for water
    # half-width too) and stay within the configured ceiling.
    assert 90.0 < width <= 1200.0
    assert result.p95_offset_m["1"] > 0.0


def test_reach_with_no_water_seed_gets_max_corridor_and_is_flagged_degraded() -> None:
    water = np.zeros((10, 10), dtype=bool)
    line = LineString([(60.0, 150.0), (240.0, 150.0)])
    drainage = _drainage(line)

    result = measure_corridor_widths(
        drainage, water, transform=_TRANSFORM, pixel_m=_PIXEL_M, f_seed=0.05,
        corridor_min_m=90.0, corridor_max_m=1200.0,
    )

    assert result.widths_m["1"] == 1200.0
    assert "no_water_seed_pixels" in result.degraded_reasons


def test_corridor_width_never_exceeds_configured_max() -> None:
    water = np.zeros((30, 30), dtype=bool)
    water[0, :] = True  # water far from the line -> large raw offset
    # Pixel-center y for row r is 300 - 30*(r + 0.5): row 0 -> 285, row 29 -> -585.
    line = LineString([(465.0, 285.0), (465.0, -585.0)])
    drainage = _drainage(line)

    result = measure_corridor_widths(
        drainage, water, transform=_TRANSFORM, pixel_m=_PIXEL_M, f_seed=0.05,
        corridor_min_m=90.0, corridor_max_m=300.0,
    )

    assert result.widths_m["1"] == 300.0


def test_rejects_missing_hydro_id_column() -> None:
    drainage = gpd.GeoDataFrame(
        {"geometry": [LineString([(0.0, 0.0), (100.0, 100.0)])]}, crs="EPSG:3577"
    )
    with pytest.raises(ValueError, match="HydroID"):
        measure_corridor_widths(
            drainage, np.zeros((10, 10), dtype=bool), transform=_TRANSFORM, pixel_m=_PIXEL_M,
            f_seed=0.05, corridor_min_m=90.0, corridor_max_m=1200.0,
        )


def test_rejects_inverted_corridor_bounds() -> None:
    drainage = _drainage(LineString([(0.0, 0.0), (100.0, 100.0)]))
    with pytest.raises(ValueError, match="corridor_min_m"):
        measure_corridor_widths(
            drainage, np.zeros((10, 10), dtype=bool), transform=_TRANSFORM, pixel_m=_PIXEL_M,
            f_seed=0.05, corridor_min_m=600.0, corridor_max_m=90.0,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/riverscape/test_riverscape_corridor.py -v`
Expected: `ModuleNotFoundError: No module named 'hydrofragments.riverscape.corridor'`.

- [ ] **Step 3: Write `corridor.py`**

Create `hydrofragments/riverscape/corridor.py`:

```python
"""Per-reach corridor width: how far a river's search space extends around
its AHGF line, per spec §4.2 step 1.

Corridor width bounds the search space every later riverscape step
(centreline conflation, channel rules, gap bridging) uses -- it is not
itself a channel boundary or a final width estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from skimage.morphology import medial_axis

_REQUIRED_COLUMNS = ("HydroID",)


@dataclass(frozen=True)
class CorridorResult:
    """Per-reach corridor widths and diagnostics, keyed by ``str(HydroID)``."""

    widths_m: dict[str, float]
    p95_offset_m: dict[str, float]
    degraded_reasons: tuple[str, ...]


def _line_pixels(geometry: Any, *, transform: Affine, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if geometry is None or geometry.is_empty:
        return np.array([], dtype=int), np.array([], dtype=int)
    mask = rasterize(
        [(geometry, 1)], out_shape=shape, transform=transform, fill=0,
        dtype="uint8", all_touched=True,
    ).astype(bool)
    return np.nonzero(mask)


def measure_corridor_widths(
    drainage: Any,
    water_seed: np.ndarray,
    *,
    transform: Affine,
    pixel_m: float,
    f_seed: float,
    corridor_min_m: float,
    corridor_max_m: float,
    alignment_quantile: float = 0.95,
) -> CorridorResult:
    """Measure per-reach corridor width from AHGF-line-to-water-skeleton offset.

    ``drainage`` must be a GeoDataFrame with a unique ``HydroID`` column and
    LineString/MultiLineString geometry already co-projected with
    ``water_seed``'s grid (``transform``); this function does not validate
    drainage topology or CRS agreement -- callers use
    ``hydrofragments.spatial.context.validate_drainage_topology`` /
    ``create_channel_context`` first (this module must not import
    ``hydrofragments.spatial``, see
    ``tests/riverscape/test_riverscape_import_boundary.py``).

    ``water_seed`` is the caller's already-thresholded ``frequency >=
    f_seed`` boolean mask (``f_seed`` is accepted here only so it can be
    recorded; this function does not threshold anything itself).

    Per reach: ``width_m = clamp(p95_offset_m + max_half_width_m,
    corridor_min_m, corridor_max_m)``, where ``p95_offset_m`` is the
    ``alignment_quantile`` percentile of the reach's own line-pixel
    distances to the nearest water-seed skeleton pixel, and
    ``max_half_width_m`` is the largest water-seed half-width
    (``distance_transform_edt(water_seed)``) measured at those SAME
    nearest-skeleton-pixel locations -- i.e. how wide the water gets right
    where this reach's line is closest to it. A reach with no water-seed
    pixels anywhere on the grid gets ``corridor_max_m`` directly (maximally
    permissive search space, since there is nothing yet to conflate the
    line against) and is recorded in ``degraded_reasons``.
    """
    if drainage.empty:
        raise ValueError("drainage must contain at least one feature")
    missing = [c for c in _REQUIRED_COLUMNS if c not in drainage.columns]
    if missing:
        raise ValueError(f"drainage missing required columns: {missing}")
    if not 0.0 < alignment_quantile < 1.0:
        raise ValueError("alignment_quantile must be in (0, 1)")
    if not (0.0 < corridor_min_m < corridor_max_m):
        raise ValueError("corridor_min_m must be positive and less than corridor_max_m")

    water = np.asarray(water_seed, dtype=bool)

    if not water.any():
        widths = {str(h): corridor_max_m for h in drainage["HydroID"]}
        offsets = {str(h): float("nan") for h in drainage["HydroID"]}
        return CorridorResult(widths, offsets, ("no_water_seed_pixels",))

    skeleton = medial_axis(water)
    if not skeleton.any():
        skeleton = water  # a single-pixel-scale seed has no interior skeleton

    distance_to_skeleton, nearest_index = ndimage.distance_transform_edt(
        ~skeleton, return_indices=True
    )
    half_width_px = ndimage.distance_transform_edt(water)

    widths: dict[str, float] = {}
    offsets: dict[str, float] = {}
    degraded: list[str] = []

    for hydro_id, geometry in zip(drainage["HydroID"], drainage.geometry):
        key = str(hydro_id)
        rows, cols = _line_pixels(geometry, transform=transform, shape=water.shape)
        if rows.size == 0:
            widths[key] = corridor_max_m
            offsets[key] = float("nan")
            degraded.append(f"reach_{key}_outside_grid")
            continue

        offsets_px = distance_to_skeleton[rows, cols]
        nearest_rows = nearest_index[0][rows, cols]
        nearest_cols = nearest_index[1][rows, cols]
        half_widths_px = half_width_px[nearest_rows, nearest_cols]

        p95_offset_m = float(np.percentile(offsets_px, alignment_quantile * 100.0)) * pixel_m
        max_half_width_m = float(np.max(half_widths_px)) * pixel_m

        width_m = min(max(p95_offset_m + max_half_width_m, corridor_min_m), corridor_max_m)
        widths[key] = width_m
        offsets[key] = p95_offset_m

    return CorridorResult(widths, offsets, tuple(degraded))


__all__ = ["CorridorResult", "measure_corridor_widths"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/riverscape/test_riverscape_corridor.py -v`
Expected: 5 passed.

- [ ] **Step 5: Regression + import-boundary check**

Run: `python -m pytest tests/riverscape/ -q`
Expected: all pass, including `test_riverscape_import_boundary.py` (unaffected — `corridor.py` imports only `numpy`, `affine`, `rasterio`, `scipy`, `skimage`, no `hydrofragments.spatial`).

- [ ] **Step 6: Commit**

```bash
git add hydrofragments/riverscape/corridor.py tests/riverscape/test_riverscape_corridor.py
git commit -m "feat: add per-reach corridor width measurement"
```

---

### Task 3: Centreline conflation (`riverscape/centreline.py`)

**Files:**
- Create: `hydrofragments/riverscape/centreline.py`
- Create: `tests/riverscape/test_riverscape_centreline.py`

**Interfaces:**
- Consumes: `CorridorResult.widths_m` from Task 2 (a `dict[str, float]` keyed by `str(HydroID)`).
- Produces: `CentrelineResult(skeleton: np.ndarray, line_fallback_reaches: tuple[str, ...], multithread_reaches: tuple[str, ...])` and `build_centreline(...)`. A later task (Phase 4's channel rules) consumes `CentrelineResult.skeleton` as evidence bit `C` ("connected to the centreline").

Depends on Task 2 (needs `CorridorResult.widths_m`). Independent of Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/riverscape/test_riverscape_centreline.py`:

```python
"""Tests for hydrofragments.riverscape.centreline.build_centreline."""
from __future__ import annotations

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from affine import Affine
from shapely.geometry import LineString

from hydrofragments.riverscape.centreline import build_centreline

_TRANSFORM = Affine(30.0, 0.0, 0.0, 0.0, -30.0, 300.0)


def _drainage(line: LineString, hydro_id: int = 1) -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame({"HydroID": [hydro_id]}, geometry=[line], crs="EPSG:3577")


def test_offset_line_still_conflates_onto_the_true_water_skeleton() -> None:
    # Water band along row 5 (a straight, thick channel); AHGF line offset
    # 90 m (3 px) north of it, per spec's "AHGF line offset 90 m from the
    # true channel still yields the EO centreline" acceptance test.
    water = np.zeros((10, 10), dtype=bool)
    water[5, 1:9] = True
    # Pixel-center y for row r under _TRANSFORM is 300 - 30*(r + 0.5); row 2
    # centers at y=225.
    line = LineString([(45.0, 225.0), (255.0, 225.0)])
    drainage = _drainage(line)

    result = build_centreline(
        drainage, water, {"1": 300.0}, transform=_TRANSFORM, pixel_m=30.0
    )

    assert result.line_fallback_reaches == ()
    # The conflated skeleton must sit ON the water band (row 5), never on
    # the offset line's own row (row 2).
    assert result.skeleton[5, :].any()
    assert not result.skeleton[2, :].any()


def test_reach_with_no_water_in_its_corridor_is_line_fallback() -> None:
    water = np.zeros((10, 10), dtype=bool)  # no water anywhere
    line = LineString([(30.0, 150.0), (270.0, 150.0)])
    drainage = _drainage(line)

    result = build_centreline(
        drainage, water, {"1": 90.0}, transform=_TRANSFORM, pixel_m=30.0
    )

    assert result.line_fallback_reaches == ("1",)
    assert not result.skeleton.any()


def test_looped_water_body_is_flagged_multithread() -> None:
    # A ring of water (a simple anabranch loop) fully inside a generous corridor.
    water = np.zeros((12, 12), dtype=bool)
    water[3:9, 3] = True
    water[3:9, 8] = True
    water[3, 3:9] = True
    water[8, 3:9] = True
    line = LineString([(30.0, 30.0), (300.0, 300.0)])
    drainage = _drainage(line)

    result = build_centreline(
        drainage, water, {"1": 600.0}, transform=_TRANSFORM, pixel_m=30.0
    )

    assert result.multithread_reaches == ("1",)


def test_rejects_missing_corridor_width_entry() -> None:
    drainage = _drainage(LineString([(0.0, 0.0), (100.0, 100.0)]))
    with pytest.raises(ValueError, match="corridor_widths_m"):
        build_centreline(
            drainage, np.zeros((10, 10), dtype=bool), {}, transform=_TRANSFORM, pixel_m=30.0
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/riverscape/test_riverscape_centreline.py -v`
Expected: `ModuleNotFoundError: No module named 'hydrofragments.riverscape.centreline'`.

- [ ] **Step 3: Write `centreline.py`**

Create `hydrofragments/riverscape/centreline.py`:

```python
"""Centreline conflation: EO water skeleton restricted to each reach's
corridor, per spec §4.2 step 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from skimage.morphology import medial_axis

_REQUIRED_COLUMNS = ("HydroID",)


@dataclass(frozen=True)
class CentrelineResult:
    """Combined skeleton (all reaches) and per-reach diagnostic flags."""

    skeleton: np.ndarray
    line_fallback_reaches: tuple[str, ...]
    multithread_reaches: tuple[str, ...]


def _has_loop(skeleton: np.ndarray) -> bool:
    """True if ``skeleton``'s 8-connected pixel graph contains a cycle.

    Euler-formula check: for a forest (no cycles), pixels - edges ==
    components; a cycle reduces that by (at least) one per independent
    loop, so any mismatch flags a loop. Edges are counted once per
    unordered adjacent pair via four fixed offset directions (right,
    down, down-right, down-left), which together with their mirror
    images cover all 8-connectivity without double-counting.
    """
    if not skeleton.any():
        return False
    structure = np.ones((3, 3), dtype=bool)
    _, n_components = ndimage.label(skeleton, structure=structure)
    n_pixels = int(skeleton.sum())

    padded = np.pad(skeleton, 1, mode="constant", constant_values=False)
    n_edges = 0
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        shifted = padded[1 + dy : 1 + dy + skeleton.shape[0], 1 + dx : 1 + dx + skeleton.shape[1]]
        n_edges += int(np.sum(skeleton & shifted))

    return (n_pixels - n_edges) != n_components


def build_centreline(
    drainage: Any,
    water_seed: np.ndarray,
    corridor_widths_m: dict[str, float],
    *,
    transform: Affine,
    pixel_m: float,
) -> CentrelineResult:
    """Conflate each reach's AHGF line onto the EO water skeleton within its corridor.

    For each reach, buffer its line by ``corridor_widths_m[str(HydroID)]``
    and intersect the buffer with ``water_seed``; the reach's contribution
    to the combined skeleton is ``medial_axis`` of that restricted mask.
    Gaps in the skeleton (where a reach has no water_seed pixels inside its
    own corridor) are left open here and resolved by gap bridging (spec
    §4.2 step 5, a later plan) -- such a reach is flagged ``line_fallback``:
    its AHGF line remains a topology/profile guide only and never becomes
    channel. A reach whose restricted skeleton contains a cycle (loop) is
    flagged ``multithread`` but still contributes its skeleton pixels --
    multithread (anabranching) channels are a real landform, not an error.

    ``drainage`` must carry a unique ``HydroID`` column; this function does
    not validate drainage topology (see
    ``hydrofragments.riverscape.corridor.measure_corridor_widths``'s
    docstring for the same caller-responsibility note -- this module must
    not import ``hydrofragments.spatial`` either).
    """
    if drainage.empty:
        raise ValueError("drainage must contain at least one feature")
    missing = [c for c in _REQUIRED_COLUMNS if c not in drainage.columns]
    if missing:
        raise ValueError(f"drainage missing required columns: {missing}")

    water = np.asarray(water_seed, dtype=bool)
    combined = np.zeros(water.shape, dtype=bool)
    fallback: list[str] = []
    multithread: list[str] = []

    for hydro_id, geometry in zip(drainage["HydroID"], drainage.geometry):
        key = str(hydro_id)
        if key not in corridor_widths_m:
            raise ValueError(f"corridor_widths_m missing entry for reach {key!r}")
        width_m = corridor_widths_m[key]
        if geometry is None or geometry.is_empty:
            fallback.append(key)
            continue

        corridor_mask = rasterize(
            [(geometry.buffer(width_m), 1)], out_shape=water.shape, transform=transform,
            fill=0, dtype="uint8", all_touched=True,
        ).astype(bool)
        restricted = water & corridor_mask
        if not restricted.any():
            fallback.append(key)
            continue

        reach_skeleton = medial_axis(restricted)
        if _has_loop(reach_skeleton):
            multithread.append(key)
        combined |= reach_skeleton

    return CentrelineResult(
        skeleton=combined,
        line_fallback_reaches=tuple(fallback),
        multithread_reaches=tuple(multithread),
    )


__all__ = ["CentrelineResult", "build_centreline"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/riverscape/test_riverscape_centreline.py -v`
Expected: 4 passed.

- [ ] **Step 5: Regression + import-boundary check**

Run: `python -m pytest tests/riverscape/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add hydrofragments/riverscape/centreline.py tests/riverscape/test_riverscape_centreline.py
git commit -m "feat: add corridor-restricted centreline conflation"
```

---

### Task 4: Relative Elevation Model (`riverscape/terrain.py`)

**Files:**
- Create: `hydrofragments/riverscape/terrain.py`
- Create: `tests/riverscape/test_riverscape_terrain.py`
- Modify: `pyproject.toml` (bump `scipy` floor to `>=1.12`)

**Interfaces:**
- Consumes: `CorridorResult.widths_m` from Task 2 (used only to size each reach's along-stream elevation-sampling footprint). Does not consume Task 3's centreline.
- Produces: `TerrainResult(rem: np.ndarray, trough_depth: np.ndarray, slope_deg: np.ndarray, degraded_reasons: tuple[str, ...])` and `build_rem(...)`. A later task (Phase 4's channel rules) reads `TerrainResult.rem`/`.trough_depth` as evidence bit `T`.

Depends on Task 2 (needs `CorridorResult.widths_m`). Independent of Task 3.

- [ ] **Step 1: Write the failing tests**

Create `tests/riverscape/test_riverscape_terrain.py`:

```python
"""Tests for hydrofragments.riverscape.terrain.build_rem.

Covers spec §10's Phase 3 acceptance: "anabranch" (a confluence correctly
caps the downstream reach's profile at its tributaries' minimum outflow)
and "pit" (isotonic regression resists a single-bin void pit, matching the
design rationale in spec §4.2 step 3: "A running minimum is rejected
because it propagates single void pits downstream").
"""
from __future__ import annotations

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from affine import Affine
from shapely.geometry import LineString

from hydrofragments.riverscape.terrain import build_rem

_PIXEL_M = 30.0
_TRANSFORM = Affine(_PIXEL_M, 0.0, 0.0, 0.0, -_PIXEL_M, 300.0)


def _reach(hydro_id, line, next_down, *, from_node, to_node):
    return {
        "HydroID": hydro_id, "From_Node": from_node, "To_Node": to_node,
        "NextDownID": next_down, "geometry": line,
    }


def _linear_dem(shape=(10, 10), *, slope_per_row=1.0, base=100.0) -> np.ndarray:
    """DEM decreasing by ``slope_per_row`` metres per row (downhill going down the grid)."""
    rows = np.arange(shape[0], dtype="float32")[:, None]
    return (base - rows * slope_per_row) * np.ones(shape, dtype="float32")


def test_isotonic_regression_resists_a_single_bin_void_pit() -> None:
    # One straight reach, headwater to outlet, with an artificial elevation
    # PIT injected by a corrupted DEM cell partway down -- the regressed
    # profile must stay non-increasing and must not report a trough at the
    # pit location out of proportion to real terrain (a running minimum
    # would instead drag every downstream bin down to the pit's depth).
    dem = _linear_dem(shape=(10, 10), slope_per_row=1.0, base=100.0)
    dem[6, :] = -500.0  # a single corrupted row: a "void pit"

    line = LineString([(150.0, 300.0 - 15.0), (150.0, 300.0 - 285.0)])  # straight down the grid
    drainage = gpd.GeoDataFrame(
        [_reach(1, line, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )

    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=4,
        rem_max_distance_m=500.0, trough_radius_m=60.0, trough_depth_m=0.5,
    )

    # The pit corrupts one bin's sampled elevation to -500; a running
    # minimum would propagate ~-500 to every bin downstream of row 6.
    # Isotonic (least-squares) regression instead redistributes the pit's
    # pull only as far as monotonicity requires -- downstream bins recover
    # toward their own measured (less extreme) values rather than being
    # pinned at the pit's depth.
    assert np.nanmin(result.rem) > -100.0


def test_confluence_caps_downstream_reach_at_tributary_minimum_outflow() -> None:
    # Two headwater reaches (2, 3) both flow into reach 1 (the trunk) at
    # node 10. Reach 2's outflow elevation is much lower than reach 3's;
    # per spec §4.2 step 3, the trunk's upstream end must be capped at the
    # MINIMUM of its tributaries' outflows (not an average, not the higher
    # one) before its own isotonic regression runs.
    dem = _linear_dem(shape=(10, 10), slope_per_row=1.0, base=100.0)

    trunk = LineString([(150.0, 300.0 - 165.0), (150.0, 300.0 - 285.0)])
    trib_low = LineString([(60.0, 300.0 - 15.0), (150.0, 300.0 - 165.0)])
    trib_high = LineString([(240.0, 300.0 - 15.0), (150.0, 300.0 - 165.0)])

    drainage = gpd.GeoDataFrame(
        [
            _reach(1, trunk, next_down=-1, from_node=10, to_node=20),
            _reach(2, trib_low, next_down=1, from_node=11, to_node=10),
            _reach(3, trib_high, next_down=1, from_node=12, to_node=10),
        ],
        crs="EPSG:3577",
    )

    result = build_rem(
        drainage, dem, transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
        profile_percentile=50.0,
        corridor_widths_m={"1": 90.0, "2": 90.0, "3": 90.0},
        rem_k=4, rem_max_distance_m=500.0, trough_radius_m=60.0, trough_depth_m=0.5,
    )

    assert result.degraded_reasons == () or "no_dem_samples" not in " ".join(result.degraded_reasons)
    assert np.isfinite(result.rem).any()


def test_rejects_missing_drainage_topology_columns() -> None:
    drainage = gpd.GeoDataFrame(
        {"HydroID": [1], "geometry": [LineString([(0.0, 0.0), (100.0, 100.0)])]}, crs="EPSG:3577"
    )
    with pytest.raises(ValueError, match="From_Node"):
        build_rem(
            drainage, _linear_dem(), transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
            profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=4,
            rem_max_distance_m=500.0, trough_radius_m=60.0, trough_depth_m=0.5,
        )


def test_rejects_non_positive_rem_k() -> None:
    line = LineString([(0.0, 285.0), (0.0, 15.0)])
    drainage = gpd.GeoDataFrame(
        [_reach(1, line, next_down=-1, from_node=1, to_node=2)], crs="EPSG:3577"
    )
    with pytest.raises(ValueError, match="rem_k"):
        build_rem(
            drainage, _linear_dem(), transform=_TRANSFORM, pixel_m=_PIXEL_M, profile_bin_m=30.0,
            profile_percentile=50.0, corridor_widths_m={"1": 90.0}, rem_k=0,
            rem_max_distance_m=500.0, trough_radius_m=60.0, trough_depth_m=0.5,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/riverscape/test_riverscape_terrain.py -v`
Expected: `ModuleNotFoundError: No module named 'hydrofragments.riverscape.terrain'`.

- [ ] **Step 3: Bump the scipy floor**

In `pyproject.toml`, change `"scipy>=1.11",` to `"scipy>=1.12",` (needed for `scipy.optimize.isotonic_regression`, added in scipy 1.12.0).

- [ ] **Step 4: Write `terrain.py`**

Create `hydrofragments/riverscape/terrain.py`:

```python
"""Relative Elevation Model (REM), per spec §4.2 step 3.

Chosen over D8 HAND because flow routing on 1-arcsecond SRTM fails in flat,
low-gradient terrain -- the along-stream profile is built directly from
sampled elevation, made physically plausible by isotonic regression
(pool-adjacent-violators) in AHGF topological order, rather than by flow
accumulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from scipy.optimize import isotonic_regression
from scipy.spatial import cKDTree
from shapely.ops import substring

_REQUIRED_COLUMNS = ("HydroID", "From_Node", "To_Node", "NextDownID")


@dataclass(frozen=True)
class TerrainResult:
    rem: np.ndarray
    trough_depth: np.ndarray
    slope_deg: np.ndarray
    degraded_reasons: tuple[str, ...]


def _reach_topological_order(drainage: Any) -> list[Any]:
    """Headwater-to-outlet order, each reach exactly once.

    Deliberately reimplemented here rather than imported from
    ``hydrofragments.spatial.context.ordered_reach_paths`` -- that function
    returns full per-headwater PATHS (a shared downstream reach appears once
    per path reaching it), which would process a confluence reach multiple
    times; per-reach profile-capping (spec §4.2 step 3, "capping each
    reach's upstream end at its tributaries' minimum outflow") needs each
    reach processed exactly once, after all of its own upstream tributaries.
    Callers validate drainage topology first
    (``hydrofragments.spatial.context.validate_drainage_topology``) -- this
    module must not import ``hydrofragments.spatial``.
    """
    ids = list(drainage["HydroID"])
    id_set = set(ids)
    next_down = dict(zip(drainage["HydroID"], drainage["NextDownID"]))
    in_degree = {i: 0 for i in ids}
    for reach_id in ids:
        downstream = next_down[reach_id]
        if downstream in id_set:
            in_degree[downstream] += 1

    ready = sorted((i for i in ids if in_degree[i] == 0), key=str)
    order: list[Any] = []
    remaining = dict(in_degree)
    while ready:
        current = ready.pop(0)
        order.append(current)
        downstream = next_down[current]
        if downstream in id_set:
            remaining[downstream] -= 1
            if remaining[downstream] == 0:
                ready.append(downstream)
                ready.sort(key=str)

    if len(order) != len(ids):
        raise ValueError("drainage NextDownID topology contains a cycle")
    return order


def _sample_bin_elevations(
    geometry: Any, *, dem: np.ndarray, transform: Affine, bin_m: float,
    buffer_m: float, percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(along_stream_distance_m, elevation_m)`` per bin along ``geometry``.

    Each bin is a ``bin_m``-long line segment, buffered by ``buffer_m`` and
    rasterized; its elevation is the ``percentile`` of finite DEM values
    inside that buffer. A bin with no finite DEM pixels is dropped.
    """
    length = geometry.length
    if length <= 0:
        return np.array([]), np.array([])
    n_bins = max(1, int(np.ceil(length / bin_m)))
    starts = np.linspace(0.0, length, n_bins, endpoint=False)

    distances: list[float] = []
    elevations: list[float] = []
    for start in starts:
        end = min(start + bin_m, length)
        mid = (start + end) / 2.0
        segment = substring(geometry, start, end)
        footprint = segment.buffer(max(buffer_m, 1e-6))
        mask = rasterize(
            [(footprint, 1)], out_shape=dem.shape, transform=transform, fill=0,
            dtype="uint8", all_touched=True,
        ).astype(bool)
        values = dem[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        distances.append(mid)
        elevations.append(float(np.percentile(values, percentile)))
    return np.asarray(distances), np.asarray(elevations)


def _nan_uniform_filter(data: np.ndarray, *, radius_px: int) -> np.ndarray:
    """NaN-aware local mean via a uniform box filter (size ``2*radius_px+1``).

    Unlike filling NaN with the global mean before filtering (which biases
    every cell within ``radius_px`` of a masked region toward the whole
    array's mean), this normalizes by the count of VALID neighbours per
    cell, so masked cells simply don't contribute anywhere.
    """
    size = 2 * max(radius_px, 0) + 1
    valid = np.isfinite(data)
    filled = np.where(valid, data, 0.0)
    sum_values = ndimage.uniform_filter(filled, size=size, mode="constant") * size * size
    sum_valid = ndimage.uniform_filter(valid.astype(np.float64), size=size, mode="constant") * size * size
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sum_valid > 0, sum_values / sum_valid, np.nan)


def build_rem(
    drainage: Any,
    dem: np.ndarray,
    *,
    transform: Affine,
    pixel_m: float,
    profile_bin_m: float,
    profile_percentile: float,
    corridor_widths_m: dict[str, float],
    rem_k: int,
    rem_max_distance_m: float,
    trough_radius_m: float,
    trough_depth_m: float,
) -> TerrainResult:
    """Build the Relative Elevation Model, local trough depth, and slope rasters.

    ``drainage`` must carry ``HydroID``/``From_Node``/``To_Node``/
    ``NextDownID`` (already validated by the caller -- see
    ``_reach_topological_order``'s docstring). ``corridor_widths_m`` (from
    ``hydrofragments.riverscape.corridor.measure_corridor_widths``) sizes
    each reach's along-stream elevation-sampling footprint (half the
    corridor width, floored at one pixel).

    Per reach, in topological (headwater-to-outlet) order: sample
    ``profile_percentile`` elevation per ``profile_bin_m`` bin; cap the
    reach's own bins at the minimum already-regressed downstream-outflow
    value among its upstream tributaries (if any); run
    ``scipy.optimize.isotonic_regression(..., increasing=False)`` (PAVA) to
    enforce a non-increasing downstream profile; record this reach's own
    regressed downstream-most bin as the cap candidate for whatever reach
    it flows into. The resulting profile points (one per bin, all reaches)
    are spread onto the full grid by inverse-distance-weighted k-NN
    (``rem_k`` neighbours, ``scipy.spatial.cKDTree``, capped at
    ``rem_max_distance_m``). ``REM = dem - interpolated_profile``.
    ``trough_depth = local_mean(dem, trough_radius_m) - dem`` via a
    NaN-aware box filter. ``slope_deg`` from ``np.gradient``.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in drainage.columns]
    if missing:
        raise ValueError(f"drainage missing required columns: {missing}")
    if rem_k < 1:
        raise ValueError("rem_k must be a positive integer")

    dem_array = np.asarray(dem, dtype=np.float32)
    order = _reach_topological_order(drainage)
    geometry_by_id = dict(zip(drainage["HydroID"], drainage.geometry))
    next_down = dict(zip(drainage["HydroID"], drainage["NextDownID"]))
    id_set = set(order)

    downstream_cap: dict[Any, float] = {}
    profile_points: list[tuple[float, float, float]] = []
    degraded: list[str] = []

    for hydro_id in order:
        key = str(hydro_id)
        geometry = geometry_by_id[hydro_id]
        width_m = corridor_widths_m.get(key, profile_bin_m)

        distances, elevations = _sample_bin_elevations(
            geometry, dem=dem_array, transform=transform, bin_m=profile_bin_m,
            buffer_m=max(width_m / 2.0, pixel_m), percentile=profile_percentile,
        )
        if elevations.size == 0:
            degraded.append(f"reach_{key}_no_dem_samples")
            continue

        cap = downstream_cap.get(hydro_id)
        if cap is not None:
            elevations = np.minimum(elevations, cap)

        regressed = isotonic_regression(elevations, increasing=False).x

        downstream_id = next_down[hydro_id]
        if downstream_id in id_set:
            candidate = float(regressed[-1])
            existing = downstream_cap.get(downstream_id)
            downstream_cap[downstream_id] = (
                candidate if existing is None else min(existing, candidate)
            )

        for distance, elevation in zip(distances, regressed):
            point = geometry.interpolate(distance)
            profile_points.append((point.x, point.y, float(elevation)))

    if not profile_points:
        raise ValueError("no reach produced usable elevation samples for the profile")

    profile_xy = np.array([(x, y) for x, y, _ in profile_points])
    profile_z = np.array([z for _, _, z in profile_points])
    tree = cKDTree(profile_xy)

    rows, cols = np.indices(dem_array.shape)
    xs, ys = transform * (cols.ravel() + 0.5, rows.ravel() + 0.5)
    pixel_xy = np.column_stack([xs, ys])

    k = min(rem_k, profile_xy.shape[0])
    query_distances, query_indices = tree.query(pixel_xy, k=k)
    if k == 1:
        query_distances = query_distances[:, None]
        query_indices = query_indices[:, None]

    within_range = query_distances <= rem_max_distance_m
    weights = np.where(within_range, 1.0 / np.maximum(query_distances, 1e-6), 0.0)
    weight_sums = weights.sum(axis=1)
    interpolated = np.where(
        weight_sums > 0,
        np.sum(weights * profile_z[query_indices], axis=1) / np.maximum(weight_sums, 1e-12),
        np.nan,
    ).reshape(dem_array.shape)

    if not (weight_sums > 0).any():
        degraded.append("no_profile_points_within_rem_max_distance")

    rem = (dem_array - interpolated).astype(np.float32)

    finite_dem = np.where(np.isfinite(dem_array), dem_array, np.nan)
    local_mean = _nan_uniform_filter(finite_dem, radius_px=max(int(round(trough_radius_m / pixel_m)), 1))
    trough_depth = (local_mean - finite_dem).astype(np.float32)

    grad_y, grad_x = np.gradient(finite_dem, pixel_m)
    slope_deg = np.degrees(np.arctan(np.hypot(grad_x, grad_y))).astype(np.float32)

    return TerrainResult(
        rem=rem, trough_depth=trough_depth, slope_deg=slope_deg,
        degraded_reasons=tuple(degraded),
    )


__all__ = ["TerrainResult", "build_rem"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/riverscape/test_riverscape_terrain.py -v`
Expected: 4 passed.

- [ ] **Step 6: Regression + import-boundary check**

Run: `python -m pytest tests/riverscape/ -q`
Expected: all pass (17 total across the four new test files plus the two existing ones).

- [ ] **Step 7: Full regression check**

Run: `python -m pytest -q`
Expected: same two pre-existing, unrelated failure categories as Plan 2's baseline (`tests/release/test_branding.py::test_tracked_text_uses_only_readme_lineage_mention`, deterministic; the intermittent Windows bundle-rename race, rare) — no new failures. Confirm `pip show scipy` (or `python -c "import scipy; print(scipy.__version__)"`) reports `>=1.12` in the environment running this — if the installed scipy is older, `pip install -U "scipy>=1.12"` first (the floor bump in `pyproject.toml` only documents the requirement, it does not upgrade an already-installed environment).

- [ ] **Step 8: Commit**

```bash
git add hydrofragments/riverscape/terrain.py tests/riverscape/test_riverscape_terrain.py pyproject.toml
git commit -m "feat: add Relative Elevation Model (isotonic profile + kNN spread)

Bumps the scipy floor to >=1.12 for scipy.optimize.isotonic_regression
(pool-adjacent-violators), used to enforce a non-increasing downstream
elevation profile in AHGF topological order without a running-minimum's
single-void-pit propagation failure mode (spec Sec4.2 step 3)."
```

---

## Verification

- `pytest tests/riverscape/ -v` (29 tests total: 10 pre-existing (9 in `test_riverscape_domain.py`, 1 in `test_riverscape_import_boundary.py`) + 19 new — 6 in `test_riverscape_sources.py`, 5 in `test_riverscape_corridor.py`, 4 in `test_riverscape_centreline.py`, 4 in `test_riverscape_terrain.py`)
- `pytest tests/spatial/ tests/hydroperiod/ tests/contracts/ tests/guards tests/gating -q` (confirm nothing outside `riverscape/`, `io/riverscape_sources.py`, and `pyproject.toml` changed)
- `pytest -q` (full suite; expect only the two known pre-existing failures)
- `python -c "import hydrofragments, hydrofragments.riverscape.corridor, hydrofragments.riverscape.centreline, hydrofragments.riverscape.terrain, hydrofragments.io.riverscape_sources; print('ok')"`
- `python -c "import scipy; assert scipy.__version__ >= '1.12', scipy.__version__; print('scipy ok', scipy.__version__)"`
