# Riverscape Zoning — Plan 6a: Landform Pipeline and Workflow Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `build_landform` orchestrator, the `zones_from_riverscape` zoning adapter, and `analyze_from_dea`'s `riverscape.mode` branch so a real DEA run can produce a landform × hydroperiod `ZoneResult` instead of occurrence-only zones.

**Architecture:** Three layers, one direction of dependency. `hydrofragments/riverscape/pipeline.py` owns source loading plus the Phase 3–5 kernels in order (corridor → centreline → REM → waterbodies → evidence → channel → bridging → riverine) and returns a `LandformResult`; it never imports `hydrofragments.spatial`, so reach-label rasters are built by the caller and passed in. `hydrofragments/spatial/zones.py` gains `zones_from_riverscape`, which derives the observed-wet domain from DEA WO statistics, calls `build_landform`, classifies hydroperiod with the bridged mask as `unobserved_mask`, and hands both layers to the existing `combine_zones`. `hydrofragments/workflow.py` resolves the `off`/`auto`/`required` mode table, validates drainage before any loader work, and records the riverscape branch's wall time.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`ndimage`, `optimize`, `spatial`), scikit-image (`medial_axis`, `MCP_Geometric`), GeoPandas/Shapely, rasterio (`features.rasterize`, `features.geometry_mask`), affine, xarray + rioxarray, odc-stac/odc-geo/pyogrio/pystac-client (via `hydrofragments.io.riverscape_sources`), pytest with `monkeypatch`.

**Spec:** `docs/superpowers/specs/2026-09-18-riverscape-phase6a-workflow-design.md` (parent: `docs/superpowers/specs/2026-09-14-riverscape-zoning-design.md`; prior phase: `docs/superpowers/specs/2026-09-17-riverscape-phase5-riverine-design.md`)

## Global Constraints

- **Import boundary (spec §1.3):** `hydrofragments.riverscape` must **not** import `hydrofragments.spatial`. Enforced by the existing `tests/riverscape/test_riverscape_import_boundary.py`, which AST-scans every `.py` under `hydrofragments/riverscape/`. `zones_from_riverscape` (in `spatial`) *may* import `riverscape.pipeline`.
- **Reach labels (spec §2 decision 9):** built by the caller (`spatial`/`workflow`) from `hydrofragments/spatial/connectivity_context.py`, never inside `riverscape.pipeline`.
- **Injectable loaders (spec §3.2):** `build_landform` takes `load_dem`, `load_fc_percentiles`, `load_waterbodies` as keyword parameters defaulting to the real `hydrofragments.io.riverscape_sources` functions. No test in this plan makes a network call.
- **No silent catch (spec §4, §3.4):** `build_landform` never catches `RiverscapeSourceUnavailable`. Only `RiverscapeSourceUnavailable` plus the declared no-drainage / no-stats cases map to the `auto` fallback in `workflow`; every other exception propagates.
- **`channel_source` codes (spec §2 decision 6):** `CHANNEL_SOURCE_NONE = 0`, `CHANNEL_SOURCE_OBSERVED = 1`, `CHANNEL_SOURCE_BRIDGED = 2`. Observed always wins over bridged.
- **`channel_confidence` nodata (spec §3.1):** `255` outside channel.
- **Bridged pixels (spec §5):** landform `1` (in-channel) and hydroperiod `4` (unobserved); legacy Zone 1 includes them. `hydrofragments.riverscape.bridging.bridge_gaps` already guarantees `bridged ⊆ ~domain & ~observed` (it masks against `protected = domain & ~observed` and `~observed`); `build_landform` re-applies that intersection once, at packaging time, so the `LandformResult` invariant holds even under an injected bridging stub. `hydrofragments.hydroperiod.classify.classify_hydroperiod` raises if `unobserved_mask` overlaps `domain`, so this invariant is load-bearing.
- **Extended landform domain:** `hydrofragments.riverscape.riverine.assemble_landform` only assigns codes inside its `domain` argument, so `build_landform` passes `domain | bridged_mask` into `build_landform_layer`. Without this, a bridged pixel would be landform `0` while hydroperiod is `4`, and `combine_zones` would raise `"landform and hydroperiod disagree on the zoned extent"`.
- **`upstr_darea` keys are reach *labels*, not `HydroID`s.** `hydrofragments/riverscape/riverine.py::_lookup_area` compares `reach_id == int(key)` against values drawn from `reach_labels`. `reach_keys: Mapping[int, str]` maps the same label ints to `str(HydroID)`.
- **Degraded reasons:** stable sorted unique tuple (`tuple(sorted(set(...)))`), unioned across corridor / centreline / terrain / evidence / channel / bridge / envelope (spec §4 step 12).
- **Provenance keys are stable strings 6b copies verbatim** (spec §9). Do not rename them in this plan's follow-ups.
- **Hash schema:** no bump. No new scientific field is introduced (spec §1.2).
- **Out of scope (spec §1.2):** manifest `zoning` polish, `riverscape_evidence` rasters, `channel_bridges` export packaging, zone-name tables, gating extension (all → 6b); `windows.py` tiling (→ Phase 7); closed-loop bridge topology (deferred); MrVBF.
- **Thresholds stay fractions in config, percent at the comparison boundary.** `cfg.f_seed` / `cfg.f_chan_high` are fractions in `[0, 1]`; `frequency` is percent `0–100`. Convert once: `frequency >= 100.0 * cfg.f_seed`.
- **Decision D1 (plan-local, needed to keep existing tests hermetic):** `RiverscapeConfig.mode` keeps its library default `"auto"`, but `hydrofragments.workflow._default_config` explicitly pins `riverscape.mode = "off"`. `_default_config` is the minimal config `analyze_from_dea` builds when `config=None`; it configures no riverscape sources, and leaving it at `"auto"` would make every existing `tests/integration/test_dea_workflow.py` case that supplies drainage attempt real STAC/WFS reads. Opting in is the caller's explicit act.
- **Decision D2 (plan-local):** `timings["riverscape"]` is carved *out of* `timings["dea_planning"]` (`dea_planning = elapsed - timings.get("riverscape", 0.0)`) so the phases stay disjoint. `hydrofragments/output/finalize.py:480-485` computes `total` as the sum of every non-`total` key; double-counting would inflate `total` past wall-clock.
- **Decision D3 (plan-local, deviates from spec §8's file list):** the workflow-mode tests live in `tests/integration/test_riverscape_modes.py`, not `tests/workflow/`. `analyze_from_dea`'s existing tests are in `tests/integration/test_dea_workflow.py`, there is no `tests/workflow/` package, and spec §7.3 explicitly permits "or extend existing DEA workflow tests".
- **Multithread reaches are not degraded.** `hydrofragments/riverscape/centreline.py` documents anabranching as "a real landform, not an error", so `multithread_reaches` goes to `provenance["reach_counts"]`, never to `degraded_reasons`. `line_fallback_reaches` *does* degrade, as `reach_{key}_line_fallback`.
- **Commands** are run from the repo root (`D:/RLH/5.6/repos/HydroFragments`) with `python -m pytest`.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `hydrofragments/riverscape/pipeline.py` | Create | `LandformResult`, `build_landform`, packaging helpers, drainage-column validation. Owns loader calls and kernel ordering. Imports no `spatial`. |
| `hydrofragments/riverscape/__init__.py` | Modify | Re-export the Phase 6a public surface. |
| `hydrofragments/spatial/zones.py` | Modify | Add `zones_from_riverscape` + `_resolve_years`; extend `__all__`. |
| `hydrofragments/spatial/__init__.py` | Modify | Re-export `zones_from_riverscape`. |
| `hydrofragments/workflow.py` | Modify | Mode table, early drainage validation, reach-label context, geobox resolution, `timings["riverscape"]`. |
| `tests/riverscape/test_riverscape_pipeline.py` | Create | Task 1 tests (stub loaders, real kernels). |
| `tests/spatial/test_zones_from_riverscape.py` | Create | Task 2 tests (stubbed `build_landform`, real `combine_zones`/`classify_hydroperiod`). |
| `tests/integration/test_riverscape_modes.py` | Create | Task 3 tests (mode table + two end-to-end `analyze_from_dea` paths). |

---

## Task 1: Landform pipeline (`riverscape/pipeline.py`)

**Files:**
- Create: `hydrofragments/riverscape/pipeline.py`
- Modify: `hydrofragments/riverscape/__init__.py`
- Test: `tests/riverscape/test_riverscape_pipeline.py`

**Interfaces:**
- Consumes (all pre-existing, signatures verified against source):
  - `hydrofragments.config.RiverscapeConfig` (frozen dataclass; fields used: `f_seed`, `f_chan_high`, `dem_product`, `dem_band`, `fc_product`, `bare_band`, `green_band`, `npv_band`, `waterbodies_source`, `corridor_min_m`, `corridor_max_m`, `alignment_quantile`, `profile_bin_m`, `profile_percentile`, `rem_k`, `rem_max_distance_m`, `trough_radius_m`, `trough_depth_m`, `h_chan_m`, `bare_threshold_floor_pct`, `bare_year_fraction`, `width_growth_factor`, `min_channel_confidence`, `include_line_fallback_in_channel`, `bridge_enabled`, `bridge_cost_weights`, `bridge_max_length_m`, `bridge_max_cost_per_m`, `bridge_rem_max_m`, `riparian_green_pct`, `narrow_width_px`, `envelope_quantile`, `envelope_min_bin_pixels`, `envelope_h_max_m`, `slope_max_deg`)
  - `hydrofragments.io.riverscape_sources.load_dem(geobox, *, product, band) -> xr.DataArray`
  - `hydrofragments.io.riverscape_sources.load_fc_percentiles(geobox, *, product, bands, years) -> dict[str, xr.DataArray]`
  - `hydrofragments.io.riverscape_sources.load_waterbodies(bounds, crs, *, source=None) -> gpd.GeoDataFrame`
  - `hydrofragments.io.riverscape_sources.RiverscapeSourceUnavailable`
  - `measure_corridor_widths(drainage, water_seed, *, transform, pixel_m, f_seed, corridor_min_m, corridor_max_m, alignment_quantile=0.95) -> CorridorResult(widths_m, p95_offset_m, degraded_reasons)`
  - `build_centreline(drainage, water_seed, corridor_widths_m, *, transform, pixel_m) -> CentrelineResult(skeleton, line_fallback_reaches, multithread_reaches)`
  - `build_rem(drainage, dem, *, transform, pixel_m, profile_bin_m, profile_percentile, corridor_widths_m, rem_k, rem_max_distance_m, trough_radius_m) -> TerrainResult(rem, trough_depth, slope_deg, degraded_reasons)`
  - `classify_waterbodies(polygons, centreline, *, transform, pixel_m) -> WaterbodyResult(polygons, riverine_mask, off_channel_mask)`
  - `build_evidence(frequency, bare_yearly, riverine_waterbody_mask, rem, trough_depth, domain, centreline, corridor_mask, *, f_chan_high, bare_threshold_floor_pct, bare_year_fraction, h_chan_m, trough_depth_m, min_calibration_pixels=200, line_fallback_mask=None) -> RiverscapeEvidence` (it calls `calibrate_bare_threshold` internally at `evidence.py:181`; the pipeline must **not** call it a second time)
  - `classify_channel(evidence, domain, centreline, water_seed, reach_labels, reach_keys, *, pixel_m, width_growth_factor, min_channel_confidence, include_line_fallback_in_channel=False) -> ChannelResult(channel, confidence, rule_id, seed_half_width_m, degraded_reasons)`
  - `find_gaps(channel, centreline, drainage, reach_labels, reach_keys, corridor_widths_m, *, transform, pixel_m) -> GapSearchResult(gaps, dangling_anchors, degraded_reasons)`
  - `bridge_gaps(gaps, observed_channel, cost_inputs, *, transform, crs, pixel_m, weights, bridge_max_length_m, bridge_max_cost_per_m, bridge_rem_max_m, trough_depth_m, riparian_green_pct, narrow_width_px) -> BridgeResult(bridged_mask, channel_bridges, unbridged, degraded_reasons)`
  - `BridgeCostInputs(rem, trough_depth, frequency, bare_fraction, green_pct, npv_pct, line_distance_m, corridor_width_m, corridor_mask, domain, off_channel_mask)`
  - `build_landform_layer(rem, slope_deg, domain, channel, reach_labels, upstr_darea, *, envelope_quantile, envelope_min_bin_pixels, envelope_h_max_m, slope_max_deg) -> LandformLayerResult(landform, envelope, degraded_reasons)`
- Produces (Task 2 and Task 3 depend on exactly these names):
  - `LandformResult` frozen dataclass with fields `landform, channel_source, channel_confidence, rule_id, rem, bridged_mask, channel_bridges, unbridged, envelope, provenance, degraded_reasons, grid=None`
  - `build_landform(domain, frequency, drainage, reach_labels, reach_keys, upstr_darea, *, geobox, transform, pixel_m, cfg, years, load_dem=..., load_fc_percentiles=..., load_waterbodies=...) -> LandformResult`
  - `validate_drainage_columns(drainage) -> None` (raises `ValueError`)
  - `with_grid(result, grid) -> LandformResult`
  - `_package_channel_source(observed, bridged) -> np.ndarray`
  - `_package_channel_confidence(confidence, channel) -> np.ndarray`
  - `_merge_degraded_reasons(*groups) -> tuple[str, ...]`
  - Constants `CHANNEL_SOURCE_NONE`, `CHANNEL_SOURCE_OBSERVED`, `CHANNEL_SOURCE_BRIDGED`, `CHANNEL_CONFIDENCE_NODATA`, `REQUIRED_DRAINAGE_COLUMNS`

### Test fixture arithmetic (read this before writing the test)

The happy-path test runs the **real** kernels on a 12 × 20 grid at 30 m. Every number below was derived by hand from the kernel source; if an assertion fails, the fixture is wrong, not the kernel.

- `TRANSFORM = Affine(30, 0, 0, 0, -30, 360)`. Pixel row `r` spans `y ∈ [360-30(r+1), 360-30r]`, centre `345-30r`. Column `c` centre `15+30c`.
- `domain[3:8, 1:19] = True` (90 pixels). `frequency = 1.0` everywhere, then `frequency[4:7, 1:19] = 60.0`.
- `f_seed = 0.05` → seed threshold 5 % → `water_seed` = rows 4–6, cols 1–18 (`60 ≥ 5`; `1.0 < 5` elsewhere). Also intersected with `domain`, which contains it.
- Reach = `LineString([(75, 195), (525, 195)])` → row 5, cols 2–17, fully inside the seed.
- Corridor: line pixels sit on the seed skeleton → `p95_offset_m = 0`. `distance_transform_edt(water_seed)` at row 5 is 2 px (nearest zero is row 3 or 7) → `max_half_width_m = 60`. `raw = 60`, clamped into `[60, 300]` → **width 60 m**, no degraded reason (not above the ceiling).
- Centreline: `line.buffer(60)` ∩ seed = rows 4–6, cols 2–17; `medial_axis` → row 5. `_has_loop` on a solid 3 × 16 rectangle → one background component → **not** multithread, **not** line-fallback.
- DEM `= 10.0`, with `dem[4:7, :] = 8.0`. `build_rem` buffers each 120 m bin by `max(width_m, pixel_m) = 60` → the polygon spans `y ∈ [135, 255]`, which `all_touched` rasterization hits on rows 3–7. The 10th-percentile DEM there is `8.0`, so the isotonic profile is flat at 8.0 and **`rem = 0` on rows 4–6, `2.0` elsewhere**. `rem_max_distance_m = 2000` covers the whole 600 × 360 m grid.
- `h_chan_m = 1.0` → `rem <= 1` is rows 4–6 only (this is why the fixture overrides the 2.0 default: at 2.0 the terrain bit would be true everywhere and stop discriminating).
- `trough_radius_m = 60` → `radius_px = 2`, box size 5. Row 5's window covers rows 3–7 → mean `(3·8 + 2·10)/5 = 8.8` → `trough = 0.8 ≥ 0.5`. Row 3's window covers rows 1–5 → mean 9.2 → `trough = -0.8`. So the trough bit is rows 4–6.
- `slope_deg`: `np.gradient(dem, 30)` gives at most `2/60` → `atan(0.0333) = 1.907° ≤ slope_max_deg = 2.0` everywhere.
- Waterbodies stub returns `gpd.GeoDataFrame(geometry=[], crs=CRS)` — the exact empty input `tests/riverscape/test_riverscape_waterbodies.py::test_empty_input_returns_typed_empty_result` already covers. Both waterbody masks are all-`False`.
- FC bare stack `(3, 12, 20)`: `40.0` on rows 4–6, `5.0` elsewhere. Calibration background is `~corridor_mask & isfinite(...)` ≈ 160 pixels < the `min_pixels=200` default, so calibration is **not** usable: threshold falls back to `bare_threshold_floor_pct = 30.0` and `"bare_threshold_fallback"` is recorded. This is expected on any grid this small — assert it is present rather than asserting the reason set is empty.
- `bare_fraction = 1.0` on rows 4–6 (`40 ≥ 30`), `0.0` elsewhere; `valid_years = 3 ≥ 2` → `bare_stable` = rows 4–6.
- Evidence confidence = `topology + water + geomorphic`. `support = domain & corridor_mask` = rows 3–7, cols 2–17, one component touching the centreline → `connected` over all of it. `water_high` (`60 ≥ 100·0.10`) = rows 4–6. So confidence is **3** on rows 4–6 cols 2–17 and **1** on rows 3 and 7 cols 2–17.
- `min_channel_confidence = 2` → `confidence_ok` = rows 4–6, cols 2–17. Seed half-width median on the centreline is 60 m; `width_growth_factor = 3` → eligible within 180 m (6 px) of row 5, i.e. every row. Rule 1 (`connected & (water_high | waterbody_riverine)`) fires → **observed channel = rows 4–6, cols 2–17 (48 px), `rule_id = RULE_CONNECTED_WATER = 1`**.
- `bridge_enabled = False` → empty `BridgeResult`, `bridged_mask` all `False`.
- Envelope: `upstr_darea` is a single value, so `log_a` is constant → one bin → `FloodplainEnvelope(a = 95th percentile of rem over the domain = 2.0, b = 0.0, n_bins = 1)` with `degraded_reasons = ("envelope_single_bin",)`.
- `classify_off_channel`: `h_fp = min(2.0·area^0, 15) = 2.0`; `rem ≤ 2.0` and `slope ≤ 2` hold across the domain → **off-channel riverine = domain ∖ channel**, so `landform` codes present are `{0, 1, 2}` and `LANDFORM_NON_RIVERINE` is absent.
- Final `degraded_reasons == ("bare_threshold_fallback", "envelope_single_bin")`.

---

- [ ] **Step 1: Write the failing pipeline tests**

Create `tests/riverscape/test_riverscape_pipeline.py`:

```python
"""``build_landform`` orchestration: stub loaders, real Phase 3-5 kernels.

Loaders are injected through ``build_landform``'s own keyword parameters
(spec 6a section 3.2) rather than monkeypatched, so no test here touches
STAC/WFS. The kernels themselves are NOT stubbed: this module's contract is
call order, argument forwarding, and packaging, and the only honest way to
prove the order is right is to let the real kernels consume each other's
output. Every expected value is derived by hand in the plan's "Test fixture
arithmetic" section.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from affine import Affine

gpd = pytest.importorskip("geopandas")
xr = pytest.importorskip("xarray")
pytest.importorskip("rasterio")
pytest.importorskip("skimage")
from shapely.geometry import LineString

from hydrofragments.config import RiverscapeConfig
from hydrofragments.io.riverscape_sources import RiverscapeSourceUnavailable
from hydrofragments.riverscape import pipeline as pipeline_module
from hydrofragments.riverscape.bridging import BridgeResult, Gap, GapSearchResult
from hydrofragments.riverscape.channel import RULE_CONNECTED_WATER
from hydrofragments.riverscape.codes import (
    LANDFORM_IN_CHANNEL,
    LANDFORM_NON_RIVERINE,
    LANDFORM_OFF_CHANNEL_RIVERINE,
    LANDFORM_OUTSIDE,
)
from hydrofragments.riverscape.pipeline import (
    CHANNEL_CONFIDENCE_NODATA,
    CHANNEL_SOURCE_BRIDGED,
    CHANNEL_SOURCE_NONE,
    CHANNEL_SOURCE_OBSERVED,
    _merge_degraded_reasons,
    _package_channel_confidence,
    _package_channel_source,
    build_landform,
    validate_drainage_columns,
    with_grid,
)

SHAPE = (12, 20)
PIXEL_M = 30.0
TRANSFORM = Affine(PIXEL_M, 0.0, 0.0, 0.0, -PIXEL_M, 360.0)
CRS = "EPSG:3577"
YEARS = (2019, 2021)
BARE_BAND = "bs_pc_50"
GREEN_BAND = "pv_pc_50"
NPV_BAND = "npv_pc_50"


def _cfg(**overrides) -> RiverscapeConfig:
    """Tiny-grid config. Every override is justified in the plan's fixture notes."""
    values = {
        "bare_band": BARE_BAND,
        "green_band": GREEN_BAND,
        "npv_band": NPV_BAND,
        "f_seed": 0.05,
        "f_chan_high": 0.10,
        "corridor_min_m": 60.0,
        "corridor_max_m": 300.0,
        "h_chan_m": 1.0,
        "trough_depth_m": 0.5,
        "trough_radius_m": 60.0,
        "profile_bin_m": 120.0,
        "rem_k": 4,
        "rem_max_distance_m": 2000.0,
        "envelope_min_bin_pixels": 1,
        "slope_max_deg": 2.0,
        "min_channel_confidence": 2,
        "bridge_enabled": False,
    }
    values.update(overrides)
    return RiverscapeConfig(**values)


def _domain_and_frequency() -> tuple[np.ndarray, np.ndarray]:
    domain = np.zeros(SHAPE, dtype=bool)
    domain[3:8, 1:19] = True
    frequency = np.full(SHAPE, 1.0, dtype=float)
    frequency[4:7, 1:19] = 60.0
    return domain, frequency


def _drainage() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame(
        {
            "HydroID": [1],
            "From_Node": [1],
            "To_Node": [2],
            "NextDownID": [-1],
            "UpstrDArea": [5000.0],
        },
        geometry=[LineString([(75.0, 195.0), (525.0, 195.0)])],
        crs=CRS,
    )


def _reach_context() -> tuple[np.ndarray, dict[int, str], dict[int, float]]:
    return np.ones(SHAPE, dtype=np.int32), {1: "1"}, {1: 5000.0}


def _dem() -> "xr.DataArray":
    dem = np.full(SHAPE, 10.0, dtype=np.float32)
    dem[4:7, :] = 8.0
    return xr.DataArray(dem, dims=("y", "x"))


def _fc() -> dict[str, "xr.DataArray"]:
    bare = np.full((3, *SHAPE), 5.0, dtype=float)
    bare[:, 4:7, :] = 40.0
    return {
        BARE_BAND: xr.DataArray(bare, dims=("time", "y", "x")),
        GREEN_BAND: xr.DataArray(np.full((3, *SHAPE), 10.0), dims=("time", "y", "x")),
        NPV_BAND: xr.DataArray(np.full((3, *SHAPE), 10.0), dims=("time", "y", "x")),
    }


def _waterbodies() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame(geometry=[], crs=CRS)


def _loaders(**overrides):
    loaders = {
        "load_dem": lambda geobox, *, product, band: _dem(),
        "load_fc_percentiles": lambda geobox, *, product, bands, years: _fc(),
        "load_waterbodies": lambda bounds, crs, *, source=None: _waterbodies(),
    }
    loaders.update(overrides)
    return loaders


def _build(cfg=None, *, domain=None, frequency=None, **loader_overrides):
    labels, reach_keys, upstr_darea = _reach_context()
    default_domain, default_frequency = _domain_and_frequency()
    return build_landform(
        default_domain if domain is None else domain,
        default_frequency if frequency is None else frequency,
        _drainage(),
        labels,
        reach_keys,
        upstr_darea,
        geobox=SimpleNamespace(),
        transform=TRANSFORM,
        pixel_m=PIXEL_M,
        cfg=cfg or _cfg(),
        years=YEARS,
        **_loaders(**loader_overrides),
    )


# --------------------------------------------------------------------------
# Happy path (spec section 7.1, bullet 1 and bullet 4)
# --------------------------------------------------------------------------


def test_happy_path_emits_in_channel_and_off_channel_riverine() -> None:
    result = _build()

    assert result.landform.dtype == np.uint8
    assert set(np.unique(result.landform).tolist()) == {
        LANDFORM_OUTSIDE,
        LANDFORM_IN_CHANNEL,
        LANDFORM_OFF_CHANNEL_RIVERINE,
    }
    assert not (result.landform == LANDFORM_NON_RIVERINE).any()
    assert result.landform[5, 10] == LANDFORM_IN_CHANNEL
    assert result.landform[3, 10] == LANDFORM_OFF_CHANNEL_RIVERINE
    assert result.landform[0, 0] == LANDFORM_OUTSIDE


def test_happy_path_channel_source_marks_observed_only() -> None:
    result = _build()
    observed = result.landform == LANDFORM_IN_CHANNEL

    assert result.channel_source.dtype == np.uint8
    assert (result.channel_source[observed] == CHANNEL_SOURCE_OBSERVED).all()
    assert (result.channel_source[~observed] == CHANNEL_SOURCE_NONE).all()
    assert not (result.channel_source == CHANNEL_SOURCE_BRIDGED).any()
    assert not result.bridged_mask.any()


def test_happy_path_confidence_is_255_outside_channel() -> None:
    result = _build()
    channel = result.landform == LANDFORM_IN_CHANNEL

    assert result.channel_confidence.dtype == np.uint8
    assert (result.channel_confidence[channel] == 3).all()
    assert (result.channel_confidence[~channel] == CHANNEL_CONFIDENCE_NODATA).all()


def test_happy_path_rule_id_is_zero_outside_channel() -> None:
    result = _build()
    channel = result.landform == LANDFORM_IN_CHANNEL

    assert (result.rule_id[channel] == RULE_CONNECTED_WATER).all()
    assert (result.rule_id[~channel] == 0).all()


def test_happy_path_rem_is_flat_channel_relative() -> None:
    result = _build()

    assert result.rem.dtype == np.float32
    assert result.rem[5, 10] == pytest.approx(0.0, abs=1e-5)
    assert result.rem[3, 10] == pytest.approx(2.0, abs=1e-5)


def test_happy_path_degraded_reasons_are_sorted_unique() -> None:
    result = _build()

    assert result.degraded_reasons == (
        "bare_threshold_fallback",
        "envelope_single_bin",
    )
    assert list(result.degraded_reasons) == sorted(set(result.degraded_reasons))


def test_happy_path_provenance_carries_stable_keys_for_phase_6b() -> None:
    result = _build()
    provenance = result.provenance

    assert provenance["channel_ruleset_version"] == "1.0.0"
    assert provenance["waterbody_ruleset_version"] == "1.0.0"
    assert provenance["riverine_ruleset_version"] == "1.0.0"
    assert provenance["dem_product"] == "ga_srtm_dem1sv1_0"
    assert provenance["fc_bands"] == (BARE_BAND, GREEN_BAND, NPV_BAND)
    assert provenance["years"] == YEARS
    assert provenance["bridge_enabled"] is False
    assert provenance["bare_threshold_pct"] == pytest.approx(30.0)
    assert provenance["envelope"]["n_bins"] == 1
    assert provenance["pixel_counts"]["domain"] == 90
    assert provenance["pixel_counts"]["observed_channel"] == 48
    assert provenance["pixel_counts"]["bridged_channel"] == 0
    assert provenance["pixel_counts"]["in_channel"] == 48
    assert provenance["pixel_counts"]["non_riverine"] == 0
    assert provenance["bridge_counts_by_cause"] == {}
    assert provenance["unbridged_counts_by_reason"] == {}
    assert provenance["reach_counts"] == {
        "total": 1,
        "line_fallback": 0,
        "multithread": 0,
    }


def test_envelope_and_bridge_frame_are_carried_through() -> None:
    result = _build()

    assert result.envelope.n_bins == 1
    assert result.envelope.a == pytest.approx(2.0)
    assert result.envelope.b == pytest.approx(0.0)
    assert result.unbridged == ()
    assert result.channel_bridges.empty
    assert "gap_cause" in result.channel_bridges.columns
    assert result.grid is None


# --------------------------------------------------------------------------
# Bridging (spec section 7.1 bullet 1, MUTANT 4)
# --------------------------------------------------------------------------


def test_bridged_pixels_are_channel_source_2_and_in_channel(monkeypatch) -> None:
    """MUTANT 4: painting bridged pixels as observed (1) must fail here."""
    bridged_rc = (1, 10)

    def fake_find_gaps(*args, **kwargs):
        return GapSearchResult(
            (
                Gap(
                    gap_id="gap-1",
                    reach_ids=("1",),
                    upstream_anchor=(5, 2),
                    downstream_anchor=(5, 17),
                    upstream_width_m=60.0,
                    downstream_width_m=60.0,
                    straight_length_m=450.0,
                ),
            ),
            (),
            (),
        )

    def fake_bridge_gaps(gaps, observed_channel, cost_inputs, **kwargs):
        painted = np.zeros(np.shape(observed_channel), dtype=bool)
        painted[bridged_rc] = True
        frame = gpd.GeoDataFrame(
            {"gap_id": ["gap-1"], "gap_cause": ["vegetated"]},
            geometry=[LineString([(75.0, 315.0), (525.0, 315.0)])],
            crs=CRS,
        )
        return BridgeResult(painted, frame, (), ())

    monkeypatch.setattr(pipeline_module, "find_gaps", fake_find_gaps)
    monkeypatch.setattr(pipeline_module, "bridge_gaps", fake_bridge_gaps)

    result = _build(_cfg(bridge_enabled=True))

    assert result.bridged_mask[bridged_rc]
    assert result.channel_source[bridged_rc] == CHANNEL_SOURCE_BRIDGED
    assert result.channel_source[5, 10] == CHANNEL_SOURCE_OBSERVED
    assert result.landform[bridged_rc] == LANDFORM_IN_CHANNEL
    assert result.provenance["pixel_counts"]["bridged_channel"] == 1
    assert result.provenance["bridge_counts_by_cause"] == {"vegetated": 1}


def test_bridged_mask_never_overlaps_the_observed_domain(monkeypatch) -> None:
    """The LandformResult invariant classify_hydroperiod depends on."""

    def fake_find_gaps(*args, **kwargs):
        return GapSearchResult((), (), ())

    def fake_bridge_gaps(gaps, observed_channel, cost_inputs, **kwargs):
        painted = np.ones(np.shape(observed_channel), dtype=bool)
        return BridgeResult(painted, _waterbodies(), (), ())

    monkeypatch.setattr(pipeline_module, "find_gaps", fake_find_gaps)
    monkeypatch.setattr(pipeline_module, "bridge_gaps", fake_bridge_gaps)

    domain, _frequency = _domain_and_frequency()
    result = _build(_cfg(bridge_enabled=True))

    assert not (result.bridged_mask & domain).any()
    assert not (result.bridged_mask & (result.channel_source == CHANNEL_SOURCE_OBSERVED)).any()


def test_bridge_disabled_never_calls_the_bridging_kernels(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("bridging must not run when bridge_enabled is False")

    monkeypatch.setattr(pipeline_module, "find_gaps", explode)
    monkeypatch.setattr(pipeline_module, "bridge_gaps", explode)

    result = _build(_cfg(bridge_enabled=False))

    assert not result.bridged_mask.any()
    assert not (result.channel_source == CHANNEL_SOURCE_BRIDGED).any()


# --------------------------------------------------------------------------
# Loader failures (spec section 7.1 bullet 2, MUTANT 5)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loader",
    ["load_dem", "load_fc_percentiles", "load_waterbodies"],
)
def test_loader_failure_propagates_unchanged(loader: str) -> None:
    """MUTANT 5: catching RiverscapeSourceUnavailable inside build_landform
    must fail here -- mode behaviour is the caller's decision, not this
    function's (spec section 4, closing paragraph)."""

    def raising(*args, **kwargs):
        raise RiverscapeSourceUnavailable(f"{loader} unavailable")

    with pytest.raises(RiverscapeSourceUnavailable, match=f"{loader} unavailable"):
        _build(**{loader: raising})


def test_missing_fc_band_is_a_source_failure() -> None:
    def partial_fc(geobox, *, product, bands, years):
        loaded = _fc()
        del loaded[GREEN_BAND]
        return loaded

    with pytest.raises(RiverscapeSourceUnavailable, match=GREEN_BAND):
        _build(load_fc_percentiles=partial_fc)


# --------------------------------------------------------------------------
# Input validation (spec section 7.1 bullet 3, section 3.2)
# --------------------------------------------------------------------------


def test_shape_mismatch_raises_value_error() -> None:
    domain, _frequency = _domain_and_frequency()
    with pytest.raises(ValueError, match="domain and frequency must share shape"):
        _build(frequency=np.ones((4, 4), dtype=float))

    with pytest.raises(ValueError, match="domain must be a boolean array"):
        _build(domain=np.ones(SHAPE, dtype=np.uint8))

    with pytest.raises(ValueError, match="domain must be a 2-D array"):
        _build(domain=np.ones((2, *SHAPE), dtype=bool))


def test_non_finite_frequency_inside_domain_raises() -> None:
    domain, frequency = _domain_and_frequency()
    frequency[5, 10] = np.nan
    with pytest.raises(ValueError, match="frequency must be finite inside the domain"):
        _build(frequency=frequency)


def test_dem_off_grid_raises_value_error() -> None:
    def wrong_shape_dem(geobox, *, product, band):
        return xr.DataArray(np.zeros((4, 4), dtype=np.float32), dims=("y", "x"))

    with pytest.raises(ValueError, match="does not match the zoning grid"):
        _build(load_dem=wrong_shape_dem)


@pytest.mark.parametrize(
    "column", ["HydroID", "NextDownID", "UpstrDArea", "From_Node", "To_Node"]
)
def test_missing_drainage_column_is_a_programming_error(column: str) -> None:
    """Spec section 3.2: missing columns raise ValueError, never a mode fallback."""
    drainage = _drainage().drop(columns=[column])
    with pytest.raises(ValueError, match=column):
        validate_drainage_columns(drainage)


def test_validate_drainage_columns_accepts_the_full_schema() -> None:
    assert validate_drainage_columns(_drainage()) is None


def test_empty_drainage_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one line feature"):
        validate_drainage_columns(_drainage().iloc[0:0])


# --------------------------------------------------------------------------
# Packaging helpers, tested directly
# --------------------------------------------------------------------------


def test_package_channel_source_prefers_observed_over_bridged() -> None:
    observed = np.array([[True, False, False]], dtype=bool)
    bridged = np.array([[True, True, False]], dtype=bool)

    source = _package_channel_source(observed, bridged)

    assert source.dtype == np.uint8
    assert source.tolist() == [
        [CHANNEL_SOURCE_OBSERVED, CHANNEL_SOURCE_BRIDGED, CHANNEL_SOURCE_NONE]
    ]


def test_package_channel_source_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="must share shape"):
        _package_channel_source(np.zeros((2, 2), bool), np.zeros((3, 3), bool))


def test_package_channel_confidence_masks_off_channel_to_255() -> None:
    confidence = np.array([[3, 2, 1]], dtype=np.uint8)
    channel = np.array([[True, True, False]], dtype=bool)

    packaged = _package_channel_confidence(confidence, channel)

    assert packaged.tolist() == [[3, 2, CHANNEL_CONFIDENCE_NODATA]]


def test_merge_degraded_reasons_sorts_and_deduplicates() -> None:
    merged = _merge_degraded_reasons(
        ("envelope_single_bin",),
        ("bare_threshold_fallback", "envelope_single_bin"),
        (),
    )
    assert merged == ("bare_threshold_fallback", "envelope_single_bin")


def test_with_grid_attaches_the_phase_6b_export_grid() -> None:
    result = _build()
    grid = SimpleNamespace(height=SHAPE[0], width=SHAPE[1])

    attached = with_grid(result, grid)

    assert result.grid is None
    assert attached.grid is grid
    assert attached.landform is result.landform
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/riverscape/test_riverscape_pipeline.py -v`

Expected: FAIL at collection — `ModuleNotFoundError: No module named 'hydrofragments.riverscape.pipeline'`.

- [ ] **Step 3: Write `hydrofragments/riverscape/pipeline.py`**

```python
"""Riverscape landform orchestrator: source loaders plus the Phase 3-5 kernels.

``build_landform`` is the single entry point that turns a DEA-derived
observed-wet domain plus a drainage network into a :class:`LandformResult`.
It owns two things and nothing else: WHICH sources get loaded (DEM,
Fractional Cover, DEA Waterbodies -- all through injectable callables, so
tests never touch the network) and IN WHAT ORDER the Phase 3-5 kernels run.
Every scientific rule stays owned by the kernel module it already lives in
(``corridor``, ``centreline``, ``terrain``, ``waterbodies``, ``evidence``,
``channel``, ``bridging``, ``riverine``); this module adds no threshold and
no classification of its own.

Import boundary (spec 6a section 1.3): this module must never import
``hydrofragments.spatial`` -- enforced by
``tests/riverscape/test_riverscape_import_boundary.py``. The consequence is
that the reach-label raster, its ``label -> HydroID`` key map, and its
``label -> UpstrDArea`` map are all built by the caller (``spatial`` or
``workflow``, which own ``connectivity_context._build_reach_label_raster``)
and passed in as plain arrays/mappings.

``RiverscapeSourceUnavailable`` is never caught here. ``riverscape.mode``
behaviour -- degrade to occurrence zoning, or propagate -- is the caller's
decision (spec 6a sections 4 and 6), and swallowing the exception here
would make ``mode="required"`` unimplementable.
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage

from hydrofragments.config import RiverscapeConfig
from hydrofragments.io.riverscape_sources import (
    RiverscapeSourceUnavailable,
    load_dem,
    load_fc_percentiles,
    load_waterbodies,
)
from hydrofragments.riverscape.bridging import (
    BridgeCostInputs,
    BridgeResult,
    _empty_bridge_frame,
    bridge_gaps,
    find_gaps,
)
from hydrofragments.riverscape.centreline import CentrelineResult, build_centreline
from hydrofragments.riverscape.channel import RULESET_VERSION as CHANNEL_RULESET_VERSION
from hydrofragments.riverscape.channel import classify_channel
from hydrofragments.riverscape.codes import (
    LANDFORM_IN_CHANNEL,
    LANDFORM_NON_RIVERINE,
    LANDFORM_OFF_CHANNEL_RIVERINE,
)
from hydrofragments.riverscape.corridor import measure_corridor_widths
from hydrofragments.riverscape.evidence import build_evidence
from hydrofragments.riverscape.riverine import (
    RIVERINE_RULESET_VERSION,
    FloodplainEnvelope,
    build_landform_layer,
)
from hydrofragments.riverscape.terrain import build_rem
from hydrofragments.riverscape.waterbodies import (
    WATERBODY_RULESET_VERSION,
    classify_waterbodies,
)

CHANNEL_SOURCE_NONE = 0
CHANNEL_SOURCE_OBSERVED = 1
CHANNEL_SOURCE_BRIDGED = 2

#: ``channel_confidence`` sentinel for "not a channel pixel" (spec 3.1).
CHANNEL_CONFIDENCE_NODATA = 255

#: Columns every Phase 3-5 kernel between them requires. ``HydroID``/
#: ``NextDownID`` come from corridor/centreline/bridging, ``From_Node``/
#: ``To_Node`` from ``terrain._REQUIRED_COLUMNS``, and ``UpstrDArea`` is the
#: floodplain envelope's scaling variable (the caller turns it into the
#: label-keyed ``upstr_darea`` mapping).
REQUIRED_DRAINAGE_COLUMNS = (
    "HydroID",
    "NextDownID",
    "UpstrDArea",
    "From_Node",
    "To_Node",
)


@dataclass(frozen=True)
class LandformResult:
    """Everything Phase 6a's zoning adapter and Phase 6b's exports consume.

    ``grid`` is declared ``Any | None`` rather than ``SpatialGrid | None``
    on purpose: ``SpatialGrid`` lives in ``hydrofragments.output.spatial``,
    which is an output-layer type this package must not reach for (see the
    module docstring's import-boundary note). It is last in the field order
    only because it is the one field with a default; spec section 3.1 lists
    it before ``provenance``.
    """

    landform: np.ndarray
    channel_source: np.ndarray
    channel_confidence: np.ndarray
    rule_id: np.ndarray
    rem: np.ndarray
    bridged_mask: np.ndarray
    channel_bridges: Any
    unbridged: tuple
    envelope: FloodplainEnvelope
    provenance: Mapping[str, Any]
    degraded_reasons: tuple[str, ...]
    grid: Any | None = None


def with_grid(result: LandformResult, grid: Any) -> LandformResult:
    """Return ``result`` with ``grid`` attached.

    The Phase 6b seam: 6b hands a real
    ``hydrofragments.output.spatial.SpatialGrid`` back to a
    :class:`LandformResult` before writing rasters. Phase 6a's
    ``zones_from_riverscape`` returns a ``ZoneResult`` (which carries its own
    grid), so it does not call this itself.
    """
    return replace(result, grid=grid)


def validate_drainage_columns(drainage: Any) -> None:
    """Raise ``ValueError`` when ``drainage`` cannot drive the pipeline.

    A missing column is a programming error in the caller's drainage
    preparation, not a data-availability problem, so it must never be
    mistaken for a ``riverscape.mode`` fallback (spec section 3.2).
    """
    if drainage is None:
        raise ValueError("drainage must contain at least one line feature")
    if getattr(drainage, "empty", True):
        raise ValueError("drainage must contain at least one line feature")
    missing = [name for name in REQUIRED_DRAINAGE_COLUMNS if name not in drainage.columns]
    if missing:
        raise ValueError(f"drainage missing required columns: {missing}")
    if getattr(drainage, "geometry", None) is None:
        raise ValueError("drainage must carry line geometry")


def _package_channel_source(observed: Any, bridged: Any) -> np.ndarray:
    """Per-pixel channel provenance: 1 observed, 2 bridged-only, 0 elsewhere.

    Observed is painted last so a pixel that is both (which
    ``bridge_gaps`` already excludes, but a caller-injected bridging stub
    might not) reads as observed -- a real observation always outranks a
    reconstruction.
    """
    observed_mask = np.asarray(observed, dtype=bool)
    bridged_mask = np.asarray(bridged, dtype=bool)
    if bridged_mask.shape != observed_mask.shape:
        raise ValueError("observed and bridged masks must share shape")
    source = np.full(observed_mask.shape, CHANNEL_SOURCE_NONE, dtype=np.uint8)
    source[bridged_mask & ~observed_mask] = CHANNEL_SOURCE_BRIDGED
    source[observed_mask] = CHANNEL_SOURCE_OBSERVED
    return source


def _package_channel_confidence(confidence: Any, channel: Any) -> np.ndarray:
    """Evidence confidence on channel pixels, ``255`` everywhere else."""
    values = np.asarray(confidence, dtype=np.uint8)
    mask = np.asarray(channel, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("confidence and channel must share shape")
    return np.where(mask, values, CHANNEL_CONFIDENCE_NODATA).astype(np.uint8)


def _merge_degraded_reasons(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Stable sorted unique union of every kernel's degraded reasons."""
    return tuple(sorted({reason for group in groups for reason in group}))


def _grid_bounds(transform: Affine, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return ``(minx, miny, maxx, maxy)`` for the zoning grid.

    Derived from ``transform``/``shape`` rather than from ``geobox`` so the
    waterbody loader's bbox never depends on the geobox object's API -- the
    geobox is only ever forwarded to the raster loaders.
    """
    height, width = shape
    west, north = transform * (0.0, 0.0)
    east, south = transform * (float(width), float(height))
    return (min(west, east), min(north, south), max(west, east), max(north, south))


def _loaded_array(data: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    values = np.asarray(getattr(data, "values", data), dtype=np.float32)
    if values.shape != shape:
        raise ValueError(
            f"{name} shape {values.shape} does not match the zoning grid {shape}"
        )
    return values


def _band_stack(bands: Mapping[str, Any], name: str, shape: tuple[int, int]) -> np.ndarray:
    """Return one Fractional Cover band as a ``(time, y, x)`` float array."""
    if name not in bands:
        raise RiverscapeSourceUnavailable(
            f"fractional cover load returned no {name!r} band"
        )
    band = bands[name]
    values = np.asarray(getattr(band, "values", band), dtype=float)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != shape:
        raise ValueError(
            f"fractional cover band {name!r} shape {values.shape} does not match "
            f"the zoning grid {shape}"
        )
    return values


def _band_median(bands: Mapping[str, Any], name: str, shape: tuple[int, int]) -> np.ndarray:
    """Across-year median of one FC band, as a 2-D float32 array.

    ``load_fc_percentiles`` already converts nodata sentinels to NaN, so an
    all-missing pixel is legitimately all-NaN; the warning that
    ``np.nanmedian`` emits for it carries no information the caller can act
    on (the resulting NaN is handled downstream by ``_unit_interval`` in
    ``bridging.build_cost_surface``).
    """
    stack = _band_stack(bands, name, shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=0).astype(np.float32)


def _rasterize(geometry: Any, *, transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    return rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)


def _corridor_rasters(
    drainage: Any,
    corridor_widths_m: Mapping[str, float],
    *,
    transform: Affine,
    shape: tuple[int, int],
    pixel_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize per-reach corridors into the three grids later steps need.

    Returns ``(corridor_mask, corridor_width_m, line_distance_m)``.
    ``corridor_width_m`` takes the widest claimant where buffers overlap
    near junctions -- the corridor is a search-space bound, so the more
    permissive value is the correct one to keep. ``line_distance_m`` is the
    Euclidean distance to the nearest rasterized AHGF line pixel, the
    ``BridgeCostInputs`` term that keeps least-cost routes near the mapped
    channel. A grid with no line pixels at all gets ``inf``, which
    ``bridging._unit_interval`` clips to the maximum penalty.
    """
    width = np.zeros(shape, dtype=np.float32)
    lines = np.zeros(shape, dtype=bool)
    for hydro_id, geometry in zip(drainage["HydroID"], drainage.geometry):
        if geometry is None or geometry.is_empty:
            continue
        key = str(hydro_id)
        if key not in corridor_widths_m:
            raise ValueError(f"corridor_widths_m missing entry for reach {key!r}")
        width_m = float(corridor_widths_m[key])
        buffered = _rasterize(geometry.buffer(width_m), transform=transform, shape=shape)
        width[buffered] = np.maximum(width[buffered], width_m)
        lines |= _rasterize(geometry, transform=transform, shape=shape)
    if lines.any():
        line_distance_m = (
            ndimage.distance_transform_edt(~lines) * pixel_m
        ).astype(np.float32)
    else:
        line_distance_m = np.full(shape, np.inf, dtype=np.float32)
    return width > 0, width, line_distance_m


def _line_fallback_raster(
    drainage: Any,
    line_fallback_reaches: tuple[str, ...],
    *,
    transform: Affine,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize the AHGF lines of reaches with no water inside their corridor.

    A line-fallback reach's line "remains a topology/profile guide only and
    never becomes channel" (``centreline.build_centreline``), so only the
    line itself is painted -- not its corridor buffer. The evidence layer
    folds this into its topology family, and it only reaches the channel
    when ``include_line_fallback_in_channel`` is on.
    """
    fallback = np.zeros(shape, dtype=bool)
    if not line_fallback_reaches:
        return fallback
    wanted = set(line_fallback_reaches)
    for hydro_id, geometry in zip(drainage["HydroID"], drainage.geometry):
        if str(hydro_id) not in wanted or geometry is None or geometry.is_empty:
            continue
        fallback |= _rasterize(geometry, transform=transform, shape=shape)
    return fallback


def _run_bridging(
    cfg: RiverscapeConfig,
    *,
    observed: np.ndarray,
    centreline: np.ndarray,
    drainage: Any,
    reach_labels: np.ndarray,
    reach_keys: Mapping[int, str],
    corridor_widths_m: Mapping[str, float],
    corridor_mask: np.ndarray,
    corridor_width_m: np.ndarray,
    line_distance_m: np.ndarray,
    frequency: np.ndarray,
    domain: np.ndarray,
    rem: np.ndarray,
    trough_depth: np.ndarray,
    bare_fraction: np.ndarray,
    green_pct: np.ndarray,
    npv_pct: np.ndarray,
    off_channel_mask: np.ndarray,
    transform: Affine,
    pixel_m: float,
) -> BridgeResult:
    """Discover and bridge channel gaps, or return a typed empty result.

    ``_empty_bridge_frame`` is reused from ``bridging`` rather than
    reconstructed here so the disabled path's ``channel_bridges`` schema can
    never drift from the enabled path's.
    """
    if not cfg.bridge_enabled:
        return BridgeResult(
            bridged_mask=np.zeros(observed.shape, dtype=bool),
            channel_bridges=_empty_bridge_frame(drainage.crs),
            unbridged=(),
            degraded_reasons=(),
        )
    gaps = find_gaps(
        observed,
        centreline,
        drainage,
        reach_labels,
        reach_keys,
        corridor_widths_m,
        transform=transform,
        pixel_m=pixel_m,
    )
    cost_inputs = BridgeCostInputs(
        rem=rem,
        trough_depth=trough_depth,
        frequency=frequency,
        bare_fraction=bare_fraction,
        green_pct=green_pct,
        npv_pct=npv_pct,
        line_distance_m=line_distance_m,
        corridor_width_m=corridor_width_m,
        corridor_mask=corridor_mask,
        domain=domain,
        off_channel_mask=off_channel_mask,
    )
    result = bridge_gaps(
        gaps.gaps,
        observed,
        cost_inputs,
        transform=transform,
        crs=drainage.crs,
        pixel_m=pixel_m,
        weights=cfg.bridge_cost_weights,
        bridge_max_length_m=cfg.bridge_max_length_m,
        bridge_max_cost_per_m=cfg.bridge_max_cost_per_m,
        bridge_rem_max_m=cfg.bridge_rem_max_m,
        trough_depth_m=cfg.trough_depth_m,
        riparian_green_pct=cfg.riparian_green_pct,
        narrow_width_px=cfg.narrow_width_px,
    )
    return replace(
        result,
        degraded_reasons=tuple(result.degraded_reasons) + tuple(gaps.degraded_reasons),
    )


def _build_provenance(
    *,
    cfg: RiverscapeConfig,
    years: tuple[int, int],
    landform: np.ndarray,
    domain: np.ndarray,
    water_seed: np.ndarray,
    observed: np.ndarray,
    bridged: np.ndarray,
    bridges: Any,
    unbridged: tuple,
    bare_threshold_pct: float,
    envelope: FloodplainEnvelope,
    centreline: CentrelineResult,
    reach_total: int,
) -> dict[str, Any]:
    """Stable provenance keys Phase 6b copies straight into the manifest."""
    causes: dict[str, int] = {}
    if len(bridges):
        causes = {
            str(cause): int(count)
            for cause, count in sorted(bridges["gap_cause"].value_counts().items())
        }
    return {
        "channel_ruleset_version": CHANNEL_RULESET_VERSION,
        "waterbody_ruleset_version": WATERBODY_RULESET_VERSION,
        "riverine_ruleset_version": RIVERINE_RULESET_VERSION,
        "dem_product": cfg.dem_product,
        "dem_band": cfg.dem_band,
        "fc_product": cfg.fc_product,
        "fc_bands": (cfg.bare_band, cfg.green_band, cfg.npv_band),
        "waterbodies_source": cfg.waterbodies_source,
        "years": (int(years[0]), int(years[1])),
        "bridge_enabled": bool(cfg.bridge_enabled),
        "bare_threshold_pct": float(bare_threshold_pct),
        "envelope": {
            "a": float(envelope.a),
            "b": float(envelope.b),
            "n_bins": int(envelope.n_bins),
        },
        "pixel_counts": {
            "domain": int(np.count_nonzero(domain)),
            "water_seed": int(np.count_nonzero(water_seed)),
            "observed_channel": int(np.count_nonzero(observed)),
            "bridged_channel": int(np.count_nonzero(bridged)),
            "in_channel": int(np.count_nonzero(landform == LANDFORM_IN_CHANNEL)),
            "off_channel_riverine": int(
                np.count_nonzero(landform == LANDFORM_OFF_CHANNEL_RIVERINE)
            ),
            "non_riverine": int(np.count_nonzero(landform == LANDFORM_NON_RIVERINE)),
        },
        "bridge_counts_by_cause": causes,
        "unbridged_counts_by_reason": {
            str(reason): int(count)
            for reason, count in sorted(Counter(item.reason for item in unbridged).items())
        },
        "reach_counts": {
            "total": int(reach_total),
            "line_fallback": len(centreline.line_fallback_reaches),
            "multithread": len(centreline.multithread_reaches),
        },
    }


def build_landform(
    domain: np.ndarray,
    frequency: np.ndarray,
    drainage: "gpd.GeoDataFrame",
    reach_labels: np.ndarray,
    reach_keys: Mapping[int, str],
    upstr_darea: Mapping[int, float],
    *,
    geobox: Any,
    transform: Affine,
    pixel_m: float,
    cfg: RiverscapeConfig,
    years: tuple[int, int],
    load_dem: Callable[..., "xr.DataArray"] = load_dem,
    load_fc_percentiles: Callable[..., dict] = load_fc_percentiles,
    load_waterbodies: Callable[..., "gpd.GeoDataFrame"] = load_waterbodies,
) -> LandformResult:
    """Load sources and run corridor -> ... -> riverine, in that order.

    ``domain`` is the observed-wet analysis domain (``riverscape.domain.
    wet_domain``); ``frequency`` is percent-scale wet-observation frequency
    on the same grid. ``reach_labels``/``reach_keys``/``upstr_darea`` come
    from the caller (see the module docstring's import-boundary note);
    ``reach_keys`` and ``upstr_darea`` are both keyed by the integer LABEL
    value found in ``reach_labels``, not by ``HydroID``.

    Raises :class:`~hydrofragments.io.riverscape_sources.RiverscapeSourceUnavailable`
    on loader failure -- propagated, never caught. Raises ``ValueError`` for
    shape/dtype/schema problems, which are programming errors rather than
    source-availability problems. Never imports ``hydrofragments.spatial``.
    """
    domain_mask = np.asarray(domain)
    frequency_values = np.asarray(frequency, dtype=float)
    labels = np.asarray(reach_labels)

    if domain_mask.dtype != np.bool_:
        raise ValueError("domain must be a boolean array")
    if domain_mask.ndim != 2:
        raise ValueError("domain must be a 2-D array")
    if frequency_values.shape != domain_mask.shape:
        raise ValueError("domain and frequency must share shape")
    if labels.shape != domain_mask.shape:
        raise ValueError("domain and reach_labels must share shape")
    if not np.all(np.isfinite(frequency_values[domain_mask])):
        raise ValueError("frequency must be finite inside the domain")
    if not np.isfinite(pixel_m) or pixel_m <= 0:
        raise ValueError("pixel_m must be finite and positive")
    validate_drainage_columns(drainage)

    shape = domain_mask.shape

    # Step 2: water seed -- the same percent-scale convention Phase 4's
    # evidence layer uses; cfg.f_seed is a fraction, so convert once here.
    water_seed = domain_mask & (frequency_values >= 100.0 * cfg.f_seed)

    # Step 3: load sources. No try/except -- see the module docstring.
    dem = _loaded_array(
        load_dem(geobox, product=cfg.dem_product, band=cfg.dem_band), shape, "dem"
    )
    fc_bands = load_fc_percentiles(
        geobox,
        product=cfg.fc_product,
        bands=(cfg.bare_band, cfg.green_band, cfg.npv_band),
        years=years,
    )
    bare_yearly = _band_stack(fc_bands, cfg.bare_band, shape)
    green_pct = _band_median(fc_bands, cfg.green_band, shape)
    npv_pct = _band_median(fc_bands, cfg.npv_band, shape)
    waterbody_polygons = load_waterbodies(
        _grid_bounds(transform, shape),
        drainage.crs,
        source=cfg.waterbodies_source,
    )

    # Step 4: corridor widths bound every later search space.
    corridor = measure_corridor_widths(
        drainage,
        water_seed,
        transform=transform,
        pixel_m=pixel_m,
        f_seed=cfg.f_seed,
        corridor_min_m=cfg.corridor_min_m,
        corridor_max_m=cfg.corridor_max_m,
        alignment_quantile=cfg.alignment_quantile,
    )

    # Step 5: centreline conflation runs on the WATER SEED, not the full
    # domain -- the skeleton must follow observed water, not the whole
    # observed-wet extent (spec section 4 step 5).
    centreline = build_centreline(
        drainage,
        water_seed,
        corridor.widths_m,
        transform=transform,
        pixel_m=pixel_m,
    )

    # Step 6: REM, local trough depth, slope.
    terrain = build_rem(
        drainage,
        dem,
        transform=transform,
        pixel_m=pixel_m,
        profile_bin_m=cfg.profile_bin_m,
        profile_percentile=cfg.profile_percentile,
        corridor_widths_m=corridor.widths_m,
        rem_k=cfg.rem_k,
        rem_max_distance_m=cfg.rem_max_distance_m,
        trough_radius_m=cfg.trough_radius_m,
    )

    corridor_mask, corridor_width_m, line_distance_m = _corridor_rasters(
        drainage,
        corridor.widths_m,
        transform=transform,
        shape=shape,
        pixel_m=pixel_m,
    )

    # Step 7: waterbody riverine/off-channel roles, judged against the
    # conflated centreline.
    waterbodies = classify_waterbodies(
        waterbody_polygons,
        centreline.skeleton,
        transform=transform,
        pixel_m=pixel_m,
    )

    # Step 8: per-pixel evidence. build_evidence owns the
    # calibrate_bare_threshold call internally; do not calibrate twice.
    evidence = build_evidence(
        frequency_values,
        bare_yearly,
        waterbodies.riverine_mask,
        terrain.rem,
        terrain.trough_depth,
        domain_mask,
        centreline.skeleton,
        corridor_mask,
        f_chan_high=cfg.f_chan_high,
        bare_threshold_floor_pct=cfg.bare_threshold_floor_pct,
        bare_year_fraction=cfg.bare_year_fraction,
        h_chan_m=cfg.h_chan_m,
        trough_depth_m=cfg.trough_depth_m,
        line_fallback_mask=_line_fallback_raster(
            drainage,
            centreline.line_fallback_reaches,
            transform=transform,
            shape=shape,
        ),
    )

    # Step 9: ordered observed-channel rules.
    channel = classify_channel(
        evidence,
        domain_mask,
        centreline.skeleton,
        water_seed,
        labels,
        reach_keys,
        pixel_m=pixel_m,
        width_growth_factor=cfg.width_growth_factor,
        min_channel_confidence=cfg.min_channel_confidence,
        include_line_fallback_in_channel=cfg.include_line_fallback_in_channel,
    )
    observed = np.asarray(channel.channel, dtype=bool)

    # Step 10: gap bridging (or a typed empty result when disabled).
    bridge = _run_bridging(
        cfg,
        observed=observed,
        centreline=centreline.skeleton,
        drainage=drainage,
        reach_labels=labels,
        reach_keys=reach_keys,
        corridor_widths_m=corridor.widths_m,
        corridor_mask=corridor_mask,
        corridor_width_m=corridor_width_m,
        line_distance_m=line_distance_m,
        frequency=frequency_values,
        domain=domain_mask,
        rem=terrain.rem,
        trough_depth=terrain.trough_depth,
        bare_fraction=evidence.bare_fraction,
        green_pct=green_pct,
        npv_pct=npv_pct,
        off_channel_mask=waterbodies.off_channel_mask,
        transform=transform,
        pixel_m=pixel_m,
    )

    # A bridged pixel means "channel we could not observe": bridge_gaps
    # already excludes the observed domain and the observed channel, and
    # re-applying that intersection here makes the invariant hold for an
    # injected bridging stub too. classify_hydroperiod depends on it --
    # it rejects an unobserved_mask that overlaps the observed domain.
    bridged = np.asarray(bridge.bridged_mask, dtype=bool) & ~domain_mask & ~observed
    channel_all = observed | bridged

    # Step 11: riverine vs non-riverine. assemble_landform only assigns
    # codes inside its domain argument, so bridged pixels must be added to
    # the domain here or they would stay landform 0 while hydroperiod says
    # 4 -- which combine_zones rejects as a zoned-extent disagreement.
    riverine = build_landform_layer(
        terrain.rem,
        terrain.slope_deg,
        domain_mask | bridged,
        channel_all,
        labels,
        upstr_darea,
        envelope_quantile=cfg.envelope_quantile,
        envelope_min_bin_pixels=cfg.envelope_min_bin_pixels,
        envelope_h_max_m=cfg.envelope_h_max_m,
        slope_max_deg=cfg.slope_max_deg,
    )

    # Step 12: package.
    return LandformResult(
        landform=riverine.landform,
        channel_source=_package_channel_source(observed, bridged),
        channel_confidence=_package_channel_confidence(channel.confidence, channel_all),
        rule_id=np.where(
            channel_all, np.asarray(channel.rule_id, dtype=np.uint8), 0
        ).astype(np.uint8),
        rem=np.asarray(terrain.rem, dtype=np.float32),
        bridged_mask=bridged,
        channel_bridges=bridge.channel_bridges,
        unbridged=bridge.unbridged,
        envelope=riverine.envelope,
        provenance=_build_provenance(
            cfg=cfg,
            years=years,
            landform=riverine.landform,
            domain=domain_mask,
            water_seed=water_seed,
            observed=observed,
            bridged=bridged,
            bridges=bridge.channel_bridges,
            unbridged=bridge.unbridged,
            bare_threshold_pct=evidence.calibration.threshold_pct,
            envelope=riverine.envelope,
            centreline=centreline,
            reach_total=len(drainage),
        ),
        degraded_reasons=_merge_degraded_reasons(
            corridor.degraded_reasons,
            tuple(
                f"reach_{key}_line_fallback"
                for key in centreline.line_fallback_reaches
            ),
            terrain.degraded_reasons,
            evidence.degraded_reasons,
            channel.degraded_reasons,
            bridge.degraded_reasons,
            riverine.envelope.degraded_reasons,
        ),
    )


__all__ = [
    "CHANNEL_CONFIDENCE_NODATA",
    "CHANNEL_SOURCE_BRIDGED",
    "CHANNEL_SOURCE_NONE",
    "CHANNEL_SOURCE_OBSERVED",
    "REQUIRED_DRAINAGE_COLUMNS",
    "LandformResult",
    "build_landform",
    "validate_drainage_columns",
    "with_grid",
]
```

- [ ] **Step 4: Re-export the Phase 6a surface**

Replace `hydrofragments/riverscape/__init__.py` with:

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
from hydrofragments.riverscape.pipeline import (
    CHANNEL_CONFIDENCE_NODATA,
    CHANNEL_SOURCE_BRIDGED,
    CHANNEL_SOURCE_NONE,
    CHANNEL_SOURCE_OBSERVED,
    LandformResult,
    build_landform,
    validate_drainage_columns,
    with_grid,
)

__all__ = [
    "CHANNEL_CONFIDENCE_NODATA",
    "CHANNEL_SOURCE_BRIDGED",
    "CHANNEL_SOURCE_NONE",
    "CHANNEL_SOURCE_OBSERVED",
    "LANDFORM_CODES",
    "LANDFORM_IN_CHANNEL",
    "LANDFORM_NAMES",
    "LANDFORM_NON_RIVERINE",
    "LANDFORM_OFF_CHANNEL_RIVERINE",
    "LANDFORM_OUTSIDE",
    "LandformResult",
    "build_landform",
    "validate_drainage_columns",
    "wet_domain",
    "with_grid",
]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/riverscape/test_riverscape_pipeline.py -v`

Expected: PASS, all tests.

- [ ] **Step 6: Re-run the import-boundary guard**

Run: `python -m pytest tests/riverscape/ -v`

Expected: PASS, including `test_riverscape_package_never_imports_spatial` — `pipeline.py` must not appear in the offenders list.

- [ ] **Step 7: Commit**

```bash
git add hydrofragments/riverscape/pipeline.py hydrofragments/riverscape/__init__.py tests/riverscape/test_riverscape_pipeline.py
git commit -m "feat(riverscape): add build_landform pipeline orchestrator (Phase 6a Task 1)"
```

---

## Task 2: `zones_from_riverscape` (`spatial/zones.py`)

**Files:**
- Modify: `hydrofragments/spatial/zones.py` (add imports at the top, `_resolve_years` + `zones_from_riverscape` after `zones_from_wo_statistics` at line 147-184, extend `__all__` at line 249)
- Modify: `hydrofragments/spatial/__init__.py:23` and `:42`
- Test: `tests/spatial/test_zones_from_riverscape.py`

**Interfaces:**
- Consumes from Task 1: `build_landform(domain, frequency, drainage, reach_labels, reach_keys, upstr_darea, *, geobox, transform, pixel_m, cfg, years) -> LandformResult` with fields `landform`, `bridged_mask`, `degraded_reasons`.
- Consumes (pre-existing): `hydrofragments.riverscape.domain.wet_domain(stats, *, min_valid_obs) -> np.ndarray`; `hydrofragments.hydroperiod.classify.classify_hydroperiod(frequency, domain, *, t_persist, t_season, unobserved_mask=None) -> HydroperiodResult` (field `classes`); `combine_zones(landform, hydroperiod, *, source="riverscape", degraded_reasons=()) -> ZoneResult`; `_attach_grid(result, template, *, require_georeference=False) -> ZoneResult`.
- Produces for Task 3: `zones_from_riverscape(stats, *, drainage, config, reach_labels, reach_keys, upstr_darea, geobox, transform, pixel_m=30.0, years=None) -> ZoneResult` with `mode == "riverscape"`, `source == stats.product`, a non-`None` `crosstab`, and a `grid` attached from `stats.frequency`.

- [ ] **Step 1: Write the failing zoning tests**

Create `tests/spatial/test_zones_from_riverscape.py`:

```python
"""``zones_from_riverscape``: DEA stats + landform -> riverscape ZoneResult.

``build_landform`` is stubbed (it has its own suite in
``tests/riverscape/test_riverscape_pipeline.py`` and needs network-shaped
loaders); ``wet_domain``, ``classify_hydroperiod`` and ``combine_zones`` all
run for real, because the behaviour under test IS how this adapter wires
those three together -- in particular that the bridged mask reaches
``classify_hydroperiod`` as ``unobserved_mask``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("rioxarray")
from affine import Affine
from shapely.geometry import LineString

from hydrofragments.config import HydroConfig
from hydrofragments.riverscape.riverine import FloodplainEnvelope
from hydrofragments.riverscape.pipeline import LandformResult
from hydrofragments.spatial import zones as zones_module
from hydrofragments.spatial.zones import zones_from_riverscape

SHAPE = (3, 4)
PIXEL_M = 30.0
TRANSFORM = Affine(PIXEL_M, 0.0, 0.0, 0.0, -PIXEL_M, 120.0)
CRS = "EPSG:3577"
PRODUCT = "ga_ls_wo_fq_myear_3"

# Domain is exactly the six cells with non-zero frequency: count_wet > 0 and
# count_clear (40) >= the default min_valid_obs (20).
FREQUENCY = np.array(
    [
        [80.0, 30.0, 5.0, 0.0],
        [80.0, 30.0, 5.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
EXPECTED_DOMAIN = np.array(
    [
        [True, True, True, False],
        [True, True, True, False],
        [False, False, False, False],
    ]
)


def _config() -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "max_water",
                "composite_owner": "upstream",
            },
            "riverscape": {"mode": "auto"},
        }
    )


def _stats(*, years=None):
    y = 120.0 - np.arange(SHAPE[0]) * PIXEL_M - PIXEL_M / 2.0
    x = np.arange(SHAPE[1]) * PIXEL_M + PIXEL_M / 2.0
    frequency = xr.DataArray(
        FREQUENCY, dims=("y", "x"), coords={"y": y, "x": x}
    ).rio.write_crs(CRS)
    stats = SimpleNamespace(
        frequency=frequency,
        count_wet=np.where(FREQUENCY > 0, 5, 0),
        count_clear=np.full(SHAPE, 40),
        product=PRODUCT,
        version="0.1.0",
        crs=CRS,
        time_span=None,
        provenance={},
    )
    if years is not None:
        stats.years = years
    return stats


def _drainage() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame(
        {
            "HydroID": [1],
            "From_Node": [1],
            "To_Node": [2],
            "NextDownID": [-1],
            "UpstrDArea": [5000.0],
        },
        geometry=[LineString([(15.0, 105.0), (105.0, 105.0)])],
        crs=CRS,
    )


def _landform_result(landform, *, bridged=None, degraded_reasons=()):
    landform = np.asarray(landform, dtype=np.uint8)
    zeros = np.zeros(landform.shape, dtype=np.uint8)
    return LandformResult(
        landform=landform,
        channel_source=zeros.copy(),
        channel_confidence=zeros.copy(),
        rule_id=zeros.copy(),
        rem=np.zeros(landform.shape, dtype=np.float32),
        bridged_mask=(
            np.zeros(landform.shape, dtype=bool) if bridged is None else np.asarray(bridged, bool)
        ),
        channel_bridges=None,
        unbridged=(),
        envelope=FloodplainEnvelope(
            a=2.0,
            b=0.0,
            n_bins=1,
            bin_log_a_mid=(3.7,),
            bin_rem_quantile=(2.0,),
            bin_counts=(90,),
            degraded_reasons=(),
        ),
        provenance={"years": (2019, 2021)},
        degraded_reasons=tuple(degraded_reasons),
    )


def _install_build_landform(monkeypatch, result: LandformResult) -> list[dict]:
    calls: list[dict] = []

    def fake_build_landform(
        domain, frequency, drainage, reach_labels, reach_keys, upstr_darea, **kwargs
    ):
        calls.append(
            {
                "domain": np.asarray(domain).copy(),
                "frequency": np.asarray(frequency).copy(),
                "drainage": drainage,
                "reach_labels": reach_labels,
                "reach_keys": reach_keys,
                "upstr_darea": upstr_darea,
                **kwargs,
            }
        )
        return result

    monkeypatch.setattr(zones_module, "build_landform", fake_build_landform)
    return calls


def _zones(monkeypatch, result: LandformResult, **overrides):
    calls = _install_build_landform(monkeypatch, result)
    kwargs = {
        "drainage": _drainage(),
        "config": _config(),
        "reach_labels": np.ones(SHAPE, dtype=np.int32),
        "reach_keys": {1: "1"},
        "upstr_darea": {1: 5000.0},
        "geobox": SimpleNamespace(),
        "transform": TRANSFORM,
        "pixel_m": PIXEL_M,
        "years": (2019, 2021),
    }
    stats = overrides.pop("stats", None) or _stats()
    kwargs.update(overrides)
    return zones_from_riverscape(stats, **kwargs), calls


# --------------------------------------------------------------------------
# Crosstab and legacy mask (spec section 7.2 bullet 1)
# --------------------------------------------------------------------------


def test_crosstab_and_legacy_mask_follow_the_landform_hydroperiod_table(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    result, _calls = _zones(monkeypatch, _landform_result(landform))

    # hydroperiod: 80 > 50 -> persistent(1); 30 in [10, 50] -> seasonal(2);
    # 5 < 10 -> marginal(3). crosstab = landform * 10 + hydroperiod.
    assert result.crosstab.tolist() == [
        [11, 22, 23, 0],
        [21, 22, 23, 0],
        [0, 0, 0, 0],
    ]
    assert result.mask.tolist() == [
        [1, 3, 4, 0],
        [2, 3, 4, 0],
        [0, 0, 0, 0],
    ]
    assert result.emitted_zones == (1, 2, 3, 4)
    assert result.has_zone_1 is True
    assert result.mode == "riverscape"


def test_result_is_stamped_with_the_dea_product_and_a_grid(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    result, _calls = _zones(monkeypatch, _landform_result(landform))

    assert result.source == PRODUCT
    assert result.grid is not None
    assert (result.grid.height, result.grid.width) == SHAPE
    assert result.as_dataarray().shape == SHAPE


# --------------------------------------------------------------------------
# Bridged pixels (spec section 7.2 bullet 1, MUTANT 3)
# --------------------------------------------------------------------------


def test_bridged_pixel_is_hydroperiod_4_and_zone_1(monkeypatch) -> None:
    """MUTANT 3: dropping unobserved_mask from the classify_hydroperiod call
    makes the bridged pixel hydroperiod 0 while landform says 1, and
    combine_zones raises "landform and hydroperiod disagree on the zoned
    extent" -- so this test fails loudly rather than silently losing code 4.
    """
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [1, 0, 0, 0],
    ]
    bridged = np.zeros(SHAPE, dtype=bool)
    bridged[2, 0] = True

    result, _calls = _zones(
        monkeypatch, _landform_result(landform, bridged=bridged)
    )

    assert int(result.crosstab[2, 0]) == 14
    assert int(result.mask[2, 0]) == 1
    assert result.has_zone_1 is True


# --------------------------------------------------------------------------
# Degraded reasons (spec section 7.2 bullet 2)
# --------------------------------------------------------------------------


def test_degraded_reasons_are_forwarded_onto_the_zone_result(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    result, _calls = _zones(
        monkeypatch,
        _landform_result(
            landform, degraded_reasons=("bare_threshold_fallback", "envelope_single_bin")
        ),
    )

    assert result.degraded_reasons == (
        "bare_threshold_fallback",
        "envelope_single_bin",
    )


# --------------------------------------------------------------------------
# Argument forwarding into build_landform
# --------------------------------------------------------------------------


def test_build_landform_receives_the_wet_domain_and_stats_frequency(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    _result, calls = _zones(monkeypatch, _landform_result(landform))

    assert len(calls) == 1
    call = calls[0]
    assert call["domain"].dtype == np.bool_
    assert call["domain"].tolist() == EXPECTED_DOMAIN.tolist()
    np.testing.assert_allclose(call["frequency"], FREQUENCY)
    assert call["reach_keys"] == {1: "1"}
    assert call["upstr_darea"] == {1: 5000.0}
    assert call["transform"] == TRANSFORM
    assert call["pixel_m"] == PIXEL_M
    assert call["years"] == (2019, 2021)
    assert call["cfg"].mode == "auto"


def test_years_are_resolved_from_stats_when_not_supplied(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    _result, calls = _zones(
        monkeypatch,
        _landform_result(landform),
        stats=_stats(years=(2018, 2019, 2020, 2021)),
        years=None,
    )

    assert calls[0]["years"] == (2018, 2021)


def test_missing_years_without_stats_years_raises(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    _install_build_landform(monkeypatch, _landform_result(landform))

    with pytest.raises(ValueError, match="years"):
        zones_from_riverscape(
            _stats(),
            drainage=_drainage(),
            config=_config(),
            reach_labels=np.ones(SHAPE, dtype=np.int32),
            reach_keys={1: "1"},
            upstr_darea={1: 5000.0},
            geobox=SimpleNamespace(),
            transform=TRANSFORM,
            years=None,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/spatial/test_zones_from_riverscape.py -v`

Expected: FAIL at collection — `ImportError: cannot import name 'zones_from_riverscape' from 'hydrofragments.spatial.zones'`.

- [ ] **Step 3: Add the imports to `hydrofragments/spatial/zones.py`**

Replace the existing `SpatialGrid` + `riverscape.codes` import block (lines 22-28) with:

```python
from hydrofragments.hydroperiod.classify import classify_hydroperiod
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.riverscape.codes import (
    LANDFORM_CODES,
    LANDFORM_IN_CHANNEL,
    LANDFORM_OFF_CHANNEL_RIVERINE,
    LANDFORM_OUTSIDE,
)
from hydrofragments.riverscape.domain import wet_domain
from hydrofragments.riverscape.pipeline import build_landform
```

(The `SpatialGrid` and `riverscape.codes` lines are already present; this block keeps them and inserts the three new imports in alphabetical order alongside them.)

- [ ] **Step 4: Add `_resolve_years` and `zones_from_riverscape`**

Insert immediately after `zones_from_wo_statistics` (i.e. after line 184, before `_validated_codes`):

```python
def _resolve_years(stats: "WoStatistics", years: tuple[int, int] | None) -> tuple[int, int]:
    """Resolve the inclusive ``(start_year, end_year)`` Fractional Cover window.

    An explicit ``years`` always wins. Otherwise ``stats.years`` is used when
    the statistics object carries it (spec section 3.3); ``WoStatistics`` does
    not declare that field today, so a caller that omits both gets a
    ``ValueError`` rather than a silently guessed window -- the FC stack's
    year range is a scientific input, not a default.
    """
    if years is not None:
        return (int(years[0]), int(years[1]))
    available = getattr(stats, "years", None)
    if not available:
        raise ValueError(
            "zones_from_riverscape requires years when stats carries no years"
        )
    return (int(min(available)), int(max(available)))


def zones_from_riverscape(
    stats: "WoStatistics",
    *,
    drainage: Any,
    config: Any,
    reach_labels: np.ndarray,
    reach_keys: Any,
    upstr_darea: Any,
    geobox: Any,
    transform: Any,
    pixel_m: float = 30.0,
    years: tuple[int, int] | None = None,
) -> ZoneResult:
    """Build a riverscape ``ZoneResult`` from DEA stats + drainage context.

    Four steps, no new science (spec section 5): the observed-wet domain
    comes from ``wet_domain`` over the same ``stats`` the occurrence path
    reads; ``build_landform`` produces the landform layer; the hydroperiod
    layer is classified over that same domain with the landform's bridged
    mask supplied as ``unobserved_mask``; and ``combine_zones`` derives the
    crosstab and legacy zone mask from the two layers.

    Bridged pixels come out landform 1 / hydroperiod 4, so legacy Zone 1
    includes them. They lie strictly outside ``domain``
    (``bridging.bridge_gaps`` and ``riverscape.pipeline.build_landform``
    both enforce that), which is what lets ``classify_hydroperiod`` accept
    them as ``unobserved_mask`` -- it rejects an overlap.

    ``reach_labels``/``reach_keys``/``upstr_darea`` are built by the caller
    (``hydrofragments.riverscape`` must not import ``hydrofragments.spatial``,
    so it cannot build them itself) and forwarded unchanged. The returned
    ``ZoneResult`` is stamped with ``source=stats.product``, matching
    ``zones_from_wo_statistics``'s convention, and carries a grid attached
    from ``stats.frequency``.
    """
    resolved_years = _resolve_years(stats, years)
    domain = wet_domain(stats, min_valid_obs=config.validity.min_valid_obs)
    frequency = np.asarray(stats.frequency, dtype=float)

    landform_result = build_landform(
        domain,
        frequency,
        drainage,
        reach_labels,
        reach_keys,
        upstr_darea,
        geobox=geobox,
        transform=transform,
        pixel_m=pixel_m,
        cfg=config.riverscape,
        years=resolved_years,
    )

    hydroperiod = classify_hydroperiod(
        frequency,
        domain,
        t_persist=config.zones.t_persist,
        t_season=config.zones.t_season,
        unobserved_mask=landform_result.bridged_mask,
    )

    result = combine_zones(
        landform_result.landform,
        hydroperiod.classes,
        source=stats.product,
        degraded_reasons=landform_result.degraded_reasons,
    )
    return _attach_grid(result, stats.frequency)
```

- [ ] **Step 5: Extend both `__all__` lists**

`hydrofragments/spatial/zones.py` (last line, replacing the current `__all__`):

```python
__all__ = [
    "ZoneResult",
    "build_zones",
    "combine_zones",
    "zones_from_riverscape",
    "zones_from_wo_statistics",
]
```

`hydrofragments/spatial/__init__.py` — change line 23 to:

```python
from hydrofragments.spatial.zones import (
    ZoneResult,
    build_zones,
    zones_from_riverscape,
    zones_from_wo_statistics,
)
```

and add `"zones_from_riverscape",` to `__all__` immediately before `"zones_from_wo_statistics",`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/spatial/test_zones_from_riverscape.py -v`

Expected: PASS, all tests.

- [ ] **Step 7: Run the surrounding suites for regressions**

Run: `python -m pytest tests/spatial/ tests/hydroperiod/ tests/riverscape/ tests/api/ -q`

Expected: PASS. `tests/spatial/test_zones.py` and `tests/spatial/test_zone_combination.py` must be unaffected — `combine_zones`, `build_zones`, and `zones_from_wo_statistics` were not modified.

- [ ] **Step 8: Commit**

```bash
git add hydrofragments/spatial/zones.py hydrofragments/spatial/__init__.py tests/spatial/test_zones_from_riverscape.py
git commit -m "feat(spatial): add zones_from_riverscape landform x hydroperiod adapter (Phase 6a Task 2)"
```

---

## Task 3: Wire `analyze_from_dea` modes (`workflow.py`)

**Files:**
- Modify: `hydrofragments/workflow.py` (imports at 103-117; `_default_config` at 119-144; `_channel_inputs` at 274-299; `analyze_from_dea` at 302-465 — specifically the AOI-load block at 339, the zone block at 351-354, and the drainage block at 396-405; new helpers inserted after `_channel_inputs`)
- Test: `tests/integration/test_riverscape_modes.py`

**Interfaces:**
- Consumes from Task 1: `validate_drainage_columns(drainage) -> None`.
- Consumes from Task 2: `zones_from_riverscape(stats, *, drainage, config, reach_labels, reach_keys, upstr_darea, geobox, transform, pixel_m, years) -> ZoneResult`.
- Consumes (pre-existing): `zones_from_wo_statistics(stats, *, config, drainage_mask=None) -> ZoneResult`; `validate_drainage_topology(drainage) -> DrainageTopology` (raises `DrainageContractError`, a `ValueError` subclass); `hydrofragments.spatial.connectivity_context._build_reach_label_raster(drainage, *, buffer_m, transform, y_coords, x_coords) -> (np.ndarray, dict)` and `_raster_transform(y_coords, x_coords) -> Affine`; `hydrofragments.io.riverscape_sources.RiverscapeSourceUnavailable`.
- Produces: `_resolve_zone_result(stats, drainage_gdf, *, config, years, pixel_m, timings) -> tuple[ZoneResult | None, tuple[str, ...]]`; `_riverscape_reach_context(drainage_gdf, frequency, *, buffer_m) -> tuple[np.ndarray, dict[int, str], dict[int, float]]`; `_frequency_geobox(frequency) -> Any`; `_with_reasons(result, reasons) -> ZoneResult`.

- [ ] **Step 1: Write the failing mode tests**

Create `tests/integration/test_riverscape_modes.py`:

```python
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
    assert timings["total"] == pytest.approx(
        sum(value for key, value in timings.items() if key != "total")
    )
    assert (Path(result.output_dir) / "run_manifest.json").exists()


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/integration/test_riverscape_modes.py -v`

Expected: FAIL at collection — `ImportError: cannot import name '_resolve_zone_result' from 'hydrofragments.workflow'`.

- [ ] **Step 3: Extend `workflow.py`'s imports**

Add to the existing stdlib/third-party import block (lines 38-46, alongside `import numpy as np`):

```python
from dataclasses import replace

from scipy import ndimage
```

Then replace the `hydrofragments.*` import block at lines 103-117 with:

```python
from hydrofragments.api import _run_core_analysis, open_water_cube
from hydrofragments.config import HydroConfig
from hydrofragments.io.cache_footprints import open_verified_cache_footprints
from hydrofragments.io.dea import open_wo_statistics_for_zoning
from hydrofragments.io.riverscape_sources import RiverscapeSourceUnavailable
from hydrofragments.metrics import ApsecRecord
from hydrofragments.models import AnalysisInputs, HydroResult
from hydrofragments.output.finalize import finalize_analysis_bundle
from hydrofragments.output.manifest import build_dea_provenance
from hydrofragments.riverscape.pipeline import validate_drainage_columns
from hydrofragments.spatial import (
    SpatialContext,
    create_channel_context,
    reach_monthly_wet_profile,
    validate_drainage_topology,
)
from hydrofragments.spatial.connectivity_context import (
    _build_reach_label_raster,
    _raster_transform,
)
from hydrofragments.spatial.zones import (
    ZoneResult,
    zones_from_riverscape,
    zones_from_wo_statistics,
)
```

- [ ] **Step 4: Pin the built-in default config to `riverscape.mode="off"`**

Replace `_default_config` with:

```python
def _default_config(*, output_dir: str | Path) -> HydroConfig:
    """A sensible default ``HydroConfig`` for a DEA-sourced watermask cube.

    ``input.kind="watermask_tsfill"`` matches the ``{-2,-1,0,1}`` coded
    cube ``hydroseason.open_completed_mask_cache`` returns.
    ``temporal.monthly_composite="max_water"``/``composite_owner="upstream"``
    record that hydroseason's WOfS acquisition already resolved daily
    observations into a monthly composite before this cube was opened --
    HydroFragments computes no compositing of its own on this path.
    ``metric_profiles`` is left at ``HydroConfig``'s own default
    (``("all_available",)``, W4.1): every runtime-wired metric whose
    dependencies this run's inputs actually supply.

    ``riverscape.mode`` is pinned to ``"off"`` here, deliberately diverging
    from :class:`~hydrofragments.config.RiverscapeConfig`'s own ``"auto"``
    default. This is the minimal config built when a caller passes
    ``config=None``; it configures no riverscape source products, so
    leaving it at ``"auto"`` would make any run that supplies drainage
    reach for DEM/Fractional Cover/Waterbodies over the network without the
    caller ever asking for riverscape zoning. Opting in is an explicit act:
    pass a ``HydroConfig`` with ``riverscape.mode`` set to ``"auto"`` or
    ``"required"``.
    """
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
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
```

- [ ] **Step 5: Take a pre-loaded GeoDataFrame in `_channel_inputs`**

Replace `_channel_inputs` (lines 274-299) with:

```python
def _channel_inputs(
    drainage_gdf: Any,
    *,
    aoi_gdf: Any,
    aoi_id: str,
    water: "Any",
    target_crs: str,
) -> tuple[SpatialContext, "np.ndarray", list[float]]:
    """Build a real channel :class:`SpatialContext` plus its monthly wet profile.

    ``drainage_gdf`` is already normalized and topology-validated by
    :func:`analyze_from_dea` (Phase 6a spec section 6 moves that validation
    to right after AOI load, so a bad schema fails before any acquisition
    or riverscape loader work); this function does not reload it.

    ``water``'s per-month, per-reach wetness comes from
    :func:`hydrofragments.spatial.reach_monthly_wet_profile` (the same
    skeleton-seeded-buffer method already used for ``wet_any_month`` gating,
    kept per-month instead of collapsed to a single OR) -- no new metric
    kernel is introduced here, only the existing one's already-computed
    intermediate is kept instead of discarded.
    """
    context = create_channel_context(
        aoi_id, aoi_gdf, drainage_gdf, drainage_id="workflow", target_crs=target_crs,
    )
    wet_profile = reach_monthly_wet_profile(
        context.drainage, water, buffer_m=_DEFAULT_CHANNEL_BUFFER_M,
    )
    segment_lengths_m = context.drainage.geometry.length.tolist()
    return context, wet_profile, segment_lengths_m
```

- [ ] **Step 6: Add the riverscape helpers**

Insert immediately after `_channel_inputs` and before `analyze_from_dea`:

```python
#: Reach-buffer radius for the riverscape reach-label raster. Matches
#: ``_DEFAULT_CHANNEL_BUFFER_M`` (two 30 m pixels), the same radius
#: ``reach_monthly_wet_profile`` already uses to attribute water to a reach.
#: Unclaimed pixels are filled by nearest-label below, so this value only
#: decides which reach wins near a junction, not how far labels reach.
_RIVERSCAPE_REACH_BUFFER_M = 60.0


def _frequency_geobox(frequency: Any) -> Any:
    """Return the odc-geo ``GeoBox`` for the zoning grid.

    ``hydrofragments.io.riverscape_sources``'s loaders take a ``geobox=``
    argument so every raster they return lands on this exact grid. The
    ``.odc`` accessor is registered as an import side effect, so the import
    is performed here rather than at module scope -- ``analyze_from_dea``
    pays for it only on a run that actually needs riverscape sources.
    """
    import odc.geo.xr  # noqa: F401 -- registers the .odc DataArray accessor

    return frequency.odc.geobox


def _riverscape_reach_context(
    drainage_gdf: Any, frequency: Any, *, buffer_m: float
) -> tuple["np.ndarray", dict[int, str], dict[int, float]]:
    """Build the reach-label raster and the two label-keyed lookups riverscape needs.

    ``hydrofragments.riverscape`` must not import ``hydrofragments.spatial``
    (enforced by ``tests/riverscape/test_riverscape_import_boundary.py``), so
    the label raster is built here, on this side of the boundary, from
    ``spatial.connectivity_context``'s existing rasterizer rather than being
    reimplemented inside the pipeline.

    Two post-processing steps turn that raster into what the riverscape
    kernels expect. Overlap pixels (``_build_reach_label_raster``'s ``-1``
    sentinel, routine where adjacent reaches share an endpoint) are resolved
    to their lowest-index claimant, because ``classify_channel`` rejects a
    label it has no key for and a negative sentinel would simply be dropped
    from the channel. Then every still-unclaimed pixel takes its nearest
    labelled pixel's reach: both ``classify_channel``'s per-reach width
    growth and ``riverine``'s floodplain envelope look up a reach for
    pixels well outside any buffer, and an unlabelled pixel can never
    become channel or floodplain.

    ``reach_keys`` and ``upstr_darea`` are keyed by the integer LABEL value
    (``index + 1``), not by ``HydroID`` -- ``riverine._lookup_area``
    compares its keys directly against ``reach_labels`` values.
    """
    y_coords = np.asarray(frequency["y"].values, dtype=float)
    x_coords = np.asarray(frequency["x"].values, dtype=float)
    labels, overlaps = _build_reach_label_raster(
        drainage_gdf,
        buffer_m=buffer_m,
        transform=_raster_transform(y_coords, x_coords),
        y_coords=y_coords,
        x_coords=x_coords,
    )
    for (row, col), indices in overlaps.items():
        labels[row, col] = min(indices) + 1
    if labels.max() > 0 and (labels == 0).any():
        _, nearest = ndimage.distance_transform_edt(labels == 0, return_indices=True)
        labels = labels[nearest[0], nearest[1]]
    reach_keys = {
        index + 1: str(value) for index, value in enumerate(drainage_gdf["HydroID"])
    }
    upstr_darea = {
        index + 1: float(value) for index, value in enumerate(drainage_gdf["UpstrDArea"])
    }
    return labels.astype(np.int32), reach_keys, upstr_darea


def _with_reasons(result: "ZoneResult", reasons: tuple[str, ...]) -> "ZoneResult":
    """Append ``reasons`` to ``result``'s degraded reasons, order-preserving."""
    merged = tuple(dict.fromkeys(tuple(result.degraded_reasons) + tuple(reasons)))
    return replace(result, degraded_reasons=merged)


def _resolve_zone_result(
    stats: Any,
    drainage_gdf: Any,
    *,
    config: HydroConfig,
    years: tuple[int, int],
    pixel_m: float,
    timings: dict[str, float],
) -> tuple["ZoneResult | None", tuple[str, ...]]:
    """Resolve this run's ``ZoneResult`` per the Phase 6a mode table (spec section 6).

    Returns ``(zone_result, degraded_reasons)``. The reasons are already
    merged onto ``zone_result`` when there is one; they are also returned
    separately so the ``stats is None`` case can still report WHY riverscape
    zoning was skipped, which has no ``ZoneResult`` to carry it (Phase 6b
    writes these into the manifest's ``zoning`` section).

    | mode | drainage | sources | result |
    |---|---|---|---|
    | ``off`` | any | any | occurrence if stats, else ``None`` |
    | ``auto`` | missing | -- | occurrence + ``riverscape_no_drainage`` |
    | ``auto`` | present | ok | ``zones_from_riverscape`` |
    | ``auto`` | present | source failure | occurrence + ``riverscape_source_unavailable`` |
    | ``auto`` | present | no stats | ``None`` + ``riverscape_no_stats`` |
    | ``required`` | missing | -- | raise |
    | ``required`` | present | source failure | propagate |
    | ``required`` | present | ok | ``zones_from_riverscape`` |

    Only ``RiverscapeSourceUnavailable`` and the declared no-drainage /
    no-stats cases map to the ``auto`` fallback. Every other exception
    propagates, so a genuine bug is never laundered into a degraded run.

    ``timings["riverscape"]`` records the riverscape branch's own wall time
    and is absent when the branch is skipped;
    :func:`analyze_from_dea` subtracts it from ``dea_planning`` so the
    manifest's phases stay disjoint.
    """
    mode = config.riverscape.mode

    if mode == "off":
        if stats is None:
            return None, ()
        return zones_from_wo_statistics(stats, config=config), ()

    if drainage_gdf is None:
        if mode == "required":
            raise RiverscapeSourceUnavailable(
                "riverscape.mode='required' needs drainage lines; none were supplied"
            )
        if stats is None:
            return None, ("riverscape_no_drainage",)
        reasons = ("riverscape_no_drainage",)
        return (
            _with_reasons(zones_from_wo_statistics(stats, config=config), reasons),
            reasons,
        )

    if stats is None:
        if mode == "required":
            raise RiverscapeSourceUnavailable(
                "riverscape.mode='required' needs DEA WO statistics; none were available"
            )
        return None, ("riverscape_no_stats",)

    started = time.perf_counter()
    try:
        reach_labels, reach_keys, upstr_darea = _riverscape_reach_context(
            drainage_gdf, stats.frequency, buffer_m=_RIVERSCAPE_REACH_BUFFER_M
        )
        zone_result = zones_from_riverscape(
            stats,
            drainage=drainage_gdf,
            config=config,
            reach_labels=reach_labels,
            reach_keys=reach_keys,
            upstr_darea=upstr_darea,
            geobox=_frequency_geobox(stats.frequency),
            transform=_raster_transform(
                np.asarray(stats.frequency["y"].values, dtype=float),
                np.asarray(stats.frequency["x"].values, dtype=float),
            ),
            pixel_m=pixel_m,
            years=years,
        )
    except RiverscapeSourceUnavailable:
        timings["riverscape"] = time.perf_counter() - started
        if mode == "required":
            raise
        reasons = ("riverscape_source_unavailable",)
        return (
            _with_reasons(zones_from_wo_statistics(stats, config=config), reasons),
            reasons,
        )
    timings["riverscape"] = time.perf_counter() - started
    return zone_result, tuple(zone_result.degraded_reasons)
```

- [ ] **Step 7: Rewire `analyze_from_dea`**

Replace the AOI-load line (`aoi_gdf = _load_geometry(aoi)`, line 339) with:

```python
    aoi_gdf = _load_geometry(aoi)

    # Parent spec section 7: validate drainage right after AOI load, before
    # any acquisition or riverscape loader work. Topology is checked
    # whatever the mode (channel metrics already need it); the extra
    # riverscape column schema is checked only when riverscape zoning can
    # actually run, so a caller with mode="off" is not forced to supply
    # UpstrDArea.
    drainage_gdf = None
    if drainage is not None:
        drainage_gdf = _load_geometry(drainage)
        validate_drainage_topology(drainage_gdf)
        if resolved_config.riverscape.mode != "off":
            validate_drainage_columns(drainage_gdf)
```

Replace the zone block (lines 351-354) with:

```python
    zone_result, _riverscape_degraded_reasons = _resolve_zone_result(
        stats,
        drainage_gdf,
        config=resolved_config,
        years=(start.year, end.year),
        pixel_m=resolution,
        timings=timings,
    )
    # timings["riverscape"] is a sub-phase of planning, not an extra phase:
    # finalize_analysis_bundle derives "total" as the sum of every other
    # key, so the riverscape branch's time is carved out of dea_planning
    # here instead of being counted twice.
    # _riverscape_degraded_reasons is Phase 6b's input for the manifest's
    # "zoning" section; 6a only has to produce it.
    timings["dea_planning"] = (
        time.perf_counter() - t0 - timings.get("riverscape", 0.0)
    )
```

Replace the drainage block (lines 396-405) with:

```python
    channel_context: SpatialContext | None = None
    channel_wet_profiles = None
    channel_segment_lengths_m = None
    if drainage_gdf is not None:
        channel_context, channel_wet_profiles, channel_segment_lengths_m = (
            _channel_inputs(
                drainage_gdf,
                aoi_gdf=aoi_gdf,
                aoi_id=aoi_id,
                water=cube.water,
                target_crs=resolved_config.spatial.target_crs,
            )
        )
```

Finally, extend `analyze_from_dea`'s docstring with a paragraph after the existing "On DEA-statistics unavailability..." block:

```python
    ``config.riverscape.mode`` selects the zoning path (Phase 6a spec
    section 6). ``"off"`` keeps today's occurrence zoning. ``"auto"`` uses
    riverscape zoning when drainage and the riverscape source products are
    both available, and otherwise degrades to occurrence zoning with a
    recorded reason (``riverscape_no_drainage`` /
    ``riverscape_source_unavailable``) rather than failing the run.
    ``"required"`` raises instead of degrading. Drainage is normalized and
    validated immediately after the AOI, before any acquisition, so a bad
    drainage schema never costs a WOfS read.
```

- [ ] **Step 8: Run the new tests to verify they pass**

Run: `python -m pytest tests/integration/test_riverscape_modes.py -v`

Expected: PASS, all tests.

- [ ] **Step 9: Run the existing DEA workflow suite for regressions**

Run: `python -m pytest tests/integration/ tests/spatial/ tests/riverscape/ tests/gating/ -q`

Expected: PASS. In particular `tests/integration/test_dea_workflow.py` must be unchanged in behaviour: `_default_config` pins `riverscape.mode="off"`, so `test_analyze_from_dea_with_drainage_computes_channel_metrics` still takes the occurrence path with no network access, and `test_analyze_from_dea_records_phase_timings` still sees exactly the five original keys (`riverscape` is absent when the branch is skipped).

- [ ] **Step 10: Run the full suite**

Run: `python -m pytest -q`

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add hydrofragments/workflow.py tests/integration/test_riverscape_modes.py
git commit -m "feat(workflow): wire riverscape.mode off/auto/required into analyze_from_dea (Phase 6a Task 3)"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Requirement | Where |
|---|---|---|
| §1.1.1 | `pipeline.py` with `build_landform` loading DEM/FC/waterbodies and running Phase 3–5 kernels | Task 1 Step 3 |
| §1.1.2 | `zones_from_riverscape`: domain → landform → hydroperiod (bridged `unobserved_mask`) → `combine_zones` + grid attach | Task 2 Step 4 |
| §1.1.3 | `analyze_from_dea` wired for `off`/`auto`/`required` with monkeypatch tests | Task 3 Steps 6-7, test file |
| §1.2 | Out of scope | Global Constraints ("Out of scope"); no export/manifest/window/MrVBF code in any task |
| §1.3 | Import boundary; reach labels from `spatial`/`workflow`; `spatial` may import `riverscape.pipeline` | Global Constraints; Task 1 Step 6 (guard re-run); `_riverscape_reach_context` in Task 3 Step 6 |
| §2.1-2.9 | All nine locked decisions | 6a/6b split (plan scope); depth (Task 3 monkeypatched loaders); orchestrator not spatial context (Task 1); no-drainage behaviour (`_resolve_zone_result`); domain/hydroperiod (Task 2 Step 4); `channel_source` 1/2/0 (`_package_channel_source`); windowing deferred; closed-loop deferred; reach labels by caller |
| §3.1 | `LandformResult` fields + `CHANNEL_SOURCE_*` constants; `grid` typed `Any` | Task 1 Step 3 (`grid` moved last for its default — noted in the docstring) |
| §3.2 | `build_landform` signature, injectable loaders, raises `RiverscapeSourceUnavailable`, required-column `ValueError` | Task 1 Step 3 (`build_landform`, `validate_drainage_columns`) |
| §3.3 | `zones_from_riverscape` signature; `years is None` → stats years else `ValueError` | Task 2 Step 4 (`_resolve_years`) |
| §3.4 | Mode resolution helper; no silent catch of unexpected exceptions | Task 3 Step 6 (`_resolve_zone_result`); `test_unexpected_exceptions_are_never_swallowed_as_a_degrade` |
| §4 steps 1-12 | Validation, water seed, loads, corridor, centreline (on seed), REM, waterbodies, evidence, channel, bridge, riverine, packaging | Task 1 Step 3, each step commented inline; per-step fixture arithmetic in Task 1's preamble |
| §4 closing | No silent catch inside `build_landform` | Task 1 Step 3 module docstring; `test_loader_failure_propagates_unchanged` (Mutant 5) |
| §5.1-5.5 | `wet_domain`, `build_landform`, `classify_hydroperiod` with bridged mask, `combine_zones`, `_attach_grid`; bridged → landform 1 + hydroperiod 4 in Zone 1 | Task 2 Step 4; `test_bridged_pixel_is_hydroperiod_4_and_zone_1` |
| §6 table | All eight rows | Task 3 `_resolve_zone_result` docstring table + one test per row |
| §6 | `stats is None` → `auto` returns `None` + `riverscape_no_stats`; `required` raises | `test_auto_without_stats_reports_no_stats_and_no_zones`, `test_required_without_stats_raises` |
| §6 | Drainage validated before loaders; parent §7 early validation even when `off` | Task 3 Step 7; `test_bad_drainage_schema_fails_before_acquisition`, `test_mode_off_still_validates_drainage_topology_early` |
| §6 | `timings["riverscape"]` records the branch only; absent when skipped | Task 3 Steps 6-7; `test_mode_off_uses_occurrence_zoning` (absent), `test_analyze_from_dea_auto_with_drainage_runs_the_riverscape_branch` (present + disjoint) |
| §7.1 | Happy path; loader raise propagates; shape mismatch; `bridge_enabled=False` | Task 1 Step 1, four test groups |
| §7.2 | Crosstab + legacy mask; bridged → 4/Zone 1; degraded reasons forwarded | Task 2 Step 1 |
| §7.3 | Six workflow-mode scenarios | Task 3 Step 1 (plus the two extra `required` and non-source-failure rows) |
| §7 Mutants 1-5 | 1, 2 → Task 3; 3 → Task 2; 4, 5 → Task 1 | Each test is labelled with its mutant number in a docstring |
| §8 | Six file touches | File Structure table; `tests/workflow/` → `tests/integration/` per Decision D3 |
| §9 | Stable provenance keys 6b copies; no alternate export DTOs | `_build_provenance` + `test_happy_path_provenance_carries_stable_keys_for_phase_6b`; `with_grid` is the only 6b seam added |

The `.superpowers/sdd/progress.md` ledger row from §8 is the executing skill's own artifact (local/gitignored), not a task deliverable.

**2. Placeholder scan**

No `TBD`, `TODO`, "implement later", "add appropriate error handling", "similar to Task N", or bare prose-only code steps. Every code step contains complete, runnable code. Every type, function, and constant referenced in a later task is defined in an earlier one (`LandformResult`, `build_landform`, `validate_drainage_columns`, `zones_from_riverscape`, `_resolve_zone_result`, `_riverscape_reach_context`, `_frequency_geobox`, `_with_reasons`).

**3. Type consistency**

- `build_landform`'s six positional parameters are `(domain, frequency, drainage, reach_labels, reach_keys, upstr_darea)` in Task 1's definition, Task 2's call, and Task 2's test stub — identical order.
- `zones_from_riverscape`'s keyword set is identical in Task 2's definition, Task 2's test harness, and Task 3's call: `drainage, config, reach_labels, reach_keys, upstr_darea, geobox, transform, pixel_m, years`.
- `reach_keys: Mapping[int, str]` and `upstr_darea: Mapping[int, float]` are label-keyed in Task 1's docstring, Task 3's `_riverscape_reach_context` return, and both test suites' `{1: "1"}` / `{1: 5000.0}`.
- `_resolve_zone_result` returns `tuple[ZoneResult | None, tuple[str, ...]]` in its definition, its docstring, `analyze_from_dea`'s unpack, and every test.
- `CHANNEL_SOURCE_*` / `CHANNEL_CONFIDENCE_NODATA` are defined once in `pipeline.py` and imported by name in Task 1's tests and `riverscape/__init__.py`.
- `LandformResult` field names match exactly between Task 1's dataclass and Task 2's `_landform_result` stub (all eleven required fields plus the defaulted `grid`).
- Degraded-reason strings are spelled identically everywhere: `riverscape_no_drainage`, `riverscape_no_stats`, `riverscape_source_unavailable`, `bare_threshold_fallback`, `envelope_single_bin`, `reach_{key}_line_fallback`.
