# Riverscape Zoning Phase 6b: Exports, Manifest, and Gating

**Date:** 2026-09-18  
**Status:** Approved  
**Parent spec:** `docs/superpowers/specs/2026-09-14-riverscape-zoning-design.md`  
**Prior phase:** Phase 6a  
`docs/superpowers/specs/2026-09-18-riverscape-phase6a-workflow-design.md`

This document specifies Phase **6b** only. Where this document is more
specific than the parent for 6b, this document governs. Phase 6a mode
wiring and landform science stay unchanged. Phase 7 windowing and Phase 8
validation remain out of scope.

## 1. Scope and boundaries

### 1.1 In scope (6b)

1. Retain riverscape science arrays past zoning via a slim export bundle
   (no second DEM/FC/waterbodies load).
2. Manifest `zoning` section on `run_manifest.json`.
3. Mode-keyed zone export names; zones GPKG columns `mode`, `landform`,
   `hydroperiod` (stable schema with nulls when unavailable).
4. New spatial product `riverscape_evidence` (rasters) and vector layer
   `channel_bridges` in `vectors/spatial.gpkg`.
5. Extend the zones-do-not-multiply-metrics gating test to riverscape.
6. Update `docs/spatial_exports.md` for the new products and columns.

### 1.2 Out of scope

- Changing `riverscape.mode` behaviour, `build_landform` ordering, or
  hydroperiod science (6a).
- `windows.py` / sub-catchment tiling → **Phase 7**.
- Manager-polygon validation harness → **Phase 8**.
- Closed-loop bridge topology; MrVBF.
- Config schema version bump (no new scientific field).

### 1.3 Import boundary

Unchanged from 6a: `hydrofragments.riverscape` must not import
`hydrofragments.spatial`. Export writers live in `hydrofragments/output/`
and are fed from `workflow` / `finalize`. `riverscape.pipeline.with_grid`
remains the seam for attaching a `SpatialGrid` to `LandformResult`.

## 2. Locked decisions

| # | Topic | Decision |
|---|---|---|
| 1 | Scope | Parent §7 leftovers: manifest, names/GPKG, evidence rasters, bridges, gating, docs |
| 2 | Architecture | `RiverscapeExportBundle` held beside `ZoneResult`; finalize writes artifacts |
| 3 | Retention | Surface `LandformResult` (and hydroperiod classes) once from zoning; do not discard |
| 4 | Manifest | `zoning` whenever a `ZoneResult` exists; thin occurrence subset vs full riverscape |
| 5 | Reasons | Prefer `ZoneResult.degraded_reasons`; workflow tuple for no-zone fallback rows |
| 6 | Zone names | Riverscape → parent §3.4; occurrence keeps current finalize names |
| 7 | GPKG schema | Always emit `mode`, `landform`, `hydroperiod` columns; null when unavailable |
| 8 | Evidence product | New `SpatialProduct` `riverscape_evidence`; write when requested **and** bundle present |
| 9 | Product default | `_default_config` does not auto-add the product (occurrence hermetic) |
| 10 | Bridges layer | Stable schema; empty GeoDataFrame → empty layer when bundle/product path runs |
| 11 | Grid | Evidence rasters require `LandformResult.grid` (via `with_grid`) or `ZoneResult.grid` |
| 12 | Gating | Extend existing gating test family so riverscape zones do not change metrics |
| 13 | Hash | No schema bump; changelog note for any hash shifts from product allowlist / D1 |
| 14 | Tests | Unit + monkeypatched integration; no live STAC/WFS in CI |

## 3. Interfaces

### 3.1 `RiverscapeExportBundle`

Produced only on a successful riverscape zoning path (not on `off`, and not
when `auto` falls back to occurrence).

```python
@dataclass(frozen=True)
class RiverscapeExportBundle:
    landform: LandformResult          # grid attached before finalize
    hydroperiod_classes: np.ndarray   # uint8; same grid as landform
    zone_result: ZoneResult           # mode == "riverscape"
```

`hydroperiod_classes` is the array already passed into `combine_zones`
(so crosstab/`zone_crosstab` stays consistent with the mask). Do not
re-classify `classify_hydroperiod` at finalize.

### 3.2 Zoning retention seam

Today `zones_from_riverscape` discards `LandformResult` after
`combine_zones`. Phase 6b changes that contract:

- `zones_from_riverscape` returns `(ZoneResult, LandformResult)` **or**
  attaches landform via a documented helper used only by workflow
  (implementation plan picks one; both must keep the public
  `ZoneResult` shape used by occurrence callers).
- Recommended: keep `zones_from_riverscape(...) -> ZoneResult` for API
  stability and add `zones_from_riverscape_with_landform(...) ->
  tuple[ZoneResult, LandformResult]` (or an optional
  `return_landform: bool = False`). Workflow uses the landform-returning
  form; existing tests that ignore the second value stay green.
- `_resolve_zone_result` returns
  `(ZoneResult | None, RiverscapeExportBundle | None, tuple[str, ...])`
  (or equivalent) so `analyze_from_dea` can pass the bundle into
  `finalize_analysis_bundle`.

`LandformResult.grid` must be set before raster export (`with_grid` using
the same template as `_attach_grid` on the `ZoneResult`).

### 3.3 Manifest `zoning` section

`build_run_manifest` gains a top-level `zoning` object whenever
`ZoneResult` is present. Shape:

```json
{
  "mode": "riverscape",
  "domain_pixel_count": 12345,
  "domain_digest": "<stable hex>",
  "degraded_reasons": ["bare_threshold_fallback"],
  "ruleset_versions": {
    "channel": "...",
    "waterbody": "...",
    "riverine": "..."
  },
  "sources": {
    "dem_product": "...",
    "dem_band": "...",
    "fc_product": "...",
    "fc_bands": ["bare", "green", "npv"],
    "waterbodies_source": "...",
    "years": [2019, 2021],
    "zone_source": "<stats.product>"
  },
  "envelope": {"a": 2.0, "b": 0.0, "n_bins": 1},
  "pixel_counts": { "...": "from LandformResult.provenance" },
  "bridge_counts_by_cause": {"vegetated": 1},
  "unbridged_counts_by_reason": {},
  "reach_counts": {"total": 1, "line_fallback": 0, "multithread": 0},
  "bridged_length_m": 0.0,
  "bridged_area_m2": 0.0,
  "non_riverine_area_m2": 0.0
}
```

**Occurrence / thin subset** (`mode == "occurrence"` or riverscape
fallback that still yields a `ZoneResult`): include at least `mode`,
`domain_pixel_count` / digest from the zone mask (or frequency domain
digest already used by DEA provenance), `degraded_reasons`, and
`zone_source`. Riverscape-only keys are omitted or set to empty
defaults — prefer **omit** for absent science, but always include
`mode` and `degraded_reasons`.

**No `ZoneResult`** (rare `auto` rows that return `None`): still record
workflow-level reasons under `zoning` with `mode` reflecting the
configured riverscape mode and `degraded_reasons` from the workflow
tuple (`riverscape_no_drainage`, `riverscape_no_stats`, …), without
fabricating domain digests.

Provenance keys copy **verbatim** from
`LandformResult.provenance` (6a §9 / `_build_provenance`) — do not rename.

Area/length fields:

- `bridged_length_m` = sum of `channel_bridges.length_m` (0 if empty).
- `bridged_area_m2` = `count(bridged_mask) * pixel_m²`.
- `non_riverine_area_m2` = `pixel_counts.non_riverine * pixel_m²`.

Domain digest: stable hash of the boolean domain (or legacy mask for
occurrence) — reuse an existing digest helper if one already exists for
`zone_mask_digest` in DEA provenance; do not invent a second algorithm.

### 3.4 Zone names and GPKG columns

**Names** (finalize `_ZONE_NAMES` becomes mode-keyed):

| Mode | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `occurrence` | `channel_connected` | `persistent` | `seasonal` | `ephemeral` |
| `riverscape` | `in_channel` | `persistent_off_channel` | `seasonal_floodplain` | `marginal_floodplain` |

**GPKG `zones` layer** always includes:

| Column | Type | Notes |
|---|---|---|
| `zone_id` | int | existing |
| `zone_name` | str | mode-keyed |
| `area_km2` | float | existing |
| `source` | str | existing |
| `mode` | str | `occurrence` / `riverscape` |
| `landform` | int or null | dominant / per-polygon from dissolved mask; see below |
| `hydroperiod` | int or null | same |
| `geometry` | polygon | existing |

Dissolved legacy zones are multipolygons per `zone_id`. For riverscape,
`landform` / `hydroperiod` on a dissolved Zone-1 polygon are not unique
(bridges mix hydroperiod 4 with 1–3). **Decision:** store null for
`landform`/`hydroperiod` on dissolved zone rows; the authoritative
per-pixel values live in `riverscape_evidence` (`landform`,
`hydroperiod`, `zone_crosstab`). Document this in `spatial_exports.md`.
(Alternative of exploding to crosstab polygons is explicitly out of
scope for 6b.)

### 3.5 Spatial product `riverscape_evidence`

Add `"riverscape_evidence"` to `SpatialProduct` / `SPATIAL_PRODUCTS`.

**Write when:** product is in `config.output.spatial_products` **and**
`RiverscapeExportBundle` is present. If the product is requested but the
bundle is missing (occurrence run), preflight fails with a clear error
(same style as zones-without-`ZoneResult`).

**Raster layers** (GeoTIFF under `rasters/`, names stable):

| File / band role | dtype | nodata / notes |
|---|---|---|
| `landform` | uint8 | 0 outside |
| `hydroperiod` | uint8 | 0 outside; 4 = unobserved bridge |
| `zone_crosstab` | uint8 or uint16 | `landform*10 + hydroperiod`; 0 outside |
| `channel_source` | uint8 | 0/1/2 per 6a |
| `channel_confidence` | uint8 | 255 outside channel |
| `rem` | float32 | metres; NaN outside domain∪bridged as packaged |

Exact on-disk naming follows existing `RASTER_PRODUCT_CONTRACTS` patterns
(plan specifies filenames). Grid must match `zones.tif`.

### 3.6 Vector layer `channel_bridges`

Written into `vectors/spatial.gpkg` as layer `channel_bridges` when the
evidence product path runs (same gate as §3.5) **or** when a dedicated
plan-local rule ties it to the bundle — recommended: **same gate as
`riverscape_evidence`** so bridges are not orphaned without evidence.

Schema matches `bridging._empty_bridge_frame`:

`gap_id`, `reach_ids`, `length_m`, `cumulative_cost`, `cost_per_m`,
`upstream_width_m`, `downstream_width_m`, `bridge_confidence`,
`gap_cause`, `geometry` (LineString).

Empty frame → empty layer (schema present). CRS = analysis CRS
(EPSG:3577).

### 3.7 Finalize / workflow plumbing

```
analyze_from_dea
  → _resolve_zone_result → (zone_result, bundle|None, reasons)
  → finalize_analysis_bundle(..., zone_result=..., riverscape_bundle=...)
       → build zoning dict
       → mode-keyed zone names + GPKG columns
       → optional riverscape_evidence rasters + channel_bridges
       → build_run_manifest(..., zoning=...)
```

`_riverscape_degraded_reasons` is no longer discarded: it feeds `zoning`
when `zone_result is None`, and otherwise should match
`zone_result.degraded_reasons` after `_with_reasons`.

## 4. Gating

Extend `tests/gating/test_zones_do_not_multiply_metrics.py` (or an
adjacent module with a unique basename) so that:

1. Branch A builds a riverscape `ZoneResult` (stubbed loaders /
   `build_landform` as in 6a tests).
2. Branch B runs `analyze()` / metric path with no zones.
3. Metric tables are identical (same assertion helper as today).

`guards/scientific.py` stays unchanged: persistence-by-zone remains
refused; landform 1 still uses water evidence.

## 5. Error handling

| Case | Behaviour |
|---|---|
| `riverscape_evidence` requested, no bundle | Preflight `ValueError` / existing preflight error type |
| Bundle present, `landform.grid` missing | Raise before write (programming error) |
| CRS mismatch on bridges vs GPKG | Reproject to bundle/zone grid CRS or raise; prefer raise if already projected in 6a |
| Manifest without `ZoneResult` but reasons present | Thin `zoning` with reasons only |

No silent catch of `RiverscapeSourceUnavailable` in export code (export
never loads sources).

## 6. Testing

- Unit: `build_zoning_section` (or equivalent) from bundle vs occurrence
  `ZoneResult`; mode-keyed names; GPKG column presence/nulls.
- Unit: evidence raster contracts (dtype, shape, nodata); empty bridges
  layer schema.
- Gating: riverscape zones do not multiply metrics.
- Integration (monkeypatch): `analyze_from_dea` with mode `auto`, product
  opted in → manifest has `zoning.mode == "riverscape"`, evidence files
  exist, bridges layer exists.
- Regression: `mode=off` / occurrence defaults — no evidence files; zone
  names unchanged; GPKG still has null `landform`/`hydroperiod`.

## 7. Documentation

Update `docs/spatial_exports.md`:

- Zones section: mode-keyed names; new columns; null landform/hydroperiod
  rationale.
- New section: `riverscape_evidence` layers and prerequisites.
- New section: `channel_bridges` layer schema.

## 8. File touch list (indicative)

| Path | Action |
|---|---|
| `hydrofragments/config.py` | Add `riverscape_evidence` to spatial product allowlist |
| `hydrofragments/spatial/zones.py` | Landform-returning helper / flag |
| `hydrofragments/workflow.py` | Bundle plumbing into finalize |
| `hydrofragments/output/manifest.py` | `zoning` section |
| `hydrofragments/output/finalize.py` | Names, GPKG columns, bundle args, bridges write |
| `hydrofragments/output/rasters.py` | Evidence contracts + writer |
| `hydrofragments/output/bundle.py` | Pass `zoning` into manifest if needed |
| `docs/spatial_exports.md` | Document products |
| `tests/gating/...` | Riverscape gating |
| `tests/output/` / `tests/integration/` | Manifest, evidence, bridges |

## 9. Relationship to 6a and later phases

6a already populates `LandformResult.provenance`, `channel_bridges`,
`degraded_reasons`, and mode behaviour. 6b only **retains and exports**.
Phase 7 must not invent a second provenance vocabulary. Phase 8 consumes
manifest `zoning` and evidence rasters for validation reports.

## 10. Acceptance

- Riverscape run with `riverscape_evidence` opted in writes full `zoning`,
  mode-correct zone names, six evidence rasters, and `channel_bridges`.
- Occurrence / `mode=off` stays hermetic (no evidence product by default).
- Gating: metrics unchanged with riverscape zones present.
- Import boundary AST test still green.
- No live network in CI tests.
