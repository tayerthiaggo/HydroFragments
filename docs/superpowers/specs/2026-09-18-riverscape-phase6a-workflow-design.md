# Riverscape Zoning Phase 6a: Landform Pipeline and Workflow Modes

**Date:** 2026-09-18  
**Status:** Approved  
**Parent spec:** `docs/superpowers/specs/2026-09-14-riverscape-zoning-design.md`  
**Prior phases:** Phase 4 design/findings; Phase 5  
`docs/superpowers/specs/2026-09-17-riverscape-phase5-riverine-design.md`

This document specifies Phase **6a** only. Phase **6b** (exports, manifest
`zoning` section, `riverscape_evidence` product, gating extension) is a
separate design. Where this document is more specific than the parent for
6a, this document governs. Phase 7 windowing and Phase 8 validation remain
unchanged.

## 1. Scope and boundaries

### 1.1 In scope (6a)

1. `hydrofragments/riverscape/pipeline.py` — `build_landform` orchestrator
   that loads DEM/FC/waterbodies and runs Phase 3–5 kernels in order.
2. `zones_from_riverscape` in `hydrofragments/spatial/zones.py` — domain →
   landform → hydroperiod (with bridged `unobserved_mask`) → `combine_zones`,
   with grid attachment from WoStatistics templates.
3. Wire `analyze_from_dea` for `riverscape.mode` ∈ {`off`, `auto`, `required`}
   with monkeypatch-style integration tests.

### 1.2 Out of scope

- Manifest `zoning` polish, `riverscape_evidence` rasters, `channel_bridges`
  export packaging, finalize zone-name tables, gating extension → **6b**.
- `windows.py` / sub-catchment tiling → **Phase 7**.
- New bridge closed-loop topology rules (Phase 4 deferral stays deferred).
- MrVBF; hash schema bump unless a new scientific field appears (none
  planned).

### 1.3 Import boundary

- `hydrofragments.riverscape` must **not** import `hydrofragments.spatial`.
- Reach-label rasters come from `create_channel_context` /
  `_build_reach_label_raster` in `spatial/connectivity_context.py`, called
  only from `spatial` or `workflow` — never from `riverscape.pipeline`.
- `zones_from_riverscape` (in `spatial`) may import `riverscape.pipeline`.

## 2. Locked decisions

| # | Topic | Decision |
|---|---|---|
| 1 | Phase 6 split | 6a pipeline+zones+workflow modes; 6b exports/manifest/gating |
| 2 | 6a depth | Library + `analyze_from_dea` wiring; monkeypatched loaders in tests |
| 3 | `build_landform` | Orchestrator: loaders + Phase 3–5 kernels (not spatial context) |
| 4 | No drainage | `auto` → occurrence + `riverscape_no_drainage`; `required` → raise |
| 5 | Domain/hydroperiod | WoStatistics frequency + `wet_domain`; bridged → hydroperiod 4 |
| 6 | `channel_source` | 1 observed, 2 bridged; else 0 |
| 7 | Windowing | Phase 7 |
| 8 | Closed-loop bridges | Deferred |
| 9 | Reach labels | Built by caller (`spatial`/`workflow`); passed into pipeline |

## 3. Interfaces

### 3.1 LandformResult

```python
CHANNEL_SOURCE_NONE = 0
CHANNEL_SOURCE_OBSERVED = 1
CHANNEL_SOURCE_BRIDGED = 2

@dataclass(frozen=True)
class LandformResult:
    landform: np.ndarray          # uint8 codes 0–3
    channel_source: np.ndarray    # uint8 0/1/2
    channel_confidence: np.ndarray  # uint8; 255 = nodata outside channel
    rule_id: np.ndarray           # from ChannelResult; 0 outside channel
    rem: np.ndarray               # float32
    bridged_mask: np.ndarray      # bool
    channel_bridges: Any          # GeoDataFrame from BridgeResult (may be empty)
    unbridged: tuple              # BridgeResult.unbridged
    envelope: FloodplainEnvelope  # from Phase 5
    grid: SpatialGrid | None      # optional; usually set by zones_from_riverscape
    provenance: Mapping[str, Any] # ruleset versions, source product ids, counts
    degraded_reasons: tuple[str, ...]
```

`SpatialGrid` is an output-layer type. To avoid `riverscape` → `spatial`
imports, `LandformResult.grid` is typed as `Any | None` inside
`pipeline.py` (or a `Protocol`); `zones_from_riverscape` attaches a real
`SpatialGrid`.

### 3.2 build_landform

```python
def build_landform(
    domain: np.ndarray,
    frequency: np.ndarray,
    drainage: gpd.GeoDataFrame,
    reach_labels: np.ndarray,
    reach_keys: Mapping[int, str],
    upstr_darea: Mapping[int, float],
    *,
    geobox: Any,
    transform: Affine,
    pixel_m: float,
    cfg: RiverscapeConfig,
    years: tuple[int, int],
    load_dem: Callable[..., xr.DataArray] = load_dem,
    load_fc_percentiles: Callable[..., dict] = load_fc_percentiles,
    load_waterbodies: Callable[..., gpd.GeoDataFrame] = load_waterbodies,
) -> LandformResult:
    """Load sources and run corridor→…→riverine. Raise RiverscapeSourceUnavailable
    on loader failure. Never import hydrofragments.spatial.
    """
```

Injectable loaders enable monkeypatch tests without network.

**Required drainage columns** (validated up front): `HydroID`, `NextDownID`,
`UpstrDArea`, `geometry` (and any fields already required by corridor/REM/
bridging). Missing columns → `ValueError` (programming error), not mode
fallback.

### 3.3 zones_from_riverscape

```python
def zones_from_riverscape(
    stats: WoStatistics,
    *,
    drainage: gpd.GeoDataFrame,
    config: HydroConfig,
    reach_labels: np.ndarray,
    reach_keys: Mapping[int, str],
    upstr_darea: Mapping[int, float],
    geobox: Any,
    transform: Affine,
    pixel_m: float = 30.0,
    years: tuple[int, int] | None = None,
) -> ZoneResult:
    """Build riverscape ZoneResult from DEA stats + drainage context."""
```

If `years is None`, use `(min(stats.years), max(stats.years))` when available,
else raise `ValueError`.

### 3.4 Mode resolution (workflow)

Workflow implements the §6 table inline (or via a private helper). Inputs:
`mode`, `stats`, `drainage`, `config`, plus reach-label context when drainage
is present. Outputs: `ZoneResult | None`. No silent catch of unexpected
exceptions — only `RiverscapeSourceUnavailable` (and the no-drainage /
no-stats cases in §6) map to `auto` fallback.

## 4. Pipeline algorithm

Order (stop and raise or degrade only as specified):

1. **Validate** shapes: `domain`, `frequency`, `reach_labels` share 2-D shape;
   `domain` boolean; frequency finite where domain.
2. **Water seed** — `frequency >= 100.0 * cfg.f_seed` within domain (same
   convention as Phase 4 evidence).
3. **Load** DEM (`cfg.dem_product` / `cfg.dem_band`), FC
   (`cfg.fc_product`, bands `(cfg.bare_band, cfg.green_band, cfg.npv_band)`
   for `years`), waterbodies (`cfg.waterbodies_source`).
4. **Corridor** — `measure_corridor_widths(drainage, water_seed, …)`.
5. **Centreline** — `build_centreline` on the water-seed mask (not the full
   domain).
6. **REM** — `build_rem(dem, drainage, corridor widths, …)`.
7. **Waterbodies** — `classify_waterbodies(polygons, centreline.skeleton, …)`.
8. **Evidence** — Phase 4 `calibrate_bare_threshold` + `build_evidence` using
   loaded FC stacks, REM/trough, riverine waterbody mask, and centreline
   connectivity.
9. **Channel** — `classify_channel(...)` → observed channel.
10. **Bridge** — if `cfg.bridge_enabled`: `find_gaps` + `bridge_gaps`; else
    empty bridge result. Channel for landform = observed ∪ bridged.
11. **Riverine** — `build_landform_layer(rem, slope, domain, channel, …)`.
12. **Package** `LandformResult`:
    - `channel_source`: 1 on observed, 2 on bridged∖observed, else 0;
    - `channel_confidence`: channel confidence on channel pixels, else 255;
    - union all `degraded_reasons` from corridor/centreline/terrain/evidence/
      channel/bridge/envelope (stable sorted unique tuple);
    - `provenance`: ruleset versions (`CHANNEL_RULESET_VERSION`,
      `WATERBODY_RULESET_VERSION`, `RIVERINE_RULESET_VERSION`), pixel
      counts, bridge counts by `gap_cause`, unbridged counts by reason.

No silent catch inside `build_landform` for `RiverscapeSourceUnavailable` —
callers (`zones_from_riverscape` / workflow) decide mode behaviour.

## 5. zones_from_riverscape and hydroperiod

1. `domain = wet_domain(stats, min_valid_obs=config.validity.min_valid_obs)`
   using the same frequency array occurrence zoning reads from `stats`.
2. `landform_result = build_landform(...)` with reach context from the caller.
3. `hydroperiod = classify_hydroperiod(
       frequency, domain,
       t_persist=config.zones.t_persist, t_season=config.zones.t_season,
       unobserved_mask=landform_result.bridged_mask,
   )`.
4. `zone = combine_zones(
       landform_result.landform, hydroperiod.classes,
       degraded_reasons=landform_result.degraded_reasons,
   )`.
5. Attach grid from the frequency/stats DataArray via existing `_attach_grid`.

Bridged pixels are landform 1 and hydroperiod 4; legacy Zone 1 includes them.

## 6. Workflow mode behaviour

Inside `analyze_from_dea`, after DEA planning yields `stats` (may be `None`):

| Mode | Drainage | Stats / sources | Result |
|---|---|---|---|
| `off` | any | any | Today’s path: `zones_from_wo_statistics` if stats else `None` |
| `auto` | missing | — | Occurrence (if stats) + `degraded_reasons+=("riverscape_no_drainage",)` |
| `auto` | present | riverscape succeeds | `zones_from_riverscape` |
| `auto` | present | `RiverscapeSourceUnavailable` or other declared source failure | Occurrence + reason string (e.g. `riverscape_source_unavailable`) |
| `required` | missing | — | Raise (`RiverscapeSourceUnavailable` or `ValueError` with clear message) |
| `required` | present | source failure | Propagate |
| `required` | present | success | `zones_from_riverscape` |

If `stats is None` and mode needs frequency/domain: `auto` → `zone_result=None`
with degrade reason `riverscape_no_stats` (metrics may still run); `required`
→ raise. Occurrence path when falling back still requires stats as today.

Drainage validation for the riverscape path runs **before** riverscape
loaders when mode ≠ `off` and drainage is present (fail fast on bad schema).
Parent §7’s “validate drainage right after AOI load” is included: when
`drainage is not None`, validate CRS/columns early even if mode is `off`
(channel metrics already need it).

`timings["riverscape"]` records wall time of the riverscape branch only
(zero/absent when skipped).

## 7. Testing

### 7.1 Unit / kernel orchestration

`tests/riverscape/test_riverscape_pipeline.py`:

- Happy path with stub loaders returning tiny arrays → landform codes present,
  `channel_source` distinguishes observed vs bridged when bridge stub paints.
- Loader raises `RiverscapeSourceUnavailable` → propagates from
  `build_landform`.
- Shape mismatch → `ValueError`.
- `bridge_enabled=False` → no bridged pixels; channel_source never 2.

### 7.2 zones_from_riverscape

`tests/spatial/test_zones_from_riverscape.py`:

- Synthetic landform+frequency → crosstab and legacy mask; bridged →
  hydroperiod 4 and Zone 1.
- Degraded reasons forwarded onto `ZoneResult`.

### 7.3 Workflow modes

`tests/workflow/test_riverscape_modes.py` (or extend existing DEA workflow
tests):

- `off` → occurrence (monkeypatch stats path).
- `auto` + no drainage → occurrence + `riverscape_no_drainage`.
- `auto` + drainage + stubbed success → `mode=="riverscape"`.
- `auto` + drainage + stubbed `RiverscapeSourceUnavailable` → occurrence +
  reason.
- `required` + no drainage → raises.
- `required` + source failure → raises.

Mutants (discriminating):

1. `auto` without drainage skips degrade reason.
2. `required` without drainage falls back instead of raising.
3. Bridged mask not passed to `classify_hydroperiod` (no code 4).
4. `channel_source` paints bridged as observed (1).
5. Catch `RiverscapeSourceUnavailable` inside `build_landform` (silent).

## 8. File touch list

| Path | Action |
|---|---|
| `hydrofragments/riverscape/pipeline.py` | Create |
| `hydrofragments/spatial/zones.py` | Add `zones_from_riverscape`; export |
| `hydrofragments/workflow.py` | Mode branch + early drainage validate + timing |
| `tests/riverscape/test_riverscape_pipeline.py` | Create |
| `tests/spatial/test_zones_from_riverscape.py` | Create |
| `tests/workflow/test_riverscape_modes.py` | Create |
| `.superpowers/sdd/progress.md` | Ledger (local/gitignored) |

## 9. Relationship to 6b

6b consumes `LandformResult` / `ZoneResult` fields already populated here and
writes them into manifest, rasters, and vectors. 6a must not invent alternate
export DTOs; provenance keys should be stable strings 6b can copy.
