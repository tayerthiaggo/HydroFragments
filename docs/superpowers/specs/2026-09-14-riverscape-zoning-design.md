# Specification: Riverscape Zoning — Landform Layer × Hydroperiod Layer

**Date:** 2026-09-14
**Status:** Approved design (brainstorming gate passed). Implementation plan to follow.
**Package version:** targets HydroFragments `0.1.0` (post-`0.1.0` work)

This document replaces the pure-threshold zone definition in
`hydrofragments/spatial/zones.py` with two independent layers — a **landform**
layer (where is the channel, where is off-channel riverine water) and a
**hydroperiod** layer (how often is it wet) — and derives the four zones from
their combination.

---

## 1. Origin and problem

The four zones HydroFragments exports are:

1. in-channel water
2. persistent off-channel waterbodies
3. seasonally flooded floodplain
4. marginal or extreme-event floodplain

Today all four are thresholds on one layer, the DEA WO multi-year frequency
(`build_zones`, `zones.py:57-117`; `t_season=0.10`, `t_persist=0.50`,
`config.py:96-98`). Verified problems:

- **Zone 1 is never produced.** `workflow.py:347` calls
  `zones_from_wo_statistics` without `drainage_mask`, so only Zones 2–4 are
  emitted in production.
- **Channel is defined by wetness.** A channel pixel that is wet only
  occasionally becomes "marginal floodplain"; a persistent in-channel pool
  becomes "persistent off-channel". Channel identity is a landform property,
  not a wetness property.
- **Zone 2 has no off-channel test.** The only channel separation in code is
  a one-pixel dilation of the drainage raster.
- **Names disagree.** Plans say in-channel / persistent off-channel /
  seasonal / marginal; `output/finalize.py:44-49` exports
  `channel_connected` / `persistent` / `seasonal` / `ephemeral`.
- **Vector drainage is misaligned with EO.** AHGF lines
  (`SrcFCName=DEMDerivedStreamsVector`) come from a coarse DEM and do not
  follow the imaged channel. Current manager practice — fixed buffers per
  Strahler order fitted by manual regression — inherits that misalignment and
  the known weakness of width scaling rules (Hou et al. 2019 show width vs
  area/gradient rules capture little real variability; Lin et al. 2020 find
  30–40%).

Scope: Australia-wide method; Fitzroy (Kimberley) is the first test catchment.
Flat, low-gradient catchments must work.

---

## 2. What already exists (reuse, do not rebuild)

| Capability | Location |
|---|---|
| DEA WO statistics adapter (`frequency`, `count_wet`, `count_clear`, EPSG:3577 30 m, Dask) | `hydrofragments/io/dea.py` (`WoStatistics`) |
| Occurrence-only zoning (kept as fallback) | `hydrofragments/spatial/zones.py` (`build_zones`, `zones_from_wo_statistics`) |
| Drainage topology checks, AOI clipping, ordered reach paths | `hydrofragments/spatial/context.py` (`create_channel_context`, `ordered_reach_paths`) |
| Water skeleton via `medial_axis`, reach label raster | `hydrofragments/spatial/connectivity_context.py` |
| Grid contract | `hydrofragments/output/spatial.py` (`SpatialGrid`) |
| Zone polygon / raster export | `output/finalize.py:163-197`, `output/rasters.py:81-86, 890-905` |
| Zone provenance digest | `output/manifest.py` (`build_dea_provenance`) |
| Zones-do-not-multiply-metrics gate | `tests/gating/test_zones_do_not_multiply_metrics.py` |
| Persistence-by-zone guard | `guards/scientific.py:24-63` |

---

## 3. Architecture

### 3.1 Analysis domain

The analysis is anchored on **observed-wet pixels** of the DEA WO statistics:

```
domain = (count_wet > 0) & isfinite(frequency) & (count_clear >= min_valid_obs)
```

This is the native 30 m wet domain `build_zones` already uses. It is **not**
hydroseason's coarsened planning footprint / `analysis_mask`, which
`hydroseason/_io_dea_stats.py:655-661` forbids feeding into zoning.

One controlled exception: **channel gap bridges** (Section 4.2 step 5). A river
channel is continuous, but WOfS breaks it where water sits under riparian
canopy, where the channel is narrower than a Landsat pixel can resolve, or
where water is spectrally unusual. Bridges may place in-channel pixels outside
the domain, but **only between two observed channel segments on the same AHGF
reach path** — both ends are anchored by EO. No channel is invented upstream of
the last observed segment or on reaches with no observed water.

Everything else outside the domain is `0` in every output layer.

**Accepted limitation.** Reaches with no observed water at all, dangling
channel ends beyond the last observed segment, and off-channel extreme floods
Landsat missed (cloud at peak, water under canopy — Lymburner et al. 2024) are
not zoned. Unbridged reaches are counted and flagged in provenance.

### 3.2 Layer 1 — landform (`hydrofragments/riverscape/`)

Answers *where*. Within the domain:

| Code | Class | Meaning |
|---|---|---|
| 0 | outside | not in domain |
| 1 | `in_channel` | part of the river channel, regardless of wetness — observed or bridged |
| 2 | `off_channel_riverine` | floodplain / off-channel water inside the valley bottom |
| 3 | `non_riverine` | water outside the valley bottom (farm dams, isolated lakes) |

A companion `channel_source` raster records how each in-channel pixel was
obtained: `1 observed` (evidence rules on domain pixels), `2 bridged` (gap
bridge). Built from fused evidence (Section 4). Knows nothing about
hydroperiod classes.

### 3.3 Layer 2 — hydroperiod (`hydrofragments/hydroperiod/`)

Answers *how often wet*. Within the domain, using `config.zones.t_persist` and
`config.zones.t_season` (reused, not duplicated):

| Code | Class | Rule (frequency, percent) |
|---|---|---|
| 0 | outside | not in domain |
| 1 | `persistent` | `freq > 100·t_persist` |
| 2 | `seasonal` | `100·t_season <= freq <= 100·t_persist` |
| 3 | `marginal` | `freq < 100·t_season` (and `> 0` by domain) |
| 4 | `unobserved` | outside the domain but supplied as a landform-1 pixel (bridged channel) |

Boundaries match `build_zones` exactly (both inclusive for seasonal).
The classifier computes codes 1–3 from frequency on the domain only; code 4 is
never inferred from neighbours — the pixel was not observed wet, and the layer
says so. The classifier accepts an optional `unobserved_mask` for code 4 and
otherwise never reads landform. Pure function; must not import
`hydrofragments.riverscape`. Future
classifiers (DEA WO seasonal summaries `ga_ls_wo_fq_nov_mar_3` /
`ga_ls_wo_fq_apr_oct_3`, hydroseason `end_dry` snapshots) replace this layer
without touching Layer 1.

### 3.4 Zones — derived, no new logic

`combine_zones(landform, hydroperiod)` produces:

- **Cross-tab** `zone_crosstab = landform*10 + hydroperiod`
  (e.g. `11` in-channel persistent = refuge pool; `13` in-channel marginal =
  rarely-wet channel; `14` in-channel unobserved = bridged gap under canopy or
  sub-pixel channel; `21` off-channel persistent = billabong).
- **Legacy four-zone view:**

| Legacy zone | Rule | Export name |
|---|---|---|
| 1 | landform 1, any hydroperiod (incl. unobserved bridges) | `in_channel` |
| 2 | landform 2 & hydroperiod 1 | `persistent_off_channel` |
| 3 | landform 2 & hydroperiod 2 | `seasonal_floodplain` |
| 4 | landform 2 & hydroperiod 3 | `marginal_floodplain` |
| 0 | outside domain, or landform 3 | — |

Non-riverine pixels are 0 in the legacy view; their count and area are
recorded in provenance.

`emitted_zones`/`has_zone_1` on the returned `ZoneResult` are **derived from
the mask actually produced**, matching occurrence mode's convention
(`build_zones` reports `(2,3,4)`/`False` with no drainage): `emitted_zones =
tuple(sorted(unique nonzero legacy zone codes present))`, `has_zone_1 = 1 in
mask`. A degraded `auto` run with zero in-channel pixels must not advertise
Zone 1. (Superseded decision, recorded 2026-09-14: `combine_zones`'s first
implementation hardcoded `(1,2,3,4)`/`True` unconditionally; that shipped in
Plan 1's Task 4 with a test pinning the hardcoded values. Plan 2 corrects
this and updates that test.)

This framing follows the landform × hydroperiod classification of
Semeniuk & Semeniuk (1995).

---

## 4. Layer 1 method

### 4.1 Evidence sources

All loaded onto the `WoStatistics` grid (EPSG:3577, 30 m) with
`odc.stac.load(like=stats.frequency.odc.geobox)`; `SpatialGrid` equality is
asserted and a mismatch raises.

| Family | Source | Use |
|---|---|---|
| water | DEA WO statistics `ga_ls_wo_fq_myear_3` | seeds, W evidence |
| water | DEA Waterbodies v3 polygons | S evidence, off-channel objects |
| bare | DEA Fractional Cover percentiles `ga_ls_fc_pc_cyear_3` (bare-soil percentile band) | B evidence: persistently bare channel bed |
| riparian | same FC percentiles product, photosynthetic-vegetation percentile band | V evidence: persistent dry-season green corridor; bridge cost and gap-cause label only |
| riparian (optional) | same FC percentiles product, non-photosynthetic-vegetation percentile band | bridge cost only (weight 0 by default) |
| terrain | SRTM 1 s `ga_srtm_dem1sv1_0`, band `dem_s` or `dem_h` (Phase 0 decides) | REM, trough, slope |
| topology | AHGF drainage lines | reach IDs, downstream order, `UpstrDArea`, search corridor |

Rules:

- **Evidence is counted by family, not by layer.** DEA Waterbodies is derived
  from WOfS, so W and S together count once.
- **AHGF geometry never enters the result.** Lines define a search corridor
  and topology only; they are never buffered into a polygon.
- **`dem_h` caution.** DEM-H is hydrologically enforced (streams burned at
  mapped positions), so its troughs follow the AHGF lines being corrected.
  Phase 0 measures the trough-to-EO-skeleton offset for `dem_s` and `dem_h`
  and picks the default. The band stays configurable.
- **Canopy bias.** SRTM measures riparian canopy tops; along-stream elevation
  uses a low percentile, not the centreline pixel.
- **FC and water.** FC percentiles exclude water observations; a missing FC
  value on a persistently wet pixel is *unknown*, not *not bare*.
- **Bare confusion.** Scalds, salt pans, tracks, burnt ground and sandy
  floodplain look bare. B is only admissible together with T or adjacency to
  accepted channel, and must be stable across years.

### 4.2 Pipeline

1. **Corridor.** Per reach, measure the p95 distance from the AHGF line to
   the EO water skeleton (`freq >= f_seed`). Corridor width =
   `clamp(p95 + max_half_width, corridor_min_m, corridor_max_m)`, pooled per
   run and recorded in provenance. The corridor is a search space only.
2. **Centreline (conflation).** Skeleton (`medial_axis`) of the water seed
   inside the corridor. Gaps in the skeleton are left open here and resolved
   by gap bridging (step 5); where a reach has no seed at all, the AHGF line
   is used only as a topology/profile guide, flagged `line_fallback`, and never
   becomes channel. Skeletons with loops flag the reach `multithread`.
3. **Terrain (Relative Elevation Model).** Chosen over D8 HAND because flow
   routing on 1 s SRTM fails in flat terrain.
   - Sample the `profile_percentile` (p10) elevation per `profile_bin_m`
     along-stream bin.
   - Enforce a non-increasing downstream profile by isotonic regression
     (pool-adjacent-violators) in AHGF topological order
     (`ordered_reach_paths`), capping each reach's upstream end at its
     tributaries' minimum outflow. A running minimum is rejected because it
     propagates single void pits downstream.
   - Spread with `scipy.spatial.cKDTree` k-NN inverse-distance weighting
     (`rem_k`, within `rem_max_distance_m`), queried per chunk.
   - `REM = dem − interpolated profile`; also local trough depth
     (`trough_radius_m`) and slope.
4. **Channel rules (landform 1).** Per-pixel evidence bits within the domain:
   - `W_high`: `freq >= f_chan_high`
   - `S`: inside a riverine DEA Waterbodies polygon
   - `B`: bare percentile `>= bare_threshold(run)` in `>= bare_year_fraction`
     of years — `bare_threshold` is **calibrated per run**, not a fixed
     constant, mirroring the envelope's own per-run calibration in step 7:
     fit from the run's own bare-vs-background contrast (e.g. a quantile of
     the bare-percentile distribution on channel-connected wet/trough pixels
     vs. the AOI background), because a fixed default cannot travel between
     catchments — Fitzroy's Phase 0 spike measured `bs_pc_50` at 20.5 AOI-wide
     vs. 31.5 on the water seed, both far under a fixed 50. `bare_threshold_pct`
     stays in config only as the calibration's floor/fallback when too few
     pixels are available to calibrate (mirroring `envelope_min_bin_pixels`'s
     degraded path), recorded as degraded when used.
   - `T`: `REM <= h_chan_m` or trough depth `>= trough_depth_m`
   - `C`: connected to the centreline

   Growth starts at the centreline and is capped per reach at
   `width_growth_factor ×` the reach's median seed half-width, so billabongs
   joined by thin low-frequency links are not captured.
   Ordered, named rules with a `RULESET_VERSION`, first match wins, e.g.
   `C & (W_high | S)`, `C & B & T`, `C & (B | T) & adjacent_to_accepted`.
   `channel_confidence` = number of agreeing families (0–3); a pixel is
   in-channel when confidence `>= min_channel_confidence` (default 2).
   Line-fallback-only support gets confidence 1 and is not in-channel by
   default.
5. **Channel gap bridging (landform 1, `channel_source = bridged`).** Follows
   established practice for reconnecting fragmented river masks: least-cost
   path search over terrain and water evidence (Wang et al. 2024; Chen et al.
   2020), overlaying EO masks with DEM-derived networks to close gaps (Yi et
   al. 2025), DEM-derived streams for sub-pixel rivers (Wortmann et al. 2025),
   and DEM-guided growth of water under vegetation from observed seeds (Rossi
   et al. 2025).
   - **Candidates.** For every AHGF reach path, order the observed channel
     segments along flow. A gap is the space between the downstream end of one
     observed segment and the upstream end of the next on the same path
     (including across a reach junction). Gaps with only one anchor are never
     bridged.
   - **Path.** `skimage.graph.MCP_Geometric` inside the corridor, from the
     upstream anchor endpoint to the downstream anchor endpoint. Cost per pixel
     (all terms configurable weights, lower = more channel-like): relative
     elevation / trough depth (T), inverse WOfS frequency where `freq > 0`,
     bare (B), riparian green (V), optional non-photosynthetic vegetation
     (dry litter / dead wood in channel beds, FC `npv` percentile; weight 0
     unless the Fitzroy calibration shows benefit), and distance to the AHGF
     line. Pixels above
     `bridge_rem_max_m` are impassable.
   - **Width.** Half-width linearly interpolated along the path between the
     anchor segments' distance-transform half-widths at their endpoints;
     painted by disc along the path, clipped to the corridor, never overwriting
     observed channel or landform 2/3 domain pixels with stronger evidence.
   - **Limits.** Bridge length `<= bridge_max_length_m`; path cost per metre
     `<= bridge_max_cost_per_m`; otherwise the gap stays open and is flagged
     `gap_unbridged` with its reason.
   - **Diagnostics (per bridge, exported as a `channel_bridges` vector layer).**
     `reach_id`, length, mean cost, anchor widths, `bridge_confidence` (1 low /
     2 medium, decreasing with length and cost), and `gap_cause`:
     `vegetated` (median V along path `>= riparian_green_pct`), `narrow`
     (both anchor widths `<= narrow_width_px` pixels), else `unobserved`.
   - Bridged pixels get hydroperiod `unobserved` (code 4).
6. **DEA Waterbodies roles.** Merge fragments cut at tile edges; classify
   each polygon `riverine` or `off_channel` by centreline overlap and
   elongation. Riverine polygons supply S; off-channel polygons give object
   consistency to landform 2.
7. **Riverine vs non-riverine (landform 2 vs 3).** For domain pixels not in
   the channel: riverine when `REM <= h_fp(A)` and `slope <= slope_max_deg`,
   else non-riverine.
   - `A` = AHGF `UpstrDArea` (m²) of the nearest reach.
   - `h_fp(A) = a·A^b`, fitted robustly in log space to per-log-A-bin REM
     quantiles (`envelope_quantile`) of channel-connected wet pixels, with at
     least `envelope_min_bin_pixels` per bin.
   - Fewer than two usable bins → `b = 0`, recorded as degraded.
   - Capped at `envelope_h_max_m`. Per-run calibration rather than global
     constants, because hydrogeomorphic scaling is unreliable in semi-arid,
     low-gradient basins (Annis et al. 2021).
   - MrVBF (Gallant & Dowling 2003) is added only if the Fitzroy check shows
     hillslope dams leaking into landform 2.
8. **Scale.** Windowed per sub-catchment group with overlap
   `>= corridor + rem_max_distance_m`, stitched by core ownership.
   `_build_reach_label_raster` (`connectivity_context.py:96-108`) is
   vectorised.

### 4.3 Modes and failure behaviour

`riverscape.mode`:

| Mode | Behaviour |
|---|---|
| `off` | occurrence zoning (`build_zones`) as today |
| `auto` (default) | riverscape zoning; if a source is unavailable or drainage is absent, fall back to occurrence zoning and record `degraded_reasons` |
| `required` | riverscape zoning or raise `RiverscapeSourceUnavailable` |

No silent fallbacks: every fallback and every degraded calibration is
recorded in `ZoneResult.degraded_reasons` and the manifest.

---

## 5. Code layout and interfaces

- `hydrofragments/io/riverscape_sources.py` — `load_dem(geobox, *, product,
  band)`, `load_fc_percentiles(geobox, *, product, bands, years)` (bare and
  photosynthetic-vegetation bands),
  `load_waterbodies(bounds, crs, *, source)`, `RiverscapeSourceUnavailable`.
  This is a deliberate exception to "hydroseason owns STAC access";
  boundary docstrings in `io/dea.py:1-16` and `workflow.py:1-10` are updated,
  and `pystac-client` is declared as a direct dependency.
- `hydrofragments/riverscape/`
  - `domain.py` — `wet_domain(stats, *, min_valid_obs) -> np.ndarray[bool]`
    (shared by both layers)
  - `evidence.py` — `RiverscapeEvidence` (frozen)
  - `corridor.py`, `centreline.py`, `terrain.py`, `waterbodies.py`,
    `channel.py`, `riverine.py`, `windows.py`
  - `bridging.py` — `find_gaps(channel, centreline, drainage) -> list[Gap]`,
    `bridge_gaps(gaps, cost_inputs, cfg) -> BridgeResult` (bridged mask,
    `channel_bridges` GeoDataFrame, unbridged gaps with reasons)
  - `pipeline.py` — `build_landform(evidence, domain, cfg) -> LandformResult`
    (`landform`, `channel_source`, `channel_confidence`, `rule_id`, `rem`,
    `bridges`, `grid`, `provenance`, `degraded_reasons`)
- `hydrofragments/hydroperiod/classify.py` —
  `classify_hydroperiod(frequency, domain, *, t_persist, t_season,
  unobserved_mask=None) -> HydroperiodResult` (`classes`,
  `method="wofs_multiyear"`, thresholds).
- `hydrofragments/spatial/zones.py` — `build_zones` unchanged; new
  `combine_zones(landform, hydroperiod) -> ZoneResult` and
  `zones_from_riverscape(stats, landform, *, config)`. `ZoneResult` gains
  `mode: Literal["occurrence", "riverscape"]`, `crosstab`, and
  `degraded_reasons: tuple[str, ...]`.

---

## 6. Configuration

New `RiverscapeConfig` beside `ZonesConfig`; parsed after `config.py:527`,
added to `_TOP_LEVEL_KEYS` and `scientific_config`;
`SCIENTIFIC_HASH_SCHEMA_VERSION` bumped to `1.2.0`. Hydroperiod reuses
`ZonesConfig.t_persist` / `t_season`.

| Field | Default |
|---|---|
| `mode` | `auto` |
| `dem_product` / `dem_band` | `ga_srtm_dem1sv1_0` / set by Phase 0 |
| `fc_product` / `bare_band` / `green_band` | `ga_ls_fc_pc_cyear_3` / confirmed in Phase 0 |
| `waterbodies_source` | `None` (Phase 0 picks WFS bbox or national file) |
| `f_seed`, `f_chan_high` | 0.05, 0.10 |
| `bare_threshold_floor_pct`, `bare_year_fraction` | 30, 0.6 (shipped; renamed from `bare_threshold_pct` and changed from 50 -- see the findings doc's basin-scale rerun section, "Implications for Plan 3" bare_threshold bullet) |
| `trough_radius_m`, `trough_depth_m`, `h_chan_m` | 150, 0.5, 2.0 |
| `corridor_min_m`, `corridor_max_m`, `alignment_quantile` | 90, 1200, 0.95 (`corridor_max_m` changed from 600 -- see the findings doc's basin-scale rerun section, "AHGF alignment (basin-wide)" and "Implications for Plan 3") |
| `width_growth_factor` | 3.0 |
| `profile_bin_m`, `profile_percentile`, `rem_k`, `rem_max_distance_m` | 300, 10, 8, 5000 |
| `envelope_quantile`, `envelope_h_max_m`, `envelope_min_bin_pixels`, `slope_max_deg` | 0.95, 15, 200, 2.0 |
| `min_channel_confidence`, `include_line_fallback_in_channel` | 2, False |
| `bridge_enabled`, `bridge_max_length_m`, `bridge_max_cost_per_m`, `bridge_rem_max_m` | True, 2000, set in Phase 3 calibration, 5.0 |
| `bridge_cost_weights` (terrain, water, bare, green, npv, line_distance) | 1.0, 1.0, 0.5, 1.0, 0.0, 0.5 |
| `riparian_green_pct`, `narrow_width_px` | 40, 2 |

Validation: fractions in `[0, 1]`; `f_seed <= f_chan_high`; distances
positive and finite; cost weights non-negative with at least one positive; `corridor_min_m < corridor_max_m`; percentile in
`(0, 100)`; quantile in `(0, 1)`; `auto`/`required` need a DEM source.

All defaults are provisional until the Phase 7 validation report; each
change after validation must cite the report.

---

## 7. Integration

- **`workflow.py`** — load and validate drainage right after AOI load
  (line 333) so bad drainage fails before acquisition; add
  `timings["riverscape"]` after DEA planning (342-347); `_channel_inputs`
  (268-293) takes the loaded GeoDataFrame.
- **`output/manifest.py`** — `zoning` section: `mode`, domain pixel count and
  digest, `degraded_reasons`, `ruleset_version`, sources (product, band,
  item ids), alignment estimate, envelope calibration, rule counts,
  non-riverine area, bridged length and area by `gap_cause`, unbridged gap
  count by reason, reaches with no observed water.
- **`output/finalize.py`** — zone names keyed by mode (riverscape names in
  3.4; occurrence keeps current names because meanings differ); zones GPKG
  gains `mode`, `landform`, `hydroperiod` columns. Update
  `docs/spatial_exports.md:179`.
- **`output/rasters.py`** + new spatial product `riverscape_evidence`:
  `landform`, `hydroperiod`, `zone_crosstab`, `channel_source` (uint8),
  `channel_confidence` (uint8, 255 nodata), `rem` (float32, m); vector layer
  `channel_bridges` (LineString path + attributes from Section 4.2 step 5).
- **`guards/scientific.py`** — unchanged; persistence metrics by zone remain
  refused (landform 1 still uses water evidence).
- Zones still never split metrics; the gating test is extended to riverscape
  mode.

---

## 8. Testing

Synthetic fixtures (`tests/fixtures/riverscape_synthetic.py`):

- AHGF line offset 90 m from the true channel still yields the EO centreline,
  and no line pixels enter the result.
- Rarely-wet (≈1%) sand bed with B and T → in-channel.
- Billabong joined by a one-pixel low-frequency link → off-channel riverine.
- Bare scald off the trough → not channel.
- Hillslope dam → non-riverine; flat-plain billabong → riverine.
- Two-thread anabranch → sensible REM; isotonic profile resists a void pit.
- Single drainage-area bin → `b = 0` and degraded flag.
- Misaligned evidence grids raise.
- W + S count as one family.
- Channel broken by a 300 m vegetated gap (never wet, high green, trough
  present) between two observed segments → bridged, `gap_cause=vegetated`,
  width interpolated, hydroperiod `unobserved`, legacy Zone 1.
- Sub-pixel reach (anchor widths 1 px) → bridged, `gap_cause=narrow`.
- Dangling end (observed segment with no downstream observed anchor) → not
  extended; reach with no observed water → no channel, flagged.
- Gap longer than `bridge_max_length_m` or path over an impassable ridge →
  `gap_unbridged` with reason; no pixels painted.
- Bridge never overwrites observed off-channel domain pixels (a billabong
  beside the gap stays landform 2).
- Hydroperiod code 4 only via `unobserved_mask`; never inferred from
  frequency or neighbours.
- Domain excludes `count_wet == 0` and low support and never reads the
  planning footprint.
- Hydroperiod boundaries 9.9 / 10 / 50 / 50.1; cross-tab ↔ legacy mapping
  table; `hydroperiod` does not import `riverscape`.

Existing `build_zones` tests stay unchanged. Integration tests monkeypatch
the loaders in the `test_dea_workflow.py` style and cover `off`, `auto`
(including fallback) and `required`. A slow Fitzroy validation script
reports results (Section 9).

---

## 9. Validation

| Reference | Independence from WOfS | Use |
|---|---|---|
| Manager polygons (to be requested) | independent | IoU, area difference, boundary offset per reach |
| DEA Waterbodies | not independent | riverine polygons should fall in landform 1, off-channel in legacy Zones 2/3 |
| Hou et al. 2019 reach widths (Geofabric v2, WOfS 1987–2014) | not independent | channel area / reach length vs published width |

Only manager polygons provide independent accuracy. DEA Waterbodies and Hou
widths are consistency checks.

---

## 10. Phases

| # | Phase | Acceptance |
|---|---|---|
| 0 | This spec; data-access spike on Fitzroy (narrow AOI) | DEM bands/CRS, FC bands, DEA Waterbodies route confirmed; AHGF offset measured; `dem_s` vs `dem_h` decided; aligned layers pass grid equality |
| 1 | `wet_domain`, `classify_hydroperiod`, `combine_zones` | domain, boundary, mapping and import-boundary tests |
| 2 | AOI widen to full Fitzroy basin + Phase 0 rerun; `combine_zones` mask-derived `emitted_zones`/`has_zone_1`; `RiverscapeConfig` skeleton + hash bump | widened findings doc with real `UpstrDArea`/AHGF-offset/FC distributions; config round-trip; updated zone-combination test |
| 3 | Loaders (`io/riverscape_sources.py`), corridor, centreline, REM | offset, anabranch and pit tests, defaults set from Phase 2's widened numbers |
| 4 | Channel rules, waterbodies, gap bridging | sand-bed, billabong, scald, family tests; vegetated/narrow/dangling/unbridged/no-overwrite bridge tests; Fitzroy bridge-cost calibration recorded |
| 5 | Riverine vs non-riverine | dam, billabong, single-bin tests |
| 6 | Workflow, exports, manifest | integration tests in all modes; gating test extended |
| 7 | Windowed execution | full Fitzroy within memory budget; no seam errors |
| 8 | Validation harness | report produced; manager-polygon slot ready |

(Phases renumbered 2026-09-14 to insert the AOI-widen prep phase; see §12.)

---

## 11. Deferred

- Seasonal hydroperiod (DEA WO Nov–Mar / Apr–Oct summaries) and hydroseason
  `end_dry` low-water snapshots — Layer 2 only.
- MrVBF — only if hillslope leakage is observed.
- Under-canopy off-channel inundation (Lymburner et al. 2024) and
  never-observed floodplain — outside the observed-wet domain by decision.
- Channel extension beyond the last observed segment (dangling ends) and full
  network completion for reaches with no observed water — rejected for now;
  flagged in provenance.

## 12. Carried into Plan 2 (Plan 1 final-review findings)

Plan 1's whole-branch review (commit `e1e4cf8`, all 5 tasks approved) raised
five findings that are correct but out of Plan 1's scope. Three are now
resolved by decision (below); the import-cycle guard was fixed immediately
(`c9e3bf0`). Two remain open as Plan 2 inputs:

**Resolved by decision (2026-09-14):**
- `emitted_zones`/`has_zone_1` semantics — resolved: derive from the mask in
  riverscape mode too (§3.4, above). Plan 2 corrects `combine_zones` and
  updates the pinned test in `tests/spatial/test_zone_combination.py`.
- `bare_threshold_pct` unsupported by measured FC values — resolved:
  per-run calibration (§4.2 step 4, above), config value becomes a
  degraded-path floor, not the primary threshold.
- Fitzroy Phase 0 AOI is mainstem-only, degenerate `UpstrDArea` — resolved:
  **widen now**, to the full Fitzroy River basin (~94,000 km², headwaters to
  mouth), before Plan 2 calibrates the envelope or the bare threshold.
  Source: query DEA/Geofabric WFS for the wider AHGF network + basin
  boundary (same access pattern as the Phase 0 DEA Waterbodies WFS query),
  replacing `data/fitzroy_kimberley_aoi.geojson` and
  `data/fitzroy_kimberley_drainage.gpkg`. Plan 2's first task re-runs a
  Phase-0-style check (grid alignment, `UpstrDArea` spread, AHGF/EO offset,
  bare-vs-background contrast) against the widened data and updates
  `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md`
  accordingly — the current findings doc's DEM-band decision and access
  routes should still hold (they don't depend on AOI size), but every
  distribution-based number (AHGF offset percentiles, `UpstrDArea`
  quantiles, FC medians) must be re-measured on the wider extent, not
  patched by inference.

**Still open, Plan 2 inputs:**
- **No test exercises the full call chain** `WoStatistics → wet_domain →
  classify_hydroperiod → combine_zones` — the exact sequence
  `analyze_from_dea` will use — including the bridged case where the
  landform extent is `domain ∪ bridges` and `combine_zones`'s
  extent-agreement check (§3.1, §3.4) is load-bearing. Plan 2's first task
  should add one synthetic end-to-end test on a small grid (dask-backed
  stats, an `unobserved_mask` disjoint from the domain, a hand-built
  landform array) before wiring in real loaders.
- **A riverscape `ZoneResult` is not yet exportable**: `combine_zones`
  leaves `grid=None`, and `ZoneResult.as_dataarray()` only serializes
  `mask`, never `crosstab`. Expected at this stage (§7 assigns grid
  attachment and crosstab export to `zones_from_riverscape` / Plan 4), noted
  here only so it isn't mistaken for an oversight when Plan 2 starts
  building the loaders that feed it.

Minor items worth folding into whichever Plan 2 task next touches the
relevant file (not worth a standalone task): `combine_zones`'s
`degraded_reasons` parameter accepts a bare `str` and silently iterates it
into single-character tuple entries — reject non-tuple/list input
explicitly; `ZoneResult.__post_init__` allows `mode="occurrence"` with a
non-`None` `crosstab` (the reverse case is correctly rejected) — reject
that combination too; `HydroperiodResult` has no `__post_init__`
validation, unlike `ZoneResult`, an asymmetry worth resolving if the two
types are meant to be handled uniformly.
- Split of in-channel by hydroperiod into separate legacy zones — available
  through the cross-tab instead.

---

## References

- Annis, A. et al. (2021). On the influence of river basin morphology and climate on hydrogeomorphic floodplain delineations. *Advances in Water Resources*. https://doi.org/10.1016/j.advwatres.2021.104078
- Bozzolan, E. et al. (2026). Enhancing active channel delineation in alluvial rivers using monthly aggregation of Sentinel-2 imagery. *Earth and Space Science*. https://doi.org/10.1029/2025ea004642
- Chen, H. et al. (2020). Extraction of connected river networks from multi-temporal remote sensing imagery using a path tracking technique. *Remote Sensing of Environment*. https://doi.org/10.1016/j.rse.2020.111868
- Chen, Q. et al. (2024). Extracting an accurate river network: Stream burning re-revisited. *Remote Sensing of Environment*. https://doi.org/10.1016/j.rse.2024.114333
- Crivellaro, M. et al. (2024). Characterization of active riverbed spatiotemporal dynamics through the definition of a framework for remote sensing procedures. *Remote Sensing*, 16(1), 184. https://doi.org/10.3390/rs16010184
- Gallant, J. C., & Dowling, T. I. (2003). A multiresolution index of valley bottom flatness for mapping depositional areas. *Water Resources Research*. https://doi.org/10.1029/2002wr001426
- Hou, J., van Dijk, A. I. J. M., & Renzullo, L. J. (2019). Hydromorphological attributes for all Australian river reaches derived from Landsat dynamic inundation remote sensing. *Earth System Science Data*, 11, 1003–1015. https://doi.org/10.5194/essd-11-1003-2019
- Krause, C. E. et al. (2021). Mapping and monitoring the multi-decadal dynamics of Australia's open waterbodies using Landsat. *Remote Sensing*, 13(8), 1437. https://doi.org/10.3390/rs13081437
- Lin, P. et al. (2020). Global estimates of reach-level bankfull river width leveraging big data geospatial analysis. *Geophysical Research Letters*. https://doi.org/10.1029/2019gl086405
- Lymburner, L. et al. (2024). Seeing the floods through the trees: Using adaptive shortwave infrared thresholds to map inundation under wooded wetlands. *Hydrological Processes*. https://doi.org/10.1002/hyp.15174
- Mueller, N. et al. (2016). Water observations from space: Mapping surface water from 25 years of Landsat imagery across Australia. *Remote Sensing of Environment*. https://doi.org/10.1016/j.rse.2015.11.003
- Nardi, F. et al. (2019). GFPLAIN250m, a global high-resolution dataset of Earth's floodplains. *Scientific Data*. https://doi.org/10.1038/sdata.2018.309
- Rossi, M. et al. (2025). Enhancing inundation mapping with geomorphological segmentation: Filling in gaps in spectral observations. *Science of the Total Environment*. https://doi.org/10.1016/j.scitotenv.2025.180180
- Semeniuk, C. A., & Semeniuk, V. (1995). A geomorphic approach to global classification for inland wetlands. *Vegetatio*. https://doi.org/10.1007/bf00045193
- Wang, Z. et al. (2021). Basin-scale high-resolution extraction of drainage networks using 10-m Sentinel-2 imagery. *Remote Sensing of Environment*. https://doi.org/10.1016/j.rse.2020.112281
- Wang, N. et al. (2024). Delineation of intermittent rivers and ephemeral streams using a hybrid method. *Remote Sensing*, 16(13), 2489. https://doi.org/10.3390/rs16132489
- Wortmann, M. et al. (2025). Global River Topology (GRIT): A bifurcating river hydrography. *Water Resources Research*. https://doi.org/10.1029/2024wr038308
- Yi, X. et al. (2025). A hydrogeomorphology-informed method for mapping continuous river networks from satellite imagery. *GIScience & Remote Sensing*. https://doi.org/10.1080/15481603.2025.2529620
- Zheng, K. et al. (2024). SHIFT: a spatial-heterogeneity improvement in DEM-based mapping of global geomorphic floodplains. *Earth System Science Data*. https://doi.org/10.5194/essd-16-3873-2024
