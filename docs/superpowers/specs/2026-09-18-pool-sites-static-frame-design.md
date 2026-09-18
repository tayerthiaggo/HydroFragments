# Specification: Pool Sites — Static Frame

**Date:** 2026-09-18
**Status:** Design approved in brainstorm; awaiting written-spec review before planning.
**Scope:** Spec 1 of 2. This document defines the *frame*: a frozen, series-global set of pool sites on a channel coordinate. Spec 2 (`pool-sites-annual-story`) will define per-year narrative metrics computed against that frame.

---

## 1. Problem

Persistence at reach resolution cannot say *which* pool within a reach persisted, whether a reach's persistence is carried by one stable pool or by different pools in different years, or whether a pool's footprint moves between years.

Existing machinery cannot answer this:

- `build_pool_id` (`hydrofragments/output/vectors.py:140`) is month-scoped and window-scoped.
- `refuge_spatial_stability` (`hydrofragments/metrics/dynamics.py`) is a raster-overlap Jaccard between consecutive end-dry refuge masks. It measures how much a footprint moved in aggregate; it cannot attribute movement to a pool.
- Chaining month-to-month patch overlap into lineages fails on intermittent rivers: at the wet peak the channel connects into one component and transitive closure merges every pool in the catchment into one identity. This failure is documented in `docs/superpowers/specs/2026-08-19-provenance-aware-pool-sites-design.md` §5.2 and is not revisited here.

A further problem, specific to this design: any per-year revisable boundary makes cross-year comparison happen on a drifting frame, so a moving pool becomes indistinguishable from a moving ruler. **Whether pools move is the first result this work must produce, not an assumption it may make.** The frame must therefore be fixed and water-independent where possible.

## 2. Relationship to existing specs

This spec **supersedes §5 of** `docs/superpowers/specs/2026-08-19-provenance-aware-pool-sites-design.md` (unapproved draft). That draft reached for the same object, conflicted with approved gate U8 (`hydrofragments/metrics/connectivity.py`, node identity locked to drainage reach `HydroID`), and left its site edge rule unresolved (§5.6).

The conflict **dissolves rather than resolves**: under §3 of this spec, pool sites never supply nodes to `metrics/connectivity.py`. `build_site_graph` / `node_source="pool_sites"` (old §5.5) is explicitly out of scope. The reach graph is untouched, U8 is untouched, and the unresolved edge-rule question (old §5.6) retires because there is no graph to define edges for.

§3 of that draft (provenance-aware inputs / WaterMask-TSFill companion bands) is **not** superseded, and is **not** a dependency of this work (§11.3).

## 3. Separation constraint (binding)

Pool-level analysis is a distinct workstream from catchment- and reach-level (AOI-level) metrics. It has its own metrics and its own workflows.

Structurally this means:

- No metric rows. No `EdgeFlag` additions. No `metrics/registry.py` entries. No use of `hydrofragments/schema.py`'s `SCHEMA_VERSION` lineage.
- Own config (`PoolsConfig`), own workflow entry, own output bundle, own manifest section. Not wired into `hydrofragments/pipeline.py` or `hydrofragments/workflow.py`.
- No dependency `pools → metrics`. Where logic is genuinely shared it is **extracted to a neutral location both import**, never called across the boundary (§6.2).

Riverscape remains a dependency. It provides a static spatial frame (reaches, centreline, corridor, landform zones, REM), not catchment metrics, and the dependency is one-way.

## 4. Vocabulary

| Term | Meaning |
|---|---|
| **`s`** | Channel coordinate: distance to outlet in metres along the AHGF reach line, decreasing downstream. |
| **Dry window** | Months `[mid_dry_month, end_dry_month]` inclusive, per hydrological year, from `hydroseason.detect_hydrological_years`. |
| **Core** | Pixels whose dry-window occurrence meets `core_occurrence_pct`. Where a pool is loyal to a location. |
| **End-dry union** | Pixels wet at ≥ `k_min_anchors` admissible end-dry anchors. |
| **Envelope** | `core ∪ end-dry union`. Where a pool has ever been. |
| **Separator** | A chainage at which the frame is cut, from merge-stop (§8) or the `m` fallback (§8.4). |
| **Site** | A maximal run of non-separator `s` intersecting the envelope. Fixed for the whole series. |
| **Seed** | A site's identity marker for the backwards walk. Per **site**, never per fragment. |
| **Min extent** | A site's water at its end-dry anchor, in a given HY. Union over that site's fragments. |
| **Max extent** | A site's water at the last month, walking backwards, at which it is still solely that site's. |

## 5. Design decisions (settled in brainstorm)

| # | Decision | Rationale |
|---|---|---|
| D1 | Occurrence restricted to the dry window, stratified by HY-season-position (months counted back from end-dry), not calendar month. | HY-relative windows drift across calendar months, so calendar strata stop being comparable. `end_dry_month` is a detected anchor, so position 0 means the same thing every year. |
| D2 | Core and end-dry union are the same estimator at two thresholds. | `k=1` union and a percentage threshold are one statistic. Outlier control becomes a threshold that can be swept, not an ad-hoc filter. |
| D3 | Envelope = `core ∪ end-dry union`. | Containment is not automatic: a margin wet every mid-dry but always dry by end-dry scores high occurrence and zero end-dry hits. Union makes `core ⊆ envelope` true by construction. |
| D4 | Pool identity and displacement measured in 1-D on `s`; 2-D mask carried as an attribute. | Removes the revisable-2-D-boundary problem entirely. Displacement becomes a signed distance in metres against a frame that does not move. |
| D5 | `s` is referenced to the **AHGF reach line**, not the conflated skeleton. | The skeleton is a function of the water data (`build_centreline` conflates onto EO water), so a coordinate built on it would drift year to year. |
| D6 | Site cuts come primarily from **merge-stop** (§8); global threshold `m` is a structural fallback. | A global threshold mis-cuts systematically wet or dry reaches. Merge-stop is local and adaptive: the cut lands where those two pools actually meet. |
| D7 | The backwards walk lives in **spec 1**, not spec 2. | Cuts define sites; sites must be frozen before per-year iteration. Putting the walk in spec 2 would make spec 1 depend on spec 2. |
| D8 | Stop rule: component contains seeds from ≥2 **distinct sites**. Envelope exceedance recorded, never blocking. | Blocking on envelope crossing would answer "did the pool exceed its boundary" by construction instead of measuring it. |
| D9 | Seeds are per-site. A site's own fragments re-merging is invisible to the stop test. | A pool that shatters in a savage dry year is still one pool. |
| D10 | HY support: `in_stream_catchment` operational; per-reach evaluated as a comparison. | The walk compares two pools month by month; different clocks make "the month before merge" ambiguous. Full-AOI extent is dominated by floodplain inundation, a different process from channel drying. |
| D11 | Minimise external-dataset dependence: REM is optional and evaluated, never required. In spec 1 this governs `cut_tiebreak` only; the parallel decision for spec 2's position estimators (downstream-most and centroid always, min-REM optional) is recorded here but implemented there. | REM is unvalidated at sub-pool scale. |
| D12 | Aggregate the **cut chainage** across years, not the max extent. | Max extent is confounded with year wetness. Cut location is set by bed topography and is far more stable. |

## 6. Architecture

### 6.1 Modules

New subpackage `hydrofragments/pools/`. Dependency direction is strictly one-way.

| Module | Responsibility | Depends on |
|---|---|---|
| `contracts.py` | Frozen dataclasses: `ChannelCoordinate`, `Site`, `SiteSet`, `PoolYearRecord`. The type boundary spec 2 consumes. | nothing in `pools` |
| `chainage.py` | Builds the network channel coordinate `s`. | riverscape, `contracts` |
| `occurrence.py` | Dry-window occurrence and its per-stratum support. | `temporal/hydroyear`, shared estimator (§6.2), `contracts` |
| `walk.py` | Backwards merge-stop walk. Emits cuts and per-year records. | `chainage`, `occurrence`, `patches/labels`, `contracts` |
| `sites.py` | Delineation. Consumes cuts, freezes the `SiteSet`. | `chainage`, `occurrence`, `walk`, `contracts` |

Order: `chainage`, `occurrence` → `walk` → `sites`. Nothing reads back.

### 6.2 Shared, not coupled

`_season_stratified_occurrence` (`hydrofragments/metrics/persistence.py:86`) hardcodes `groupby("time.month")`. It is **extracted to a neutral shared location** and parametrised to take a stratum label coordinate, defaulting to `time.month`. Both `metrics/persistence.py` and `pools/occurrence.py` import it; neither imports the other.

Acceptance: existing metric outputs bit-identical after the move.

### 6.3 Adapter extension

`hydrofragments/temporal/hydroyear.py` currently keeps only `end_dry_month` from the anchors table. `hydroseason` 0.2.0 emits `peak_month`, `mid_dry_month`, `mid_extent_pct`, `end_dry_month`, `end_extent_pct`, `hy_start`, `hy_end`, `confidence` (verified against source at `D:\RLH\5.6\repos\hydroseason`, `fd53e18`, `version = "0.2.0"`). The adapter is extended to carry all of them. It remains a pure adapter: no detector logic is added.

`pyproject.toml:25` currently pins `hydroseason==0.1.0` while `0.1.1` is installed. This pre-existing drift is fixed in the same change; the pin becomes `==0.2.0`.

## 7. Static layers

### 7.1 Channel coordinate (`chainage.py`)

Reference geometry is the AHGF reach line (D5). Skeleton and mask pixels are projected onto it with `geom.project(point)` — the same call as `riverscape/bridging.py:207`, but unnormalised, returning metres.

`s` = distance to outlet, decreasing downstream. Per reach:

```
s(point) = s_reach_start + geom.length - geom.project(point)
```

`s_reach_start` is the cumulative length of all reaches downstream, accumulated over `riverscape/terrain.py:114` `_reach_topological_order` run in reverse. `riverscape/bridging.py:121` `_direction_sign` handles reach lines digitised against flow. A projected CRS in metres is required; `hydrofragments/spatial/crs.py` already enforces this.

**Tree limitation, stated not hidden.** `s` is monotone along any single flow path, but two tributaries above a confluence can share an `s` value. The unique address is `(reach set, s)`.

- A site confined to one flow path has a genuine interval `[s_min, s_max]`. This is the normal case.
- A site spanning a confluence is a subtree. `[s_min, s_max]` is then a projection that hides branch structure. Such sites are flagged `confluence_spanning` and carry their member reaches. Displacement in `s` remains meaningful; *length* does not.

**AHGF sinuosity differs from real channel sinuosity.** `s` is AHGF-metres, internally consistent and comparable across years — which is what displacement requires — but not survey distance, and must not be reported as such. The AHGF offset measured in riverscape Phase 3 bounds the error.

**Output** — `ChannelCoordinate`: per-centreline-pixel `s`; per-reach `(s_start, s_end)`; the topological order used; the flow-path/branch id of each reach.

### 7.2 Dry-window occurrence (`occurrence.py`)

**Window** is `[mid_dry_month, end_dry_month]` inclusive, per HY, read from the anchors table. Its depth is **detected, not configured** — `mid_dry_month` is a first-class anchor in 0.2.0 (`hydroseason/hydro_year.py:430-442`). Note `label_hydrological_months` returns only `Wet`/`Dry`, so the window comes from anchors, not month labels.

**Strata** are months counted back from end-dry: position 0 is the end-dry month, 1 the month before, and so on to the mid-dry edge. Counting backwards (not forwards from mid-dry) is what makes a stratum mean the same thing in every year.

**Uneven support is the hazard.** Strata near end-dry have support from nearly all years; deep strata only from years with long dry windows. The equal-weight mean would give a stratum backed by 3 years the same weight as one backed by 40. Guard: `min_stratum_years` floor — strata below it drop out of the mean rather than being equally weighted. Years with shorter windows contribute NaN at deep strata and fall out via `skipna`, exactly as unsupported calendar months already do.

**Support floor is re-derived, not inherited.** `validity.min_valid_obs` was calibrated against 12 strata × ~40 years. This window is a handful of strata × ~40. Reusing that value would suppress most of the channel. `PoolsConfig` carries its own floor, defaulted from the Fitzroy distribution.

**Output** — occurrence percentage over the dry window; per-stratum year-support counts; realised window depth per year. The support counts are consumed by §7.3, not discarded.

### 7.3 Core, union, envelope (`sites.py`, first stage)

- **core** = dry-window occurrence ≥ `core_occurrence_pct`
- **end-dry union** = wet at ≥ `k_min_anchors` admissible end-dry anchors, `k_min_anchors ≥ 2`
- **envelope** = `core ∪ end-dry union` (D3)

`k ≥ 2` excludes singletons, so no single anomalous year can extend the envelope anywhere.

**Anchor admissibility.** An end-dry anchor enters the union only if:

- `confidence` ≥ `hy_confidence_min`, from the anchors table
- valid-observation support at that anchor ≥ the §7.2 floor

Rejected anchors are listed by year with reason (§10). `k` counts admissible anchors, so effective `N` shrinks on rejection and is reported alongside.

**Anchor labelling** reuses `patches/labels.py:325` `label_components` with its `min_patch_pixels` filter and its cross-chunk reconciliation. Window-straddling pools are already handled there and are not rebuilt.

**Restricted to `LANDFORM_IN_CHANNEL`** (`riverscape/codes.py`). Not a preference: `s` is defined only on the channel network, so off-channel water has no coordinate. `LANDFORM_OFF_CHANNEL_RIVERINE` refugia are real and are explicitly out of scope here (§12).

## 8. Backwards merge-stop walk (`walk.py`)

### 8.1 Mechanism

No watershed transform and no synthetic region growing. The walk replays **observed** monthly masks backwards from the end-dry anchor toward `mid_dry_month`, labelling each month with `label_components`, and attributing components to sites.

Per site, per HY:

1. **Seed.** All end-dry fragments whose `s` falls in the site's interval share that site's seed identity (D9). A site with no water at its end-dry anchor has no seed: record `present_at_end_dry = false`, no walk, and do not skip the row.
2. **Min extent.** Union of the site's end-dry fragments, with `fragment_count`. The count is the intra-pool fragmentation signal.
3. **Walk backwards** month by month to `mid_dry_month`.
4. **Stop** at the first month `t` where the component containing this site's seed also contains a seed from a **different** site. Max extent is the state at `t-1`. Re-assembly of the site's own fragments never triggers this.
5. **Or stop** at `mid_dry_month` with `stop_reason = window_edge`.

### 8.2 Censoring

`stop_reason = window_edge` means the maximum is **right-censored**, not observed. Mixing censored and uncensored maxima in one distribution biases it downward. `stop_reason` is mandatory in the output and censoring must be respected in any aggregation. `mid_extent_pct` / `end_extent_pct` from the anchors table give the per-year wetness covariate against which censoring is interpreted.

### 8.3 Cut location

At the merge month the two sites are already one component, so the information is at `t-1`: site A ends at `s_a`, site B begins at `s_b`, dry between. The cut lies in `[s_a, s_b]`.

- **Default tie-break:** midpoint in `s`. Assumes nothing and needs no external data.
- **Optional:** minimum-REM point within the gap, behind a flag, evaluated not assumed (D11).

`s_b - s_a` at `t-1` is recorded as the **pre-merge barrier gap** — the dry barrier length between two refuges immediately before reconnection.

**Ties.** Two sites merging within the same month give one cut with no ordering. The midpoint rule resolves this deterministically. Any alternative tie-break must also be deterministic or cut chainages become run-dependent.

**Adjacency is not connection.** Two masks meeting at 30 m pixels is adjacency, not observed flow. This caveat carries verbatim from the 2026-08-19 draft §6 and must appear in outputs and prose.

### 8.4 From cuts to separators

Across years, each site pair yields a set of cut chainages. These cluster at riffles.

- A pair's **modal cut** becomes a separator once its supporting year count meets `min_cut_support_years`.
- Pairs below that floor fall back to the global threshold `m`: `s` is a separator iff dry at ≥ `m` admissible anchors.

**The fallback is structural, not decorative.** Many pairs never merge inside the window in most years, so `m` will fire routinely and must be implemented and calibrated, not treated as an edge case.

**`k` and `m` are independent knobs.** `k` sets envelope extent; `m` sets cuts. A chainage wet at 5 of 40 end-dries is simultaneously inside the envelope (`k=2`) and a legitimate separator (dry three years in four). Tying `m` to `N-k+1` would collapse them into one knob and reintroduce the one-ribbon failure.

## 9. Site delineation (`sites.py`, second stage)

**Project to 1-D.** Each corridor pixel is attributed to its nearest centreline pixel — `distance_transform_edt(..., return_indices=True)`, a channel Voronoi — and inherits that pixel's `s`. A chainage is *wet at anchor y* if any pixel attributed to it is wet at y.

**Sites** = maximal runs of non-separator `s` intersecting the envelope. Runs shorter than `min_site_length_m` are dropped. (`min_patch_pixels` removes 2-D speckle; `min_site_length_m` is a different, additionally required filter on 1-D length.)

**Cores name sites.** Core components projected to 1-D and assigned to their containing site:

| Cores in site | `core_status` | Meaning |
|---|---|---|
| 1 | `cored` | Stable identity. |
| ≥2 | `multi_cored` | Candidate over-merge. Kept and flagged; the count of these says whether `m` / `min_cut_support_years` is too lenient. |
| 0 | `mobile` | Kept and flagged. **This count is a headline result**, not a data-quality problem. |

**Identity is geometric, never ordinal.** `site_id` derives from `(outlet-most member reach, s_min at fixed precision)`. Label iteration order must never enter it, or IDs shuffle between runs and spec 2 silently compares different pools.

**`site_id` is distinct from `build_pool_id`** (`output/vectors.py:140`), which is month- and window-scoped and unchanged. The two must not be conflated in schema or prose.

## 10. Outputs

### 10.1 `pool_sites` — one row per site (the frozen frame)

`site_id`, `s_min`, `s_max`, member reaches, `branch_id`, `core_status`, `confluence_spanning`, core count, envelope length (m), envelope area (m²), and per boundary: cut provenance (`merge_modal` | `m_fallback`) with its supporting year count.

Written once. Byte-identical across reruns on identical inputs is the determinism assertion target.

### 10.2 `pool_site_years` — one row per site per HY

`site_id`, `hy`, `hy_confidence`, `anchor_admissible` + reason, `present_at_end_dry`, min extent (area, `s`-interval, `fragment_count`), max extent (area, `s`-interval), `stop_reason` (`merge` | `window_edge`), `stop_partner_site_id`, `stop_month_offset` (months back from end-dry), `envelope_exceeded` + exceedance length (m), `premerge_barrier_gap_m`, and the site's own drying month with its offset from the basin `end_dry_month` (D10).

Spec 2 adds **columns** here. It never adds rows.

The two tables stay separate: the frame is not a per-year fact, and merging them would let a per-year value quietly redefine a frozen one.

### 10.3 Rasters

Dry-window occurrence; envelope; site label raster. Site labels are `site_id`, not sequential integers, so raster and table join without a lookup that can drift.

### 10.4 Schema version

New constant, own version. **Not** a bump of `schema.py`'s `SCHEMA_VERSION = "1.1.0"` — these are not metric rows (§3), and a pools change must never force a version bump on the metric schema.

### 10.5 Manifest

Resolved `hydroseason` version (not the pin); `PoolsConfig` hash; `hy_support` as a first-class field; the admissible-anchor list with per-year rejection reasons; realised window depth per year; regime-screening verdicts (§11.1); and the threshold-sweep results used to set defaults.

Runs differing in `hy_support` are **not comparable** and the manifest must make that legible.

## 11. Calibration and evaluation

### 11.1 HY support: B operational, C evaluated

`hy_support` config field. Default `in_stream_catchment` (B): the extent series feeding `hydroseason` is water within `LANDFORM_IN_CHANNEL`. The in-channel zone is a static landform product built once, so masking a time series with it is not circular.

Per-reach (C) is evaluated in two tiers, using `hydroseason.run_hydroseason_many` (`hydroseason/batch.py:104`), which provides per-AOI batch detection with per-item error isolation:

- **Tier 1 — anchor spread.** Distribution of `mid_dry_month` / `end_dry_month` across reaches against the basin value. If reach anchors sit within a month of the basin value, B is safe and the spec says so with a number. If per-reach batch over the full Fitzroy reach set is too costly, run tier 1 on a stratified reach sample (headwater / mid / lowland) — the question is whether spread exists and where, not a per-reach inventory.
- **Tier 2 — full delineation under both**, only if tier 1 shows real spread. `pool_sites` computed twice; agreement measured as site-count difference, boundary displacement (m), and `core_status` churn.

To make C runnable end-to-end, **a cross-reach pool pair uses the downstream reach's clock**. Deterministic and defensible: the downstream pool governs when the connection is observed. Without this rule C can only produce anchor spreads, not a comparable site set.

**Regime screening is mandatory** on every series before its anchors are trusted: `hydroseason.assess_water_regime` (`hydroseason/_regime.py:138`) and `classify_seasonal_pattern` (`hydroseason/_seasonality.py:78`). The in-stream extent series is small at end-dry, so seasonality cannot be assumed. Under C, a reach the library calls aseasonal has no meaningful `mid_dry_month` and must be handled explicitly rather than producing a silent bad clock.

### 11.2 Sensitivity sweeps (deliverables, not checks)

- **Leave-one-year-out** over admissible anchors: envelope area as a function of which year is dropped. Worst-case shrinkage says whether `k` is doing its job.
- **`k` and `core_occurrence_pct` sweep**: envelope and core area against each.
- **`m` and `min_cut_support_years` sweep**: site count, median site length, `mobile` fraction, `multi_cored` count.
- **`m`-fallback firing rate** on Fitzroy: how often merge-stop actually decides a boundary.
- **Merge-stop vs `m`**, run once during calibration: agreement between the two delineations. `m` is a knob that would need defending for the length of a paper; merge-stop replaces it with a cut location the data picks.

These curves justify the defaults. Defaults are set from Fitzroy distributions, not guessed — the same way riverscape Phase 3 defaults came from Phase 2's widened numbers.

### 11.3 Not a dependency

WaterMask-TSFill companion bands (`observed`, `method_flag`, `confidence`) are **not** required. There is no companion-band gate in §7.3 admissibility. Under a gap-filled input a reconstructed wet pixel can bridge two sites and fabricate a merge; this is a **stated limitation**, recorded in the manifest as merge detection being ungated, not a mechanism this spec builds.

## 12. Config

New `PoolsConfig`. Config hash bump (as riverscape Phase 2 did).

| Field | Meaning |
|---|---|
| `hy_support` | `in_stream_catchment` (default) \| `reach` |
| `core_occurrence_pct` | Core threshold on dry-window occurrence |
| `k_min_anchors` | Envelope threshold, ≥ 2 |
| `m_dry_anchors` | Global separator fallback threshold |
| `min_cut_support_years` | Year support before a pair's modal cut is trusted |
| `hy_confidence_min` | Anchor admissibility |
| `min_valid_obs_dry_window` | Support floor, re-derived for the dry window |
| `min_stratum_years` | Per-stratum year-support floor |
| `min_site_length_m` | Minimum site length along `s` |
| `cut_tiebreak` | `midpoint` (default) \| `min_rem` |

## 13. Testing

### 13.1 Spine

Determinism and chunk-invariance, now across the 40-year walk rather than raster maths alone. `pool_sites` byte-identical across reruns and across active-window chunkings, including sites straddling window boundaries and merges detected near a seam.

### 13.2 Behavioural (synthetic series, known answers)

| Test | Expected |
|---|---|
| Pool dries fully, refills | One site with a dry interval, not a death and a birth |
| Pool shatters into N fragments at end-dry, never meets another site | Walks to window edge; `stop_reason = window_edge`; never `merge` (D9) |
| Two pools merge in the same place every year | Stable modal cut |
| Two pools merge at different chainages each year | Wide cut distribution; flagged, never silently averaged |
| Site with no core | Retained, `core_status = mobile` |
| Reach wetter than basin average holding two distinct pools | Cut by merge-stop where global `m` fails to cut — **the test that justifies the whole design over a threshold** |
| Every max extent hits the window edge | All `window_edge`; must not be reported as an observed maximum distribution |

### 13.3 Coordinate invariants

`s` strictly decreases downstream along every flow path; total network length matches summed reach lengths; a pixel's `s` is invariant to active-window chunking (the streaming byte budget must not move the ruler); reach lines digitised upstream-to-downstream and reversed give identical `s`; a confluence reach gets exactly one `s_start`.

### 13.4 Mutation testing

Required on `chainage.py`, `walk.py`, and cut-selection logic, per the standing requirement that algorithmic code is mutated rather than only read. Mutations most likely to survive a reading review: sign flips in downstream direction, off-by-one in the backwards month walk, `≥` vs `>` in `k` / `m` / `min_stratum_years` thresholds, and per-fragment vs per-site seed identity in the stop test (D9).

### 13.5 Refactor safety

Existing metric outputs bit-identical after the §6.2 estimator extraction.

## 14. Out of scope

- Per-year narrative metrics: positions, displacement statistics, contraction trajectories, fragmentation time series. These are **spec 2**, consuming the frozen `SiteSet` and the `pool_site_years` records.
- Off-channel riverine refugia (`LANDFORM_OFF_CHANNEL_RIVERINE`). They have no `s` and need a different frame. They are expected to be less mobile and simpler to assess — plausibly a static footprint suffices, which is the null hypothesis this work tests for in-channel pools. Own later spec.
- Any node source for `metrics/connectivity.py` (§2).
- Morphological bridging, dilation, erosion, or DEM-guided reconnection of the water mask. Sites derive from the mask as given.
- Provenance-aware input contracts (§11.3).
- Any claim that site co-occurrence in one component is observed hydraulic connectivity (§8.3).

## 15. Open questions

1. Whether REM resolves sub-pool relief. Decides if `cut_tiebreak = min_rem` and the min-REM position estimator are available at all. Tested, not assumed.
2. `min_site_length_m` value. Calibration item.
3. `min_cut_support_years` value, and the observed `m`-fallback firing rate on Fitzroy. Calibration item (§11.2).
4. Whether tier-2 of the HY-support comparison is needed, which tier 1 decides (§11.1).
