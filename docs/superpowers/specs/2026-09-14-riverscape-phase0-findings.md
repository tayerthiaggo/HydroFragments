# Riverscape Zoning — Phase 0 Findings (Fitzroy, Kimberley)

**Date:** 2026-09-14
**Source JSON:** `output/spikes/riverscape_phase0.json` (not committed)
**Script:** `scripts/spikes/riverscape_phase0_spike.py`

## Grid
- CRS / shape / resolution: `EPSG:3577` / `[539, 1117]` / `[30.0, -30.0]`
- WO product / time span: `ga_ls_wo_fq_myear_3` / `1987-01-01T00:00:00Z/2025-12-31T23:59:59.999999Z`

## DEM (`ga_srtm_dem1sv1_0`)
- STAC URL: `https://explorer.sandbox.dea.ga.gov.au/stac`
- Assets: `dem`, `dem_h`, `dem_s`
- Grid equal after `odc.stac.load(geobox=...)`: `true`

| Band | finite fraction | skeleton→trough p50 / p95 (m) | line→trough p50 / p95 (m) |
|---|---|---|---|
| dem_s | 1.0 | 270.0 / 818.84 | 234.31 / 807.77 |
| dem_h | 1.0 | 30.0 / 450.0 | 0.0 / 276.59 |

**Caveat — finite fraction is not a valid nodata check in this run:** every
band above reports `finite fraction = 1.0`. This is a script artifact, not
evidence that the AOI is fully populated: the spike casts each band to float
via `np.asarray(data, dtype=float)` without ever introducing a NaN nodata
value (DEM bands may carry a float nodata sentinel rather than NaN), so this
column cannot be relied on as a nodata check in this run — whether it would
ever show < 1.0 depends on whether `dem_s`/`dem_h`'s actual nodata sentinel
is NaN or some other float value, which was not verified here. Any evidence
loader written for
Plan 2 must load with an explicit NaN nodata path (e.g.
`odc.stac.load(..., dtype="float32")` plus masking against the declared
nodata value) rather than trusting `np.isfinite` on a raw cast.

**Decision:** the literal rule's mechanical output is `dem_band = dem_h`
(kept below, and in the table above, for audit purposes). **The corrected,
adopted default is `dem_band = dem_s`** — see the override below.

Rule applied: choose the band with the lower skeleton→trough p50. If the two
p50 values differ by 30 m or less, choose `dem_s`, because `dem_h` troughs are
stream-burned at AHGF line positions (record both line→trough p50 values as
evidence of that bias).

Measured: dem_s skeleton→trough p50 = 270.0 m; dem_h skeleton→trough p50 =
30.0 m. Gap = 240.0 m, which is greater than the 30 m tie threshold, so the
tie-break override to `dem_s` does not trigger and the literal rule selects
`dem_h`. However, dem_h's line→trough p50 is exactly 0.0 m (vs dem_s's
234.31 m) — a strong stream-burning signature, i.e. dem_h's trough pixels sit
essentially on top of the rasterized AHGF line, not on independently
DEM-derived terrain. See "Implications for Plan 2" below: this is exactly the
bias pattern the tie-break clause exists to catch, but the large p50 gap
means it isn't caught here, so Plan 2 should not treat `dem_h`'s channel
geometry as independent evidence of channel position — see below.

**Override (task-review finding, plan-level):** the literal rule gates on the
skeleton→trough p50 gap (≤30 m), which is the wrong signal here — the direct
signal, `dem_h`'s line→trough p50, is available and decisive: it is exactly
0.0 m, i.e. at or below one pixel (30 m), which is direct proof that `dem_h`'s
trough sits on the rasterized AHGF line rather than on independent terrain.
The corrected default for this AOI is therefore `dem_band = dem_s`, overriding
the literal rule's `dem_h` output. **Corrected general rule for Plan 2:** gate
directly on `line_to_trough.p50_m` for `dem_h` — if it is at or below one
pixel (30 m), treat `dem_h` as stream-burned and choose `dem_s` regardless of
the skeleton→trough gap.

## Fractional Cover percentiles (`ga_ls_fc_pc_cyear_3`)
- STAC URL / items: `https://explorer.sandbox.dea.ga.gov.au/stac` / `2`
- Percentile bands present: `bs_pc_10`, `bs_pc_50`, `bs_pc_90`, `npv_pc_10`, `npv_pc_50`, `npv_pc_90`, `pv_pc_10`, `pv_pc_50`, `pv_pc_90`
- Grid equal: `true`
- Band medians (seed water / trough / `dem_s` trough / AOI), from
  `fc.band_stats`:

| Band | finite fraction | median on seed water | median on `dem_s` trough | median AOI |
|---|---|---|---|---|
| bs_pc_10 | 1.0 | 21.5 | 12.5 | 12.0 |
| bs_pc_50 | 1.0 | 31.5 | 23.0 | 20.5 |
| bs_pc_90 | 1.0 | 38.5 | 32.0 | 29.5 |
| npv_pc_10 | 1.0 | 36.0 | 39.0 | 36.5 |
| npv_pc_50 | 1.0 | 47.0 | 50.5 | 49.0 |
| npv_pc_90 | 1.0 | 56.0 | 58.5 | 56.0 |
| pv_pc_10 | 1.0 | 18.5 | 14.5 | 18.0 |
| pv_pc_50 | 1.0 | 23.5 | 21.5 | 25.0 |
| pv_pc_90 | 1.0 | 35.0 | 39.5 | 44.5 |

Footnote: the "median on `dem_s` trough" column was computed using `dem_s`'s
trough mask, hardcoded in the spike script, regardless of which DEM band is
ultimately selected as `dem_band`. Since the corrected decision above adopts
`dem_s` as the default, this column is now consistent with the selected band
— but the label is explicit here so a reader does not have to infer which
band's trough it refers to.

**Caveat — finite fraction is not a valid nodata check in this run:** as with
the DEM table above, every band here reports `finite fraction = 1.0` purely
because Fractional Cover percentile bands are documented as uint8 with a 255
nodata sentinel (per the DEA FC product spec — not independently verified
against this run's raw values) and the spike casts them to float via
`np.asarray(data,
dtype=float)` without ever introducing a NaN — so `np.isfinite(...).mean()`
can never be anything but 1.0 in this run, regardless of actual nodata
coverage. Plan 2's evidence loaders must mask against the declared nodata
value (255 for these bands) and produce real NaNs before computing any
finite-fraction diagnostic.

**Time aggregation caveat:** the Fractional Cover STAC search returned
exactly 2 items (`fc.item_count = 2`, above), and the spike's `_band_array`
helper takes `.median("time")` over whatever items were returned — so every
median reported in the table above is really the mean of 2 yearly composite
values (which is why they all end in `.0` or `.5`).

- Chosen names: `bare_band = bs_pc_50`,
  `green_band = pv_pc_50`,
  `npv_band = npv_pc_50`

(All three spec-default percentile band names are present verbatim on the
STAC item's assets — no fallback to a "closest present" name was needed.)

## DEA Waterbodies
- Route: WFS `https://geoserver.dea.ga.gov.au/geoserver/wfs`, typename `dea:DigitalEarthAustraliaWaterbodies_v3`
- Features in AOI bbox: `112`
- Columns: `area_m2`, `dt_created`, `dt_satpass`, `dt_updated`, `dt_wetobs`, `geometry`, `id`, `meta_url`, `perimetr_m`, `timeseries`, `uid`, `wet_sa_m2`

## AHGF alignment
- line→EO skeleton (lines within ~1 km, 990 m / 33 px): p50 108.17 m, p95
  630.71 m, n 15883
- Retention / all-lines context: `ahgf_offset.line_to_skeleton_all` (all AHGF
  line pixels, not restricted to the ~1 km search radius) has p50 174.93 m,
  p95 3461.58 m, n 19455 — versus n 15883 within ~1 km. That means roughly
  81.6% of AHGF line pixels have an EO water skeleton within ~1 km
  (15883/19455) — an upper bound on the fraction with no nearby water at
  all, since a water pixel itself can sit off its own medial-axis skeleton —
  and 18.4% do not, with a tail out past 3.4 km. This is material context for
  Plan 2's corridor design: a meaningful minority of the network has no
  nearby EO water signal at all.
- Corridor seed: the template formula `p95 + max half-width`, clamped to
  90–600 m, is now evaluated against the measured p95 above (630.71 m).
  630.71 m already exceeds the 600 m upper clamp before adding any channel
  half-width term, so for this AOI the corridor seed is unconditionally
  600 m (the clamp ceiling) regardless of per-reach half-width. The 600 m
  ceiling appears too tight for this catchment's measured AHGF/EO offset and
  should be reconsidered when Plan 2 sets `corridor_max_m`.
- `UpstrDArea` 5/50/95 %: 671305.95 / 53181024764.69 / 54159056415.92 m²

## Implications for Plan 2
- `dem_h`'s line→trough p50 is 0.0 m — a clear stream-burning signature (its
  trough coincides almost exactly with the rasterized AHGF line). The
  decision rule's tie-break protection against this bias only triggers when
  the skeleton→trough p50 gap is ≤30 m; here the gap is 240 m, so the literal
  rule still selects `dem_h` **— overridden above: the adopted default for
  this AOI is `dem_band = dem_s`** (see the Override in the DEM section).
  Plan 2 should treat `dem_h`'s apparent channel
  position as AHGF-line-conditioned, not as independent terrain evidence,
  when combining it with the AHGF-offset analysis below — e.g. do not use
  `dem_h` trough position as an independent cross-check on AHGF line
  placement, since the two are not independent by construction.
- DEA Waterbodies must be addressed via the versioned literal typename
  `dea:DigitalEarthAustraliaWaterbodies_v3` (not an unversioned
  `dea:DigitalEarthAustraliaWaterbodies` guess) when Plan 2 hard-codes a WFS
  query.
- `hydroseason.open_wo_statistics` (used by
  `open_wo_statistics_for_zoning`) scopes `AWS_NO_SIGN_REQUEST` and related
  GDAL env vars to its own STAC-search-and-load call and restores the
  caller's prior (unset) environment in a `finally` before returning —  but
  the returned dataset is Dask-lazy, so any later `.compute()`/`np.asarray()`
  on it happens after that restore. Any Plan 2 code (not just this spike)
  that materializes `WoStatistics.frequency`/`count_wet`/`count_clear` after
  calling `open_wo_statistics_for_zoning` must itself set
  `AWS_NO_SIGN_REQUEST=YES` (e.g. via `os.environ.setdefault(...)` before
  calling `open_wo_statistics_for_zoning`, so hydroseason's own
  snapshot/restore cycle preserves it) rather than assuming unsigned S3
  access stays configured after the call returns.

## Plan 1 completion
- Full suite: 894 passed, 1 failed (tests/release/test_branding.py::test_tracked_text_uses_only_readme_lineage_mention) — both known pre-existing failure categories in this repo are unrelated to this plan: the one that fired this run, tests/release/test_branding.py::test_tracked_text_uses_only_readme_lineage_mention (deterministic, pre-existing since commit 6bd26c6, failing because docs/superpowers/plans/2026-07-20-user-ready-implementation.md — last touched at 8a70e74 — mentions a retired package name), and a second category that did not fire this run but is also pre-existing and accepted: an intermittent Windows bundle-rename race (`PermissionError: [WinError 5] Access is denied`) in `hydrofragments/output/bundle.py::commit_staged_bundle`, documented in `docs/superpowers/plans/2026-08-12-dynamics-and-spatial-exports.md`.
- Commit: 37e3095

## Basin-scale rerun (Plan 2, Task 3)

**Source JSON:** `output/spikes/riverscape_phase0_basin.json` (not committed)
**Script:** `scripts/spikes/riverscape_phase0_basin_spike.py`
**AOI:** `data/fitzroy_basin_aoi.geojson` (full Fitzroy River basin, ~97,200 km²), replacing the narrow AOI for distributional (not band/route) decisions.

This does not re-decide `dem_band`, FC band names, or the DEA Waterbodies
route — those came from the narrow-AOI Phase 0 spike above and don't depend
on AOI size. It re-measures the distributions that were degenerate there.

**Nodata-masking bug found and fixed during this task (not in the original
brief text):** the first two runs of this script (matching the brief's
script verbatim, plus the required PROJ env fix — see "Run cost" below)
completed with `errors: {}` but produced degenerate numbers: `dem.trough_pixels
= 0` basin-wide, and every FC band's `median_on_seed_water` read exactly
`255.0` (FC's uint8 nodata sentinel) for all three bands. Root-caused via a
standalone diagnostic against the same AOI/geobox:
- FC bands declare `nodata=255`. 83.9% of this basin's 12,426,644 seed-water
  pixels are FC-nodata (Fractional Cover has no land-cover estimate over
  persistent water), so an unmasked `nanmedian` over the water mask returned
  the sentinel itself, not a cover fraction. Masking only *after*
  `.median("time")` would still be wrong: a pixel with a mix of
  valid/nodata observations across time already has its per-pixel temporal
  median pulled toward 255 (observed on-water max of 176.5 on a ~0-100
  band) unless nodata is masked to NaN *before* the time reduction.
- The DEM's declared nodata is `-3.4028234663852886e+38` (STAC
  `raster:bands[0].nodata`, confirmed via the item's own metadata) — a
  large-magnitude finite float, not NaN, so `np.isfinite()` alone does not
  catch it (finite fraction read 99.87%, i.e. misleadingly "clean"). Left
  unmasked, these cells fed `ndimage.uniform_filter`'s box-average and blew
  up `local_mean` to NaN across virtually the whole raster, so
  `depth >= TROUGH_DEPTH_M` was never true.

Both are the exact gap this doc's DEM/FC caveats above already flagged for
Plan 2 loaders ("must mask against the declared nodata value ... rather
than trusting `np.isfinite` on a raw cast") — at basin scale it stopped
being cosmetic and actively corrupted the trough/on-water statistics. Fixed
in `scripts/spikes/riverscape_phase0_basin_spike.py`'s `_band_array` by
masking each band's declared `nodata` attribute to NaN before any time
reduction; the numbers below are from the corrected rerun (`errors: {}`,
`dem.finite_fraction = 0.9987`, `dem.trough_pixels = 43,074,902`).

### Grid
- Shape / pixel count: `[12813, 16033]` / `205,430,829`

### AHGF alignment (basin-wide)
- line→EO skeleton (within ~1 km): p50 364.97 m, p95 924.18 m, n 407,211
- `UpstrDArea` quantiles (m²): p1 278,736.85, p5 558,117.67, p25
  1,698,952.54, p50 5,248,549.48, p75 33,181,307.65, p95 3,064,075,197.49,
  p99 53,593,517,641.37
- Reach count: 31,318

**Comparison with the narrow AOI:** the narrow AOI's `UpstrDArea` p50≈p95
(≈53,000–54,000 km²) is now the basin's **p99** (53,593.52 km² vs the
narrow AOI's 53,181.02–54,159.06 km²) — the narrow AOI's entire observed
range sits almost exactly at the top 1% of basin reaches by upstream
drainage area, confirming it sampled only the largest (near-outlet)
reaches, not a representative cross-section. The envelope fit (spec §4.2
step 7) now has real per-log-A bins to fit against, spanning roughly five
orders of magnitude (p1 ≈ 0.28 km² to p99 ≈ 53,594 km²).

### Fractional Cover medians (basin-wide, `dem_s` trough)
| Band | median on seed water | median on trough | median AOI |
|---|---|---|---|
| bs_pc_50 | 7.5 | 22.0 | 24.5 |
| pv_pc_50 | 4.5 | 21.5 | 20.5 |
| npv_pc_50 | 76.0 | 50.5 | 49.0 |

**Comparison with the narrow AOI:** narrow-AOI `bs_pc_50` was 31.5 (seed) /
20.5 (AOI) — the basin-wide **AOI** median (24.5) is broadly similar (within
~20%), but the basin-wide **seed-water** median (7.5) is materially
different, more than 4x lower than the narrow AOI's 31.5. This is itself
evidence *for* the per-run-calibration decision in §4.2 step 4 rather than a
fixed threshold: the ~97,200 km² basin spans landscapes (coastal mudflats
through arid interior reaches) far more heterogeneous than the narrow AOI's
single river stretch, so a bare-cover value calibrated on one AOI does not
transfer to another. Whether *per-reach* (not just per-run) calibration is
also needed can't be assessed from this script, which only reports
basin-wide aggregate medians, not a per-reach breakdown — that would need a
follow-up binned by reach or region.

### Run cost
- Wall-clock time: 721.6 s (~12.0 minutes)
- Peak memory: 18,659,270,656 bytes (~17.4 GiB / 18.7 GB) (method: PowerShell
  polling `Get-Process -Id <pid> | Select WorkingSet64` every 15 s for the
  process lifetime, tracking the running maximum — Windows equivalent of
  `resource.getrusage(...).ru_maxrss`). This exceeds the brief's own
  ~800 MB–1.2 GB per-step estimate by roughly 15x: the nodata-masking fix
  (`values.where(values != nodata)`) builds xarray-level intermediate
  copies beyond the raw float32 path the original design assumed. It
  stayed well within this machine's headroom throughout (system had
  66.9 GB total / ≥28.6 GB free at every observed peak; no swapping, no OS
  kill), so it did not warrant a BLOCKED report, but a memory-conscious
  follow-up should mask nodata via plain `numpy.where` on the already-cast
  float32 array rather than through xarray's `.where`.
- FC-band computation approach used: one-band-at-a-time eager
  materialization (as designed), with nodata masked to NaN via xarray
  `.where()` before each band's `.median("time", skipna=True)` reduction.

### Implications for Plan 3
- `corridor_min_m`/`corridor_max_m` (spec §6, currently 90/600): the
  basin-wide line→skeleton p95 (924.18 m) still saturates — and now exceeds
  by more, not less — the 600 m ceiling the narrow AOI's 630.71 m already
  saturated. The 600 m ceiling remains too tight for this catchment's
  measured AHGF/EO offset at basin scale.
- `bare_threshold` calibration inputs: basin-wide bare-cover medians differ
  materially from the narrow AOI's (seed-water 7.5 vs 31.5), which supports
  per-run calibration (not a value hardcoded from the narrow-AOI spike) as
  the minimum needed; this script's basin-wide aggregates can't rule out
  needing finer per-reach calibration too, since they average over the
  whole basin's heterogeneous landscape rather than resolving it spatially.
