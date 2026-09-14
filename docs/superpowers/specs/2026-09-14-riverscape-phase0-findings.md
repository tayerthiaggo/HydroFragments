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
