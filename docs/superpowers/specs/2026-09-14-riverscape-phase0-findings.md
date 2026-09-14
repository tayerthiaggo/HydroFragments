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

**Decision:** default `dem_band = dem_h`.
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

## Fractional Cover percentiles (`ga_ls_fc_pc_cyear_3`)
- STAC URL / items: `https://explorer.sandbox.dea.ga.gov.au/stac` / `2`
- Percentile bands present: `bs_pc_10`, `bs_pc_50`, `bs_pc_90`, `npv_pc_10`, `npv_pc_50`, `npv_pc_90`, `pv_pc_10`, `pv_pc_50`, `pv_pc_90`
- Grid equal: `true`
- Band medians (seed water / trough / AOI), from `fc.band_stats`:

| Band | finite fraction | median on seed water | median on trough | median AOI |
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
- line→EO skeleton (lines within 1 km): p50 108.17 m, p95 630.71 m, n 15883
- Implied corridor seed: `p95 + max half-width`, clamped to 90–600 m
- `UpstrDArea` 5/50/95 %: 671305.95 / 53181024764.69 / 54159056415.92 m²

## Implications for Plan 2
- `dem_h`'s line→trough p50 is 0.0 m — a clear stream-burning signature (its
  trough coincides almost exactly with the rasterized AHGF line). The
  decision rule's tie-break protection against this bias only triggers when
  the skeleton→trough p50 gap is ≤30 m; here the gap is 240 m, so the literal
  rule still selects `dem_h`. Plan 2 should treat `dem_h`'s apparent channel
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
