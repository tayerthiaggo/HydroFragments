# Spatial exports

Optional GIS-ready vector and raster products complement the canonical tidy metric tables. Spatial exports are **off by default** so the default tabular workflow pays no polygonization, checkpoint, or serialization cost.

## Quick decision guide

| Goal | Use |
|------|-----|
| Statistical modelling, dashboards, database loads | Metric tables (`metrics/` Parquet or CSV) |
| Cartography in QGIS/ArcGIS | GeoTIFF rasters and GeoPackage vectors |
| Multidimensional scientific exchange | Opt-in NetCDF (`pip install hydrofragments[netcdf]`) |

## Configuration

Use configuration schema **`1.1.0`**. Spatial product selection belongs in `execution_config()` and does not change the scientific configuration hash.

```python
from hydrofragments import HydroConfig

config = HydroConfig.from_mapping(
    {
        "config_schema_version": "1.1.0",
        "input": {"kind": "generic_binary"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "supplied",
            "composite_owner": "caller",
        },
        "output": {
            "output_dir": "runs/demo_01",  # required when spatial_products is non-empty
            "formats": ["parquet"],        # validated: parquet, csv
            "spatial_products": [],        # default: exports off
            "raster_formats": ["geotiff"], # geotiff (default) and/or netcdf
        },
    }
)
```

### Opt-in examples

Persistence rasters only:

```python
"output": {
    "output_dir": "runs/persistence",
    "spatial_products": ["persistence_rasters"],
}
```

Monthly pool polygons:

```python
"output": {
    "output_dir": "runs/pools",
    "spatial_products": ["monthly_pools"],
}
```

All products appropriate for a georeferenced cube with hydrological-year inputs:

```python
"output": {
    "output_dir": "runs/full",
    "spatial_products": [
        "monthly_pools",
        "zones",
        "persistence_rasters",
        "temporal_rasters",
        "refuge_stability_rasters",
        "reach_profiles",
        "riverscape_evidence",
    ],
}
```

> `riverscape_evidence` is the only product with a *workflow-mode* prerequisite: it requires an `analyze_from_dea` run whose `riverscape.mode` resolved to riverscape zoning. Requesting it on an occurrence run fails preflight.

The deprecated `include_vectors: true` alias maps to `monthly_pools` for one schema cycle. Do not set both to conflicting values.

## Output layout

`output_dir` names the **final run directory**, not a shared parent folder. The directory must be absent or empty before analysis starts. HydroFragments stages artifacts, validates them, then commits the bundle with one directory rename. `run_manifest.json` is written **last**.

```text
<output_dir>/
  config.json
  metrics/
  metric_coverage.csv
  vectors/spatial.gpkg
    monthly_pools
    zones
    reaches
    reach_wet_monthly
    channel_bridges                 # riverscape_evidence only
  rasters/landform.tif              # riverscape_evidence only
  rasters/hydroperiod.tif           # riverscape_evidence only
  rasters/zone_crosstab.tif         # riverscape_evidence only
  rasters/channel_source.tif        # riverscape_evidence only
  rasters/channel_confidence.tif    # riverscape_evidence only
  rasters/rem.tif                   # riverscape_evidence only
  rasters/occurrence.tif
  rasters/valid_observation_count.tif
  rasters/refuge_mask.tif
  rasters/zones.tif
  rasters/recurrence.tif
  rasters/recurrence_valid_year_count.tif
  rasters/hydroperiod_by_year.tif
  rasters/hydroperiod_valid_month_count_by_year.tif
  rasters/refuge_overlap_by_hy.tif
  rasters/refuge_stability_frequency.tif
  rasters/refuge_stability_union_pair_count.tif
  rasters/spatial.nc              # when raster_formats includes netcdf
  run_manifest.json
```

Only requested and applicable paths are created. Unavailable requested products fail **preflight** with `SpatialProductUnavailable` rather than silently omitting files.

## CRS and grid

Exports preserve the source cube grid and CRS. HydroFragments does **not** reproject during export. A cube without resolvable CRS/transform fails early when spatial output is requested; tabular analysis remains allowed.

Every raster and vector product is validated against a frozen `SpatialGrid` contract (CRS, affine transform, coordinate order, shape). Equal-shaped arrays with a shifted transform are rejected.

## Product reference

### `persistence_rasters`

| Artifact | dtype | nodata | units |
|----------|------:|--------|-------|
| `rasters/occurrence.tif` | float32 | NaN | percent, 0–100 |
| `rasters/valid_observation_count.tif` | uint32 | 4294967295 | months |
| `rasters/refuge_mask.tif` | uint8 | 255 | 0=false, 1=true |

**Prerequisites:** Georeferenced water cube with valid CRS/transform.

**Workflow:** `analyze()` and `analyze_from_dea()` when cube grid is valid.

**Cost:** Moderate I/O; counters are accumulated in the monthly pass (same pass as scalar persistence metrics). Enabling exports reuses completed checkpoints rather than re-reading the cube.

### `temporal_rasters`

| Artifact | dtype | nodata | units |
|----------|------:|--------|-------|
| `rasters/recurrence.tif` | float32 | NaN | percent, 0–100 |
| `rasters/recurrence_valid_year_count.tif` | uint16 | 65535 | calendar years |
| `rasters/hydroperiod_by_year.tif` | float32 | NaN | fraction, 0–1 (multi-band by calendar year) |
| `rasters/hydroperiod_valid_month_count_by_year.tif` | uint8 | 255 | months, 0–12 |

**Prerequisites:** Georeferenced cube; sufficient temporal record for the estimator.

### `refuge_stability_rasters`

| Artifact | dtype | nodata | units / codes |
|----------|------:|--------|---------------|
| `rasters/refuge_overlap_by_hy.tif` | uint8 | 255 | 0=dry, 1=lost, 2=new, 3=stable (multi-band by HY pair) |
| `rasters/refuge_stability_frequency.tif` | float32 | NaN | percent, 0–100 per-pixel stability frequency |
| `rasters/refuge_stability_union_pair_count.tif` | uint16 | 65535 | valid HY pairs wet in either year |

**Prerequisites:** Hydrological-year anchors from `hydroyear_extent` and at least two valid end-dry states.

**Nodata semantics:** Pixels with zero eligible HY pairs are nodata. The per-pixel frequency raster is **not** the scalar Jaccard `refuge_spatial_stability` metric.

### `monthly_pools`

**Path:** `vectors/spatial.gpkg` layer `monthly_pools`

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime64[ns] | Month timestamp |
| `pool_id` | string | `YYYY-MM-DD:<window_id>:<label_id>` |
| `label_id` | int32 | Connected-component label |
| `n_pixels` | int32 | Pixel count |
| `area_m2` | float64 | Polygon area |
| `perimeter_m` | float64 | Perimeter |
| `major_axis_length_m` | float64 | Major axis |
| `width_m` | float64 | Nullable width |
| `elongation_ratio` | float64 | Nullable |
| `shape_index` | float64 | Nullable |
| `geometry` | Polygon/MultiPolygon | Source CRS |

Aggregate monthly metrics (AWRE, AWMSI) are **not** duplicated on every feature. Polygon area and count match the measured label properties within raster/vector tolerance.

**Checkpoint-only design:** Pool polygons are polygonized during the monthly pass into durable checkpoint partitions, then streamed into the GeoPackage. `HydroResult.write()` and `write_output_tables()` reject an in-memory run-wide `GeoDataFrame`.

### `zones`

**Paths:** `vectors/spatial.gpkg` layer `zones`, `rasters/zones.tif`

| Column | Type | Notes |
|---|---|---|
| `zone_id` | int | 1–4 |
| `zone_name` | str | Mode-keyed, see below |
| `area_km2` | float | Dissolved polygon area |
| `source` | str | Zone provenance (DEA product id, or `occurrence`) |
| `mode` | str | `occurrence` or `riverscape` |
| `landform` | int or null | Always null, see below |
| `hydroperiod` | int or null | Always null, see below |
| `geometry` | MultiPolygon | Analysis CRS |

Zone names depend on the zoning mode:

| Mode | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `occurrence` | `channel_connected` | `persistent` | `seasonal` | `ephemeral` |
| `riverscape` | `in_channel` | `persistent_off_channel` | `seasonal_floodplain` | `marginal_floodplain` |

`landform` and `hydroperiod` are **always null** on this layer, in both modes. Zone polygons are dissolved per `zone_id`, and a dissolved riverscape Zone 1 mixes hydroperiod 4 (bridged, unobserved) with hydroperiods 1–3, so no single value is truthful at polygon granularity. The columns exist anyway so the GeoPackage schema does not change between modes. The authoritative per-pixel values ship as the `riverscape_evidence` rasters.

**Prerequisites:** Explicit zone input (`AnalysisInputs.zones` or DEA workflow zone result). Unavailable from a cube-only `analyze()` call.

### `riverscape_evidence`

**Paths:** six GeoTIFFs under `rasters/`, plus the `channel_bridges` layer in `vectors/spatial.gpkg`.

| File | dtype | nodata | Meaning |
|---|---|---|---|
| `landform.tif` | uint8 | 0 | 0=outside, 1=in_channel, 2=off_channel_riverine, 3=non_riverine |
| `hydroperiod.tif` | uint8 | 0 | 0=outside, 1=persistent, 2=seasonal, 3=marginal, 4=unobserved (bridged) |
| `zone_crosstab.tif` | uint16 | 0 | `landform * 10 + hydroperiod`; 0 outside |
| `channel_source.tif` | uint8 | 255 | 0=not channel, 1=observed, 2=bridged |
| `channel_confidence.tif` | uint8 | 255 | 0–3 inside channel; 255 outside channel |
| `rem.tif` | float32 | NaN | Relative elevation above the channel, metres |

All six share the `zones.tif` grid (same CRS, transform, and shape).

**Prerequisites:** An `analyze_from_dea` run whose `riverscape.mode` resolved to riverscape zoning — that is, `auto` or `required` with drainage and the riverscape source products both available. `mode="off"`, and any `auto` run that fell back to occurrence zoning, cannot produce these layers, and requesting the product on such a run fails preflight with `SpatialProductUnavailable`. The product is **not** added by default: a run with `config=None` stays occurrence-only and writes no evidence.

### `channel_bridges`

**Path:** `vectors/spatial.gpkg` layer `channel_bridges`

One LineString per reconstructed channel gap, in the analysis CRS (EPSG:3577).

| Column | Type | Notes |
|---|---|---|
| `gap_id` | int | Gap identifier |
| `reach_ids` | str | Comma-joined `HydroID`s the gap spans |
| `length_m` | float | Bridge centreline length |
| `cumulative_cost` | float | Least-cost path cost |
| `cost_per_m` | float | `cumulative_cost / length_m` |
| `upstream_width_m` | float | Corridor width at the upstream anchor |
| `downstream_width_m` | float | Corridor width at the downstream anchor |
| `bridge_confidence` | int | 1 or 2 |
| `gap_cause` | str | `vegetated`, `narrow`, or `unobserved` |
| `geometry` | LineString | Analysis CRS |

**Prerequisites:** Same gate as `riverscape_evidence`, so bridges are never written without the rasters that explain them. When no gap was bridged the layer is still created, empty, with the full schema.

### Manifest `zoning`

Every run that produces a `ZoneResult` — or that has a recorded reason for producing none — writes a top-level `zoning` object into `run_manifest.json`. A riverscape run records the full science provenance (ruleset versions, source products, floodplain envelope, pixel counts, bridge and reach counts, bridged length/area, non-riverine area). An occurrence run records the thin subset: `mode`, `domain_pixel_count`, `domain_digest`, `degraded_reasons`, `zone_source`, with riverscape-only keys omitted rather than emitted empty. A run whose zoning returned nothing records only `mode` (the configured `riverscape.mode`) and `degraded_reasons`.

### `reach_profiles`

**Paths:** `vectors/spatial.gpkg` layers `reaches` (geometry) and `reach_wet_monthly` (non-spatial table keyed by `reach_id`, `date` with `is_wet`, `length_m`, `lpsec_contribution_pct`).

**Prerequisites:** Real channel context (`SpatialContext` with drainage geometry). Unavailable from a cube-only `analyze()` call.

## Performance and storage

Controlled benchmark evidence (synthetic fixtures, export-off median regression ≤10%, all-products peak RSS ≤125% of core) is recorded in `benchmarks/results/dynamics_spatial_exports.md`.

Rules of thumb:

- **Export off:** No polygonization, vector checkpoint writes, or raster serialization.
- **Export on:** Reuses the same monthly materialization and label pass as metrics; no second full-cube read.
- **GeoTIFF default:** Tiled 256×256, DEFLATE compression — not advertised as Cloud Optimized GeoTIFF (COG).
- **NetCDF:** Opt-in extra; single write pass, slower than GeoTIFF for large grids.

## Opening outputs

### Python

```python
from pathlib import Path

import geopandas as gpd
import rioxarray  # noqa: F401
import xarray as xr

from hydrofragments.output.manifest import validate_result_bundle

bundle = Path("runs/demo_01")
manifest = validate_result_bundle(bundle)
print(manifest["manifest_schema_version"])  # 1.1.0

occurrence = xr.open_dataarray(bundle / "rasters" / "occurrence.tif")
pools = gpd.read_file(bundle / "vectors" / "spatial.gpkg", layer="monthly_pools")
```

### QGIS / GDAL

```bash
gdalinfo runs/demo_01/rasters/occurrence.tif
ogrinfo -al -so runs/demo_01/vectors/spatial.gpkg monthly_pools
```

Load rasters with **Render type → Singleband pseudocolor** for percent products. Check band descriptions for hydroperiod calendar years and refuge-overlap HY pairs.

## Version boundaries

| Version | Scope |
|---------|-------|
| Package | `0.1.0` (`hydrofragments.__version__`) |
| Config schema | `1.0.0` (accepted), `1.1.0` (spatial products) |
| Metric row schema | `1.1.0` (new `EdgeFlag` values for dynamics) |
| Run manifest schema | `1.1.0` (artifact inventory with digests) |

Readers accept legacy metric/manifest `1.0.0` datasets under their original contracts.

## See also

- [Dynamics metrics](metrics/dynamics.md) — reconnection timing and refuge stability scalars
- [Offline example](../examples/spatial_exports.py) — synthetic cube, bundle write, manifest validation
- [Final metrics covered](final_metrics_covered.md) — metric availability matrix
