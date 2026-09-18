"""Single owner for validated result-bundle publication from both public workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from rasterio import features as rio_features
from rasterio.crs import CRS as RioCRS
from shapely.geometry import shape as shapely_shape

from hydrofragments._version import __version__
from hydrofragments.config import HydroConfig
from hydrofragments.models import AnalysisInputs, HydroResult, WaterCube
from hydrofragments.output.bundle import (
    ArtifactRegistration,
    BundleError,
    assert_output_dir_available,
    open_bundle_transaction,
)
from hydrofragments.output.core import CoreAnalysisResult, build_in_memory_manifest
from hydrofragments.output.manifest import (
    build_run_manifest,
    build_zoning_section,
    validate_result_bundle,
)
from hydrofragments.output.riverscape_export import RiverscapeExportBundle
from hydrofragments.riverscape.bridging import _empty_bridge_frame
from hydrofragments.output.rasters import export_rasters_from_checkpoint
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.output.tables import write_metric_coverage, write_output_tables
from hydrofragments.output.vectors import (
    MONTHLY_POOLS_LAYER,
    SPATIAL_GPKG_NAME,
    export_vectors_from_checkpoint,
)
from hydrofragments.spatial import SpatialContext, ZoneResult
from hydrofragments.spatial.zones import ZoneResult as ZoneResultType
from hydrofragments.temporal.hydroyear import HyAnchorResult

ZONES_LAYER = "zones"
REACHES_LAYER = "reaches"
REACH_WET_MONTHLY_LAYER = "reach_wet_monthly"
CHANNEL_BRIDGES_LAYER = "channel_bridges"

#: Zone names by zoning mode (spec 6b section 3.4). Occurrence keeps the
#: names the bundle has always written; riverscape uses the parent spec's
#: landform-aware vocabulary, so a Zone 2 polygon is not mislabelled
#: "persistent" when it actually means "persistent off-channel".
_ZONE_NAMES_BY_MODE: dict[str, dict[int, str]] = {
    "occurrence": {
        1: "channel_connected",
        2: "persistent",
        3: "seasonal",
        4: "ephemeral",
    },
    "riverscape": {
        1: "in_channel",
        2: "persistent_off_channel",
        3: "seasonal_floodplain",
        4: "marginal_floodplain",
    },
}


def _zone_names_for(mode: str) -> dict[int, str]:
    """Zone-id -> name table for one zoning mode.

    An unrecognised mode falls back to the occurrence table rather than
    raising: ``ZoneResult.__post_init__`` already rejects any mode outside
    ``{"occurrence", "riverscape"}``, so reaching the fallback means a
    caller hand-built a frame, and a legacy-named export beats a crash in
    the writer.
    """
    return _ZONE_NAMES_BY_MODE.get(mode, _ZONE_NAMES_BY_MODE["occurrence"])


class SpatialProductUnavailable(ValueError):
    """Raised when a requested spatial product lacks runtime prerequisites."""


def _resolve_zone_result(
    inputs: AnalysisInputs,
    *,
    zone_result: ZoneResult | None,
) -> ZoneResult | None:
    if zone_result is not None:
        return zone_result
    if inputs.zones is not None:
        return inputs.zones
    return None


def preflight_spatial_outputs(
    config: HydroConfig,
    *,
    cube: WaterCube,
    inputs: AnalysisInputs,
    hydroyear_result: HyAnchorResult | None,
    zone_result: ZoneResult | None = None,
    riverscape_bundle: RiverscapeExportBundle | None = None,
) -> SpatialGrid | None:
    """Validate spatial products, source grid, writers, and output paths."""

    config.validate_output_preflight()
    products = config.output.spatial_products
    if not products and config.output.output_dir is None:
        return None

    grid: SpatialGrid | None = None
    if products:
        template = cube.water.isel(time=0)
        grid = SpatialGrid.from_dataarray(template, require_georeference=True)

    resolved_zones = _resolve_zone_result(inputs, zone_result=zone_result)
    drainage = inputs.drainage
    has_channel = isinstance(drainage, SpatialContext) and drainage.has_real_channel

    for product in products:
        if product == "monthly_pools":
            if grid is None:
                raise SpatialProductUnavailable(
                    "monthly_pools requires a georeferenced water cube grid"
                )
            if config.patches.min_patch_pixels < 1:
                raise SpatialProductUnavailable(
                    "monthly_pools requires patches.min_patch_pixels >= 1"
                )
        elif product == "zones":
            if resolved_zones is None:
                raise SpatialProductUnavailable(
                    "zones requires an explicit zone input or DEA zone derivation"
                )
            if resolved_zones.grid is None:
                raise SpatialProductUnavailable(
                    "zones requires a georeferenced zone mask grid"
                )
        elif product == "persistence_rasters":
            if grid is None:
                raise SpatialProductUnavailable(
                    "persistence_rasters requires a georeferenced water cube grid"
                )
        elif product == "temporal_rasters":
            if grid is None:
                raise SpatialProductUnavailable(
                    "temporal_rasters requires a georeferenced water cube grid"
                )
        elif product == "refuge_stability_rasters":
            if grid is None:
                raise SpatialProductUnavailable(
                    "refuge_stability_rasters requires a georeferenced water cube grid"
                )
            if hydroyear_result is None or len(hydroyear_result.anchors) < 2:
                raise SpatialProductUnavailable(
                    "refuge_stability_rasters requires at least two hydrological-year anchors"
                )
        elif product == "reach_profiles":
            if not has_channel:
                raise SpatialProductUnavailable(
                    "reach_profiles requires a real channel SpatialContext and drainage"
                )
            if (
                inputs.channel_wet_profiles is None
                or inputs.channel_segment_lengths_m is None
            ):
                raise SpatialProductUnavailable(
                    "reach_profiles requires channel_wet_profiles and "
                    "channel_segment_lengths_m"
                )
        elif product == "riverscape_evidence":
            if riverscape_bundle is None:
                raise SpatialProductUnavailable(
                    "riverscape_evidence requires a riverscape zoning run; "
                    "no RiverscapeExportBundle was produced"
                )
            if riverscape_bundle.landform.grid is None:
                raise SpatialProductUnavailable(
                    "riverscape_evidence requires a georeferenced landform grid"
                )
        else:
            raise SpatialProductUnavailable(f"unsupported spatial product: {product}")

    if config.output.output_dir is not None:
        assert_output_dir_available(Path(config.output.output_dir))
        output_root = Path(config.output.output_dir)
        collision_paths = [
            output_root / "run_manifest.json",
            output_root / "config.json",
            output_root / "metrics",
            output_root / "vectors" / SPATIAL_GPKG_NAME,
            output_root / "rasters",
        ]
        for path in collision_paths:
            if path.exists():
                raise BundleError(f"refusing to overwrite existing artifact: {path}")

    return grid


_ZONE_COLUMNS = [
    "zone_id",
    "zone_name",
    "area_km2",
    "source",
    "mode",
    "landform",
    "hydroperiod",
    "geometry",
]


def _zones_geodataframe(
    zone_result: ZoneResultType, *, pixel_size_m: float
) -> gpd.GeoDataFrame:
    """Dissolve the zone mask into one polygon per zone id.

    ``landform``/``hydroperiod`` are always ``None`` (spec 6b section 3.4).
    A dissolved Zone 1 multipolygon mixes hydroperiod 4 (bridged, unobserved)
    with 1-3, so no single value is honest at this granularity; the
    authoritative per-pixel layers ship as ``riverscape_evidence``. The
    columns exist in both modes so the GPKG schema is stable.
    """
    if zone_result.grid is None:
        raise SpatialProductUnavailable("zones vector export requires a georeferenced zone mask")
    grid = zone_result.grid
    mask = np.asarray(zone_result.mask, dtype=np.uint8)
    zone_names = _zone_names_for(zone_result.mode)
    cell_area_m2 = float(pixel_size_m) ** 2
    rows: list[dict[str, object]] = []
    for zone_id in sorted(set(int(value) for value in np.unique(mask) if int(value) > 0)):
        zone_pixels = mask == zone_id
        shapes = rio_features.shapes(
            zone_pixels.astype(np.uint8),
            mask=zone_pixels,
            transform=grid.transform,
        )
        geometries = [shapely_shape(geom) for geom, value in shapes if int(value) == 1]
        if not geometries:
            continue
        geometry = geometries[0] if len(geometries) == 1 else gpd.GeoSeries(geometries).union_all()
        area_m2 = float(zone_pixels.sum()) * cell_area_m2
        rows.append(
            {
                "zone_id": zone_id,
                "zone_name": zone_names.get(zone_id, f"zone_{zone_id}"),
                "area_km2": area_m2 / 1_000_000.0,
                "source": zone_result.source,
                "mode": zone_result.mode,
                "landform": None,
                "hydroperiod": None,
                "geometry": geometry,
            }
        )
    if not rows:
        return gpd.GeoDataFrame(
            columns=_ZONE_COLUMNS,
            geometry="geometry",
            crs=grid.crs,
        )
    return gpd.GeoDataFrame(rows, columns=_ZONE_COLUMNS, geometry="geometry", crs=grid.crs)


def _riverscape_evidence_arrays(
    bundle: RiverscapeExportBundle,
) -> dict[str, np.ndarray]:
    """Band name -> array for the ``riverscape_evidence`` product.

    Every array comes straight out of the bundle. ``zone_crosstab`` is
    upcast from ``combine_zones``'s ``uint8`` to ``uint16`` (decision L4):
    the widest code today is ``34``, but a uint16 band leaves headroom if
    the landform vocabulary ever grows past 25 codes, and costs nothing at
    these raster sizes.
    """
    landform = bundle.landform
    return {
        "landform": np.asarray(landform.landform, dtype=np.uint8),
        "hydroperiod": np.asarray(bundle.hydroperiod_classes, dtype=np.uint8),
        "zone_crosstab": np.asarray(bundle.zone_result.crosstab, dtype=np.uint16),
        "channel_source": np.asarray(landform.channel_source, dtype=np.uint8),
        "channel_confidence": np.asarray(landform.channel_confidence, dtype=np.uint8),
        "rem": np.asarray(landform.rem, dtype=np.float32),
    }


def _same_crs(left: object, right: object) -> bool:
    try:
        left_crs = RioCRS.from_user_input(left)
        right_crs = RioCRS.from_user_input(right)
    except Exception:
        return str(left) == str(right)
    left_epsg = left_crs.to_epsg()
    right_epsg = right_crs.to_epsg()
    if left_epsg is not None and right_epsg is not None:
        return left_epsg == right_epsg
    return bool(left_crs == right_crs)


#: Column -> pandas dtype for the ``channel_bridges`` layer. Pinned so an
#: empty frame (all-object columns out of ``_empty_bridge_frame``) writes a
#: GPKG layer with the same field types a populated frame would.
_BRIDGE_COLUMN_DTYPES: dict[str, str] = {
    "gap_id": "int64",
    "reach_ids": "object",
    "length_m": "float64",
    "cumulative_cost": "float64",
    "cost_per_m": "float64",
    "upstream_width_m": "float64",
    "downstream_width_m": "float64",
    "bridge_confidence": "int64",
    "gap_cause": "object",
}


def _channel_bridges_geodataframe(
    bundle: RiverscapeExportBundle, *, crs
) -> gpd.GeoDataFrame:
    """Normalize ``LandformResult.channel_bridges`` for GPKG publication.

    An absent frame (a stubbed ``LandformResult``) becomes the typed-empty
    schema from ``bridging._empty_bridge_frame``, so the layer exists with
    its full field list even when no gap was bridged (spec 6b section 3.6).

    A frame already carrying a different CRS is a bug upstream, not
    something to fix here: Phase 6a builds bridge geometry directly on the
    analysis grid, so a mismatch means the wrong frame arrived. Spec section
    5 prefers raising over silently reprojecting.
    """
    frame = bundle.landform.channel_bridges
    if frame is None:
        frame = _empty_bridge_frame(crs)
    frame = gpd.GeoDataFrame(frame).copy()
    if frame.crs is not None and not _same_crs(frame.crs, crs):
        raise SpatialProductUnavailable(
            "channel_bridges CRS does not match the riverscape export grid CRS"
        )
    if frame.crs is None:
        frame = frame.set_crs(crs)
    missing = [name for name in _BRIDGE_COLUMN_DTYPES if name not in frame.columns]
    if missing:
        raise SpatialProductUnavailable(
            f"channel_bridges is missing required columns: {missing}"
        )
    for column, dtype in _BRIDGE_COLUMN_DTYPES.items():
        frame[column] = frame[column].astype(dtype)
    return frame[[*_BRIDGE_COLUMN_DTYPES, "geometry"]]


def _reach_wet_monthly_table(
    *,
    context: SpatialContext,
    cube: WaterCube,
    wet_profiles: Sequence[Sequence[bool]],
    segment_lengths_m: Sequence[float],
) -> pd.DataFrame:
    drainage = context.drainage
    if drainage is None:
        raise SpatialProductUnavailable("reach_profiles requires drainage geometry")
    reach_ids = drainage["HydroID"].astype(str).tolist()
    lengths = np.asarray(segment_lengths_m, dtype=float)
    wet = np.asarray(wet_profiles, dtype=bool)
    times = pd.to_datetime(cube.water["time"].values)
    rows: list[dict[str, object]] = []
    for time_index, timestamp in enumerate(times):
        month_wet = wet[time_index]
        wetted_length_m = float(lengths[month_wet].sum())
        total_length_m = float(lengths.sum())
        lpsec_pct = (
            100.0 * wetted_length_m / total_length_m if total_length_m > 0 else float("nan")
        )
        for reach_index, reach_id in enumerate(reach_ids):
            rows.append(
                {
                    "reach_id": reach_id,
                    "date": pd.Timestamp(timestamp),
                    "is_wet": bool(month_wet[reach_index]),
                    "length_m": float(lengths[reach_index]),
                    "lpsec_contribution_pct": lpsec_pct if month_wet[reach_index] else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _write_spatial_vectors(
    staging_root: Path,
    *,
    config: HydroConfig,
    core: CoreAnalysisResult,
    cube: WaterCube,
    inputs: AnalysisInputs,
    zone_result: ZoneResult | None,
    riverscape_bundle: RiverscapeExportBundle | None,
    pixel_size_m: float,
) -> list[ArtifactRegistration]:
    products = set(config.output.spatial_products)
    if not products:
        return []

    registrations: list[ArtifactRegistration] = []
    vectors_dir = staging_root / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = vectors_dir / SPATIAL_GPKG_NAME
    if gpkg_path.exists():
        raise BundleError(f"refusing to overwrite existing artifact: {gpkg_path}")

    if "monthly_pools" in products:
        if core.pool_checkpoint_root is None:
            raise SpatialProductUnavailable(
                "monthly_pools export requires a completed pool vector checkpoint"
            )
        export_vectors_from_checkpoint(core.pool_checkpoint_root, vectors_dir)
        registrations.append(
            ArtifactRegistration(
                name="spatial_vectors",
                relative_path=f"vectors/{SPATIAL_GPKG_NAME}",
                media_type="application/geopackage+sqlite3",
            )
        )

    resolved_zones = _resolve_zone_result(inputs, zone_result=zone_result)
    if "zones" in products and resolved_zones is not None:
        zones_gdf = _zones_geodataframe(resolved_zones, pixel_size_m=pixel_size_m)
        pyogrio.write_dataframe(
            zones_gdf,
            gpkg_path,
            layer=ZONES_LAYER,
            driver="GPKG",
            encoding="UTF-8",
            append=gpkg_path.exists(),
        )
        registrations.append(
            ArtifactRegistration(
                name="zones_vector",
                relative_path=f"vectors/{SPATIAL_GPKG_NAME}",
                media_type="application/geopackage+sqlite3",
            )
        )

    if "reach_profiles" in products:
        context = inputs.drainage
        if not isinstance(context, SpatialContext) or context.drainage is None:
            raise SpatialProductUnavailable("reach_profiles requires channel drainage geometry")
        reaches = context.drainage.copy()
        reaches["reach_id"] = reaches["HydroID"].astype(str)
        reach_table = _reach_wet_monthly_table(
            context=context,
            cube=cube,
            wet_profiles=inputs.channel_wet_profiles or (),
            segment_lengths_m=inputs.channel_segment_lengths_m or (),
        )
        pyogrio.write_dataframe(
            reaches,
            gpkg_path,
            layer=REACHES_LAYER,
            driver="GPKG",
            encoding="UTF-8",
            append=gpkg_path.exists(),
        )
        pyogrio.write_dataframe(
            gpd.GeoDataFrame(reach_table, geometry=None),
            gpkg_path,
            layer=REACH_WET_MONTHLY_LAYER,
            driver="GPKG",
            encoding="UTF-8",
            append=True,
        )
        registrations.append(
            ArtifactRegistration(
                name="reach_profiles",
                relative_path=f"vectors/{SPATIAL_GPKG_NAME}",
                media_type="application/geopackage+sqlite3",
            )
        )

    if "riverscape_evidence" in products:
        if riverscape_bundle is None:
            raise SpatialProductUnavailable(
                "riverscape_evidence requires a riverscape zoning run; "
                "no RiverscapeExportBundle was produced"
            )
        grid = riverscape_bundle.landform.grid
        if grid is None:
            raise SpatialProductUnavailable(
                "riverscape_evidence requires a georeferenced landform grid"
            )
        bridges_gdf = _channel_bridges_geodataframe(riverscape_bundle, crs=grid.crs)
        pyogrio.write_dataframe(
            bridges_gdf,
            gpkg_path,
            layer=CHANNEL_BRIDGES_LAYER,
            driver="GPKG",
            encoding="UTF-8",
            geometry_type="LineString",
            append=gpkg_path.exists(),
        )
        registrations.append(
            ArtifactRegistration(
                name="channel_bridges",
                relative_path=f"vectors/{SPATIAL_GPKG_NAME}",
                media_type="application/geopackage+sqlite3",
            )
        )

    return registrations


def _write_spatial_rasters(
    staging_root: Path,
    *,
    config: HydroConfig,
    core: CoreAnalysisResult,
    inputs: AnalysisInputs,
    zone_result: ZoneResult | None,
    riverscape_bundle: RiverscapeExportBundle | None,
    analysis_mask: np.ndarray | None,
) -> list[ArtifactRegistration]:
    products = set(config.output.spatial_products)
    raster_products = products & {
        "persistence_rasters",
        "temporal_rasters",
        "refuge_stability_rasters",
        "zones",
        "riverscape_evidence",
    }
    wants_zones = "zones" in products
    wants_evidence = "riverscape_evidence" in products
    if not raster_products:
        return []
    if core.raster_checkpoint is None and not wants_zones and not wants_evidence:
        return []

    raster_dir = staging_root / "rasters"
    registrations: list[ArtifactRegistration] = []
    zone_mask = None
    resolved_zones = _resolve_zone_result(inputs, zone_result=zone_result)
    if "zones" in config.output.spatial_products and resolved_zones is not None:
        zone_mask = np.asarray(resolved_zones.mask, dtype=np.uint8)

    if core.raster_checkpoint is not None:
        artifacts = export_rasters_from_checkpoint(
            core.raster_checkpoint,
            raster_dir,
            config=config,
            raster_formats=config.output.raster_formats,
            zone_mask=zone_mask if "zones" in config.output.spatial_products else None,
            analysis_mask=analysis_mask,
        )
        for name, path in artifacts.items():
            registrations.append(
                ArtifactRegistration(
                    name=name,
                    relative_path=str(path.relative_to(staging_root)).replace("\\", "/"),
                    media_type="image/tiff" if path.suffix.lower() in {".tif", ".tiff"} else None,
                )
            )
    elif zone_mask is not None:
        from hydrofragments.output.rasters import RASTER_PRODUCT_CONTRACTS, write_zones_geotiff

        if core.spatial_grid is None:
            raise SpatialProductUnavailable("zones raster export requires a spatial grid")
        zones_path = raster_dir / RASTER_PRODUCT_CONTRACTS["zones"].filename
        write_zones_geotiff(
            zone_mask,
            zones_path,
            grid=core.spatial_grid,
            metadata={
                "algorithm_version": "1.0.0",
                "scientific_config_hash": config.config_hash,
            },
        )
        registrations.append(
            ArtifactRegistration(
                name="zones",
                relative_path=f"rasters/{RASTER_PRODUCT_CONTRACTS['zones'].filename}",
                media_type="image/tiff",
            )
        )

    if wants_evidence:
        if riverscape_bundle is None:
            raise SpatialProductUnavailable(
                "riverscape_evidence requires a riverscape zoning run; "
                "no RiverscapeExportBundle was produced"
            )
        evidence_grid = riverscape_bundle.landform.grid
        if evidence_grid is None:
            raise SpatialProductUnavailable(
                "riverscape_evidence requires a georeferenced landform grid"
            )
        from hydrofragments.output.rasters import write_riverscape_evidence_geotiffs

        evidence_paths = write_riverscape_evidence_geotiffs(
            _riverscape_evidence_arrays(riverscape_bundle),
            raster_dir,
            grid=evidence_grid,
            metadata={
                "algorithm_version": "1.0.0",
                "scientific_config_hash": config.config_hash,
            },
        )
        for name, path in evidence_paths.items():
            registrations.append(
                ArtifactRegistration(
                    name=name,
                    relative_path=str(path.relative_to(staging_root)).replace("\\", "/"),
                    media_type="image/tiff",
                )
            )

    return registrations


def finalize_analysis_bundle(
    config: HydroConfig,
    core: CoreAnalysisResult,
    *,
    cube: WaterCube,
    inputs: AnalysisInputs | None = None,
    pixel_size_m: float = 30.0,
    zone_result: ZoneResult | None = None,
    riverscape_bundle: RiverscapeExportBundle | None = None,
    zoning_reasons: Sequence[str] = (),
    dea_provenance: Mapping[str, object] | None = None,
    timings_seconds: Mapping[str, float] | None = None,
    peak_rss_bytes: int | None = None,
) -> HydroResult:
    """Publish tables, spatial products, and one validated manifest atomically."""

    inputs = inputs or AnalysisInputs()
    output_dir = Path(config.output.output_dir)
    analysis_mask_np = None
    if cube.analysis_mask is not None:
        analysis_mask_np = np.asarray(cube.analysis_mask.values, dtype=bool)

    write_started = time.perf_counter()
    transaction = open_bundle_transaction(
        output_dir, run_id=core.run_id, config=config
    )
    try:
        write_output_tables(
            core.metrics_table,
            transaction.root,
            formats=config.output.formats,
            export_csv="csv" in config.output.formats,
        )
        transaction.register_artifact(
            ArtifactRegistration(name="metrics", relative_path="metrics")
        )
        write_metric_coverage(core.metric_coverage, transaction.root)
        transaction.register_artifact(
            ArtifactRegistration(
                name="metric_coverage",
                relative_path="metric_coverage.csv",
                media_type="text/csv",
            )
        )

        vector_regs = _write_spatial_vectors(
            transaction.root,
            config=config,
            core=core,
            cube=cube,
            inputs=inputs,
            zone_result=zone_result,
            riverscape_bundle=riverscape_bundle,
            pixel_size_m=pixel_size_m,
        )
        for registration in vector_regs:
            transaction.register_artifact(registration)

        raster_regs = _write_spatial_rasters(
            transaction.root,
            config=config,
            core=core,
            inputs=inputs,
            zone_result=zone_result,
            riverscape_bundle=riverscape_bundle,
            analysis_mask=analysis_mask_np,
        )
        for registration in raster_regs:
            transaction.register_artifact(registration)

        transaction.write_config()
        transaction.register_artifact(
            ArtifactRegistration(
                name="config",
                relative_path="config.json",
                media_type="application/json",
            )
        )

        resolved_timings = dict(timings_seconds or {})
        resolved_timings["output_write"] = time.perf_counter() - write_started
        if resolved_timings.keys() - {"total"}:
            resolved_timings["total"] = sum(
                value
                for key, value in resolved_timings.items()
                if key != "total"
            )

        resolved_zones = _resolve_zone_result(inputs, zone_result=zone_result)
        zoning_section = build_zoning_section(
            resolved_zones,
            riverscape_bundle,
            pixel_m=pixel_size_m,
            configured_mode=config.riverscape.mode,
            workflow_reasons=tuple(zoning_reasons),
        )
        if resolved_zones is None and not zoning_section["degraded_reasons"]:
            zoning_section = None  # nothing to say: no zones, no reasons

        artifacts = transaction.finalize(
            package_version=__version__,
            git_sha=core.git_sha,
            input_fingerprint=core.input_fingerprint,
            planned_backend=str(core.execution_plan_mapping.get("planned_backend", "cpu")),
            actual_backend_by_stage=dict(
                core.execution_plan_mapping.get("actual_backend_by_stage", {})
            ),
            backend_capabilities=dict(
                core.execution_plan_mapping.get("backend_capabilities", {})
            ),
            skipped_metrics=[
                {"metric_id": metric_id, "reason": reason}
                for metric_id, reason in core.skipped_metrics
            ],
            warnings=list(core.report_warnings),
            comparison_context=core.comparison_context,
            timings_seconds=resolved_timings,
            dea_provenance=dict(dea_provenance) if dea_provenance else None,
            zoning=zoning_section,
            peak_rss_bytes=peak_rss_bytes,
        )
    except Exception:
        transaction.abort()
        raise

    manifest = validate_result_bundle(output_dir)
    manifest_dict = dict(manifest)
    manifest_dict["manifest_path"] = str(output_dir / "run_manifest.json")
    return HydroResult(
        metrics_table=core.metrics_table,
        manifest=manifest_dict,
        output_dir=output_dir,
        run_id=core.run_id,
        metric_coverage=core.metric_coverage,
    )


__all__ = [
    "CoreAnalysisResult",
    "SpatialProductUnavailable",
    "build_in_memory_manifest",
    "finalize_analysis_bundle",
    "preflight_spatial_outputs",
]
