"""One public user-input-to-table entry point: :func:`analyze_from_dea`.

Orchestration only (task W4.2). This module contains no metric formula and
no acquisition internals -- it calls hydroseason's public DEA-statistics and
WOfS-acquisition APIs, HydroFragments' own zoning/cache-footprint/hydroyear
adapters, and :func:`hydrofragments.api.analyze` exactly once, then writes
the final tables and an enriched manifest. Every scientific computation it
touches (zoning thresholds, APSEC, LPSEC, hydro-year detection, metric
registry resolution) already lives in, and stays owned by, the modules it
calls.

Phase timings recorded in the run manifest's ``timings_seconds``:

- ``dea_planning``: the native DEA Water Observation Statistics read, zone
  assignment, and coarse wet-pixel planning-footprint build.
- ``wofs_query_and_acquisition``: one call to
  ``hydroseason.acquire_wofs_cache`` -- hydroseason's public contract fuses
  its STAC search and the resumable annual Zarr write into a single call,
  so this is the finest phase boundary observable from this side of the
  repository boundary (see ``hydroseason.acquire_wofs_cache``'s own
  docstring: "Queries STAC exactly once for the whole interval ... writes
  one annual Zarr group per calendar year not already completed").
- ``metric_processing``: opening the verified cache footprints/water cube,
  deriving hydro-year and dual-composite inputs, and the single
  :func:`hydrofragments.api._run_core_analysis` call.
- ``output_write``: atomic bundle finalization (tables, spatial products,
  validated manifest).
- ``total``: wall-clock sum of the four phases above.

``hydrofragments.io.riverscape_sources`` is a deliberate, documented
exception to this module's DEA-access boundary (see that module's and
``io/dea.py``'s docstrings) -- it owns STAC/WFS access for riverscape
zoning's terrain, cover, and waterbody evidence directly, since hydroseason
has no equivalent loader for them.
"""
from __future__ import annotations

import importlib.metadata
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hydroseason
import hydroseason._io_dea_stats  # noqa: F401 -- accessed as module attrs below
from hydroseason._io_dea_stats import DEAStatsUnavailable, WoStatisticsUnavailable

_DEA_PRODUCT = "ga_ls_wo_fq_myear_3"
_DEA_STAC_URL = "https://explorer.sandbox.dea.ga.gov.au/stac"
_DEFAULT_CHANNEL_BUFFER_M = 60.0
_GIT_SHA_ENV = "HYDROFRAGMENTS_GIT_SHA"
_PACKAGE_METADATA_REVISION_KEYS = (
    "Source-Revision-Id",
    "Revision-Id",
    "Git-Commit",
)


def resolve_git_sha() -> str:
    """Resolve one git revision for an entire analysis run.

    Precedence: CI environment variable, installed package metadata, local
    Git ``HEAD``, then the literal ``unknown``.
    """

    env_value = os.environ.get(_GIT_SHA_ENV, "").strip()
    if env_value:
        return env_value

    try:
        metadata = importlib.metadata.metadata("hydrofragments")
        for key in _PACKAGE_METADATA_REVISION_KEYS:
            value = metadata.get(key, "").strip()
            if value:
                return value
    except importlib.metadata.PackageNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if completed.returncode == 0:
            head = completed.stdout.strip()
            if head:
                return head
    except (OSError, subprocess.SubprocessError):
        pass

    return "unknown"


from hydrofragments.api import _run_core_analysis, open_water_cube
from hydrofragments.config import HydroConfig
from hydrofragments.io.cache_footprints import open_verified_cache_footprints
from hydrofragments.io.dea import open_wo_statistics_for_zoning
from hydrofragments.metrics import ApsecRecord
from hydrofragments.models import AnalysisInputs, HydroResult
from hydrofragments.output.finalize import finalize_analysis_bundle
from hydrofragments.output.manifest import build_dea_provenance
from hydrofragments.spatial import (
    SpatialContext,
    create_channel_context,
    reach_monthly_wet_profile,
)
from hydrofragments.spatial.zones import zones_from_wo_statistics


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
            "output": {"output_dir": str(output_dir)},
        }
    )


def _load_geometry(source: Any):
    """Normalize ``str | Path | GeoDataFrame`` via hydroseason's public loader.

    ``hydroseason.load_aoi`` already validates non-empty, non-null,
    geometrically valid geometry for a vector path or GeoDataFrame -- reused
    here instead of duplicating that validation for ``aoi``/``drainage``.
    """
    return hydroseason.load_aoi(source)


def _resolve_dea_planning(
    aoi: Any, *, requested_years: list[int], resolution: float, crs: str
):
    """Phase 1: native DEA statistics -> zone mask + coarse planning footprint.

    Returns ``(stats, footprint)``. On ``WoStatisticsUnavailable`` (the DEA
    statistics read itself failed/timed out/returned nothing) or
    ``DEAStatsUnavailable`` (the read succeeded but the wet-pixel planning
    footprint could not be established -- e.g. requested years outside its
    covered range, an incompatible source lineage, or zero wet pixels),
    this is the documented fail-open signal: return ``(None, None)`` so the
    caller falls back to full-AOI acquisition rather than pruning on an
    unproven mask.
    """
    try:
        stats = open_wo_statistics_for_zoning(
            aoi, product=_DEA_PRODUCT, stac_url=_DEA_STAC_URL,
            resolution=resolution, crs=crs,
        )
    except WoStatisticsUnavailable:
        return None, None

    stats_dataset = _wo_statistics_as_dataset(stats)
    try:
        footprint = hydroseason._io_dea_stats.build_wet_planning_footprint(
            stats_dataset, requested_years=requested_years,
        )
    except DEAStatsUnavailable:
        return stats, None

    return stats, footprint


def _wo_statistics_as_dataset(stats) -> "Any":
    """Reassemble hydroseason's raw ``xr.Dataset`` shape from ``WoStatistics``.

    ``build_wet_planning_footprint`` expects the same ``xr.Dataset`` shape
    ``hydroseason.open_wo_statistics`` returns (``count_wet``/``count_clear``/
    ``frequency`` with ``.attrs["provenance"]``); ``open_wo_statistics_for_zoning``
    already unpacked exactly those fields into :class:`WoStatistics`, so this
    reassembles them rather than re-querying hydroseason a second time.
    """
    import xarray as xr

    dataset = xr.Dataset(
        {
            "count_wet": stats.count_wet,
            "count_clear": stats.count_clear,
            "frequency": stats.frequency,
        }
    )
    dataset.attrs["provenance"] = dict(stats.provenance)
    return dataset


def _dual_extent_inputs(
    dual_counts: "pd.DataFrame | None",
) -> tuple[pd.Series | None, list[ApsecRecord] | None, list[ApsecRecord] | None]:
    """Derive hydro-year extent + dual-composite APSEC records from counts.

    ``dual_counts`` is ``hydroseason.open_completed_dual_extent_counts``'s
    return value: ``None`` means "not available this run" (incomplete
    cache, or acquired without ``composite_bundle='hydrofragments_v1'``) --
    treated here as "dynamics/extent_contraction inputs unavailable", never
    raised. The percentage convention (``100 * n_water / aoi_pixel_count``)
    matches :func:`hydrofragments.metrics.extent.compute_apsec`'s own
    ``wetted_area / a_ref_m2 * 100`` formula, with the fixed reference area
    expressed here in pixel-count terms (both sides share the same
    ``cell_area_m2`` factor, so it cancels).

    ``dual_counts`` carries two distinct denominator columns:
    ``aoi_pixel_count`` (full catchment) and ``analysis_mask_pixel_count``
    (the conservative potential-water footprint, a subset of the AOI). Per
    the plan's Global Constraints, APSEC/LPI/reference-area denominators
    stay pinned to the full ``aoi_mask`` -- so this function must use
    ``aoi_pixel_count`` here, never ``analysis_mask_pixel_count``. The
    latter is reserved for the *monthly coverage* fraction computed
    elsewhere in this codebase (see ``AnalysisMaskCoverageResult`` in
    ``hydrofragments.metrics.extent``), which is deliberately denominated
    by the smaller, conservative footprint -- a different metric with a
    different denominator by design, not an interchangeable choice.
    """
    if dual_counts is None or dual_counts.empty:
        return None, None, None

    denominator = dual_counts["aoi_pixel_count"].astype(float)
    extent_pct = 100.0 * dual_counts["n_max_water"].astype(float) / denominator
    hydroyear_extent = pd.Series(
        extent_pct.to_numpy(), index=dual_counts.index, name="extent_pct"
    )

    max_water_records = [
        ApsecRecord(
            date=timestamp.to_pydatetime(),
            value=float(
                100.0 * row["n_max_water"] / row["aoi_pixel_count"]
            ),
            n_water_pixels=int(row["n_max_water"]),
            a_ref_m2=float(row["aoi_pixel_count"]),
            cell_area_m2=1.0,
        )
        for timestamp, row in dual_counts.iterrows()
    ]
    median_records = [
        ApsecRecord(
            date=timestamp.to_pydatetime(),
            value=float(
                100.0 * row["n_median_water"] / row["aoi_pixel_count"]
            ),
            n_water_pixels=int(row["n_median_water"]),
            a_ref_m2=float(row["aoi_pixel_count"]),
            cell_area_m2=1.0,
        )
        for timestamp, row in dual_counts.iterrows()
    ]
    return hydroyear_extent, max_water_records, median_records


def _channel_inputs(
    drainage_source: Any,
    *,
    aoi_gdf: Any,
    aoi_id: str,
    water: "Any",
    target_crs: str,
) -> tuple[SpatialContext, "np.ndarray", list[float]]:
    """Build a real channel :class:`SpatialContext` plus its monthly wet profile.

    ``water``'s per-month, per-reach wetness comes from
    :func:`hydrofragments.spatial.reach_monthly_wet_profile` (the same
    skeleton-seeded-buffer method already used for ``wet_any_month`` gating,
    kept per-month instead of collapsed to a single OR) -- no new metric
    kernel is introduced here, only the existing one's already-computed
    intermediate is kept instead of discarded.
    """
    drainage_gdf = _load_geometry(drainage_source)
    context = create_channel_context(
        aoi_id, aoi_gdf, drainage_gdf, drainage_id="workflow", target_crs=target_crs,
    )
    wet_profile = reach_monthly_wet_profile(
        context.drainage, water, buffer_m=_DEFAULT_CHANNEL_BUFFER_M,
    )
    segment_lengths_m = context.drainage.geometry.length.tolist()
    return context, wet_profile, segment_lengths_m


def analyze_from_dea(
    aoi: Any,
    start_date: str,
    end_date: str,
    *,
    aoi_id: str,
    drainage: Any | None = None,
    config: HydroConfig | None = None,
    cache_dir: str | Path = "output/wofs_cache",
) -> HydroResult:
    """Run catchment analysis end-to-end from a user AOI/date range to tables.

    Calls hydroseason's public DEA-statistics and WOfS-acquisition APIs,
    creates verified ``aoi_mask``/``analysis_mask``, opens the canonical
    cache, derives hydro-year/dual-composite inputs automatically, creates
    channel inputs when ``drainage`` is supplied, calls
    :func:`hydrofragments.api.analyze` exactly once, and writes final
    artifacts (metrics table, metric-coverage table, DEA-enriched manifest).

    ``config=None`` builds a minimal default :class:`HydroConfig` for a
    ``watermask_tsfill`` cube (see :func:`_default_config`); a caller wanting
    non-default zone thresholds, validity policy, or metric profiles should
    build and pass their own resolved ``HydroConfig`` instead.

    On DEA-statistics unavailability (``WoStatisticsUnavailable``) or an
    unprovable wet-pixel planning footprint (``DEAStatsUnavailable`` --
    requested years outside coverage, incompatible source lineage, or zero
    wet pixels), this falls open to full-AOI acquisition (no pruning) rather
    than failing the run. An invalid/tampered cache mask digest
    (``CacheFootprintVerificationError``) is never swallowed: a wrong
    denominator must never be silently accepted, so it propagates.
    """
    timings: dict[str, float] = {}

    resolved_config = config or _default_config(
        output_dir=Path(cache_dir).parent / "output"
    )
    aoi_gdf = _load_geometry(aoi)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    requested_years = list(range(start.year, end.year + 1))
    resolution = 30.0
    dea_crs = "EPSG:3577"

    # --- Phase 1: DEA planning ------------------------------------------------
    t0 = time.perf_counter()
    stats, footprint = _resolve_dea_planning(
        aoi_gdf, requested_years=requested_years, resolution=resolution, crs=dea_crs,
    )
    zone_result = None
    if stats is not None:
        zone_result = zones_from_wo_statistics(stats, config=resolved_config)
    timings["dea_planning"] = time.perf_counter() - t0

    # --- Phase 2/3: WOfS query + acquisition -----------------------------------
    t0 = time.perf_counter()
    handle = hydroseason.acquire_wofs_cache(
        _DEA_STAC_URL,
        "ga_ls_wo_3",
        aoi_gdf,
        start_date,
        end_date,
        cache_root=cache_dir,
        crs=dea_crs,
        resolution=resolution,
        wet_mask="dea_stats" if footprint is not None else "off",
        planning_footprint=footprint,
        composite_bundle="hydrofragments_v1",
    )
    timings["wofs_query_and_acquisition"] = time.perf_counter() - t0

    # --- Phase 4: local metric processing --------------------------------------
    t0 = time.perf_counter()
    verified_footprints = open_verified_cache_footprints(handle)
    mask_cube = hydroseason.open_completed_mask_cache(handle, start_date, end_date)

    aoi_mask_da = _as_spatial_mask(verified_footprints.aoi_mask, mask_cube)
    analysis_mask_da = _as_spatial_mask(verified_footprints.analysis_mask, mask_cube)

    cube = open_water_cube(
        mask_cube,
        input_kind="watermask_tsfill",
        aoi_mask=aoi_mask_da,
        analysis_mask=analysis_mask_da,
    )

    dual_counts = hydroseason.open_completed_dual_extent_counts(
        handle, start_date, end_date
    )
    hydroyear_extent, max_water_apsec, median_apsec = _dual_extent_inputs(dual_counts)

    channel_context: SpatialContext | None = None
    channel_wet_profiles = None
    channel_segment_lengths_m = None
    if drainage is not None:
        channel_context, channel_wet_profiles, channel_segment_lengths_m = (
            _channel_inputs(
                drainage,
                aoi_gdf=aoi_gdf,
                aoi_id=aoi_id,
                water=cube.water,
                target_crs=resolved_config.spatial.target_crs,
            )
        )

    inputs = AnalysisInputs(
        drainage=channel_context,
        hydroyear_extent=hydroyear_extent,
        max_water_apsec=max_water_apsec,
        median_apsec=median_apsec,
        channel_wet_profiles=channel_wet_profiles,
        channel_segment_lengths_m=channel_segment_lengths_m,
    )
    core = _run_core_analysis(
        cube,
        aoi_id,
        config=resolved_config,
        inputs=inputs,
        pixel_size_m=resolution,
        git_sha=resolve_git_sha(),
        zone_result=zone_result,
    )
    timings["metric_processing"] = time.perf_counter() - t0

    # --- Phase 5: output write --------------------------------------------------
    t0 = time.perf_counter()
    dea_provenance = None
    if stats is not None and zone_result is not None:
        dea_provenance = build_dea_provenance(
            resolved_config,
            product=stats.product,
            version=stats.version,
            item_ids=list(stats.provenance.get("item_ids", ())),
            crs=stats.crs,
            resolution=resolution,
            time_span=stats.time_span,
            zone_mask=zone_result.mask,
            planning_footprint=(
                {
                    "digest": footprint.digest,
                    "factor": footprint.factor,
                    "safety_cells": footprint.safety_cells,
                    "covered_years": list(footprint.covered_years),
                    "source_collection": footprint.source_collection,
                    "source_version": footprint.source_version,
                    "source_lineage": footprint.source_lineage,
                }
                if footprint is not None
                else None
            ),
        )
    result = finalize_analysis_bundle(
        resolved_config,
        core,
        cube=cube,
        inputs=inputs,
        pixel_size_m=resolution,
        zone_result=zone_result,
        dea_provenance=dea_provenance,
        timings_seconds=timings,
    )
    timings.update(result.manifest.get("timings_seconds", {}))

    return result


def _as_spatial_mask(mask: "np.ndarray", reference: "Any"):
    """Wrap a verified 2-D boolean mask as an ``xr.DataArray`` aligned to ``reference``."""
    import xarray as xr

    spatial_dims = tuple(dim for dim in reference.dims if dim != "time")
    coords = {
        name: coord
        for name, coord in reference.coords.items()
        if set(coord.dims) <= set(spatial_dims)
    }
    return xr.DataArray(mask, dims=spatial_dims, coords=coords)


__all__ = ["analyze_from_dea", "resolve_git_sha"]
