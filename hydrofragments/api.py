"""Public HydroFragments v1.2 API."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments._version import __version__
from hydrofragments.section_analysis import (
    DynamicsSupportCollector,
    SectionSpatialCollector,
    analyze_section_rows,
)
from hydrofragments.config import HydroConfig
from hydrofragments.compute import resolve_execution_plan
from hydrofragments.guards import ComparisonGuardError, guard_comparison
from hydrofragments.io.adapters import parse_watermask_tsfill
from hydrofragments.io.alignment import validate_alignment
from hydrofragments.metrics import (
    ApsecRecord,
    MetricPlan,
    RUNTIME_WIRED_METRIC_IDS,
    compute_extent_contraction,
    compute_hydroperiod,
    compute_inter_pool_gaps,
    compute_lpsec,
    compute_recurrence,
    registry_wide_plan,
    resolve_metrics,
)
from hydrofragments.models import (
    AnalysisInputs,
    HydroResult,
    MetricRecord,
    ValidationReport,
    WaterCube,
)
if TYPE_CHECKING:
    from hydrofragments.output.core import CoreAnalysisResult
from hydrofragments.output.tables import records_to_frame, resolve_git_sha
from hydrofragments.schema import (
    EdgeFlag,
    MetricDependency,
    MetricFamily,
    SCHEMA_VERSION,
    Statistic,
    ValueType,
    WarningFlag,
)
from hydrofragments.spatial import SpatialContext
from hydrofragments.metrics.dynamics import (
    DynamicsSupport,
    EndDryState,
    compute_reconnection_timing,
    evaluate_refuge_spatial_stability,
    _lpsec_complete_for_window,
    _month_key,
    _search_window_months,
    _series_lookup,
    _slice_series_for_window,
)
from hydrofragments.temporal import detect_cadence, detect_hy_anchors
from hydrofragments.temporal.hydroyear import HyAnchorResult
from hydrofragments.spatial.zones import ZoneResult


def _describe_chunks(array: xr.DataArray) -> str:
    """Serializable description of an array's actual Dask chunk sizes.

    Returns ``"none"`` for a non-Dask-backed (eager numpy) array, otherwise
    a compact ``"dim=size,dim=size,..."`` string using each dimension's
    first block size -- enough to verify what chunking was actually applied
    without embedding full nested chunk tuples in provenance/manifest text.
    """
    chunks = array.chunks
    if chunks is None:
        return "none"
    return ",".join(
        f"{dim}={sizes[0]}" for dim, sizes in zip(array.dims, chunks)
    )


def _coerce_dataarray(source: xr.DataArray | xr.Dataset) -> xr.DataArray:
    if isinstance(source, xr.Dataset):
        if "water" in source:
            return source["water"]
        if len(source.data_vars) == 1:
            return next(iter(source.data_vars.values()))
        raise ValueError("Dataset must expose a single water variable or 'water'")
    return source


_WATER_VALIDITY_ERROR = "water=True requires valid_obs=True for every pixel/month"


def _can_check_water_validity_eager(
    water: xr.DataArray, valid_obs: xr.DataArray
) -> bool:
    return water.chunks is None and valid_obs.chunks is None


def _has_water_without_valid_obs(water: xr.DataArray, valid_obs: xr.DataArray) -> bool:
    if not _can_check_water_validity_eager(water, valid_obs):
        return False
    invalid_water = water.astype(bool) & ~valid_obs.astype(bool)
    return bool(np.any(np.asarray(invalid_water.values, dtype=bool)))


def _ensure_water_implies_valid_obs(
    water: xr.DataArray, valid_obs: xr.DataArray
) -> None:
    if _has_water_without_valid_obs(water, valid_obs):
        raise ValueError(_WATER_VALIDITY_ERROR)


def open_water_cube(
    source: xr.DataArray | xr.Dataset | str | Path,
    *,
    valid_obs: xr.DataArray | None = None,
    variable_map: Mapping[str, str] | None = None,
    chunks: Mapping[str, int] | None = None,
    input_kind: str = "generic_binary",
    aoi_mask: xr.DataArray | None = None,
    analysis_mask: xr.DataArray | None = None,
) -> WaterCube:
    """Open a canonical aligned water/valid cube from supported sources.

    ``aoi_mask``/``analysis_mask`` are optional aligned 2-D boolean masks
    passed straight through to :class:`hydrofragments.models.WaterCube`
    (which validates alignment and defaults either to all-true when
    omitted -- see its docstring). Every caller that does not pass them
    keeps today's unpruned full-grid behaviour unchanged.
    """
    del variable_map  # reserved for later adapter expansion

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".zarr" or path.name.endswith(".zarr"):
            dataset = xr.open_zarr(path, chunks=chunks if chunks is not None else "auto")
            raw = dataset["water_mask"] if "water_mask" in dataset else dataset["water"]
            water, valid = parse_watermask_tsfill(raw)
            cadence = detect_cadence(water)
            crs = water.rio.crs.to_string() if hasattr(water, "rio") and water.rio.crs else None
            return WaterCube(
                water=water,
                valid_obs=valid,
                source=str(path),
                cadence=cadence,
                crs=crs,
                provenance=(
                    ("adapter", "watermask_tsfill"),
                    ("chunks", _describe_chunks(water)),
                ),
                aoi_mask=aoi_mask,
                analysis_mask=analysis_mask,
            )
        raise ValueError(f"unsupported source path: {path}")

    array = _coerce_dataarray(source)
    if chunks is not None:
        array = array.chunk(chunks)
    if input_kind == "watermask_tsfill":
        water, valid = parse_watermask_tsfill(array)
    else:
        water = (array == 1).astype(bool)
        if valid_obs is None:
            valid = xr.ones_like(water, dtype=bool)
        else:
            valid = valid_obs.astype(bool)
            if chunks is not None:
                valid = valid.chunk(chunks)
            validate_alignment(water, valid)
            _ensure_water_implies_valid_obs(water, valid)
    cadence = detect_cadence(water)
    crs = water.rio.crs.to_string() if hasattr(water, "rio") and water.rio.crs else None
    return WaterCube(
        water=water,
        valid_obs=valid,
        source=input_kind,
        cadence=cadence,
        crs=crs,
        provenance=(
            ("input_kind", input_kind),
            ("chunks", _describe_chunks(water)),
        ),
        aoi_mask=aoi_mask,
        analysis_mask=analysis_mask,
    )


def _available_dependencies(
    *,
    config: HydroConfig,
    drainage: Any | None = None,
    hydroyear_available: bool = False,
    dual_composites_available: bool = False,
    wet_any_month: Mapping[str, bool] | None = None,
) -> set[MetricDependency]:
    """Assemble the set of metric dependencies actually present in this run.

    Single source of truth for "what inputs did this run actually supply,"
    shared by both :func:`validate_inputs` (which reports skip reasons
    without computing anything) and :func:`analyze` (which uses the exact
    same availability to decide what to execute). Before this helper existed
    the two call sites independently re-derived this set, and it was
    possible for them to silently drift apart.
    """
    available = {MetricDependency.VALIDITY}
    if config.patches.min_patch_pixels > 0:
        available.add(MetricDependency.PATCHES)
    has_real_channel = (
        isinstance(drainage, SpatialContext) and drainage.has_real_channel
    )
    if has_real_channel:
        available.add(MetricDependency.CHANNEL)
    if config.patches.width_resolution_floor_pixels is not None:
        available.add(MetricDependency.WIDTH_FLOOR)
    if hydroyear_available:
        available.add(MetricDependency.HY_ANCHOR)
    if dual_composites_available:
        available.add(MetricDependency.DUAL_COMPOSITE)
    if (
        has_real_channel
        and wet_any_month is not None
        and any(wet_any_month.values())
    ):
        available.add(MetricDependency.FIXED_NODES)
        available.add(MetricDependency.GRAPH)
    return available


def validate_inputs(
    cube: WaterCube,
    aoi_id: str,
    *,
    config: HydroConfig,
    drainage: Any | None = None,
    hydroyear_available: bool = False,
    dual_composites_available: bool = False,
    wet_any_month: Mapping[str, bool] | None = None,
) -> ValidationReport:
    """Validate contracts without running metric kernels."""
    errors: list[str] = []
    warnings: list[str] = []
    if cube.water.sizes != cube.valid_obs.sizes:
        errors.append("water and valid_obs must share dimensions")
    if cube.water.dims != cube.valid_obs.dims:
        errors.append("water and valid_obs must share dimension order")
    try:
        validate_alignment(cube.water, cube.valid_obs)
    except ValueError as error:
        errors.append(str(error))
    else:
        if _has_water_without_valid_obs(cube.water, cube.valid_obs):
            errors.append(_WATER_VALIDITY_ERROR)

    available = _available_dependencies(
        config=config,
        drainage=drainage,
        hydroyear_available=hydroyear_available,
        dual_composites_available=dual_composites_available,
        wet_any_month=wet_any_month,
    )
    plan = resolve_metrics(config.metric_profiles, available_dependencies=available, metric_overrides=config.metric_overrides)
    skipped = tuple((item.metric_id, item.reason) for item in plan.skipped)
    if plan.skipped:
        warnings.append("some requested metrics are unavailable with current inputs")

    return ValidationReport(
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        resolved_profiles=config.metric_profiles,
        skipped_metrics=skipped,
    )


def _metric_record(
    *,
    run_id: str,
    config: HydroConfig,
    git_sha: str,
    catchment_id: str,
    aoi_id: str,
    metric: str,
    metric_family: MetricFamily,
    value: float | None,
    unit: str,
    value_type: ValueType,
    source: str,
    timestamp: datetime | None = None,
    resolution_m: float,
    crs: str,
    n_pools: int | None = None,
    zone: str = "AOI",
    statistic: Statistic | None = None,
    metric_dependency: MetricDependency = MetricDependency.NONE,
    warning_flags: tuple[WarningFlag, ...] = (WarningFlag.LENGTH_CRS_CAVEAT,),
    n_valid_pixels: int | None = None,
    n_water_pixels: int | None = None,
    valid_fraction_month: float | None = None,
    min_valid_fraction_month: float | None = None,
    edge_flag: EdgeFlag | None = None,
    low_coverage_flag: bool | None = None,
    is_reportable: bool | None = None,
) -> MetricRecord:
    reportable = value is not None and np.isfinite(value)
    if is_reportable is not None:
        reportable = is_reportable
    return MetricRecord(
        run_id=run_id,
        config_hash=config.config_hash,
        package_version=__version__,
        git_sha=git_sha,
        catchment_id=catchment_id,
        aoi_id=aoi_id,
        zone=zone,
        date=timestamp,
        metric=metric,
        metric_family=metric_family,
        statistic=statistic,
        value=value,
        unit=unit,
        value_type=value_type,
        n_pools=n_pools,
        n_valid_pixels=n_valid_pixels,
        n_water_pixels=n_water_pixels,
        valid_fraction_month=valid_fraction_month,
        min_valid_fraction_month=min_valid_fraction_month,
        edge_flag=edge_flag,
        low_coverage_flag=low_coverage_flag,
        warning_flags=warning_flags,
        is_reportable=reportable,
        source=source,
        resolution_m=resolution_m,
        crs=crs,
        area_unit="m2",
        length_unit="m",
        monthly_composite=config.temporal.monthly_composite,
        min_patch_pixels=config.patches.min_patch_pixels,
        min_patch_area_m2=float(resolution_m) ** 2 * config.patches.min_patch_pixels,
        connectivity_rule=config.patches.connectivity_rule,
        metric_dependency=metric_dependency,
    )


def _channel_profile_records(
    *,
    cube: WaterCube,
    context: SpatialContext,
    wet_profiles: Sequence[Sequence[bool]],
    segment_lengths_m: Sequence[float],
    run_id: str,
    config: HydroConfig,
    catchment_id: str,
    aoi_id: str,
    resolution_m: float,
    crs: str,
    source: str,
    git_sha: str,
) -> list[MetricRecord]:
    wet = np.asarray(wet_profiles, dtype=bool)
    lengths = np.asarray(segment_lengths_m, dtype=float)
    if wet.ndim != 2:
        raise ValueError("channel_wet_profiles must have shape (time, segment)")
    if lengths.ndim != 1 or wet.shape[1] != lengths.size:
        raise ValueError("channel profile and segment lengths must align")
    if wet.shape[0] != cube.water.sizes.get("time", 0):
        raise ValueError("channel profile time axis must align with water cube")
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("channel_segment_lengths_m must be positive and finite")

    records: list[MetricRecord] = []
    times = pd.to_datetime(cube.water["time"].values)
    for timestamp, states in zip(times, wet):
        wetted_length_m = float(lengths[states].sum())
        lpsec = compute_lpsec(wetted_length_m, context=context)
        records.append(
            _metric_record(
                run_id=run_id,
                config=config,
                git_sha=git_sha,
                catchment_id=catchment_id,
                aoi_id=aoi_id,
                metric="lpsec",
                metric_family=MetricFamily.EXTENT,
                value=lpsec.value,
                unit="percent",
                value_type=ValueType.MONTHLY,
                source=source,
                timestamp=timestamp.to_pydatetime(),
                resolution_m=resolution_m,
                crs=crs,
                zone="1",
                metric_dependency=MetricDependency.CHANNEL,
            )
        )
        gaps = compute_inter_pool_gaps(states, segment_lengths_m=lengths)
        for statistic, value_m in (
            (Statistic.MEAN, gaps.mean_m),
            (Statistic.MEDIAN, gaps.median_m),
            (Statistic.MAX, gaps.max_m),
            (Statistic.CV, gaps.cv),
        ):
            if not np.isfinite(value_m):
                continue
            value = float(value_m if statistic is Statistic.CV else value_m / 1000.0)
            records.append(
                _metric_record(
                    run_id=run_id,
                    config=config,
                    git_sha=git_sha,
                    catchment_id=catchment_id,
                    aoi_id=aoi_id,
                    metric="inter_pool_gap",
                    metric_family=MetricFamily.CLUSTERING,
                    statistic=statistic,
                    value=value,
                    unit="dimensionless" if statistic is Statistic.CV else "km",
                    value_type=ValueType.MONTHLY,
                    source=source,
                    timestamp=timestamp.to_pydatetime(),
                    resolution_m=resolution_m,
                    crs=crs,
                    zone="1",
                    metric_dependency=MetricDependency.CHANNEL,
                )
            )
    return records


# Metric ids that can ever originate from analyze_section_rows() /
# _records_from_compat_rows() below (the legacy wide-row bridge). Kept in
# sync with the "mapping" table inside _records_from_compat_rows -- used to
# skip the whole compat-row compute path when a narrow profile selects none
# of these (B1).
_SECTION_ROW_METRIC_IDS = frozenset(
    {
        "apsec",
        "number_of_pools",
        "lpi",
        "awre",
        "awmsi",
        "pool_width",
        "occurrence",
        "refuge_area",
    }
)


def _records_from_section_rows(
    rows: list[dict[str, object]],
    *,
    run_id: str,
    config: HydroConfig,
    catchment_id: str,
    aoi_id: str,
    resolution_m: float,
    crs: str,
    source: str,
    git_sha: str,
) -> list[MetricRecord]:
    mapping = {
        "APSEC": (MetricFamily.EXTENT, "apsec", "percent", ValueType.MONTHLY, None),
        "n_patches": (
            MetricFamily.FRAGMENTATION,
            "number_of_pools",
            "count",
            ValueType.MONTHLY,
            None,
        ),
        "LPI": (
            MetricFamily.FRAGMENTATION,
            "lpi",
            "percent",
            ValueType.MONTHLY,
            None,
        ),
        "AWRe": (
            MetricFamily.MORPHOLOGY,
            "awre",
            "dimensionless",
            ValueType.MONTHLY,
            None,
        ),
        "AWMSI": (
            MetricFamily.MORPHOLOGY,
            "awmsi",
            "dimensionless",
            ValueType.MONTHLY,
            None,
        ),
        "pool_width_mean": (
            MetricFamily.MORPHOLOGY,
            "pool_width",
            "m",
            ValueType.MONTHLY,
            Statistic.MEAN,
        ),
        "pool_width_median": (
            MetricFamily.MORPHOLOGY,
            "pool_width",
            "m",
            ValueType.MONTHLY,
            Statistic.MEDIAN,
        ),
        "pool_width_max": (
            MetricFamily.MORPHOLOGY,
            "pool_width",
            "m",
            ValueType.MONTHLY,
            Statistic.MAX,
        ),
        "pool_width_cv": (
            MetricFamily.MORPHOLOGY,
            "pool_width",
            "dimensionless",
            ValueType.MONTHLY,
            Statistic.CV,
        ),
        "pp_mean_%": (
            MetricFamily.PERSISTENCE,
            "occurrence",
            "percent",
            ValueType.RASTER_SUMMARY,
            None,
        ),
        "ra_area_km2": (
            MetricFamily.PERSISTENCE,
            "refuge_area",
            "km2",
            ValueType.RASTER_SUMMARY,
            None,
        ),
    }
    records: list[MetricRecord] = []
    for row in rows:
        timestamp = pd.Timestamp(row["date"]).to_pydatetime()
        for column, (family, metric_id, unit, value_type, statistic) in mapping.items():
            value = row.get(column)
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                numeric_value = None
            else:
                numeric_value = float(value)
            if metric_id == "pool_width" and (
                numeric_value is None or not np.isfinite(numeric_value)
            ):
                continue
            monthly_metric = value_type is ValueType.MONTHLY
            low_coverage = monthly_metric and bool(row.get("low_coverage_flag", False))
            records.append(
                _metric_record(
                    run_id=run_id,
                    config=config,
                    git_sha=git_sha,
                    catchment_id=catchment_id,
                    aoi_id=aoi_id,
                    metric=metric_id,
                    metric_family=family,
                    value=numeric_value,
                    unit=unit,
                    value_type=value_type,
                    statistic=statistic,
                    timestamp=timestamp,
                    resolution_m=resolution_m,
                    crs=crs,
                    source=source,
                    n_pools=(
                        int(row["n_patches"])
                        if metric_id == "number_of_pools" and row.get("n_patches") is not None
                        else None
                    ),
                    n_valid_pixels=(
                        int(row["n_valid_pixels"])
                        if monthly_metric and row.get("n_valid_pixels") is not None
                        else None
                    ),
                    n_water_pixels=(
                        int(row["APSEC_n_water_pixels"])
                        if metric_id == "apsec"
                        and row.get("APSEC_n_water_pixels") is not None
                        else None
                    ),
                    valid_fraction_month=(
                        float(row["valid_fraction_month"])
                        if monthly_metric
                        and row.get("valid_fraction_month") is not None
                        else None
                    ),
                    min_valid_fraction_month=(
                        float(row["min_valid_fraction_month"])
                        if monthly_metric
                        and row.get("min_valid_fraction_month") is not None
                        else None
                    ),
                    edge_flag=(
                        EdgeFlag.LOW_VALID_OBS if low_coverage else None
                    ),
                    low_coverage_flag=low_coverage if monthly_metric else None,
                    warning_flags=(
                        row["pool_width_warning_flags"]
                        if metric_id == "pool_width"
                        and row.get("pool_width_warning_flags") is not None
                        else (WarningFlag.LENGTH_CRS_CAVEAT,)
                    ),
                    metric_dependency=(
                        MetricDependency.WIDTH_FLOOR
                        if metric_id == "pool_width"
                        else MetricDependency.NONE
                    ),
                    is_reportable=False if low_coverage else None,
                )
            )
    return records


def _temporal_profile_records(
    monthly: xr.Dataset,
    *,
    run_id: str,
    config: HydroConfig,
    catchment_id: str,
    aoi_id: str,
    source: str,
    resolution_m: float,
    crs: str,
    git_sha: str,
) -> list[MetricRecord]:
    """Emit AOI summaries for pixel-temporal kernels through canonical schema."""
    records: list[MetricRecord] = []
    recurrence = compute_recurrence(monthly, config=config)
    hydroperiod = compute_hydroperiod(monthly, config=config).hydroperiod
    years = [int(year) for year in hydroperiod.coords["year"].values]

    # Batch the AOI-mean recurrence scalar and every per-year AOI-mean
    # hydroperiod scalar into a single Dataset so they share one Dask graph
    # execution instead of one independent `.item()` materialization each
    # (m8: materialization count must not scale with the number of years).
    summary_vars: dict[str, xr.DataArray] = {
        "recurrence": recurrence.recurrence.mean(skipna=True),
    }
    for year in years:
        summary_vars[f"hydroperiod_{year}"] = (
            hydroperiod.sel(year=year).mean(skipna=True).drop_vars("year")
        )
    summary_ds = xr.Dataset(summary_vars).compute()

    recurrence_value = summary_ds["recurrence"].item()
    if recurrence_value is not None and np.isfinite(recurrence_value):
        records.append(
            _metric_record(
                run_id=run_id,
                config=config,
                git_sha=git_sha,
                catchment_id=catchment_id,
                aoi_id=aoi_id,
                metric="recurrence",
                metric_family=MetricFamily.PERSISTENCE,
                value=float(recurrence_value),
                unit="percent",
                value_type=ValueType.RASTER_SUMMARY,
                source=source,
                resolution_m=resolution_m,
                crs=crs,
            )
        )

    for year in years:
        value = summary_ds[f"hydroperiod_{year}"].item()
        if value is None or not np.isfinite(value):
            continue
        records.append(
            _metric_record(
                run_id=run_id,
                config=config,
                git_sha=git_sha,
                catchment_id=catchment_id,
                aoi_id=aoi_id,
                metric="hydroperiod",
                metric_family=MetricFamily.PERSISTENCE,
                value=float(value),
                unit="fraction",
                value_type=ValueType.RASTER_SUMMARY,
                source=source,
                timestamp=datetime(int(year), 1, 1),
                resolution_m=resolution_m,
                crs=crs,
            )
        )
    return records


_DYNAMICS_PROFILE_METRIC_IDS = frozenset(
    {
        "extent_contraction",
        "reconnection_timing",
        "refuge_spatial_stability",
    }
)
_DYNAMICS_CUBE_METRIC_IDS = frozenset(
    {"reconnection_timing", "refuge_spatial_stability"}
)


def _build_lpsec_by_month(
    *,
    cube: WaterCube,
    context: SpatialContext,
    wet_profiles: Sequence[Sequence[bool]],
    segment_lengths_m: Sequence[float],
) -> tuple[tuple[datetime, float], ...]:
    wet = np.asarray(wet_profiles, dtype=bool)
    lengths = np.asarray(segment_lengths_m, dtype=float)
    times = pd.to_datetime(cube.water["time"].values)
    series: list[tuple[datetime, float]] = []
    for timestamp, states in zip(times, wet):
        wetted_length_m = float(lengths[states].sum())
        lpsec = compute_lpsec(wetted_length_m, context=context)
        series.append((timestamp.to_pydatetime(), float(lpsec.value)))
    return tuple(series)


def _dynamics_profile_records(
    *,
    selected_metric_ids: frozenset[str],
    hy_anchors: HyAnchorResult,
    monthly_support: DynamicsSupport,
    refuge_states: Sequence[EndDryState],
    config: HydroConfig,
    run_context: Mapping[str, object],
    max_water: Sequence[ApsecRecord] | None = None,
    median: Sequence[ApsecRecord] | None = None,
    analysis_mask: np.ndarray | None = None,
) -> list[MetricRecord]:
    run_id = str(run_context["run_id"])
    catchment_id = str(run_context["catchment_id"])
    aoi_id = str(run_context["aoi_id"])
    source = str(run_context["source"])
    resolution_m = float(run_context["resolution_m"])
    crs = str(run_context["crs"])
    git_sha = str(run_context["git_sha"])

    records: list[MetricRecord] = []
    anchors = hy_anchors.anchors.sort_values("hy", kind="stable")
    anchor_rows = anchors.to_dict(orient="records")
    end_dry_by_hy = {
        int(state.hy): state
        for state in refuge_states
    }
    lpi_lookup = _series_lookup(monthly_support.lpi_by_month)
    lpsec_lookup = (
        _series_lookup(monthly_support.lpsec_by_month)
        if monthly_support.lpsec_by_month is not None
        else {}
    )
    cube_month_keys = sorted(lpi_lookup)

    for index, anchor in enumerate(anchor_rows):
        hy = int(anchor["hy"])
        end_dry_month = anchor.get("end_dry_month")
        next_end_dry = (
            anchor_rows[index + 1].get("end_dry_month")
            if index + 1 < len(anchor_rows)
            else None
        )
        confidence = str(anchor.get("confidence", "unassigned"))

        if "extent_contraction" in selected_metric_ids:
            if max_water is not None and median is not None:
                result = compute_extent_contraction(
                    max_water=max_water,
                    median=median,
                    anchor=anchor,
                    config=config,
                )
                if result is not None:
                    for composite, value, low_df in (
                        ("max_water", result.slope_pct_per_month, result.low_df),
                        ("median", result.median_slope_pct_per_month, result.median_low_df),
                    ):
                        records.append(
                            MetricRecord(
                                run_id=run_id,
                                config_hash=config.config_hash,
                                package_version=__version__,
                                git_sha=git_sha,
                                catchment_id=catchment_id,
                                aoi_id=aoi_id,
                                zone="AOI",
                                hy=result.hy,
                                metric="extent_contraction",
                                metric_family=MetricFamily.DYNAMICS,
                                value=None
                                if low_df or not np.isfinite(value)
                                else float(value),
                                unit="percent_per_month",
                                value_type=ValueType.HY_SUMMARY,
                                hy_confidence=result.hy_confidence,
                                composite_sensitive=result.composite_sensitive,
                                monthly_composite=composite,
                                metric_dependency=MetricDependency.HY_ANCHOR,
                                is_reportable=not low_df and np.isfinite(value),
                                warning_flags=(
                                    (WarningFlag.COMPOSITE_SENSITIVE,)
                                    if result.composite_sensitive
                                    else ()
                                ),
                                source=source,
                                resolution_m=resolution_m,
                                crs=crs,
                                area_unit="m2",
                                length_unit="m",
                                min_patch_pixels=config.patches.min_patch_pixels,
                                min_patch_area_m2=resolution_m**2
                                * config.patches.min_patch_pixels,
                                connectivity_rule=config.patches.connectivity_rule,
                            )
                        )

        if "reconnection_timing" in selected_metric_ids:
            edge_flag = None
            value = None
            is_reportable = False
            warning_flags: tuple[WarningFlag, ...] = ()
            reconnection_metric_used = None
            proxy_reconnection_flag = None
            connected_threshold = None

            if end_dry_month is None or pd.isna(end_dry_month):
                edge_flag = EdgeFlag.MISSING_HY_ANCHOR
            else:
                end_dry_dt = pd.Timestamp(end_dry_month).to_pydatetime()
                window_months = _search_window_months(
                    end_dry_month=end_dry_dt,
                    next_end_dry_month=next_end_dry,
                    cube_month_keys=cube_month_keys,
                )
                lpi_window = _slice_series_for_window(
                    lpi_lookup, window_months=window_months
                )
                lpsec_window = _slice_series_for_window(
                    lpsec_lookup, window_months=window_months
                )
                use_lpsec = _lpsec_complete_for_window(
                    lpsec_lookup=lpsec_lookup,
                    window_months=window_months,
                )
                if use_lpsec:
                    result = compute_reconnection_timing(
                        lpi_series=lpi_window,
                        end_dry_month=end_dry_dt,
                        lpi_threshold=config.dynamics.reconnection_lpi_threshold_pct,
                        lpsec_series=lpsec_window,
                        lpsec_threshold=config.dynamics.reconnection_lpsec_threshold_pct,
                    )
                    connected_threshold = (
                        config.dynamics.reconnection_lpsec_threshold_pct
                    )
                else:
                    result = compute_reconnection_timing(
                        lpi_series=lpi_window,
                        end_dry_month=end_dry_dt,
                        lpi_threshold=config.dynamics.reconnection_lpi_threshold_pct,
                    )
                    connected_threshold = config.dynamics.reconnection_lpi_threshold_pct

                reconnection_metric_used = result.reconnection_metric_used
                proxy_reconnection_flag = result.proxy_reconnection_flag
                if result.proxy_reconnection_flag:
                    warning_flags = (WarningFlag.PROXY_RECONNECTION,)
                if result.t_reconnect_months is None:
                    edge_flag = EdgeFlag.NO_THRESHOLD_CROSSING
                else:
                    value = float(result.t_reconnect_months)
                    is_reportable = True

            records.append(
                MetricRecord(
                    run_id=run_id,
                    config_hash=config.config_hash,
                    package_version=__version__,
                    git_sha=git_sha,
                    catchment_id=catchment_id,
                    aoi_id=aoi_id,
                    zone="AOI",
                    date=pd.Timestamp(end_dry_month).to_pydatetime()
                    if end_dry_month is not None and not pd.isna(end_dry_month)
                    else None,
                    hy=hy,
                    hy_anchor="end_dry",
                    metric="reconnection_timing",
                    metric_family=MetricFamily.DYNAMICS,
                    value=value,
                    unit="month",
                    value_type=ValueType.HY_ANCHOR,
                    hy_confidence=confidence,
                    edge_flag=edge_flag,
                    warning_flags=warning_flags,
                    is_reportable=is_reportable,
                    connected_wet_metric=reconnection_metric_used,
                    connected_wet_threshold=connected_threshold,
                    reconnection_metric_used=reconnection_metric_used,
                    proxy_reconnection_flag=proxy_reconnection_flag,
                    metric_dependency=MetricDependency.HY_ANCHOR,
                    source=source,
                    resolution_m=resolution_m,
                    crs=crs,
                    area_unit="m2",
                    length_unit="m",
                    min_patch_pixels=config.patches.min_patch_pixels,
                    min_patch_area_m2=resolution_m**2 * config.patches.min_patch_pixels,
                    connectivity_rule=config.patches.connectivity_rule,
                )
            )

        if "refuge_spatial_stability" in selected_metric_ids:
            current_state = end_dry_by_hy.get(hy)
            previous_state = end_dry_by_hy.get(hy - 1)
            if current_state is None:
                evaluation = evaluate_refuge_spatial_stability(
                    current=EndDryState(
                        hy=hy,
                        date=pd.Timestamp(end_dry_month).to_pydatetime()
                        if end_dry_month is not None and not pd.isna(end_dry_month)
                        else datetime(1, 1, 1),
                        water=np.zeros((1, 1), dtype=bool),
                        valid_obs=np.zeros((1, 1), dtype=bool),
                        hy_confidence=confidence,
                        anchor_missing=True,
                    ),
                    previous=previous_state,
                    analysis_mask=analysis_mask,
                    min_valid_fraction=config.validity.min_valid_fraction_month,
                )
            else:
                evaluation = evaluate_refuge_spatial_stability(
                    current=current_state,
                    previous=previous_state,
                    analysis_mask=analysis_mask,
                    min_valid_fraction=config.validity.min_valid_fraction_month,
                )

            jaccard = evaluation.jaccard
            is_reportable = (
                evaluation.edge_flag is None
                and jaccard is not None
                and np.isfinite(jaccard)
            )
            records.append(
                MetricRecord(
                    run_id=run_id,
                    config_hash=config.config_hash,
                    package_version=__version__,
                    git_sha=git_sha,
                    catchment_id=catchment_id,
                    aoi_id=aoi_id,
                    zone="AOI",
                    date=current_state.date if current_state is not None else None,
                    hy=hy,
                    hy_anchor="end_dry",
                    metric="refuge_spatial_stability",
                    metric_family=MetricFamily.DYNAMICS,
                    value=float(jaccard)
                    if is_reportable and jaccard is not None
                    else None,
                    unit="dimensionless",
                    value_type=ValueType.HY_ANCHOR,
                    hy_confidence=confidence,
                    edge_flag=evaluation.edge_flag,
                    is_reportable=is_reportable,
                    valid_fraction_month=evaluation.common_valid_fraction,
                    min_valid_fraction_month=config.validity.min_valid_fraction_month,
                    n_valid_pixels=evaluation.n_common_valid_pixels,
                    n_water_pixels=evaluation.n_union_pixels,
                    metric_dependency=MetricDependency.HY_ANCHOR,
                    source=source,
                    resolution_m=resolution_m,
                    crs=crs,
                    area_unit="m2",
                    length_unit="m",
                    min_patch_pixels=config.patches.min_patch_pixels,
                    min_patch_area_m2=resolution_m**2 * config.patches.min_patch_pixels,
                    connectivity_rule=config.patches.connectivity_rule,
                )
            )

    return records


_COVERAGE_COLUMNS = (
    "metric",
    "runtime_wired",
    "status",
    "rows",
    "reportable_rows",
    "reason",
)


def _quality_reason(rows: pd.DataFrame) -> str:
    """Best-effort human reason a computed metric has zero reportable rows.

    Only called once a metric's kernel has already run (i.e. it is
    ``computed``): the metric was never dependency-skipped, but every row it
    produced failed the ``is_reportable`` check for a data-quality reason
    (e.g. too few valid observations for the month, or a zero-patch month),
    or the kernel ran and legitimately produced no rows at all (e.g. zero
    resolved hydrological-year anchors). Prefers the recorded ``edge_flag``
    values as the most specific signal; falls back to a generic message when
    none were set or none were produced.
    """
    if rows.empty:
        return "data quality: kernel ran, no rows produced"
    flags = sorted(
        {
            str(flag)
            for flag in rows["edge_flag"].dropna().unique()
            if flag not in (None, "")
        }
    )
    if flags:
        return "data quality: " + ", ".join(flags)
    return "data quality: zero reportable rows"


def _build_metric_coverage(
    frame: pd.DataFrame, coverage_plan: MetricPlan, *, selected_ids: set[str]
) -> pd.DataFrame:
    """Build the one-row-per-registry-metric coverage table (Step 5).

    ``coverage_plan`` (from :func:`hydrofragments.metrics.registry_wide_plan`)
    reports what ``all_available`` would resolve to for this run's inputs --
    every registry metric, split into ``selected`` (runtime-wired and
    dependency-satisfied) and ``skipped`` (either a runtime-wired metric
    missing a dependency, or a registry metric that is never runtime-wired at
    all -- see ``NOT_RUNTIME_WIRED_REASONS``). That is independent of which
    profile the caller actually asked for, so a runtime-wired,
    dependency-satisfied metric that an explicit narrower profile (e.g.
    ``contracts_core``) never selected must not be reported ``computed``: its
    kernel genuinely did not run this time. ``selected_ids`` -- this run's
    real execution set from ``resolve_metrics(config.metric_profiles, ...)``
    -- is what distinguishes "available and actually attempted" from
    "available but this profile didn't ask for it".

    A metric in both ``coverage_plan.selected`` and ``selected_ids`` is
    ``computed`` even when it produced zero reportable rows -- that is a
    data-quality outcome, recorded in ``reason``, never re-labelled as a
    dependency skip.
    """
    skip_reason_by_id = {item.metric_id: item.reason for item in coverage_plan.skipped}
    rows: list[dict[str, object]] = []
    for spec in coverage_plan.selected:
        metric_id = spec.metric_id
        if metric_id not in selected_ids:
            rows.append(
                {
                    "metric": metric_id,
                    "runtime_wired": True,
                    "status": "skipped (not selected by profile)",
                    "rows": 0,
                    "reportable_rows": 0,
                    "reason": "skipped (not selected by profile)",
                }
            )
            continue
        metric_rows = frame[frame["metric"] == metric_id]
        n_rows = int(len(metric_rows))
        n_reportable = (
            int(metric_rows["is_reportable"].fillna(False).sum()) if n_rows else 0
        )
        reason = "" if n_reportable > 0 else _quality_reason(metric_rows)
        rows.append(
            {
                "metric": metric_id,
                "runtime_wired": True,
                "status": "computed",
                "rows": n_rows,
                "reportable_rows": n_reportable,
                "reason": reason,
            }
        )
    for metric_id, reason in skip_reason_by_id.items():
        is_runtime_wired = metric_id in RUNTIME_WIRED_METRIC_IDS
        status = (
            "skipped (missing dependency)"
            if is_runtime_wired
            else reason
        )
        rows.append(
            {
                "metric": metric_id,
                "runtime_wired": is_runtime_wired,
                "status": status,
                "rows": 0,
                "reportable_rows": 0,
                "reason": reason,
            }
        )
    coverage = pd.DataFrame(rows, columns=_COVERAGE_COLUMNS)
    return coverage.sort_values("metric", kind="stable").reset_index(drop=True)


def _run_core_analysis(
    cube: WaterCube,
    aoi_id: str,
    *,
    config: HydroConfig,
    inputs: AnalysisInputs | None = None,
    pixel_size_m: float = 30.0,
    catchment_id: str | None = None,
    git_sha: str | None = None,
    zone_result: ZoneResult | None = None,
    riverscape_bundle: Any = None,
) -> CoreAnalysisResult:
    """Compute metrics and optional spatial checkpoints without publishing."""

    from hydrofragments.output.core import CoreAnalysisResult

    if config.output.spatial_products or config.output.output_dir is not None:
        from hydrofragments.output.finalize import preflight_spatial_outputs

    inputs = inputs or AnalysisInputs()
    drainage = inputs.drainage
    hydroyear_extent = inputs.hydroyear_extent
    max_water_apsec = inputs.max_water_apsec
    median_apsec = inputs.median_apsec
    channel_wet_profiles = inputs.channel_wet_profiles
    channel_segment_lengths_m = inputs.channel_segment_lengths_m

    execution_plan = resolve_execution_plan(
        accelerator=config.compute.accelerator,
        cuda_strict=config.compute.cuda_strict,
    )
    report = validate_inputs(
        cube,
        aoi_id,
        config=config,
        drainage=drainage,
        hydroyear_available=hydroyear_extent is not None,
        dual_composites_available=(
            max_water_apsec is not None and median_apsec is not None
        ),
    )
    if not report.is_valid:
        raise ValueError("; ".join(report.errors))

    resolved_git_sha = git_sha or resolve_git_sha()
    run_id = uuid4().hex
    catchment = catchment_id or aoi_id
    crs = cube.crs or config.spatial.target_crs
    section_area_km2 = (
        float(np.count_nonzero(cube.aoi_mask.values)) * pixel_size_m**2 / 1_000_000.0
    )
    monthly = xr.Dataset(
        {
            "water": cube.water.astype(bool),
            "valid_obs": cube.valid_obs.astype(bool),
        }
    )
    hydroyear_result = None
    if hydroyear_extent is not None:
        hydroyear_result = detect_hy_anchors(
            hydroyear_extent, hydrofragments_config=config
        )

    if config.output.spatial_products or config.output.output_dir is not None:
        preflight_spatial_outputs(
            config,
            cube=cube,
            inputs=inputs,
            hydroyear_result=hydroyear_result,
            zone_result=zone_result or inputs.zones,
            riverscape_bundle=riverscape_bundle,
        )

    available_dependencies = _available_dependencies(
        config=config,
        drainage=drainage,
        hydroyear_available=hydroyear_result is not None,
        dual_composites_available=(
            max_water_apsec is not None and median_apsec is not None
        ),
    )
    metric_plan = resolve_metrics(
        config.metric_profiles,
        available_dependencies=available_dependencies,
        metric_overrides=config.metric_overrides,
    )
    selected_ids = {spec.metric_id for spec in metric_plan.selected}
    internal_ids = {spec.metric_id for spec in metric_plan.internal_support}
    compute_ids = selected_ids | internal_ids
    coverage_plan = registry_wide_plan(available_dependencies=available_dependencies)

    need_section_rows = bool(compute_ids & _SECTION_ROW_METRIC_IDS)
    need_dynamics_cube = bool(selected_ids & _DYNAMICS_CUBE_METRIC_IDS)
    dynamics_collector = None
    if need_dynamics_cube and hydroyear_result is not None:
        dynamics_collector = DynamicsSupportCollector(
            end_dry_anchors=hydroyear_result.anchors,
        )

    spatial_collector = SectionSpatialCollector()
    records: list[MetricRecord] = []
    dynamics_support: DynamicsSupport | None = None
    refuge_states: tuple[EndDryState, ...] = ()
    if need_section_rows or need_dynamics_cube:
        rows = analyze_section_rows(
            monthly["water"],
            section=aoi_id,
            section_area_km2=section_area_km2,
            pixel_size_m=pixel_size_m,
            config=config,
            selected_ids=compute_ids,
            valid_obs=monthly["valid_obs"],
            analysis_mask=cube.analysis_mask,
            dynamics_collector=dynamics_collector,
            end_dry_anchors=(
                hydroyear_result.anchors if hydroyear_result is not None else None
            ),
            spatial_collector=spatial_collector,
        )
        if need_section_rows:
            records = _records_from_section_rows(
                rows,
                run_id=run_id,
                config=config,
                catchment_id=catchment,
                aoi_id=aoi_id,
                resolution_m=pixel_size_m,
                crs=crs,
                source=cube.source,
                git_sha=resolved_git_sha,
            )
        if dynamics_collector is not None:
            dynamics_support, refuge_states = dynamics_collector.finalize()
    records = [record for record in records if record.metric in selected_ids]
    if {"lpsec", "inter_pool_gap"} & selected_ids:
        if not isinstance(drainage, SpatialContext) or not drainage.has_real_channel:
            raise ValueError("channel profile requires a real SpatialContext")
        if channel_wet_profiles is None or channel_segment_lengths_m is None:
            raise ValueError(
                "channel profile requires channel_wet_profiles and "
                "channel_segment_lengths_m"
            )
        records.extend(
            _channel_profile_records(
                cube=cube,
                context=drainage,
                wet_profiles=channel_wet_profiles,
                segment_lengths_m=channel_segment_lengths_m,
                run_id=run_id,
                config=config,
                catchment_id=catchment,
                aoi_id=aoi_id,
                resolution_m=pixel_size_m,
                crs=crs,
                source=cube.source,
                git_sha=resolved_git_sha,
            )
        )
    if {"recurrence", "hydroperiod"} & selected_ids:
        records.extend(
            _temporal_profile_records(
                monthly,
                run_id=run_id,
                config=config,
                catchment_id=catchment,
                aoi_id=aoi_id,
                source=cube.source,
                resolution_m=pixel_size_m,
                crs=crs,
                git_sha=resolved_git_sha,
            )
        )
    if selected_ids & _DYNAMICS_PROFILE_METRIC_IDS and hydroyear_result is not None:
        if dynamics_support is None:
            dynamics_support = DynamicsSupport(lpi_by_month=())
        lpsec_by_month = dynamics_support.lpsec_by_month
        if (
            isinstance(drainage, SpatialContext)
            and drainage.has_real_channel
            and channel_wet_profiles is not None
            and channel_segment_lengths_m is not None
        ):
            lpsec_by_month = _build_lpsec_by_month(
                cube=cube,
                context=drainage,
                wet_profiles=channel_wet_profiles,
                segment_lengths_m=channel_segment_lengths_m,
            )
        dynamics_support = DynamicsSupport(
            lpi_by_month=dynamics_support.lpi_by_month,
            lpsec_by_month=lpsec_by_month,
        )
        analysis_mask_np = None
        if cube.analysis_mask is not None:
            analysis_mask_np = np.asarray(cube.analysis_mask.values, dtype=bool)
        records.extend(
            _dynamics_profile_records(
                selected_metric_ids=frozenset(
                    selected_ids & _DYNAMICS_PROFILE_METRIC_IDS
                ),
                hy_anchors=hydroyear_result,
                monthly_support=dynamics_support,
                refuge_states=refuge_states,
                config=config,
                run_context={
                    "run_id": run_id,
                    "catchment_id": catchment,
                    "aoi_id": aoi_id,
                    "source": cube.source,
                    "resolution_m": pixel_size_m,
                    "crs": crs,
                    "git_sha": resolved_git_sha,
                },
                max_water=max_water_apsec,
                median=median_apsec,
                analysis_mask=analysis_mask_np,
            )
        )
    frame = records_to_frame(records)
    metric_coverage = _build_metric_coverage(
        frame, coverage_plan, selected_ids=selected_ids
    )

    backend_warnings = list(report.warnings)
    if (
        config.compute.accelerator == "auto"
        and execution_plan.fallback_reason is not None
    ):
        backend_warnings.append(
            f"accelerator_fallback: {execution_plan.fallback_reason}"
        )

    return CoreAnalysisResult(
        metrics_table=frame,
        metric_coverage=metric_coverage,
        run_id=run_id,
        git_sha=resolved_git_sha,
        report_warnings=tuple(backend_warnings),
        skipped_metrics=report.skipped_metrics,
        execution_plan_mapping={
            "planned_backend": execution_plan.planned_backend,
            "actual_backend_by_stage": {
                "analyze": "cpu",
                **execution_plan.actual_backend_by_stage,
            },
            "backend_capabilities": execution_plan.capabilities.to_mapping(),
        },
        input_fingerprint={
            "source": cube.source,
            "cadence": cube.cadence,
            "shape": dict(cube.water.sizes),
            "chunks": _describe_chunks(cube.water),
        },
        comparison_context={
            "aoi_id": aoi_id,
            "source": cube.source,
            "resolution_m": pixel_size_m,
            "crs": crs,
            "hydroseason_hy_count": (
                0 if hydroyear_result is None else len(hydroyear_result.anchors)
            ),
        },
        hydroyear_result=hydroyear_result,
        raster_checkpoint=spatial_collector.raster_checkpoint,
        pool_checkpoint_root=spatial_collector.pool_checkpoint_root,
        spatial_grid=spatial_collector.grid,
    )


def analyze(
    cube: WaterCube,
    aoi_id: str,
    *,
    config: HydroConfig,
    inputs: AnalysisInputs | None = None,
    pixel_size_m: float = 30.0,
    catchment_id: str | None = None,
) -> HydroResult:
    """Execute configured metric profiles for one AOI.

    ``inputs`` bundles optional advanced inputs (drainage context, hydroyear
    extent, dual-composite APSEC, channel profiles) -- see
    :class:`hydrofragments.models.AnalysisInputs`. ``inputs.hydroyear_extent``
    enables the external hydroseason adapter. Dynamics additionally requires
    both ``inputs.max_water_apsec`` and ``inputs.median_apsec``; absent
    either composite, the registry reports an explicit dependency skip.
    """
    from hydrofragments.output.core import build_in_memory_manifest

    core = _run_core_analysis(
        cube,
        aoi_id,
        config=config,
        inputs=inputs,
        pixel_size_m=pixel_size_m,
        catchment_id=catchment_id,
    )

    if config.output.output_dir is None:
        manifest_dict = build_in_memory_manifest(config, core)
        return HydroResult(
            metrics_table=core.metrics_table,
            manifest=manifest_dict,
            output_dir=None,
            run_id=core.run_id,
            metric_coverage=core.metric_coverage,
        )

    from hydrofragments.output.finalize import finalize_analysis_bundle

    return finalize_analysis_bundle(
        config,
        core,
        cube=cube,
        inputs=inputs,
        pixel_size_m=pixel_size_m,
    )


def compare_results(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    overrides: Mapping[str, str] | None = None,
):
    """Refuse incompatible scientific settings unless explicitly overridden."""
    return guard_comparison(left, right, overrides=overrides)


__all__ = [
    "AnalysisInputs",
    "ComparisonGuardError",
    "HydroConfig",
    "HydroResult",
    "MetricPlan",
    "SCHEMA_VERSION",
    "ValidationReport",
    "WaterCube",
    "__version__",
    "analyze",
    "compare_results",
    "open_water_cube",
    "validate_inputs",
]
