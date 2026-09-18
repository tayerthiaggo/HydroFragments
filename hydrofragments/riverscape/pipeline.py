"""Riverscape landform orchestrator: source loaders plus the Phase 3-5 kernels.

``build_landform`` is the single entry point that turns a DEA-derived
observed-wet domain plus a drainage network into a :class:`LandformResult`.
It owns two things and nothing else: WHICH sources get loaded (DEM,
Fractional Cover, DEA Waterbodies -- all through injectable callables, so
tests never touch the network) and IN WHAT ORDER the Phase 3-5 kernels run.
Every scientific rule stays owned by the kernel module it already lives in
(``corridor``, ``centreline``, ``terrain``, ``waterbodies``, ``evidence``,
``channel``, ``bridging``, ``riverine``); this module adds no threshold and
no classification of its own.

Import boundary (spec 6a section 1.3): this module must never import
``hydrofragments.spatial`` -- enforced by
``tests/riverscape/test_riverscape_import_boundary.py``. The consequence is
that the reach-label raster, its ``label -> HydroID`` key map, and its
``label -> UpstrDArea`` map are all built by the caller (``spatial`` or
``workflow``, which own ``connectivity_context._build_reach_label_raster``)
and passed in as plain arrays/mappings.

``RiverscapeSourceUnavailable`` is never caught here. ``riverscape.mode``
behaviour -- degrade to occurrence zoning, or propagate -- is the caller's
decision (spec 6a sections 4 and 6), and swallowing the exception here
would make ``mode="required"`` unimplementable.
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage

from hydrofragments.config import RiverscapeConfig
from hydrofragments.io.riverscape_sources import (
    RiverscapeSourceUnavailable,
    load_dem,
    load_fc_percentiles,
    load_waterbodies,
)
from hydrofragments.riverscape.bridging import (
    BridgeCostInputs,
    BridgeResult,
    _empty_bridge_frame,
    bridge_gaps,
    find_gaps,
)
from hydrofragments.riverscape.centreline import CentrelineResult, build_centreline
from hydrofragments.riverscape.channel import RULESET_VERSION as CHANNEL_RULESET_VERSION
from hydrofragments.riverscape.channel import classify_channel
from hydrofragments.riverscape.codes import (
    LANDFORM_IN_CHANNEL,
    LANDFORM_NON_RIVERINE,
    LANDFORM_OFF_CHANNEL_RIVERINE,
)
from hydrofragments.riverscape.corridor import measure_corridor_widths
from hydrofragments.riverscape.evidence import build_evidence
from hydrofragments.riverscape.riverine import (
    RIVERINE_RULESET_VERSION,
    FloodplainEnvelope,
    build_landform_layer,
)
from hydrofragments.riverscape.terrain import build_rem
from hydrofragments.riverscape.waterbodies import (
    WATERBODY_RULESET_VERSION,
    classify_waterbodies,
)

CHANNEL_SOURCE_NONE = 0
CHANNEL_SOURCE_OBSERVED = 1
CHANNEL_SOURCE_BRIDGED = 2

#: ``channel_confidence`` sentinel for "not a channel pixel" (spec 3.1).
CHANNEL_CONFIDENCE_NODATA = 255

#: Columns every Phase 3-5 kernel between them requires. ``HydroID``/
#: ``NextDownID`` come from corridor/centreline/bridging, ``From_Node``/
#: ``To_Node`` from ``terrain._REQUIRED_COLUMNS``, and ``UpstrDArea`` is the
#: floodplain envelope's scaling variable (the caller turns it into the
#: label-keyed ``upstr_darea`` mapping).
REQUIRED_DRAINAGE_COLUMNS = (
    "HydroID",
    "NextDownID",
    "UpstrDArea",
    "From_Node",
    "To_Node",
)


@dataclass(frozen=True)
class LandformResult:
    """Everything Phase 6a's zoning adapter and Phase 6b's exports consume.

    ``grid`` is declared ``Any | None`` rather than ``SpatialGrid | None``
    on purpose: ``SpatialGrid`` lives in ``hydrofragments.output.spatial``,
    which is an output-layer type this package must not reach for (see the
    module docstring's import-boundary note). It is last in the field order
    only because it is the one field with a default; spec section 3.1 lists
    it before ``provenance``.
    """

    landform: np.ndarray
    channel_source: np.ndarray
    channel_confidence: np.ndarray
    rule_id: np.ndarray
    rem: np.ndarray
    bridged_mask: np.ndarray
    channel_bridges: Any
    unbridged: tuple
    envelope: FloodplainEnvelope
    provenance: Mapping[str, Any]
    degraded_reasons: tuple[str, ...]
    grid: Any | None = None


def with_grid(result: LandformResult, grid: Any) -> LandformResult:
    """Return ``result`` with ``grid`` attached.

    The Phase 6b seam: 6b hands a real
    ``hydrofragments.output.spatial.SpatialGrid`` back to a
    :class:`LandformResult` before writing rasters. Phase 6a's
    ``zones_from_riverscape`` returns a ``ZoneResult`` (which carries its own
    grid), so it does not call this itself.
    """
    return replace(result, grid=grid)


def validate_drainage_columns(drainage: Any) -> None:
    """Raise ``ValueError`` when ``drainage`` cannot drive the pipeline.

    A missing column is a programming error in the caller's drainage
    preparation, not a data-availability problem, so it must never be
    mistaken for a ``riverscape.mode`` fallback (spec section 3.2).
    """
    if drainage is None:
        raise ValueError("drainage must contain at least one line feature")
    if getattr(drainage, "empty", True):
        raise ValueError("drainage must contain at least one line feature")
    missing = [name for name in REQUIRED_DRAINAGE_COLUMNS if name not in drainage.columns]
    if missing:
        raise ValueError(f"drainage missing required columns: {missing}")
    if getattr(drainage, "geometry", None) is None:
        raise ValueError("drainage must carry line geometry")


def _package_channel_source(observed: Any, bridged: Any) -> np.ndarray:
    """Per-pixel channel provenance: 1 observed, 2 bridged-only, 0 elsewhere.

    Observed is painted last so a pixel that is both (which
    ``bridge_gaps`` already excludes, but a caller-injected bridging stub
    might not) reads as observed -- a real observation always outranks a
    reconstruction.
    """
    observed_mask = np.asarray(observed, dtype=bool)
    bridged_mask = np.asarray(bridged, dtype=bool)
    if bridged_mask.shape != observed_mask.shape:
        raise ValueError("observed and bridged masks must share shape")
    source = np.full(observed_mask.shape, CHANNEL_SOURCE_NONE, dtype=np.uint8)
    source[bridged_mask & ~observed_mask] = CHANNEL_SOURCE_BRIDGED
    source[observed_mask] = CHANNEL_SOURCE_OBSERVED
    return source


def _package_channel_confidence(confidence: Any, channel: Any) -> np.ndarray:
    """Evidence confidence on channel pixels, ``255`` everywhere else."""
    values = np.asarray(confidence, dtype=np.uint8)
    mask = np.asarray(channel, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("confidence and channel must share shape")
    return np.where(mask, values, CHANNEL_CONFIDENCE_NODATA).astype(np.uint8)


def _merge_degraded_reasons(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Stable sorted unique union of every kernel's degraded reasons."""
    return tuple(sorted({reason for group in groups for reason in group}))


def _grid_bounds(transform: Affine, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return ``(minx, miny, maxx, maxy)`` for the zoning grid.

    Derived from ``transform``/``shape`` rather than from ``geobox`` so the
    waterbody loader's bbox never depends on the geobox object's API -- the
    geobox is only ever forwarded to the raster loaders.
    """
    height, width = shape
    west, north = transform * (0.0, 0.0)
    east, south = transform * (float(width), float(height))
    return (min(west, east), min(north, south), max(west, east), max(north, south))


def _loaded_array(data: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    values = np.asarray(getattr(data, "values", data), dtype=np.float32)
    if values.shape != shape:
        raise ValueError(
            f"{name} shape {values.shape} does not match the zoning grid {shape}"
        )
    return values


def _band_stack(bands: Mapping[str, Any], name: str, shape: tuple[int, int]) -> np.ndarray:
    """Return one Fractional Cover band as a ``(time, y, x)`` float array."""
    if name not in bands:
        raise RiverscapeSourceUnavailable(
            f"fractional cover load returned no {name!r} band"
        )
    band = bands[name]
    values = np.asarray(getattr(band, "values", band), dtype=float)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != shape:
        raise ValueError(
            f"fractional cover band {name!r} shape {values.shape} does not match "
            f"the zoning grid {shape}"
        )
    return values


def _band_median(bands: Mapping[str, Any], name: str, shape: tuple[int, int]) -> np.ndarray:
    """Across-year median of one FC band, as a 2-D float32 array.

    ``load_fc_percentiles`` already converts nodata sentinels to NaN, so an
    all-missing pixel is legitimately all-NaN; the warning that
    ``np.nanmedian`` emits for it carries no information the caller can act
    on (the resulting NaN is handled downstream by ``_unit_interval`` in
    ``bridging.build_cost_surface``).
    """
    stack = _band_stack(bands, name, shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=0).astype(np.float32)


def _rasterize(geometry: Any, *, transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    return rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)


def _corridor_rasters(
    drainage: Any,
    corridor_widths_m: Mapping[str, float],
    *,
    transform: Affine,
    shape: tuple[int, int],
    pixel_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize per-reach corridors into the three grids later steps need.

    Returns ``(corridor_mask, corridor_width_m, line_distance_m)``.
    ``corridor_width_m`` takes the widest claimant where buffers overlap
    near junctions -- the corridor is a search-space bound, so the more
    permissive value is the correct one to keep. ``line_distance_m`` is the
    Euclidean distance to the nearest rasterized AHGF line pixel, the
    ``BridgeCostInputs`` term that keeps least-cost routes near the mapped
    channel. A grid with no line pixels at all gets ``inf``, which
    ``bridging._unit_interval`` clips to the maximum penalty.
    """
    width = np.zeros(shape, dtype=np.float32)
    lines = np.zeros(shape, dtype=bool)
    for hydro_id, geometry in zip(drainage["HydroID"], drainage.geometry):
        if geometry is None or geometry.is_empty:
            continue
        key = str(hydro_id)
        if key not in corridor_widths_m:
            raise ValueError(f"corridor_widths_m missing entry for reach {key!r}")
        width_m = float(corridor_widths_m[key])
        buffered = _rasterize(geometry.buffer(width_m), transform=transform, shape=shape)
        width[buffered] = np.maximum(width[buffered], width_m)
        lines |= _rasterize(geometry, transform=transform, shape=shape)
    if lines.any():
        line_distance_m = (
            ndimage.distance_transform_edt(~lines) * pixel_m
        ).astype(np.float32)
    else:
        line_distance_m = np.full(shape, np.inf, dtype=np.float32)
    return width > 0, width, line_distance_m


def _line_fallback_raster(
    drainage: Any,
    line_fallback_reaches: tuple[str, ...],
    *,
    transform: Affine,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize the AHGF lines of reaches with no water inside their corridor.

    A line-fallback reach's line "remains a topology/profile guide only and
    never becomes channel" (``centreline.build_centreline``), so only the
    line itself is painted -- not its corridor buffer. The evidence layer
    folds this into its topology family, and it only reaches the channel
    when ``include_line_fallback_in_channel`` is on.
    """
    fallback = np.zeros(shape, dtype=bool)
    if not line_fallback_reaches:
        return fallback
    wanted = set(line_fallback_reaches)
    for hydro_id, geometry in zip(drainage["HydroID"], drainage.geometry):
        if str(hydro_id) not in wanted or geometry is None or geometry.is_empty:
            continue
        fallback |= _rasterize(geometry, transform=transform, shape=shape)
    return fallback


def _run_bridging(
    cfg: RiverscapeConfig,
    *,
    observed: np.ndarray,
    centreline: np.ndarray,
    drainage: Any,
    reach_labels: np.ndarray,
    reach_keys: Mapping[int, str],
    corridor_widths_m: Mapping[str, float],
    corridor_mask: np.ndarray,
    corridor_width_m: np.ndarray,
    line_distance_m: np.ndarray,
    frequency: np.ndarray,
    domain: np.ndarray,
    rem: np.ndarray,
    trough_depth: np.ndarray,
    bare_fraction: np.ndarray,
    green_pct: np.ndarray,
    npv_pct: np.ndarray,
    off_channel_mask: np.ndarray,
    transform: Affine,
    pixel_m: float,
) -> BridgeResult:
    """Discover and bridge channel gaps, or return a typed empty result.

    ``_empty_bridge_frame`` is reused from ``bridging`` rather than
    reconstructed here so the disabled path's ``channel_bridges`` schema can
    never drift from the enabled path's.
    """
    if not cfg.bridge_enabled:
        return BridgeResult(
            bridged_mask=np.zeros(observed.shape, dtype=bool),
            channel_bridges=_empty_bridge_frame(drainage.crs),
            unbridged=(),
            degraded_reasons=(),
        )
    gaps = find_gaps(
        observed,
        centreline,
        drainage,
        reach_labels,
        reach_keys,
        corridor_widths_m,
        transform=transform,
        pixel_m=pixel_m,
    )
    cost_inputs = BridgeCostInputs(
        rem=rem,
        trough_depth=trough_depth,
        frequency=frequency,
        bare_fraction=bare_fraction,
        green_pct=green_pct,
        npv_pct=npv_pct,
        line_distance_m=line_distance_m,
        corridor_width_m=corridor_width_m,
        corridor_mask=corridor_mask,
        domain=domain,
        off_channel_mask=off_channel_mask,
    )
    result = bridge_gaps(
        gaps.gaps,
        observed,
        cost_inputs,
        transform=transform,
        crs=drainage.crs,
        pixel_m=pixel_m,
        weights=cfg.bridge_cost_weights,
        bridge_max_length_m=cfg.bridge_max_length_m,
        bridge_max_cost_per_m=cfg.bridge_max_cost_per_m,
        bridge_rem_max_m=cfg.bridge_rem_max_m,
        trough_depth_m=cfg.trough_depth_m,
        riparian_green_pct=cfg.riparian_green_pct,
        narrow_width_px=cfg.narrow_width_px,
    )
    return replace(
        result,
        degraded_reasons=tuple(result.degraded_reasons) + tuple(gaps.degraded_reasons),
    )


def _build_provenance(
    *,
    cfg: RiverscapeConfig,
    years: tuple[int, int],
    landform: np.ndarray,
    domain: np.ndarray,
    water_seed: np.ndarray,
    observed: np.ndarray,
    bridged: np.ndarray,
    bridges: Any,
    unbridged: tuple,
    bare_threshold_pct: float,
    envelope: FloodplainEnvelope,
    centreline: CentrelineResult,
    reach_total: int,
) -> dict[str, Any]:
    """Stable provenance keys Phase 6b copies straight into the manifest."""
    causes: dict[str, int] = {}
    if len(bridges):
        causes = {
            str(cause): int(count)
            for cause, count in sorted(bridges["gap_cause"].value_counts().items())
        }
    return {
        "channel_ruleset_version": CHANNEL_RULESET_VERSION,
        "waterbody_ruleset_version": WATERBODY_RULESET_VERSION,
        "riverine_ruleset_version": RIVERINE_RULESET_VERSION,
        "dem_product": cfg.dem_product,
        "dem_band": cfg.dem_band,
        "fc_product": cfg.fc_product,
        "fc_bands": (cfg.bare_band, cfg.green_band, cfg.npv_band),
        "waterbodies_source": cfg.waterbodies_source,
        "years": (int(years[0]), int(years[1])),
        "bridge_enabled": bool(cfg.bridge_enabled),
        "bare_threshold_pct": float(bare_threshold_pct),
        "envelope": {
            "a": float(envelope.a),
            "b": float(envelope.b),
            "n_bins": int(envelope.n_bins),
        },
        "pixel_counts": {
            "domain": int(np.count_nonzero(domain)),
            "water_seed": int(np.count_nonzero(water_seed)),
            "observed_channel": int(np.count_nonzero(observed)),
            "bridged_channel": int(np.count_nonzero(bridged)),
            "in_channel": int(np.count_nonzero(landform == LANDFORM_IN_CHANNEL)),
            "off_channel_riverine": int(
                np.count_nonzero(landform == LANDFORM_OFF_CHANNEL_RIVERINE)
            ),
            "non_riverine": int(np.count_nonzero(landform == LANDFORM_NON_RIVERINE)),
        },
        "bridge_counts_by_cause": causes,
        "unbridged_counts_by_reason": {
            str(reason): int(count)
            for reason, count in sorted(Counter(item.reason for item in unbridged).items())
        },
        "reach_counts": {
            "total": int(reach_total),
            "line_fallback": len(centreline.line_fallback_reaches),
            "multithread": len(centreline.multithread_reaches),
        },
    }


def build_landform(
    domain: np.ndarray,
    frequency: np.ndarray,
    drainage: "gpd.GeoDataFrame",
    reach_labels: np.ndarray,
    reach_keys: Mapping[int, str],
    upstr_darea: Mapping[int, float],
    *,
    geobox: Any,
    transform: Affine,
    pixel_m: float,
    cfg: RiverscapeConfig,
    years: tuple[int, int],
    load_dem: Callable[..., "xr.DataArray"] = load_dem,
    load_fc_percentiles: Callable[..., dict] = load_fc_percentiles,
    load_waterbodies: Callable[..., "gpd.GeoDataFrame"] = load_waterbodies,
) -> LandformResult:
    """Load sources and run corridor -> ... -> riverine, in that order.

    ``domain`` is the observed-wet analysis domain (``riverscape.domain.
    wet_domain``); ``frequency`` is percent-scale wet-observation frequency
    on the same grid. ``reach_labels``/``reach_keys``/``upstr_darea`` come
    from the caller (see the module docstring's import-boundary note);
    ``reach_keys`` and ``upstr_darea`` are both keyed by the integer LABEL
    value found in ``reach_labels``, not by ``HydroID``.

    Raises :class:`~hydrofragments.io.riverscape_sources.RiverscapeSourceUnavailable`
    on loader failure -- propagated, never caught. Raises ``ValueError`` for
    shape/dtype/schema problems, which are programming errors rather than
    source-availability problems. Never imports ``hydrofragments.spatial``.
    """
    domain_mask = np.asarray(domain)
    frequency_values = np.asarray(frequency, dtype=float)
    labels = np.asarray(reach_labels)

    if domain_mask.dtype != np.bool_:
        raise ValueError("domain must be a boolean array")
    if domain_mask.ndim != 2:
        raise ValueError("domain must be a 2-D array")
    if frequency_values.shape != domain_mask.shape:
        raise ValueError("domain and frequency must share shape")
    if labels.shape != domain_mask.shape:
        raise ValueError("domain and reach_labels must share shape")
    if not np.all(np.isfinite(frequency_values[domain_mask])):
        raise ValueError("frequency must be finite inside the domain")
    if not np.isfinite(pixel_m) or pixel_m <= 0:
        raise ValueError("pixel_m must be finite and positive")
    validate_drainage_columns(drainage)

    shape = domain_mask.shape

    # Step 2: water seed -- the same percent-scale convention Phase 4's
    # evidence layer uses; cfg.f_seed is a fraction, so convert once here.
    water_seed = domain_mask & (frequency_values >= 100.0 * cfg.f_seed)

    # Step 3: load sources. No try/except -- see the module docstring.
    dem = _loaded_array(
        load_dem(geobox, product=cfg.dem_product, band=cfg.dem_band), shape, "dem"
    )
    fc_bands = load_fc_percentiles(
        geobox,
        product=cfg.fc_product,
        bands=(cfg.bare_band, cfg.green_band, cfg.npv_band),
        years=years,
    )
    bare_yearly = _band_stack(fc_bands, cfg.bare_band, shape)
    green_pct = _band_median(fc_bands, cfg.green_band, shape)
    npv_pct = _band_median(fc_bands, cfg.npv_band, shape)
    waterbody_polygons = load_waterbodies(
        _grid_bounds(transform, shape),
        drainage.crs,
        source=cfg.waterbodies_source,
    )

    # Step 4: corridor widths bound every later search space.
    corridor = measure_corridor_widths(
        drainage,
        water_seed,
        transform=transform,
        pixel_m=pixel_m,
        f_seed=cfg.f_seed,
        corridor_min_m=cfg.corridor_min_m,
        corridor_max_m=cfg.corridor_max_m,
        alignment_quantile=cfg.alignment_quantile,
    )

    # Step 5: centreline conflation runs on the WATER SEED, not the full
    # domain -- the skeleton must follow observed water, not the whole
    # observed-wet extent (spec section 4 step 5).
    centreline = build_centreline(
        drainage,
        water_seed,
        corridor.widths_m,
        transform=transform,
        pixel_m=pixel_m,
    )

    # Step 6: REM, local trough depth, slope.
    terrain = build_rem(
        drainage,
        dem,
        transform=transform,
        pixel_m=pixel_m,
        profile_bin_m=cfg.profile_bin_m,
        profile_percentile=cfg.profile_percentile,
        corridor_widths_m=corridor.widths_m,
        rem_k=cfg.rem_k,
        rem_max_distance_m=cfg.rem_max_distance_m,
        trough_radius_m=cfg.trough_radius_m,
    )

    corridor_mask, corridor_width_m, line_distance_m = _corridor_rasters(
        drainage,
        corridor.widths_m,
        transform=transform,
        shape=shape,
        pixel_m=pixel_m,
    )

    # Step 7: waterbody riverine/off-channel roles, judged against the
    # conflated centreline.
    waterbodies = classify_waterbodies(
        waterbody_polygons,
        centreline.skeleton,
        transform=transform,
        pixel_m=pixel_m,
    )

    # Step 8: per-pixel evidence. build_evidence owns the
    # calibrate_bare_threshold call internally; do not calibrate twice.
    evidence = build_evidence(
        frequency_values,
        bare_yearly,
        waterbodies.riverine_mask,
        terrain.rem,
        terrain.trough_depth,
        domain_mask,
        centreline.skeleton,
        corridor_mask,
        f_chan_high=cfg.f_chan_high,
        bare_threshold_floor_pct=cfg.bare_threshold_floor_pct,
        bare_year_fraction=cfg.bare_year_fraction,
        h_chan_m=cfg.h_chan_m,
        trough_depth_m=cfg.trough_depth_m,
        line_fallback_mask=_line_fallback_raster(
            drainage,
            centreline.line_fallback_reaches,
            transform=transform,
            shape=shape,
        ),
    )

    # Step 9: ordered observed-channel rules.
    channel = classify_channel(
        evidence,
        domain_mask,
        centreline.skeleton,
        water_seed,
        labels,
        reach_keys,
        pixel_m=pixel_m,
        width_growth_factor=cfg.width_growth_factor,
        min_channel_confidence=cfg.min_channel_confidence,
        include_line_fallback_in_channel=cfg.include_line_fallback_in_channel,
    )
    observed = np.asarray(channel.channel, dtype=bool)

    # Step 10: gap bridging (or a typed empty result when disabled).
    bridge = _run_bridging(
        cfg,
        observed=observed,
        centreline=centreline.skeleton,
        drainage=drainage,
        reach_labels=labels,
        reach_keys=reach_keys,
        corridor_widths_m=corridor.widths_m,
        corridor_mask=corridor_mask,
        corridor_width_m=corridor_width_m,
        line_distance_m=line_distance_m,
        frequency=frequency_values,
        domain=domain_mask,
        rem=terrain.rem,
        trough_depth=terrain.trough_depth,
        bare_fraction=evidence.bare_fraction,
        green_pct=green_pct,
        npv_pct=npv_pct,
        off_channel_mask=waterbodies.off_channel_mask,
        transform=transform,
        pixel_m=pixel_m,
    )

    # A bridged pixel means "channel we could not observe": bridge_gaps
    # already excludes the observed domain and the observed channel, and
    # re-applying that intersection here makes the invariant hold for an
    # injected bridging stub too. classify_hydroperiod depends on it --
    # it rejects an unobserved_mask that overlaps the observed domain.
    bridged = np.asarray(bridge.bridged_mask, dtype=bool) & ~domain_mask & ~observed
    channel_all = observed | bridged

    # Step 11: riverine vs non-riverine. assemble_landform only assigns
    # codes inside its domain argument, so bridged pixels must be added to
    # the domain here or they would stay landform 0 while hydroperiod says
    # 4 -- which combine_zones rejects as a zoned-extent disagreement.
    riverine = build_landform_layer(
        terrain.rem,
        terrain.slope_deg,
        domain_mask | bridged,
        channel_all,
        labels,
        upstr_darea,
        envelope_quantile=cfg.envelope_quantile,
        envelope_min_bin_pixels=cfg.envelope_min_bin_pixels,
        envelope_h_max_m=cfg.envelope_h_max_m,
        slope_max_deg=cfg.slope_max_deg,
    )

    # Step 12: package.
    return LandformResult(
        landform=riverine.landform,
        channel_source=_package_channel_source(observed, bridged),
        channel_confidence=_package_channel_confidence(channel.confidence, channel_all),
        rule_id=np.where(
            channel_all, np.asarray(channel.rule_id, dtype=np.uint8), 0
        ).astype(np.uint8),
        rem=np.asarray(terrain.rem, dtype=np.float32),
        bridged_mask=bridged,
        channel_bridges=bridge.channel_bridges,
        unbridged=bridge.unbridged,
        envelope=riverine.envelope,
        provenance=_build_provenance(
            cfg=cfg,
            years=years,
            landform=riverine.landform,
            domain=domain_mask,
            water_seed=water_seed,
            observed=observed,
            bridged=bridged,
            bridges=bridge.channel_bridges,
            unbridged=bridge.unbridged,
            bare_threshold_pct=evidence.calibration.threshold_pct,
            envelope=riverine.envelope,
            centreline=centreline,
            reach_total=len(drainage),
        ),
        degraded_reasons=_merge_degraded_reasons(
            corridor.degraded_reasons,
            tuple(
                f"reach_{key}_line_fallback"
                for key in centreline.line_fallback_reaches
            ),
            terrain.degraded_reasons,
            evidence.degraded_reasons,
            channel.degraded_reasons,
            bridge.degraded_reasons,
            riverine.envelope.degraded_reasons,
        ),
    )


__all__ = [
    "CHANNEL_CONFIDENCE_NODATA",
    "CHANNEL_SOURCE_BRIDGED",
    "CHANNEL_SOURCE_NONE",
    "CHANNEL_SOURCE_OBSERVED",
    "REQUIRED_DRAINAGE_COLUMNS",
    "LandformResult",
    "build_landform",
    "validate_drainage_columns",
    "with_grid",
]
