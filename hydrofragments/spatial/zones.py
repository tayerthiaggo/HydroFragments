"""Zone masks: occurrence-only zoning and landform x hydroperiod combination."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from scipy import ndimage

import rioxarray  # noqa: F401 — registers the .rio accessor for DataArray

from hydrofragments.hydroperiod.codes import (
    HYDROPERIOD_CODES,
    HYDROPERIOD_MARGINAL,
    HYDROPERIOD_OUTSIDE,
    HYDROPERIOD_PERSISTENT,
    HYDROPERIOD_SEASONAL,
    HYDROPERIOD_UNOBSERVED,
)
from hydrofragments.hydroperiod.classify import classify_hydroperiod
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.riverscape.codes import (
    LANDFORM_CODES,
    LANDFORM_IN_CHANNEL,
    LANDFORM_OFF_CHANNEL_RIVERINE,
    LANDFORM_OUTSIDE,
)
from hydrofragments.riverscape.domain import wet_domain
from hydrofragments.riverscape.pipeline import build_landform, with_grid

if TYPE_CHECKING:
    from hydrofragments.io.dea import WoStatistics

_ZONE_MODES = frozenset({"occurrence", "riverscape"})


@dataclass(frozen=True)
class ZoneResult:
    mask: np.ndarray
    emitted_zones: tuple[int, ...]
    has_zone_1: bool
    source: str = "occurrence"
    grid: SpatialGrid | None = None
    mode: str = "occurrence"
    crosstab: np.ndarray | None = None
    degraded_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in _ZONE_MODES:
            raise ValueError(f"ZoneResult mode must be one of {sorted(_ZONE_MODES)}")
        if self.mode == "riverscape" and self.crosstab is None:
            raise ValueError("riverscape ZoneResult requires a crosstab")
        if self.crosstab is not None and np.shape(self.crosstab) != np.shape(self.mask):
            raise ValueError("ZoneResult crosstab must share the mask shape")

    def as_dataarray(self) -> xr.DataArray:
        """Return the zone mask as a georeferenced ``DataArray``."""
        if self.grid is None:
            raise ValueError("ZoneResult has no spatial grid contract")
        data = xr.DataArray(
            self.mask,
            dims=(self.grid.y_dim, self.grid.x_dim),
            coords={self.grid.y_dim: self.grid.y, self.grid.x_dim: self.grid.x},
            attrs={"source": self.source},
        )
        return data.rio.write_crs(self.grid.crs)


def _attach_grid(
    result: ZoneResult,
    template: xr.DataArray | np.ndarray | None,
    *,
    require_georeference: bool = False,
) -> ZoneResult:
    if template is None or not isinstance(template, xr.DataArray):
        return result
    grid = SpatialGrid.from_dataarray(template, require_georeference=require_georeference)
    if grid is None:
        return result
    if result.mask.shape != (grid.height, grid.width):
        raise ValueError("zone mask shape does not align with spatial grid")
    return replace(result, grid=grid)


def build_zones(
    occurrence: np.ndarray,
    *,
    max_wet_mask: np.ndarray,
    valid_count: np.ndarray,
    drainage_mask: np.ndarray | None = None,
    t_persist: float = 0.50,
    t_season: float = 0.10,
    min_valid_obs: int = 20,
) -> ZoneResult:
    """Build mutually exclusive Zones 1-4; zero means outside zoned extent.

    Without drainage, Zone 1 is absent. Zone 2 then represents persistent
    water collectively and is never split using wet-mask morphology.

    ``occurrence`` is PERCENT-scale (0-100), matching the convention used
    throughout the rest of the codebase (``compute_occurrence``,
    ``WoStatistics.frequency``). ``t_persist``/``t_season`` are received as
    FRACTIONS in ``[0, 1]`` (``config.py``'s validated range for
    ``ZonesConfig``) and are converted to percent exactly once, at this
    function's own boundary, before being compared against ``occurrence``.
    """
    frequency = np.asarray(occurrence, dtype=float)
    max_wet = np.asarray(max_wet_mask, dtype=bool)
    support = np.asarray(valid_count)
    if frequency.ndim != 2:
        raise ValueError("occurrence must be a 2-D array")
    if max_wet.shape != frequency.shape:
        raise ValueError("occurrence and max_wet_mask must share shape")
    if support.shape != frequency.shape:
        raise ValueError("occurrence and valid_count must share shape")
    if min_valid_obs < 1:
        raise ValueError("min_valid_obs must be at least 1")
    if not 0.0 <= t_season < t_persist <= 1.0:
        raise ValueError("zone thresholds require 0 <= t_season < t_persist <= 1")

    # Normalize fraction thresholds to percent ONCE, at this boundary, since
    # occurrence is percent-scale. Do this after the fraction-range
    # validation above so config.py's [0, 1] invariant is still checked in
    # its native (fraction) units.
    t_persist_pct = t_persist * 100.0
    t_season_pct = t_season * 100.0

    valid = max_wet & np.isfinite(frequency) & (support >= min_valid_obs)
    mask = np.zeros(frequency.shape, dtype=np.uint8)
    mask[valid & (frequency < t_season_pct)] = 4
    mask[valid & (frequency >= t_season_pct) & (frequency <= t_persist_pct)] = 3
    mask[valid & (frequency > t_persist_pct)] = 2

    if drainage_mask is None:
        return ZoneResult(mask=mask, emitted_zones=(2, 3, 4), has_zone_1=False)

    drainage = np.asarray(drainage_mask, dtype=bool)
    if drainage.shape != frequency.shape:
        raise ValueError("occurrence and drainage_mask must share shape")
    adjacent = ndimage.binary_dilation(
        drainage, structure=np.ones((3, 3), dtype=bool)
    )
    zone_1 = drainage | (adjacent & valid & (frequency > t_persist_pct))
    mask[zone_1] = 1
    return ZoneResult(mask=mask, emitted_zones=(1, 2, 3, 4), has_zone_1=True)


def zones_from_wo_statistics(
    stats: "WoStatistics",
    *,
    config: Any,
    drainage_mask: np.ndarray | None = None,
) -> ZoneResult:
    """Adapt a W1.1 ``WoStatistics`` object into a ``build_zones`` call.

    Maps ``stats.frequency`` -> ``occurrence`` (both already percent-scale,
    0-100), ``stats.count_clear`` -> ``valid_count``, and
    ``stats.count_wet > 0`` -> ``max_wet_mask``. Thresholds and the support
    floor are read from ``config.zones.t_persist``/``config.zones.t_season``
    and ``config.validity.min_valid_obs`` -- this adapter does not invent its
    own defaults, it forwards the caller's resolved configuration.

    ``drainage_mask`` is passed straight through, unchanged.

    The returned ``ZoneResult`` is stamped with ``source=stats.product`` (the
    DEA product id from W1.1), not the generic ``"occurrence"`` default, so a
    caller can tell a DEA-derived ``ZoneResult`` apart from a local-cube one
    and see exactly which product built it.

    ``stats.frequency``/``count_wet``/``count_clear`` may be Dask-backed
    (per ``WoStatistics``'s own contract); this adapter does not force
    materialization itself -- ``build_zones``'s internal ``np.asarray(...)``
    calls perform that conversion regardless, the moment it runs.
    """
    result = build_zones(
        stats.frequency,
        max_wet_mask=stats.count_wet > 0,
        valid_count=stats.count_clear,
        drainage_mask=drainage_mask,
        t_persist=config.zones.t_persist,
        t_season=config.zones.t_season,
        min_valid_obs=config.validity.min_valid_obs,
    )
    result = replace(result, source=stats.product)
    return _attach_grid(result, stats.frequency)


def _resolve_years(stats: "WoStatistics", years: tuple[int, int] | None) -> tuple[int, int]:
    """Resolve the inclusive ``(start_year, end_year)`` Fractional Cover window.

    An explicit ``years`` always wins. Otherwise ``stats.years`` is used when
    the statistics object carries it (spec section 3.3); ``WoStatistics`` does
    not declare that field today, so a caller that omits both gets a
    ``ValueError`` rather than a silently guessed window -- the FC stack's
    year range is a scientific input, not a default.
    """
    if years is not None:
        return (int(years[0]), int(years[1]))
    available = getattr(stats, "years", None)
    if not available:
        raise ValueError(
            "zones_from_riverscape requires years when stats carries no years"
        )
    return (int(min(available)), int(max(available)))


def zones_from_riverscape_with_landform(
    stats: "WoStatistics",
    *,
    drainage: Any,
    config: Any,
    reach_labels: np.ndarray,
    reach_keys: Any,
    upstr_darea: Any,
    geobox: Any,
    transform: Any,
    pixel_m: float = 30.0,
    years: tuple[int, int] | None = None,
) -> tuple[ZoneResult, "LandformResult"]:
    """Build a riverscape ``ZoneResult`` AND return the landform behind it.

    Four steps, no new science (parent spec section 5): the observed-wet
    domain comes from ``wet_domain`` over the same ``stats`` the occurrence
    path reads; ``build_landform`` produces the landform layer; the
    hydroperiod layer is classified over that same domain with the
    landform's bridged mask supplied as ``unobserved_mask``; and
    ``combine_zones`` derives the crosstab and legacy zone mask from the two
    layers.

    Bridged pixels come out landform 1 / hydroperiod 4, so legacy Zone 1
    includes them. They lie strictly outside ``domain``
    (``bridging.bridge_gaps`` and ``riverscape.pipeline.build_landform`` both
    enforce that), which is what lets ``classify_hydroperiod`` accept them as
    ``unobserved_mask`` -- it rejects an overlap.

    ``reach_labels``/``reach_keys``/``upstr_darea`` are built by the caller
    (``hydrofragments.riverscape`` must not import ``hydrofragments.spatial``,
    so it cannot build them itself) and forwarded unchanged. The returned
    ``ZoneResult`` is stamped with ``source=stats.product``, matching
    ``zones_from_wo_statistics``'s convention, and carries a grid attached
    from ``stats.frequency``.

    Phase 6b (spec 6b section 3.2) added the second return value. Phase 6a
    dropped the ``LandformResult`` here, which forced any exporter to reload
    DEM / Fractional Cover / Waterbodies to get it back. The same grid that
    lands on the ``ZoneResult`` is attached to the ``LandformResult`` via
    ``riverscape.pipeline.with_grid`` -- the documented seam for handing an
    output-layer ``SpatialGrid`` back across the import boundary.
    """
    resolved_years = _resolve_years(stats, years)
    domain = wet_domain(stats, min_valid_obs=config.validity.min_valid_obs)
    frequency = np.asarray(stats.frequency, dtype=float)

    landform_result = build_landform(
        domain,
        frequency,
        drainage,
        reach_labels,
        reach_keys,
        upstr_darea,
        geobox=geobox,
        transform=transform,
        pixel_m=pixel_m,
        cfg=config.riverscape,
        years=resolved_years,
    )

    hydroperiod = classify_hydroperiod(
        frequency,
        domain,
        t_persist=config.zones.t_persist,
        t_season=config.zones.t_season,
        unobserved_mask=landform_result.bridged_mask,
    )

    result = combine_zones(
        landform_result.landform,
        hydroperiod.classes,
        source=stats.product,
        degraded_reasons=landform_result.degraded_reasons,
    )
    zone_result = _attach_grid(result, stats.frequency)
    return zone_result, with_grid(landform_result, zone_result.grid)


def zones_from_riverscape(
    stats: "WoStatistics",
    *,
    drainage: Any,
    config: Any,
    reach_labels: np.ndarray,
    reach_keys: Any,
    upstr_darea: Any,
    geobox: Any,
    transform: Any,
    pixel_m: float = 30.0,
    years: tuple[int, int] | None = None,
) -> ZoneResult:
    """Build a riverscape ``ZoneResult`` from DEA stats + drainage context.

    Thin wrapper over :func:`zones_from_riverscape_with_landform`, kept for
    API stability (spec 6b section 3.2): callers that only need zoning keep
    a single return value. ``workflow`` uses the landform-returning form.
    """
    zone_result, _landform = zones_from_riverscape_with_landform(
        stats,
        drainage=drainage,
        config=config,
        reach_labels=reach_labels,
        reach_keys=reach_keys,
        upstr_darea=upstr_darea,
        geobox=geobox,
        transform=transform,
        pixel_m=pixel_m,
        years=years,
    )
    return zone_result


def _validated_codes(values: np.ndarray, allowed: frozenset[int], name: str) -> np.ndarray:
    invalid = sorted(set(np.unique(values).tolist()) - set(allowed))
    if invalid:
        raise ValueError(f"{name} has invalid codes: {invalid}")
    return values.astype(np.uint8)


def combine_zones(
    landform,
    hydroperiod,
    *,
    source: str = "riverscape",
    degraded_reasons: tuple[str, ...] = (),
) -> ZoneResult:
    """Derive zones from the landform and hydroperiod layers; no new logic.

    ``crosstab = landform * 10 + hydroperiod`` (0 outside). Legacy view:
    Z1 = in-channel (any hydroperiod, including unobserved bridges);
    Z2/Z3/Z4 = off-channel riverine x persistent/seasonal/marginal;
    non-riverine water is 0. The two layers must agree on the zoned extent,
    and ``unobserved`` is only valid on in-channel pixels.

    ``emitted_zones``/``has_zone_1`` are derived from ``mask`` (the zones
    actually present), matching ``build_zones``'s convention -- not declared
    by mode. A degraded run with zero in-channel pixels reports
    ``has_zone_1=False``.
    """
    landform_values = np.asarray(landform)
    hydroperiod_values = np.asarray(hydroperiod)
    if landform_values.ndim != 2:
        raise ValueError("landform must be a 2-D array")
    if hydroperiod_values.shape != landform_values.shape:
        raise ValueError("landform and hydroperiod must share shape")
    lf = _validated_codes(landform_values, LANDFORM_CODES, "landform")
    hp = _validated_codes(hydroperiod_values, HYDROPERIOD_CODES, "hydroperiod")

    if np.any((lf == LANDFORM_OUTSIDE) != (hp == HYDROPERIOD_OUTSIDE)):
        raise ValueError("landform and hydroperiod disagree on the zoned extent")
    if np.any((hp == HYDROPERIOD_UNOBSERVED) & (lf != LANDFORM_IN_CHANNEL)):
        raise ValueError("unobserved hydroperiod is only valid for in-channel landform")

    crosstab = np.where(lf == LANDFORM_OUTSIDE, 0, lf * 10 + hp).astype(np.uint8)
    off_channel = lf == LANDFORM_OFF_CHANNEL_RIVERINE
    mask = np.zeros(lf.shape, dtype=np.uint8)
    mask[lf == LANDFORM_IN_CHANNEL] = 1
    mask[off_channel & (hp == HYDROPERIOD_PERSISTENT)] = 2
    mask[off_channel & (hp == HYDROPERIOD_SEASONAL)] = 3
    mask[off_channel & (hp == HYDROPERIOD_MARGINAL)] = 4

    emitted_zones = tuple(sorted(int(z) for z in np.unique(mask) if z != 0))

    return ZoneResult(
        mask=mask,
        emitted_zones=emitted_zones,
        has_zone_1=bool(np.any(mask == 1)),
        source=source,
        mode="riverscape",
        crosstab=crosstab,
        degraded_reasons=tuple(degraded_reasons),
    )


__all__ = [
    "ZoneResult",
    "build_zones",
    "combine_zones",
    "zones_from_riverscape",
    "zones_from_riverscape_with_landform",
    "zones_from_wo_statistics",
]
