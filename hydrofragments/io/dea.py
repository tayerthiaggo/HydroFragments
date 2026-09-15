"""Adapter over hydroseason's native DEA Water Observation Statistics loader.

hydroseason owns STAC search, unsigned-COG/GDAL configuration, and the
Dask-backed load itself (``hydroseason.open_wo_statistics``). This module's
only job is to convert that raw ``xr.Dataset`` into the frozen
:class:`WoStatistics` value HydroFragments' zoning code will consume, and to
enforce the one check that legitimately belongs on this side of the
repository boundary: refusing a geographic CRS for an area metric
(``guard_area_metric_crs``, spec §8 guard 8), because hydroseason has no
``pyproj`` dependency and must never import from HydroFragments (dependency
direction is one-way: HydroFragments -> hydroseason).

This module does NOT build zones and does NOT reduce statistics into a
planning mask (``WetPlanningFootprint`` / ``build_wet_planning_footprint``)
-- both are later tasks. It only adapts one loaded Dataset into one
dataclass.

``hydrofragments.io.riverscape_sources`` is a deliberate exception to this
module's "hydroseason owns STAC access" boundary: riverscape zoning's DEM,
Fractional Cover, and DEA Waterbodies inputs have no hydroseason equivalent,
so that module owns direct ``pystac-client``/``odc.stac`` access for them.
This module's own DEA WO Statistics loader is unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import hydroseason

from hydrofragments.guards.scientific import guard_area_metric_crs


@dataclass(frozen=True)
class WoStatistics:
    """Frozen DEA Water Observation Statistics, adapted for zoning use.

    ``frequency`` is the 0-100 float32 wet-observation frequency, lazily
    derived by hydroseason's loader as ``100 * count_wet / count_clear`` and
    passed through here unchanged (still Dask-backed; never materialized by
    this adapter). ``count_wet``/``count_clear`` are the raw per-pixel
    observation counts the frequency was derived from. ``product``,
    ``version`` (the installed ``hydroseason`` package version that produced
    this data, matching the convention already used for run-manifest
    provenance in ``hydrofragments.output.manifest`` and
    ``hydrofragments.temporal.hydroyear``), ``crs``, ``time_span``, and
    ``provenance`` are recorded so a later task can attach them to the run
    manifest without re-deriving anything.
    """

    frequency: Any
    count_wet: Any
    count_clear: Any
    product: str
    version: str | None
    crs: str
    time_span: str | None
    provenance: Mapping[str, Any]


def open_wo_statistics_for_zoning(
    aoi: Any,
    *,
    product: str = "ga_ls_wo_fq_myear_3",
    stac_url: str = "https://explorer.sandbox.dea.ga.gov.au/stac",
    resolution: float = 30.0,
    crs: str = "EPSG:3577",
    chunks: Mapping[str, int] | None = None,
) -> WoStatistics:
    """Load native DEA WO Statistics via hydroseason and adapt for zoning.

    Delegates entirely to ``hydroseason.open_wo_statistics`` for STAC search,
    unsigned-COG reads, and the Dask-backed load -- this function adds no
    acquisition logic of its own. ``resolution`` selects hydroseason's native
    output grid explicitly (default DEA's native 30 m Albers grid); it is
    passed straight through and must not be reinterpreted as a
    scientific-resolution knob here.

    Raises whatever ``hydroseason.open_wo_statistics`` raises on a failed or
    empty search (propagated unchanged, so callers using this as a zoning
    source can fall back to their local-cube zoning path). Additionally
    raises :class:`~hydrofragments.guards.scientific.ScientificGuardError`
    if the returned dataset's CRS is geographic -- this is the one check the
    brief requires this adapter (not hydroseason) to own, since only this
    side of the boundary can import ``guard_area_metric_crs``.
    """
    dataset = hydroseason.open_wo_statistics(
        aoi,
        product=product,
        stac_url=stac_url,
        resolution=resolution,
        crs=crs,
        chunks=chunks,
    )

    resolved_crs = dataset.rio.crs
    crs_text = resolved_crs.to_string() if resolved_crs is not None else str(crs)
    guard_area_metric_crs(crs_text, area_method="projected")

    frequency = dataset["frequency"].astype("float32")

    provenance = dict(dataset.attrs.get("provenance", {}))
    time_span = provenance.get("time_span")

    return WoStatistics(
        frequency=frequency,
        count_wet=dataset["count_wet"],
        count_clear=dataset["count_clear"],
        product=provenance.get("product", product),
        version=getattr(hydroseason, "__version__", None),
        crs=crs_text,
        time_span=time_span,
        provenance=provenance,
    )


__all__ = ["WoStatistics", "open_wo_statistics_for_zoning"]
