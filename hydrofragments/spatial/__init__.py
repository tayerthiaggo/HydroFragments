"""Fixed AOI, drainage topology, zones, windows, and CRS validation."""

from hydrofragments.spatial.connectivity_context import (
    reach_monthly_wet_profile,
    reach_wet_any_month,
)
from hydrofragments.spatial.context import (
    DrainageContractError,
    DrainageTopology,
    SpatialContext,
    create_channel_context,
    create_spatial_context,
    ordered_reach_paths,
    validate_drainage_topology,
)
from hydrofragments.spatial.crs import normalize_spatial_inputs
from hydrofragments.spatial.windows import (
    SpatialWindow,
    create_channel_windows,
    create_drainage_windows,
    create_regular_grid_windows,
)
from hydrofragments.spatial.zones import (
    ZoneResult,
    build_zones,
    zones_from_riverscape,
    zones_from_wo_statistics,
)

__all__ = [
    "DrainageContractError",
    "DrainageTopology",
    "SpatialContext",
    "SpatialWindow",
    "ZoneResult",
    "build_zones",
    "create_channel_context",
    "create_channel_windows",
    "create_drainage_windows",
    "create_regular_grid_windows",
    "create_spatial_context",
    "normalize_spatial_inputs",
    "ordered_reach_paths",
    "reach_monthly_wet_profile",
    "reach_wet_any_month",
    "validate_drainage_topology",
    "zones_from_riverscape",
    "zones_from_wo_statistics",
]
