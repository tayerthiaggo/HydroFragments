"""The slim bundle that carries riverscape science from zoning to export.

Phase 6a's ``zones_from_riverscape`` discarded its ``LandformResult`` the
moment ``combine_zones`` had run. Phase 6b needs those arrays again to write
``riverscape_evidence`` rasters, the ``channel_bridges`` layer, and the
manifest's ``zoning`` section -- and re-deriving them would mean a second
DEM / Fractional Cover / Waterbodies load (spec 6b section 1.1). This module
owns the one type that carries them across the
``workflow`` -> ``finalize`` seam.

It lives under ``hydrofragments/output/`` rather than
``hydrofragments/spatial/`` on purpose: ``spatial`` stays zoning-only, and
the riverscape -> spatial import boundary (spec section 1.3) is untouched --
``riverscape`` imports nothing from here, and ``spatial.zones`` does not
import this module either. Only ``workflow`` and ``output.finalize`` do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydrofragments.riverscape.pipeline import LandformResult
from hydrofragments.spatial.zones import ZoneResult


@dataclass(frozen=True)
class RiverscapeExportBundle:
    """Landform + hydroperiod + zones for one successful riverscape run.

    Produced only on a genuinely successful riverscape zoning path -- never
    for ``riverscape.mode="off"``, and never when ``auto`` falls back to
    occurrence zoning (spec section 3.1).
    """

    landform: LandformResult
    hydroperiod_classes: np.ndarray
    zone_result: ZoneResult

    def __post_init__(self) -> None:
        if self.zone_result.mode != "riverscape":
            raise ValueError(
                "RiverscapeExportBundle requires a riverscape ZoneResult"
            )
        if np.shape(self.hydroperiod_classes) != np.shape(self.landform.landform):
            raise ValueError(
                "hydroperiod_classes must share the landform shape"
            )

    @classmethod
    def from_zoning(
        cls,
        zone_result: ZoneResult,
        landform: LandformResult,
    ) -> "RiverscapeExportBundle":
        """Build a bundle, recovering hydroperiod from the crosstab.

        ``combine_zones`` writes ``crosstab = landform * 10 + hydroperiod``
        with ``0`` outside the zoned extent, and validates that the two
        layers agree on that extent
        (``hydrofragments/spatial/zones.py``). Inside, hydroperiod is one of
        ``{1, 2, 3, 4}``, so ``crosstab % 10`` returns it exactly; outside,
        ``crosstab`` is ``0`` and so is hydroperiod. That identity is why
        this never calls ``classify_hydroperiod`` a second time (spec
        section 3.1) -- re-classifying would also silently drop code 4 on
        bridged pixels, which only exists because the first classification
        was handed an ``unobserved_mask``.
        """
        if zone_result.mode != "riverscape":
            raise ValueError(
                "RiverscapeExportBundle requires a riverscape ZoneResult"
            )
        if zone_result.crosstab is None:
            raise ValueError(
                "RiverscapeExportBundle requires a ZoneResult crosstab"
            )
        crosstab = np.asarray(zone_result.crosstab, dtype=np.uint16)
        hydroperiod_classes = (crosstab % 10).astype(np.uint8)
        return cls(
            landform=landform,
            hydroperiod_classes=hydroperiod_classes,
            zone_result=zone_result,
        )


__all__ = ["RiverscapeExportBundle"]
