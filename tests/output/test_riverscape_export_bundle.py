"""``RiverscapeExportBundle``: the slim retention type Phase 6b exports from.

The bundle carries no science of its own. Its whole job is to hold the
``LandformResult`` that Phase 6a discarded, alongside the exact hydroperiod
array ``combine_zones`` consumed -- recovered from the crosstab rather than
re-classified (spec 6b section 3.1).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("rioxarray")

from hydrofragments.output.riverscape_export import RiverscapeExportBundle
from hydrofragments.riverscape.riverine import FloodplainEnvelope
from hydrofragments.riverscape.pipeline import LandformResult
from hydrofragments.spatial.zones import build_zones, combine_zones

SHAPE = (3, 4)


def _landform_result(landform, *, bridged=None, provenance=None, channel_bridges=None):
    """Compact LandformResult builder.

    Same shape as ``tests/spatial/test_zones_from_riverscape.py::_landform_result``
    (that module is not importable from here -- ``tests/spatial`` has no
    ``__init__.py`` -- so the pattern is repeated, not shared).
    """
    landform = np.asarray(landform, dtype=np.uint8)
    zeros = np.zeros(landform.shape, dtype=np.uint8)
    return LandformResult(
        landform=landform,
        channel_source=zeros.copy(),
        channel_confidence=np.full(landform.shape, 255, dtype=np.uint8),
        rule_id=zeros.copy(),
        rem=np.zeros(landform.shape, dtype=np.float32),
        bridged_mask=(
            np.zeros(landform.shape, dtype=bool)
            if bridged is None
            else np.asarray(bridged, dtype=bool)
        ),
        channel_bridges=channel_bridges,
        unbridged=(),
        envelope=FloodplainEnvelope(
            a=2.0,
            b=0.0,
            n_bins=1,
            bin_log_a_mid=(3.7,),
            bin_rem_quantile=(2.0,),
            bin_counts=(90,),
            degraded_reasons=(),
        ),
        provenance=dict(provenance or {"years": (2019, 2021)}),
        degraded_reasons=(),
    )


def _riverscape_pair():
    landform = np.array(
        [
            [1, 2, 2, 0],
            [2, 2, 2, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    hydroperiod = np.array(
        [
            [1, 1, 2, 0],
            [1, 2, 3, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    zone_result = combine_zones(landform, hydroperiod, source="ga_ls_wo_fq_myear_3")
    return zone_result, hydroperiod, _landform_result(landform)


def test_from_zoning_recovers_the_exact_hydroperiod_array_from_the_crosstab() -> None:
    """MUTANT: deriving hydroperiod any other way (re-running
    classify_hydroperiod, or ``crosstab // 10``) breaks this equality.
    Code 4 on the bridged in-channel pixel is the discriminating case: it
    only exists because ``combine_zones`` was handed an unobserved mask.
    """
    zone_result, hydroperiod, landform = _riverscape_pair()

    bundle = RiverscapeExportBundle.from_zoning(zone_result, landform)

    assert bundle.hydroperiod_classes.dtype == np.uint8
    np.testing.assert_array_equal(bundle.hydroperiod_classes, hydroperiod)
    assert int(bundle.hydroperiod_classes[2, 0]) == 4


def test_bundle_holds_the_landform_and_zone_result_by_identity() -> None:
    zone_result, _hydroperiod, landform = _riverscape_pair()

    bundle = RiverscapeExportBundle.from_zoning(zone_result, landform)

    assert bundle.zone_result is zone_result
    assert bundle.landform is landform


def test_bundle_is_frozen() -> None:
    zone_result, _hydroperiod, landform = _riverscape_pair()
    bundle = RiverscapeExportBundle.from_zoning(zone_result, landform)

    with pytest.raises(Exception):
        bundle.zone_result = zone_result  # type: ignore[misc]


def test_occurrence_zone_result_is_rejected() -> None:
    """MUTANT: accepting an occurrence ZoneResult would let an occurrence run
    write riverscape evidence rasters with no landform behind them."""
    occurrence = build_zones(
        np.full(SHAPE, 80.0),
        max_wet_mask=np.ones(SHAPE, dtype=bool),
        valid_count=np.full(SHAPE, 40),
    )
    assert occurrence.mode == "occurrence"

    with pytest.raises(ValueError, match="riverscape"):
        RiverscapeExportBundle.from_zoning(
            occurrence, _landform_result(np.zeros(SHAPE, dtype=np.uint8))
        )


def test_shape_mismatch_between_landform_and_hydroperiod_is_rejected() -> None:
    zone_result, _hydroperiod, _landform = _riverscape_pair()

    with pytest.raises(ValueError, match="shape"):
        RiverscapeExportBundle(
            landform=_landform_result(np.zeros((2, 2), dtype=np.uint8)),
            hydroperiod_classes=np.zeros(SHAPE, dtype=np.uint8),
            zone_result=zone_result,
        )
