"""``build_zoning_section``: the manifest's ``zoning`` object (spec 6b section 3.3).

Three shapes, one function: full riverscape from a bundle, the thin
occurrence subset from a bare ``ZoneResult``, and a reasons-only record when
``auto`` returned no zones at all. Provenance keys are copied verbatim from
``LandformResult.provenance`` -- this module asserts the literal key names,
because Phase 7 and Phase 8 both read them.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("rioxarray")

import geopandas as gpd
from shapely.geometry import LineString

from hydrofragments.output.manifest import _hash_array, build_zoning_section
from hydrofragments.output.riverscape_export import RiverscapeExportBundle
from hydrofragments.riverscape.riverine import FloodplainEnvelope
from hydrofragments.riverscape.pipeline import LandformResult
from hydrofragments.spatial.zones import build_zones, combine_zones

PIXEL_M = 30.0
CRS = "EPSG:3577"

_PROVENANCE = {
    "channel_ruleset_version": "c-1.2.3",
    "waterbody_ruleset_version": "w-2.0.0",
    "riverine_ruleset_version": "r-3.1.0",
    "dem_product": "ga_srtm_dem1sv1_0",
    "dem_band": "elevation",
    "fc_product": "ga_ls_fc_pc_cyear_3",
    "fc_bands": ("bs_pc_50", "pv_pc_50", "npv_pc_50"),
    "waterbodies_source": "dea_waterbodies",
    "years": (2019, 2021),
    "bridge_enabled": True,
    "bare_threshold_pct": 30.0,
    "envelope": {"a": 2.0, "b": 0.0, "n_bins": 1},
    "pixel_counts": {
        "domain": 6,
        "water_seed": 4,
        "observed_channel": 2,
        "bridged_channel": 1,
        "in_channel": 3,
        "off_channel_riverine": 4,
        "non_riverine": 2,
    },
    "bridge_counts_by_cause": {"vegetated": 1},
    "unbridged_counts_by_reason": {},
    "reach_counts": {"total": 1, "line_fallback": 0, "multithread": 0},
}


def _bridges(lengths=(120.0, 80.0)):
    return gpd.GeoDataFrame(
        {
            "gap_id": list(range(len(lengths))),
            "reach_ids": ["1,2"] * len(lengths),
            "length_m": list(lengths),
            "cumulative_cost": [1.0] * len(lengths),
            "cost_per_m": [0.01] * len(lengths),
            "upstream_width_m": [30.0] * len(lengths),
            "downstream_width_m": [30.0] * len(lengths),
            "bridge_confidence": [2] * len(lengths),
            "gap_cause": ["vegetated"] * len(lengths),
        },
        geometry=[LineString([(0.0, 0.0), (length, 0.0)]) for length in lengths],
        crs=CRS,
    )


def _landform_result(landform, *, bridged, channel_bridges, degraded_reasons=()):
    landform = np.asarray(landform, dtype=np.uint8)
    zeros = np.zeros(landform.shape, dtype=np.uint8)
    return LandformResult(
        landform=landform,
        channel_source=zeros.copy(),
        channel_confidence=np.full(landform.shape, 255, dtype=np.uint8),
        rule_id=zeros.copy(),
        rem=np.zeros(landform.shape, dtype=np.float32),
        bridged_mask=np.asarray(bridged, dtype=bool),
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
        provenance=dict(_PROVENANCE),
        degraded_reasons=tuple(degraded_reasons),
    )


def _riverscape_bundle(*, channel_bridges=None, degraded_reasons=("bare_threshold_fallback",)):
    landform = np.array(
        [
            [1, 2, 2, 0],
            [2, 2, 3, 0],
            [1, 3, 1, 0],
        ],
        dtype=np.uint8,
    )
    hydroperiod = np.array(
        [
            [1, 1, 2, 0],
            [2, 3, 1, 0],
            [4, 2, 1, 0],
        ],
        dtype=np.uint8,
    )
    bridged = np.zeros(landform.shape, dtype=bool)
    bridged[2, 0] = True
    zone_result = combine_zones(
        landform,
        hydroperiod,
        source="ga_ls_wo_fq_myear_3",
        degraded_reasons=degraded_reasons,
    )
    result = _landform_result(
        landform,
        bridged=bridged,
        channel_bridges=_bridges() if channel_bridges is None else channel_bridges,
        degraded_reasons=degraded_reasons,
    )
    return RiverscapeExportBundle.from_zoning(zone_result, result), zone_result


def _occurrence_zone_result():
    frequency = np.array(
        [
            [80.0, 30.0, 5.0, 0.0],
            [80.0, 30.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    return build_zones(
        frequency,
        max_wet_mask=frequency > 0,
        valid_count=np.full(frequency.shape, 40),
    )


# --------------------------------------------------------------------------
# Shape 1: full riverscape
# --------------------------------------------------------------------------


def test_riverscape_section_copies_every_provenance_key_verbatim() -> None:
    """MUTANT: renaming any provenance key (e.g. ``dem_product`` ->
    ``dem``) breaks Phase 7/8's contract and fails here (spec 6b section 3.3)."""
    bundle, _zone_result = _riverscape_bundle()

    section = build_zoning_section(
        bundle.zone_result,
        bundle,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=(),
    )

    assert section["mode"] == "riverscape"
    assert section["ruleset_versions"] == {
        "channel": "c-1.2.3",
        "waterbody": "w-2.0.0",
        "riverine": "r-3.1.0",
    }
    assert section["sources"] == {
        "dem_product": "ga_srtm_dem1sv1_0",
        "dem_band": "elevation",
        "fc_product": "ga_ls_fc_pc_cyear_3",
        "fc_bands": ["bs_pc_50", "pv_pc_50", "npv_pc_50"],
        "waterbodies_source": "dea_waterbodies",
        "years": [2019, 2021],
        "zone_source": "ga_ls_wo_fq_myear_3",
    }
    assert section["envelope"] == {"a": 2.0, "b": 0.0, "n_bins": 1}
    assert section["pixel_counts"] == _PROVENANCE["pixel_counts"]
    assert section["bridge_counts_by_cause"] == {"vegetated": 1}
    assert section["unbridged_counts_by_reason"] == {}
    assert section["reach_counts"] == {"total": 1, "line_fallback": 0, "multithread": 0}
    assert section["bridge_enabled"] is True
    assert section["bare_threshold_pct"] == 30.0


def test_riverscape_area_and_length_fields_use_the_documented_formulas() -> None:
    """bridged_length_m = sum(channel_bridges.length_m);
    bridged_area_m2 = count(bridged_mask) * pixel_m**2;
    non_riverine_area_m2 = pixel_counts.non_riverine * pixel_m**2."""
    bundle, _zone_result = _riverscape_bundle()

    section = build_zoning_section(
        bundle.zone_result,
        bundle,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=(),
    )

    assert section["bridged_length_m"] == pytest.approx(200.0)
    assert section["bridged_area_m2"] == pytest.approx(900.0)
    assert section["non_riverine_area_m2"] == pytest.approx(1800.0)


def test_riverscape_empty_bridges_report_zero_length() -> None:
    bundle, _zone_result = _riverscape_bundle(channel_bridges=_bridges(lengths=()))

    section = build_zoning_section(
        bundle.zone_result,
        bundle,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=(),
    )

    assert section["bridged_length_m"] == 0.0


def test_riverscape_domain_fields_use_provenance_and_the_shared_digest() -> None:
    """MUTANT: inventing a second hashing algorithm instead of reusing
    ``_hash_array`` (the helper behind ``dea_provenance.zone_mask_digest``)
    fails here (spec 6b section 3.3, decision L8)."""
    bundle, zone_result = _riverscape_bundle()

    section = build_zoning_section(
        zone_result, bundle, pixel_m=PIXEL_M, configured_mode="auto", workflow_reasons=()
    )

    assert section["domain_pixel_count"] == 6
    assert section["domain_digest"] == _hash_array(zone_result.mask)


def test_riverscape_degraded_reasons_come_from_the_zone_result() -> None:
    bundle, _zone_result = _riverscape_bundle(
        degraded_reasons=("bare_threshold_fallback", "envelope_single_bin")
    )

    section = build_zoning_section(
        bundle.zone_result,
        bundle,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=("ignored_because_zone_result_wins",),
    )

    assert section["degraded_reasons"] == [
        "bare_threshold_fallback",
        "envelope_single_bin",
    ]


def test_riverscape_section_is_json_serializable_with_allow_nan_false() -> None:
    """``manifest._json_bytes`` uses ``allow_nan=False``; tuples and numpy
    scalars must already be normalized by this builder."""
    bundle, _zone_result = _riverscape_bundle()

    section = build_zoning_section(
        bundle.zone_result,
        bundle,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=(),
    )

    json.dumps(section, allow_nan=False, sort_keys=True)


# --------------------------------------------------------------------------
# Shape 2: thin occurrence subset
# --------------------------------------------------------------------------


def test_occurrence_section_is_the_thin_subset_and_omits_riverscape_keys() -> None:
    """MUTANT: emitting riverscape keys as empty defaults instead of
    omitting them makes an occurrence run look like a degraded riverscape
    run (spec 6b section 3.3: 'prefer omit for absent science')."""
    zone_result = _occurrence_zone_result()

    section = build_zoning_section(
        zone_result, None, pixel_m=PIXEL_M, configured_mode="off", workflow_reasons=()
    )

    assert section["mode"] == "occurrence"
    assert section["zone_source"] == "occurrence"
    assert section["degraded_reasons"] == []
    assert section["domain_pixel_count"] == int(np.count_nonzero(zone_result.mask))
    assert section["domain_digest"] == _hash_array(zone_result.mask)
    for omitted in (
        "ruleset_versions",
        "sources",
        "envelope",
        "pixel_counts",
        "bridge_counts_by_cause",
        "unbridged_counts_by_reason",
        "reach_counts",
        "bridged_length_m",
        "bridged_area_m2",
        "non_riverine_area_m2",
    ):
        assert omitted not in section


def test_occurrence_fallback_reasons_ride_on_the_zone_result() -> None:
    from dataclasses import replace

    zone_result = replace(
        _occurrence_zone_result(),
        degraded_reasons=("riverscape_source_unavailable",),
    )

    section = build_zoning_section(
        zone_result,
        None,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=("riverscape_source_unavailable",),
    )

    assert section["mode"] == "occurrence"
    assert section["degraded_reasons"] == ["riverscape_source_unavailable"]


# --------------------------------------------------------------------------
# Shape 3: no ZoneResult at all
# --------------------------------------------------------------------------


def test_no_zone_result_records_configured_mode_and_workflow_reasons() -> None:
    section = build_zoning_section(
        None,
        None,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=("riverscape_no_stats",),
    )

    assert section == {
        "mode": "auto",
        "degraded_reasons": ["riverscape_no_stats"],
    }


def test_no_zone_result_never_fabricates_a_domain_digest() -> None:
    """MUTANT: hashing an empty/zero array to fill ``domain_digest`` would
    publish a digest for a domain that was never computed."""
    section = build_zoning_section(
        None,
        None,
        pixel_m=PIXEL_M,
        configured_mode="required",
        workflow_reasons=(),
    )

    assert "domain_digest" not in section
    assert "domain_pixel_count" not in section
    assert section["degraded_reasons"] == []


# --------------------------------------------------------------------------
# Wiring into build_run_manifest
# --------------------------------------------------------------------------


def test_build_run_manifest_emits_zoning_when_supplied(tmp_path) -> None:
    from hydrofragments.config import HydroConfig
    from hydrofragments.output.manifest import build_run_manifest

    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
        }
    )
    manifest = build_run_manifest(
        config,
        run_id="run-1",
        package_version="0.0.0",
        git_sha="deadbeef",
        input_fingerprint={"kind": "generic_binary"},
        planned_backend="cpu",
        actual_backend_by_stage={},
        zoning={"mode": "riverscape", "degraded_reasons": []},
    )

    assert manifest["zoning"] == {"mode": "riverscape", "degraded_reasons": []}


def test_build_run_manifest_omits_zoning_when_absent() -> None:
    from hydrofragments.config import HydroConfig
    from hydrofragments.output.manifest import build_run_manifest

    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
        }
    )
    manifest = build_run_manifest(
        config,
        run_id="run-1",
        package_version="0.0.0",
        git_sha="deadbeef",
        input_fingerprint={"kind": "generic_binary"},
        planned_backend="cpu",
        actual_backend_by_stage={},
    )

    assert "zoning" not in manifest
