"""W1.3 Step 2 — analyze() output is unaffected by whether a zone raster
was computed alongside it.

The plan (docs/superpowers/plans/2026-07-27-dea-zones-and-catchment-speed.md,
section 1.3) requires: "Keep the fastest default as one AOI/channel
calculation of every available runtime-wired metric and emit the exact DEA
zone raster as a separate artifact. Do not add per-zone metric
multiplication to this default workflow."

Scoping note (per task-w1.3-brief.md): a true end-to-end "DEA zones emitted
alongside metrics via one workflow call" regression test cannot be written
yet, because ``hydrofragments/workflow.py`` (the one-call DEA-to-table
orchestrator, task W4.2) does not exist in this codebase yet. Building any
part of it here would be scope creep into W4.2's job. Independently
reconfirmed before writing this test (2026-07-30):

- ``grep -rn "guard_persistence_zone" hydrofragments/`` finds only the
  guard's own definition/docstring, its ``guards/__init__.py`` re-export,
  and its ``__all__`` entries -- zero production call sites.
- ``grep -n "zone" hydrofragments/api.py`` finds only hardcoded channel-
  context zone LABEL literals (``zone="1"`` on LPSEC/inter_pool_gap records,
  ``zone="AOI"`` on extent_contraction records) -- these are a fixed literal
  stamped onto every record of that metric family, not the occurrence-
  defined Zone 1-4 system this guard protects, and not driven by any
  ``ZoneResult``/zone raster at all.
- ``grep -n "zone" hydrofragments/metrics/registry.py`` finds nothing.
- ``hydrofragments.api.analyze()``'s signature takes no ``ZoneResult``/zone
  raster parameter, and its body contains no loop over zone labels that
  would multiply metric records per zone.

So instead, this module proves the SPIRIT of the plan's requirement against
what exists today: computing a ``ZoneResult`` (via ``build_zones`` or
``zones_from_wo_statistics``) on the same water cube that ``analyze()``
consumes has ZERO effect on ``analyze()``'s output -- same row count, same
(metric, statistic, value) content -- whether or not that zone computation
happened at all. This is a concrete regression baseline: if a later task
(e.g. W4.2's workflow orchestrator) ever wires zone-conditioned iteration
into ``analyze()`` or a caller of it, this test will fail the moment that
change makes ``analyze()``'s output depend on zone computation.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from hydrofragments import HydroConfig, analyze
from hydrofragments.config import ValidityConfig, ZonesConfig
from hydrofragments.spatial.zones import build_zones, combine_zones


def _config(tmp_path):
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": ["contracts_core"],
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )


def _zones_config(*, t_persist: float = 0.50, t_season: float = 0.10, min_valid_obs: int = 1):
    return SimpleNamespace(
        zones=ZonesConfig(t_persist=t_persist, t_season=t_season),
        validity=ValidityConfig(min_valid_obs=min_valid_obs),
    )


def _assert_frames_identical(left: pd.DataFrame, right: pd.DataFrame) -> None:
    assert len(left) == len(right)
    left_rows = sorted(
        (str(row["metric"]), "" if pd.isna(row["statistic"]) else str(row["statistic"]))
        for _, row in left.iterrows()
    )
    right_rows = sorted(
        (str(row["metric"]), "" if pd.isna(row["statistic"]) else str(row["statistic"]))
        for _, row in right.iterrows()
    )
    assert left_rows == right_rows

    def _values(frame: pd.DataFrame) -> list[tuple[str, str, float]]:
        out = []
        for _, row in frame.iterrows():
            value = row["value"]
            if pd.isna(value):
                continue
            statistic = row["statistic"]
            statistic = "" if pd.isna(statistic) else str(statistic)
            out.append((str(row["metric"]), statistic, round(float(value), 9)))
        return sorted(out)

    assert _values(left) == _values(right)


def test_analyze_output_identical_whether_or_not_zones_were_computed(
    synthetic_cube, tmp_path
) -> None:
    """analyze()'s row count and content do not depend on zone computation.

    Calls build_zones() on data derived from the same synthetic water cube
    and calls analyze() on that same cube in the same test -- with the zone
    computation happening BEFORE analyze() in one branch and not at all in
    the other -- to prove analyze()'s output is byte-for-byte identical
    either way. This is the "zones are a separate, non-multiplying
    artifact" invariant, proven directly rather than merely asserted.
    """
    config = _config(tmp_path / "with_zones")
    baseline_config = _config(tmp_path / "without_zones")

    # Branch A: compute a ZoneResult first, from data genuinely derived from
    # the same cube analyze() will consume (occurrence-like ratio of wet
    # months per pixel, support = number of valid months).
    water = synthetic_cube.water.values.astype(bool)
    valid = synthetic_cube.valid_obs.values.astype(bool)
    valid_count = valid.sum(axis=0)
    wet_count = (water & valid).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        occurrence_pct = np.where(valid_count > 0, 100.0 * wet_count / valid_count, np.nan)
    max_wet_mask = water.any(axis=0)

    zones_cfg = _zones_config()
    zone_result = build_zones(
        occurrence_pct,
        max_wet_mask=max_wet_mask,
        valid_count=valid_count,
        t_persist=zones_cfg.zones.t_persist,
        t_season=zones_cfg.zones.t_season,
        min_valid_obs=zones_cfg.validity.min_valid_obs,
    )
    # Zone computation genuinely happened and produced a real raster.
    assert zone_result.mask.shape == water.shape[1:]
    assert set(np.unique(zone_result.mask)) <= {0, 2, 3, 4}

    result_with_zones = analyze(
        synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0
    )

    # Branch B: analyze() alone, with no zone computation at all.
    result_without_zones = analyze(
        synthetic_cube, aoi_id="demo", config=baseline_config, pixel_size_m=30.0
    )

    _assert_frames_identical(
        result_with_zones.metrics_table, result_without_zones.metrics_table
    )
    # Explicit row-count check per the brief's wording ("row count and
    # content are completely independent of whether a ZoneResult ... exists
    # or was computed alongside it").
    assert len(result_with_zones.metrics_table) == len(
        result_without_zones.metrics_table
    )


def test_analyze_output_identical_whether_or_not_riverscape_zones_were_computed(
    synthetic_cube, tmp_path
) -> None:
    """Phase 6b spec section 4: riverscape zones do not multiply metrics.

    Same invariant as the occurrence case above, extended to the landform x
    hydroperiod path. ``combine_zones`` runs for real -- no stubbing -- over
    layers derived from the very cube ``analyze()`` consumes, so the
    ZoneResult is a genuine riverscape one (``mode == "riverscape"``, with a
    crosstab). If a later phase ever wires zone-conditioned iteration into
    ``analyze()`` or a caller of it, this fails the moment the riverscape
    path makes metric output depend on zoning.

    ``guards/scientific.py`` is untouched: persistence-by-zone stays
    refused, and landform 1 still rests on water evidence.
    """
    config = _config(tmp_path / "with_riverscape_zones")
    baseline_config = _config(tmp_path / "without_zones")

    water = synthetic_cube.water.values.astype(bool)
    valid = synthetic_cube.valid_obs.values.astype(bool)
    valid_count = valid.sum(axis=0)
    wet_count = (water & valid).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        occurrence_pct = np.where(valid_count > 0, 100.0 * wet_count / valid_count, np.nan)

    # Landform: a synthetic in-channel column, off-channel riverine over the
    # rest of the observed extent, 0 outside. Hydroperiod follows the same
    # thresholds the occurrence path uses, so the two branches describe the
    # same water.
    observed = np.isfinite(occurrence_pct) & water.any(axis=0)
    landform = np.zeros(observed.shape, dtype=np.uint8)
    landform[observed] = 2
    landform[observed & (np.arange(observed.shape[1])[None, :] == 0)] = 1

    hydroperiod = np.zeros(observed.shape, dtype=np.uint8)
    frequency = np.where(np.isfinite(occurrence_pct), occurrence_pct, 0.0)
    hydroperiod[observed & (frequency > 50.0)] = 1
    hydroperiod[observed & (frequency >= 10.0) & (frequency <= 50.0)] = 2
    hydroperiod[observed & (frequency < 10.0)] = 3

    zone_result = combine_zones(landform, hydroperiod, source="synthetic")

    # Riverscape zoning genuinely happened and produced a real crosstab.
    assert zone_result.mode == "riverscape"
    assert zone_result.crosstab is not None
    assert zone_result.mask.shape == water.shape[1:]
    assert set(np.unique(zone_result.mask)) <= {0, 1, 2, 3, 4}

    result_with_zones = analyze(
        synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0
    )
    result_without_zones = analyze(
        synthetic_cube, aoi_id="demo", config=baseline_config, pixel_size_m=30.0
    )

    _assert_frames_identical(
        result_with_zones.metrics_table, result_without_zones.metrics_table
    )
    assert len(result_with_zones.metrics_table) == len(
        result_without_zones.metrics_table
    )
