from __future__ import annotations

import numpy as np

from hydrofragments.riverscape.channel import (
    RULE_ADJACENT_GEOMORPHIC,
    RULE_CONNECTED_BARE_TERRAIN,
    RULE_CONNECTED_WATER,
    classify_channel,
)
from hydrofragments.riverscape.evidence import BareCalibration, RiverscapeEvidence


def _evidence(
    shape,
    *,
    water=(),
    bare=(),
    terrain=(),
    connected=(),
    fallback=(),
) -> RiverscapeEvidence:
    def mask(points):
        result = np.zeros(shape, bool)
        for point in points:
            result[point] = True
        return result

    w, b, t, c, f = map(mask, (water, bare, terrain, connected, fallback))
    confidence = (
        (c | f).astype(np.uint8)
        + w.astype(np.uint8)
        + (b | t).astype(np.uint8)
    )
    return RiverscapeEvidence(
        water_high=w,
        waterbody_riverine=np.zeros(shape, bool),
        bare_stable=b,
        terrain=t,
        connected=c,
        line_fallback=f,
        bare_fraction=b.astype(np.float32),
        bare_valid_years=np.full(shape, 2, np.uint16),
        confidence=confidence,
        calibration=BareCalibration(30.0, None, None, 0, 0, ()),
        degraded_reasons=(),
    )


def _run(evidence, centreline, water_seed, *, factor=3.0):
    labels = np.ones(centreline.shape, np.int32)
    return classify_channel(
        evidence,
        np.ones(centreline.shape, bool),
        centreline,
        water_seed,
        labels,
        {1: "reach-1"},
        pixel_m=30.0,
        width_growth_factor=factor,
        min_channel_confidence=2,
    )


def test_rarely_wet_sand_bed_uses_bare_terrain_rule() -> None:
    shape = (5, 7)
    points = [(2, col) for col in range(1, 6)]
    centreline = np.zeros(shape, bool)
    centreline[2, 1:6] = True
    evidence = _evidence(
        shape, bare=points, terrain=points, connected=points
    )

    result = _run(evidence, centreline, centreline)

    assert result.channel[2, 1:6].all()
    assert np.all(
        result.rule_id[2, 1:6] == RULE_CONNECTED_BARE_TERRAIN
    )


def test_bare_scald_off_trough_and_not_adjacent_is_rejected() -> None:
    shape = (7, 9)
    centreline = np.zeros(shape, bool)
    centreline[3, 1:5] = True
    water = [(3, col) for col in range(1, 5)]
    # Inside the 90 m width cap, but two diagonal cells away from accepted
    # channel: B alone must not seed rule 2 or bypass rule 3 adjacency.
    scald = [(1, 6), (1, 7)]
    evidence = _evidence(
        shape,
        water=water,
        bare=scald,
        connected=water + scald,
    )

    result = _run(evidence, centreline, centreline)

    assert result.channel[3, 1:5].all()
    assert not result.channel[1, 6:8].any()


def test_first_match_keeps_connected_water_rule() -> None:
    shape = (3, 3)
    point = (1, 1)
    centreline = np.zeros(shape, bool)
    centreline[point] = True
    evidence = _evidence(
        shape, water=[point], bare=[point], terrain=[point], connected=[point]
    )

    result = _run(evidence, centreline, centreline)

    assert result.rule_id[point] == RULE_CONNECTED_WATER


def test_adjacent_geomorphic_growth_iterates_to_convergence() -> None:
    shape = (3, 7)
    centreline = np.zeros(shape, bool)
    centreline[1, 1:6] = True
    water = [(1, 1)]
    growth = [(1, col) for col in range(2, 6)]
    evidence = _evidence(
        shape, water=water, terrain=growth, connected=water + growth
    )

    result = _run(evidence, centreline, centreline)

    assert result.channel[1, 1:6].all()
    assert np.all(result.rule_id[1, 2:6] == RULE_ADJACENT_GEOMORPHIC)


def test_width_cap_stops_attached_billabong_growth() -> None:
    shape = (11, 11)
    centreline = np.zeros(shape, bool)
    centreline[5, 1:10] = True
    water_seed = centreline.copy()  # one-pixel seed half-width = 30 m
    all_points = [(row, col) for row in range(11) for col in range(11)]
    evidence = _evidence(
        shape,
        water=[(5, col) for col in range(1, 10)],
        terrain=all_points,
        connected=all_points,
    )

    result = _run(evidence, centreline, water_seed, factor=2.0)

    assert result.channel[5, 1:10].all()
    assert result.channel[4, 5]  # 30 m — inside cap
    assert result.channel[3, 5]  # 60 m — inclusive boundary
    assert not result.channel[2, 5]  # 90 m — beyond cap
    assert not result.channel[0, 5]
    assert result.seed_half_width_m["reach-1"] == 30.0


def test_missing_seed_width_degrades_and_rejects_reach() -> None:
    shape = (3, 3)
    centreline = np.zeros(shape, bool)
    evidence = _evidence(shape, connected=[(1, 1)], water=[(1, 1)])
    result = _run(evidence, centreline, np.zeros(shape, bool))
    assert not result.channel.any()
    assert result.degraded_reasons == ("reach_reach-1_missing_seed_width",)
