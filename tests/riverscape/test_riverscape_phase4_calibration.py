from __future__ import annotations

import numpy as np
import pytest

from scripts.spikes.riverscape_phase4_calibration import (
    BASELINE_NAME,
    calibration_candidates,
    candidate_score,
    choose_candidate,
    choose_cost_cap,
    path_f1,
)


def test_candidate_set_is_fixed_and_baseline_first() -> None:
    candidates = calibration_candidates()
    assert list(candidates)[0] == BASELINE_NAME
    assert len(candidates) == 10
    assert candidates[BASELINE_NAME] == {
        "terrain": 1.0,
        "water": 1.0,
        "bare": 0.5,
        "green": 1.0,
        "npv": 0.0,
        "line_distance": 0.5,
    }


def test_candidate_needs_margin_and_eighty_percent_routable() -> None:
    # Use a real candidate key: choose_candidate only considers names in
    # calibration_candidates(), not arbitrary score keys.
    challenger = "terrain_heavy"
    scores = {BASELINE_NAME: 0.60, challenger: 0.619}
    assert (
        choose_candidate(scores, {BASELINE_NAME: 100, challenger: 100}, 100)
        == BASELINE_NAME
    )
    scores[challenger] = 0.621
    assert (
        choose_candidate(scores, {BASELINE_NAME: 100, challenger: 79}, 100)
        == BASELINE_NAME
    )
    assert (
        choose_candidate(scores, {BASELINE_NAME: 100, challenger: 80}, 100)
        == challenger
    )


def test_cost_cap_is_p95_of_good_routes_or_one() -> None:
    costs = np.arange(1.0, 101.0)
    f1 = np.ones(100)
    assert choose_cost_cap(costs, f1) == np.percentile(costs, 95)
    assert choose_cost_cap(costs[:79], f1[:79]) == 1.0


def test_path_f1_uses_one_pixel_tolerance() -> None:
    truth = np.zeros((5, 7), bool)
    routed = np.zeros_like(truth)
    truth[2, 1:6] = True
    routed[3, 1:6] = True
    assert path_f1(routed, truth) == 1.0


def test_candidate_score_penalizes_off_channel_crossing() -> None:
    clean = candidate_score([0.8, 0.9], [0.0, 0.0])
    crossing = candidate_score([0.8, 0.9], [0.1, 0.1])
    assert clean - crossing == pytest.approx(0.2)


def test_connected_section_stays_local_and_connected() -> None:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString

    from scripts.spikes.riverscape_phase4_calibration import (
        _connected_section_drainage,
        _section_aoi,
    )

    # Linear chain of 20 reaches spaced 1 km apart.
    rows = []
    for index in range(20):
        rows.append(
            {
                "HydroID": index + 1,
                "NextDownID": index + 2 if index < 19 else -1,
                "UpstrDArea": float((index + 1) * 10),
                "geometry": LineString(
                    [(index * 1000.0, 0.0), ((index + 1) * 1000.0, 0.0)]
                ),
            }
        )
    drainage = gpd.GeoDataFrame(rows, crs="EPSG:3577")
    rng = np.random.default_rng(0)
    section = _connected_section_drainage(drainage, rng=rng, target_reaches=8)
    assert 1 <= len(section) <= 8
    aoi = _section_aoi(section, buffer_m=100.0)
    assert aoi.geometry.iloc[0].area > 0
    # Local section must be far smaller than buffering the whole chain.
    full = _section_aoi(drainage, buffer_m=100.0)
    assert aoi.geometry.iloc[0].area < 0.5 * full.geometry.iloc[0].area
