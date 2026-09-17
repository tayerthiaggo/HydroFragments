from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString

from hydrofragments.config import BridgeCostWeights
from hydrofragments.riverscape.bridging import (
    BridgeCostInputs,
    Gap,
    _gap_routing_window,
    _path_length,
    bridge_gaps,
    build_cost_surface,
    find_gaps,
)

TRANSFORM = Affine(30, 0, 0, 0, -30, 330)
WEIGHTS = BridgeCostWeights()


def _inputs(shape=(11, 15), *, ridge=False, protected=False, green=0.0):
    rem = np.zeros(shape, np.float32)
    if ridge:
        rem[:, 7] = 10.0
    domain = np.zeros(shape, bool)
    off_channel = np.zeros(shape, bool)
    if protected:
        domain[5, 7] = True
        off_channel[5, 7] = True
    return BridgeCostInputs(
        rem=rem,
        trough_depth=np.ones(shape, np.float32),
        frequency=np.zeros(shape, np.float32),
        bare_fraction=np.ones(shape, np.float32),
        green_pct=np.full(shape, green, np.float32),
        npv_pct=np.zeros(shape, np.float32),
        line_distance_m=np.zeros(shape, np.float32),
        corridor_width_m=np.full(shape, 300.0, np.float32),
        corridor_mask=np.ones(shape, bool),
        domain=domain,
        off_channel_mask=off_channel,
    )


def _gap(length_px=10, width_m=30.0):
    return Gap(
        gap_id="gap-1",
        reach_ids=("1",),
        upstream_anchor=(5, 2),
        downstream_anchor=(5, 2 + length_px),
        upstream_width_m=width_m,
        downstream_width_m=width_m,
        straight_length_m=length_px * 30.0,
    )


def _bridge(gap, inputs, **overrides):
    observed = np.zeros(inputs.rem.shape, bool)
    observed[gap.upstream_anchor] = True
    observed[gap.downstream_anchor] = True
    values = {
        "bridge_max_length_m": 2000.0,
        "bridge_max_cost_per_m": 1.1,
        "bridge_rem_max_m": 5.0,
        "trough_depth_m": 0.5,
        "riparian_green_pct": 40.0,
        "narrow_width_px": 2,
    }
    values.update(overrides)
    return bridge_gaps(
        [gap],
        observed,
        inputs,
        transform=TRANSFORM,
        crs="EPSG:3577",
        pixel_m=30.0,
        weights=WEIGHTS,
        **values,
    )


def test_vegetated_300m_gap_is_bridged_and_labelled() -> None:
    result = _bridge(_gap(), _inputs(green=80.0))
    assert result.bridged_mask[5, 3:12].all()
    assert result.unbridged == ()
    row = result.channel_bridges.iloc[0]
    assert row.gap_cause == "vegetated"
    assert row.length_m == pytest.approx(300.0)
    assert row.cost_per_m == pytest.approx(0.300001, rel=1e-5)


def test_one_pixel_anchors_label_narrow_gap() -> None:
    result = _bridge(_gap(length_px=5, width_m=30.0), _inputs())
    assert result.channel_bridges.iloc[0].gap_cause == "narrow"


def test_ridge_is_impassable() -> None:
    result = _bridge(_gap(), _inputs(ridge=True))
    assert not result.bridged_mask.any()
    assert result.unbridged[0].reason == "no_finite_path"


def test_gap_over_max_length_is_rejected_without_painting() -> None:
    result = _bridge(
        _gap(), _inputs(), bridge_max_length_m=299.0
    )
    assert not result.bridged_mask.any()
    assert result.unbridged[0].reason == "straight_length_exceeds_max"


def test_bridge_never_overwrites_observed_off_channel_domain() -> None:
    result = _bridge(_gap(), _inputs(protected=True))
    assert not result.bridged_mask[5, 7]


def test_width_is_interpolated_between_anchors() -> None:
    gap = Gap("gap-1", ("1",), (5, 2), (5, 12), 30.0, 90.0, 300.0)
    result = _bridge(gap, _inputs())
    assert result.bridged_mask[:, 11].sum() > result.bridged_mask[:, 3].sum()


def test_gap_routing_window_pads_by_max_length() -> None:
    row_sl, col_sl, local_up, local_down = _gap_routing_window(
        (100, 50),
        (100, 60),
        (500, 500),
        pixel_m=30.0,
        bridge_max_length_m=300.0,
        paint_radius_m=30.0,
    )
    # pad = ceil(300/30) + ceil(30/30) + 1 = 12
    assert row_sl == slice(88, 113)
    assert col_sl == slice(38, 73)
    assert local_up == (12, 12)
    assert local_down == (12, 22)


def test_bridge_on_large_raster_uses_local_window() -> None:
    """MCP must stay local: full-basin routing on a 2k grid would be very slow."""
    import time

    shape = (2000, 2000)
    inputs = _inputs(shape)
    gap = Gap(
        "gap-large",
        ("1",),
        (1000, 990),
        (1000, 1010),
        30.0,
        30.0,
        600.0,
    )
    started = time.perf_counter()
    result = _bridge(gap, inputs, bridge_max_length_m=2000.0)
    elapsed = time.perf_counter() - started
    assert result.unbridged == ()
    assert result.bridged_mask[1000, 991:1010].all()
    assert elapsed < 5.0


def test_dangling_end_has_no_gap() -> None:
    channel = np.zeros((7, 9), bool)
    centreline = np.zeros_like(channel)
    channel[3, 1:4] = True
    centreline[3, 1:4] = True
    labels = np.ones_like(channel, np.int32)
    drainage = gpd.GeoDataFrame(
        {"HydroID": [1], "NextDownID": [-1]},
        geometry=[LineString([(30, 225), (240, 225)])],
        crs="EPSG:3577",
    )
    result = find_gaps(
        channel,
        centreline,
        drainage,
        labels,
        {1: "1"},
        {"1": 300.0},
        transform=TRANSFORM,
        pixel_m=30.0,
    )
    assert result.gaps == ()
    assert result.dangling_anchors


def test_two_segments_on_one_reach_form_one_gap() -> None:
    channel = np.zeros((7, 12), bool)
    centreline = np.zeros_like(channel)
    channel[3, 1:4] = channel[3, 8:11] = True
    centreline |= channel
    labels = np.ones_like(channel, np.int32)
    drainage = gpd.GeoDataFrame(
        {"HydroID": [1, 2], "NextDownID": [2, -1]},
        geometry=[
            LineString([(30, 225), (330, 225)]),
            LineString([(330, 225), (390, 225)]),
        ],
        crs="EPSG:3577",
    )
    result = find_gaps(
        channel, centreline, drainage, labels, {1: "1"}, {"1": 300.0},
        transform=TRANSFORM, pixel_m=30.0,
    )
    assert len(result.gaps) == 1
    assert result.gaps[0].reach_ids == ("1",)


def test_components_across_reach_junction_form_one_gap() -> None:
    channel = np.zeros((7, 12), bool)
    centreline = np.zeros_like(channel)
    channel[3, 1:4] = channel[3, 8:11] = True
    centreline |= channel
    labels = np.zeros_like(channel, np.int32)
    labels[:, :6] = 1
    labels[:, 6:] = 2
    drainage = gpd.GeoDataFrame(
        {"HydroID": [10, 20], "NextDownID": [20, -1]},
        geometry=[
            LineString([(30, 225), (180, 225)]),
            LineString([(180, 225), (330, 225)]),
        ],
        crs="EPSG:3577",
    )
    result = find_gaps(
        channel, centreline, drainage, labels, {1: "10", 2: "20"},
        {"10": 300.0, "20": 300.0},
        transform=TRANSFORM, pixel_m=30.0,
    )
    assert len(result.gaps) == 1
    assert result.gaps[0].reach_ids == ("10", "20")


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("water", [0.0, 100.0]),
        ("bare", [0.0, 1.0]),
        ("green", [0.0, 100.0]),
        ("npv", [0.0, 100.0]),
        ("line_distance", [300.0, 0.0]),
    ],
)
def test_channel_like_value_lowers_each_cost_term(field, values) -> None:
    inputs = _inputs(shape=(1, 2))
    input_values = dict(inputs.__dict__)
    input_name = {
        "water": "frequency",
        "bare": "bare_fraction",
        "green": "green_pct",
        "npv": "npv_pct",
        "line_distance": "line_distance_m",
    }[field]
    input_values[input_name] = np.array([values], np.float32)
    isolated = {
        "terrain": 0.0,
        "water": 0.0,
        "bare": 0.0,
        "green": 0.0,
        "npv": 0.0,
        "line_distance": 0.0,
    }
    isolated[field] = 1.0
    cost = build_cost_surface(
        BridgeCostInputs(**input_values),
        weights=BridgeCostWeights(**isolated),
        bridge_rem_max_m=5.0,
        trough_depth_m=0.5,
    )
    assert cost[0, 1] < cost[0, 0]


def test_deep_trough_lowers_terrain_cost() -> None:
    inputs = _inputs(shape=(1, 2))
    values = dict(inputs.__dict__)
    values["rem"] = np.full((1, 2), 5.0, np.float32)
    values["trough_depth"] = np.array([[0.0, 0.5]], np.float32)
    terrain_only = BridgeCostWeights(1, 0, 0, 0, 0, 0)
    cost = build_cost_surface(
        BridgeCostInputs(**values),
        weights=terrain_only,
        bridge_rem_max_m=5.0,
        trough_depth_m=0.5,
    )
    assert cost[0, 1] < cost[0, 0]


def test_diagonal_path_length_uses_metric_hypotenuse() -> None:
    assert _path_length([(0, 0), (1, 1), (2, 2)], 30.0) == pytest.approx(
        2 * np.sqrt(2) * 30.0
    )


def test_three_segments_create_only_two_consecutive_gaps() -> None:
    channel = np.zeros((7, 15), bool)
    centreline = np.zeros_like(channel)
    channel[3, 1:3] = channel[3, 6:8] = channel[3, 11:13] = True
    centreline |= channel
    labels = np.ones_like(channel, np.int32)
    drainage = gpd.GeoDataFrame(
        {"HydroID": [1, 2], "NextDownID": [2, -1]},
        geometry=[
            LineString([(30, 225), (390, 225)]),
            LineString([(390, 225), (450, 225)]),
        ],
        crs="EPSG:3577",
    )
    result = find_gaps(
        channel, centreline, drainage, labels, {1: "1"}, {"1": 300.0},
        transform=TRANSFORM, pixel_m=30.0,
    )
    assert len(result.gaps) == 2
