"""Two-anchor channel-gap discovery and least-cost bridging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
from affine import Affine
from scipy import ndimage
from shapely.geometry import LineString, Point
from skimage.graph import MCP_Geometric

from hydrofragments.config import BridgeCostWeights


@dataclass(frozen=True)
class Gap:
    gap_id: str
    reach_ids: tuple[str, ...]
    upstream_anchor: tuple[int, int]
    downstream_anchor: tuple[int, int]
    upstream_width_m: float
    downstream_width_m: float
    straight_length_m: float


@dataclass(frozen=True)
class GapSearchResult:
    gaps: tuple[Gap, ...]
    dangling_anchors: tuple[tuple[int, int], ...]
    degraded_reasons: tuple[str, ...]


@dataclass(frozen=True)
class BridgeCostInputs:
    rem: np.ndarray
    trough_depth: np.ndarray
    frequency: np.ndarray
    bare_fraction: np.ndarray
    green_pct: np.ndarray
    npv_pct: np.ndarray
    line_distance_m: np.ndarray
    corridor_width_m: np.ndarray
    corridor_mask: np.ndarray
    domain: np.ndarray
    off_channel_mask: np.ndarray


@dataclass(frozen=True)
class UnbridgedGap:
    gap: Gap
    reason: str


@dataclass(frozen=True)
class BridgeResult:
    bridged_mask: np.ndarray
    channel_bridges: gpd.GeoDataFrame
    unbridged: tuple[UnbridgedGap, ...]
    degraded_reasons: tuple[str, ...]


def _pixel_xy(rc: tuple[int, int], transform: Affine) -> tuple[float, float]:
    row, col = rc
    x, y = transform * (col + 0.5, row + 0.5)
    return float(x), float(y)


def _component_endpoints(mask: np.ndarray) -> list[tuple[int, int]]:
    neighbours = ndimage.convolve(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    rows, cols = np.nonzero(mask & (neighbours <= 2))
    if rows.size == 0:
        rows, cols = np.nonzero(mask)
    return [(int(row), int(col)) for row, col in zip(rows, cols)]


def _downstream_chain(start: str, next_down: Mapping[str, str]) -> tuple[str, ...]:
    chain: list[str] = [start]
    seen = {start}
    current = start
    while next_down.get(current) not in {None, "", "-1", "0"}:
        current = next_down[current]
        if current in seen:
            raise ValueError("drainage NextDownID topology contains a cycle")
        seen.add(current)
        chain.append(current)
    return tuple(chain)


def _geometry_endpoints(geometry) -> tuple[Point, Point] | None:
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type == "LineString":
        parts = [geometry]
    elif geometry.geom_type == "MultiLineString":
        parts = list(geometry.geoms)
    elif geometry.geom_type == "GeometryCollection":
        parts = [
            part for part in geometry.geoms
            if part.geom_type in {"LineString", "MultiLineString"}
            and not part.is_empty
        ]
        flattened = []
        for part in parts:
            flattened.extend(
                list(part.geoms) if part.geom_type == "MultiLineString" else [part]
            )
        parts = flattened
    elif geometry.geom_type == "Point":
        return None
    else:
        raise ValueError(
            f"unsupported reach geometry type: {geometry.geom_type!r}"
        )
    if not parts:
        return None
    return Point(parts[0].coords[0]), Point(parts[-1].coords[-1])


def _direction_sign(
    key: str,
    geometry: Mapping[str, Any],
    next_down: Mapping[str, str],
) -> int | None:
    endpoints = _geometry_endpoints(geometry[key])
    if endpoints is None:
        return None
    start, end = endpoints
    downstream = next_down.get(key)
    if downstream in geometry:
        start_distance = start.distance(geometry[downstream])
        end_distance = end.distance(geometry[downstream])
        if np.isclose(start_distance, end_distance):
            return None
        return 1 if end_distance < start_distance else -1
    upstream = [
        upstream_key
        for upstream_key, target in next_down.items()
        if target == key and upstream_key in geometry
    ]
    if not upstream:
        return None
    start_distance = min(start.distance(geometry[item]) for item in upstream)
    end_distance = min(end.distance(geometry[item]) for item in upstream)
    if np.isclose(start_distance, end_distance):
        return None
    return 1 if start_distance < end_distance else -1


def find_gaps(
    channel: np.ndarray,
    centreline: np.ndarray,
    drainage: gpd.GeoDataFrame,
    reach_labels: np.ndarray,
    reach_keys: Mapping[int, str],
    corridor_widths_m: Mapping[str, float],
    *,
    transform: Affine,
    pixel_m: float,
) -> GapSearchResult:
    channel = np.asarray(channel, bool)
    centreline = np.asarray(centreline, bool)
    reach_labels = np.asarray(reach_labels)
    if channel.ndim != 2 or centreline.shape != channel.shape or reach_labels.shape != channel.shape:
        raise ValueError("channel, centreline and reach_labels must share a 2-D shape")
    required = {"HydroID", "NextDownID"}
    missing = sorted(required - set(drainage.columns))
    if missing:
        raise ValueError(f"drainage missing required columns: {missing}")
    next_down = {
        str(hydro_id): str(downstream)
        for hydro_id, downstream in zip(
            drainage["HydroID"], drainage["NextDownID"]
        )
    }
    geometry = {
        str(hydro_id): geom
        for hydro_id, geom in zip(drainage["HydroID"], drainage.geometry)
    }
    directions = {
        key: _direction_sign(key, geometry, next_down) for key in geometry
    }
    spine_labels, count = ndimage.label(
        channel & centreline, structure=np.ones((3, 3), int)
    )
    if count < 2:
        dangling = tuple(_component_endpoints(spine_labels == 1)) if count else ()
        return GapSearchResult((), dangling, ())

    width_px = ndimage.distance_transform_edt(channel) * pixel_m
    records: list[dict[str, Any]] = []
    degraded: list[str] = []
    for component in range(1, count + 1):
        for rc in _component_endpoints(spine_labels == component):
            label = int(reach_labels[rc])
            key_value = reach_keys.get(label)
            key = None if key_value is None else str(key_value)
            if key is None or key not in corridor_widths_m:
                degraded.append(f"component_{component}_anchor_without_reach")
                continue
            geom = geometry[key]
            if directions.get(key) is None:
                degraded.append(f"reach_{key}_ambiguous_direction")
                continue
            point = Point(_pixel_xy(rc, transform))
            raw_chainage = float(geom.project(point, normalized=True))
            chainage = (
                raw_chainage if directions[key] == 1 else 1.0 - raw_chainage
            )
            records.append(
                {
                    "component": component,
                    "rc": rc,
                    "reach": key,
                    "chainage": chainage,
                    "width": float(width_px[rc]),
                }
            )

    by_component: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_component.setdefault(record["component"], []).append(record)
    pairs: dict[tuple[int, int], tuple[tuple, dict, dict]] = {}
    for left_component, left_records in by_component.items():
        for right_component, right_records in by_component.items():
            if left_component >= right_component:
                continue
            best = None
            for left in left_records:
                chain = _downstream_chain(left["reach"], next_down)
                for right in right_records:
                    if right["reach"] not in chain:
                        reverse_chain = _downstream_chain(right["reach"], next_down)
                        if left["reach"] not in reverse_chain:
                            continue
                        upstream, downstream = right, left
                        used_chain = reverse_chain
                    else:
                        upstream, downstream = left, right
                        used_chain = chain
                    if upstream["reach"] == downstream["reach"] and (
                        upstream["chainage"] >= downstream["chainage"]
                    ):
                        upstream, downstream = downstream, upstream
                    distance = float(
                        np.hypot(
                            upstream["rc"][0] - downstream["rc"][0],
                            upstream["rc"][1] - downstream["rc"][1],
                        )
                        * pixel_m
                    )
                    rank_gap = used_chain.index(downstream["reach"])
                    score = (rank_gap, distance)
                    if best is None or score < best[0]:
                        best = (score, upstream, downstream)
            if best is not None:
                pairs[(left_component, right_component)] = best

    # Each component keeps its nearest downstream neighbour; this removes
    # non-consecutive component pairs while retaining cross-junction gaps.
    selected: dict[int, tuple[tuple, dict, dict]] = {}
    for candidate in pairs.values():
        upstream_component = candidate[1]["component"]
        if upstream_component not in selected or candidate[0] < selected[upstream_component][0]:
            selected[upstream_component] = candidate

    gaps: list[Gap] = []
    anchored_components: set[int] = set()
    for number, candidate in enumerate(
        sorted(selected.values(), key=lambda value: (value[1]["reach"], value[1]["chainage"])),
        start=1,
    ):
        _, upstream, downstream = candidate
        chain = _downstream_chain(upstream["reach"], next_down)
        reach_ids = chain[: chain.index(downstream["reach"]) + 1]
        distance = float(
            np.hypot(
                upstream["rc"][0] - downstream["rc"][0],
                upstream["rc"][1] - downstream["rc"][1],
            )
            * pixel_m
        )
        gaps.append(
            Gap(
                gap_id=f"gap-{number}",
                reach_ids=reach_ids,
                upstream_anchor=upstream["rc"],
                downstream_anchor=downstream["rc"],
                upstream_width_m=upstream["width"],
                downstream_width_m=downstream["width"],
                straight_length_m=distance,
            )
        )
        anchored_components.update(
            (upstream["component"], downstream["component"])
        )
    dangling = tuple(
        record["rc"]
        for component, component_records in sorted(by_component.items())
        if component not in anchored_components
        for record in component_records[:1]
    )
    return GapSearchResult(
        tuple(gaps), dangling, tuple(dict.fromkeys(degraded))
    )


def _unit_interval(values: np.ndarray, *, nan_value: float = 1.0) -> np.ndarray:
    return np.clip(
        np.nan_to_num(np.asarray(values, float), nan=nan_value), 0.0, 1.0
    )


def build_cost_surface(
    inputs: BridgeCostInputs,
    *,
    weights: BridgeCostWeights,
    bridge_rem_max_m: float,
    trough_depth_m: float,
) -> np.ndarray:
    shape = np.shape(inputs.rem)
    fields = (
        inputs.trough_depth,
        inputs.frequency,
        inputs.bare_fraction,
        inputs.green_pct,
        inputs.npv_pct,
        inputs.line_distance_m,
        inputs.corridor_width_m,
        inputs.corridor_mask,
        inputs.domain,
        inputs.off_channel_mask,
    )
    if len(shape) != 2 or any(np.shape(field) != shape for field in fields):
        raise ValueError("all bridge cost inputs must share a 2-D shape")
    if bridge_rem_max_m <= 0 or trough_depth_m <= 0:
        raise ValueError("bridge terrain thresholds must be positive")
    rem = np.asarray(inputs.rem, float)
    terrain_rem = _unit_interval(np.maximum(rem, 0) / bridge_rem_max_m)
    terrain_trough = 1.0 - _unit_interval(
        np.maximum(inputs.trough_depth, 0) / trough_depth_m,
        nan_value=0.0,
    )
    terms = (
        np.minimum(terrain_rem, terrain_trough),
        1.0 - _unit_interval(
            np.asarray(inputs.frequency) / 100.0, nan_value=0.0
        ),
        1.0 - _unit_interval(inputs.bare_fraction, nan_value=0.0),
        1.0 - _unit_interval(np.asarray(inputs.green_pct) / 100.0, nan_value=0.0),
        1.0 - _unit_interval(np.asarray(inputs.npv_pct) / 100.0, nan_value=0.0),
        _unit_interval(
            np.divide(
                inputs.line_distance_m,
                inputs.corridor_width_m,
                out=np.ones(shape, float),
                where=np.asarray(inputs.corridor_width_m) > 0,
            )
        ),
    )
    values = np.array(
        [
            weights.terrain,
            weights.water,
            weights.bare,
            weights.green,
            weights.npv,
            weights.line_distance,
        ],
        float,
    )
    if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
        raise ValueError("bridge weights must be finite, non-negative, and not all zero")
    cost = sum(weight * term for weight, term in zip(values, terms)) / values.sum()
    blocked = (
        ~np.asarray(inputs.corridor_mask, bool)
        | (np.isfinite(rem) & (rem > bridge_rem_max_m))
        | (np.asarray(inputs.domain, bool) & np.asarray(inputs.off_channel_mask, bool))
    )
    return np.where(blocked, np.inf, cost + 1e-6)


def _path_length(path: Sequence[tuple[int, int]], pixel_m: float) -> float:
    return float(
        sum(
            np.hypot(right[0] - left[0], right[1] - left[1]) * pixel_m
            for left, right in zip(path, path[1:])
        )
    )


def _paint_disc(mask: np.ndarray, rc: tuple[int, int], radius_px: float) -> None:
    row, col = rc
    radius = max(float(radius_px), 0.5)
    row0, row1 = max(0, int(np.floor(row - radius))), min(mask.shape[0], int(np.ceil(row + radius)) + 1)
    col0, col1 = max(0, int(np.floor(col - radius))), min(mask.shape[1], int(np.ceil(col + radius)) + 1)
    rows, cols = np.ogrid[row0:row1, col0:col1]
    mask[row0:row1, col0:col1] |= (
        (rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2
    )


def _empty_bridge_frame(crs: Any) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "gap_id": [],
            "reach_ids": [],
            "length_m": [],
            "cumulative_cost": [],
            "cost_per_m": [],
            "upstream_width_m": [],
            "downstream_width_m": [],
            "bridge_confidence": [],
            "gap_cause": [],
            "geometry": [],
        },
        geometry="geometry",
        crs=crs,
    )


def bridge_gaps(
    gaps: Sequence[Gap],
    observed_channel: np.ndarray,
    cost_inputs: BridgeCostInputs,
    *,
    transform: Affine,
    crs: Any,
    pixel_m: float,
    weights: BridgeCostWeights,
    bridge_max_length_m: float,
    bridge_max_cost_per_m: float,
    bridge_rem_max_m: float,
    trough_depth_m: float,
    riparian_green_pct: float,
    narrow_width_px: int,
) -> BridgeResult:
    observed = np.asarray(observed_channel, bool)
    if observed.shape != np.shape(cost_inputs.rem):
        raise ValueError("observed_channel and cost inputs must share shape")
    base_cost = build_cost_surface(
        cost_inputs,
        weights=weights,
        bridge_rem_max_m=bridge_rem_max_m,
        trough_depth_m=trough_depth_m,
    )
    bridged = np.zeros(observed.shape, bool)
    rows: list[dict[str, Any]] = []
    unbridged: list[UnbridgedGap] = []
    protected = np.asarray(cost_inputs.domain, bool) & ~observed

    for gap in gaps:
        if gap.straight_length_m > bridge_max_length_m:
            unbridged.append(UnbridgedGap(gap, "straight_length_exceeds_max"))
            continue
        cost = base_cost.copy()
        cost[protected] = np.inf
        cost[gap.upstream_anchor] = base_cost[gap.upstream_anchor]
        cost[gap.downstream_anchor] = base_cost[gap.downstream_anchor]
        mcp = MCP_Geometric(
            cost, fully_connected=True, sampling=(pixel_m, pixel_m)
        )
        cumulative, _ = mcp.find_costs(
            [gap.upstream_anchor], [gap.downstream_anchor]
        )
        total_cost = float(cumulative[gap.downstream_anchor])
        if not np.isfinite(total_cost):
            unbridged.append(UnbridgedGap(gap, "no_finite_path"))
            continue
        path = [tuple(point) for point in mcp.traceback(gap.downstream_anchor)]
        length_m = _path_length(path, pixel_m)
        if length_m > bridge_max_length_m:
            unbridged.append(UnbridgedGap(gap, "path_length_exceeds_max"))
            continue
        cost_per_m = total_cost / max(length_m, pixel_m)
        if cost_per_m > bridge_max_cost_per_m:
            unbridged.append(UnbridgedGap(gap, "cost_per_m_exceeds_max"))
            continue

        path_distance = np.concatenate(
            [[0.0], np.cumsum([
                np.hypot(b[0] - a[0], b[1] - a[1]) * pixel_m
                for a, b in zip(path, path[1:])
            ])]
        )
        widths = np.interp(
            path_distance,
            [0.0, max(length_m, pixel_m)],
            [gap.upstream_width_m, gap.downstream_width_m],
        )
        painted = np.zeros(observed.shape, bool)
        for point, width in zip(path, widths):
            _paint_disc(painted, point, width / pixel_m)
        painted &= np.asarray(cost_inputs.corridor_mask, bool)
        painted &= ~observed
        painted &= ~protected
        bridged |= painted

        green_values = np.array(
            [cost_inputs.green_pct[point] for point in path], float
        )
        finite_green = green_values[np.isfinite(green_values)]
        median_green = float(np.median(finite_green)) if finite_green.size else float("-inf")
        if median_green >= riparian_green_pct:
            cause = "vegetated"
        elif (
            gap.upstream_width_m <= narrow_width_px * pixel_m
            and gap.downstream_width_m <= narrow_width_px * pixel_m
        ):
            cause = "narrow"
        else:
            cause = "unobserved"
        confidence = int(
            length_m <= bridge_max_length_m / 2.0
            and cost_per_m <= bridge_max_cost_per_m / 2.0
        ) + 1
        rows.append(
            {
                "gap_id": gap.gap_id,
                "reach_ids": ",".join(gap.reach_ids),
                "length_m": length_m,
                "cumulative_cost": total_cost,
                "cost_per_m": cost_per_m,
                "upstream_width_m": gap.upstream_width_m,
                "downstream_width_m": gap.downstream_width_m,
                "bridge_confidence": confidence,
                "gap_cause": cause,
                "geometry": LineString(
                    [_pixel_xy(point, transform) for point in path]
                ),
            }
        )

    frame = (
        gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
        if rows
        else _empty_bridge_frame(crs)
    )
    return BridgeResult(bridged, frame, tuple(unbridged), ())


__all__ = [
    "BridgeCostInputs",
    "BridgeResult",
    "Gap",
    "GapSearchResult",
    "UnbridgedGap",
    "bridge_gaps",
    "build_cost_surface",
    "find_gaps",
]
