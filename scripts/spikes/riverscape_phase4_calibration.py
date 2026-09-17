"""Masked-gap calibration for riverscape Phase 4.

Run from repository root:
    python scripts/spikes/riverscape_phase4_calibration.py

Writes raw measurements to output/spikes/riverscape_phase4_calibration.json
and a concise committed record to
docs/superpowers/specs/2026-09-15-riverscape-phase4-findings.md.
"""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence


def _fix_proj_data_env() -> None:
    """Clear inherited PROJ_LIB/PROJ_DATA and point pyproj at rasterio's proj db."""
    os.environ.pop("PROJ_LIB", None)
    os.environ.pop("PROJ_DATA", None)
    try:
        import rasterio

        proj_data = Path(rasterio.__file__).parent / "proj_data"
        if proj_data.exists():
            os.environ["PROJ_DATA"] = str(proj_data)
    except Exception:
        pass


_fix_proj_data_env()
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage

from hydrofragments.config import BridgeCostWeights, RiverscapeConfig
from hydrofragments.io.dea import open_wo_statistics_for_zoning
from hydrofragments.io.riverscape_sources import (
    load_dem,
    load_fc_percentiles,
    load_waterbodies,
)
from hydrofragments.riverscape.bridging import (
    BridgeCostInputs,
    Gap,
    _gap_routing_window,
    bridge_gaps,
)
from hydrofragments.riverscape.centreline import build_centreline
from hydrofragments.riverscape.channel import classify_channel
from hydrofragments.riverscape.corridor import measure_corridor_widths
from hydrofragments.riverscape.domain import wet_domain
from hydrofragments.riverscape.evidence import build_evidence
from hydrofragments.riverscape.terrain import build_rem
from hydrofragments.riverscape.waterbodies import classify_waterbodies

REPO = Path(__file__).resolve().parents[2]
AOI_PATH = REPO / "data" / "fitzroy_basin_aoi.geojson"
DRAINAGE_PATH = REPO / "data" / "fitzroy_basin_drainage.gpkg"
JSON_PATH = REPO / "output" / "spikes" / "riverscape_phase4_calibration.json"
FINDINGS_PATH = (
    REPO
    / "docs"
    / "superpowers"
    / "specs"
    / "2026-09-15-riverscape-phase4-findings.md"
)
BASELINE_NAME = "baseline"
SAMPLE_COUNT = 100
MIN_SPAN_M = 300.0
MAX_SPAN_M = 1800.0


def calibration_candidates() -> OrderedDict[str, dict[str, float]]:
    baseline = {
        "terrain": 1.0,
        "water": 1.0,
        "bare": 0.5,
        "green": 1.0,
        "npv": 0.0,
        "line_distance": 0.5,
    }
    candidates = OrderedDict([(BASELINE_NAME, baseline)])
    for key in ("terrain", "water", "bare", "green", "line_distance"):
        value = dict(baseline)
        value[key] = 0.0
        candidates[f"without_{key}"] = value
    npv = dict(baseline)
    npv["npv"] = 0.5
    candidates["npv_enabled"] = npv
    terrain = dict(baseline)
    terrain["terrain"] = 2.0
    candidates["terrain_heavy"] = terrain
    water = dict(baseline)
    water["water"] = 0.5
    candidates["water_light"] = water
    line = dict(baseline)
    line["line_distance"] = 0.25
    candidates["line_light"] = line
    return candidates


def path_f1(routed: np.ndarray, truth: np.ndarray) -> float:
    routed = np.asarray(routed, bool)
    truth = np.asarray(truth, bool)
    structure = np.ones((3, 3), bool)
    routed_near = ndimage.binary_dilation(routed, structure=structure)
    truth_near = ndimage.binary_dilation(truth, structure=structure)
    precision = (
        float((routed & truth_near).sum()) / float(routed.sum())
        if routed.any()
        else 0.0
    )
    recall = (
        float((truth & routed_near).sum()) / float(truth.sum())
        if truth.any()
        else 0.0
    )
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def choose_candidate(
    scores: Mapping[str, float],
    routable: Mapping[str, int],
    total_cases: int,
) -> str:
    baseline_score = float(scores[BASELINE_NAME])
    winner = BASELINE_NAME
    winner_score = baseline_score
    for name in calibration_candidates():
        if name == BASELINE_NAME or name not in scores:
            continue
        if routable.get(name, 0) < int(np.ceil(0.8 * total_cases)):
            continue
        score = float(scores[name])
        if score >= baseline_score + 0.02 and score > winner_score:
            winner, winner_score = name, score
    return winner


def choose_cost_cap(cost_per_m: Sequence[float], f1: Sequence[float]) -> float:
    cost = np.asarray(cost_per_m, float)
    quality = np.asarray(f1, float)
    usable = cost[np.isfinite(cost) & (quality >= 0.8)]
    return float(np.percentile(usable, 95)) if usable.size >= 80 else 1.0


def candidate_score(
    f1_values: Sequence[float], crossing_values: Sequence[float]
) -> float:
    if not f1_values:
        return -2.0
    return float(np.median(f1_values) - 2.0 * np.mean(crossing_values))


def _rasterize_reaches(drainage, shape, transform, widths):
    labels = np.zeros(shape, np.int32)
    corridor = np.zeros(shape, bool)
    width_raster = np.zeros(shape, np.float32)
    key_map: dict[int, str] = {}
    for label, (hydro_id, geometry) in enumerate(
        zip(drainage["HydroID"], drainage.geometry), start=1
    ):
        key = str(hydro_id)
        key_map[label] = key
        if geometry is None or geometry.is_empty:
            continue
        mask = rasterize(
            [(geometry.buffer(widths[key]), 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)
        unclaimed = mask & (labels == 0)
        labels[unclaimed] = label
        width_raster[unclaimed] = widths[key]
        corridor |= mask
    return labels, key_map, corridor, width_raster


def _ordered_simple_path(component: np.ndarray) -> list[tuple[int, int]]:
    points = {tuple(int(value) for value in rc) for rc in np.argwhere(component)}
    if not points:
        return []
    neighbours = {}
    for point in points:
        row, col = point
        neighbours[point] = sorted(
            candidate
            for candidate in points
            if candidate != point
            and abs(candidate[0] - row) <= 1
            and abs(candidate[1] - col) <= 1
        )
    endpoints = sorted(point for point, adjacent in neighbours.items() if len(adjacent) == 1)
    if len(endpoints) != 2 or any(len(adjacent) > 2 for adjacent in neighbours.values()):
        return []  # reject loops/branches; calibration truth must be unambiguous
    ordered = [endpoints[0]]
    previous = None
    current = endpoints[0]
    while current != endpoints[1]:
        choices = [point for point in neighbours[current] if point != previous]
        if len(choices) != 1:
            return []
        previous, current = current, choices[0]
        ordered.append(current)
    return ordered


def _linestring_pixel_path(
    geometry, *, transform: Affine, shape: tuple[int, int], pixel_m: float
) -> list[tuple[int, int]]:
    """Densify a reach geometry into an ordered unique pixel path."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "MultiLineString":
        parts = list(geometry.geoms)
        if not parts:
            return []
        geometry = max(parts, key=lambda part: part.length)
    if geometry.geom_type != "LineString":
        return []
    if geometry.length < pixel_m:
        coords = list(geometry.coords)
    else:
        distances = np.arange(0.0, geometry.length + pixel_m * 0.5, pixel_m * 0.5)
        coords = [geometry.interpolate(float(distance)).coords[0] for distance in distances]
    path: list[tuple[int, int]] = []
    rows, cols = shape
    for x, y in coords:
        col, row = ~transform * (x, y)
        row_i, col_i = int(round(row)), int(round(col))
        if not (0 <= row_i < rows and 0 <= col_i < cols):
            continue
        point = (row_i, col_i)
        if not path or path[-1] != point:
            path.append(point)
    return path


def _sample_spans(channel, centreline, reach_labels, drainage, pixel_m, transform):
    rng = np.random.default_rng(20260915)
    min_pixels = int(np.ceil(MIN_SPAN_M / pixel_m)) + 1
    max_pixels = int(np.floor(MAX_SPAN_M / pixel_m)) + 1
    area_by_label = {
        index: float(area)
        for index, area in enumerate(drainage["UpstrDArea"], start=1)
    }
    quantiles = np.quantile(list(area_by_label.values()), np.linspace(0, 1, 11))
    anchor_width = ndimage.distance_transform_edt(channel) * pixel_m
    cases = []

    def _append_from_path(label: int, path: list[tuple[int, int]], tag: str) -> bool:
        if len(path) < min_pixels:
            return False
        span_pixels = min(len(path), max_pixels)
        start_index = int(rng.integers(0, len(path) - span_pixels + 1))
        points = np.asarray(path[start_index : start_index + span_pixels], dtype=int)
        start = tuple(int(value) for value in points[0])
        end = tuple(int(value) for value in points[-1])
        cases.append(
            {
                "label": label,
                "truth_points": points,
                "gap": Gap(
                    gap_id=f"{tag}-{len(cases)}",
                    reach_ids=(str(drainage.iloc[label - 1].HydroID),),
                    upstream_anchor=start,
                    downstream_anchor=end,
                    upstream_width_m=float(max(anchor_width[start], pixel_m)),
                    downstream_width_m=float(max(anchor_width[end], pixel_m)),
                    straight_length_m=float(
                        np.hypot(start[0] - end[0], start[1] - end[1]) * pixel_m
                    ),
                ),
            }
        )
        return True

    for stratum in range(10):
        labels = [
            label for label, area in area_by_label.items()
            if quantiles[stratum] <= area <= quantiles[stratum + 1]
        ]
        rng.shuffle(labels)
        stratum_hits = 0
        for label in labels:
            geometry = drainage.geometry.iloc[label - 1]
            path = _linestring_pixel_path(
                geometry,
                transform=transform,
                shape=channel.shape,
                pixel_m=pixel_m,
            )
            if not _append_from_path(label, path, f"heldout-{stratum}"):
                continue
            stratum_hits += 1
            if stratum_hits >= 10 or len(cases) >= SAMPLE_COUNT:
                break
        if len(cases) >= SAMPLE_COUNT:
            break

    if len(cases) < SAMPLE_COUNT:
        remaining = [
            label
            for label in area_by_label
            if all(case["label"] != label for case in cases)
        ]
        rng.shuffle(remaining)
        for label in remaining:
            geometry = drainage.geometry.iloc[label - 1]
            path = _linestring_pixel_path(
                geometry,
                transform=transform,
                shape=channel.shape,
                pixel_m=pixel_m,
            )
            if not _append_from_path(label, path, "heldout-extra"):
                continue
            if len(cases) >= SAMPLE_COUNT:
                break
    return cases[:SAMPLE_COUNT]


def _connected_section_drainage(
    drainage: gpd.GeoDataFrame,
    *,
    rng: np.random.Generator,
    target_reaches: int = 1200,
    min_seed_length_m: float = MIN_SPAN_M,
) -> gpd.GeoDataFrame:
    """Grow a connected AHGF neighbourhood from a long seed reach.

    Stratifying across the whole basin first would buffer into a near-full
    catchment again. A connected walk keeps the section local. Seeding from
    reaches long enough to hold a calibration span avoids tiny headwater
    clusters that cannot yield 100 held-out paths.
    """
    if drainage.empty:
        raise ValueError("drainage is empty")
    by_id = {
        int(hydro_id): int(next_down)
        for hydro_id, next_down in zip(drainage["HydroID"], drainage["NextDownID"])
    }
    upstream_of: dict[int, list[int]] = {}
    for hydro_id, next_down in by_id.items():
        if next_down < 0:
            continue
        upstream_of.setdefault(next_down, []).append(hydro_id)
    lengths = {
        int(hydro_id): float(geometry.length)
        for hydro_id, geometry in zip(drainage["HydroID"], drainage.geometry)
        if geometry is not None and not geometry.is_empty
    }
    seed_pool = [
        hydro_id
        for hydro_id, length_m in lengths.items()
        if length_m >= min_seed_length_m
    ]
    if not seed_pool:
        seed_pool = list(by_id)
    seed = int(seed_pool[int(rng.integers(0, len(seed_pool)))])
    ordered = [seed]
    seen = {seed}
    queue = [seed]
    while queue and len(ordered) < target_reaches:
        current = queue.pop(0)
        neighbours = []
        next_down = by_id.get(current, -1)
        if next_down >= 0 and next_down not in seen:
            neighbours.append(next_down)
        for upstream in upstream_of.get(current, ()):
            if upstream not in seen:
                neighbours.append(upstream)
        neighbours.sort(key=lambda hydro_id: lengths.get(hydro_id, 0.0), reverse=True)
        for neighbour in neighbours:
            if neighbour in seen:
                continue
            seen.add(neighbour)
            ordered.append(neighbour)
            queue.append(neighbour)
            if len(ordered) >= target_reaches:
                break
    return drainage[drainage["HydroID"].isin(ordered)].copy().reset_index(drop=True)


def _sample_reach_ids(
    drainage: gpd.GeoDataFrame,
    *,
    rng: np.random.Generator,
    per_stratum: int = 25,
) -> list[Any]:
    """Stratify HydroIDs by UpstrDArea inside an already-local drainage subset."""
    areas = drainage["UpstrDArea"].to_numpy(dtype=float)
    quantiles = np.quantile(areas, np.linspace(0, 1, 11))
    chosen: list[Any] = []
    for stratum in range(10):
        mask = (areas >= quantiles[stratum]) & (areas <= quantiles[stratum + 1])
        ids = drainage.loc[mask, "HydroID"].to_numpy().copy()
        if ids.size == 0:
            continue
        rng.shuffle(ids)
        chosen.extend(ids[:per_stratum].tolist())
    # Always keep the whole connected section if it is already small.
    if len(drainage) <= max(len(chosen), 1):
        return drainage["HydroID"].tolist()
    return chosen


def _expand_reach_neighbourhood(
    drainage: gpd.GeoDataFrame, hydro_ids: Sequence[Any]
) -> gpd.GeoDataFrame:
    """Keep sampled reaches plus one hop up/downstream for topology continuity."""
    wanted = {int(value) for value in hydro_ids}
    by_id = {
        int(hydro_id): int(next_down)
        for hydro_id, next_down in zip(drainage["HydroID"], drainage["NextDownID"])
    }
    upstream_of: dict[int, list[int]] = {}
    for hydro_id, next_down in by_id.items():
        if next_down < 0:
            continue
        upstream_of.setdefault(next_down, []).append(hydro_id)
    expanded = set(wanted)
    for hydro_id in list(wanted):
        next_down = by_id.get(hydro_id, -1)
        if next_down >= 0:
            expanded.add(next_down)
        expanded.update(upstream_of.get(hydro_id, ()))
    return drainage[drainage["HydroID"].isin(expanded)].copy().reset_index(drop=True)


def _section_aoi(
    drainage: gpd.GeoDataFrame, *, buffer_m: float
) -> gpd.GeoDataFrame:
    """Union buffered reaches into one local AOI polygon (with CRS preserved)."""
    if drainage.empty:
        raise ValueError("section drainage is empty")
    buffered = drainage.geometry.buffer(buffer_m)
    union = buffered.union_all()
    return gpd.GeoDataFrame({"section": [1]}, geometry=[union], crs=drainage.crs)


def _render_findings(payload: Mapping[str, Any]) -> str:
    chosen = payload["selected_candidate"]
    return (
        "# Riverscape Phase 4 Fitzroy Bridge Calibration Findings\n\n"
        f"**Run date:** 2026-09-15  \n"
        f"**Cases:** {payload['case_count']}  \n"
        f"**Runtime seconds:** {payload['runtime_seconds']:.1f}  \n"
        f"**Selected candidate:** {chosen}  \n"
        f"**Selected weights:** {json.dumps(payload['selected_weights'], sort_keys=True)}  \n"
        f"**bridge_max_cost_per_m:** {payload['bridge_max_cost_per_m']:.6g}  \n"
        f"**Threshold fallback:** {payload['cost_cap_fallback']}\n\n"
        "## Method\n\n"
        "Reaches were stratified by UpstrDArea inside a connected local AHGF "
        "neighbourhood, buffered into a section AOI (corridor_max + "
        "bridge_max_length halo) so corridor/channel/REM never run on the full "
        "catchment. One hundred 300-1800 m spans were sampled along AHGF "
        "linestrings on that section. Water/domain evidence was hidden on each "
        "span; routes retained terrain, FC, and line distance, with least-cost "
        "MCP cropped to each gap window. Candidate score was median "
        "one-pixel-tolerance F1 minus twice off-channel crossing rate. Baseline "
        "was retained unless a candidate improved by at least 0.02 with at least "
        "80% routable cases.\n\n"
        "## Candidate results\n\n"
        + "\n".join(
            f"- {name}: score={values['score']:.6f}, "
            f"routable={values['routable']}, "
            f"median_f1={values['median_f1']:.6f}, "
            f"off_channel_rate={values['off_channel_rate']:.6f}"
            for name, values in payload["candidates"].items()
        )
        + "\n"
    )


def main() -> int:
    started = time.perf_counter()

    def _mark(label: str) -> None:
        print(
            f"[phase4-calib] {label}: {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    cfg = RiverscapeConfig()
    rng = np.random.default_rng(20260915)
    basin_aoi = gpd.read_file(AOI_PATH)
    drainage_full = gpd.read_file(DRAINAGE_PATH)
    target_crs = str(basin_aoi.crs) if basin_aoi.crs is not None else "EPSG:3577"
    drainage_full = drainage_full.to_crs(target_crs)
    _mark(f"loaded drainage rows={len(drainage_full)}")

    local = _connected_section_drainage(
        drainage_full, rng=rng, target_reaches=500
    )
    # Keep the whole connected section — do not thin it before buffering.
    drainage = local
    # MCP uses its own gap halo; section buffer only needs corridor headroom.
    buffer_m = float(cfg.corridor_max_m + 100.0)
    aoi = _section_aoi(drainage, buffer_m=buffer_m)
    drainage = drainage[
        drainage.intersects(aoi.geometry.iloc[0])
    ].copy().reset_index(drop=True)
    _mark(
        f"section aoi reaches={len(drainage)} buffer_m={buffer_m:.0f}"
    )

    stats = open_wo_statistics_for_zoning(aoi)
    _mark("loaded WO statistics")
    geobox = stats.frequency.odc.geobox
    drainage = drainage.to_crs(str(geobox.crs))
    transform: Affine = geobox.transform
    pixel_m = abs(float(transform.a))
    frequency = np.asarray(stats.frequency, np.float32)
    domain = wet_domain(stats, min_valid_obs=20)
    water_seed = np.isfinite(frequency) & (frequency >= 100.0 * cfg.f_seed)
    _mark(f"domain ready shape={frequency.shape}")

    corridor_result = measure_corridor_widths(
        drainage,
        water_seed,
        transform=transform,
        pixel_m=pixel_m,
        f_seed=cfg.f_seed,
        corridor_min_m=cfg.corridor_min_m,
        corridor_max_m=cfg.corridor_max_m,
        alignment_quantile=cfg.alignment_quantile,
    )
    _mark("corridor widths")
    reach_labels, reach_keys, corridor_mask, corridor_width_raster = (
        _rasterize_reaches(
            drainage, frequency.shape, transform, corridor_result.widths_m
        )
    )
    _mark("rasterized reaches")
    centreline_result = build_centreline(
        drainage,
        water_seed,
        corridor_result.widths_m,
        transform=transform,
        pixel_m=pixel_m,
    )
    _mark("centreline")
    dem = np.asarray(
        load_dem(geobox, product=cfg.dem_product, band=cfg.dem_band),
        np.float32,
    )
    _mark("DEM")
    terrain = build_rem(
        drainage,
        dem,
        transform=transform,
        pixel_m=pixel_m,
        profile_bin_m=cfg.profile_bin_m,
        profile_percentile=cfg.profile_percentile,
        corridor_widths_m=corridor_result.widths_m,
        rem_k=cfg.rem_k,
        rem_max_distance_m=cfg.rem_max_distance_m,
        trough_radius_m=cfg.trough_radius_m,
    )
    _mark("REM")
    fc = load_fc_percentiles(
        geobox,
        product=cfg.fc_product,
        bands=[cfg.bare_band, cfg.green_band, cfg.npv_band],
        years=(2022, 2023),
    )
    bare_yearly = np.asarray(fc[cfg.bare_band], np.float32)
    green = np.nanmedian(np.asarray(fc[cfg.green_band], np.float32), axis=0)
    npv = np.nanmedian(np.asarray(fc[cfg.npv_band], np.float32), axis=0)
    _mark("FC")
    bounds = tuple(float(value) for value in geobox.extent.boundingbox)
    waterbody_polygons = load_waterbodies(
        bounds, str(geobox.crs), source=cfg.waterbodies_source
    )
    waterbodies = classify_waterbodies(
        waterbody_polygons,
        centreline_result.skeleton,
        transform=transform,
        pixel_m=pixel_m,
    )
    _mark("waterbodies")
    evidence = build_evidence(
        frequency,
        bare_yearly,
        waterbodies.riverine_mask,
        terrain.rem,
        terrain.trough_depth,
        domain,
        centreline_result.skeleton,
        corridor_mask,
        f_chan_high=cfg.f_chan_high,
        bare_threshold_floor_pct=cfg.bare_threshold_floor_pct,
        bare_year_fraction=cfg.bare_year_fraction,
        h_chan_m=cfg.h_chan_m,
        trough_depth_m=cfg.trough_depth_m,
        min_calibration_pixels=cfg.envelope_min_bin_pixels,
    )
    channel = classify_channel(
        evidence,
        domain,
        centreline_result.skeleton,
        water_seed,
        reach_labels,
        reach_keys,
        pixel_m=pixel_m,
        width_growth_factor=cfg.width_growth_factor,
        min_channel_confidence=cfg.min_channel_confidence,
    )
    _mark("channel")
    cases = _sample_spans(
        channel.channel,
        centreline_result.skeleton,
        reach_labels,
        drainage,
        pixel_m,
        transform,
    )
    _mark(f"sampled cases={len(cases)}")
    if len(cases) != SAMPLE_COUNT:
        raise RuntimeError(
            f"calibration requires {SAMPLE_COUNT} cases, found {len(cases)}"
        )

    line_mask = rasterize(
        [(geometry, 1) for geometry in drainage.geometry if geometry is not None and not geometry.is_empty],
        out_shape=frequency.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    line_distance = ndimage.distance_transform_edt(~line_mask) * pixel_m
    candidate_results: dict[str, Any] = {}
    raw_costs: dict[str, list[float]] = {}
    raw_f1: dict[str, list[float]] = {}
    for name, values in calibration_candidates().items():
        f1_values: list[float] = []
        crossing_values: list[float] = []
        costs: list[float] = []
        routable = 0
        for case in cases:
            points = case["truth_points"]
            gap = case["gap"]
            row_sl, col_sl, local_up, local_down = _gap_routing_window(
                gap.upstream_anchor,
                gap.downstream_anchor,
                frequency.shape,
                pixel_m=pixel_m,
                bridge_max_length_m=cfg.bridge_max_length_m,
                paint_radius_m=max(gap.upstream_width_m, gap.downstream_width_m),
            )
            # Shift transform origin to the crop so bridge geometries stay metric.
            window_transform = transform * Affine.translation(col_sl.start, row_sl.start)
            truth_w = np.zeros(
                (row_sl.stop - row_sl.start, col_sl.stop - col_sl.start), bool
            )
            truth_w[points[:, 0] - row_sl.start, points[:, 1] - col_sl.start] = True
            freq_w = np.array(frequency[row_sl, col_sl], copy=True)
            freq_w[truth_w] = 0.0
            observed_w = channel.channel[row_sl, col_sl] & ~truth_w
            local_gap = Gap(
                gap_id=gap.gap_id,
                reach_ids=gap.reach_ids,
                upstream_anchor=local_up,
                downstream_anchor=local_down,
                upstream_width_m=gap.upstream_width_m,
                downstream_width_m=gap.downstream_width_m,
                straight_length_m=gap.straight_length_m,
            )
            inputs = BridgeCostInputs(
                rem=terrain.rem[row_sl, col_sl],
                trough_depth=terrain.trough_depth[row_sl, col_sl],
                frequency=freq_w,
                bare_fraction=evidence.bare_fraction[row_sl, col_sl],
                green_pct=green[row_sl, col_sl],
                npv_pct=npv[row_sl, col_sl],
                line_distance_m=line_distance[row_sl, col_sl],
                corridor_width_m=corridor_width_raster[row_sl, col_sl],
                corridor_mask=corridor_mask[row_sl, col_sl],
                # Calibration disables production hard protection so
                # off-channel preference is measurable in the score below.
                domain=np.zeros_like(truth_w),
                off_channel_mask=np.zeros_like(truth_w),
            )
            result = bridge_gaps(
                [local_gap],
                observed_w,
                inputs,
                transform=window_transform,
                crs=str(geobox.crs),
                pixel_m=pixel_m,
                weights=BridgeCostWeights(**values),
                bridge_max_length_m=cfg.bridge_max_length_m,
                bridge_max_cost_per_m=float("inf"),
                bridge_rem_max_m=cfg.bridge_rem_max_m,
                trough_depth_m=cfg.trough_depth_m,
                riparian_green_pct=cfg.riparian_green_pct,
                narrow_width_px=cfg.narrow_width_px,
            )
            if result.channel_bridges.empty:
                continue
            routable += 1
            f1_values.append(path_f1(result.bridged_mask, truth_w))
            crossing_values.append(
                float(
                    (
                        result.bridged_mask
                        & waterbodies.off_channel_mask[row_sl, col_sl]
                    ).sum()
                )
                / max(int(result.bridged_mask.sum()), 1)
            )
            costs.append(float(result.channel_bridges.iloc[0].cost_per_m))
        median_f1 = float(np.median(f1_values)) if f1_values else 0.0
        crossing = float(np.mean(crossing_values)) if crossing_values else 1.0
        candidate_results[name] = {
            "score": candidate_score(f1_values, crossing_values),
            "routable": routable,
            "median_f1": median_f1,
            "off_channel_rate": crossing,
        }
        raw_costs[name] = costs
        raw_f1[name] = f1_values
        _mark(f"candidate {name} routable={routable}")

    selected = choose_candidate(
        {name: values["score"] for name, values in candidate_results.items()},
        {name: values["routable"] for name, values in candidate_results.items()},
        len(cases),
    )
    cost_cap = choose_cost_cap(raw_costs[selected], raw_f1[selected])
    payload = {
        "case_count": len(cases),
        "candidates": candidate_results,
        "selected_candidate": selected,
        "selected_weights": calibration_candidates()[selected],
        "bridge_max_cost_per_m": cost_cap,
        "cost_cap_fallback": bool(
            cost_cap == 1.0
            and int(np.sum(np.asarray(raw_f1[selected]) >= 0.8)) < 80
        ),
        "runtime_seconds": time.perf_counter() - started,
        "section_reach_count": int(len(drainage)),
        "section_shape": [int(frequency.shape[0]), int(frequency.shape[1])],
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    FINDINGS_PATH.write_text(_render_findings(payload), encoding="utf-8")
    _mark(f"done selected={selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
