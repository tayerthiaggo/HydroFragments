# Riverscape Zoning — Plan 4: Evidence, Channel Rules, Waterbodies, and Bridging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build Phase 4's per-pixel evidence, DEA Waterbodies roles, observed-channel classifier, least-cost gap bridges, and recorded Fitzroy bridge-cost calibration.

**Architecture:** Five independently reviewable tasks build pure kernels before integration. FC loading preserves yearly observations; evidence and waterbody modules produce immutable results; channel rules fuse three evidence families under per-reach width caps; bridging discovers only two-anchor gaps and routes over normalized cost; a final masked-gap Fitzroy run validates provisional bridge defaults. Phase 6 remains sole owner of build_landform and workflow wiring.

**Tech Stack:** Python 3.10+, numpy, scipy.ndimage, scikit-image 0.22+ (medial_axis and graph.MCP_Geometric), rasterio, shapely 2.x, geopandas, xarray, odc-stac, pytest.

**Spec:** docs/superpowers/specs/2026-09-15-riverscape-phase4-channel-bridging-design.md

## Global Constraints

- Read parent spec docs/superpowers/specs/2026-09-14-riverscape-zoning-design.md sections 3-6, 8, and 10 before starting.
- hydrofragments/riverscape must never import hydrofragments.spatial. Callers validate drainage and build reach labels.
- CorridorResult.widths_m is a radius, never a diameter.
- W and S are one observed-water family. Confidence families are topology C, observed water W|S, and geomorphic B|T.
- FC nodata must become NaN before any temporal reduction. load_fc_percentiles preserves time.
- Bridges add only unobserved pixels. Any domain pixel not already observed channel is protected.
- No silent fallback. Scientific degradation is returned in degraded_reasons; ordinary rejected gaps are returned with exact reasons.
- Phase 4 does not create pipeline.py, modify workflow.py, or export files.
- Every task uses TDD. Passing tests is insufficient: reviewer must apply each named mutant, prove the listed test turns red, restore production code, and rerun tests.
- Test basenames remain unique repo-wide.
- Sonnet implements each task; Opus reviews each task and runs mutation checks. Final whole-plan review uses Opus.

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| hydrofragments/io/riverscape_sources.py | Modify | Preserve nodata-masked yearly FC stacks |
| hydrofragments/riverscape/evidence.py | Create | Bare calibration and evidence bits/confidence |
| tests/riverscape/test_riverscape_evidence.py | Create | Loader/evidence/calibration tests |
| hydrofragments/riverscape/waterbodies.py | Create | Fragment merging and waterbody roles |
| tests/riverscape/test_riverscape_waterbodies.py | Create | Tile-fragment, traversal, billabong tests |
| hydrofragments/riverscape/channel.py | Create | Ordered observed-channel rules and width caps |
| tests/riverscape/test_riverscape_channel.py | Create | Sand-bed, scald, family/order/growth tests |
| hydrofragments/config.py | Modify | BridgeCostWeights, validation, hash schema 1.3.0 |
| tests/contracts/test_config.py | Modify | Weight parse/validation/hash-shape tests |
| hydrofragments/riverscape/bridging.py | Create | Gap search, cost surface, routing, bridge painting |
| tests/riverscape/test_riverscape_bridging.py | Create | Vegetated/narrow/dangling/rejection/protection tests |
| scripts/spikes/riverscape_phase4_calibration.py | Create | Deterministic Fitzroy masked-gap calibration |
| tests/riverscape/test_riverscape_phase4_calibration.py | Create | Candidate selection and threshold decision tests |
| docs/superpowers/specs/2026-09-15-riverscape-phase4-findings.md | Create during calibration run | Committed calibration evidence |
| .superpowers/sdd/progress.md | Modify during execution | Task and review ledger |

---

### Task 1: Preserve yearly FC data and build evidence

**Files:**
- Modify: hydrofragments/io/riverscape_sources.py
- Modify: tests/riverscape/test_riverscape_sources.py
- Create: hydrofragments/riverscape/evidence.py
- Create: tests/riverscape/test_riverscape_evidence.py

**Interfaces:**
- Consumes: Phase 3 loader; CentrelineResult.skeleton; TerrainResult.rem and trough_depth; caller-built domain and corridor masks.
- Produces: BareCalibration, RiverscapeEvidence, calibrate_bare_threshold(...), build_evidence(...). Tasks 3 and 4 consume RiverscapeEvidence fields.

- [ ] **Step 1: Change loader tests to require time preservation**

In tests/riverscape/test_riverscape_sources.py, replace test_load_fc_percentiles_masks_nodata_before_median with:

~~~python
def test_load_fc_percentiles_masks_nodata_and_preserves_time(monkeypatch) -> None:
    monkeypatch.setattr(
        pystac_client.Client, "open",
        staticmethod(lambda url, **kw: _FakeClient(["item-1"])),
    )
    monkeypatch.setattr(odc_stac, "configure_rio", lambda **kw: None)
    monkeypatch.setattr(
        odc_stac,
        "load",
        lambda items, **kw: _fc_dataset_with_time_and_nodata(
            band=kw["bands"][0]
        ),
    )

    result = load_fc_percentiles(
        _FakeGeobox(),
        product="ga_ls_fc_pc_cyear_3",
        bands=["bs_pc_50"],
        years=(2022, 2023),
    )["bs_pc_50"]

    assert result.dims == ("time", "y", "x")
    assert result.shape == (2, 1, 1)
    assert float(result.isel(time=0, y=0, x=0)) == 50.0
    assert np.isnan(float(result.isel(time=1, y=0, x=0)))
~~~

- [ ] **Step 2: Run loader test and verify old temporal median fails**

Run: python -m pytest tests/riverscape/test_riverscape_sources.py::test_load_fc_percentiles_masks_nodata_and_preserves_time -v

Expected: FAIL because current result is two-dimensional.

- [ ] **Step 3: Replace temporal reduction with nodata masking**

In hydrofragments/io/riverscape_sources.py replace _reduce_time with:

~~~python
def _mask_nodata(data: xr.DataArray) -> xr.DataArray:
    """Convert a declared nodata sentinel to NaN without reducing time."""
    nodata = data.attrs.get("nodata")
    if nodata is None:
        nodata = data.odc.nodata
    if nodata is None:
        return data
    return data.where(data != nodata)


def _reduce_dem_time(data: xr.DataArray) -> xr.DataArray:
    """Reduce a repeated DEM time dimension after masking nodata."""
    masked = _mask_nodata(data)
    return masked.median("time", skipna=True) if "time" in masked.dims else masked
~~~

Change load_dem's return to:

~~~python
    return _reduce_dem_time(dataset[band])
~~~

Change load_fc_percentiles' assignment to:

~~~python
        result[band] = _mask_nodata(dataset[band])
~~~

Update its docstring to state that the time dimension is preserved and scientific reductions belong to evidence.py.

- [ ] **Step 4: Add failing evidence tests**

Create tests/riverscape/test_riverscape_evidence.py:

~~~python
from __future__ import annotations

import numpy as np
import pytest

from hydrofragments.riverscape.evidence import (
    build_evidence,
    calibrate_bare_threshold,
)


def test_calibration_uses_midpoint_when_channel_is_barer() -> None:
    yearly = np.array(
        [
            [[60, 60, 20, 20], [60, 60, 20, 20]],
            [[70, 70, 30, 30], [70, 70, 30, 30]],
        ],
        dtype=float,
    )
    candidate = np.zeros((2, 4), bool)
    candidate[:, :2] = True
    background = ~candidate

    result = calibrate_bare_threshold(
        yearly, candidate, background, floor_pct=30.0, min_pixels=4
    )

    assert result.candidate_median_pct == 65.0
    assert result.background_median_pct == 25.0
    assert result.threshold_pct == 45.0
    assert result.degraded_reasons == ()


def test_calibration_falls_back_when_contrast_reverses() -> None:
    yearly = np.array([[[10, 10, 50, 50]], [[20, 20, 60, 60]]], float)
    candidate = np.array([[True, True, False, False]])
    background = ~candidate

    result = calibrate_bare_threshold(
        yearly, candidate, background, floor_pct=30.0, min_pixels=2
    )

    assert result.threshold_pct == 30.0
    assert result.degraded_reasons == ("bare_threshold_fallback",)


def test_b_requires_fraction_boundary_and_two_valid_years() -> None:
    shape = (1, 3)
    bare = np.array(
        [
            [[30.0, 40.0, 30.0]],
            [[30.0, np.nan, 20.0]],
            [[20.0, np.nan, 30.0]],
            [[20.0, np.nan, 20.0]],
            [[30.0, np.nan, 30.0]],
        ]
    )
    result = build_evidence(
        frequency=np.full(shape, 20.0),
        bare_yearly=bare,
        riverine_waterbody_mask=np.zeros(shape, bool),
        rem=np.zeros(shape),
        trough_depth=np.zeros(shape),
        domain=np.ones(shape, bool),
        centreline=np.ones(shape, bool),
        corridor_mask=np.ones(shape, bool),
        f_chan_high=0.1,
        bare_threshold_floor_pct=30.0,
        bare_year_fraction=0.6,
        h_chan_m=2.0,
        trough_depth_m=0.5,
        min_calibration_pixels=99,
    )

    assert result.bare_fraction.tolist() == [[0.6, 1.0, 0.6]]
    assert result.bare_valid_years.tolist() == [[5, 1, 5]]
    assert result.bare_stable.tolist() == [[True, False, True]]


def test_confidence_counts_w_and_s_once() -> None:
    shape = (3, 3)
    result = build_evidence(
        frequency=np.full(shape, 20.0),
        bare_yearly=np.full((2, *shape), 10.0),
        riverine_waterbody_mask=np.ones(shape, bool),
        rem=np.full(shape, 10.0),
        trough_depth=np.zeros(shape),
        domain=np.ones(shape, bool),
        centreline=np.zeros(shape, bool),
        corridor_mask=np.ones(shape, bool),
        line_fallback_mask=np.ones(shape, bool),
        f_chan_high=0.1,
        bare_threshold_floor_pct=30.0,
        bare_year_fraction=0.6,
        h_chan_m=2.0,
        trough_depth_m=0.5,
        min_calibration_pixels=99,
    )

    assert np.all(result.confidence == 2)  # topology + one water family


def test_connected_requires_domain_component_touching_centreline() -> None:
    domain = np.zeros((5, 7), bool)
    domain[2, 0:3] = True
    domain[2, 4:7] = True
    centreline = np.zeros_like(domain)
    centreline[2, 0] = True

    result = build_evidence(
        frequency=np.full(domain.shape, 20.0),
        bare_yearly=np.full((2, *domain.shape), 10.0),
        riverine_waterbody_mask=np.zeros_like(domain),
        rem=np.full(domain.shape, 10.0),
        trough_depth=np.zeros(domain.shape),
        domain=domain,
        centreline=centreline,
        corridor_mask=np.ones_like(domain),
        f_chan_high=0.1,
        bare_threshold_floor_pct=30.0,
        bare_year_fraction=0.6,
        h_chan_m=2.0,
        trough_depth_m=0.5,
    )

    assert result.connected[2, 0:3].all()
    assert not result.connected[2, 4:7].any()


def test_water_high_converts_fraction_threshold_to_percent_once() -> None:
    shape = (1, 2)
    result = build_evidence(
        frequency=np.array([[9.9, 10.0]]),
        bare_yearly=np.full((2, *shape), 10.0),
        riverine_waterbody_mask=np.zeros(shape, bool),
        rem=np.full(shape, 10.0),
        trough_depth=np.zeros(shape),
        domain=np.ones(shape, bool),
        centreline=np.ones(shape, bool),
        corridor_mask=np.ones(shape, bool),
        f_chan_high=0.10,
        bare_threshold_floor_pct=30.0,
        bare_year_fraction=0.6,
        h_chan_m=2.0,
        trough_depth_m=0.5,
    )
    assert result.water_high.tolist() == [[False, True]]


def test_build_evidence_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share shape"):
        build_evidence(
            frequency=np.zeros((2, 2)),
            bare_yearly=np.zeros((2, 3, 3)),
            riverine_waterbody_mask=np.zeros((2, 2), bool),
            rem=np.zeros((2, 2)),
            trough_depth=np.zeros((2, 2)),
            domain=np.zeros((2, 2), bool),
            centreline=np.zeros((2, 2), bool),
            corridor_mask=np.zeros((2, 2), bool),
            f_chan_high=0.1,
            bare_threshold_floor_pct=30.0,
            bare_year_fraction=0.6,
            h_chan_m=2.0,
            trough_depth_m=0.5,
        )
~~~

- [ ] **Step 5: Run evidence tests and verify missing module failure**

Run: python -m pytest tests/riverscape/test_riverscape_evidence.py -v

Expected: collection ERROR with ModuleNotFoundError for evidence.

- [ ] **Step 6: Implement evidence.py**

Create hydrofragments/riverscape/evidence.py:

~~~python
"""Per-pixel evidence for riverscape observed-channel rules."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class BareCalibration:
    threshold_pct: float
    candidate_median_pct: float | None
    background_median_pct: float | None
    candidate_pixels: int
    background_pixels: int
    degraded_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RiverscapeEvidence:
    water_high: np.ndarray
    waterbody_riverine: np.ndarray
    bare_stable: np.ndarray
    terrain: np.ndarray
    connected: np.ndarray
    line_fallback: np.ndarray
    bare_fraction: np.ndarray
    bare_valid_years: np.ndarray
    confidence: np.ndarray
    calibration: BareCalibration
    degraded_reasons: tuple[str, ...]


def _year_stack(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim != 3:
        raise ValueError("bare_yearly must be 2-D or 3-D (time, y, x)")
    return array


def calibrate_bare_threshold(
    bare_yearly: np.ndarray,
    candidate_mask: np.ndarray,
    background_mask: np.ndarray,
    *,
    floor_pct: float,
    min_pixels: int = 200,
) -> BareCalibration:
    yearly = _year_stack(bare_yearly)
    candidate = np.asarray(candidate_mask, bool)
    background = np.asarray(background_mask, bool)
    if candidate.shape != yearly.shape[1:] or background.shape != candidate.shape:
        raise ValueError("bare_yearly and calibration masks must share shape")
    if not 0.0 <= floor_pct <= 100.0:
        raise ValueError("floor_pct must be in [0, 100]")
    if min_pixels < 1:
        raise ValueError("min_pixels must be positive")
    with np.errstate(all="ignore"):
        per_pixel = np.nanmedian(yearly, axis=0)
    candidate_values = per_pixel[candidate & np.isfinite(per_pixel)]
    background_values = per_pixel[background & np.isfinite(per_pixel)]
    candidate_median = (
        float(np.median(candidate_values)) if candidate_values.size else None
    )
    background_median = (
        float(np.median(background_values)) if background_values.size else None
    )
    usable = (
        candidate_values.size >= min_pixels
        and background_values.size >= min_pixels
        and candidate_median is not None
        and background_median is not None
        and candidate_median > background_median
    )
    threshold = (
        max(floor_pct, (candidate_median + background_median) / 2.0)
        if usable
        else floor_pct
    )
    return BareCalibration(
        threshold_pct=float(threshold),
        candidate_median_pct=candidate_median,
        background_median_pct=background_median,
        candidate_pixels=int(candidate_values.size),
        background_pixels=int(background_values.size),
        degraded_reasons=() if usable else ("bare_threshold_fallback",),
    )


def build_evidence(
    frequency: np.ndarray,
    bare_yearly: np.ndarray,
    riverine_waterbody_mask: np.ndarray,
    rem: np.ndarray,
    trough_depth: np.ndarray,
    domain: np.ndarray,
    centreline: np.ndarray,
    corridor_mask: np.ndarray,
    *,
    f_chan_high: float,
    bare_threshold_floor_pct: float,
    bare_year_fraction: float,
    h_chan_m: float,
    trough_depth_m: float,
    min_calibration_pixels: int = 200,
    line_fallback_mask: np.ndarray | None = None,
) -> RiverscapeEvidence:
    frequency = np.asarray(frequency, float)
    yearly = _year_stack(bare_yearly)
    arrays = [
        np.asarray(riverine_waterbody_mask, bool),
        np.asarray(rem, float),
        np.asarray(trough_depth, float),
        np.asarray(domain, bool),
        np.asarray(centreline, bool),
        np.asarray(corridor_mask, bool),
    ]
    if frequency.ndim != 2 or any(a.shape != frequency.shape for a in arrays):
        raise ValueError("all 2-D evidence arrays must share shape")
    if yearly.shape[1:] != frequency.shape:
        raise ValueError("bare_yearly and 2-D evidence arrays must share shape")
    if not 0.0 <= f_chan_high <= 1.0:
        raise ValueError("f_chan_high must be in [0, 1]")
    if not 0.0 <= bare_year_fraction <= 1.0:
        raise ValueError("bare_year_fraction must be in [0, 1]")

    waterbody, rem, trough, domain, centreline, corridor = arrays
    fallback = (
        np.zeros(frequency.shape, bool)
        if line_fallback_mask is None
        else np.asarray(line_fallback_mask, bool)
    )
    if fallback.shape != frequency.shape:
        raise ValueError("line_fallback_mask must share shape")

    water_high = np.isfinite(frequency) & (frequency >= 100.0 * f_chan_high)
    terrain = (
        (np.isfinite(rem) & (rem <= h_chan_m))
        | (np.isfinite(trough) & (trough >= trough_depth_m))
    )
    support = domain & corridor
    labels, _ = ndimage.label(support, structure=np.ones((3, 3), int))
    touching = np.unique(labels[centreline & support])
    touching = touching[touching != 0]
    connected = np.isin(labels, touching)

    calibration = calibrate_bare_threshold(
        yearly,
        connected & (water_high | terrain),
        (~corridor) & np.isfinite(yearly).any(axis=0),
        floor_pct=bare_threshold_floor_pct,
        min_pixels=min_calibration_pixels,
    )
    valid_years = np.isfinite(yearly).sum(axis=0).astype(np.uint16)
    exceed = np.isfinite(yearly) & (yearly >= calibration.threshold_pct)
    bare_fraction = np.divide(
        exceed.sum(axis=0),
        valid_years,
        out=np.zeros(frequency.shape, dtype=np.float32),
        where=valid_years > 0,
    )
    bare_stable = (valid_years >= 2) & (
        bare_fraction >= bare_year_fraction
    )
    topology_family = connected | fallback
    water_family = water_high | waterbody
    geomorphic_family = bare_stable | terrain
    confidence = (
        topology_family.astype(np.uint8)
        + water_family.astype(np.uint8)
        + geomorphic_family.astype(np.uint8)
    )
    return RiverscapeEvidence(
        water_high=water_high,
        waterbody_riverine=waterbody,
        bare_stable=bare_stable,
        terrain=terrain,
        connected=connected,
        line_fallback=fallback,
        bare_fraction=bare_fraction,
        bare_valid_years=valid_years,
        confidence=confidence,
        calibration=calibration,
        degraded_reasons=calibration.degraded_reasons,
    )


__all__ = [
    "BareCalibration",
    "RiverscapeEvidence",
    "build_evidence",
    "calibrate_bare_threshold",
]
~~~

- [ ] **Step 7: Run tests**

Run: python -m pytest tests/riverscape/test_riverscape_sources.py tests/riverscape/test_riverscape_evidence.py -v

Expected: all pass.

- [ ] **Step 8: Prove discriminating tests with mutants**

Apply each mutant separately, run named test, then restore:

1. Change candidate_median > background_median to <; reversed-contrast test must fail.
2. Change midpoint / 2.0 to no division; midpoint test must fail.
3. Change yearly >= threshold to >; fraction-boundary test must fail.
4. Change valid_years >= 2 to >= 1; one-year pixel assertion must fail.
5. Sum water_high and waterbody separately in confidence; W/S family test must fail.
6. Restore old FC median reduction; loader time-preservation test must fail.
7. Remove the 100 multiplier from f_chan_high; percent-boundary test fails.

Then run: python -m pytest tests/riverscape/ -q

- [ ] **Step 9: Commit**

~~~bash
git add hydrofragments/io/riverscape_sources.py hydrofragments/riverscape/evidence.py tests/riverscape/test_riverscape_sources.py tests/riverscape/test_riverscape_evidence.py
git commit -m "feat: build yearly riverscape evidence and bare calibration"
~~~

---

### Task 2: Merge and classify DEA Waterbodies

**Files:**
- Create: hydrofragments/riverscape/waterbodies.py
- Create: tests/riverscape/test_riverscape_waterbodies.py

**Interfaces:**
- Consumes: loaded Waterbodies GeoDataFrame and CentrelineResult.skeleton.
- Produces: WaterbodyResult and classify_waterbodies(...). Task 1's build_evidence consumes riverine_mask; Task 4 protects off_channel_mask.

- [ ] **Step 1: Write failing role tests**

Create tests/riverscape/test_riverscape_waterbodies.py:

~~~python
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

gpd = pytest.importorskip("geopandas")

from hydrofragments.riverscape.waterbodies import classify_waterbodies

TRANSFORM = Affine(30, 0, 0, 0, -30, 300)


def test_touching_tile_fragments_merge() -> None:
    polygons = gpd.GeoDataFrame(
        {"source": [0, 1]},
        geometry=[box(30, 120, 120, 180), box(120, 120, 240, 180)],
        crs="EPSG:3577",
    )
    centreline = np.zeros((10, 10), bool)
    centreline[4:6, 1:8] = True

    result = classify_waterbodies(
        polygons, centreline, transform=TRANSFORM, pixel_m=30.0
    )

    assert len(result.polygons) == 1
    assert int(result.polygons.iloc[0].waterbody_id) == 1


def test_two_contact_elongated_body_is_riverine() -> None:
    polygons = gpd.GeoDataFrame(
        geometry=[box(30, 120, 270, 180)], crs="EPSG:3577"
    )
    centreline = np.zeros((10, 10), bool)
    centreline[4, 0:10] = True

    result = classify_waterbodies(
        polygons, centreline, transform=TRANSFORM, pixel_m=30.0
    )

    row = result.polygons.iloc[0]
    assert row.role == "riverine"
    assert row.boundary_contacts >= 2
    assert 0.5 <= row.centreline_span_fraction <= 1.0
    assert result.riverine_mask.any()
    assert not result.off_channel_mask.any()


def test_one_ended_billabong_branch_is_off_channel() -> None:
    polygons = gpd.GeoDataFrame(
        geometry=[box(120, 30, 180, 210)], crs="EPSG:3577"
    )
    centreline = np.zeros((10, 10), bool)
    centreline[3:9, 5] = True
    centreline[3, 0:6] = True  # branch enters body once and terminates

    result = classify_waterbodies(
        polygons, centreline, transform=TRANSFORM, pixel_m=30.0
    )

    assert result.polygons.iloc[0].role == "off_channel"
    assert result.off_channel_mask.any()
    assert not np.any(result.riverine_mask & result.off_channel_mask)


def test_compact_body_fails_elongation_even_when_crossed() -> None:
    polygons = gpd.GeoDataFrame(
        geometry=[box(90, 90, 210, 210)], crs="EPSG:3577"
    )
    centreline = np.zeros((10, 10), bool)
    centreline[5, 0:10] = True

    result = classify_waterbodies(
        polygons, centreline, transform=TRANSFORM, pixel_m=30.0
    )

    assert result.polygons.iloc[0].elongation < 2.0
    assert result.polygons.iloc[0].role == "off_channel"


def test_empty_input_returns_typed_empty_result() -> None:
    polygons = gpd.GeoDataFrame(geometry=[], crs="EPSG:3577")
    result = classify_waterbodies(
        polygons, np.zeros((4, 4), bool), transform=TRANSFORM, pixel_m=30.0
    )
    assert result.polygons.empty
    assert result.riverine_mask.shape == (4, 4)
~~~

- [ ] **Step 2: Verify missing module failure**

Run: python -m pytest tests/riverscape/test_riverscape_waterbodies.py -v

Expected: collection ERROR with ModuleNotFoundError.

- [ ] **Step 3: Implement waterbodies.py**

Create hydrofragments/riverscape/waterbodies.py:

~~~python
"""DEA Waterbodies fragment merging and riverine-role classification."""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

WATERBODY_RULESET_VERSION = "1.0.0"
MIN_RIVERINE_ELONGATION = 2.0
MIN_CENTRELINE_SPAN_FRACTION = 0.5
MIN_BOUNDARY_CONTACTS = 2


@dataclass(frozen=True)
class WaterbodyResult:
    polygons: gpd.GeoDataFrame
    riverine_mask: np.ndarray
    off_channel_mask: np.ndarray


def _polygon_parts(geometry):
    if isinstance(geometry, Polygon) and not geometry.is_empty:
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    return []


def _merged_components(polygons: gpd.GeoDataFrame) -> list[tuple[int, object]]:
    parts: list[tuple[int, object]] = []
    for position, geometry in enumerate(polygons.geometry):
        parts.extend((position, part) for part in _polygon_parts(geometry))
    parent = list(range(len(parts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(parts)):
        for right in range(left + 1, len(parts)):
            if parts[left][1].intersects(parts[right][1]):
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(len(parts)):
        groups.setdefault(find(index), []).append(index)
    merged = []
    for indices in groups.values():
        source_index = min(parts[index][0] for index in indices)
        merged.append(
            (source_index, unary_union([parts[index][1] for index in indices]))
        )
    return sorted(merged, key=lambda item: item[0])


def _rectangle_axes(
    geometry, pixel_m: float
) -> tuple[float, float, np.ndarray]:
    rectangle = geometry.minimum_rotated_rectangle
    coords = np.asarray(rectangle.exterior.coords, float)
    edges = np.diff(coords, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    major_index = int(np.argmax(lengths))
    major = float(lengths[major_index])
    minor = float(np.min(lengths))
    unit = edges[major_index] / max(major, pixel_m)
    return major, max(minor, pixel_m), unit


def _pixel_centres(mask: np.ndarray, transform: Affine) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.empty((0, 2), float)
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    return np.column_stack([xs, ys])


def _projected_span(points: np.ndarray, unit: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    projected = points @ unit
    return float(projected.max() - projected.min())


def classify_waterbodies(
    polygons: gpd.GeoDataFrame,
    centreline: np.ndarray,
    *,
    transform: Affine,
    pixel_m: float,
) -> WaterbodyResult:
    centreline = np.asarray(centreline, bool)
    if centreline.ndim != 2:
        raise ValueError("centreline must be 2-D")
    if pixel_m <= 0:
        raise ValueError("pixel_m must be positive")
    rows: list[dict[str, object]] = []
    riverine = np.zeros(centreline.shape, bool)
    off_channel = np.zeros(centreline.shape, bool)
    structure = np.ones((3, 3), bool)

    for waterbody_id, (_, geometry) in enumerate(
        _merged_components(polygons), start=1
    ):
        body = rasterize(
            [(geometry, 1)],
            out_shape=centreline.shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)
        inside_line = centreline & body
        exterior_ring = ndimage.binary_dilation(body, structure=structure) & ~body
        crossing_inside = inside_line & ndimage.binary_dilation(
            centreline & exterior_ring, structure=structure
        )
        _, contacts = ndimage.label(crossing_inside, structure=structure)
        major, minor, major_unit = _rectangle_axes(geometry, pixel_m)
        elongation = major / minor
        span_fraction = _projected_span(
            _pixel_centres(inside_line, transform), major_unit
        ) / max(major, pixel_m)
        is_riverine = (
            contacts >= MIN_BOUNDARY_CONTACTS
            and elongation >= MIN_RIVERINE_ELONGATION
            and span_fraction >= MIN_CENTRELINE_SPAN_FRACTION
        )
        role = "riverine" if is_riverine else "off_channel"
        rows.append(
            {
                "waterbody_id": waterbody_id,
                "role": role,
                "elongation": float(elongation),
                "centreline_span_fraction": float(span_fraction),
                "boundary_contacts": int(contacts),
                "geometry": geometry,
            }
        )
        if is_riverine:
            riverine |= body
        else:
            off_channel |= body

    frame = gpd.GeoDataFrame(
        rows,
        columns=[
            "waterbody_id",
            "role",
            "elongation",
            "centreline_span_fraction",
            "boundary_contacts",
            "geometry",
        ],
        geometry="geometry",
        crs=polygons.crs,
    )
    return WaterbodyResult(frame, riverine, off_channel)


__all__ = [
    "MIN_BOUNDARY_CONTACTS",
    "MIN_CENTRELINE_SPAN_FRACTION",
    "MIN_RIVERINE_ELONGATION",
    "WATERBODY_RULESET_VERSION",
    "WaterbodyResult",
    "classify_waterbodies",
]
~~~

- [ ] **Step 4: Run role tests**

Run: python -m pytest tests/riverscape/test_riverscape_waterbodies.py -v

Expected: all pass.

- [ ] **Step 5: Mutation checks**

Apply separately and restore:

1. Replace the three-way AND with OR; compact-body or billabong test must fail.
2. Change MIN_BOUNDARY_CONTACTS to 1; billabong test must fail.
3. Change >= MIN_RIVERINE_ELONGATION to <=; crossed-body tests must fail.
4. Remove union(left, right); fragment count test must fail.
5. Replace span/major with span/minor; bounded span-fraction assertion fails.

Run after restoration: python -m pytest tests/riverscape/ -q

- [ ] **Step 6: Commit**

~~~bash
git add hydrofragments/riverscape/waterbodies.py tests/riverscape/test_riverscape_waterbodies.py
git commit -m "feat: classify merged DEA Waterbodies roles"
~~~

---
### Task 3: Apply ordered observed-channel rules

**Files:**
- Create: hydrofragments/riverscape/channel.py
- Create: tests/riverscape/test_riverscape_channel.py

**Interfaces:**
- Consumes: Task 1 RiverscapeEvidence, caller reach labels, CentrelineResult.skeleton, and water seed.
- Produces: ChannelResult and classify_channel(...). Task 4 consumes ChannelResult.channel and seed_half_width_m.

- [ ] **Step 1: Write failing channel tests**

Create tests/riverscape/test_riverscape_channel.py:

~~~python
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
    centreline[1, 1] = True
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
    assert result.channel[4, 5]
    assert not result.channel[0, 5]
    assert result.seed_half_width_m["reach-1"] == 30.0


def test_missing_seed_width_degrades_and_rejects_reach() -> None:
    shape = (3, 3)
    centreline = np.zeros(shape, bool)
    evidence = _evidence(shape, connected=[(1, 1)], water=[(1, 1)])
    result = _run(evidence, centreline, np.zeros(shape, bool))
    assert not result.channel.any()
    assert result.degraded_reasons == ("reach_reach-1_missing_seed_width",)
~~~

- [ ] **Step 2: Verify missing module failure**

Run: python -m pytest tests/riverscape/test_riverscape_channel.py -v

Expected: collection ERROR with ModuleNotFoundError.

- [ ] **Step 3: Implement channel.py**

Create hydrofragments/riverscape/channel.py:

~~~python
"""Ordered, named observed-channel rules for riverscape zoning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage

from hydrofragments.riverscape.evidence import RiverscapeEvidence

RULESET_VERSION = "1.0.0"
RULE_NONE = 0
RULE_CONNECTED_WATER = 1
RULE_CONNECTED_BARE_TERRAIN = 2
RULE_ADJACENT_GEOMORPHIC = 3
RULE_LINE_FALLBACK = 4


@dataclass(frozen=True)
class ChannelResult:
    channel: np.ndarray
    confidence: np.ndarray
    rule_id: np.ndarray
    seed_half_width_m: dict[str, float]
    degraded_reasons: tuple[str, ...]


def classify_channel(
    evidence: RiverscapeEvidence,
    domain: np.ndarray,
    centreline: np.ndarray,
    water_seed: np.ndarray,
    reach_labels: np.ndarray,
    reach_keys: Mapping[int, str],
    *,
    pixel_m: float,
    width_growth_factor: float,
    min_channel_confidence: int,
    include_line_fallback_in_channel: bool = False,
) -> ChannelResult:
    domain = np.asarray(domain, bool)
    centreline = np.asarray(centreline, bool)
    water_seed = np.asarray(water_seed, bool)
    reach_labels = np.asarray(reach_labels)
    shape = domain.shape
    evidence_arrays = (
        evidence.water_high,
        evidence.waterbody_riverine,
        evidence.bare_stable,
        evidence.terrain,
        evidence.connected,
        evidence.line_fallback,
        evidence.confidence,
    )
    if domain.ndim != 2 or any(np.shape(array) != shape for array in evidence_arrays):
        raise ValueError("domain and evidence arrays must share a 2-D shape")
    if centreline.shape != shape or water_seed.shape != shape or reach_labels.shape != shape:
        raise ValueError("centreline, water_seed, reach_labels and domain must share shape")
    if pixel_m <= 0:
        raise ValueError("pixel_m must be positive")
    if width_growth_factor < 1.0:
        raise ValueError("width_growth_factor must be at least 1")
    if not 1 <= min_channel_confidence <= 3:
        raise ValueError("min_channel_confidence must be in [1, 3]")

    positive_labels = set(int(value) for value in np.unique(reach_labels) if value > 0)
    missing_keys = positive_labels - set(reach_keys)
    if missing_keys:
        raise ValueError(f"reach_keys missing labels: {sorted(missing_keys)}")

    half_width = ndimage.distance_transform_edt(water_seed) * pixel_m
    distance_to_centreline = ndimage.distance_transform_edt(~centreline) * pixel_m
    width_eligible = np.zeros(shape, bool)
    seed_widths: dict[str, float] = {}
    degraded: list[str] = list(evidence.degraded_reasons)
    for label in sorted(positive_labels):
        key = str(reach_keys[label])
        samples = half_width[
            (reach_labels == label) & centreline & water_seed
        ]
        samples = samples[np.isfinite(samples) & (samples > 0)]
        if samples.size == 0:
            degraded.append(f"reach_{key}_missing_seed_width")
            continue
        median_width = float(np.median(samples))
        seed_widths[key] = median_width
        width_eligible |= (
            (reach_labels == label)
            & (distance_to_centreline <= width_growth_factor * median_width)
        )

    confidence_ok = evidence.confidence >= min_channel_confidence
    ordinary = domain & width_eligible & confidence_ok
    water_family = evidence.water_high | evidence.waterbody_riverine
    channel = np.zeros(shape, bool)
    rule_id = np.zeros(shape, np.uint8)

    first = ordinary & evidence.connected & water_family
    channel[first] = True
    rule_id[first] = RULE_CONNECTED_WATER

    second = (
        ordinary
        & ~channel
        & evidence.connected
        & evidence.bare_stable
        & evidence.terrain
    )
    channel[second] = True
    rule_id[second] = RULE_CONNECTED_BARE_TERRAIN

    growth_candidates = (
        ordinary
        & ~channel
        & evidence.connected
        & (evidence.bare_stable | evidence.terrain)
    )
    structure = np.ones((3, 3), bool)
    while True:
        grown = growth_candidates & ndimage.binary_dilation(
            channel, structure=structure
        )
        if not grown.any():
            break
        channel[grown] = True
        rule_id[grown] = RULE_ADJACENT_GEOMORPHIC
        growth_candidates[grown] = False

    if include_line_fallback_in_channel:
        fallback = (
            domain
            & ~channel
            & evidence.line_fallback
            & confidence_ok
        )
        channel[fallback] = True
        rule_id[fallback] = RULE_LINE_FALLBACK

    return ChannelResult(
        channel=channel,
        confidence=np.asarray(evidence.confidence, np.uint8).copy(),
        rule_id=rule_id,
        seed_half_width_m=seed_widths,
        degraded_reasons=tuple(dict.fromkeys(degraded)),
    )


__all__ = [
    "ChannelResult",
    "RULESET_VERSION",
    "RULE_ADJACENT_GEOMORPHIC",
    "RULE_CONNECTED_BARE_TERRAIN",
    "RULE_CONNECTED_WATER",
    "RULE_LINE_FALLBACK",
    "RULE_NONE",
    "classify_channel",
]
~~~

- [ ] **Step 4: Run channel tests**

Run: python -m pytest tests/riverscape/test_riverscape_channel.py -v

Expected: all pass.

- [ ] **Step 5: Mutation checks**

Apply separately and restore:

1. Count W and S independently in Task 1 confidence; family-count test fails.
2. Replace rule 2 B & T with B | T; scald test fails.
3. Remove adjacency from rule 3; scald test fails.
4. Remove ~channel from rule 2; first-match rule-ID test fails.
5. Replace width_growth_factor * median_width with division; width-cap boundary fixture fails.
6. Replace confidence >= threshold with >; a new exact-confidence-two assertion fails.
7. Execute only one growth iteration; convergence test fails.

Run after restoration: python -m pytest tests/riverscape/ -q

- [ ] **Step 6: Commit**

~~~bash
git add hydrofragments/riverscape/channel.py tests/riverscape/test_riverscape_channel.py
git commit -m "feat: classify observed channel from ordered evidence rules"
~~~

---
### Task 4: Add bridge configuration, discover gaps, and route bridges

**Files:**
- Modify: hydrofragments/config.py
- Modify: tests/contracts/test_config.py
- Create: hydrofragments/riverscape/bridging.py
- Create: tests/riverscape/test_riverscape_bridging.py

**Interfaces:**
- Consumes: Task 2 off_channel_mask; Task 3 ChannelResult.channel and seed_half_width_m; validated drainage and caller reach labels.
- Produces: BridgeCostWeights, Gap, GapSearchResult, BridgeCostInputs, UnbridgedGap, BridgeResult, find_gaps(...), build_cost_surface(...), bridge_gaps(...).

- [ ] **Step 1: Add failing config tests**

Append to tests/contracts/test_config.py:

~~~python
def test_bridge_cost_weights_defaults_and_hash_shape() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())
    weights = config.riverscape.bridge_cost_weights
    assert (
        weights.terrain,
        weights.water,
        weights.bare,
        weights.green,
        weights.npv,
        weights.line_distance,
    ) == (1.0, 1.0, 0.5, 1.0, 0.0, 0.5)
    assert config.scientific_config()["riverscape"]["bridge_cost_weights"] == {
        "bare": 0.5,
        "green": 1.0,
        "line_distance": 0.5,
        "npv": 0.0,
        "terrain": 1.0,
        "water": 1.0,
    }
    changed = HydroConfig.from_mapping(
        minimal_config(
            riverscape={"bridge_cost_weights": {"terrain": 2.0}}
        )
    )
    assert changed.scientific_hash() != config.scientific_hash()


@pytest.mark.parametrize(
    "weights,match",
    [
        ({"terrain": -1}, "non-negative"),
        ({"terrain": float("nan")}, "finite"),
        (
            {
                "terrain": 0,
                "water": 0,
                "bare": 0,
                "green": 0,
                "npv": 0,
                "line_distance": 0,
            },
            "at least one positive",
        ),
        ({"mystery": 1}, "unknown config key"),
    ],
)
def test_bridge_cost_weights_reject_invalid_values(weights, match) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match=match):
        HydroConfig.from_mapping(
            minimal_config(riverscape={"bridge_cost_weights": weights})
        )
~~~

- [ ] **Step 2: Verify config tests fail**

Run: python -m pytest tests/contracts/test_config.py -k bridge_cost_weights -v

Expected: FAIL because RiverscapeConfig has no bridge_cost_weights.

- [ ] **Step 3: Add and parse BridgeCostWeights**

In hydrofragments/config.py:

1. Change SCIENTIFIC_HASH_SCHEMA_VERSION to "1.3.0".
2. Insert before RiverscapeConfig:

~~~python
@dataclass(frozen=True)
class BridgeCostWeights:
    terrain: float = 1.0
    water: float = 1.0
    bare: float = 0.5
    green: float = 1.0
    npv: float = 0.0
    line_distance: float = 0.5
~~~

3. Add this field to RiverscapeConfig after bridge_rem_max_m:

~~~python
    bridge_cost_weights: BridgeCostWeights = field(
        default_factory=BridgeCostWeights
    )
~~~

4. Add bridge_cost_weights to the allowed riverscape keys. Immediately before constructing RiverscapeConfig, parse:

~~~python
        weight_defaults = BridgeCostWeights()
        weight_raw = riverscape_raw.get("bridge_cost_weights", {})
        if not isinstance(weight_raw, Mapping):
            raise ConfigError("riverscape.bridge_cost_weights must be a mapping")
        weight_keys = {
            "terrain", "water", "bare", "green", "npv", "line_distance"
        }
        unknown_weight_keys = set(weight_raw) - weight_keys
        if unknown_weight_keys:
            joined = ", ".join(
                f"riverscape.bridge_cost_weights.{key}"
                for key in sorted(unknown_weight_keys)
            )
            raise ConfigError(f"unknown config key(s): {joined}")
        bridge_cost_weights = BridgeCostWeights(
            **{
                key: float(weight_raw.get(key, getattr(weight_defaults, key)))
                for key in weight_keys
            }
        )
        weight_values = tuple(
            getattr(bridge_cost_weights, key) for key in sorted(weight_keys)
        )
        if not all(math.isfinite(value) for value in weight_values):
            raise ConfigError(
                "riverscape.bridge_cost_weights values must be finite"
            )
        if not all(value >= 0 for value in weight_values):
            raise ConfigError(
                "riverscape.bridge_cost_weights values must be non-negative"
            )
        if not any(value > 0 for value in weight_values):
            raise ConfigError(
                "riverscape.bridge_cost_weights requires at least one positive value"
            )
~~~

5. Pass bridge_cost_weights=bridge_cost_weights into RiverscapeConfig.
6. Add this nested mapping inside scientific_config's riverscape mapping:

~~~python
                "bridge_cost_weights": {
                    "bare": self.riverscape.bridge_cost_weights.bare,
                    "green": self.riverscape.bridge_cost_weights.green,
                    "line_distance": (
                        self.riverscape.bridge_cost_weights.line_distance
                    ),
                    "npv": self.riverscape.bridge_cost_weights.npv,
                    "terrain": self.riverscape.bridge_cost_weights.terrain,
                    "water": self.riverscape.bridge_cost_weights.water,
                },
~~~

7. Export BridgeCostWeights in __all__.

- [ ] **Step 4: Run config tests**

Run: python -m pytest tests/contracts/test_config.py tests/contracts/test_hashing.py -q

Expected: all pass.

- [ ] **Step 5: Write failing bridging tests**

Create tests/riverscape/test_riverscape_bridging.py:

~~~python
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
~~~

- [ ] **Step 6: Verify missing module failure**

Run: python -m pytest tests/riverscape/test_riverscape_bridging.py -v

Expected: collection ERROR with ModuleNotFoundError.

- [ ] **Step 7: Implement bridging.py**

Create hydrofragments/riverscape/bridging.py:

~~~python
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
~~~

- [ ] **Step 8: Run bridging tests**

Run: python -m pytest tests/riverscape/test_riverscape_bridging.py -v

Expected: all pass.

- [ ] **Step 9: Add consecutive-segment and cross-junction gap tests**

Extend the test file with two fixtures:

~~~python
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
~~~

Run: python -m pytest tests/riverscape/test_riverscape_bridging.py -v

Expected: all pass.

- [ ] **Step 10: Mutation checks**

Apply separately and restore:

1. Invert each of water, bare, green, NPV, and line cost terms; a parameterized cost-order test must fail for every term.
2. Replace min(terrain_rem, terrain_trough) with max; trough-preference test fails.
3. Remove REM ridge from blocked; ridge test fails.
4. Remove sampling=(pixel_m, pixel_m); exact 300 m length/cost test fails.
5. Replace diagonal hypot with 1; diagonal path-length test fails.
6. Divide total cost by path pixel count; exact cost-per-m test fails.
7. Paint upstream width for every path cell; interpolation test fails.
8. Remove painted &= ~protected; no-overwrite test fails.
9. Reverse > in both length checks; accepted/rejected boundary tests fail.
10. Remove consecutive-neighbour selection; a three-segment fixture must fail by returning three gaps instead of two.

Run after restoration:

~~~bash
python -m pytest tests/riverscape/test_riverscape_bridging.py tests/contracts/test_config.py -q
python -m pytest tests/riverscape/ -q
~~~

- [ ] **Step 11: Commit**

~~~bash
git add hydrofragments/config.py hydrofragments/riverscape/bridging.py tests/contracts/test_config.py tests/riverscape/test_riverscape_bridging.py
git commit -m "feat: add two-anchor least-cost channel bridging"
~~~

---

### Task 5: Calibrate bridge weights on masked Fitzroy gaps

**Files:**
- Create: scripts/spikes/riverscape_phase4_calibration.py
- Create: tests/riverscape/test_riverscape_phase4_calibration.py
- Create at run time: docs/superpowers/specs/2026-09-15-riverscape-phase4-findings.md
- Modify only if selection rule chooses new defaults: hydrofragments/config.py and tests/contracts/test_config.py

**Interfaces:**
- Consumes: Tasks 1-4 and all Phase 3 results. Network/data execution uses data/fitzroy_basin_aoi.geojson and data/fitzroy_basin_drainage.gpkg.
- Produces: ignored output/spikes/riverscape_phase4_calibration.json and committed Phase 4 findings. May replace bridge weight/cost defaults only through the predeclared rule.

- [ ] **Step 1: Write failing selection tests**

Create tests/riverscape/test_riverscape_phase4_calibration.py:

~~~python
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
    scores = {BASELINE_NAME: 0.60, "better": 0.619}
    assert choose_candidate(scores, {BASELINE_NAME: 100, "better": 100}, 100) == BASELINE_NAME
    scores["better"] = 0.621
    assert choose_candidate(scores, {BASELINE_NAME: 100, "better": 79}, 100) == BASELINE_NAME
    assert choose_candidate(scores, {BASELINE_NAME: 100, "better": 80}, 100) == "better"


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
~~~

- [ ] **Step 2: Verify missing script failure**

Run: python -m pytest tests/riverscape/test_riverscape_phase4_calibration.py -v

Expected: collection ERROR because the script does not exist.

- [ ] **Step 3: Implement deterministic selection kernel**

Create scripts/spikes/riverscape_phase4_calibration.py with this header and selection kernel:

~~~python
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
~~~

- [ ] **Step 4: Add exact Fitzroy case preparation**

Continue the same script with these helpers and orchestration rules:

~~~python
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


def _sample_spans(channel, centreline, reach_labels, drainage, pixel_m):
    rng = np.random.default_rng(20260915)
    min_pixels = int(np.ceil(MIN_SPAN_M / pixel_m)) + 1
    max_pixels = int(np.floor(MAX_SPAN_M / pixel_m)) + 1
    area_by_label = {
        index: float(area)
        for index, area in enumerate(drainage["UpstrDArea"], start=1)
    }
    quantiles = np.quantile(list(area_by_label.values()), np.linspace(0, 1, 11))
    cases = []
    for stratum in range(10):
        labels = [
            label for label, area in area_by_label.items()
            if quantiles[stratum] <= area <= quantiles[stratum + 1]
        ]
        rng.shuffle(labels)
        for label in labels:
            spine = channel & centreline & (reach_labels == label)
            component_labels, count = ndimage.label(
                spine, structure=np.ones((3, 3), int)
            )
            paths = [
                _ordered_simple_path(component_labels == component)
                for component in range(1, count + 1)
            ]
            paths = [path for path in paths if len(path) >= min_pixels]
            if not paths:
                continue
            path = max(paths, key=len)
            span_pixels = min(len(path), max_pixels)
            start_index = int(rng.integers(0, len(path) - span_pixels + 1))
            points = np.asarray(
                path[start_index : start_index + span_pixels], dtype=int
            )
            truth = np.zeros(channel.shape, bool)
            truth[points[:, 0], points[:, 1]] = True
            anchor_width = ndimage.distance_transform_edt(channel) * pixel_m
            start = tuple(int(value) for value in points[0])
            end = tuple(int(value) for value in points[-1])
            cases.append(
                {
                    "label": label,
                    "truth": truth,
                    "gap": Gap(
                        gap_id=f"heldout-{stratum}-{len(cases)}",
                        reach_ids=(str(drainage.iloc[label - 1].HydroID),),
                        upstream_anchor=start,
                        downstream_anchor=end,
                        upstream_width_m=float(anchor_width[start]),
                        downstream_width_m=float(anchor_width[end]),
                        straight_length_m=float(
                            np.hypot(start[0] - end[0], start[1] - end[1])
                            * pixel_m
                        ),
                    ),
                }
            )
            if sum(case["label"] in labels for case in cases) >= 10:
                break
    return cases[:SAMPLE_COUNT]


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
        "One hundred 300-1800 m observed-channel spans were sampled "
        "deterministically across UpstrDArea deciles. Water/domain evidence "
        "was hidden on each span; routes retained terrain, FC, and line "
        "distance. Candidate score was median one-pixel-tolerance F1 minus "
        "twice off-channel crossing rate. Baseline was retained unless a "
        "candidate improved by at least 0.02 with at least 80% routable cases.\n\n"
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
~~~

Complete main() using these exact data-flow calls, in this order:

~~~python
def main() -> int:
    started = time.perf_counter()
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    cfg = RiverscapeConfig()
    aoi = gpd.read_file(AOI_PATH)
    drainage = gpd.read_file(DRAINAGE_PATH)
    stats = open_wo_statistics_for_zoning(aoi)
    geobox = stats.frequency.odc.geobox
    drainage = drainage.to_crs(str(geobox.crs))
    transform: Affine = geobox.transform
    pixel_m = abs(float(transform.a))
    frequency = np.asarray(stats.frequency, np.float32)
    domain = wet_domain(stats, min_valid_obs=20)
    water_seed = np.isfinite(frequency) & (frequency >= 100.0 * cfg.f_seed)

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
    reach_labels, reach_keys, corridor_mask, corridor_width_raster = (
        _rasterize_reaches(
            drainage, frequency.shape, transform, corridor_result.widths_m
        )
    )
    centreline_result = build_centreline(
        drainage,
        water_seed,
        corridor_result.widths_m,
        transform=transform,
        pixel_m=pixel_m,
    )
    dem = np.asarray(
        load_dem(geobox, product=cfg.dem_product, band=cfg.dem_band),
        np.float32,
    )
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
    fc = load_fc_percentiles(
        geobox,
        product=cfg.fc_product,
        bands=[cfg.bare_band, cfg.green_band, cfg.npv_band],
        years=(2022, 2023),
    )
    bare_yearly = np.asarray(fc[cfg.bare_band], np.float32)
    green = np.nanmedian(np.asarray(fc[cfg.green_band], np.float32), axis=0)
    npv = np.nanmedian(np.asarray(fc[cfg.npv_band], np.float32), axis=0)
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
    cases = _sample_spans(
        channel.channel,
        centreline_result.skeleton,
        reach_labels,
        drainage,
        pixel_m,
    )
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
            truth = case["truth"]
            hidden_frequency = frequency.copy()
            hidden_frequency[truth] = 0.0
            observed = channel.channel & ~truth
            inputs = BridgeCostInputs(
                rem=terrain.rem,
                trough_depth=terrain.trough_depth,
                frequency=hidden_frequency,
                bare_fraction=evidence.bare_fraction,
                green_pct=green,
                npv_pct=npv,
                line_distance_m=line_distance,
                corridor_width_m=corridor_width_raster,
                corridor_mask=corridor_mask,
                # Calibration disables production hard protection so
                # off-channel preference is measurable in the score below.
                domain=np.zeros_like(domain),
                off_channel_mask=np.zeros_like(waterbodies.off_channel_mask),
            )
            result = bridge_gaps(
                [case["gap"]],
                observed,
                inputs,
                transform=transform,
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
            f1_values.append(path_f1(result.bridged_mask, truth))
            crossing_values.append(
                float((result.bridged_mask & waterbodies.off_channel_mask).sum())
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
        "cost_cap_fallback": cost_cap == 1.0 and sum(
            np.asarray(raw_f1[selected]) >= 0.8
        ) < 80,
        "runtime_seconds": time.perf_counter() - started,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    FINDINGS_PATH.write_text(_render_findings(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
~~~

- [ ] **Step 5: Run unit tests**

Run: python -m pytest tests/riverscape/test_riverscape_phase4_calibration.py -v

Expected: all pass without network access.

- [ ] **Step 6: Run real Fitzroy calibration**

Run: python scripts/spikes/riverscape_phase4_calibration.py

Expected:
- exit code 0;
- exactly 100 cases;
- JSON written under ignored output/spikes;
- findings Markdown contains fully rendered measured values;
- every candidate has score, routable count, median F1, and off-channel rate.

This run previously cost roughly 12 minutes and 17.4 GiB for Phase 0's basin arrays. Record actual runtime and peak memory. If source access fails, preserve exact error and rerun; do not fabricate findings.

- [ ] **Step 7: Apply calibration decision mechanically**

Read selected_candidate and bridge_max_cost_per_m from JSON.

- If selected_candidate is baseline and cost_cap_fallback is true, leave config defaults unchanged.
- Otherwise replace BridgeCostWeights defaults and/or RiverscapeConfig.bridge_max_cost_per_m with JSON values, update exact assertions in test_riverscape_config_has_documented_defaults, and add one sentence to findings stating which defaults changed.
- Never select by visual preference or rerun until a preferred candidate wins.

Run:

~~~bash
python -m pytest tests/contracts/test_config.py tests/contracts/test_hashing.py tests/riverscape/ -q
python -m pytest -q
~~~

Expected: no new failures beyond documented pre-existing branding and intermittent Windows bundle-rename categories.

- [ ] **Step 8: Mutation checks**

Apply separately and restore:

1. Remove +0.02 margin in choose_candidate; margin test fails.
2. Change 0.8 routable threshold to 0.79; routable test fails.
3. Flip subtraction of off-channel penalty to addition; score-penalty test fails.
4. Replace percentile(..., 95) with mean; cost-cap test fails.
5. Remove fewer-than-80 fallback; cost-cap test fails.
6. Change one-pixel dilation to no dilation; path tolerance test fails.

- [ ] **Step 9: Commit calibration**

~~~bash
git add scripts/spikes/riverscape_phase4_calibration.py tests/riverscape/test_riverscape_phase4_calibration.py docs/superpowers/specs/2026-09-15-riverscape-phase4-findings.md hydrofragments/config.py tests/contracts/test_config.py
git commit -m "calibrate: validate riverscape bridge costs on Fitzroy"
~~~

Do not add output/spikes/riverscape_phase4_calibration.json.

---

## Final Verification

Run:

~~~bash
python -m pytest tests/riverscape/ -v
python -m pytest tests/contracts/test_config.py tests/contracts/test_hashing.py -q
python -m pytest tests/spatial/ tests/hydroperiod/ tests/contracts/ tests/guards tests/gating -q
python -m pytest -q
python -c "from hydrofragments.riverscape.evidence import RiverscapeEvidence; from hydrofragments.riverscape.waterbodies import WaterbodyResult; from hydrofragments.riverscape.channel import ChannelResult; from hydrofragments.riverscape.bridging import BridgeResult; print('ok')"
git status --short
~~~

Final reviewer must:

1. Compare every parent-spec Phase 4 acceptance fixture against a named test.
2. Run every task's listed mutants, not merely inspect tests.
3. Check 0-100 versus 0-1 conversions at all frequency/config boundaries.
4. Check no hydrofragments.riverscape module imports hydrofragments.spatial.
5. Check bridge painting cannot change any observed non-channel domain pixel.
6. Check config weights and cost cap appear in scientific_config and alter scientific_hash.
7. Check only task files and progress ledger changed; leave examples_out and output scratch trees untouched.

After review passes, update .superpowers/sdd/progress.md with task commits, review rounds, mutation results, calibration summary, and final HEAD. Ask user before any git push origin development.

## Execution Handoff

Recommended: subagent-driven execution, fresh Sonnet implementer and Opus reviewer per task, with explicit mutation-test prompts. Inline execution remains possible via superpowers:executing-plans. User approval is required before pushing.
