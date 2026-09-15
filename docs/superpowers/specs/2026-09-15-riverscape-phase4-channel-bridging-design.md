# Riverscape Zoning Phase 4: Evidence, Channel Rules, Waterbodies, and Bridging

**Date:** 2026-09-15  
**Status:** Approved design  
**Parent spec:** `docs/superpowers/specs/2026-09-14-riverscape-zoning-design.md`  
**Prior findings:** `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md`  
**Prior implementation:** `docs/superpowers/plans/2026-09-15-riverscape-loaders-corridor-centreline-rem.md`

This document resolves the Phase 4 ambiguities left deliberately open by the
parent spec. It is an additive design: where this document is more specific
than the parent spec for Phase 4, this document governs. Phases 5-8 remain
unchanged.

## 1. Scope and boundaries

Phase 4 builds four pure, independently testable kernels and one calibration
script:

1. per-pixel evidence construction and per-run bare-threshold calibration;
2. DEA Waterbodies fragment merging and riverine/off-channel classification;
3. ordered observed-channel rules with confidence and width caps;
4. two-anchor gap discovery and least-cost bridge construction; and
5. a deterministic Fitzroy masked-gap calibration report.

Phase 4 does not add `build_landform`, workflow modes, exports, manifest
wiring, or window orchestration. Those remain Phase 6/7 responsibilities.
Callers pass plain arrays, a validated drainage `GeoDataFrame`, and a
caller-built reach-label raster. No module under
`hydrofragments/riverscape/` imports `hydrofragments.spatial`.

## 2. FC loader correction

`load_fc_percentiles` currently masks nodata and reduces every requested band
to a two-dimensional temporal median. That cannot implement the parent
spec's B evidence, which requires a threshold to be met in at least
`bare_year_fraction` of years.

Phase 4 changes the loader contract:

```python
def load_fc_percentiles(
    geobox: Any,
    *,
    product: str,
    bands: Sequence[str],
    years: tuple[int, int],
) -> dict[str, xr.DataArray]:
    """Return nodata-masked FC arrays, preserving any time dimension."""
```

Nodata is converted to NaN before return. If `odc.stac.load` supplies a time
dimension, it remains `(time, y, x)`; an already two-dimensional source stays
two-dimensional. Evidence construction, not source acquisition, owns all
scientific temporal reductions. The loader is not yet consumed by production
workflow, so this is an intentional pre-integration contract correction.

## 3. Waterbody roles

### 3.1 Interface

```python
WATERBODY_RULESET_VERSION = "1.0.0"

@dataclass(frozen=True)
class WaterbodyResult:
    polygons: geopandas.GeoDataFrame
    riverine_mask: np.ndarray
    off_channel_mask: np.ndarray

def classify_waterbodies(
    polygons: geopandas.GeoDataFrame,
    centreline: np.ndarray,
    *,
    transform: Affine,
    pixel_m: float,
) -> WaterbodyResult: ...
```

Input polygons must use the same projected CRS represented by `transform`.
The result `GeoDataFrame` contains `waterbody_id`, `role`, `elongation`,
`centreline_span_fraction`, `boundary_contacts`, and `geometry`.

### 3.2 Fragment merging

Explode multipart inputs, discard empty/non-polygonal parts, and build
connected components using exact polygon intersection/touching. Union each
component. This merges WFS/tile-edge fragments without closing real spatial
gaps. Stable `waterbody_id` values follow the merged components' sorted
minimum original row index.

### 3.3 Classification

Constants are versioned scientific rules, not user configuration until Phase
7 validation:

```python
MIN_RIVERINE_ELONGATION = 2.0
MIN_CENTRELINE_SPAN_FRACTION = 0.5
MIN_BOUNDARY_CONTACTS = 2
```

Elongation is the longer side divided by the shorter side of the minimum
rotated rectangle; the denominator is floored at `pixel_m`. Centreline span
is the range of centreline-pixel centres projected onto that rectangle's
major-axis unit vector, divided by the major-axis length. Boundary contacts are distinct
8-connected groups of centreline pixels that cross between the rasterized
polygon and its one-pixel exterior ring.

A body is `riverine` only when all three thresholds pass. Every other body is
`off_channel`. This deliberately rejects a one-ended skeleton branch into an
attached billabong even when that branch overlaps the polygon. Raster masks
are non-overlapping boolean arrays matching `centreline.shape`.

## 4. Evidence construction

### 4.1 Interfaces

```python
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

def calibrate_bare_threshold(
    bare_yearly: np.ndarray,
    candidate_mask: np.ndarray,
    background_mask: np.ndarray,
    *,
    floor_pct: float,
    min_pixels: int = 200,
) -> BareCalibration: ...

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
) -> RiverscapeEvidence: ...
```

`bare_yearly` is always normalized internally to `(time, y, x)`; a 2-D array
becomes a one-year stack but cannot establish B because at least two valid
years are required per pixel.

### 4.2 Bare calibration

Compute each pixel's median across finite yearly bare values. Calibration
candidates are computed as centreline-connected pixels supported by
`W_high | T`. Background pixels are finite-FC pixels outside all reach
corridors.

If each population has at least `min_pixels` values and the candidate median
is strictly greater than the background median, use:

```text
threshold = max(floor_pct, (candidate_median + background_median) / 2)
```

Otherwise use `floor_pct` and record `bare_threshold_fallback`. This is a
per-run calibration. Per-reach or regional calibration is deferred to Phase
7 because the Phase 0 basin findings establish heterogeneity but do not
provide enough per-reach evidence to select a defensible method.

For each pixel, `bare_fraction` is the fraction of finite yearly values at or
above the calibrated threshold. B is true only when at least two years are
finite and `bare_fraction >= bare_year_fraction`.

### 4.3 Other bits and confidence

```text
W_high = finite(frequency) and frequency >= 100 * f_chan_high
S      = riverine_waterbody_mask
T      = (REM <= h_chan_m) or (trough_depth >= trough_depth_m)
C      = in an 8-connected component of (domain and corridor_mask)
         that touches centreline
```

Confidence has exactly three families:

1. topology: `C` or explicit line-fallback support;
2. observed water: `W_high | S`; W and S together still count once; and
3. geomorphic: `B | T`.

Therefore confidence is an unsigned 8-bit raster in `[0, 3]`. Line-fallback
support alone scores one.

## 5. Observed-channel rules

### 5.1 Interface

```python
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
) -> ChannelResult: ...
```

Positive integer `reach_labels` identify the owning reach; zero means no
reach. `reach_keys` maps raster labels to string HydroIDs.

### 5.2 Width cap

Compute the Euclidean distance transform of `water_seed`. For each reach,
take the median half-width at centreline pixels carrying that reach label.
Eligible pixels must lie no farther from centreline than
`width_growth_factor * median_half_width`. Missing seed/centreline support
produces `reach_<id>_missing_seed_width`; that reach accepts no ordinary
channel pixels rather than running uncapped. `CorridorResult.widths_m` remains
a radius and is never halved.

### 5.3 Ordered rules

Apply first match only, always within domain, reach ownership, width cap, and
`confidence >= min_channel_confidence`:

1. `C & (W_high | S)`;
2. `C & B & T`;
3. iteratively grow through `C & (B | T)` pixels 8-adjacent to already
   accepted pixels, stopping at convergence; and
4. when explicitly enabled, accept line-fallback pixels that also meet the
   confidence threshold.

Earlier rule IDs never change during growth. First-match ordering is part of
`RULESET_VERSION`.

## 6. Gap discovery and bridging

### 6.1 Types and interfaces

```python
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
    channel_bridges: geopandas.GeoDataFrame
    unbridged: tuple[UnbridgedGap, ...]
    degraded_reasons: tuple[str, ...]

def find_gaps(
    channel: np.ndarray,
    centreline: np.ndarray,
    drainage: geopandas.GeoDataFrame,
    reach_labels: np.ndarray,
    reach_keys: Mapping[int, str],
    corridor_widths_m: Mapping[str, float],
    *,
    transform: Affine,
    pixel_m: float,
) -> GapSearchResult: ...

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
) -> BridgeResult: ...
```

### 6.2 Gap discovery

Label 8-connected components of `channel & centreline`. Assign component
endpoints to the nearest reach within that reach's corridor. Build the local
reach DAG from `HydroID` and `NextDownID`; do not import spatial helpers.
Resolve within-reach direction from connectivity to immediate upstream and
downstream reach geometries. If direction is geometrically ambiguous, skip
the candidate and record a degraded reason in the caller-visible diagnostic
path rather than guessing.

For every headwater-to-outlet path, sort component endpoints by reach order
and normalized within-reach chainage. Pair consecutive distinct components.
Canonicalize pairs so shared downstream trunks do not duplicate gaps. A
single observed segment at a dangling end has no second anchor and yields no
gap. Anchor widths come from the observed-channel Euclidean distance
transform. AHGF geometry determines ordering and cost only; no line pixel is
painted directly.

### 6.3 Cost surface

All finite terms are normalized to `[0, 1]`, where lower is more
channel-like:

```text
terrain = min(
    clip(max(REM, 0) / bridge_rem_max_m, 0, 1),
    1 - clip(max(trough_depth, 0) / trough_depth_m, 0, 1),
)
water = 1 - clip(frequency / 100, 0, 1)    # NaN/non-positive becomes 1
bare = 1 - clip(bare_fraction, 0, 1)       # NaN becomes 1
green = 1 - clip(green_pct / 100, 0, 1)   # NaN becomes 1
npv = 1 - clip(npv_pct / 100, 0, 1)       # NaN becomes 1
line = clip(line_distance_m / corridor_width_m, 0, 1)
cost = weighted_sum / sum(weights) + 1e-6
```

Pixels outside the corridor, pixels with `REM > bridge_rem_max_m`, and
observed domain pixels that are neither existing channel nor anchors are
impassable. `MCP_Geometric(cost, fully_connected=True,
sampling=(pixel_m, pixel_m))` is used; current scikit-image interprets each
cell value as cost per unit distance and returns a distance-weighted
cumulative cost.

Reject in this order: `straight_length_exceeds_max`, `no_finite_path`,
`path_length_exceeds_max`, `cost_per_m_exceeds_max`. Path length is the sum
of orthogonal/diagonal centre-to-centre distances, not pixel count.

### 6.4 Width, confidence, and cause

Interpolate anchor half-widths along cumulative path distance. Paint a disc
at every path cell, clip it to the corridor, and remove existing channel plus
all protected observed-domain/off-channel pixels. Thus bridges can add only
unobserved pixels.

`gap_cause` is `vegetated` when median finite green cover on the centre path
is at least `riparian_green_pct`; otherwise `narrow` when both anchor widths
are at most `narrow_width_px * pixel_m`; otherwise `unobserved`.

`bridge_confidence` is 2 only when path length and cost per metre are each at
most half their configured maxima; otherwise it is 1. The vector layer
records gap/reach IDs, geometry, length, cumulative and per-metre cost,
anchor widths, confidence, and cause.

## 7. Configuration

Add a frozen nested value type:

```python
@dataclass(frozen=True)
class BridgeCostWeights:
    terrain: float = 1.0
    water: float = 1.0
    bare: float = 0.5
    green: float = 1.0
    npv: float = 0.0
    line_distance: float = 0.5
```

`RiverscapeConfig.bridge_cost_weights` uses this type. Parsing accepts one
mapping with exactly these keys, rejects unknown keys, requires every value
to be finite and non-negative, and requires at least one positive value. The
nested mapping enters `scientific_config`; `SCIENTIFIC_HASH_SCHEMA_VERSION`
becomes `1.3.0`.

The initial defaults are the parent spec's values. Calibration may replace
them only under the predeclared selection rule below.

## 8. Fitzroy calibration

`scripts/spikes/riverscape_phase4_calibration.py` runs after the four kernels
exist. It deterministically samples 100 continuous observed-channel spans,
300-1800 m long, stratified over `UpstrDArea`. It hides water/domain evidence
along each held-out span while retaining terrain, FC, and line-distance
inputs, then routes between the original anchors.

Calibration routes deliberately disable the production hard-protection mask
so candidate weights can be penalized when they prefer an off-channel body;
the score measures that crossing against the real off-channel mask. Production
bridging keeps the hard protection invariant from Section 6.

Candidates are the parent-spec baseline; each of its five non-zero terms
ablated once; NPV set to 0.5; terrain doubled; water halved; and line distance
halved. Score is median one-pixel-tolerance path F1 minus twice the
off-channel crossing rate. Keep the baseline unless another candidate beats
it by at least 0.02 and at least 80% of the 100 spans are routable. Ties use
the declared candidate order.

For the selected weights, set `bridge_max_cost_per_m` to the p95 cost per
metre among routes with F1 at least 0.8. If fewer than 80 qualifying routes
exist, retain 1.0 and record insufficient calibration. Raw JSON is written to
ignored `output/spikes/riverscape_phase4_calibration.json`; method, sample
counts, candidate scores, selected weights, threshold decision, runtime, and
peak memory are committed to
`docs/superpowers/specs/2026-09-15-riverscape-phase4-findings.md`.

## 9. Verification and mutation testing

Each task follows TDD and ends with reviewer-run mutation checks. Passing
tests without killing the named mutants is not acceptance.

- Loader/evidence: time collapse, nodata masking after reduction, calibration
  contrast sign, midpoint operator, `>=` year-fraction boundary, one-year B.
- Waterbodies: replace AND with OR, one contact accepted, elongation/span
  thresholds inverted, touching fragments left separate.
- Channel: W and S counted twice, B/T OR substituted for rule 2's AND,
  adjacency removed, first-match overwrite, width factor multiplied/divided
  incorrectly, confidence comparison flipped.
- Bridging: each evidence cost sign inverted, ridge barrier removed, sampling
  omitted, diagonal length counted as pixels, cost divided by pixel count,
  width interpolation replaced with constant width, protected-pixel mask
  removed.
- Calibration: baseline-retention margin removed, off-channel penalty sign
  flipped, p95 replaced by mean, insufficient-sample fallback removed.

Synthetic acceptance covers the parent spec's sand bed, billabong, scald,
family counting, vegetated gap, narrow gap, dangling end, length/ridge
rejections, and no-overwrite fixtures. Full regression must introduce no new
failures beyond the two documented pre-existing categories in the Phase 3
handoff.
