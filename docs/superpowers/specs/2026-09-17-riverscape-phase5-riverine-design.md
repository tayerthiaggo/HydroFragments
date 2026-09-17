# Riverscape Zoning Phase 5: Riverine vs Non-Riverine Landform

**Date:** 2026-09-17  
**Status:** Draft for review  

**Parent spec:** `docs/superpowers/specs/2026-09-14-riverscape-zoning-design.md`  
**Prior phase design:** `docs/superpowers/specs/2026-09-15-riverscape-phase4-channel-bridging-design.md`  
**Prior calibration:** `docs/superpowers/specs/2026-09-15-riverscape-phase4-findings.md`

This document resolves Phase 5 ambiguities left open by the parent spec. Where
this document is more specific than the parent for Phase 5, this document
governs. Phases 6–8 remain unchanged.

## 1. Scope and boundaries

Phase 5 builds one pure, independently testable module and a thin assembler:

1. per-run floodplain envelope calibration `h_fp(A) = a·A^b`;
2. classification of domain∖channel pixels into off-channel riverine vs
   non-riverine using REM and slope; and
3. assembly of the full landform code raster (0 / 1 / 2 / 3).

Phase 5 does **not** add `pipeline.py`, workflow modes, exports, manifest
wiring, window orchestration, MrVBF, hydroperiod combination, or
`channel_source` / bridge-vector packaging. Those remain Phase 6+
responsibilities.

Callers pass plain arrays and a per-reach `UpstrDArea` mapping. No module under
`hydrofragments/riverscape/` imports `hydrofragments.spatial`.

DEA Waterbodies `off_channel` roles do **not** override landform 2 vs 3.
Waterbody riverine masks already feed channel evidence (Phase 4); off-channel
polygons remain provenance / object inventory only for this phase.

## 2. Locked decisions

| # | Topic | Decision |
|---|---|---|
| 1 | Scope | Kernel + assemble landform raster; no pipeline/workflow |
| 2 | DEA `off_channel` | REM/slope only; no waterbody override of 2 vs 3 |
| 3 | Envelope sample | Domain pixels 8-connected to observed∪bridged channel |
| 4 | Per-pixel `A` | `UpstrDArea` of the reach owning the nearest channel pixel |
| 5 | Paint order | Outside domain → 0; channel → 1; else REM∧slope → 2; else → 3 |
| 6 | Fit | Fixed 0.5-dex `log10(A)` bins; per-bin REM quantile; OLS in log-log if ≥2 bins |
| 7 | Cap | `h_fp = min(a·A^b, envelope_h_max_m)` |
| 8 | Zero bins | Fail-closed: `a=0`, `b=0`, degrade `envelope_no_bins` |
| 9 | Single bin | `b=0`, `a=q`, degrade `envelope_single_bin` |
| 10 | MrVBF | Deferred until validation shows hillslope-dam leakage |
| 11 | Module | `hydrofragments/riverscape/riverine.py` + tests; reuse `codes.py` |

## 3. Interfaces

```python
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np

RIVERINE_RULESET_VERSION = "1.0.0"

@dataclass(frozen=True)
class FloodplainEnvelope:
    a: float
    b: float
    n_bins: int
    bin_log_a_mid: tuple[float, ...]
    bin_rem_quantile: tuple[float, ...]
    bin_counts: tuple[int, ...]
    degraded_reasons: tuple[str, ...]

@dataclass(frozen=True)
class LandformLayerResult:
    landform: np.ndarray  # uint8, codes from riverscape.codes
    envelope: FloodplainEnvelope
    degraded_reasons: tuple[str, ...]


def fit_floodplain_envelope(
    rem: np.ndarray,
    domain: np.ndarray,
    channel: np.ndarray,
    reach_labels: np.ndarray,
    upstr_darea: Mapping[int, float],
    *,
    envelope_quantile: float,
    envelope_min_bin_pixels: int,
) -> FloodplainEnvelope: ...


def classify_off_channel(
    rem: np.ndarray,
    slope_deg: np.ndarray,
    domain: np.ndarray,
    channel: np.ndarray,
    reach_labels: np.ndarray,
    upstr_darea: Mapping[int, float],
    envelope: FloodplainEnvelope,
    *,
    slope_max_deg: float,
    envelope_h_max_m: float,
) -> np.ndarray:
    """Return boolean mask of landform-2 candidates (domain & ~channel)."""


def assemble_landform(
    domain: np.ndarray,
    channel: np.ndarray,
    off_channel_riverine: np.ndarray,
) -> np.ndarray:
    """Return uint8 landform codes 0/1/2/3."""


def build_landform_layer(
    rem: np.ndarray,
    slope_deg: np.ndarray,
    domain: np.ndarray,
    channel: np.ndarray,
    reach_labels: np.ndarray,
    upstr_darea: Mapping[int, float],
    *,
    envelope_quantile: float,
    envelope_min_bin_pixels: int,
    envelope_h_max_m: float,
    slope_max_deg: float,
) -> LandformLayerResult:
    """fit → classify → assemble; union degraded reasons."""
```

`channel` is the boolean union of observed channel and bridged pixels.
`reach_labels` is the same integer raster used by corridor/REM (0 = no reach).
`upstr_darea` maps reach id → AHGF `UpstrDArea` in m².

Config fields already exist on `RiverscapeConfig` (`envelope_quantile`,
`envelope_h_max_m`, `envelope_min_bin_pixels`, `slope_max_deg`). Phase 5 does
not bump `SCIENTIFIC_HASH_SCHEMA_VERSION` unless a new scientific field is
added (none planned).

## 4. Envelope fit algorithm

### 4.1 Sample mask

1. Require `domain` and `channel` boolean, same shape as `rem`.
2. Build `connected`: pixels in `domain` that are 8-connected to at least one
   `channel` pixel (seed from `channel & domain`, dilate within `domain`).
3. Restrict to finite `rem`.
4. Assign each sample pixel the reach id of the **nearest channel pixel**
   (Euclidean distance transform indices into channel coordinates). Drop
   pixels whose nearest channel has label 0 or whose `UpstrDArea` is missing
   or not positive and finite.

### 4.2 Bins and quantile

1. Let `log_a = log10(A)` for sample pixels.
2. Partition with fixed bin width **0.5 in log₁₀(A)** (half order of
   magnitude), aligned so the first edge is `floor(min(log_a) / 0.5) * 0.5`
   and bins cover through `max(log_a)`. If `min == max`, use a single bin
   of width 0.5 centred on that value (edges `v - 0.25` .. `v + 0.25`).
3. Keep a bin only when its count ≥ `envelope_min_bin_pixels`.
4. For each kept bin: `log_a_mid` = bin centre; `A_mid = 10**log_a_mid`;
   `q = nanpercentile(rem_in_bin, 100 * envelope_quantile)`. Require `q > 0`
   and finite; otherwise discard the bin.

### 4.3 Power-law parameters

- **≥2 kept bins:** OLS of `log(q)` on `log(A_mid)` (natural log).  
  `b = slope`, `a = exp(intercept)`, `n_bins = len(kept)`.
- **Exactly 1 kept bin:** `b = 0`, `a = q`, `n_bins = 1`, append
  `envelope_single_bin` to `degraded_reasons`.
- **0 kept bins:** `b = 0`, `a = 0`, `n_bins = 0`, append
  `envelope_no_bins`. Fail-closed: no pixel satisfies `rem <= h_fp(A)` via a
  positive envelope.

`envelope_h_max_m` is not used as a substitute `a` when bins are missing; it
only caps evaluated heights.

### 4.4 Evaluate

For `A > 0` finite: `h_fp(A) = min(a * (A ** b), envelope_h_max_m)`.  
For invalid `A`: pixel cannot become landform 2.

## 5. Classification and assembly

### 5.1 Off-channel riverine mask

A domain pixel that is not channel is landform-2 candidate when all hold:

- `rem` and `slope_deg` finite;
- nearest-channel reach yields valid `A > 0`;
- `rem <= h_fp(A)`;
- `slope_deg <= slope_max_deg`.

Otherwise a domain∖channel pixel is non-riverine (landform 3).

### 5.2 Assemble

```text
landform[...] = 0
where domain:
  if channel:            1
  elif off_channel_mask: 2
  else:                  3
```

Channel always wins. `off_channel_riverine` bits outside `domain & ~channel`
are ignored (assembler may AND them defensively).

### 5.3 Validation

Shape or dtype contract failures raise `ValueError` (mismatched shapes,
non-2-D arrays). No silent coercion of landform codes from float NaN.

## 6. Acceptance mapping (parent §8)

| Parent fixture | Phase 5 test responsibility |
|---|---|
| Hillslope dam → non-riverine | High REM and/or slope > `slope_max_deg` → landform 3 |
| Flat-plain billabong → riverine | Low REM, gentle slope, valid A → landform 2 |
| Single drainage-area bin → `b = 0` and degraded | `envelope_single_bin` |
| Bridge never overwrites off-channel domain | Channel/bridge already landform 1; Phase 5 must not paint 2/3 onto channel (assembler invariant) |

Parent billabong “joined by one-pixel low-frequency link → off-channel
riverine” remains a **channel** width-cap concern (Phase 4); Phase 5 only
requires that such off-channel wet pixels classify as 2 when REM/slope pass.

## 7. Testing and mutants

File: `tests/riverscape/test_riverscape_riverine.py` (basename unique
repo-wide).

Required cases:

- flat-plain billabong → code 2;
- hillslope dam → code 3;
- single usable bin → `b == 0` and `envelope_single_bin`;
- zero usable bins → `a == 0`, `envelope_no_bins`, no code-2 from REM;
- channel pixels remain 1 when REM/slope would pass floodplain;
- outside domain remains 0;
- shape mismatch raises.

Discriminating mutants (apply, expect red, restore):

1. REM/slope conjunction → disjunction;
2. remove slope gate;
3. single-bin path sets `b = 0` without recording degrade;
4. 8-connected sample → 4-connected;
5. assembler paints 2 over channel;
6. zero-bin path sets `a = envelope_h_max_m` instead of 0.

## 8. Out of scope

- `build_landform` pipeline / workflow / gating / exports / manifest
- Windowed sub-catchment execution (Phase 7)
- MrVBF filter
- Changing bridge or waterbody kernels
- Fitzroy full-basin landform spike (optional later; not required for Phase 5
  acceptance — synthetic fixtures suffice)

## 9. Relationship to later phases

Phase 6 owns wiring: loaders → evidence → channel → bridges → **Phase 5
landform layer** → hydroperiod → `combine_zones`, plus
`channel_source` / confidence / bridge vectors in `LandformResult`-shaped
pipeline outputs. Phase 5's `LandformLayerResult` is the landform-code core
that pipeline will embed, not the final export DTO.
