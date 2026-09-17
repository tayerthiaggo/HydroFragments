# Riverscape Zoning — Plan 5: Riverine vs Non-Riverine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build Phase 5 floodplain envelope calibration, off-channel riverine vs non-riverine classification, and landform code assembly.

**Architecture:** One pure module `riverine.py` with fit → classify → assemble, plus a thin `build_landform_layer` wrapper. No pipeline/workflow. Reuse existing `codes.py` and `RiverscapeConfig` envelope/slope fields.

**Tech Stack:** Python 3.10+, numpy, scipy.ndimage, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-riverscape-phase5-riverine-design.md`

## Global Constraints

- No `hydrofragments.riverscape` → `hydrofragments.spatial` imports.
- TDD required; mutants in spec §7 must turn named tests red.
- Test basename `test_riverscape_riverine.py` unique repo-wide.
- Phase 5 does not create `pipeline.py`, bump hash schema, or add MrVBF.
- DEA off_channel waterbodies do not override landform 2 vs 3.
- Channel = observed ∪ bridged boolean supplied by caller.

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `hydrofragments/riverscape/riverine.py` | Create | Envelope fit, classify, assemble, wrapper |
| `tests/riverscape/test_riverscape_riverine.py` | Create | Acceptance + edge + mutant-discriminating tests |
| `hydrofragments/riverscape/__init__.py` | Modify | Re-export public API if file already exports peers |
| `.superpowers/sdd/progress.md` | Modify | Task ledger (gitignored; update locally) |

---

### Task 1: Floodplain envelope fit

**Files:**
- Create: `hydrofragments/riverscape/riverine.py`
- Create: `tests/riverscape/test_riverscape_riverine.py`

**Interfaces:**
- Consumes: `rem`, `domain`, `channel`, `reach_labels`, `upstr_darea`, `envelope_quantile`, `envelope_min_bin_pixels`
- Produces: `FloodplainEnvelope`, `fit_floodplain_envelope`

- [ ] **Step 1: Write failing envelope tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from hydrofragments.riverscape.codes import (
    LANDFORM_IN_CHANNEL,
    LANDFORM_NON_RIVERINE,
    LANDFORM_OFF_CHANNEL_RIVERINE,
    LANDFORM_OUTSIDE,
)
from hydrofragments.riverscape.riverine import (
    assemble_landform,
    build_landform_layer,
    classify_off_channel,
    fit_floodplain_envelope,
)


def test_single_bin_sets_b_zero_and_degrades() -> None:
    shape = (20, 20)
    rem = np.full(shape, 1.0, np.float32)
    domain = np.ones(shape, bool)
    channel = np.zeros(shape, bool)
    channel[10, 5:15] = True
    reach_labels = np.zeros(shape, np.int32)
    reach_labels[channel] = 1
    # One A only → one log bin; lower min_bin so the bin is kept
    envelope = fit_floodplain_envelope(
        rem,
        domain,
        channel,
        reach_labels,
        {1: 1.0e6},
        envelope_quantile=0.95,
        envelope_min_bin_pixels=10,
    )
    assert envelope.b == 0.0
    assert envelope.n_bins == 1
    assert envelope.a > 0.0
    assert "envelope_single_bin" in envelope.degraded_reasons


def test_zero_bins_fail_closed() -> None:
    shape = (8, 8)
    rem = np.full(shape, 1.0, np.float32)
    domain = np.ones(shape, bool)
    channel = np.zeros(shape, bool)
    channel[4, 2:6] = True
    reach_labels = np.zeros(shape, np.int32)
    reach_labels[channel] = 1
    envelope = fit_floodplain_envelope(
        rem,
        domain,
        channel,
        reach_labels,
        {1: 1.0e6},
        envelope_quantile=0.95,
        envelope_min_bin_pixels=10_000,
    )
    assert envelope.a == 0.0
    assert envelope.b == 0.0
    assert envelope.n_bins == 0
    assert "envelope_no_bins" in envelope.degraded_reasons
```

- [ ] **Step 2: Run tests — expect ImportError / fail**

Run: `python -m pytest tests/riverscape/test_riverscape_riverine.py::test_single_bin_sets_b_zero_and_degrades tests/riverscape/test_riverscape_riverine.py::test_zero_bins_fail_closed -v`

- [ ] **Step 3: Implement `fit_floodplain_envelope` per spec §4**

Constants: `LOG_BIN_WIDTH = 0.5`, `RIVERINE_RULESET_VERSION = "1.0.0"`.

Use `scipy.ndimage.binary_propagation` (8-connected) for sample connectivity;
`distance_transform_edt(~channel, return_indices=True)` for nearest reach;
OLS via `np.polyfit(np.log(A_mid), np.log(q), 1)` when ≥2 bins.

- [ ] **Step 4: Run envelope tests — expect pass**

- [ ] **Step 5: Commit**

```bash
git add hydrofragments/riverscape/riverine.py tests/riverscape/test_riverscape_riverine.py
git commit -m "feat: fit riverscape floodplain envelope from channel-connected REM"
```

---

### Task 2: Classify, assemble, wrapper, mutants

**Files:**
- Modify: `hydrofragments/riverscape/riverine.py`
- Modify: `tests/riverscape/test_riverscape_riverine.py`
- Modify: `hydrofragments/riverscape/__init__.py` (re-exports only if peers are exported)

**Interfaces:**
- Consumes: Task 1 envelope API
- Produces: `classify_off_channel`, `assemble_landform`, `build_landform_layer`, `LandformLayerResult`

- [ ] **Step 1: Write failing classify/assemble tests**

```python
def _two_bin_fixture():
    """Two A values spanning >0.5 dex with enough pixels per bin."""
    shape = (40, 40)
    rem = np.full(shape, 2.0, np.float32)
    slope = np.full(shape, 0.5, np.float32)
    domain = np.ones(shape, bool)
    channel = np.zeros(shape, bool)
    channel[20, 5:35] = True
    reach_labels = np.zeros(shape, np.int32)
    # Left half of channel = small A, right = large A (different 0.5-dex bins)
    reach_labels[20, 5:20] = 1
    reach_labels[20, 20:35] = 2
    upstr = {1: 1.0e4, 2: 1.0e7}
    # Seed enough connected domain REM samples: fill rem low on floodplain
    rem[:, :] = 1.0
    rem[10:15, 10:15] = 0.5  # billabong pocket
    rem[2:6, 2:6] = 20.0  # hillslope dam pocket
    slope[2:6, 2:6] = 5.0
    return rem, slope, domain, channel, reach_labels, upstr


def test_flat_plain_billabong_is_off_channel_riverine() -> None:
    rem, slope, domain, channel, labels, upstr = _two_bin_fixture()
    result = build_landform_layer(
        rem,
        slope,
        domain,
        channel,
        labels,
        upstr,
        envelope_quantile=0.95,
        envelope_min_bin_pixels=5,
        envelope_h_max_m=15.0,
        slope_max_deg=2.0,
    )
    assert result.landform[12, 12] == LANDFORM_OFF_CHANNEL_RIVERINE


def test_hillslope_dam_is_non_riverine() -> None:
    rem, slope, domain, channel, labels, upstr = _two_bin_fixture()
    result = build_landform_layer(
        rem,
        slope,
        domain,
        channel,
        labels,
        upstr,
        envelope_quantile=0.95,
        envelope_min_bin_pixels=5,
        envelope_h_max_m=15.0,
        slope_max_deg=2.0,
    )
    assert result.landform[4, 4] == LANDFORM_NON_RIVERINE


def test_channel_wins_over_floodplain_criteria() -> None:
    rem, slope, domain, channel, labels, upstr = _two_bin_fixture()
    result = build_landform_layer(
        rem,
        slope,
        domain,
        channel,
        labels,
        upstr,
        envelope_quantile=0.95,
        envelope_min_bin_pixels=5,
        envelope_h_max_m=15.0,
        slope_max_deg=2.0,
    )
    assert np.all(result.landform[channel] == LANDFORM_IN_CHANNEL)


def test_outside_domain_is_zero() -> None:
    rem, slope, domain, channel, labels, upstr = _two_bin_fixture()
    domain[0, 0] = False
    result = build_landform_layer(
        rem,
        slope,
        domain,
        channel,
        labels,
        upstr,
        envelope_quantile=0.95,
        envelope_min_bin_pixels=5,
        envelope_h_max_m=15.0,
        slope_max_deg=2.0,
    )
    assert result.landform[0, 0] == LANDFORM_OUTSIDE


def test_shape_mismatch_raises() -> None:
    rem = np.zeros((4, 4), np.float32)
    with pytest.raises(ValueError):
        assemble_landform(
            np.ones((4, 4), bool),
            np.zeros((3, 4), bool),
            np.zeros((4, 4), bool),
        )


def test_zero_bins_yield_no_landform_two_from_rem() -> None:
    shape = (10, 10)
    rem = np.full(shape, 0.1, np.float32)
    slope = np.full(shape, 0.1, np.float32)
    domain = np.ones(shape, bool)
    channel = np.zeros(shape, bool)
    channel[5, 2:8] = True
    labels = np.zeros(shape, np.int32)
    labels[channel] = 1
    result = build_landform_layer(
        rem,
        slope,
        domain,
        channel,
        labels,
        {1: 1e6},
        envelope_quantile=0.95,
        envelope_min_bin_pixels=10_000,
        envelope_h_max_m=15.0,
        slope_max_deg=2.0,
    )
    assert "envelope_no_bins" in result.degraded_reasons
    assert not np.any(result.landform == LANDFORM_OFF_CHANNEL_RIVERINE)
```

Tune `_two_bin_fixture` during green so both bins keep ≥5 samples and billabong rem is below fitted `h_fp`.

- [ ] **Step 2: Run new tests — expect fail**

- [ ] **Step 3: Implement classify / assemble / `build_landform_layer`**

- [ ] **Step 4: All riverine tests pass**

Run: `python -m pytest tests/riverscape/test_riverscape_riverine.py -v`

- [ ] **Step 5: Mutation checks** (apply, expect RED, restore; clear `__pycache__`)

1. REM∧slope → REM∨slope — billabong/dam discrimination fails or dam becomes 2
2. Remove slope gate — dam with high rem only may still fail; prefer dam with rem low but slope high → must stay 3 only with slope gate
3. Single-bin without degrade string — `test_single_bin_*` fails
4. 8-connect → 4-connect structure — add a diagonal-only connected sample test if needed, or mutant against connectivity helper
5. Assemble paints 2 over channel — `test_channel_wins_*` fails
6. Zero-bin `a = envelope_h_max_m` — `test_zero_bins_*` fails

- [ ] **Step 6: Commit**

```bash
git add hydrofragments/riverscape/riverine.py tests/riverscape/test_riverscape_riverine.py hydrofragments/riverscape/__init__.py
git commit -m "feat: classify off-channel riverine landform from REM envelope"
```

---

## Final Verification

```bash
python -m pytest tests/riverscape/test_riverscape_riverine.py -v
python -m pytest tests/riverscape/ tests/contracts/test_config.py -q
python -c "from hydrofragments.riverscape.riverine import build_landform_layer; print('ok')"
git status --short
```

Update `.superpowers/sdd/progress.md` with Plan 5 commits and mutant counts. Ask before `git push`.

## Execution Handoff

User requested immediate implementation after plan write — execute inline with TDD in this session.
