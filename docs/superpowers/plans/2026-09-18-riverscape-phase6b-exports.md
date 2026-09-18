# Riverscape Phase 6b Exports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain the riverscape science arrays past zoning and export them — a manifest `zoning` section, mode-keyed zone names, stable `mode`/`landform`/`hydroperiod` GPKG columns, six `riverscape_evidence` GeoTIFFs, and a `channel_bridges` vector layer — without a second DEM/FC/waterbodies load and without changing any metric value.

**Architecture:** Four seams, one direction of flow. (1) `hydrofragments/spatial/zones.py` gains `zones_from_riverscape_with_landform`, which returns the `LandformResult` that `zones_from_riverscape` used to throw away; `zones_from_riverscape` becomes a thin wrapper so every existing caller and test is untouched. (2) `hydrofragments/output/riverscape_export.py` owns the frozen `RiverscapeExportBundle` that carries `(landform, hydroperiod_classes, zone_result)` from `workflow._resolve_zone_result` to `finalize_analysis_bundle`; it lives under `output/` so `spatial/` stays zoning-only. (3) `hydrofragments/output/manifest.py` gains `build_zoning_section`, a pure function over `(zone_result, bundle)` that produces the full riverscape shape, the thin occurrence subset, or the no-`ZoneResult` reasons-only shape. (4) `hydrofragments/output/finalize.py` picks zone names by mode, always writes the three new GPKG columns, and — only when the `riverscape_evidence` product is opted in **and** a bundle is present — writes six rasters plus the `channel_bridges` layer.

**Tech Stack:** Python, pytest, geopandas, rasterio/rioxarray, existing HydroFragments output bundle

**Spec:** `docs/superpowers/specs/2026-09-18-riverscape-phase6b-exports-design.md` (prior phase: `docs/superpowers/specs/2026-09-18-riverscape-phase6a-workflow-design.md`; parent: `docs/superpowers/specs/2026-09-14-riverscape-zoning-design.md`)

## Global Constraints

- **Import boundary (spec §1.3).** `hydrofragments.riverscape` must **not** import `hydrofragments.spatial`, enforced by the AST scan in `tests/riverscape/test_riverscape_import_boundary.py`. Nothing in this plan adds an import to any module under `hydrofragments/riverscape/`. The new `hydrofragments/output/riverscape_export.py` imports *from* `riverscape.pipeline` and `spatial.zones`, which is the allowed direction. `riverscape.pipeline.with_grid` stays the only seam for attaching a `SpatialGrid` to a `LandformResult`.
- **No second source load (spec §1.1 item 1).** Export code never calls `load_dem`, `load_fc_percentiles`, `load_waterbodies`, `build_landform`, or `classify_hydroperiod`. Everything exported comes out of the bundle produced during zoning.
- **No re-classification of hydroperiod (spec §3.1).** `hydroperiod_classes` is the exact array that went into `combine_zones`. Because `combine_zones` writes `crosstab = landform * 10 + hydroperiod` with `0` outside (`hydrofragments/spatial/zones.py:320`) and both layers are validated to agree on the zoned extent (`zones.py:315`), the identity `hydroperiod == crosstab % 10` holds pixel-for-pixel — inside the extent because `hydroperiod ∈ {1,2,3,4}`, outside because `crosstab == 0`. `RiverscapeExportBundle.from_zoning` recovers the array with that identity instead of re-running `classify_hydroperiod`. This is load-bearing and has its own test and mutant.
- **No config schema version bump (spec §1.2, §2 row 13).** `ACCEPTED_CONFIG_SCHEMA_VERSIONS` and `SCIENTIFIC_HASH_SCHEMA_VERSION` in `hydrofragments/config.py` are untouched. Adding `riverscape_evidence` to `SPATIAL_PRODUCTS` widens an allowlist; it changes no default, so a config that does not request the product hashes identically to before.
- **No manifest schema version bump.** `MANIFEST_SCHEMA_VERSION` stays `"1.1.0"`. `zoning` is an additive optional top-level key; `validate_result_bundle` does not reject unknown keys.
- **Provenance keys are copied verbatim (spec §3.3).** Every key under `LandformResult.provenance` — built by `_build_provenance` at `hydrofragments/riverscape/pipeline.py:396-457` — is reproduced in `zoning` under the same name: `channel_ruleset_version`, `waterbody_ruleset_version`, `riverine_ruleset_version`, `dem_product`, `dem_band`, `fc_product`, `fc_bands`, `waterbodies_source`, `years`, `bridge_enabled`, `bare_threshold_pct`, `envelope`, `pixel_counts`, `bridge_counts_by_cause`, `unbridged_counts_by_reason`, `reach_counts`. Renaming any of them is a plan violation. The only regrouping permitted is the spec's own: the three `*_ruleset_version` keys are additionally surfaced under `ruleset_versions` as `channel`/`waterbody`/`riverine`, and the source keys additionally under `sources`, exactly as spec §3.3's JSON shows.
- **GPKG zone rows carry null landform/hydroperiod (spec §3.4).** Dissolved multipolygons mix hydroperiod values, so `landform` and `hydroperiod` are always `None` on the `zones` layer, in both modes. The authoritative per-pixel values are the `riverscape_evidence` rasters. Exploding to crosstab polygons is out of scope.
- **Evidence product is opt-in (spec §2 row 9).** `hydrofragments.workflow._default_config` must **not** add `riverscape_evidence`. An occurrence run stays hermetic.
- **Evidence write gate (spec §3.5, §3.6).** Rasters *and* the `channel_bridges` layer are written only when `"riverscape_evidence" in config.output.spatial_products` **and** `riverscape_bundle is not None`. Requested-without-bundle fails preflight with `SpatialProductUnavailable`. Bridges use the same gate so they are never orphaned.
- **Grid is mandatory for evidence (spec §2 row 11).** `bundle.landform.grid` must be a real `SpatialGrid` before any raster write; a missing grid is a programming error and raises, it is not degraded.
- **No silent catch of `RiverscapeSourceUnavailable` in export code (spec §5).** Export never loads sources, so it never has a reason to catch it.
- **Metrics must not change (spec §4).** Nothing in this plan touches `hydrofragments/guards/scientific.py`, the metric registry, or `analyze()`'s record construction. The gating test proves it.
- **No live network in CI (spec §2 row 14, §10).** Every test here either calls pure functions on hand-built arrays or monkeypatches the hydroseason/riverscape entry points, following `tests/integration/test_riverscape_modes.py`'s established convention.
- **Commands** run from the repo root `D:/RLH/5.6/repos/HydroFragments` with `python -m pytest`.

## Locked decisions (no ambiguity)

| # | Topic | Decision |
|---|---|---|
| L1 | Retention API | Companion function `zones_from_riverscape_with_landform(...) -> tuple[ZoneResult, LandformResult]`. **Not** a `return_landform: bool` flag. `zones_from_riverscape` stays `-> ZoneResult`. |
| L2 | Bundle module | `hydrofragments/output/riverscape_export.py`. Not `spatial/`. |
| L3 | `hydroperiod_classes` source | Derived as `crosstab % 10` in `RiverscapeExportBundle.from_zoning`; never re-classified. |
| L4 | Crosstab raster dtype | `uint16`, nodata `0`. (`combine_zones` produces `uint8`; the export upcasts.) |
| L5 | Evidence filenames | `rasters/landform.tif`, `rasters/hydroperiod.tif`, `rasters/zone_crosstab.tif`, `rasters/channel_source.tif`, `rasters/channel_confidence.tif`, `rasters/rem.tif` |
| L6 | Evidence contract keys | `riverscape_landform`, `riverscape_hydroperiod`, `riverscape_zone_crosstab`, `riverscape_channel_source`, `riverscape_channel_confidence`, `riverscape_rem` in `RASTER_PRODUCT_CONTRACTS` |
| L7 | Bridges layer name | `CHANNEL_BRIDGES_LAYER = "channel_bridges"` in `hydrofragments/output/finalize.py` |
| L8 | `domain_digest` | `hydrofragments.output.manifest._hash_array(zone_result.mask)` — the exact helper `build_dea_provenance` already uses for `zone_mask_digest` (`manifest.py:202-218`, called at `manifest.py:245`). No second algorithm. |
| L9 | `domain_pixel_count` | Riverscape: `provenance["pixel_counts"]["domain"]`. Occurrence: `int(np.count_nonzero(zone_result.mask))`. |
| L10 | Manifest wiring | `build_run_manifest(..., zoning=...)` keyword, forwarded through `BundleTransaction.finalize(..., zoning=...)`. |
| L11 | Preflight plumbing | `preflight_spatial_outputs(..., riverscape_bundle=None)`; `api._run_core_analysis(..., riverscape_bundle=None)` forwards it. |
| L12 | Bridges test module | Task 4's integration test requests `("riverscape_evidence",)` only — **not** `("zones",)` — because the zones-raster branch at `finalize.py:378-398` needs `core.spatial_grid`, which is populated by `SectionSpatialCollector` and is an orthogonal dependency. Zone names and GPKG columns are unit-tested directly in Task 3. |

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `hydrofragments/output/riverscape_export.py` | Create | `RiverscapeExportBundle` frozen dataclass + `from_zoning` constructor. |
| `hydrofragments/spatial/zones.py` | Modify | Add `zones_from_riverscape_with_landform`; make `zones_from_riverscape` a wrapper; import `with_grid`. |
| `hydrofragments/spatial/__init__.py` | Modify | Re-export `zones_from_riverscape_with_landform`. |
| `hydrofragments/workflow.py` | Modify | `_resolve_zone_result` 3-tuple; build the bundle; plumb into `_run_core_analysis` and `finalize_analysis_bundle`. |
| `hydrofragments/api.py` | Modify | `_run_core_analysis(..., riverscape_bundle=None)` → `preflight_spatial_outputs`. |
| `hydrofragments/config.py` | Modify | Add `riverscape_evidence` to `SpatialProduct` and `SPATIAL_PRODUCTS`. |
| `hydrofragments/output/manifest.py` | Modify | `build_zoning_section`; `build_run_manifest(..., zoning=...)`. |
| `hydrofragments/output/bundle.py` | Modify | `BundleTransaction.finalize(..., zoning=...)` passthrough. |
| `hydrofragments/output/rasters.py` | Modify | Six evidence contracts + `write_riverscape_evidence_geotiffs`. |
| `hydrofragments/output/finalize.py` | Modify | Mode-keyed names, GPKG columns, evidence preflight + write, bridges layer, zoning wiring. |
| `tests/output/test_riverscape_export_bundle.py` | Create | Task 1 unit tests. |
| `tests/spatial/test_zones_from_riverscape.py` | Modify | Task 1: add `with_landform` cases; existing cases unchanged. |
| `tests/integration/test_riverscape_modes.py` | Modify | Task 1: 3-tuple unpack + patch the new function name. |
| `tests/output/test_manifest_zoning.py` | Create | Task 2 unit tests. |
| `tests/output/test_riverscape_evidence_exports.py` | Create | Task 3 unit tests. |
| `tests/integration/test_riverscape_exports.py` | Create | Task 4 end-to-end test. |
| `tests/gating/test_zones_do_not_multiply_metrics.py` | Modify | Task 4: riverscape branch. |
| `docs/spatial_exports.md` | Modify | Task 4: zones columns, evidence, bridges. |

---

## Task 1: Retention seam + `RiverscapeExportBundle`

**Files:**
- Create: `hydrofragments/output/riverscape_export.py`
- Modify: `hydrofragments/spatial/zones.py`, `hydrofragments/spatial/__init__.py`, `hydrofragments/workflow.py`
- Test: `tests/output/test_riverscape_export_bundle.py` (create), `tests/spatial/test_zones_from_riverscape.py` (modify), `tests/integration/test_riverscape_modes.py` (modify)

**Interfaces:**

*Consumes* (all verified against source):
- `hydrofragments.spatial.zones.ZoneResult(mask, emitted_zones, has_zone_1, source="occurrence", grid=None, mode="occurrence", crosstab=None, degraded_reasons=())` — frozen dataclass, `zones.py:39-68`
- `hydrofragments.spatial.zones._attach_grid(result, template, *, require_georeference=False) -> ZoneResult` — `zones.py:71-84`; returns `result` unchanged when `template` is not an `xr.DataArray`
- `hydrofragments.spatial.zones.combine_zones(landform, hydroperiod, *, source="riverscape", degraded_reasons=()) -> ZoneResult` — `zones.py:286`
- `hydrofragments.riverscape.pipeline.LandformResult` — frozen, fields `landform, channel_source, channel_confidence, rule_id, rem, bridged_mask, channel_bridges, unbridged, envelope, provenance, degraded_reasons, grid=None` (`pipeline.py:96-119`)
- `hydrofragments.riverscape.pipeline.with_grid(result, grid) -> LandformResult` — `pipeline.py:122-131`
- `hydrofragments.output.spatial.SpatialGrid` — `output/spatial.py:115`

*Produces* (Tasks 2–4 depend on exactly these names):
- `hydrofragments.output.riverscape_export.RiverscapeExportBundle(landform, hydroperiod_classes, zone_result)` frozen, plus `RiverscapeExportBundle.from_zoning(zone_result, landform) -> RiverscapeExportBundle`
- `hydrofragments.spatial.zones.zones_from_riverscape_with_landform(stats, *, drainage, config, reach_labels, reach_keys, upstr_darea, geobox, transform, pixel_m=30.0, years=None) -> tuple[ZoneResult, LandformResult]`
- `hydrofragments.workflow._resolve_zone_result(stats, drainage_gdf, *, config, years, pixel_m, timings) -> tuple[ZoneResult | None, RiverscapeExportBundle | None, tuple[str, ...]]`

### Why the bundle is built in `workflow`, not in `zones`

`zones_from_riverscape_with_landform` returns the two science objects and nothing else, so `hydrofragments/spatial/` keeps knowing only about zoning. `workflow._resolve_zone_result` — which already owns the `off`/`auto`/`required` mode table — is the one place that knows whether *this* row actually took the riverscape path, so it is the one place that can honestly decide whether a bundle exists. `mode="off"`, the `auto` occurrence fallbacks, and the no-`ZoneResult` rows all return `bundle=None`.

---

- [ ] **Step 1: Write the failing bundle tests**

Create `tests/output/test_riverscape_export_bundle.py`:

```python
"""``RiverscapeExportBundle``: the slim retention type Phase 6b exports from.

The bundle carries no science of its own. Its whole job is to hold the
``LandformResult`` that Phase 6a discarded, alongside the exact hydroperiod
array ``combine_zones`` consumed -- recovered from the crosstab rather than
re-classified (spec 6b section 3.1).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("rioxarray")

from hydrofragments.output.riverscape_export import RiverscapeExportBundle
from hydrofragments.riverscape.riverine import FloodplainEnvelope
from hydrofragments.riverscape.pipeline import LandformResult
from hydrofragments.spatial.zones import build_zones, combine_zones

SHAPE = (3, 4)


def _landform_result(landform, *, bridged=None, provenance=None, channel_bridges=None):
    """Compact LandformResult builder.

    Same shape as ``tests/spatial/test_zones_from_riverscape.py::_landform_result``
    (that module is not importable from here -- ``tests/spatial`` has no
    ``__init__.py`` -- so the pattern is repeated, not shared).
    """
    landform = np.asarray(landform, dtype=np.uint8)
    zeros = np.zeros(landform.shape, dtype=np.uint8)
    return LandformResult(
        landform=landform,
        channel_source=zeros.copy(),
        channel_confidence=np.full(landform.shape, 255, dtype=np.uint8),
        rule_id=zeros.copy(),
        rem=np.zeros(landform.shape, dtype=np.float32),
        bridged_mask=(
            np.zeros(landform.shape, dtype=bool)
            if bridged is None
            else np.asarray(bridged, dtype=bool)
        ),
        channel_bridges=channel_bridges,
        unbridged=(),
        envelope=FloodplainEnvelope(
            a=2.0,
            b=0.0,
            n_bins=1,
            bin_log_a_mid=(3.7,),
            bin_rem_quantile=(2.0,),
            bin_counts=(90,),
            degraded_reasons=(),
        ),
        provenance=dict(provenance or {"years": (2019, 2021)}),
        degraded_reasons=(),
    )


def _riverscape_pair():
    landform = np.array(
        [
            [1, 2, 2, 0],
            [2, 2, 2, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    hydroperiod = np.array(
        [
            [1, 1, 2, 0],
            [1, 2, 3, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    zone_result = combine_zones(landform, hydroperiod, source="ga_ls_wo_fq_myear_3")
    return zone_result, hydroperiod, _landform_result(landform)


def test_from_zoning_recovers_the_exact_hydroperiod_array_from_the_crosstab() -> None:
    """MUTANT: deriving hydroperiod any other way (re-running
    classify_hydroperiod, or ``crosstab // 10``) breaks this equality.
    Code 4 on the bridged in-channel pixel is the discriminating case: it
    only exists because ``combine_zones`` was handed an unobserved mask.
    """
    zone_result, hydroperiod, landform = _riverscape_pair()

    bundle = RiverscapeExportBundle.from_zoning(zone_result, landform)

    assert bundle.hydroperiod_classes.dtype == np.uint8
    np.testing.assert_array_equal(bundle.hydroperiod_classes, hydroperiod)
    assert int(bundle.hydroperiod_classes[2, 0]) == 4


def test_bundle_holds_the_landform_and_zone_result_by_identity() -> None:
    zone_result, _hydroperiod, landform = _riverscape_pair()

    bundle = RiverscapeExportBundle.from_zoning(zone_result, landform)

    assert bundle.zone_result is zone_result
    assert bundle.landform is landform


def test_bundle_is_frozen() -> None:
    zone_result, _hydroperiod, landform = _riverscape_pair()
    bundle = RiverscapeExportBundle.from_zoning(zone_result, landform)

    with pytest.raises(Exception):
        bundle.zone_result = zone_result  # type: ignore[misc]


def test_occurrence_zone_result_is_rejected() -> None:
    """MUTANT: accepting an occurrence ZoneResult would let an occurrence run
    write riverscape evidence rasters with no landform behind them."""
    occurrence = build_zones(
        np.full(SHAPE, 80.0),
        max_wet_mask=np.ones(SHAPE, dtype=bool),
        valid_count=np.full(SHAPE, 40),
    )
    assert occurrence.mode == "occurrence"

    with pytest.raises(ValueError, match="riverscape"):
        RiverscapeExportBundle.from_zoning(
            occurrence, _landform_result(np.zeros(SHAPE, dtype=np.uint8))
        )


def test_shape_mismatch_between_landform_and_hydroperiod_is_rejected() -> None:
    zone_result, _hydroperiod, _landform = _riverscape_pair()

    with pytest.raises(ValueError, match="shape"):
        RiverscapeExportBundle(
            landform=_landform_result(np.zeros((2, 2), dtype=np.uint8)),
            hydroperiod_classes=np.zeros(SHAPE, dtype=np.uint8),
            zone_result=zone_result,
        )
```

Run:

```
python -m pytest tests/output/test_riverscape_export_bundle.py -q
```

Expected: **FAIL** — `ModuleNotFoundError: No module named 'hydrofragments.output.riverscape_export'`.

---

- [ ] **Step 2: Create `hydrofragments/output/riverscape_export.py`**

```python
"""The slim bundle that carries riverscape science from zoning to export.

Phase 6a's ``zones_from_riverscape`` discarded its ``LandformResult`` the
moment ``combine_zones`` had run. Phase 6b needs those arrays again to write
``riverscape_evidence`` rasters, the ``channel_bridges`` layer, and the
manifest's ``zoning`` section -- and re-deriving them would mean a second
DEM / Fractional Cover / Waterbodies load (spec 6b section 1.1). This module
owns the one type that carries them across the
``workflow`` -> ``finalize`` seam.

It lives under ``hydrofragments/output/`` rather than
``hydrofragments/spatial/`` on purpose: ``spatial`` stays zoning-only, and
the riverscape -> spatial import boundary (spec section 1.3) is untouched --
``riverscape`` imports nothing from here, and ``spatial.zones`` does not
import this module either. Only ``workflow`` and ``output.finalize`` do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydrofragments.riverscape.pipeline import LandformResult
from hydrofragments.spatial.zones import ZoneResult


@dataclass(frozen=True)
class RiverscapeExportBundle:
    """Landform + hydroperiod + zones for one successful riverscape run.

    Produced only on a genuinely successful riverscape zoning path -- never
    for ``riverscape.mode="off"``, and never when ``auto`` falls back to
    occurrence zoning (spec section 3.1).
    """

    landform: LandformResult
    hydroperiod_classes: np.ndarray
    zone_result: ZoneResult

    def __post_init__(self) -> None:
        if self.zone_result.mode != "riverscape":
            raise ValueError(
                "RiverscapeExportBundle requires a riverscape ZoneResult"
            )
        if np.shape(self.hydroperiod_classes) != np.shape(self.landform.landform):
            raise ValueError(
                "hydroperiod_classes must share the landform shape"
            )

    @classmethod
    def from_zoning(
        cls,
        zone_result: ZoneResult,
        landform: LandformResult,
    ) -> "RiverscapeExportBundle":
        """Build a bundle, recovering hydroperiod from the crosstab.

        ``combine_zones`` writes ``crosstab = landform * 10 + hydroperiod``
        with ``0`` outside the zoned extent, and validates that the two
        layers agree on that extent
        (``hydrofragments/spatial/zones.py``). Inside, hydroperiod is one of
        ``{1, 2, 3, 4}``, so ``crosstab % 10`` returns it exactly; outside,
        ``crosstab`` is ``0`` and so is hydroperiod. That identity is why
        this never calls ``classify_hydroperiod`` a second time (spec
        section 3.1) -- re-classifying would also silently drop code 4 on
        bridged pixels, which only exists because the first classification
        was handed an ``unobserved_mask``.
        """
        if zone_result.crosstab is None:
            raise ValueError(
                "RiverscapeExportBundle requires a ZoneResult crosstab"
            )
        crosstab = np.asarray(zone_result.crosstab, dtype=np.uint16)
        hydroperiod_classes = (crosstab % 10).astype(np.uint8)
        return cls(
            landform=landform,
            hydroperiod_classes=hydroperiod_classes,
            zone_result=zone_result,
        )


__all__ = ["RiverscapeExportBundle"]
```

Run:

```
python -m pytest tests/output/test_riverscape_export_bundle.py -q
```

Expected: **PASS** (5 tests).

---

- [ ] **Step 3: Write the failing `with_landform` tests**

Append to `tests/spatial/test_zones_from_riverscape.py` (the module's existing helpers `_zones`, `_landform_result`, `_install_build_landform`, `_stats`, `_drainage`, `_config`, `SHAPE`, `TRANSFORM`, `PIXEL_M` are reused verbatim):

```python
# --------------------------------------------------------------------------
# Phase 6b: the landform-returning companion (spec 6b section 3.2)
# --------------------------------------------------------------------------


def _zones_with_landform(monkeypatch, result: LandformResult, **overrides):
    calls = _install_build_landform(monkeypatch, result)
    kwargs = {
        "drainage": _drainage(),
        "config": _config(),
        "reach_labels": np.ones(SHAPE, dtype=np.int32),
        "reach_keys": {1: "1"},
        "upstr_darea": {1: 5000.0},
        "geobox": SimpleNamespace(),
        "transform": TRANSFORM,
        "pixel_m": PIXEL_M,
        "years": (2019, 2021),
    }
    stats = overrides.pop("stats", None) or _stats()
    kwargs.update(overrides)
    return zones_from_riverscape_with_landform(stats, **kwargs), calls


def test_with_landform_returns_the_same_zone_result_as_the_wrapper(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    (paired_zones, _paired_landform), _calls = _zones_with_landform(
        monkeypatch, _landform_result(landform)
    )
    wrapper_zones, _calls_again = _zones(monkeypatch, _landform_result(landform))

    np.testing.assert_array_equal(paired_zones.mask, wrapper_zones.mask)
    np.testing.assert_array_equal(paired_zones.crosstab, wrapper_zones.crosstab)
    assert paired_zones.mode == wrapper_zones.mode == "riverscape"
    assert paired_zones.source == wrapper_zones.source


def test_with_landform_attaches_the_zone_grid_to_the_landform_result(monkeypatch) -> None:
    """MUTANT: returning the landform without calling ``with_grid`` leaves
    ``grid=None``, and Task 3's evidence writer then has no grid to write
    against (spec 6b section 2 row 11)."""
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    (zone_result, landform_result), _calls = _zones_with_landform(
        monkeypatch, _landform_result(landform)
    )

    assert zone_result.grid is not None
    assert landform_result.grid is not None
    assert landform_result.grid == zone_result.grid


def test_with_landform_returns_the_landform_build_landform_produced(monkeypatch) -> None:
    """MUTANT: discarding the landform again (returning a fresh/empty one)
    loses ``provenance``, which Task 2's manifest section copies verbatim."""
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    source = _landform_result(landform)
    (_zone_result, landform_result), _calls = _zones_with_landform(monkeypatch, source)

    assert landform_result.provenance == source.provenance
    np.testing.assert_array_equal(landform_result.landform, source.landform)
    np.testing.assert_array_equal(landform_result.rem, source.rem)


def test_wrapper_still_returns_a_bare_zone_result(monkeypatch) -> None:
    """API stability (spec 6b section 3.2): occurrence-shaped callers keep
    a single return value."""
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    result, _calls = _zones(monkeypatch, _landform_result(landform))

    assert isinstance(result, ZoneResult)
    assert not isinstance(result, tuple)
```

Also extend the module's imports:

```python
from hydrofragments.spatial.zones import (
    ZoneResult,
    zones_from_riverscape,
    zones_from_riverscape_with_landform,
)
```

Run:

```
python -m pytest tests/spatial/test_zones_from_riverscape.py -q
```

Expected: **FAIL** — `ImportError: cannot import name 'zones_from_riverscape_with_landform'`.

---

- [ ] **Step 4: Split `zones_from_riverscape` in `hydrofragments/spatial/zones.py`**

Extend the existing import at `zones.py:31`:

```python
from hydrofragments.riverscape.pipeline import build_landform, with_grid
```

Replace the body of `zones_from_riverscape` (`zones.py:209-276`) with the companion plus a wrapper. The docstring moves to the companion; the wrapper keeps a short one pointing at it.

```python
def zones_from_riverscape_with_landform(
    stats: "WoStatistics",
    *,
    drainage: Any,
    config: Any,
    reach_labels: np.ndarray,
    reach_keys: Any,
    upstr_darea: Any,
    geobox: Any,
    transform: Any,
    pixel_m: float = 30.0,
    years: tuple[int, int] | None = None,
) -> tuple[ZoneResult, "LandformResult"]:
    """Build a riverscape ``ZoneResult`` AND return the landform behind it.

    Four steps, no new science (parent spec section 5): the observed-wet
    domain comes from ``wet_domain`` over the same ``stats`` the occurrence
    path reads; ``build_landform`` produces the landform layer; the
    hydroperiod layer is classified over that same domain with the
    landform's bridged mask supplied as ``unobserved_mask``; and
    ``combine_zones`` derives the crosstab and legacy zone mask from the two
    layers.

    Bridged pixels come out landform 1 / hydroperiod 4, so legacy Zone 1
    includes them. They lie strictly outside ``domain``
    (``bridging.bridge_gaps`` and ``riverscape.pipeline.build_landform`` both
    enforce that), which is what lets ``classify_hydroperiod`` accept them as
    ``unobserved_mask`` -- it rejects an overlap.

    ``reach_labels``/``reach_keys``/``upstr_darea`` are built by the caller
    (``hydrofragments.riverscape`` must not import ``hydrofragments.spatial``,
    so it cannot build them itself) and forwarded unchanged. The returned
    ``ZoneResult`` is stamped with ``source=stats.product``, matching
    ``zones_from_wo_statistics``'s convention, and carries a grid attached
    from ``stats.frequency``.

    Phase 6b (spec 6b section 3.2) added the second return value. Phase 6a
    dropped the ``LandformResult`` here, which forced any exporter to reload
    DEM / Fractional Cover / Waterbodies to get it back. The same grid that
    lands on the ``ZoneResult`` is attached to the ``LandformResult`` via
    ``riverscape.pipeline.with_grid`` -- the documented seam for handing an
    output-layer ``SpatialGrid`` back across the import boundary.
    """
    resolved_years = _resolve_years(stats, years)
    domain = wet_domain(stats, min_valid_obs=config.validity.min_valid_obs)
    frequency = np.asarray(stats.frequency, dtype=float)

    landform_result = build_landform(
        domain,
        frequency,
        drainage,
        reach_labels,
        reach_keys,
        upstr_darea,
        geobox=geobox,
        transform=transform,
        pixel_m=pixel_m,
        cfg=config.riverscape,
        years=resolved_years,
    )

    hydroperiod = classify_hydroperiod(
        frequency,
        domain,
        t_persist=config.zones.t_persist,
        t_season=config.zones.t_season,
        unobserved_mask=landform_result.bridged_mask,
    )

    result = combine_zones(
        landform_result.landform,
        hydroperiod.classes,
        source=stats.product,
        degraded_reasons=landform_result.degraded_reasons,
    )
    zone_result = _attach_grid(result, stats.frequency)
    return zone_result, with_grid(landform_result, zone_result.grid)


def zones_from_riverscape(
    stats: "WoStatistics",
    *,
    drainage: Any,
    config: Any,
    reach_labels: np.ndarray,
    reach_keys: Any,
    upstr_darea: Any,
    geobox: Any,
    transform: Any,
    pixel_m: float = 30.0,
    years: tuple[int, int] | None = None,
) -> ZoneResult:
    """Build a riverscape ``ZoneResult`` from DEA stats + drainage context.

    Thin wrapper over :func:`zones_from_riverscape_with_landform`, kept for
    API stability (spec 6b section 3.2): callers that only need zoning keep
    a single return value. ``workflow`` uses the landform-returning form.
    """
    zone_result, _landform = zones_from_riverscape_with_landform(
        stats,
        drainage=drainage,
        config=config,
        reach_labels=reach_labels,
        reach_keys=reach_keys,
        upstr_darea=upstr_darea,
        geobox=geobox,
        transform=transform,
        pixel_m=pixel_m,
        years=years,
    )
    return zone_result
```

Extend `__all__` at `zones.py:341`:

```python
__all__ = [
    "ZoneResult",
    "build_zones",
    "combine_zones",
    "zones_from_riverscape",
    "zones_from_riverscape_with_landform",
    "zones_from_wo_statistics",
]
```

And re-export from `hydrofragments/spatial/__init__.py` (add to both the import block near line 26 and `__all__` near line 47):

```python
    zones_from_riverscape,
    zones_from_riverscape_with_landform,
```

Run:

```
python -m pytest tests/spatial/test_zones_from_riverscape.py -q
```

Expected: **PASS** (existing 7 + new 4 = 11 tests).

---

- [ ] **Step 5: Write the failing `_resolve_zone_result` bundle test**

Modify `tests/integration/test_riverscape_modes.py`.

First, adapt the shared helpers. Replace `_install_riverscape` (currently at lines 298-307) and add two helpers:

```python
from dataclasses import replace

from hydrofragments.output.riverscape_export import RiverscapeExportBundle
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.riverscape.pipeline import LandformResult, with_grid
from hydrofragments.riverscape.riverine import FloodplainEnvelope


def _landform_result(landform=None, *, provenance=None, channel_bridges=None):
    """Compact LandformResult for the stubbed riverscape path.

    Mirrors ``tests/spatial/test_zones_from_riverscape.py::_landform_result``
    (not importable across test packages -- ``tests/spatial`` has no
    ``__init__.py``).
    """
    values = (
        np.full(_SHAPE, 1, dtype=np.uint8)
        if landform is None
        else np.asarray(landform, dtype=np.uint8)
    )
    zeros = np.zeros(_SHAPE, dtype=np.uint8)
    return LandformResult(
        landform=values,
        channel_source=np.full(_SHAPE, 1, dtype=np.uint8),
        channel_confidence=np.full(_SHAPE, 3, dtype=np.uint8),
        rule_id=zeros.copy(),
        rem=np.zeros(_SHAPE, dtype=np.float32),
        bridged_mask=np.zeros(_SHAPE, dtype=bool),
        channel_bridges=channel_bridges,
        unbridged=(),
        envelope=FloodplainEnvelope(
            a=2.0,
            b=0.0,
            n_bins=1,
            bin_log_a_mid=(3.7,),
            bin_rem_quantile=(2.0,),
            bin_counts=(16,),
            degraded_reasons=(),
        ),
        provenance=dict(provenance or _PROVENANCE),
        degraded_reasons=(),
    )


_PROVENANCE = {
    "channel_ruleset_version": "1.0.0",
    "waterbody_ruleset_version": "1.0.0",
    "riverine_ruleset_version": "1.0.0",
    "dem_product": "ga_srtm_dem1sv1_0",
    "dem_band": "elevation",
    "fc_product": "ga_ls_fc_pc_cyear_3",
    "fc_bands": ("bs_pc_50", "pv_pc_50", "npv_pc_50"),
    "waterbodies_source": "dea_waterbodies",
    "years": (2020, 2020),
    "bridge_enabled": False,
    "bare_threshold_pct": 30.0,
    "envelope": {"a": 2.0, "b": 0.0, "n_bins": 1},
    "pixel_counts": {
        "domain": 16,
        "water_seed": 16,
        "observed_channel": 16,
        "bridged_channel": 0,
        "in_channel": 16,
        "off_channel_riverine": 0,
        "non_riverine": 0,
    },
    "bridge_counts_by_cause": {},
    "unbridged_counts_by_reason": {},
    "reach_counts": {"total": 1, "line_fallback": 0, "multithread": 0},
}


def _install_riverscape(monkeypatch, recorder: _Recorder, *, raises=None, landform=None):
    def fake_zones_from_riverscape_with_landform(stats, **kwargs):
        recorder.riverscape_calls.append(kwargs)
        if raises is not None:
            raise raises
        grid = SpatialGrid.from_dataarray(stats.frequency, require_georeference=True)
        zone_result = replace(_riverscape_zone_result(), grid=grid)
        return zone_result, with_grid(landform or _landform_result(), grid)

    monkeypatch.setattr(
        workflow_module,
        "zones_from_riverscape_with_landform",
        fake_zones_from_riverscape_with_landform,
    )


def _resolve(*args, **kwargs):
    """2-tuple view of the Phase 6b 3-tuple, for the mode-table cases that
    do not care about the export bundle."""
    zone_result, _bundle, reasons = _resolve_zone_result(*args, **kwargs)
    return zone_result, reasons
```

Then, in these six existing tests, change the call from `_resolve_zone_result(` to `_resolve(`:

- `test_mode_off_uses_occurrence_zoning`
- `test_mode_off_without_stats_returns_no_zones`
- `test_auto_without_drainage_degrades_to_occurrence_with_a_reason`
- `test_auto_with_drainage_uses_riverscape_zoning`
- `test_auto_with_source_failure_degrades_to_occurrence_with_a_reason`
- `test_auto_without_stats_reports_no_stats_and_no_zones`

The `pytest.raises(...)` cases (`test_required_without_drainage_raises`, `test_required_without_stats_raises`, `test_required_propagates_a_source_failure`, `test_unexpected_exceptions_are_never_swallowed_as_a_degrade`, `test_resolve_zone_result_rejects_unreprojected_wgs84_at_build_landform`, `test_resolve_zone_result_passes_projected_drainage_to_build_landform`) unpack nothing and stay as they are. `wrapped_resolve_zone_result` in `test_dea_planning_carves_out_riverscape_timing` passes the whole tuple through and stays as it is.

In `test_unexpected_exceptions_are_never_swallowed_as_a_degrade`, change the monkeypatch target:

```python
    monkeypatch.setattr(
        workflow_module,
        "zones_from_riverscape_with_landform",
        exploding_zones_from_riverscape,
    )
```

Finally add the new bundle cases:

```python
# --------------------------------------------------------------------------
# Phase 6b: the export bundle rides beside the ZoneResult (spec 6b section 3.2)
# --------------------------------------------------------------------------


def test_riverscape_path_produces_an_export_bundle(monkeypatch) -> None:
    recorder = _Recorder()
    _install_riverscape(monkeypatch, recorder)
    monkeypatch.setattr(
        workflow_module, "_frequency_geobox", lambda frequency: SimpleNamespace()
    )

    zone_result, bundle, reasons = _resolve_zone_result(
        _stats(),
        _drainage(),
        config=_config("auto"),
        years=(2020, 2020),
        pixel_m=30.0,
        timings={},
    )

    assert reasons == ()
    assert isinstance(bundle, RiverscapeExportBundle)
    assert bundle.zone_result is zone_result
    assert bundle.landform.grid is not None
    assert bundle.hydroperiod_classes.shape == _SHAPE


def test_mode_off_produces_no_bundle(monkeypatch) -> None:
    """MUTANT: building a bundle on the occurrence path would let a
    ``mode="off"`` run write riverscape evidence rasters."""
    zone_result, bundle, _reasons = _resolve_zone_result(
        _stats(),
        _drainage(),
        config=_config("off"),
        years=(2020, 2020),
        pixel_m=30.0,
        timings={},
    )

    assert zone_result is not None
    assert bundle is None


def test_auto_occurrence_fallback_produces_no_bundle(monkeypatch) -> None:
    recorder = _Recorder()
    _install_riverscape(
        monkeypatch, recorder, raises=RiverscapeSourceUnavailable("DEM unavailable")
    )
    monkeypatch.setattr(
        workflow_module, "_frequency_geobox", lambda frequency: SimpleNamespace()
    )

    zone_result, bundle, reasons = _resolve_zone_result(
        _stats(),
        _drainage(),
        config=_config("auto"),
        years=(2020, 2020),
        pixel_m=30.0,
        timings={},
    )

    assert zone_result is not None
    assert zone_result.mode == "occurrence"
    assert bundle is None
    assert reasons == ("riverscape_source_unavailable",)


def test_no_zone_result_rows_produce_no_bundle() -> None:
    zone_result, bundle, reasons = _resolve_zone_result(
        None,
        _drainage(),
        config=_config("auto"),
        years=(2020, 2020),
        pixel_m=30.0,
        timings={},
    )

    assert (zone_result, bundle) == (None, None)
    assert reasons == ("riverscape_no_stats",)
```

Run:

```
python -m pytest tests/integration/test_riverscape_modes.py -q
```

Expected: **FAIL** — `ValueError: not enough values to unpack (expected 3, got 2)` on the new tests, plus `AttributeError: <module 'hydrofragments.workflow'> has no attribute 'zones_from_riverscape_with_landform'` from `_install_riverscape`.

---

- [ ] **Step 6: Make `_resolve_zone_result` return the bundle**

In `hydrofragments/workflow.py`, change the zones import block (currently lines 125-129):

```python
from hydrofragments.output.riverscape_export import RiverscapeExportBundle
from hydrofragments.spatial.zones import (
    ZoneResult,
    zones_from_riverscape_with_landform,
    zones_from_wo_statistics,
)
```

(`zones_from_riverscape` is no longer imported here — the workflow always uses the landform-returning form, and leaving the old name bound would let a stale monkeypatch silently no-op.)

Change the signature and every return of `_resolve_zone_result` (`workflow.py:443-540`):

```python
def _resolve_zone_result(
    stats: Any,
    drainage_gdf: Any,
    *,
    config: HydroConfig,
    years: tuple[int, int],
    pixel_m: float,
    timings: dict[str, float],
) -> tuple["ZoneResult | None", "RiverscapeExportBundle | None", tuple[str, ...]]:
    """Resolve this run's ``ZoneResult`` per the Phase 6a mode table (spec 6a section 6).

    Returns ``(zone_result, riverscape_bundle, degraded_reasons)``. The
    reasons are already merged onto ``zone_result`` when there is one; they
    are also returned separately so the ``stats is None`` case can still
    report WHY riverscape zoning was skipped, which has no ``ZoneResult`` to
    carry it (Phase 6b writes these into the manifest's ``zoning`` section).

    ``riverscape_bundle`` is non-``None`` on exactly one branch: the
    successful riverscape path. ``off``, both ``auto`` occurrence fallbacks,
    and every ``None``-zones row return ``None`` (spec 6b section 3.1) --
    an occurrence run must never be able to write riverscape evidence.

    | mode | drainage | sources | result |
    |---|---|---|---|
    | ``off`` | any | any | occurrence if stats, else ``None`` |
    | ``auto`` | missing | -- | occurrence + ``riverscape_no_drainage`` |
    | ``auto`` | present | ok | riverscape zoning + bundle |
    | ``auto`` | present | source failure | occurrence + ``riverscape_source_unavailable`` |
    | ``auto`` | present | no stats | ``None`` + ``riverscape_no_stats`` |
    | ``required`` | missing | -- | raise |
    | ``required`` | present | source failure | propagate |
    | ``required`` | present | ok | riverscape zoning + bundle |

    Only ``RiverscapeSourceUnavailable`` and the declared no-drainage /
    no-stats cases map to the ``auto`` fallback. Every other exception
    propagates, so a genuine bug is never laundered into a degraded run.

    ``timings["riverscape"]`` records the riverscape branch's own wall time
    and is absent when the branch is skipped;
    :func:`analyze_from_dea` subtracts it from ``dea_planning`` so the
    manifest's phases stay disjoint.
    """
    mode = config.riverscape.mode

    if mode == "off":
        if stats is None:
            return None, None, ()
        return zones_from_wo_statistics(stats, config=config), None, ()

    if drainage_gdf is None:
        if mode == "required":
            raise RiverscapeSourceUnavailable(
                "riverscape.mode='required' needs drainage lines; none were supplied"
            )
        if stats is None:
            return None, None, ("riverscape_no_drainage",)
        reasons = ("riverscape_no_drainage",)
        return (
            _with_reasons(zones_from_wo_statistics(stats, config=config), reasons),
            None,
            reasons,
        )

    if stats is None:
        if mode == "required":
            raise RiverscapeSourceUnavailable(
                "riverscape.mode='required' needs DEA WO statistics; none were available"
            )
        return None, None, ("riverscape_no_stats",)

    started = time.perf_counter()
    try:
        projected_drainage = _project_drainage_to_frequency_grid(
            drainage_gdf, stats.frequency
        )
        reach_labels, reach_keys, upstr_darea = _riverscape_reach_context(
            projected_drainage, stats.frequency, buffer_m=_RIVERSCAPE_REACH_BUFFER_M
        )
        zone_result, landform_result = zones_from_riverscape_with_landform(
            stats,
            drainage=projected_drainage,
            config=config,
            reach_labels=reach_labels,
            reach_keys=reach_keys,
            upstr_darea=upstr_darea,
            geobox=_frequency_geobox(stats.frequency),
            transform=_raster_transform(
                np.asarray(stats.frequency["y"].values, dtype=float),
                np.asarray(stats.frequency["x"].values, dtype=float),
            ),
            pixel_m=pixel_m,
            years=years,
        )
    except RiverscapeSourceUnavailable:
        timings["riverscape"] = time.perf_counter() - started
        if mode == "required":
            raise
        reasons = ("riverscape_source_unavailable",)
        return (
            _with_reasons(zones_from_wo_statistics(stats, config=config), reasons),
            None,
            reasons,
        )
    timings["riverscape"] = time.perf_counter() - started
    bundle = RiverscapeExportBundle.from_zoning(zone_result, landform_result)
    return zone_result, bundle, tuple(zone_result.degraded_reasons)
```

Update the single production call site in `analyze_from_dea` (`workflow.py:616-623`) so the module still imports and runs; full plumbing lands in Task 4:

```python
    zone_result, riverscape_bundle, riverscape_degraded_reasons = _resolve_zone_result(
        stats,
        drainage_gdf,
        config=resolved_config,
        years=(start.year, end.year),
        pixel_m=resolution,
        timings=timings,
    )
```

Update the surrounding comment block (`workflow.py:628-629`) to drop the now-stale "6b only has to produce it" note:

```python
    # riverscape_degraded_reasons feeds the manifest's "zoning" section when
    # there is no ZoneResult to carry them; riverscape_bundle carries the
    # landform science to finalize (Phase 6b spec section 3.7).
```

Run:

```
python -m pytest tests/integration/test_riverscape_modes.py tests/spatial/test_zones_from_riverscape.py tests/output/test_riverscape_export_bundle.py -q
```

Expected: **PASS**.

---

- [ ] **Step 7: Full-suite regression + commit**

```
python -m pytest tests/ -q -x
python -m pytest tests/riverscape/test_riverscape_import_boundary.py -q
```

Expected: **PASS**. The import-boundary AST scan must still be green — Task 1 added no import to any `hydrofragments/riverscape/` module.

```
git add hydrofragments/output/riverscape_export.py hydrofragments/spatial/zones.py hydrofragments/spatial/__init__.py hydrofragments/workflow.py tests/output/test_riverscape_export_bundle.py tests/spatial/test_zones_from_riverscape.py tests/integration/test_riverscape_modes.py
git commit -m "Retain LandformResult past zoning behind a RiverscapeExportBundle (6b Task 1)"
```

---

## Task 2: Manifest `zoning` builder

**Files:**
- Modify: `hydrofragments/output/manifest.py`, `hydrofragments/output/bundle.py`
- Test: `tests/output/test_manifest_zoning.py` (create)

**Interfaces:**

*Consumes* (from Task 1):
- `RiverscapeExportBundle(landform, hydroperiod_classes, zone_result)`
- `LandformResult.provenance` keys exactly as listed in Global Constraints
- `hydrofragments.output.manifest._hash_array(array) -> str` (`manifest.py:202-218`) — the same digest helper `build_dea_provenance` uses (L8)

*Produces* (Task 3/4 depend on these):
- `hydrofragments.output.manifest.build_zoning_section(zone_result, bundle, *, pixel_m, configured_mode, workflow_reasons) -> dict[str, object]`
- `build_run_manifest(..., zoning: Mapping[str, object] | None = None)` → sets `manifest["zoning"]`
- `BundleTransaction.finalize(..., zoning: Mapping[str, object] | None = None)` passthrough

### Type imports and why they are guarded

`build_zoning_section` takes `zone_result` and `bundle` as `Any` at runtime, with real annotations only under `TYPE_CHECKING`. `manifest.py` sits below `output/bundle.py` and `output/finalize.py` in the import graph, and `RiverscapeExportBundle` imports `spatial.zones`, which imports `output.spatial`. A runtime import here would add a new edge across three packages for no benefit — the function only reads attributes.

---

- [ ] **Step 1: Write the failing zoning-section tests**

Create `tests/output/test_manifest_zoning.py`:

```python
"""``build_zoning_section``: the manifest's ``zoning`` object (spec 6b section 3.3).

Three shapes, one function: full riverscape from a bundle, the thin
occurrence subset from a bare ``ZoneResult``, and a reasons-only record when
``auto`` returned no zones at all. Provenance keys are copied verbatim from
``LandformResult.provenance`` -- this module asserts the literal key names,
because Phase 7 and Phase 8 both read them.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("rioxarray")

import geopandas as gpd
from shapely.geometry import LineString

from hydrofragments.output.manifest import _hash_array, build_zoning_section
from hydrofragments.output.riverscape_export import RiverscapeExportBundle
from hydrofragments.riverscape.riverine import FloodplainEnvelope
from hydrofragments.riverscape.pipeline import LandformResult
from hydrofragments.spatial.zones import build_zones, combine_zones

PIXEL_M = 30.0
CRS = "EPSG:3577"

_PROVENANCE = {
    "channel_ruleset_version": "c-1.2.3",
    "waterbody_ruleset_version": "w-2.0.0",
    "riverine_ruleset_version": "r-3.1.0",
    "dem_product": "ga_srtm_dem1sv1_0",
    "dem_band": "elevation",
    "fc_product": "ga_ls_fc_pc_cyear_3",
    "fc_bands": ("bs_pc_50", "pv_pc_50", "npv_pc_50"),
    "waterbodies_source": "dea_waterbodies",
    "years": (2019, 2021),
    "bridge_enabled": True,
    "bare_threshold_pct": 30.0,
    "envelope": {"a": 2.0, "b": 0.0, "n_bins": 1},
    "pixel_counts": {
        "domain": 6,
        "water_seed": 4,
        "observed_channel": 2,
        "bridged_channel": 1,
        "in_channel": 3,
        "off_channel_riverine": 4,
        "non_riverine": 2,
    },
    "bridge_counts_by_cause": {"vegetated": 1},
    "unbridged_counts_by_reason": {},
    "reach_counts": {"total": 1, "line_fallback": 0, "multithread": 0},
}


def _bridges(lengths=(120.0, 80.0)):
    return gpd.GeoDataFrame(
        {
            "gap_id": list(range(len(lengths))),
            "reach_ids": ["1,2"] * len(lengths),
            "length_m": list(lengths),
            "cumulative_cost": [1.0] * len(lengths),
            "cost_per_m": [0.01] * len(lengths),
            "upstream_width_m": [30.0] * len(lengths),
            "downstream_width_m": [30.0] * len(lengths),
            "bridge_confidence": [2] * len(lengths),
            "gap_cause": ["vegetated"] * len(lengths),
        },
        geometry=[LineString([(0.0, 0.0), (length, 0.0)]) for length in lengths],
        crs=CRS,
    )


def _landform_result(landform, *, bridged, channel_bridges, degraded_reasons=()):
    landform = np.asarray(landform, dtype=np.uint8)
    zeros = np.zeros(landform.shape, dtype=np.uint8)
    return LandformResult(
        landform=landform,
        channel_source=zeros.copy(),
        channel_confidence=np.full(landform.shape, 255, dtype=np.uint8),
        rule_id=zeros.copy(),
        rem=np.zeros(landform.shape, dtype=np.float32),
        bridged_mask=np.asarray(bridged, dtype=bool),
        channel_bridges=channel_bridges,
        unbridged=(),
        envelope=FloodplainEnvelope(
            a=2.0,
            b=0.0,
            n_bins=1,
            bin_log_a_mid=(3.7,),
            bin_rem_quantile=(2.0,),
            bin_counts=(90,),
            degraded_reasons=(),
        ),
        provenance=dict(_PROVENANCE),
        degraded_reasons=tuple(degraded_reasons),
    )


def _riverscape_bundle(*, channel_bridges=None, degraded_reasons=("bare_threshold_fallback",)):
    landform = np.array(
        [
            [1, 2, 2, 0],
            [2, 2, 3, 0],
            [1, 3, 1, 0],
        ],
        dtype=np.uint8,
    )
    hydroperiod = np.array(
        [
            [1, 1, 2, 0],
            [2, 3, 1, 0],
            [4, 2, 1, 0],
        ],
        dtype=np.uint8,
    )
    bridged = np.zeros(landform.shape, dtype=bool)
    bridged[2, 0] = True
    zone_result = combine_zones(
        landform,
        hydroperiod,
        source="ga_ls_wo_fq_myear_3",
        degraded_reasons=degraded_reasons,
    )
    result = _landform_result(
        landform,
        bridged=bridged,
        channel_bridges=_bridges() if channel_bridges is None else channel_bridges,
        degraded_reasons=degraded_reasons,
    )
    return RiverscapeExportBundle.from_zoning(zone_result, result), zone_result


def _occurrence_zone_result():
    frequency = np.array(
        [
            [80.0, 30.0, 5.0, 0.0],
            [80.0, 30.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    return build_zones(
        frequency,
        max_wet_mask=frequency > 0,
        valid_count=np.full(frequency.shape, 40),
    )


# --------------------------------------------------------------------------
# Shape 1: full riverscape
# --------------------------------------------------------------------------


def test_riverscape_section_copies_every_provenance_key_verbatim() -> None:
    """MUTANT: renaming any provenance key (e.g. ``dem_product`` ->
    ``dem``) breaks Phase 7/8's contract and fails here (spec 6b section 3.3)."""
    bundle, _zone_result = _riverscape_bundle()

    section = build_zoning_section(
        bundle.zone_result,
        bundle,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=(),
    )

    assert section["mode"] == "riverscape"
    assert section["ruleset_versions"] == {
        "channel": "c-1.2.3",
        "waterbody": "w-2.0.0",
        "riverine": "r-3.1.0",
    }
    assert section["sources"] == {
        "dem_product": "ga_srtm_dem1sv1_0",
        "dem_band": "elevation",
        "fc_product": "ga_ls_fc_pc_cyear_3",
        "fc_bands": ["bs_pc_50", "pv_pc_50", "npv_pc_50"],
        "waterbodies_source": "dea_waterbodies",
        "years": [2019, 2021],
        "zone_source": "ga_ls_wo_fq_myear_3",
    }
    assert section["envelope"] == {"a": 2.0, "b": 0.0, "n_bins": 1}
    assert section["pixel_counts"] == _PROVENANCE["pixel_counts"]
    assert section["bridge_counts_by_cause"] == {"vegetated": 1}
    assert section["unbridged_counts_by_reason"] == {}
    assert section["reach_counts"] == {"total": 1, "line_fallback": 0, "multithread": 0}
    assert section["bridge_enabled"] is True
    assert section["bare_threshold_pct"] == 30.0


def test_riverscape_area_and_length_fields_use_the_documented_formulas() -> None:
    """bridged_length_m = sum(channel_bridges.length_m);
    bridged_area_m2 = count(bridged_mask) * pixel_m**2;
    non_riverine_area_m2 = pixel_counts.non_riverine * pixel_m**2."""
    bundle, _zone_result = _riverscape_bundle()

    section = build_zoning_section(
        bundle.zone_result,
        bundle,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=(),
    )

    assert section["bridged_length_m"] == pytest.approx(200.0)
    assert section["bridged_area_m2"] == pytest.approx(900.0)
    assert section["non_riverine_area_m2"] == pytest.approx(1800.0)


def test_riverscape_empty_bridges_report_zero_length() -> None:
    bundle, _zone_result = _riverscape_bundle(channel_bridges=_bridges(lengths=()))

    section = build_zoning_section(
        bundle.zone_result,
        bundle,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=(),
    )

    assert section["bridged_length_m"] == 0.0


def test_riverscape_domain_fields_use_provenance_and_the_shared_digest() -> None:
    """MUTANT: inventing a second hashing algorithm instead of reusing
    ``_hash_array`` (the helper behind ``dea_provenance.zone_mask_digest``)
    fails here (spec 6b section 3.3, decision L8)."""
    bundle, zone_result = _riverscape_bundle()

    section = build_zoning_section(
        zone_result, bundle, pixel_m=PIXEL_M, configured_mode="auto", workflow_reasons=()
    )

    assert section["domain_pixel_count"] == 6
    assert section["domain_digest"] == _hash_array(zone_result.mask)


def test_riverscape_degraded_reasons_come_from_the_zone_result() -> None:
    bundle, _zone_result = _riverscape_bundle(
        degraded_reasons=("bare_threshold_fallback", "envelope_single_bin")
    )

    section = build_zoning_section(
        bundle.zone_result,
        bundle,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=("ignored_because_zone_result_wins",),
    )

    assert section["degraded_reasons"] == [
        "bare_threshold_fallback",
        "envelope_single_bin",
    ]


def test_riverscape_section_is_json_serializable_with_allow_nan_false() -> None:
    """``manifest._json_bytes`` uses ``allow_nan=False``; tuples and numpy
    scalars must already be normalized by this builder."""
    bundle, _zone_result = _riverscape_bundle()

    section = build_zoning_section(
        bundle.zone_result,
        bundle,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=(),
    )

    json.dumps(section, allow_nan=False, sort_keys=True)


# --------------------------------------------------------------------------
# Shape 2: thin occurrence subset
# --------------------------------------------------------------------------


def test_occurrence_section_is_the_thin_subset_and_omits_riverscape_keys() -> None:
    """MUTANT: emitting riverscape keys as empty defaults instead of
    omitting them makes an occurrence run look like a degraded riverscape
    run (spec 6b section 3.3: 'prefer omit for absent science')."""
    zone_result = _occurrence_zone_result()

    section = build_zoning_section(
        zone_result, None, pixel_m=PIXEL_M, configured_mode="off", workflow_reasons=()
    )

    assert section["mode"] == "occurrence"
    assert section["zone_source"] == "occurrence"
    assert section["degraded_reasons"] == []
    assert section["domain_pixel_count"] == int(np.count_nonzero(zone_result.mask))
    assert section["domain_digest"] == _hash_array(zone_result.mask)
    for omitted in (
        "ruleset_versions",
        "sources",
        "envelope",
        "pixel_counts",
        "bridge_counts_by_cause",
        "unbridged_counts_by_reason",
        "reach_counts",
        "bridged_length_m",
        "bridged_area_m2",
        "non_riverine_area_m2",
    ):
        assert omitted not in section


def test_occurrence_fallback_reasons_ride_on_the_zone_result() -> None:
    from dataclasses import replace

    zone_result = replace(
        _occurrence_zone_result(),
        degraded_reasons=("riverscape_source_unavailable",),
    )

    section = build_zoning_section(
        zone_result,
        None,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=("riverscape_source_unavailable",),
    )

    assert section["mode"] == "occurrence"
    assert section["degraded_reasons"] == ["riverscape_source_unavailable"]


# --------------------------------------------------------------------------
# Shape 3: no ZoneResult at all
# --------------------------------------------------------------------------


def test_no_zone_result_records_configured_mode_and_workflow_reasons() -> None:
    section = build_zoning_section(
        None,
        None,
        pixel_m=PIXEL_M,
        configured_mode="auto",
        workflow_reasons=("riverscape_no_stats",),
    )

    assert section == {
        "mode": "auto",
        "degraded_reasons": ["riverscape_no_stats"],
    }


def test_no_zone_result_never_fabricates_a_domain_digest() -> None:
    """MUTANT: hashing an empty/zero array to fill ``domain_digest`` would
    publish a digest for a domain that was never computed."""
    section = build_zoning_section(
        None,
        None,
        pixel_m=PIXEL_M,
        configured_mode="required",
        workflow_reasons=(),
    )

    assert "domain_digest" not in section
    assert "domain_pixel_count" not in section
    assert section["degraded_reasons"] == []
```

Run:

```
python -m pytest tests/output/test_manifest_zoning.py -q
```

Expected: **FAIL** — `ImportError: cannot import name 'build_zoning_section'`.

---

- [ ] **Step 2: Implement `build_zoning_section` in `hydrofragments/output/manifest.py`**

Add to the imports at the top of the module:

```python
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hydrofragments.output.riverscape_export import RiverscapeExportBundle
    from hydrofragments.spatial.zones import ZoneResult
```

Insert after `build_dea_provenance` (i.e. after `manifest.py:258`):

```python
def _finite_float(value: object, *, field: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ManifestError(f"zoning.{field} must be finite")
    return number


def _bridged_length_m(bridges: Any) -> float:
    """Total bridged centreline length in metres; 0 for an empty/absent frame.

    ``LandformResult.channel_bridges`` is either
    ``bridging._empty_bridge_frame``'s typed-empty GeoDataFrame or a frame
    with one row per bridged gap. A stub may leave it ``None``.
    """
    if bridges is None or len(bridges) == 0:
        return 0.0
    return _finite_float(
        np.asarray(bridges["length_m"], dtype=float).sum(),
        field="bridged_length_m",
    )


def build_zoning_section(
    zone_result: "ZoneResult | None",
    bundle: "RiverscapeExportBundle | None",
    *,
    pixel_m: float,
    configured_mode: str,
    workflow_reasons: Sequence[str] = (),
) -> dict[str, object]:
    """Build the manifest's top-level ``zoning`` object (spec 6b section 3.3).

    Three shapes, chosen by what actually exists this run:

    1. ``bundle`` present -> the full riverscape record. Every key under
       ``LandformResult.provenance`` is copied **verbatim** (spec section
       3.3); the ruleset versions and source products are additionally
       regrouped under ``ruleset_versions``/``sources`` exactly as the spec's
       JSON shows. Area and length fields are derived here, not in the
       pipeline, because ``pixel_m`` is an output-layer fact.
    2. ``zone_result`` present without a bundle (occurrence zoning, or an
       ``auto`` run that fell back) -> the thin subset: ``mode``,
       ``domain_pixel_count``, ``domain_digest``, ``degraded_reasons``,
       ``zone_source``. Riverscape-only keys are **omitted**, never emitted
       as empty defaults, so a reader cannot mistake absent science for
       degraded science.
    3. Neither -> ``mode`` (the *configured* riverscape mode, since no
       ``ZoneResult`` exists to report an actual one) plus
       ``degraded_reasons`` from the workflow tuple. No domain digest is
       fabricated.

    ``degraded_reasons`` prefers ``ZoneResult.degraded_reasons`` whenever a
    ``ZoneResult`` exists (spec section 2 row 5); ``workflow_reasons`` are
    already merged onto it by ``workflow._with_reasons``, and are only read
    directly in shape 3.

    ``domain_digest`` reuses :func:`_hash_array` -- the same helper behind
    ``dea_provenance.zone_mask_digest`` -- so the codebase has exactly one
    mask-digest algorithm.
    """
    if zone_result is None:
        return {
            "mode": str(configured_mode),
            "degraded_reasons": [str(reason) for reason in workflow_reasons],
        }

    section: dict[str, object] = {
        "mode": str(zone_result.mode),
        "degraded_reasons": [str(reason) for reason in zone_result.degraded_reasons],
        "domain_digest": _hash_array(zone_result.mask),
    }

    if bundle is None:
        section["domain_pixel_count"] = int(np.count_nonzero(zone_result.mask))
        section["zone_source"] = str(zone_result.source)
        return section

    provenance = dict(bundle.landform.provenance)
    pixel_counts = {
        str(name): int(count)
        for name, count in dict(provenance["pixel_counts"]).items()
    }
    cell_area_m2 = float(pixel_m) ** 2

    section["domain_pixel_count"] = int(pixel_counts["domain"])
    section["ruleset_versions"] = {
        "channel": provenance["channel_ruleset_version"],
        "waterbody": provenance["waterbody_ruleset_version"],
        "riverine": provenance["riverine_ruleset_version"],
    }
    section["sources"] = {
        "dem_product": provenance["dem_product"],
        "dem_band": provenance["dem_band"],
        "fc_product": provenance["fc_product"],
        "fc_bands": [str(band) for band in provenance["fc_bands"]],
        "waterbodies_source": provenance["waterbodies_source"],
        "years": [int(year) for year in provenance["years"]],
        "zone_source": str(zone_result.source),
    }
    section["bridge_enabled"] = bool(provenance["bridge_enabled"])
    section["bare_threshold_pct"] = _finite_float(
        provenance["bare_threshold_pct"], field="bare_threshold_pct"
    )
    envelope = dict(provenance["envelope"])
    section["envelope"] = {
        "a": _finite_float(envelope["a"], field="envelope.a"),
        "b": _finite_float(envelope["b"], field="envelope.b"),
        "n_bins": int(envelope["n_bins"]),
    }
    section["pixel_counts"] = pixel_counts
    section["bridge_counts_by_cause"] = {
        str(cause): int(count)
        for cause, count in dict(provenance["bridge_counts_by_cause"]).items()
    }
    section["unbridged_counts_by_reason"] = {
        str(reason): int(count)
        for reason, count in dict(provenance["unbridged_counts_by_reason"]).items()
    }
    section["reach_counts"] = {
        str(name): int(count)
        for name, count in dict(provenance["reach_counts"]).items()
    }
    section["bridged_length_m"] = _bridged_length_m(bundle.landform.channel_bridges)
    section["bridged_area_m2"] = (
        float(np.count_nonzero(bundle.landform.bridged_mask)) * cell_area_m2
    )
    section["non_riverine_area_m2"] = float(pixel_counts["non_riverine"]) * cell_area_m2
    return section
```

Add `"build_zoning_section"` to `manifest.py`'s `__all__`.

---

- [ ] **Step 3: Wire `zoning` through `build_run_manifest` and `BundleTransaction.finalize`**

In `build_run_manifest` (`manifest.py:261-365`), add the keyword after `dea_provenance`:

```python
    dea_provenance: Mapping[str, object] | None = None,
    zoning: Mapping[str, object] | None = None,
    manifest_schema_version: str | None = None,
```

and, beside the existing `dea_provenance` assignment near `manifest.py:363`:

```python
    if dea_provenance is not None:
        manifest["dea_provenance"] = dict(dea_provenance)
    if zoning is not None:
        manifest["zoning"] = dict(zoning)
    return manifest
```

In `hydrofragments/output/bundle.py`, add the parameter to `BundleTransaction.finalize` (after `dea_provenance`, `bundle.py:99`):

```python
        dea_provenance: Mapping[str, object] | None = None,
        zoning: Mapping[str, object] | None = None,
```

and forward it in the `build_run_manifest(...)` call (`bundle.py:127-147`), beside `dea_provenance=dea_provenance`:

```python
            dea_provenance=dea_provenance,
            zoning=zoning,
```

---

- [ ] **Step 4: Add the wiring test and run**

Append to `tests/output/test_manifest_zoning.py`:

```python
# --------------------------------------------------------------------------
# Wiring into build_run_manifest
# --------------------------------------------------------------------------


def test_build_run_manifest_emits_zoning_when_supplied(tmp_path) -> None:
    from hydrofragments.config import HydroConfig
    from hydrofragments.output.manifest import build_run_manifest

    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
        }
    )
    manifest = build_run_manifest(
        config,
        run_id="run-1",
        package_version="0.0.0",
        git_sha="deadbeef",
        input_fingerprint={"kind": "generic_binary"},
        planned_backend="cpu",
        actual_backend_by_stage={},
        zoning={"mode": "riverscape", "degraded_reasons": []},
    )

    assert manifest["zoning"] == {"mode": "riverscape", "degraded_reasons": []}


def test_build_run_manifest_omits_zoning_when_absent() -> None:
    from hydrofragments.config import HydroConfig
    from hydrofragments.output.manifest import build_run_manifest

    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
        }
    )
    manifest = build_run_manifest(
        config,
        run_id="run-1",
        package_version="0.0.0",
        git_sha="deadbeef",
        input_fingerprint={"kind": "generic_binary"},
        planned_backend="cpu",
        actual_backend_by_stage={},
    )

    assert "zoning" not in manifest
```

Run:

```
python -m pytest tests/output/test_manifest_zoning.py -q
python -m pytest tests/output -q
```

Expected: **PASS** (13 new tests; `tests/output` suite green — `zoning` is additive and defaults to `None`).

---

- [ ] **Step 5: Regression + commit**

```
python -m pytest tests/ -q
git add hydrofragments/output/manifest.py hydrofragments/output/bundle.py tests/output/test_manifest_zoning.py
git commit -m "Add the manifest zoning section and its three shapes (6b Task 2)"
```

---

## Task 3: Finalize names, GPKG columns, evidence rasters, `channel_bridges`

**Files:**
- Modify: `hydrofragments/config.py`, `hydrofragments/output/rasters.py`, `hydrofragments/output/finalize.py`
- Test: `tests/output/test_riverscape_evidence_exports.py` (create)

**Interfaces:**

*Consumes*:
- `RiverscapeExportBundle` (Task 1)
- `hydrofragments.output.rasters.write_verified_geotiff(*, bands, destination, grid, contract, band_descriptions, metadata) -> Path` (`rasters.py:718`) — writes to a temp file, reopens and validates dtype/CRS/transform/tiling/tags/values, then atomically replaces
- `hydrofragments.output.rasters.preflight_raster_artifacts(destination, *, filenames)` (`rasters.py:555`)
- `hydrofragments.riverscape.bridging._empty_bridge_frame(crs) -> gpd.GeoDataFrame` (`bridging.py:443`) — the schema of record for the bridges layer (spec §3.6 names it explicitly)
- `hydrofragments.riverscape.codes` / `hydrofragments.hydroperiod.codes` for the raster codebooks

*Produces* (Task 4 depends on these):
- `SpatialProduct` / `SPATIAL_PRODUCTS` include `"riverscape_evidence"`
- `RASTER_PRODUCT_CONTRACTS` keys from L6; `RIVERSCAPE_EVIDENCE_BANDS`; `write_riverscape_evidence_geotiffs(arrays, destination, *, grid, metadata) -> dict[str, Path]`
- `finalize._ZONE_NAMES_BY_MODE`, `finalize._zone_names_for(mode) -> Mapping[int, str]`
- `finalize.CHANNEL_BRIDGES_LAYER = "channel_bridges"`
- `preflight_spatial_outputs(..., riverscape_bundle=None)`
- `finalize_analysis_bundle(..., riverscape_bundle=None, zoning_reasons=())`

---

- [ ] **Step 1: Write the failing export tests**

Create `tests/output/test_riverscape_evidence_exports.py`:

```python
"""Phase 6b export surfaces: zone names, GPKG columns, evidence rasters, bridges.

Everything here is unit-level and calls the finalize/raster helpers
directly. The end-to-end wiring is Task 4's
``tests/integration/test_riverscape_exports.py``.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
xr = pytest.importorskip("xarray")
pytest.importorskip("rasterio")
pytest.importorskip("rioxarray")
pyogrio = pytest.importorskip("pyogrio")

import rasterio
from shapely.geometry import LineString

from hydrofragments.output import finalize as finalize_module
from hydrofragments.output.finalize import (
    CHANNEL_BRIDGES_LAYER,
    SpatialProductUnavailable,
    _channel_bridges_geodataframe,
    _riverscape_evidence_arrays,
    _zone_names_for,
    _zones_geodataframe,
)
from hydrofragments.output.rasters import (
    RASTER_PRODUCT_CONTRACTS,
    RIVERSCAPE_EVIDENCE_BANDS,
    write_riverscape_evidence_geotiffs,
)
from hydrofragments.output.riverscape_export import RiverscapeExportBundle
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.riverscape.bridging import _empty_bridge_frame
from hydrofragments.riverscape.riverine import FloodplainEnvelope
from hydrofragments.riverscape.pipeline import LandformResult
from hydrofragments.spatial.zones import build_zones, combine_zones

SHAPE = (3, 4)
PIXEL_M = 30.0
CRS = "EPSG:3577"


def _grid() -> SpatialGrid:
    y = 120.0 - np.arange(SHAPE[0]) * PIXEL_M - PIXEL_M / 2.0
    x = np.arange(SHAPE[1]) * PIXEL_M + PIXEL_M / 2.0
    template = xr.DataArray(
        np.zeros(SHAPE, dtype=np.float32), dims=("y", "x"), coords={"y": y, "x": x}
    ).rio.write_crs(CRS)
    return SpatialGrid.from_dataarray(template, require_georeference=True)


_LANDFORM = np.array(
    [
        [1, 2, 2, 0],
        [2, 2, 3, 0],
        [1, 3, 1, 0],
    ],
    dtype=np.uint8,
)
_HYDROPERIOD = np.array(
    [
        [1, 1, 2, 0],
        [2, 3, 1, 0],
        [4, 2, 1, 0],
    ],
    dtype=np.uint8,
)


def _landform_result(*, channel_bridges, grid):
    bridged = np.zeros(SHAPE, dtype=bool)
    bridged[2, 0] = True
    return LandformResult(
        landform=_LANDFORM,
        channel_source=np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [2, 0, 1, 0]], dtype=np.uint8
        ),
        channel_confidence=np.array(
            [[3, 255, 255, 255], [255, 255, 255, 255], [2, 255, 3, 255]],
            dtype=np.uint8,
        ),
        rule_id=np.zeros(SHAPE, dtype=np.uint8),
        rem=np.array(
            [[0.0, 1.5, 2.0, np.nan], [1.0, 1.0, 3.0, np.nan], [0.0, 2.5, 0.0, np.nan]],
            dtype=np.float32,
        ),
        bridged_mask=bridged,
        channel_bridges=channel_bridges,
        unbridged=(),
        envelope=FloodplainEnvelope(
            a=2.0,
            b=0.0,
            n_bins=1,
            bin_log_a_mid=(3.7,),
            bin_rem_quantile=(2.0,),
            bin_counts=(9,),
            degraded_reasons=(),
        ),
        provenance={"years": (2019, 2021)},
        degraded_reasons=(),
        grid=grid,
    )


def _bridges(crs=CRS):
    return gpd.GeoDataFrame(
        {
            "gap_id": [0],
            "reach_ids": ["1,2"],
            "length_m": [120.0],
            "cumulative_cost": [1.0],
            "cost_per_m": [0.01],
            "upstream_width_m": [30.0],
            "downstream_width_m": [30.0],
            "bridge_confidence": [2],
            "gap_cause": ["vegetated"],
        },
        geometry=[LineString([(15.0, 15.0), (105.0, 15.0)])],
        crs=crs,
    )


def _bundle(*, channel_bridges=None):
    grid = _grid()
    zone_result = replace(
        combine_zones(_LANDFORM, _HYDROPERIOD, source="ga_ls_wo_fq_myear_3"),
        grid=grid,
    )
    landform = _landform_result(
        channel_bridges=_bridges() if channel_bridges is None else channel_bridges,
        grid=grid,
    )
    return RiverscapeExportBundle.from_zoning(zone_result, landform)


def _occurrence_zone_result():
    frequency = np.array(
        [
            [80.0, 30.0, 5.0, 0.0],
            [80.0, 30.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    drainage = np.zeros(SHAPE, dtype=bool)
    drainage[0, 0] = True
    result = build_zones(
        frequency,
        max_wet_mask=frequency > 0,
        valid_count=np.full(SHAPE, 40),
        drainage_mask=drainage,
    )
    return replace(result, grid=_grid())


# --------------------------------------------------------------------------
# Mode-keyed zone names (spec 6b section 3.4)
# --------------------------------------------------------------------------


def test_zone_names_are_keyed_by_mode() -> None:
    """MUTANT: leaving ``_ZONE_NAMES`` mode-blind labels a riverscape Zone 2
    'persistent' instead of 'persistent_off_channel'."""
    assert _zone_names_for("occurrence") == {
        1: "channel_connected",
        2: "persistent",
        3: "seasonal",
        4: "ephemeral",
    }
    assert _zone_names_for("riverscape") == {
        1: "in_channel",
        2: "persistent_off_channel",
        3: "seasonal_floodplain",
        4: "marginal_floodplain",
    }


def test_unknown_mode_falls_back_to_occurrence_names() -> None:
    assert _zone_names_for("something_else")[1] == "channel_connected"


# --------------------------------------------------------------------------
# GPKG zones layer columns (spec 6b section 3.4)
# --------------------------------------------------------------------------


def test_riverscape_zones_gdf_uses_riverscape_names_and_null_science_columns() -> None:
    bundle = _bundle()

    frame = _zones_geodataframe(bundle.zone_result, pixel_size_m=PIXEL_M)

    assert list(frame.columns) == [
        "zone_id",
        "zone_name",
        "area_km2",
        "source",
        "mode",
        "landform",
        "hydroperiod",
        "geometry",
    ]
    assert set(frame["mode"]) == {"riverscape"}
    assert frame["landform"].isna().all()
    assert frame["hydroperiod"].isna().all()
    names = dict(zip(frame["zone_id"], frame["zone_name"]))
    assert names[1] == "in_channel"
    assert names[2] == "persistent_off_channel"


def test_occurrence_zones_gdf_keeps_legacy_names_and_still_has_the_columns() -> None:
    """Regression (spec 6b section 6): occurrence zone names are unchanged,
    but the schema is stable -- the new columns are present and null."""
    frame = _zones_geodataframe(_occurrence_zone_result(), pixel_size_m=PIXEL_M)

    assert set(frame["mode"]) == {"occurrence"}
    assert frame["landform"].isna().all()
    assert frame["hydroperiod"].isna().all()
    names = dict(zip(frame["zone_id"], frame["zone_name"]))
    assert names[1] == "channel_connected"
    assert names[2] == "persistent"


def test_empty_zone_mask_still_returns_the_full_column_set() -> None:
    empty = replace(
        _occurrence_zone_result(), mask=np.zeros(SHAPE, dtype=np.uint8)
    )

    frame = _zones_geodataframe(empty, pixel_size_m=PIXEL_M)

    assert len(frame) == 0
    assert set(["mode", "landform", "hydroperiod"]).issubset(frame.columns)


# --------------------------------------------------------------------------
# Evidence rasters (spec 6b section 3.5)
# --------------------------------------------------------------------------


def test_evidence_filenames_and_dtypes_match_the_locked_contract() -> None:
    expected = {
        "riverscape_landform": ("landform.tif", "uint8"),
        "riverscape_hydroperiod": ("hydroperiod.tif", "uint8"),
        "riverscape_zone_crosstab": ("zone_crosstab.tif", "uint16"),
        "riverscape_channel_source": ("channel_source.tif", "uint8"),
        "riverscape_channel_confidence": ("channel_confidence.tif", "uint8"),
        "riverscape_rem": ("rem.tif", "float32"),
    }
    assert [key for key, _band in RIVERSCAPE_EVIDENCE_BANDS] == list(expected)
    for key, (filename, dtype) in expected.items():
        contract = RASTER_PRODUCT_CONTRACTS[key]
        assert contract.filename == filename
        assert contract.dtype.name == dtype


def test_writing_evidence_produces_six_grid_aligned_rasters(tmp_path: Path) -> None:
    bundle = _bundle()
    grid = bundle.landform.grid

    written = write_riverscape_evidence_geotiffs(
        _riverscape_evidence_arrays(bundle),
        tmp_path,
        grid=grid,
        metadata={"algorithm_version": "1.0.0", "scientific_config_hash": "abc"},
    )

    assert set(written) == {key for key, _band in RIVERSCAPE_EVIDENCE_BANDS}
    for key, path in written.items():
        contract = RASTER_PRODUCT_CONTRACTS[key]
        assert path.name == contract.filename
        with rasterio.open(path) as dataset:
            assert dataset.dtypes[0] == contract.dtype.name
            assert (dataset.height, dataset.width) == SHAPE
            assert dataset.transform == grid.transform
            assert dataset.crs.to_epsg() == 3577


def test_zone_crosstab_raster_is_uint16_and_equals_landform_times_ten_plus_hydroperiod(
    tmp_path: Path,
) -> None:
    """MUTANT: writing the crosstab as uint8, or writing ``landform`` into
    the crosstab band, fails here."""
    bundle = _bundle()

    written = write_riverscape_evidence_geotiffs(
        _riverscape_evidence_arrays(bundle),
        tmp_path,
        grid=bundle.landform.grid,
        metadata={"algorithm_version": "1.0.0", "scientific_config_hash": "abc"},
    )

    with rasterio.open(written["riverscape_zone_crosstab"]) as dataset:
        values = dataset.read(1)
        assert dataset.dtypes[0] == "uint16"
    np.testing.assert_array_equal(
        values, np.where(_LANDFORM == 0, 0, _LANDFORM * 10 + _HYDROPERIOD)
    )


def test_hydroperiod_raster_keeps_code_four_on_the_bridged_pixel(tmp_path: Path) -> None:
    bundle = _bundle()

    written = write_riverscape_evidence_geotiffs(
        _riverscape_evidence_arrays(bundle),
        tmp_path,
        grid=bundle.landform.grid,
        metadata={"algorithm_version": "1.0.0", "scientific_config_hash": "abc"},
    )

    with rasterio.open(written["riverscape_hydroperiod"]) as dataset:
        assert int(dataset.read(1)[2, 0]) == 4


def test_rem_raster_round_trips_nan_outside_the_domain(tmp_path: Path) -> None:
    bundle = _bundle()

    written = write_riverscape_evidence_geotiffs(
        _riverscape_evidence_arrays(bundle),
        tmp_path,
        grid=bundle.landform.grid,
        metadata={"algorithm_version": "1.0.0", "scientific_config_hash": "abc"},
    )

    with rasterio.open(written["riverscape_rem"]) as dataset:
        values = dataset.read(1)
    assert np.isnan(values[:, 3]).all()
    assert values[0, 0] == pytest.approx(0.0)


def test_evidence_write_refuses_an_array_off_the_grid(tmp_path: Path) -> None:
    bundle = _bundle()
    arrays = _riverscape_evidence_arrays(bundle)
    arrays["landform"] = np.zeros((2, 2), dtype=np.uint8)

    from hydrofragments.output.rasters import RasterExportError

    with pytest.raises(RasterExportError, match="landform"):
        write_riverscape_evidence_geotiffs(
            arrays,
            tmp_path,
            grid=bundle.landform.grid,
            metadata={"algorithm_version": "1.0.0", "scientific_config_hash": "abc"},
        )


# --------------------------------------------------------------------------
# Preflight gate (spec 6b sections 3.5, 5)
# --------------------------------------------------------------------------


def test_riverscape_evidence_without_a_bundle_fails_preflight(tmp_path: Path) -> None:
    """MUTANT: writing evidence without checking for the bundle would
    AttributeError deep inside the raster writer instead of failing the run
    up front (spec 6b section 5)."""
    from hydrofragments.config import HydroConfig
    from hydrofragments.models import AnalysisInputs, WaterCube
    from hydrofragments.output.finalize import preflight_spatial_outputs

    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {
                "output_dir": str(tmp_path / "run"),
                "spatial_products": ["riverscape_evidence"],
            },
        }
    )
    cube = _water_cube()

    with pytest.raises(SpatialProductUnavailable, match="riverscape_evidence requires"):
        preflight_spatial_outputs(
            config,
            cube=cube,
            inputs=AnalysisInputs(),
            hydroyear_result=None,
            zone_result=None,
            riverscape_bundle=None,
        )


def test_riverscape_evidence_with_a_gridless_bundle_fails_preflight(tmp_path: Path) -> None:
    from hydrofragments.config import HydroConfig
    from hydrofragments.models import AnalysisInputs
    from hydrofragments.output.finalize import preflight_spatial_outputs

    bundle = _bundle()
    gridless = RiverscapeExportBundle(
        landform=replace(bundle.landform, grid=None),
        hydroperiod_classes=bundle.hydroperiod_classes,
        zone_result=bundle.zone_result,
    )
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {
                "output_dir": str(tmp_path / "run"),
                "spatial_products": ["riverscape_evidence"],
            },
        }
    )

    with pytest.raises(SpatialProductUnavailable, match="georeferenced"):
        preflight_spatial_outputs(
            config,
            cube=_water_cube(),
            inputs=AnalysisInputs(),
            hydroyear_result=None,
            zone_result=bundle.zone_result,
            riverscape_bundle=gridless,
        )


def _water_cube():
    """Minimal georeferenced WaterCube for preflight only."""
    import pandas as pd

    from hydrofragments import open_water_cube

    times = pd.date_range("2020-01-01", periods=3, freq="MS")
    y = 120.0 - np.arange(SHAPE[0]) * PIXEL_M - PIXEL_M / 2.0
    x = np.arange(SHAPE[1]) * PIXEL_M + PIXEL_M / 2.0
    water = xr.DataArray(
        np.zeros((3, *SHAPE), dtype=np.uint8),
        dims=("time", "y", "x"),
        coords={"time": times, "y": y, "x": x},
    ).rio.write_crs(CRS)
    return open_water_cube(water, input_kind="generic_binary")


# --------------------------------------------------------------------------
# channel_bridges layer (spec 6b section 3.6)
# --------------------------------------------------------------------------


def test_bridges_frame_keeps_the_bridging_schema_and_order() -> None:
    bundle = _bundle()

    frame = _channel_bridges_geodataframe(bundle, crs=bundle.landform.grid.crs)

    assert list(frame.columns) == list(_empty_bridge_frame(CRS).columns)
    assert len(frame) == 1
    assert frame.iloc[0]["gap_cause"] == "vegetated"


def test_empty_bridges_produce_an_empty_but_typed_frame() -> None:
    bundle = _bundle(channel_bridges=_empty_bridge_frame(CRS))

    frame = _channel_bridges_geodataframe(bundle, crs=bundle.landform.grid.crs)

    assert len(frame) == 0
    assert list(frame.columns) == list(_empty_bridge_frame(CRS).columns)
    assert frame["length_m"].dtype == np.float64


def test_absent_bridges_frame_falls_back_to_the_empty_schema() -> None:
    bundle = _bundle(channel_bridges=None)

    frame = _channel_bridges_geodataframe(bundle, crs=bundle.landform.grid.crs)

    assert len(frame) == 0
    assert list(frame.columns) == list(_empty_bridge_frame(CRS).columns)


def test_bridges_in_a_foreign_crs_are_refused(tmp_path: Path) -> None:
    """Spec 6b section 5: prefer raising over silently reprojecting, since
    6a already projected bridges onto the analysis grid."""
    bundle = _bundle(channel_bridges=_bridges(crs="EPSG:4326"))

    with pytest.raises(SpatialProductUnavailable, match="CRS"):
        _channel_bridges_geodataframe(bundle, crs=bundle.landform.grid.crs)


def test_empty_bridges_write_an_empty_gpkg_layer(tmp_path: Path) -> None:
    bundle = _bundle(channel_bridges=_empty_bridge_frame(CRS))
    frame = _channel_bridges_geodataframe(bundle, crs=bundle.landform.grid.crs)
    path = tmp_path / "spatial.gpkg"

    pyogrio.write_dataframe(
        frame,
        path,
        layer=CHANNEL_BRIDGES_LAYER,
        driver="GPKG",
        encoding="UTF-8",
        geometry_type="LineString",
    )

    info = pyogrio.read_info(path, layer=CHANNEL_BRIDGES_LAYER)
    assert CHANNEL_BRIDGES_LAYER in pyogrio.read_info(path)["layer_names"]
    assert info["features"] == 0
    assert "gap_cause" in info["fields"]
```

Run:

```
python -m pytest tests/output/test_riverscape_evidence_exports.py -q
```

Expected: **FAIL** — `ImportError: cannot import name 'CHANNEL_BRIDGES_LAYER'` (and the rest).

---

- [ ] **Step 2: Add `riverscape_evidence` to the config allowlist**

In `hydrofragments/config.py`, extend both literals (lines 17-36):

```python
SpatialProduct = Literal[
    "monthly_pools",
    "zones",
    "persistence_rasters",
    "temporal_rasters",
    "refuge_stability_rasters",
    "reach_profiles",
    "riverscape_evidence",
]
RasterFormat = Literal["geotiff", "netcdf"]

SPATIAL_PRODUCTS = frozenset(
    {
        "monthly_pools",
        "zones",
        "persistence_rasters",
        "temporal_rasters",
        "refuge_stability_rasters",
        "reach_profiles",
        "riverscape_evidence",
    }
)
```

No default changes: `spatial_products` still defaults to `()` (or `("monthly_pools",)` under the deprecated `include_vectors` alias) at `config.py:1141-1151`.

---

- [ ] **Step 3: Add the six raster contracts and the evidence writer**

In `hydrofragments/output/rasters.py`, append to `RASTER_PRODUCT_CONTRACTS` (after the `refuge_stability_union_pair_count` entry, `rasters.py:129-135`):

```python
    "riverscape_landform": RasterProductContract(
        filename="landform.tif",
        dtype=np.dtype(np.uint8),
        nodata=np.uint8(0),
        units="landform_code",
        codebook="0=outside,1=in_channel,2=off_channel_riverine,3=non_riverine",
    ),
    "riverscape_hydroperiod": RasterProductContract(
        filename="hydroperiod.tif",
        dtype=np.dtype(np.uint8),
        nodata=np.uint8(0),
        units="hydroperiod_code",
        codebook="0=outside,1=persistent,2=seasonal,3=marginal,4=unobserved",
    ),
    "riverscape_zone_crosstab": RasterProductContract(
        filename="zone_crosstab.tif",
        dtype=np.dtype(np.uint16),
        nodata=np.uint16(0),
        units="crosstab_code",
        codebook="landform*10+hydroperiod; 0=outside",
    ),
    "riverscape_channel_source": RasterProductContract(
        filename="channel_source.tif",
        dtype=np.dtype(np.uint8),
        nodata=_UINT8_NODATA,
        units="channel_source_code",
        codebook="0=not channel,1=observed,2=bridged,255=nodata",
    ),
    "riverscape_channel_confidence": RasterProductContract(
        filename="channel_confidence.tif",
        dtype=np.dtype(np.uint8),
        nodata=_UINT8_NODATA,
        units="confidence_score",
        codebook="0-3 inside channel,255=not a channel pixel",
    ),
    "riverscape_rem": RasterProductContract(
        filename="rem.tif",
        dtype=np.dtype(np.float32),
        nodata=_FLOAT32_NODATA,
        units="metres",
    ),
```

`channel_source` and `channel_confidence` use `255` rather than `0` as nodata because `0` is a *meaningful* code for both ("not a channel pixel" / a real confidence of zero); `landform`, `hydroperiod`, and `zone_crosstab` use `0`, matching the existing `zones` contract's "0=outside/no zone" convention.

Add the band table and writer just after `write_zones_geotiff` (`rasters.py:908`):

```python
#: On-disk order and band names for the ``riverscape_evidence`` product
#: (spec 6b section 3.5). Pairs are ``(RASTER_PRODUCT_CONTRACTS key, array
#: key)``; the array key doubles as the GeoTIFF band description.
RIVERSCAPE_EVIDENCE_BANDS: tuple[tuple[str, str], ...] = (
    ("riverscape_landform", "landform"),
    ("riverscape_hydroperiod", "hydroperiod"),
    ("riverscape_zone_crosstab", "zone_crosstab"),
    ("riverscape_channel_source", "channel_source"),
    ("riverscape_channel_confidence", "channel_confidence"),
    ("riverscape_rem", "rem"),
)


def write_riverscape_evidence_geotiffs(
    arrays: Mapping[str, np.ndarray],
    destination: Path | str,
    *,
    grid: SpatialGrid,
    metadata: Mapping[str, object],
) -> dict[str, Path]:
    """Write the six ``riverscape_evidence`` rasters onto one shared grid.

    ``arrays`` is a plain mapping keyed by band name, not a
    ``RiverscapeExportBundle`` -- this module stays free of the export
    bundle type, and ``output.finalize`` owns the translation. Every band
    goes through ``write_verified_geotiff``, so each file is reopened and
    checked against its contract (dtype, CRS, transform, tiling,
    compression, tags, values) before the atomic replace.
    """
    raster_dir = Path(destination)
    raster_dir.mkdir(parents=True, exist_ok=True)
    preflight_raster_artifacts(
        raster_dir,
        filenames=[
            RASTER_PRODUCT_CONTRACTS[key].filename
            for key, _band in RIVERSCAPE_EVIDENCE_BANDS
        ],
    )

    written: dict[str, Path] = {}
    for key, band in RIVERSCAPE_EVIDENCE_BANDS:
        contract = RASTER_PRODUCT_CONTRACTS[key]
        values = np.asarray(arrays[band])
        if values.shape != (grid.height, grid.width):
            raise RasterExportError(
                f"riverscape evidence band {band} does not align with the export grid"
            )
        written[key] = write_verified_geotiff(
            bands=[values.astype(contract.dtype)],
            destination=raster_dir / contract.filename,
            grid=grid,
            contract=contract,
            band_descriptions=[band],
            metadata={**metadata, "source_name": band},
        )
    return written
```

Extend `rasters.py`'s `__all__` with `"RIVERSCAPE_EVIDENCE_BANDS"` and `"write_riverscape_evidence_geotiffs"`.

---

- [ ] **Step 4: Mode-keyed names and the new GPKG columns in `finalize.py`**

Replace `_ZONE_NAMES` (`finalize.py:44-49`) with:

```python
ZONES_LAYER = "zones"
REACHES_LAYER = "reaches"
REACH_WET_MONTHLY_LAYER = "reach_wet_monthly"
CHANNEL_BRIDGES_LAYER = "channel_bridges"

#: Zone names by zoning mode (spec 6b section 3.4). Occurrence keeps the
#: names the bundle has always written; riverscape uses the parent spec's
#: landform-aware vocabulary, so a Zone 2 polygon is not mislabelled
#: "persistent" when it actually means "persistent off-channel".
_ZONE_NAMES_BY_MODE: dict[str, dict[int, str]] = {
    "occurrence": {
        1: "channel_connected",
        2: "persistent",
        3: "seasonal",
        4: "ephemeral",
    },
    "riverscape": {
        1: "in_channel",
        2: "persistent_off_channel",
        3: "seasonal_floodplain",
        4: "marginal_floodplain",
    },
}


def _zone_names_for(mode: str) -> dict[int, str]:
    """Zone-id -> name table for one zoning mode.

    An unrecognised mode falls back to the occurrence table rather than
    raising: ``ZoneResult.__post_init__`` already rejects any mode outside
    ``{"occurrence", "riverscape"}``, so reaching the fallback means a
    caller hand-built a frame, and a legacy-named export beats a crash in
    the writer.
    """
    return _ZONE_NAMES_BY_MODE.get(mode, _ZONE_NAMES_BY_MODE["occurrence"])
```

Rewrite `_zones_geodataframe` (`finalize.py:163-197`):

```python
_ZONE_COLUMNS = [
    "zone_id",
    "zone_name",
    "area_km2",
    "source",
    "mode",
    "landform",
    "hydroperiod",
    "geometry",
]


def _zones_geodataframe(
    zone_result: ZoneResultType, *, pixel_size_m: float
) -> gpd.GeoDataFrame:
    """Dissolve the zone mask into one polygon per zone id.

    ``landform``/``hydroperiod`` are always ``None`` (spec 6b section 3.4).
    A dissolved Zone 1 multipolygon mixes hydroperiod 4 (bridged, unobserved)
    with 1-3, so no single value is honest at this granularity; the
    authoritative per-pixel layers ship as ``riverscape_evidence``. The
    columns exist in both modes so the GPKG schema is stable.
    """
    if zone_result.grid is None:
        raise SpatialProductUnavailable("zones vector export requires a georeferenced zone mask")
    grid = zone_result.grid
    mask = np.asarray(zone_result.mask, dtype=np.uint8)
    zone_names = _zone_names_for(zone_result.mode)
    cell_area_m2 = float(pixel_size_m) ** 2
    rows: list[dict[str, object]] = []
    for zone_id in sorted(set(int(value) for value in np.unique(mask) if int(value) > 0)):
        zone_pixels = mask == zone_id
        shapes = rio_features.shapes(
            zone_pixels.astype(np.uint8),
            mask=zone_pixels,
            transform=grid.transform,
        )
        geometries = [shapely_shape(geom) for geom, value in shapes if int(value) == 1]
        if not geometries:
            continue
        geometry = geometries[0] if len(geometries) == 1 else gpd.GeoSeries(geometries).union_all()
        area_m2 = float(zone_pixels.sum()) * cell_area_m2
        rows.append(
            {
                "zone_id": zone_id,
                "zone_name": zone_names.get(zone_id, f"zone_{zone_id}"),
                "area_km2": area_m2 / 1_000_000.0,
                "source": zone_result.source,
                "mode": zone_result.mode,
                "landform": None,
                "hydroperiod": None,
                "geometry": geometry,
            }
        )
    if not rows:
        return gpd.GeoDataFrame(
            columns=_ZONE_COLUMNS,
            geometry="geometry",
            crs=grid.crs,
        )
    return gpd.GeoDataFrame(rows, columns=_ZONE_COLUMNS, geometry="geometry", crs=grid.crs)
```

---

- [ ] **Step 5: Evidence arrays, bridges frame, and the preflight gate**

Add near the top of `finalize.py`, beside the existing imports:

```python
from rasterio.crs import CRS as RioCRS

from hydrofragments.output.riverscape_export import RiverscapeExportBundle
from hydrofragments.riverscape.bridging import _empty_bridge_frame
```

`_empty_bridge_frame` is module-private in `bridging.py` but is the schema of record that spec §3.6 names explicitly; importing it here is deliberate, so the export layer cannot drift from the producer's column list.

Add these helpers after `_zones_geodataframe`:

```python
def _riverscape_evidence_arrays(
    bundle: RiverscapeExportBundle,
) -> dict[str, np.ndarray]:
    """Band name -> array for the ``riverscape_evidence`` product.

    Every array comes straight out of the bundle. ``zone_crosstab`` is
    upcast from ``combine_zones``'s ``uint8`` to ``uint16`` (decision L4):
    the widest code today is ``34``, but a uint16 band leaves headroom if
    the landform vocabulary ever grows past 25 codes, and costs nothing at
    these raster sizes.
    """
    landform = bundle.landform
    return {
        "landform": np.asarray(landform.landform, dtype=np.uint8),
        "hydroperiod": np.asarray(bundle.hydroperiod_classes, dtype=np.uint8),
        "zone_crosstab": np.asarray(bundle.zone_result.crosstab, dtype=np.uint16),
        "channel_source": np.asarray(landform.channel_source, dtype=np.uint8),
        "channel_confidence": np.asarray(landform.channel_confidence, dtype=np.uint8),
        "rem": np.asarray(landform.rem, dtype=np.float32),
    }


def _same_crs(left: object, right: object) -> bool:
    try:
        left_crs = RioCRS.from_user_input(left)
        right_crs = RioCRS.from_user_input(right)
    except Exception:
        return str(left) == str(right)
    left_epsg = left_crs.to_epsg()
    right_epsg = right_crs.to_epsg()
    if left_epsg is not None and right_epsg is not None:
        return left_epsg == right_epsg
    return bool(left_crs == right_crs)


#: Column -> pandas dtype for the ``channel_bridges`` layer. Pinned so an
#: empty frame (all-object columns out of ``_empty_bridge_frame``) writes a
#: GPKG layer with the same field types a populated frame would.
_BRIDGE_COLUMN_DTYPES: dict[str, str] = {
    "gap_id": "int64",
    "reach_ids": "object",
    "length_m": "float64",
    "cumulative_cost": "float64",
    "cost_per_m": "float64",
    "upstream_width_m": "float64",
    "downstream_width_m": "float64",
    "bridge_confidence": "int64",
    "gap_cause": "object",
}


def _channel_bridges_geodataframe(
    bundle: RiverscapeExportBundle, *, crs
) -> gpd.GeoDataFrame:
    """Normalize ``LandformResult.channel_bridges`` for GPKG publication.

    An absent frame (a stubbed ``LandformResult``) becomes the typed-empty
    schema from ``bridging._empty_bridge_frame``, so the layer exists with
    its full field list even when no gap was bridged (spec 6b section 3.6).

    A frame already carrying a different CRS is a bug upstream, not
    something to fix here: Phase 6a builds bridge geometry directly on the
    analysis grid, so a mismatch means the wrong frame arrived. Spec section
    5 prefers raising over silently reprojecting.
    """
    frame = bundle.landform.channel_bridges
    if frame is None:
        frame = _empty_bridge_frame(crs)
    frame = gpd.GeoDataFrame(frame).copy()
    if frame.crs is not None and not _same_crs(frame.crs, crs):
        raise SpatialProductUnavailable(
            "channel_bridges CRS does not match the riverscape export grid CRS"
        )
    if frame.crs is None:
        frame = frame.set_crs(crs)
    missing = [name for name in _BRIDGE_COLUMN_DTYPES if name not in frame.columns]
    if missing:
        raise SpatialProductUnavailable(
            f"channel_bridges is missing required columns: {missing}"
        )
    for column, dtype in _BRIDGE_COLUMN_DTYPES.items():
        frame[column] = frame[column].astype(dtype)
    return frame[[*_BRIDGE_COLUMN_DTYPES, "geometry"]]
```

Extend `preflight_spatial_outputs` (`finalize.py:68-160`) with the new keyword and branch:

```python
def preflight_spatial_outputs(
    config: HydroConfig,
    *,
    cube: WaterCube,
    inputs: AnalysisInputs,
    hydroyear_result: HyAnchorResult | None,
    zone_result: ZoneResult | None = None,
    riverscape_bundle: RiverscapeExportBundle | None = None,
) -> SpatialGrid | None:
```

and insert before the final `else:` at `finalize.py:143`:

```python
        elif product == "riverscape_evidence":
            if riverscape_bundle is None:
                raise SpatialProductUnavailable(
                    "riverscape_evidence requires a riverscape zoning run; "
                    "no RiverscapeExportBundle was produced"
                )
            if riverscape_bundle.landform.grid is None:
                raise SpatialProductUnavailable(
                    "riverscape_evidence requires a georeferenced landform grid"
                )
```

---

- [ ] **Step 6: Write the evidence rasters and the bridges layer**

In `_write_spatial_vectors` (`finalize.py:235-325`), add the parameter and the bridges block just before `return registrations`:

```python
def _write_spatial_vectors(
    staging_root: Path,
    *,
    config: HydroConfig,
    core: CoreAnalysisResult,
    cube: WaterCube,
    inputs: AnalysisInputs,
    zone_result: ZoneResult | None,
    riverscape_bundle: RiverscapeExportBundle | None,
    pixel_size_m: float,
) -> list[ArtifactRegistration]:
```

```python
    if "riverscape_evidence" in products:
        if riverscape_bundle is None:
            raise SpatialProductUnavailable(
                "riverscape_evidence requires a riverscape zoning run; "
                "no RiverscapeExportBundle was produced"
            )
        grid = riverscape_bundle.landform.grid
        if grid is None:
            raise SpatialProductUnavailable(
                "riverscape_evidence requires a georeferenced landform grid"
            )
        bridges_gdf = _channel_bridges_geodataframe(riverscape_bundle, crs=grid.crs)
        pyogrio.write_dataframe(
            bridges_gdf,
            gpkg_path,
            layer=CHANNEL_BRIDGES_LAYER,
            driver="GPKG",
            encoding="UTF-8",
            geometry_type="LineString",
            append=gpkg_path.exists(),
        )
        registrations.append(
            ArtifactRegistration(
                name="channel_bridges",
                relative_path=f"vectors/{SPATIAL_GPKG_NAME}",
                media_type="application/geopackage+sqlite3",
            )
        )

    return registrations
```

`geometry_type="LineString"` is passed explicitly because an all-empty geometry column gives pyogrio nothing to infer from. If the installed pyogrio rejects the keyword, drop it and the empty layer is written as geometry type `Unknown`; `test_empty_bridges_write_an_empty_gpkg_layer` still passes either way, since it asserts the layer, feature count, and fields, not the geometry type.

In `_write_spatial_rasters` (`finalize.py:328-400`), add the parameter and restructure the two early-return guards, then append the evidence block:

```python
def _write_spatial_rasters(
    staging_root: Path,
    *,
    config: HydroConfig,
    core: CoreAnalysisResult,
    inputs: AnalysisInputs,
    zone_result: ZoneResult | None,
    riverscape_bundle: RiverscapeExportBundle | None,
    analysis_mask: np.ndarray | None,
) -> list[ArtifactRegistration]:
    products = set(config.output.spatial_products)
    raster_products = products & {
        "persistence_rasters",
        "temporal_rasters",
        "refuge_stability_rasters",
        "zones",
        "riverscape_evidence",
    }
    wants_zones = "zones" in products
    wants_evidence = "riverscape_evidence" in products
    if not raster_products:
        return []
    if core.raster_checkpoint is None and not wants_zones and not wants_evidence:
        return []
```

(the rest of the existing body is unchanged, up to the final `return registrations`), then immediately before that `return`:

```python
    if wants_evidence:
        if riverscape_bundle is None:
            raise SpatialProductUnavailable(
                "riverscape_evidence requires a riverscape zoning run; "
                "no RiverscapeExportBundle was produced"
            )
        evidence_grid = riverscape_bundle.landform.grid
        if evidence_grid is None:
            raise SpatialProductUnavailable(
                "riverscape_evidence requires a georeferenced landform grid"
            )
        from hydrofragments.output.rasters import write_riverscape_evidence_geotiffs

        evidence_paths = write_riverscape_evidence_geotiffs(
            _riverscape_evidence_arrays(riverscape_bundle),
            raster_dir,
            grid=evidence_grid,
            metadata={
                "algorithm_version": "1.0.0",
                "scientific_config_hash": config.config_hash,
            },
        )
        for name, path in evidence_paths.items():
            registrations.append(
                ArtifactRegistration(
                    name=name,
                    relative_path=str(path.relative_to(staging_root)).replace("\\", "/"),
                    media_type="image/tiff",
                )
            )

    return registrations
```

Finally, thread the new parameters through `finalize_analysis_bundle` (`finalize.py:403-521`) and build the `zoning` section there:

```python
def finalize_analysis_bundle(
    config: HydroConfig,
    core: CoreAnalysisResult,
    *,
    cube: WaterCube,
    inputs: AnalysisInputs | None = None,
    pixel_size_m: float = 30.0,
    zone_result: ZoneResult | None = None,
    riverscape_bundle: RiverscapeExportBundle | None = None,
    zoning_reasons: Sequence[str] = (),
    dea_provenance: Mapping[str, object] | None = None,
    timings_seconds: Mapping[str, float] | None = None,
    peak_rss_bytes: int | None = None,
) -> HydroResult:
```

Inside, pass `riverscape_bundle=riverscape_bundle` to both `_write_spatial_vectors` and `_write_spatial_rasters`, and build the zoning section just before `transaction.finalize(...)`:

```python
        resolved_zones = _resolve_zone_result(inputs, zone_result=zone_result)
        zoning_section = build_zoning_section(
            resolved_zones,
            riverscape_bundle,
            pixel_m=pixel_size_m,
            configured_mode=config.riverscape.mode,
            workflow_reasons=tuple(zoning_reasons),
        )
        if resolved_zones is None and not zoning_section["degraded_reasons"]:
            zoning_section = None  # nothing to say: no zones, no reasons
```

and add `zoning=zoning_section,` to the `transaction.finalize(...)` call beside `dea_provenance=...`. Import `build_zoning_section` alongside the existing `build_run_manifest` import at `finalize.py:27`.

The `zoning_section = None` shortcut keeps a plain cube-only `analyze()` run byte-identical to today: no zones and no reasons means no `zoning` key at all, so no existing manifest snapshot shifts.

Run:

```
python -m pytest tests/output/test_riverscape_evidence_exports.py -q
python -m pytest tests/output tests/integration/test_spatial_exports.py -q
```

Expected: **PASS**.

---

- [ ] **Step 7: Regression + commit**

```
python -m pytest tests/ -q
```

Expected: **PASS**. If `tests/contracts/` or `tests/release/` pins the `SPATIAL_PRODUCTS` set or a config-hash snapshot, update the pinned list — widening an allowlist does not change any default, so no recorded hash for an existing config should move. If one does, stop and investigate before editing the snapshot.

```
git add hydrofragments/config.py hydrofragments/output/rasters.py hydrofragments/output/finalize.py tests/output/test_riverscape_evidence_exports.py
git commit -m "Mode-keyed zone names, stable GPKG columns, riverscape_evidence rasters, channel_bridges (6b Task 3)"
```

---

## Task 4: Workflow→finalize plumbing + gating + docs

**Files:**
- Modify: `hydrofragments/api.py`, `hydrofragments/workflow.py`, `docs/spatial_exports.md`
- Test: `tests/integration/test_riverscape_exports.py` (create), `tests/gating/test_zones_do_not_multiply_metrics.py` (modify)

**Interfaces:**

*Consumes*: everything Tasks 1–3 produced.

*Produces*: a run in which `analyze_from_dea(..., config=<mode auto + riverscape_evidence>)` writes `zoning`, six rasters, and the bridges layer.

---

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_riverscape_exports.py`:

```python
"""End-to-end: a riverscape run writes zoning, evidence rasters, and bridges.

Reuses ``tests/integration/test_riverscape_modes.py``'s monkeypatch
convention -- every hydroseason entry point is patched as a module attribute
on the real ``hydroseason`` package, and the riverscape zoning call is
replaced with a stub -- so nothing here touches STAC/WFS.

Only ``riverscape_evidence`` is requested, deliberately: the ``zones``
raster branch in ``output/finalize.py`` additionally needs
``core.spatial_grid`` from ``SectionSpatialCollector``, which is an
orthogonal dependency. Zone names and the GPKG zone columns are unit-tested
in ``tests/output/test_riverscape_evidence_exports.py``.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
xr = pytest.importorskip("xarray")
pytest.importorskip("rasterio")
pytest.importorskip("rioxarray")
pyogrio = pytest.importorskip("pyogrio")

import hydroseason
import rasterio
from shapely import wkb
from shapely.geometry import LineString, box

from hydrofragments import workflow as workflow_module
from hydrofragments.config import HydroConfig
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.riverscape.pipeline import LandformResult, with_grid
from hydrofragments.riverscape.riverine import FloodplainEnvelope
from hydrofragments.spatial.zones import combine_zones
from hydrofragments.workflow import _default_config, analyze_from_dea

_SHAPE = (4, 4)
_TRANSFORM = (30.0, 0.0, 0.0, 0.0, -30.0, 120.0)
_CRS = "EPSG:3577"
```

Then copy, unchanged, these helpers from `tests/integration/test_riverscape_modes.py` (same file, same values — they are module-local there and `tests/integration` has no `__init__.py`, so they cannot be imported):

- `_aoi()`, `_drainage()`, `_wo_statistics_dataset()`, `_stats()`
- `_Recorder`, `_install_happy_path(monkeypatch, recorder, tmp_path)`

and add:

```python
_PROVENANCE = {
    "channel_ruleset_version": "c-1.0.0",
    "waterbody_ruleset_version": "w-1.0.0",
    "riverine_ruleset_version": "r-1.0.0",
    "dem_product": "ga_srtm_dem1sv1_0",
    "dem_band": "elevation",
    "fc_product": "ga_ls_fc_pc_cyear_3",
    "fc_bands": ("bs_pc_50", "pv_pc_50", "npv_pc_50"),
    "waterbodies_source": "dea_waterbodies",
    "years": (2020, 2020),
    "bridge_enabled": True,
    "bare_threshold_pct": 30.0,
    "envelope": {"a": 2.0, "b": 0.0, "n_bins": 1},
    "pixel_counts": {
        "domain": 16,
        "water_seed": 16,
        "observed_channel": 12,
        "bridged_channel": 1,
        "in_channel": 13,
        "off_channel_riverine": 2,
        "non_riverine": 1,
    },
    "bridge_counts_by_cause": {"vegetated": 1},
    "unbridged_counts_by_reason": {},
    "reach_counts": {"total": 1, "line_fallback": 0, "multithread": 0},
}


def _bridges():
    return gpd.GeoDataFrame(
        {
            "gap_id": [0],
            "reach_ids": ["1"],
            "length_m": [90.0],
            "cumulative_cost": [1.0],
            "cost_per_m": [0.011],
            "upstream_width_m": [30.0],
            "downstream_width_m": [30.0],
            "bridge_confidence": [2],
            "gap_cause": ["vegetated"],
        },
        geometry=[LineString([(15.0, 15.0), (105.0, 15.0)])],
        crs=_CRS,
    )


_LANDFORM = np.array(
    [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 2, 2, 3],
    ],
    dtype=np.uint8,
)
_HYDROPERIOD = np.array(
    [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 4],
        [1, 1, 2, 3],
    ],
    dtype=np.uint8,
)


def _landform_result():
    bridged = np.zeros(_SHAPE, dtype=bool)
    bridged[2, 3] = True
    return LandformResult(
        landform=_LANDFORM,
        channel_source=np.where(bridged, 2, 1).astype(np.uint8),
        channel_confidence=np.full(_SHAPE, 3, dtype=np.uint8),
        rule_id=np.ones(_SHAPE, dtype=np.uint8),
        rem=np.zeros(_SHAPE, dtype=np.float32),
        bridged_mask=bridged,
        channel_bridges=_bridges(),
        unbridged=(),
        envelope=FloodplainEnvelope(
            a=2.0,
            b=0.0,
            n_bins=1,
            bin_log_a_mid=(3.7,),
            bin_rem_quantile=(2.0,),
            bin_counts=(16,),
            degraded_reasons=(),
        ),
        provenance=dict(_PROVENANCE),
        degraded_reasons=("bare_threshold_fallback",),
    )


def _install_riverscape(monkeypatch, recorder):
    def fake_zones_from_riverscape_with_landform(stats, **kwargs):
        recorder.riverscape_calls.append(kwargs)
        grid = SpatialGrid.from_dataarray(stats.frequency, require_georeference=True)
        zone_result = replace(
            combine_zones(
                _LANDFORM,
                _HYDROPERIOD,
                source="ga_ls_wo_fq_myear_3",
                degraded_reasons=("bare_threshold_fallback",),
            ),
            grid=grid,
        )
        return zone_result, with_grid(_landform_result(), grid)

    monkeypatch.setattr(
        workflow_module,
        "zones_from_riverscape_with_landform",
        fake_zones_from_riverscape_with_landform,
    )
    monkeypatch.setattr(
        workflow_module, "_frequency_geobox", lambda frequency: SimpleNamespace()
    )


def _evidence_config(output_dir: Path, mode: str = "auto") -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "max_water",
                "composite_owner": "upstream",
            },
            "riverscape": {"mode": mode},
            "output": {
                "output_dir": str(output_dir),
                "spatial_products": ["riverscape_evidence"],
            },
        }
    )


def _run(monkeypatch, tmp_path, output_dir, *, config=None, drainage=None):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)
    _install_riverscape(monkeypatch, recorder)
    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        drainage=_drainage() if drainage is None else drainage,
        cache_dir=tmp_path / "wofs_cache",
        config=config or _evidence_config(output_dir),
    )
    return result, recorder


# --------------------------------------------------------------------------
# Acceptance (spec 6b section 10)
# --------------------------------------------------------------------------


def test_riverscape_run_writes_the_full_zoning_section(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "out_zoning"
    result, _recorder = _run(monkeypatch, tmp_path, output_dir)

    manifest = json.loads(
        (Path(result.output_dir) / "run_manifest.json").read_text(encoding="utf-8")
    )
    zoning = manifest["zoning"]
    assert zoning["mode"] == "riverscape"
    assert zoning["degraded_reasons"] == ["bare_threshold_fallback"]
    assert zoning["domain_pixel_count"] == 16
    assert zoning["sources"]["dem_product"] == "ga_srtm_dem1sv1_0"
    assert zoning["sources"]["zone_source"] == "ga_ls_wo_fq_myear_3"
    assert zoning["ruleset_versions"]["channel"] == "c-1.0.0"
    assert zoning["bridge_counts_by_cause"] == {"vegetated": 1}
    assert zoning["bridged_length_m"] == pytest.approx(90.0)
    assert zoning["bridged_area_m2"] == pytest.approx(900.0)
    assert zoning["non_riverine_area_m2"] == pytest.approx(900.0)


def test_riverscape_run_writes_six_evidence_rasters(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "out_rasters"
    result, _recorder = _run(monkeypatch, tmp_path, output_dir)
    raster_dir = Path(result.output_dir) / "rasters"

    for filename in (
        "landform.tif",
        "hydroperiod.tif",
        "zone_crosstab.tif",
        "channel_source.tif",
        "channel_confidence.tif",
        "rem.tif",
    ):
        assert (raster_dir / filename).exists(), filename

    with rasterio.open(raster_dir / "hydroperiod.tif") as dataset:
        assert int(dataset.read(1)[2, 3]) == 4
    with rasterio.open(raster_dir / "zone_crosstab.tif") as dataset:
        assert dataset.dtypes[0] == "uint16"
        assert int(dataset.read(1)[2, 3]) == 14

    inventory = {
        item["relative_path"]
        for item in result.manifest["artifact_inventory"]
    }
    assert "rasters/landform.tif" in inventory
    assert "rasters/rem.tif" in inventory


def test_riverscape_run_writes_the_channel_bridges_layer(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "out_bridges"
    result, _recorder = _run(monkeypatch, tmp_path, output_dir)
    gpkg = Path(result.output_dir) / "vectors" / "spatial.gpkg"

    assert gpkg.exists()
    assert "channel_bridges" in pyogrio.read_info(gpkg)["layer_names"]
    frame = gpd.read_file(gpkg, layer="channel_bridges")
    assert len(frame) == 1
    assert frame.iloc[0]["gap_cause"] == "vegetated"
    assert set(
        [
            "gap_id",
            "reach_ids",
            "length_m",
            "cumulative_cost",
            "cost_per_m",
            "upstream_width_m",
            "downstream_width_m",
            "bridge_confidence",
            "gap_cause",
        ]
    ).issubset(frame.columns)


# --------------------------------------------------------------------------
# Hermetic occurrence defaults (spec 6b sections 2 row 9, 6, 10)
# --------------------------------------------------------------------------


def test_default_config_does_not_opt_into_riverscape_evidence(tmp_path) -> None:
    """MUTANT: adding ``riverscape_evidence`` to ``_default_config`` would
    make every ``config=None`` run try to write evidence it cannot produce."""
    config = _default_config(output_dir=tmp_path)

    assert "riverscape_evidence" not in config.output.spatial_products


def test_mode_off_run_writes_no_evidence_and_a_thin_zoning_section(
    monkeypatch, tmp_path
) -> None:
    output_dir = tmp_path / "out_off"
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "max_water",
                "composite_owner": "upstream",
            },
            "riverscape": {"mode": "off"},
            "output": {"output_dir": str(output_dir)},
        }
    )
    result, _recorder = _run(monkeypatch, tmp_path, output_dir, config=config)

    manifest = json.loads(
        (Path(result.output_dir) / "run_manifest.json").read_text(encoding="utf-8")
    )
    zoning = manifest["zoning"]
    assert zoning["mode"] == "occurrence"
    assert "sources" not in zoning
    assert "bridged_length_m" not in zoning
    assert not (Path(result.output_dir) / "rasters" / "landform.tif").exists()


def test_evidence_requested_on_an_occurrence_run_fails_preflight(
    monkeypatch, tmp_path
) -> None:
    """Spec 6b section 5: requested-without-bundle is a preflight failure,
    not a silently skipped product."""
    from hydrofragments.output.finalize import SpatialProductUnavailable

    output_dir = tmp_path / "out_no_bundle"
    with pytest.raises(SpatialProductUnavailable, match="riverscape_evidence requires"):
        _run(
            monkeypatch,
            tmp_path,
            output_dir,
            config=_evidence_config(output_dir, mode="off"),
        )
```

Run:

```
python -m pytest tests/integration/test_riverscape_exports.py -q
```

Expected: **FAIL** — `KeyError: 'zoning'` and missing raster files; the workflow does not yet forward the bundle.

---

- [ ] **Step 2: Plumb the bundle through `api._run_core_analysis`**

In `hydrofragments/api.py`, add the parameter to `_run_core_analysis` (line 1118):

```python
    zone_result: ZoneResult | None = None,
    riverscape_bundle: Any = None,
```

and forward it in the `preflight_spatial_outputs(...)` call (lines 1172-1178):

```python
        preflight_spatial_outputs(
            config,
            cube=cube,
            inputs=inputs,
            hydroyear_result=hydroyear_result,
            zone_result=zone_result or inputs.zones,
            riverscape_bundle=riverscape_bundle,
        )
```

The annotation is `Any` rather than `RiverscapeExportBundle` so `api.py` does not gain a module-level import of the export type; `api.py` only hands it through.

---

- [ ] **Step 3: Plumb the bundle and reasons through `analyze_from_dea`**

In `hydrofragments/workflow.py`, pass the bundle to `_run_core_analysis` (around line 693-701):

```python
    core = _run_core_analysis(
        cube,
        aoi_id,
        config=resolved_config,
        inputs=inputs,
        pixel_size_m=resolution,
        git_sha=resolve_git_sha(),
        zone_result=zone_result,
        riverscape_bundle=riverscape_bundle,
    )
```

and to `finalize_analysis_bundle` (around line 731-740):

```python
    result = finalize_analysis_bundle(
        resolved_config,
        core,
        cube=cube,
        inputs=inputs,
        pixel_size_m=resolution,
        zone_result=zone_result,
        riverscape_bundle=riverscape_bundle,
        zoning_reasons=riverscape_degraded_reasons,
        dea_provenance=dea_provenance,
        timings_seconds=timings,
    )
```

Extend `analyze_from_dea`'s docstring with one paragraph:

```
    When ``config.output.spatial_products`` includes ``"riverscape_evidence"``
    and the riverscape branch actually ran, the run also writes six evidence
    rasters under ``rasters/`` and a ``channel_bridges`` layer into
    ``vectors/spatial.gpkg`` (Phase 6b spec section 3.5/3.6). Requesting the
    product on a run that cannot produce a bundle -- ``mode="off"``, or an
    ``auto`` run that fell back to occurrence zoning -- fails preflight
    rather than silently skipping the product.
```

Run:

```
python -m pytest tests/integration/test_riverscape_exports.py -q
python -m pytest tests/integration -q
```

Expected: **PASS**.

---

- [ ] **Step 4: Write the failing gating extension**

Modify `tests/gating/test_zones_do_not_multiply_metrics.py`. Add imports:

```python
from hydrofragments.spatial.zones import build_zones, combine_zones
```

and append:

```python
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
```

Run:

```
python -m pytest tests/gating/test_zones_do_not_multiply_metrics.py -q
```

Expected: **PASS immediately** — this is a *characterisation* test: it asserts an invariant the codebase already upholds, and its value is as a tripwire for later phases. If it fails now, the riverscape path is already leaking into metrics and that is a stop-and-investigate result, not a fixture problem. (If `combine_zones` raises `"landform and hydroperiod disagree on the zoned extent"`, the fixture's `observed` mask and the two layers disagree — narrow `observed` until they match; do not relax `combine_zones`.)

---

- [ ] **Step 5: Update `docs/spatial_exports.md`**

Three edits.

**(a)** In the all-products config example (lines 61-73), add the product with a note:

```python
"output": {
    "output_dir": "runs/full",
    "spatial_products": [
        "monthly_pools",
        "zones",
        "persistence_rasters",
        "temporal_rasters",
        "refuge_stability_rasters",
        "reach_profiles",
        "riverscape_evidence",
    ],
}
```

immediately followed by:

> `riverscape_evidence` is the only product with a *workflow-mode* prerequisite: it requires an `analyze_from_dea` run whose `riverscape.mode` resolved to riverscape zoning. Requesting it on an occurrence run fails preflight.

**(b)** In the directory tree (lines 81-104), add the new artifacts:

```text
  vectors/spatial.gpkg
    monthly_pools
    zones
    reaches
    reach_wet_monthly
    channel_bridges                 # riverscape_evidence only
  ...
  rasters/landform.tif              # riverscape_evidence only
  rasters/hydroperiod.tif           # riverscape_evidence only
  rasters/zone_crosstab.tif         # riverscape_evidence only
  rasters/channel_source.tif        # riverscape_evidence only
  rasters/channel_confidence.tif    # riverscape_evidence only
  rasters/rem.tif                   # riverscape_evidence only
```

**(c)** Replace the `### zones` section (lines 175-181) and add two new sections before `### reach_profiles`:

```markdown
### `zones`

**Paths:** `vectors/spatial.gpkg` layer `zones`, `rasters/zones.tif`

| Column | Type | Notes |
|---|---|---|
| `zone_id` | int | 1–4 |
| `zone_name` | str | Mode-keyed, see below |
| `area_km2` | float | Dissolved polygon area |
| `source` | str | Zone provenance (DEA product id, or `occurrence`) |
| `mode` | str | `occurrence` or `riverscape` |
| `landform` | int or null | Always null, see below |
| `hydroperiod` | int or null | Always null, see below |
| `geometry` | MultiPolygon | Analysis CRS |

Zone names depend on the zoning mode:

| Mode | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `occurrence` | `channel_connected` | `persistent` | `seasonal` | `ephemeral` |
| `riverscape` | `in_channel` | `persistent_off_channel` | `seasonal_floodplain` | `marginal_floodplain` |

`landform` and `hydroperiod` are **always null** on this layer, in both modes. Zone polygons are dissolved per `zone_id`, and a dissolved riverscape Zone 1 mixes hydroperiod 4 (bridged, unobserved) with hydroperiods 1–3, so no single value is truthful at polygon granularity. The columns exist anyway so the GeoPackage schema does not change between modes. The authoritative per-pixel values ship as the `riverscape_evidence` rasters.

**Prerequisites:** Explicit zone input (`AnalysisInputs.zones` or DEA workflow zone result). Unavailable from a cube-only `analyze()` call.

### `riverscape_evidence`

**Paths:** six GeoTIFFs under `rasters/`, plus the `channel_bridges` layer in `vectors/spatial.gpkg`.

| File | dtype | nodata | Meaning |
|---|---|---|---|
| `landform.tif` | uint8 | 0 | 0=outside, 1=in_channel, 2=off_channel_riverine, 3=non_riverine |
| `hydroperiod.tif` | uint8 | 0 | 0=outside, 1=persistent, 2=seasonal, 3=marginal, 4=unobserved (bridged) |
| `zone_crosstab.tif` | uint16 | 0 | `landform * 10 + hydroperiod`; 0 outside |
| `channel_source.tif` | uint8 | 255 | 0=not channel, 1=observed, 2=bridged |
| `channel_confidence.tif` | uint8 | 255 | 0–3 inside channel; 255 outside channel |
| `rem.tif` | float32 | NaN | Relative elevation above the channel, metres |

All six share the `zones.tif` grid (same CRS, transform, and shape).

**Prerequisites:** An `analyze_from_dea` run whose `riverscape.mode` resolved to riverscape zoning — that is, `auto` or `required` with drainage and the riverscape source products both available. `mode="off"`, and any `auto` run that fell back to occurrence zoning, cannot produce these layers, and requesting the product on such a run fails preflight with `SpatialProductUnavailable`. The product is **not** added by default: a run with `config=None` stays occurrence-only and writes no evidence.

### `channel_bridges`

**Path:** `vectors/spatial.gpkg` layer `channel_bridges`

One LineString per reconstructed channel gap, in the analysis CRS (EPSG:3577).

| Column | Type | Notes |
|---|---|---|
| `gap_id` | int | Gap identifier |
| `reach_ids` | str | Comma-joined `HydroID`s the gap spans |
| `length_m` | float | Bridge centreline length |
| `cumulative_cost` | float | Least-cost path cost |
| `cost_per_m` | float | `cumulative_cost / length_m` |
| `upstream_width_m` | float | Corridor width at the upstream anchor |
| `downstream_width_m` | float | Corridor width at the downstream anchor |
| `bridge_confidence` | int | 1 or 2 |
| `gap_cause` | str | `vegetated`, `narrow`, or `unobserved` |
| `geometry` | LineString | Analysis CRS |

**Prerequisites:** Same gate as `riverscape_evidence`, so bridges are never written without the rasters that explain them. When no gap was bridged the layer is still created, empty, with the full schema.

### Manifest `zoning`

Every run that produces a `ZoneResult` — or that has a recorded reason for producing none — writes a top-level `zoning` object into `run_manifest.json`. A riverscape run records the full science provenance (ruleset versions, source products, floodplain envelope, pixel counts, bridge and reach counts, bridged length/area, non-riverine area). An occurrence run records the thin subset: `mode`, `domain_pixel_count`, `domain_digest`, `degraded_reasons`, `zone_source`, with riverscape-only keys omitted rather than emitted empty. A run whose zoning returned nothing records only `mode` (the configured `riverscape.mode`) and `degraded_reasons`.
```

If `tests/docs/test_vocabulary_scan.py` flags a new term (`zone_crosstab`, `channel_bridges`, `riverscape_evidence`), add it to that test's approved vocabulary — these are artifact names, not new science terms.

---

- [ ] **Step 6: Full regression + commit**

```
python -m pytest tests/ -q
python -m pytest tests/riverscape/test_riverscape_import_boundary.py tests/gating -q
```

Expected: **PASS**.

```
git add hydrofragments/api.py hydrofragments/workflow.py docs/spatial_exports.md tests/integration/test_riverscape_exports.py tests/gating/test_zones_do_not_multiply_metrics.py
git commit -m "Plumb the riverscape export bundle into finalize; gating + docs (6b Task 4)"
```

---

## Self-review coverage checklist

Work through this after Task 4. Every spec section must land on a task and a test.

| Spec § | Requirement | Task | Evidence |
|---|---|---|---|
| §1.1.1 | Retain science past zoning, no second source load | 1 | `zones_from_riverscape_with_landform`; `test_with_landform_returns_the_landform_build_landform_produced`; export code calls no loader |
| §1.1.2 | Manifest `zoning` section | 2 | `build_zoning_section`; `tests/output/test_manifest_zoning.py` |
| §1.1.3 | Mode-keyed names; `mode`/`landform`/`hydroperiod` columns | 3 | `_zone_names_for`; `test_riverscape_zones_gdf_uses_riverscape_names_and_null_science_columns` |
| §1.1.4 | `riverscape_evidence` product + `channel_bridges` layer | 3 | Six contracts; `write_riverscape_evidence_geotiffs`; `_channel_bridges_geodataframe` |
| §1.1.5 | Gating extended to riverscape | 4 | `test_analyze_output_identical_whether_or_not_riverscape_zones_were_computed` |
| §1.1.6 | `docs/spatial_exports.md` updated | 4 | Task 4 Step 5 (a)(b)(c) |
| §1.2 | No `windows.py`, no validation harness, no config schema bump | — | Nothing touched; `ACCEPTED_CONFIG_SCHEMA_VERSIONS` unchanged |
| §1.3 | Import boundary held; `with_grid` is the seam | 1 | `test_riverscape_import_boundary.py` re-run in Task 1 Step 7; `with_grid` used in `zones_from_riverscape_with_landform` |
| §2 row 2 | Bundle held beside `ZoneResult`; finalize writes | 1, 3 | `_resolve_zone_result` 3-tuple; `_write_spatial_rasters`/`_write_spatial_vectors` |
| §2 row 3 | Surface landform once, do not discard | 1 | `test_with_landform_returns_the_landform_build_landform_produced` (mutant: discard again) |
| §2 row 5 | Prefer `ZoneResult.degraded_reasons`; workflow tuple for no-zone rows | 2 | `test_riverscape_degraded_reasons_come_from_the_zone_result`; `test_no_zone_result_records_configured_mode_and_workflow_reasons` |
| §2 row 9 | `_default_config` does not auto-add the product | 4 | `test_default_config_does_not_opt_into_riverscape_evidence` |
| §2 row 11 | Evidence requires a grid | 1, 3 | `test_with_landform_attaches_the_zone_grid_to_the_landform_result`; `test_riverscape_evidence_with_a_gridless_bundle_fails_preflight` |
| §2 row 13 | No hash/schema bump; allowlist widened only | 3 | `config.py` diff adds one member to two literals |
| §3.1 | Bundle fields; produced only on success; no re-classification | 1 | `test_from_zoning_recovers_the_exact_hydroperiod_array_from_the_crosstab`; `test_mode_off_produces_no_bundle`; `test_auto_occurrence_fallback_produces_no_bundle` |
| §3.2 | Companion function; `ZoneResult` shape stable; 3-tuple resolve | 1 | `test_wrapper_still_returns_a_bare_zone_result`; `test_riverscape_path_produces_an_export_bundle` |
| §3.3 full shape | Verbatim provenance keys; envelope; counts | 2 | `test_riverscape_section_copies_every_provenance_key_verbatim` |
| §3.3 thin shape | Occurrence subset omits riverscape keys | 2 | `test_occurrence_section_is_the_thin_subset_and_omits_riverscape_keys` |
| §3.3 no-zones shape | Reasons only, no fabricated digest | 2 | `test_no_zone_result_never_fabricates_a_domain_digest` |
| §3.3 area/length | Three formulas | 2 | `test_riverscape_area_and_length_fields_use_the_documented_formulas` |
| §3.3 digest | Reuse the `zone_mask_digest` helper | 2 | `test_riverscape_domain_fields_use_provenance_and_the_shared_digest` (decision L8) |
| §3.4 names | Both mode tables exact | 3 | `test_zone_names_are_keyed_by_mode` |
| §3.4 columns | Eight columns, nulls, both modes | 3 | `test_riverscape_zones_gdf_...`, `test_occurrence_zones_gdf_...`, `test_empty_zone_mask_still_returns_the_full_column_set` |
| §3.5 product | Added to `SpatialProduct`/`SPATIAL_PRODUCTS` | 3 | `config.py` diff |
| §3.5 gate | Requested + bundle present; else preflight error | 3 | `test_riverscape_evidence_without_a_bundle_fails_preflight`; `test_evidence_requested_on_an_occurrence_run_fails_preflight` |
| §3.5 layers | Six files, dtypes, nodata, grid | 3 | `test_evidence_filenames_and_dtypes_match_the_locked_contract`; `test_writing_evidence_produces_six_grid_aligned_rasters` |
| §3.6 bridges | Same gate; `_empty_bridge_frame` schema; empty layer OK; CRS | 3 | Four `_channel_bridges_geodataframe` tests + `test_empty_bridges_write_an_empty_gpkg_layer` |
| §3.7 plumbing | `analyze_from_dea` → resolve → finalize → manifest | 4 | `tests/integration/test_riverscape_exports.py` (three acceptance tests) |
| §3.7 reasons | `_riverscape_degraded_reasons` no longer discarded | 4 | `zoning_reasons=riverscape_degraded_reasons` in the finalize call; `test_mode_off_run_writes_a_thin_zoning_section` |
| §4 | Riverscape zones do not multiply metrics; guards unchanged | 4 | Gating test; `guards/scientific.py` not in any diff |
| §5 | Five error cases | 3, 4 | No-bundle preflight, gridless preflight, bridges CRS raise, thin zoning with reasons only, no `RiverscapeSourceUnavailable` catch in export code |
| §6 | Unit + gating + integration + regression; no live network | 1–4 | 4 new test modules + 2 modified; every source call monkeypatched or hand-built |
| §7 | Docs: zones, evidence, bridges | 4 | Task 4 Step 5 |
| §8 | File touch list | all | "File Structure" table |
| §10 | Acceptance criteria | 4 | Three acceptance tests + two hermetic-default tests + import-boundary re-run |

### Mutants this plan kills

| Mutant | Killed by |
|---|---|
| Discard the `LandformResult` again (return a fresh empty one) | `test_with_landform_returns_the_landform_build_landform_produced` |
| Skip `with_grid`, leaving `landform.grid is None` | `test_with_landform_attaches_the_zone_grid_to_the_landform_result`, `test_riverscape_evidence_with_a_gridless_bundle_fails_preflight` |
| Re-run `classify_hydroperiod` (or use `crosstab // 10`) for `hydroperiod_classes` | `test_from_zoning_recovers_the_exact_hydroperiod_array_from_the_crosstab` (code 4 on the bridged pixel is the discriminator) |
| Build a bundle on the occurrence path | `test_mode_off_produces_no_bundle`, `test_auto_occurrence_fallback_produces_no_bundle`, `test_occurrence_zone_result_is_rejected` |
| Leave `_ZONE_NAMES` mode-blind | `test_zone_names_are_keyed_by_mode` |
| Rename a provenance key in `zoning` | `test_riverscape_section_copies_every_provenance_key_verbatim` |
| Invent a second mask-digest algorithm | `test_riverscape_domain_fields_use_provenance_and_the_shared_digest` |
| Emit riverscape keys as empty defaults on an occurrence run | `test_occurrence_section_is_the_thin_subset_and_omits_riverscape_keys` |
| Fabricate a `domain_digest` when there are no zones | `test_no_zone_result_never_fabricates_a_domain_digest` |
| Write evidence without checking for the bundle | `test_riverscape_evidence_without_a_bundle_fails_preflight`, `test_evidence_requested_on_an_occurrence_run_fails_preflight` |
| Write `zone_crosstab` as uint8, or write `landform` into that band | `test_zone_crosstab_raster_is_uint16_and_equals_landform_times_ten_plus_hydroperiod` |
| Silently reproject foreign-CRS bridges | `test_bridges_in_a_foreign_crs_are_refused` |
| Skip the bridges layer when the frame is empty | `test_empty_bridges_write_an_empty_gpkg_layer`, `test_absent_bridges_frame_falls_back_to_the_empty_schema` |
| Add `riverscape_evidence` to `_default_config` | `test_default_config_does_not_opt_into_riverscape_evidence` |
| Let riverscape zoning change metric output | `test_analyze_output_identical_whether_or_not_riverscape_zones_were_computed` |
