# Specification: Provenance-Aware Inputs and Pool-Resolution Site Nodes

**Date:** 2026-08-19
**Status:** Draft proposal — NOT approved. Requires a decision gate before implementation.
**Package version:** targets HydroFragments `0.1.0` (post-`0.1.0` work)

This document proposes two related additions to HydroFragments. It is a
forward-looking draft, unlike
`docs/superpowers/specs/2026-08-12-dynamics-and-spatial-export-design.md`,
which records frozen shipped contracts.

Section 5 conflicts with an approved gate (U8) and cannot be implemented
without reopening or extending that decision. Nothing here is authorized by
this document alone.

---

## 1. Origin and scope

This design was extracted from a WaterMask-TSFill session that mistakenly
began specifying pool post-processing inside that repository. WaterMask-TSFill
declares connectivity metrics and pool-point tracking out of scope in its
`docs/paper-scope-and-method-boundary.md`; those specs were withdrawn and the
work reassigned here, where the relevant machinery already exists.

Scope is operational: input contracts, node identity, and metric support.
No publication claim is made or implied. Scientific novelty is explicitly
**not** argued in this document and must not be inferred from it.

---

## 2. What already exists (do not rebuild)

An earlier draft proposed building per-month pool labeling from scratch. That
was redundant. The following already ship and must be reused:

| Capability | Location |
|---|---|
| Per-month component labeling | `hydrofragments/patches/labels.py`, `components.py` |
| Patch morphology (`area_m2`, `perimeter_m`, `major_axis_length_m`, widths, bbox) | `hydrofragments/patches/morphology.py` (`PatchProperties`) |
| Patch metrics (pool count, LPI, AWRe, AWMSI) | `hydrofragments/metrics/patches.py` |
| Month-local pool identifiers | `hydrofragments/output/vectors.py` (`build_pool_id`) |
| Persistence / refuge area / occurrence | `hydrofragments/metrics/persistence.py` |
| Refuge spatial stability, reconnection timing, extent contraction | `hydrofragments/metrics/dynamics.py` |
| Fixed-graph structural connectivity (RC, TCF) | `hydrofragments/metrics/connectivity.py` |
| HY anchors (`peak_month`, `end_dry_month`, `confidence`) | `hydrofragments/temporal/hydroyear.py` |
| Window streaming under a byte budget, checkpoints | `hydrofragments/analysis/window_stream.py` |

Two existing properties constrain everything below:

- `build_pool_id(date, window_id, label_id)` is **month-scoped and
  window-scoped**. A pool straddling an active-window boundary is currently
  two records. Any site derivation must handle this explicitly rather than
  assume catchment-global labels.
- `refuge_spatial_stability` is a **scalar Jaccard between consecutive
  end-dry refuge masks on common-valid support**. It is a raster-overlap
  statistic, not object identity. It answers "how much did the refuge
  footprint move" and cannot answer "which pool persisted".

---

## 3. Gap 1: filled pixels are counted as observations

### 3.1 Current behaviour

`input.kind` already accepts `"watermask_tsfill"`
(`hydrofragments/config.py:386`). The validity policy is
(`hydrofragments/io/validity.py`):

```python
def apply_validity_policy(da_in):
    # Valid = 0 or 1. Invalid = 254 (outside AOI) or 255 (unobserved)
    return (da_in == 0) | (da_in == 1)
```

WaterMask-TSFill's product is a *gap-filled* mask. Every reconstructed pixel
is written as `0` or `1`. Under the policy above, a reconstructed pixel is
therefore indistinguishable from a directly observed one and is counted as a
valid observation.

WaterMask-TSFill separately exports `observed`, `confidence`, and
`method_flag` bands that carry exactly this distinction. HydroFragments
currently ingests only the water mask and discards them.

### 3.2 Operational consequence

Every metric that divides by valid-observation support, or that gates on
observation sufficiency, is affected. Concretely:

- `TCF` = `active_months / valid_months`. Filled months inflate the
  denominator and, where filled wet, the numerator.
- `refuge_area` and `occurrence` are computed over a support that includes
  reconstructed pixels.
- `EdgeFlag.LOW_VALID_OBS` and `LOW_COMMON_VALID_SUPPORT` will not fire for a
  month that was mostly reconstructed, because reconstruction removed the
  invalid values that would have triggered them.
- `refuge_spatial_stability`'s "common-valid support" can be satisfied by
  reconstructed pixels on both sides of the comparison.

None of these are wrong given the current input contract. They are wrong
given a gap-filled input, and the current contract cannot express the
difference.

### 3.3 Proposed contract change

Extend the `watermask_tsfill` input kind to optionally accept companion
bands, via the existing `input.variable_map`:

- `observed` — boolean; pixel was directly observed rather than reconstructed.
- `method_flag` — categorical provenance (WaterMask-TSFill `contracts.py`).
- `confidence` — `0..100`, `255` = N/A. Not a calibrated probability;
  WaterMask-TSFill documents this explicitly and the caveat must be carried,
  not dropped.

Introduce a distinction the current code does not have:

- **valid** — a value is present and usable (today's meaning; unchanged).
- **observed** — that value came from a direct observation.

Where the companion bands are absent, `observed` defaults to `valid` and
behaviour is bit-identical to today. This keeps `generic_binary` and existing
`watermask_tsfill` runs unaffected.

Metric support counts should then be reported on both bases, or on an
explicitly configured basis, rather than silently on the filled basis. Which
of those two is correct is an open question (§7).

### 3.4 Required new flags

Add to `EdgeFlag` (metric row schema bump to `1.2.0`; no column changes):

- `low_observed_support` — support threshold met only by counting
  reconstructed pixels.
- `reconstructed_dominant` — a reportable value whose support is majority
  reconstructed.

---

## 4. Gap 2: no pool-resolution persistence

`TCF` answers persistence at drainage-reach resolution: node identity is a
reach `HydroID`. A reach may contain several hydrologically distinct pools.
Reach-level persistence therefore cannot say *which* pool within a reach
persisted, or whether a reach's persistence is carried by one stable pool or
by different pools in different years.

Nothing currently in the package answers that. `build_pool_id` is month-local;
`refuge_spatial_stability` is a raster overlap.

---

## 5. Proposed: pool-site node source (conflicts with gate U8)

### 5.1 The conflict, stated plainly

`hydrofragments/metrics/connectivity.py` locks node identity (U8, approved
2026-07-17, spec §6.13/§6.11):

> Nodes are drainage reach `HydroID` values ... never derived from monthly
> water-mask patch labels. There is no transient monthly patch identity in
> this module — a node exists for the whole series or not at all.

This proposal adds a data-derived node source. It must not be implemented by
weakening that contract.

### 5.2 Why the exclusion was correct

The exclusion targets **transient monthly patch identity**, and it is
correct. Chaining month-to-month patch overlap into lineages fails on
intermittent rivers: at the wet peak the channel connects into one component,
so transitive closure over overlap edges merges every pool upstream and
downstream into a single identity. The result is one catchment-wide id per
record. Any design based on chained overlap lineages is unusable and must be
rejected.

### 5.3 Sites are not transient patches

Define a **site** as a fixed location that exists for the entire series
whether wet or dry — the same ontology as a reach node, at finer resolution.

Derivation, which must occur once, before any monthly iteration:

1. Take every HY end-dry anchor in the series (`detect_hy_anchors`,
   `hydrofragments/temporal/hydroyear.py`).
2. Label components at each anchor month using existing `label_components`.
3. Reconcile across active-window boundaries so a pool spanning windows is
   one site, not two (see §2).
4. Take the union across **all** anchors in the series as the site set.
5. Fix the site set. It is not revised by any later monthly pass.

Because the site set is a function of the whole series, it satisfies U8's
"exists for the whole series or not at all" contract. A site derived from a
2004 anchor exists as a node in 1988 — dry, valid, and reportable — exactly
as a structurally-dry reach would be.

This also resolves the case that breaks lineage chaining: a pool that dries
completely and refills is one site with a dry interval, not a death followed
by a birth.

### 5.4 Attribution, not identity transfer

Per month, each labeled component is **attributed** to the sites it covers.
Attribution is many-to-one and is a relation, never an identity claim:

- At end-dry, typically one component per site.
- At peak, one component may cover many sites. This is recorded as those
  sites being inundated within one connected body. It must never be recorded
  as those sites being one pool.

Site-level connectivity then falls out of the existing model without new
graph machinery: two sites are connected in month `t` iff the same component
covers both. This is the same shape as the existing edge rule and is
compatible with `connectivity_gap_threshold`.

### 5.5 Integration seam

`node_source` is already plumbed as a first-class concept:

- `hydrofragments/config.py:138` — `channel.node_source` config field
- `hydrofragments/models.py:81` — `node_source` model field
- `hydrofragments/schema.py:132,185` — `node_source` metric-row column
- `hydrofragments/metrics/connectivity.py:96` — the only place the value is
  pinned, hardcoded to `"external_network"` inside `build_fixed_graph`

`FixedGraph`'s docstring already says `node_source` is *"always
`external_network` in v1.2"*, which anticipates alternatives rather than
forbidding them.

Proposed: add `build_site_graph(...) -> FixedGraph` with
`node_source="pool_sites"`, leaving `build_fixed_graph` untouched.
`FixedGraph`, `compute_tcf`, and RC consume it unchanged, since they depend
only on `nodes`/`edges` and on node identity being series-stable.

The reach graph remains the default and remains authoritative for structural
connectivity. The two node sources coexist; results are labeled by
`node_source` and must never be compared across sources without saying so.

### 5.6 Open architectural question

Reach edges come from drainage topology (`From_Node`/`To_Node`). Site edges
have no equivalent external topology. Candidate rules — channel-network
adjacency, along-centreline neighbours, or the observed
co-occurrence-in-one-component relation from §5.4 — are not evaluated here.
This must be settled before implementation; §5.4 gives site connectivity per
month but not a fixed edge set.

---

## 6. Non-goals

- No morphological bridging, dilation, erosion, DEM-guided reconnection, or
  any modification of pixel classes. Sites are derived from the mask as
  given.
- No change to `build_fixed_graph`, the reach node set, or existing RC/TCF
  results.
- No claim that site co-occurrence in one component is observed hydraulic
  connectivity.
- No publication or novelty claim (§1).
- No modification of the WaterMask-TSFill repository.

---

## 7. Open questions requiring decisions

1. Should support counts default to the observed basis or the filled basis,
   and is that configurable or locked? (§3.3)
2. Does a metric computed on majority-reconstructed support get suppressed
   like `low_df`, or reported with a flag? (§3.4)
3. What is the fixed edge rule for site nodes? (§5.6)
4. Minimum site size, and how it relates to WaterMask-TSFill's own cleanup
   parameters, which currently determine the smallest surviving component in
   its output.
5. Cross-window site reconciliation: correctness and cost under the existing
   streaming byte budget. (§5.3 step 3)
6. Whether an anchor month that is majority-reconstructed is admissible for
   site derivation at all.

---

## 8. Validation required before any gate approval

- Site derivation is deterministic and reproducible from identical inputs.
- Site set is invariant to active-window chunking, including for pools
  straddling boundaries.
- A synthetic series where the channel fully connects at peak yields a stable
  multi-site set, demonstrating the §5.2 collapse does not occur.
- A pool that dries fully and refills remains one site.
- With companion bands absent, all existing metric outputs are bit-identical
  to `0.1.0`.
- With companion bands present, support counts and new flags change only
  where reconstruction actually occurred.
