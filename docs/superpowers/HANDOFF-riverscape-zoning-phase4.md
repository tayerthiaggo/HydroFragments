# Handoff: Riverscape Zoning — Phase 4 and Beyond

**Written:** 2026-09-15, end of session that completed Plans 1-3 (design spec Phases 0-3).
**For:** whichever agent/session picks this up next to plan and implement Phase 4 onward.
**Repo state at handoff:** branch `development`, HEAD `e97ff3c`, pushed to `origin/development`. Clean except untracked `examples_out/`/`output/` scratch dirs (leave them, unrelated).

## Read these first, in order

1. `docs/superpowers/specs/2026-09-14-riverscape-zoning-design.md` — the approved design spec. **§10 is the phase table** (source of truth for what's done and what's next). §4.2 steps 4-6 are Phase 4's actual algorithm content.
2. `docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md` — real measured numbers from Fitzroy basin data (AHGF offset, bare-cover medians, UpstrDArea distribution). Phase 4's `bare_threshold` calibration and bridge-cost weights should cite this, not guess.
3. `.superpowers/sdd/progress.md` — the full SDD ledger for this project. Read it to avoid re-deriving what already happened; trust it over your own memory after any compaction.
4. `docs/superpowers/plans/2026-09-15-riverscape-loaders-corridor-centreline-rem.md` — Plan 3, the most recently executed plan. Read this as a **template** for how much code-level detail a plan needs and how tasks should be scoped — Phase 4 is comparably complex and should get the same treatment (exact function signatures, exact test code, no placeholders).

## What's done (Phases 0-3, design spec §10)

- **Phase 0-1**: `hydrofragments/riverscape/domain.py` (`wet_domain`), `hydrofragments/hydroperiod/classify.py` (`classify_hydroperiod`), `hydrofragments/spatial/zones.py` (`combine_zones`, mask-derived `emitted_zones`/`has_zone_1`).
- **Phase 2**: `data/fitzroy_basin_aoi.geojson`/`fitzroy_basin_drainage.gpkg` (full basin, 31,318 reaches), `hydrofragments/config.py`'s `RiverscapeConfig` (all fields except `bridge_cost_weights` — see Deferred below).
- **Phase 3**: `hydrofragments/io/riverscape_sources.py` (`load_dem`, `load_fc_percentiles`, `load_waterbodies`), `hydrofragments/riverscape/corridor.py` (`measure_corridor_widths` → `CorridorResult`), `centreline.py` (`build_centreline` → `CentrelineResult`), `terrain.py` (`build_rem` → `TerrainResult`).

None of Phase 3's outputs are wired into anything yet — they're built and tested in isolation, consumed only by their own test files. Phase 4 is where they first get used together.

## What's next — Phase 4 (design spec §4.2 steps 4-6, §10 row 4)

**Channel rules, waterbodies, gap bridging.** Per the code-layout table in spec §5:
- `hydrofragments/riverscape/evidence.py` — `RiverscapeEvidence` (frozen dataclass), per-pixel evidence bits: `W_high` (`freq >= f_chan_high`), `S` (inside riverine DEA Waterbodies polygon), `B` (bare percentile ≥ calibrated threshold, ≥ `bare_year_fraction` of years), `T` (`REM <= h_chan_m` or trough depth ≥ `trough_depth_m`), `C` (connected to centreline).
- `hydrofragments/riverscape/channel.py` — ordered named rules with `RULESET_VERSION`, first-match-wins (spec gives examples: `C & (W_high | S)`, `C & B & T`, `C & (B | T) & adjacent_to_accepted`), `channel_confidence` = count of agreeing evidence families (0-3), in-channel when `>= min_channel_confidence`. Growth capped at `width_growth_factor ×` median seed half-width per reach.
- `hydrofragments/riverscape/waterbodies.py` — merge DEA Waterbodies polygon fragments cut at tile edges, classify each `riverine` vs `off_channel` by centreline overlap and elongation.
- `hydrofragments/riverscape/bridging.py` — `find_gaps(channel, centreline, drainage) -> list[Gap]`, `bridge_gaps(gaps, cost_inputs, cfg) -> BridgeResult`. Least-cost path (`skimage.graph.MCP_Geometric`) between two observed channel segments on the same AHGF reach path. Cost weights: terrain/water/bare/green/npv/line_distance (this is exactly `bridge_cost_weights`, currently NOT a `RiverscapeConfig` field — see Deferred).

**Test fixtures spec §8 assigns to this phase** (not yet built — the earlier phases' tests covered only §8's "offset, anabranch, pit" items):
- Rarely-wet (~1%) sand bed with B and T → in-channel
- Billabong joined by one-pixel low-frequency link → off-channel riverine
- Bare scald off the trough → not channel
- W + S count as one evidence family (not two)
- Channel broken by a 300m vegetated gap between two observed segments → bridged, `gap_cause=vegetated`
- Sub-pixel reach (anchor widths 1px) → bridged, `gap_cause=narrow`
- Dangling end (no downstream observed anchor) → not extended
- Gap longer than `bridge_max_length_m` or path over impassable ridge → `gap_unbridged` with reason
- Bridge never overwrites observed off-channel domain pixels

## Critical process learning from this session — DO NOT SKIP MUTATION TESTING

This is the single most important thing to carry forward. **Twice** in Plan 3, a task's tests passed on first submission while the algorithm was subtly wrong in a way that a plausible-looking test could not catch:

1. Task 3's anabranch-loop detector (`centreline.py`) used a mathematically-well-formed Euler-characteristic pixel-graph cycle check that flagged *every* ordinary channel corner as a loop — verified by a reviewer who reverted the fix and found all 32 existing tests still passed.
2. Task 4's REM (`terrain.py`) had two headline tests (PAVA pit-resistance, confluence min-capping) that a reviewer proved non-discriminating by literally swapping the real algorithm for the design it was explicitly chosen over (`np.minimum.accumulate`, `max()`/mean instead of `min()`) — every test stayed green.

Then, in the **final whole-branch review alone**, three more rounds of real bugs surfaced in one function (`terrain.py`'s `_linear_parts`, handling `MultiLineString` reaches from AOI clipping): a crash, then a fix that introduced a *worse* bug (GEOS `linemerge` silently reorders disjoint parts, reversing the along-stream profile), then two more edge cases (empty `LineString`, `GeometryCollection`) that still crashed after that.

**What worked**: dispatching reviewers with explicit instructions to *run mutation tests* — swap the real algorithm for the specific alternative design it was chosen over, or flip a boolean/sign/comparison operator, and confirm the test suite actually goes red. "Tests pass" and "the algorithm is right" are not the same claim, especially for real numerical/graph code (isotonic regression, distance transforms, topological sort, loop detection). When dispatching implementers or reviewers for Phase 4's evidence-fusion and bridging logic (which is *more* algorithmically dense than Phase 3), explicitly ask for this. See any of the Task 3/Task 4 review dispatch prompts in this session's transcript, or the final-review dispatch prompts, for the exact phrasing that worked.

## Workflow used this session (repeat for Phase 4)

1. `Skill: superpowers:brainstorming` if the spec doesn't already cover the phase at code-level detail (it mostly does for Phase 3; check whether §4.2 steps 4-6 are detailed enough or need a fresh brainstorm — they're less mechanically specified than corridor/REM were, more likely to need real design discussion, e.g. exact rule ordering, exact bridging cost function tuning).
2. `Skill: superpowers:writing-plans` — write a plan to `docs/superpowers/plans/YYYY-MM-DD-<name>.md` with full code in every step, following Plan 3's structure exactly (Global Constraints section, File Structure table, per-task Interfaces block naming exact signatures for cross-task dependencies).
3. `Skill: superpowers:subagent-driven-development` — dispatch fresh implementer per task (Sonnet, per this project's established preference — user explicitly asked for "Sonnet implement, Opus review" on both Plan 2 and Plan 3), dispatch fresh reviewer per task (Opus), loop fix→re-review until clean, update `.superpowers/sdd/progress.md` after each task.
4. Final whole-branch review (Opus, most capable model) after all tasks — expect to iterate; this session needed 4 rounds on the final review alone. Don't be surprised if Phase 4 needs similarly many, given it's more complex than Phase 3.
5. `Skill: superpowers:finishing-a-development-branch` at the end — this repo works directly on `development` (no separate feature branches per plan), push to `origin/development` when done. Confirm with the user before pushing (this session asked each time).

## Deferred items (from Plans 1-3, still open)

- **`bridge_cost_weights`** — spec §6 lists it as a `RiverscapeConfig` field (terrain, water, bare, green, npv, line_distance weights); it was never added, deliberately, because Plan 3 doesn't implement bridging. **Phase 4 needs this field** — add it when `bridging.py` is built, with the non-negative/at-least-one-positive validation spec §6 requires.
- `_fraction`/`_percentage` helpers in `config.py` use closed intervals `[0,1]`/`[0,100]`; spec §6 wants open `(0,1)`/`(0,100)`. Repo-wide pre-existing pattern, not Phase-3-specific; still open.
- Several `RiverscapeConfig` numeric fields raise bare `ValueError`/`OverflowError` instead of `ConfigError` on malformed (non-numeric) input, mirroring a pre-existing pattern elsewhere in `config.py`. Still open.
- GitHub Dependabot flagged 1 moderate vulnerability on `development` — nobody has looked at it yet.
- Minor items recorded in `.superpowers/sdd/progress.md` throughout Plans 2-3 (unused `f_seed`/`pixel_m` params in `corridor.py`/`centreline.py`, dead `distances` variable in `terrain.py`, no 3+-part MultiLineString ordering test, `CentrelineResult` lacking a `degraded_reasons` field while `CorridorResult`/`TerrainResult` have one, DRY opportunity for the `rasterize(...).astype(bool)` pattern duplicated across all three riverscape modules) — low priority, listed here so they're not lost, worth sweeping up whenever Phase 4 next touches these files.

## Key semantic gotchas (would waste time to rediscover)

- **`CorridorResult.widths_m` is a radius, not a diameter.** `centreline.py` uses it correctly as `geometry.buffer(width_m)`. This was implemented backwards in `terrain.py` (halved) and caught only in the final review — if Phase 4 reads this value, treat it as a one-sided distance from the line, matching centreline's usage.
- **`hydrofragments/riverscape/` must never import `hydrofragments.spatial`** (guarded by `tests/riverscape/test_riverscape_import_boundary.py`, an AST-based `rglob` scan — it will catch new files automatically). Every riverscape function takes already-validated `drainage` as a plain `GeoDataFrame`; validation (`validate_drainage_topology`, `create_channel_context`) is the caller's job, always in `hydrofragments.spatial.context`, never imported into `riverscape/`.
- Drainage geometry can legitimately be `LineString`, `MultiLineString`, `Point` (a reach that only touches the AOI at one point), or `GeometryCollection` (a reach crossing one AOI polygon part and touching another) after `create_channel_context`'s clipping. `terrain.py`'s `_linear_parts` function is the reference implementation for handling all four cases correctly — copy its pattern rather than re-deriving it if Phase 4 needs similar geometry handling.
- `bare_threshold` is **per-run calibrated**, not a config constant (spec §4.2 step 4) — `bare_threshold_floor_pct` in `RiverscapeConfig` (currently 30.0) is only the degraded-path fallback floor, used when too few pixels exist to calibrate. Don't wire it in as the primary threshold.

## User preferences observed this session

- Model split: **Sonnet 5 implements, Opus reviews** — explicit instruction, held across two plans.
- Caveman-mode communication (terse) was active throughout — check if still active; it's a slash-command toggle, not necessarily persistent across sessions.
- User wants to be asked before any push (`git push origin development`), even though the pattern is now established — asked and confirmed both times rather than assuming.
- User is comfortable with multi-round review/fix cycles when real bugs are found — did not push back on the 4-round final review for Plan 3, engaged with `AskUserQuestion` decisions on plan-mandated deviations (e.g. CRS-fallback strictness in Plan 2, `bridge_cost_weights` deferral in Plan 2's final review) promptly.
