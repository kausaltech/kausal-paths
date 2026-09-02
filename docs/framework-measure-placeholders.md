# Framework measure placeholders: how the lookup works, and why it keeps breaking

*Produced by Claude Opus 5.0 on 2026-09-02.*
*Responsible: Jouni Tuomisto.*

Written after the second silent failure of this feature, as background for a refactoring of the
framework measure system. It is not a change proposal. It describes the mechanism, the two
regressions it has suffered, the structural reason it is fragile, and what a refactoring would have
to preserve.

## 1. What the feature is

On the *Additional Historical Data* tab of nzc-data-studio, a city fills in what actually happened
in each year since its baseline. Every cell it has **not** filled shows, in grey, the value the
model currently plans for that year — a hint, not data.

Those hints are `Measure.placeholderDataPoints` in the GraphQL schema. Producing one means
answering a question that turns out to be surprisingly hard:

> Which node in this city's computation graph holds the series for MeasureTemplate `<uuid>`?

Everything difficult about this feature follows from that question having no recorded answer.

## 2. The mechanism

Two stages, in two different files.

**Stage 1 — build a uuid → node map** (`FrameworkConfig.measure_template_uuid_to_node_dimension_selection`,
`frameworks/models.py`). Walk every node in the instance. For each, collect the measure uuids it
claims, by one of two routes:

- **Legacy**: the node *is* a `gpc.DatasetNode`, a thin wrapper over one dataset that holds a
  `UUID` column. Matched on concrete type.
- **Binding**: the node has an input dataset binding tagged `city_data`. The uuid lives in the
  binding's source frame.

A uuid claimed by more than one node then goes through a tie-break: binding tag (`historical` beats
untagged beats `goal`, **per node**), then level-over-delta, then node-id heuristics (`*_observed`
wins, `*_historical` loses), then a graph walk that redirects `goal → action → output`.

**Stage 2 — read the values** (`MeasureType._get_placeholder_df` / `_narrow_to_placeholders`,
`frameworks/schema.py`). Compute the selected node under the default scenario, select the metric
column the binding named, convert units to the template's, cut to the year window
(`reference_year < year <= today`), filter to the selection's dimension categories, and refuse if
more than one row per year survives.

### The load-bearing asymmetry

**The binding is how a measure is *found*. It is not where its values live.**

Under the legacy route those were the same object: a `DatasetNode` wrapped one dataset, so the
`UUID` column and the values sat in one frame, and identifying a measure and serving it were a
single act. Under the binding route they are two different frames — the uuid is in the binding's
*source*, the values come from `node.get_output_pl()`.

Almost every complication in the current code is a consequence of that split:

| Piece | Exists because |
| --- | --- |
| `get_uuid_frame()` | The overlay that folds in the city's own data drops `uuid`, so the read has to stop short of it |
| Dimension-category matching, containment rule | Node output carries no uuid, so **categories are the only handle left** on a measure |
| `_prefer_historical_bindings` | One dataset holds one uuid column beside several value columns, so several bindings claim one uuid |
| `metric_col` | A multi-metric node renames `Value`, so there is nothing to read under the usual name |
| Unit conversion | On the legacy route the node's unit *was* the client's; on the binding route it can differ by scale |
| `_prefer_the_full_trajectory` | A goal node holds only the target end of a series, so it has nothing in the window |
| `_claimed_uuids` | The map is one `cached_property` for a whole framework config, so one unreadable source would blank every cell |

None of these is gratuitous. All of them are downstream of one missing fact.

## 3. The two regressions

Both produced an empty `placeholderDataPoints`, which the UI renders as a blank cell. **A blank
cell is also what "the city has not filled this in" looks like.** Neither failure raised anything.

**Regression 1 — the model swap (late June).** The node lookup matched on concrete type:

```python
# Intentionally test for concrete type, filter out subclasses
if type(node) is not DatasetNode:
    continue
```

Replacing the NZC model with the lucia-derived one took `configs/nzc.yaml` from six live
`gpc.DatasetNode` entries to zero, and added 127 `city_data` tags. Nothing matched; every measure
resolved to a null node. This was an *incomplete migration*, not config drift:
`get_measure_datapoint_years()` was moved to the tagged-binding world eight days later, and this
call site was left behind. It was the last exact-type `DatasetNode` check in the repo.

**Regression 2 — the overlay moved (discovered 2026-09-02).** The fix for regression 1 added
`get_uuid_frame()`, which read the binding's source and stopped "short of `post_process`" to keep
the `uuid` column. Correct when written. But `_override_with_measure_datapoints` — which drops
`uuid` — had since moved into `before_temporal_fill`, and `_filter_and_process_df` calls that hook
between its two transformation groups. So a method whose entire purpose was to preserve `uuid` was
routed through the one call that removes it. Every binding reported carrying no measures; the map
came out empty; the original bug was back, unchanged in symptom.

The cure is to run only the pre-temporal transformation group and stop at the hook.

### Why neither was caught

The fix for regression 1 shipped with 68 tests. Not one executed `get_uuid_frame`'s body: every
test either passed a `SimpleNamespace(get_uuid_frame=lambda: df)` or constructed a real dataset
with `_uuid_frame_loaded = True` pre-seeded, so that it "answers without reaching for DVC or the
database". The method whose correctness depended on the surrounding pipeline was stubbed in all 68.

This is the general lesson and it is worth stating plainly: **the seam between this feature and the
dataset pipeline is exactly where it breaks, and mocking that seam is what makes the breakage
invisible.** `src/frameworks/tests/test_placeholder_graphql.py` now hands the binding a payload
store instead of a memo, so the real read runs; it fails on both historical regressions, which was
verified by reintroducing each.

## 4. Delta versus level, and the target-year measures

An adjacent question, settled differently than it first appeared.

For action-node measures the model stores **cumulative change from the baseline year**, not a
level: `DatasetReduceAction` and `DatasetReduceAction2` both end on `col - pl.first(col)`. A
"share of heating from fossil fuels" cell would show `-3.7` — the planned *reduction* — where a
city expects the planned share. The general rule is:

> The cell should show the **level** of the column the measure names, which is `anchor + delta`.
> The raw delta equals the level **if and only if the anchor is zero.**

That test is per-measure, not per-class or even per-column: in the framework's own default data,
`share_of_new_buildings_built` holds one uuid anchored at `1.0` and another at `0.0`, so within a
single column one measure's delta is the level and the other's is wrong by the whole baseline.

In practice this is handled by the graph rather than by arithmetic in the resolver: for all five
delta actions carrying city data, every `historical`-tagged column is **also** bound by a
downstream level `GenericNode` with the same column, and `_prefer_a_level_over_a_delta` routes the
measure there. Converting delta to level inside the resolver was considered and rejected — it would
reconstruct in a consumer exactly what the level node already computes.

**The 21 withheld measures are a different thing entirely.** Every one is claimed *only* by a
`goal`-tagged binding on a target-year column, and their own metadata says what they are: sections
titled *"…in the target year"*, `default_value_source` reading *"forecast assumption – 2030"*. They
are 2030 inputs, and the tab asks about the years up to today. They have no historical series, and
their uuid landed on a goal column because that is the only column carrying it — not because the
target is the answer.

So the withholding is keyed on the **binding role**, not on `output_is_baseline_delta`. Both
criteria select the same 21 measures on real data, but only one of them says something true about
why. If it is ever decided that cities should see their 2030 assumption echoed on that tab, the
value to show is the goal itself, constant across the window — not the action's output.

`Node.output_is_baseline_delta` stays, and earns its place in the tie-break: with it 159 measures
map, without it 141. It is not what the withholding rests on.

## 5. The fragility, stated structurally

Three properties, in decreasing order of how much they matter.

**A measure's identity is reconstructed, never recorded.** The lookup infers it from dimension
categories, binding tags, node-id suffixes and graph shape. Node output carries no uuid; no binding
records which measure template it serves; no spec embeds one. Every rule in the tie-break is a
heuristic standing in for a fact nobody wrote down. This is the root cause, and the single change
that would collapse most of section 2's table is for **a binding to declare the measure template it
serves**.

**Failure is indistinguishable from absence.** A blank cell means either "no hint available" or
"the lookup broke". The UI cannot tell, the city cannot tell, and for weeks at a time nobody did.
Any refactoring should make an unresolved measure *say so* — `correspondingNode: null` already
carries the signal and nothing surfaces it.

**Categories cannot always discriminate.** Two uuids in one column with no distinguishing category
collapse to the empty selector and eliminate each other; ~40 measures per city are unmapped, 44
uuids dropped by the containment rule on one measured instance. Sometimes the real discriminator is
the *year role* (a current value at the reference year versus a target at the target year), which
the category rule cannot see. Measured, that recovers only 2 of 44 — so it is not worth building
generally, but it shows that categories are the wrong primary key.

## 6. What a refactoring must preserve

Verified behaviour, worth keeping as an acceptance list:

- **159 measures map** on a real NZC city; **21 withheld** as target-year; **134 resolve to a full
  seven-year series** with no duplicate years and no compute errors.
- **Waste treatment shares sum to exactly 100 %** within each of the six waste types. This is the
  strongest single check available — it breaks immediately if a wrong node or a crossed series is
  picked, and it is cheap to re-run.
- Counts are **identical across cities**, since every NZC instance loads the same
  `configs/nzc.yaml`. A per-city difference in the number of resolved measures is a bug.
- Renaming a node must not change the numbers a city sees. The `goal → action → output` walk reads
  the graph precisely so that a `_goal` suffix is not load-bearing; the surviving node-id
  heuristics (`*_observed`, `*_historical`) are the remaining exception and are worth removing.
- One unreadable source must cost its own measures and no others.

## 7. Adjacent cleanups

Tracked in `data/TODO.md`:

- **45** — retire Athens (166 of 291 live `gpc.DatasetNode` entries), then migrate the remaining
  instances off the legacy action classes. Also: the legacy `DatasetNode` branch in this lookup is
  **unreachable** — no deployment has a live one — and is pinned by tests rather than by data.
  Three complications exist only to serve it.
- **46** — `gpc.SCurveAction` is superseded by `simple.SCurveAction`.
- **47** — replace `DatasetReduceAction2` and `DatasetDifferenceAction`; `gpc.DatasetDifferenceAction2`
  has zero uses anywhere and can probably just go.
- **48** — three measures show a blank cell to every city, and need itemising.

## 8. An operational note that cost an afternoon

Two things made the 2026-09-02 verification much slower than it needed to be, both worth knowing
before debugging this feature again:

- **`serverDeployment.gitRevision` is already in the studio's own GraphQL query.** Read it before
  interpreting any UI behaviour against a backend change. The tab was being served by a staging
  deployment 28 commits stale, which looked exactly like the fix not working.
- **An unauthorised GraphQL request returns `config: null` and `measure: null` with no `errors`
  entry at all**, because `cache.framework_configs` is permission-filtered rather than raising.
  That is indistinguishable from a broken resolver. Server-side `shell_plus` needs no auth and
  answers faster.

And locally: nzc-data-studio and kausal-paths-ui both want port 3000, but only `:3000` is a
registered OIDC redirect URI for the Data Studio client. Whichever starts second lands on `:3001`
and the login fails with a 500 served by the *other* application.
