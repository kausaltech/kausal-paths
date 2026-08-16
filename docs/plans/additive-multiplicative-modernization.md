# Modernizing AdditiveNode and MultiplicativeNode

*Produced by Claude Opus 5.0 on 2026-08-15.*
*Version 2 produced by Claude Opus 5.0 on 2026-08-16.*
*Responsible: Jouni Tuomisto.*

`AdditiveNode` and `MultiplicativeNode` are the oldest node classes in the codebase.
`GenericNode` was built to unify them, and it does — but the model editor draws a node's
*class* in its GUI, and "GenericNode" tells a modeller nothing about whether the node sums
or multiplies its inputs. So models should go back to the specific classes wherever the
specific class can express what the node does (`data/TODO.md` §19).

That only works if the specific classes are worth returning to. Today they are not:

- they predate the current join machinery and do not share it with `GenericNode` or
  `FormulaNode`;
- they treat an input *dataset* differently from an input *node*, even though both are
  just a `PathsDataFrame` by the time arithmetic happens;
- `MultiplicativeNode` cannot take a dataset at all — it loads one and ignores it.

This document specifies what the two classes should become, records what they do today
(measured, not assumed), and sets out the migration.

## 1. Objectives

1. **Nodes and datasets are interchangeable inputs.** Both are `PathsDataFrame`s to the
   node. Today `AdditiveNode` accepts 0–1 datasets and `MultiplicativeNode` accepts none.
2. **Both classes compute on the modern machinery** — the same joins and binary operations
   as `GenericNode` and `FormulaNode`, not their own hand-rolled versions.
3. **`AdditiveNode` has exactly one additive multiport.** Any number of inputs may connect
   if they comply: identical dimensions, compatible units, missing values count as zero.
4. **`MultiplicativeNode` has at least two single-input factor ports**, whose product takes
   the union of the incoming dimensions and joins inner; plus one additive multiport added
   to the product, which must match its dimensions and unit.
5. **The real models should already compute these results** — the rebuild is meant to
   change the code, not the numbers. Verified per model, not assumed.
6. **Dataset preprocessing should stop depending on the node class.**
7. **Staged as `AdditiveNode2` / `MultiplicativeNode2`**, migrating model by model, with the
   original names reclaimed at the end.

## 2. What the classes do today

Pinned down by `nodes/tests/test_add_multiply_semantics.py`, which asserts each of the
behaviours below. Where `GenericNode` should agree, the test asserts that it does; where it
does not, the test names the divergence.

| # | Finding | Where |
| --- | --- | --- |
| F1 | `AdditiveNode` takes 0–1 datasets; two is a hard error. `MultiplicativeNode` never reads `input_dataset_instances` in `_compute`, so a dataset bound to it contributes nothing, silently. One live instance of this: `athens/total_cost_of_ownership` binds `gpc/athens_nzc` and ignores it. (The same node in `configs/modules/transportation/transport_cost.yaml` is dormant — only `ev-policy` includes that module, and that instance does not load; `nzc.yaml` inlined the node as a `GenericNode` with no dataset.) | `nodes/simple.py:397`, `:896` |
| F2 | A dataset input is extended to the model end year (`extend_last_historical_value_pl`); a node input is not. The two are not interchangeable even where both work. | `nodes/simple.py:376` |
| F3 | Addition treats nulls as zero — **but only when no unit conversion happened first**. `ensure_unit` round-trips the column through numpy, where a polars null becomes `np.nan` and returns as NaN. NaN then poisons the sum, and `Node.check()` rejects NaNs outright. | `common/polars.py:502-504` |
| F4 | Multiplication with a null factor: `MultiplicativeNode` **drops** the row (`drop_nulls` after the product); `GenericNode` keeps it and emits NaN. Same input, two different wrong answers. | `nodes/simple.py:892` |
| F5 | Order-independence already holds for both classes — ragged years, ragged categories, disjoint dimensions, three factors reversed. Only column order and `primary_keys` order vary, so `how='left'` → `how='inner'` in `perform_operation` is a clarity fix, not a correctness fix. | `nodes/simple.py:887` |
| F6 | A plain `AdditiveNode` collects `non_additive` inputs into `na_nodes` and never uses them. Subclasses consume them; the base class drops them silently. | `nodes/simple.py:403` |
| F7 | `only_historical` is dead code — it reassigns `outputs` after that list has been consumed and never reads it again. `extend_rows` has zero config uses. | `nodes/simple.py:941-945` |
| F8 | Two interpolation mechanisms coexist: the per-binding dataset-level `_linear_interpolate` (opt-in, cache-keyed, available to every node class) and `GenericNode`'s `_add_missing_years` (unconditional, and additionally back-fills *leading* nulls). | `nodes/datasets.py:200`, `common/polars_ext.py:1020` |

F5 is the reassuring one: objective 5 largely holds. The rebuild is not expected to move
values, except where F1–F4 say the current behaviour is indefensible.

## 3. How much dataset preprocessing actually differs (objective 6)

Every YAML instance that loads (60 of 64) was scanned, and for each node↔dataset binding
the **raw** frame was inspected — with any `interpolate` flag temporarily switched off, so
the gaps interpolation is currently filling stay visible. A binding "would change" if the
raw data has interior year gaps or interior nulls.

| | bindings | interp on | interp off | off **and** gapped | on, nothing to fill |
| --- | ---: | ---: | ---: | ---: | ---: |
| additive family | 1155 | 358 | 797 | **140** | 105 |
| generic family | 1264 | 66 | 1198 | 254 | 64 |
| other classes | 783 | 99 | 684 | 201 | 33 |
| multiplicative family | 7 | 1 | 6 | 2 | 1 |
| **total** | **3209** | **524** | **2685** | **597** | **203** |

Read this as three facts:

- Interpolation-by-default is **not** a no-op. It would change values in 597 bindings,
  140 of them in the additive family.
- The `interpolate` flag is decorative in 203 of the 524 bindings that set it — the data
  has no gap to fill.
- 191 bindings have *leading* nulls (53 generic, 38 additive). Those are back-filled today
  only on the `GenericNode` path, via `_add_missing_years`. The dataset-level mechanism
  does not do it, so choosing the dataset-level mechanism drops that behaviour.

Four instances are excluded because they do not load at all: `ev-policy` (missing include
`configs/modules/transportation/active_mobility.yaml`), `helsinki` (`KeyError:
'default_language'`), `kpr` (`Value for field owner missing`), plus 15 individual datasets
that failed to load inside otherwise healthy instances.

## 4. Decisions

1. **Null × x = null.** The product keeps the row with a null value. Not dropped (today's
   `MultiplicativeNode`), not NaN (today's `GenericNode`). Downstream tolerates nulls;
   `Node.check()` rejects NaNs.
2. **Interpolation defaults on for the v2 classes only.** A dataset bound to
   `AdditiveNode2` / `MultiplicativeNode2` is interpolated unless the binding says
   `interpolate: false`. The v1 classes keep today's opt-in, so the 140 affected additive
   bindings change one model at a time, as each migrates, instead of all at once.
3. **Dataset extension becomes a per-binding flag, off by default.** A dataset input
   behaves exactly like a node input unless the binding asks to be extended.
4. **v2 keeps `inventory_only` and drops the rest.** Config usage: `inventory_only` 78,
   `use_input_node_unit_when_adding` 8, `drop_nans` 5, `fill_gaps_using_input_dataset` 4,
   `multiplier` 2, `replace_output_using_input_dataset` 2, `only_historical` and
   `extend_rows` 0. The ~21 sites using a dropped parameter are converted as their models
   migrate; each has a clearer expression as an explicit edge, a dataset processor or a tag.

Two more decisions came out of the pilot, where the migration hit behaviour `GenericNode`
performed silently:

5. **Leading nulls are back-filled only when a binding asks.** `_add_missing_years` copies
   each category's first known value backwards over the years before it — a back-cast, and
   the reason four longmont-dev series have values for years their data does not cover. The
   behaviour stays available as `backfill: true` on the binding, off by default. 191
   bindings across all instances have leading nulls, so this would otherwise have been a
   silent change in every one of them.
6. **Empty dimensions are dropped explicitly, not automatically.** When a dataset holds
   metrics of different shapes, selecting one column leaves the others' dimension columns
   behind with nothing in them. `GenericNode` dropped those via `_drop_unnecessary_levels`;
   v2 does not, and the binding says `filters: - column: <dim>` instead — the idiom already
   documented in `configs/cork-nzc.yaml`. Whether to make it automatic again is deliberately
   left open. Note that the wide format cannot represent a null category (`metric@dim:`
   collapses to `metric@`), so interpolating such a frame fails until the column is dropped.

## 5. The contract

**Operand.** A `PathsDataFrame` from either an input node or an input dataset. Role
assignment is identical for both: an explicit tag (`non_additive`, `additive`, `impute`)
wins; otherwise unit compatibility with the node's own unit decides — compatible means
additive, incompatible means factor.

**Dataset preprocessing**, shared by both classes: pick the metric column (the single
metric column, else the `metric` parameter, else the one unit-compatible column), then
apply whatever the binding asked for — `interpolate` (on by default for these classes),
`backfill`, `extend`, and any `filters` — convert to the node's unit, and treat exactly
like a node operand. Nothing is inferred: no `_drop_unnecessary_levels`, no leading-null
back-fill, no extension unless the binding says so.

### AdditiveNode2

One additive multiport taking any number of operands, plus the `impute` port.

- All operands must carry the same dimension set; a mismatch is an error.
- Units must be compatible; every operand is converted to the node's unit.
- Outer join over the index. A missing row **or** a null value counts as zero.
- `impute` operands overlay the result last.
- A `non_additive` operand is an **error**, not a silent drop (F6).

### MultiplicativeNode2

At least two single-input factor ports, one additive multiport, plus the `impute` port.

- Factors: inner join, output dimensions are the union of the factors' dimensions, units
  multiply and are then converted to the node's unit.
- A null in any factor propagates as null.
- The additive multiport is added to the product, and must match its dimensions and unit.
- `impute` operands overlay last.

## 6. Migration

Steps 0–4 are done, each verified by a full `test_instance` state comparison over the 53
instances that load and compute (the seven `*bisko`/`reutlingen` models were under active
edit for another project and are excluded; four instances do not load at all, and eleven
fail on `main` for reasons that predate this work).

**Step 0 — `ensure_unit` must stop destroying nulls.** ✅ `common/polars.py` converted via
`to_numpy()`, which maps null → `np.nan`; the null mask is now restored afterwards. Until
this was fixed, neither "null counts as zero" nor "null propagates" was reachable, and any
converted null reached `Node.check()` as a NaN and failed it. Fixing it also settled
decision 1 for free: `GenericNode`'s multiply already propagates nulls correctly, so the v2
class inherits the agreed behaviour from the shared machinery. *Diff: none, 59 instances.*

**Step 1 — the shared operand layer.** ✅ `nodes/operands.py` holds the rule that was
written out four times, reusing `PathsExt.add_with_dims` / `multiply_with_dims`,
`Node.add_nodes_pl` / `multiply_nodes_pl` and `Node.get_input_datasets_pl`.
`GenericNode._get_add_multiply_nodes` went from 36 lines to 3 by calling it. Two deliberate
changes rather than a byte-for-byte port: an input claimed by another operation's tag is
now *listed* (`claimed_elsewhere`) rather than dropped on the floor, and an unreadable unit
raises a `NodeError` that names the fix instead of tripping a bare `assert` inside
`is_compatible_unit`. *Diff: none, 53 instances.*

**Step 2 — the v2 classes.** ✅ In `nodes/simple.py` alongside the originals. Wiring them
into the editor turned up a coupling problem worth naming: `instance_parser` decided whether
a node has an additive multiport with `issubclass(node_class, AdditiveNode)`, so
`AdditiveNode2` would have rendered without one — the entire editor payoff. The question now
belongs to the class (`Node.additive_multiport_declaration(tags)`), which is what
`docs/architecture/principles.md` §6 asks for. *Diff: none, 53 instances.*

**Step 3 — binding flags.** ✅ `interpolate` (default flipped for the v2 classes), plus
`extend` and `backfill`, threaded through `instance_loader`, `instance_parser`,
`DatasetPortSpec` and `spec_export`, and applied in `Dataset.post_process` in the order
interior → leading → trailing. Two traps handled: a class default is *not* the legacy
`input_dataset_processors` entry (that one overrides a binding, a default yields to it), and
a class default must not manufacture a processor entry in the export, or parse and export
stop agreeing — `tools/parse_oracle.py` catches exactly that. All three flags are in the
dataset cache key, or flipping one would serve a stale frame. *Diff: none, 53 instances.*

> A defect in this step survived until the bisko migration: `spec_export` rebuilds an
> `InputDatasetDef` from the runtime dataset field by field, and it was not taught to read
> `extend` or `backfill`. Both flags would have been dropped on the way into the DB, so a
> DB-sourced instance would quietly compute something different from its YAML. The oracle
> caught it the first time a config actually set `extend` — which is the argument for
> running `parse_oracle.py` *after* a migration and not only after a code change.

**Step 4 — migrate, model by model.** ✅ for **`longmont-dev`**: 64 of its 74 convertible
nodes now run on the v2 classes, computing byte-identically — 292 comparisons across all
278 node outputs and action impacts, zero differences. (`data/TODO.md`
§19.2 proposed `*bisko` first; `longmont-dev` goes first instead because the bisko models
are under active edit for a separate project, and a migration diff is only readable
against a model that is otherwise still.) Capture `test_instance --state-dir` before each
model, convert, diff, and explain every value that moves.

`longmont-dev` was surveyed against the *loaded* instance — which resolves edges declared
as `output_nodes` elsewhere and the unit-driven add/multiply split, the two things a
YAML-only survey gets wrong. Of its 82 `GenericNode`s:

| bucket | count | notes |
| --- | ---: | --- |
| `AdditiveNode2` candidates | 45 | 34 are a bare dataset (`1ds+0add+0mul`); the rest mix a dataset with 1–4 additive nodes |
| `MultiplicativeNode2` candidates | 20 | 16 are two factor nodes; one has an additive side input; **three** (`waste_composted`, `waste_landfilled`, `waste_recycled`) are dataset × node |
| stays `GenericNode` | 17 | 9 are `GenericAction`/`ConstantNode` subclasses; 8 use an operation no v2 class expresses (`do_correction`, `extend_all`, `extend_to_history`, `drop_infs,drop_nans`, `add_to_existing_dims`, `inventory_only,extend_values`), or have only one factor |

The three dataset × node nodes are the pilot's proof case for objective 1: today
`MultiplicativeNode` cannot express them at all, which is why they are `GenericNode`s.

Only two operations across the whole model fall outside what a v2 class can express
(`do_correction`, `inventory_only`), so the conversion is nearly complete rather than
partial — a good sign for the classes' expressiveness, on one real model.

### What the pilot had to say out loud

Reproducing the old values needed four markers, each for something `GenericNode` did in
silence. The count is the useful part: it says how much of a model's behaviour was
invisible.

| marker | uses | what it makes explicit |
| --- | ---: | --- |
| `extend: true` | 38 | the dataset's last value carried to the model end year |
| `filters: - column: <dim>` | 14 | dimensions belonging to the dataset's *other* metrics |
| `backfill: true` | 4 | a series' first value back-cast into earlier years |
| `tags: [non_additive]` | 3 | a factor whose unit matches the node's, which the unit rule alone would add |

Two nodes stayed `GenericNode` for reasons rather than failures.
`electricity_replacing_natural_gas_residential` applies `multiplier: 0.33` to its single
input; decision 4 drops that parameter, and the honest conversion is an explicit factor in
the graph, which is a modelling change and not something to slip into a retype. The other
is `avoided_emissions_if_included`, which has one factor where a product needs two.

**Step 5 — reclaim the names**, still to do, once no config references the v1 classes.
Delete `nodes/tests/test_zz_scratch_order.py` then too — the scratch probe the semantics
test replaced.

### The bisko family

Migrated 2026-08-16, once the transport work on those models had settled. The nodes live in
two shared modules rather than in an instance, so a conversion has to be right for every
instance that includes them: the survey was run against each of the six loadable instances
(`bisko`, `augsburg-bisko`, `bayreuth-bisko`, `duesseldorf-bisko`, `mainz-bisko`,
`schwerin-bisko`) and only nodes classified identically in all of them were converted.
Nine nodes classified differently somewhere; every one turned out to be a city-level
override in that instance's own YAML, so the module definition it shadows converts safely.

**42 nodes converted** — 21 in `configs/modules/bisko/model.yaml` (15 additive, 6
multiplicative) and all 21 in `configs/modules/bisko/municipal_balance.yaml` (15 additive,
6 multiplicative). `municipal_balance.yaml` converted completely; `model.yaml` keeps 9
`GenericNode`s, all either subclasses (`BiskoChpNode`, `BiskoExergeticAllocationNode`,
`ConstantNode`) or users of `do_correction`, `split_dims` or `extend_all`.

**Verifying a config change needs the instance to be reading that config.** `bisko` and
`mainz-bisko` have `config_source='database'`, so `test_instance` loads their stored DB spec
and never opens the edited YAML at all — a comparison over them passes trivially and proves
nothing. `municipal_balance.yaml` is included *only* by `mainz-bisko`, so its 21 conversions
had no coverage whatsoever until this was noticed. Check `config_source` before trusting a
migration diff; the run log names the source (`Creating instance from YAML file: …`) and
says nothing when it came from the DB.

| instance | source | how verified | result |
| --- | --- | --- | --- |
| `augsburg-bisko`, `bayreuth-bisko`, `duesseldorf-bisko`, `schwerin-bisko` | yaml | `test_instance --all-nodes` | 600 comparisons, 0 diffs |
| `bisko` | database | YAML loaded directly, before vs after | 68 nodes identical |
| `mainz-bisko` | database | YAML loaded directly, before vs after | 79 identical, 11 failing in both |
| `reutlingen-klimabilanz` | yaml | YAML loaded directly, before vs after | 75 nodes identical |

Comparing a DB-sourced instance means loading its YAML by hand and diffing node outputs
**keyed by index, not by row position**: the v2 classes build `primary_keys` from a set, so
row order legitimately differs (finding F5). A positional comparison reports every reordered
row as a change — it showed 2397 of 3660 rows "differing" in `final_energy_use`, with the
tell-tale symmetry of 764 values becoming zero and 764 zeros becoming values. Keyed, the
node is identical.

The markers needed: 25 × `extend: true`, 4 × `backfill: true`, 2 × `tags: [non_additive]`,
and one removal — the six `municipal_*_emissions` nodes carried `params: {operations:
multiply}`, which `MultiplicativeNode2` refuses because it has no `operations` parameter at
all. The parameter was redundant on a class that multiplies natively, so dropping it *is*
the conversion. Worth checking for on any node being retyped: a v2 class accepts only
`metric` (and `inventory_only` on the additive one), and rejects everything else at load.
Both `non_additive` tags are the same trap the longmont waste nodes hit — a dataset that is
the left operand of a multiplication while carrying the node's own unit, so the unit rule
alone reads it as an addend. `district_heating_fuel_emission_factors` and `final_energy_use`
are the `1ds+1add+1mul` shape that §19.2 of `data/TODO.md` could not classify from YAML
alone and predicted would need the live graph; it does, and it is a product plus an addend.

`mainz-bisko` carries 22 pre-existing node failures (`Unknown categories in dimension
column 'road_type': total`, from `bisko/energy_shares`) that predate this work and are
unchanged by it. `reutlingen-klimabilanz` has no local `InstanceConfig` and was not tested.

### What is left

- Migrate the remaining models; the classifier runs against any loaded instance.
- Open the model editor on a migrated `longmont-dev` node and confirm the additive and
  factor ports render. Everything else is verified by state comparison, but this is the
  reason the work exists and only an eye can check it.
- Decide whether empty dimensions should be dropped automatically again (decision 6).
- Convert the ~21 config sites that use a dropped parameter, `multiplier` first.
- `MultiplicativeNode2.lower_to_pipeline_ir` still refuses datasets and additive side
  inputs, as v1 did. Lift when the pipeline IR gains a dataset binding.

## 7. Verification

```bash
python -m pytest --reuse-db nodes/tests/test_add_multiply_semantics.py \
                            nodes/tests/test_operands.py \
                            nodes/tests/test_dataset_binding_flags.py
python -m pytest --reuse-db nodes/ common/
python tools/parse_oracle.py -i <instance>                       # parse == export
python manage.py test_instance --state-dir <dir> --store         # before a step
python manage.py test_instance --state-dir <dir> --compare       # after it
```

Steps 0–3 are refactors and must produce an **empty** state diff across all instances.
A migration step is verified per model with `--all-nodes`, which compares every node's
output rather than only the outcome nodes — a value can move in the middle of a graph and
still wash out at `net_emissions`:

```bash
python manage.py test_instance --state-dir <dir> --all-nodes --store  --only longmont-dev
# convert nodes, then
python manage.py test_instance --state-dir <dir> --all-nodes --compare --only longmont-dev
```

Every moved value is a finding to explain rather than a rounding artefact. In the pilot,
each one turned out to be something `GenericNode` was doing silently, and the fix was to
say it in the binding.
