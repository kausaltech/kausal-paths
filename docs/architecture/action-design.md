# How an action attaches to the graph

*Produced by Claude Opus 5.0 on 2026-09-09.*
*Responsible: Jouni Tuomisto.*

An action is a node whose output is switched on and off by the scenario. Deciding
*what* it emits and *which* node it feeds is the part that goes wrong, and it goes
wrong quietly: the model still computes, the chart still draws, and the number is
built on a claim nobody meant to make.

These are the rules. Apply them in order — the first that fits is the one to use.


## 1. Prefer an absolute change, in the output node's own units

An action that says "300 fewer MWh a year by 2035" is added to its target node. Use
this whenever the effect can honestly be stated as a quantity.

Two things make it the default:

* **Missing categories are free.** Addition treats an absent row as zero, so an
  action that only speaks about one sector or one carrier can say so and stay
  silent about the rest. Nothing has to be filled in.
* **It is what the target node measures.** A quantity added to a quantity needs no
  interpretation, and the impact the UI reports is the number the action states.

The target node is an `AdditiveNode2` (or anything else that sums), and the action
carries **every** dimension the target has — an additive input with fewer is refused
outright:

```
Dimensions do not match with <action>
  (['energy_carrier', 'ghg_category'] vs
   ['energy_carrier', 'ghg_category', 'municipal_facility_type'])
```

That is the mechanical reason. The better one is that the missing dimension is usually
part of the measure: *which* buildings convert, *which* sites the meals are served at.
Naming them in the data keeps the allocation visible and lets a city that knows the
answer change it, instead of the model spreading the effect on a guess.

Note the asymmetry with the *categories* of those dimensions, which is what rule 1's
freedom actually buys: every dimension must be present, but within a dimension only the
categories the measure touches need rows.


## 2. Where absolute is impractical, use a relative factor

Some effects are only known as a proportion — "renovation cuts heat demand by a
third", "70 % of district heat is renewable by 2035". Writing those out absolutely
means multiplying by a base year and freezing that base into the measure, so a later
restatement of the underlying data silently invalidates it. In that case the action
emits a **factor**: the share of the target that remains, `1.0` meaning unchanged.

Three rules come with it:

* **A dimension the factor treats uniformly is left out of the action.** A
  dimensionless factor broadcasts across every row of the target. If the factor
  arrives carrying a dimension it does not actually differentiate on, flatten that
  dimension on the edge and it broadcasts the same way.
* **A dimension the factor treats differently must appear in the action's
  dataset**, carrying the **no-effect value** on every category the measure does not
  touch. This is not tidiness. `multiply_with_dims` joins **inner** and has no
  fill-with-one option, so a category absent from the factor is absent from the
  product — the untouched rows are *deleted*, not passed through.
* **`to_dimensions` is not used for this.** It accepts a single category per edge
  (`_edge_to_transforms` in `nodes/instance_parser.py` raises otherwise) and a node
  id cannot repeat within one `input_nodes` list, so building the missing categories
  edge by edge costs one node each. The dataset is where they belong.

Derive the untouched categories from the target node's full category set rather than
listing them by hand. A forgotten category deletes real activity and nothing says so.

The target node is a `generic.GenericNode` — see rule 5.


## 3. Keep the data at the granularity the source has; split in the model

Rule 1 says an absolute change carries every dimension of its target. That is a
statement about the *shape the node needs*, not about what you should claim to
know. When the source is coarser than the target — and it usually is — do not
fill the gap in the dataset. Write the rows at the granularity the source
actually has, and let the model distribute them.

`split_dims` is the operation. Tag the input `splittee` and the node spreads it
across the dimensions it lacks, in proportion to what is already there:

```yaml
type: generic.GenericNode
params:
  operations: add_datasets,split_dims,add,multiply
```

```
input:  heat / natural_gas  −4 185 MWh        (what the source says)
base:   heat / natural_gas × 7 property groups (what the node holds)
result: −4 185 MWh spread over the seven groups by their current gas use
```

### Why this is not merely convenient

A dataset is a record of what is known. Precomputing the split and storing it
puts an assumption into the same table as the observations, in the same columns,
with nothing to tell them apart — and a reader, or a later modeller, has no way
to see that seven of those numbers were invented by a division. Distributing in
the model keeps the two separable: the dataset says what the source said, the
operation says what we assumed, and the assumption is visible in the graph
rather than buried in a producer script.

It also stays correct for longer. A split written into the data is a frozen copy
of a distribution that was true on the day it was computed; `split_dims` reads
the distribution as it is when the model runs, so a restatement of the
underlying data carries through instead of quietly disagreeing with it.

And it is reversible in the right direction. A city that *does* know which of its
buildings are being connected can say so by adding the dimension back to those
rows — at which point there is nothing left to split and the operation stands
down. Going the other way, recovering "we did not actually know this" from a
table of numbers, is not possible.

### What it does not license

Splitting is still an assumption, and proportional-to-current is only the most
defensible default. Say which assumption is in play — in the node's description,
where a reader of the model will meet it. If the real allocation is known to be
different (new connections following the existing network rather than each
building's own boiler, say), the honest split is not the proportional one and
the model should say what it is instead.

Two mechanics worth knowing:

* `split_dims` is a no-op when nothing is tagged `splitter` or `splittee`, so a
  shared module can list it on a node that only some cities attach a measure to.
* It both distributes *and* adds, so it replaces `add` for that input rather than
  running before it.

## 4. A target value, rather than a deviation, is `DatasetReduceAction`

Where the estimate is "this reaches X by 2035" rather than "this is Y less than
business as usual", use `DatasetReduceAction`. It takes the target values *and* the
historical values and emits the absolute change between them, which is then added to
the output node.

The historical values normally come from the same dataset and the same rows the
output node reads. Bind a **copy** rather than the node itself: the action feeds the
node, so reading the node would close a cycle.


## 5. Choose the node class for the actions it will attract, not for today

A node a relative action will probably hit should be a `generic.GenericNode` **from
the start**, so that no city has to change a node class in a module it merely
includes. A node that will only ever be added to should be an `AdditiveNode2`.

This matters most in shared modules, where changing a class later means changing it
under every city at once.

### Why `GenericNode` and not `MultiplicativeNode2`

`MultiplicativeNode2` is the obvious choice and the wrong one. It requires **two**
factors:

```
Multiplication needs at least two inputs, got none
```

A node whose only input is its own dataset has none — a unit-compatible dataset is
classified as an addend, not a factor. So a measure-bearing node in a module that
some cities include *without* measures would need the module to ship a neutral `1.0`
factor purely to keep it computing.

`GenericNode` needs no such prop. Its operations are explicit and each is a no-op
when its bucket is empty:

```yaml
type: generic.GenericNode
params:
  operations: add_datasets,add,multiply
```

* `add_datasets` (**not** `get_single_dataset`, even where a node reads exactly one —
  see the extend caveat below) puts the node's own data on the table.
* `add` applies every input **node** tagged `additive`, or untagged with a compatible
  unit — the absolute changes of rule 1.
* `multiply` applies the rest: every input tagged `non_additive`, or untagged with a
  unit that cannot be added. One factor is fine; none is fine.

With no measure bound, the last two find nothing and the output is the dataset.

**Add before multiply.** An absolute change describes the activity, and a factor scales
the activity as it then stands. A building connected to district heating should still
benefit from the renovation measure, so the carrier switch is added first and the
renovation factor applies to the result. Multiplying first would leave the added
quantity untouched by every factor on the node.

**Datasets do not go through the add/multiply buckets** — those are resolved from
input nodes (`resolve_input_nodes` in `nodes/operands.py`) — so a dataset binding
needs no tag, and tagging one is misleading.

### Operation order is a correctness question, not a style one

`add_datasets,multiply,add` multiplies the *whole* base and then adds. Putting the
node's second dataset in the additive bucket instead — which is what
`MultiplicativeNode2` does with any unit-compatible dataset — adds it **after** the
multiply, so that part of the branch escapes the measure entirely.

That is not hypothetical. In `modules/gpc/municipal_balance.yaml`,
`municipal_building_energy` reads two datasets: the building management records and
the rented properties the records do not cover. Under `MultiplicativeNode2` the
rented properties were an addend and no renovation measure touched them, understating
the effect by about 225 t CO2e at 2035 — a quiet 5 % error in the answer, from an
operation order.

### One caveat when converting an existing node

`AdditiveNode2` honours the binding's `extend` flag; `GenericNode` does not, and a
series that stopped will start being held forward. `extend: false` on the binding
does not suppress it, and `get_single_dataset` makes it worse — it calls
`_extend_values` itself, so prefer `add_datasets` even for a single dataset. Snapshot the branch before converting and diff it after —
this is exactly the kind of change that does not announce itself.

The disagreement is a known defect awaiting a fix; see `data/TODO.md` item 50.

## 6. Target the node that makes causal sense

An action attaches where the effect physically happens. A measure that reduces heat
demand attaches to the heat demand; a measure that decarbonises a supply attaches to
its emission factor.

The temptation to break this is mechanical convenience — an emissions node is
usually already multiplicative, so a factor can be hung there with no other change,
and the total comes out the same. It is still wrong. The energy series then does not
move when the measure that reduces energy is switched on, the explanation the model
gives is false, and the next person to read the graph cannot see what the measure
does.

If the causally correct node cannot take the action, change that node (rule 5)
rather than moving the action.


## Where the numbers live

Not in the config. `historical_values` and `forecast_values` in YAML are
**deprecated**: they are fine as a first draft while a model is being built, and they
must move into a dataset before it is finished.

The reason is not tidiness either. A number in a config file is invisible to the
people who own it — a city user can open a dataset in the admin, see what an
assumption is and change it, and can do none of that with a literal in YAML. The
assumptions behind a measure are exactly the numbers they should own.

Publishing a dataset goes through `tools/upload_new_dataset.py`. Write the producer
script alongside the model work rather than treating inline values as finished
because the upload is someone else's step.


## A worked example

`configs/mainz-bisko.yaml` carries fifteen measures of a city-administration climate
concept, all of them rule 2 (the source states percentages of a base). The shape:

* Five datasets, one per factor **shape** rather than one per target node — two nodes
  that differentiate on the same dimension share a dataset, and a `measure` dimension
  keeps their rows apart. Each action filters that dimension to its own category and
  flattens it away.
* Each dataset carries every category its target node has, with `1.0` on the ones the
  measure does not touch.
* Both ambition variants of the source concept live in the data under a
  `measure_scenario` dimension. `select_variant` picks one and `selected_number` is
  the dial, so a scenario overrides one parameter per action and no action is
  duplicated.
* `modules/gpc/municipal_balance.yaml` declares its eight measure-bearing nodes
  `generic.GenericNode` with `operations: add_datasets,multiply,add`.

`data/mainz/create_knsv_measures.py` is the producer, and the derivation of every
value that is not quoted straight from the source is in the row's own `Comment`.


## Checklist

- [ ] Can the effect be stated as a quantity? Use rule 1.
- [ ] If it is a factor: does every category of the target appear, with the no-effect
      value where the measure is silent?
- [ ] Is any dimension in the dataset finer than the source actually is? If so, take it
      out and let `split_dims` do it.
- [ ] Is the target the node where the effect physically happens?
- [ ] Does the target's class match what will attach to it, in every city that
      includes it — and does it still compute where no action is attached?
- [ ] Does the operation order multiply the whole base, or does part of it slip in
      after the multiply?
- [ ] Is the branch unchanged when the action is off? Snapshot and diff.
- [ ] Are the numbers in a dataset rather than in the config?
