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
carries the target's dimensions.


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

The target node is a `MultiplicativeNode2` — see rule 4 — or a `GenericNode` when
the branch needs operations beyond multiplication.


## 3. A target value, rather than a deviation, is `DatasetReduceAction`

Where the estimate is "this reaches X by 2035" rather than "this is Y less than
business as usual", use `DatasetReduceAction`. It takes the target values *and* the
historical values and emits the absolute change between them, which is then added to
the output node.

The historical values normally come from the same dataset and the same rows the
output node reads. Bind a **copy** rather than the node itself: the action feeds the
node, so reading the node would close a cycle.


## 4. Choose the node class for the actions it will attract, not for today

A node a relative action will probably hit should be a `MultiplicativeNode2` **from
the start**, so that no city has to change a node class in a module it merely
includes. A node that will only ever be added to should be an `AdditiveNode2`.

This matters most in shared modules, where changing a class later means changing it
under every city at once.

### The two-input rule, and the neutral factor

`MultiplicativeNode2` raises

```
Multiplication needs at least two inputs, got none
```

when its only input is a unit-compatible dataset — such a dataset is classified as
an *addend*, not a factor, so the node has nothing to multiply. A measure-bearing
node in a module that some cities include **without** measures therefore needs a
second factor to exist at all.

The convention is for the module to ship a constant `1.0` and wire it into each such
node. It is inelegant, and it is the least bad of the four options:

| Option | Why not |
|---|---|
| Let a multiplicative node accept one factor | Loses the point of the class — nothing distinguishes it from an additive node |
| Express the relative change as an absolute one | Dresses an intensive quantity up as extensive data (rule 1 exists because absolute is *honest* when it is available, not because it is always available) |
| Put the factor on a downstream node that is already multiplicative | Breaks rule 5 |
| A scoped passthrough per part of a branch | One node per part, rather than one in total |

A neutral factor changes no value — the product of a quantity and `1.0` is the
quantity — but say so with a test rather than by assertion. Snapshot the branch
before the change and diff it after.

Two mechanics to know when converting an existing node:

* Tag the dataset that carries the quantity `non_additive`, or it stays an addend
  and the product is the neutral factor alone. A node summing **two** datasets keeps
  one as the factor and leaves the other additive.
* Converting to `generic.GenericNode` instead needs
  `operations: add_datasets,add,multiply`. Plain `add,multiply` returns nothing when
  the inputs are datasets, because `add` collects only nodes.


## 5. Target the node that makes causal sense

An action attaches where the effect physically happens. A measure that reduces heat
demand attaches to the heat demand; a measure that decarbonises a supply attaches to
its emission factor.

The temptation to break this is mechanical convenience — an emissions node is
usually already multiplicative, so a factor can be hung there with no other change,
and the total comes out the same. It is still wrong. The energy series then does not
move when the measure that reduces energy is switched on, the explanation the model
gives is false, and the next person to read the graph cannot see what the measure
does.

If the causally correct node cannot take the action, change that node (rule 4)
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
* `modules/gpc/municipal_balance.yaml` supplies the neutral factor and declares the
  eight measure-bearing nodes multiplicative.

`data/mainz/create_knsv_measures.py` is the producer, and the derivation of every
value that is not quoted straight from the source is in the row's own `Comment`.


## Checklist

- [ ] Can the effect be stated as a quantity? Use rule 1.
- [ ] If it is a factor: does every category of the target appear, with the no-effect
      value where the measure is silent?
- [ ] Is the target the node where the effect physically happens?
- [ ] Does the target's class match what will attach to it, in every city that
      includes it?
- [ ] Is the branch unchanged when the action is off? Snapshot and diff.
- [ ] Are the numbers in a dataset rather than in the config?
