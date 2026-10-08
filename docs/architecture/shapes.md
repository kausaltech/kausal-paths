# Shapes

A **shape** is a named, declared set of category combinations, together with the combinations
among them that are required. It states a fact about data that flows through the model: "end
energy in BISKO is broken down into these sectors and carriers, and electricity must be reported
for the whole municipality". Datasets and input ports *refer* to a shape instead of each carrying
their own copy of the list.

Shapes replaced three mechanisms that each held part of this fact, attached to the wrong thing:

- the `category_domain` declared on a dataset, which describes the rows a dataset may hold but
  says nothing once a framework instance replaces that dataset with a calculation;
- the dataset rules `required_combinations` and `allowed_combinations`;
- the `combinations` written by hand into a node's `input_validation`.

Shapes are **declared**. They are not to be confused with the **observed** shape of a dataset
(`DatasetShapeProfile` in `datasets/shapes.py`): the categories that the data of one dataset
version actually holds, frozen beside each revision pin. A declared shape says what is meant; an
observed profile says what is there. The static check below compares the two.


## What a shape is

```yaml
shapes:
- id: bisko/end_energy
  name_de: Endenergieverbrauch nach BISKO
  name_en: End energy use, BISKO
  dimensions: [sector, energy_carrier]
  combinations:
  - {id: private_households_electricity, categories: {sector: private_households, energy_carrier: electricity}}
  - {id: industry_electricity, categories: {sector: industry, energy_carrier: electricity}}
  # ...
  required:
  - id: electricity                   # satisfied by any sector that reports it
    combinations: [private_households_electricity, industry_electricity]
    qualifiers: {bisko_quality.coverage: {min: 1}, bisko_quality.score: {min: 1}}
```

- **`dimensions`** are the dimensions the shape constrains. Every combination names a category in
  each of them, and nothing else.
- **`combinations`** are the category tuples the shape defines. Each has its own identity, because
  required groups and data-entry rows refer to it.
- **`required`** lists groups. A group is satisfied in a year when at least one of its
  combinations has a value, and all of the group's values meet its qualifier requirements when
  stated. Groups express "heat from oil *or* gas", or BISKO's "electricity for the whole
  municipality, in whatever sectors it is reported", which a flat list of required cells cannot.
- **`closed`** (default `false`) says whether the shape's combinations are the *only* ones allowed.
  An open shape is a minimum: other combinations are accepted, just not as part of the shape.
- **`inherits`** lists shapes this one extends; see below.

Shapes are declarations of the instance spec, like parameters and scenarios. They have a UUID;
the identifier is optional, a bridge for YAML authoring and for readable references. It follows
the dataset identifier pattern, so `bisko/end_energy` is valid, and a prefix like `bisko/` is a
naming convention for the reader. No logic depends on the form of an identifier.


## Inheritance

A shape's **effective** combinations and required groups are the union of its own and those of
every shape it inherits, transitively. A child can only add: it cannot remove a combination or a
requirement of its parent. That is the guarantee that anything conforming to a child also
covers the parent, so a municipality's shape still covers BISKO's minimum. `closed` is not
inherited; each shape decides it for itself. All shapes in an inheritance chain constrain the
same dimensions. Cycles are rejected.

Inheritance is resolved when the runtime shape is built, never stored hydrated. A shape records
the identities it inherits from, and the effective shape is computed from whatever those
resolve to in the composed instance. Moving a municipality's template pin therefore brings in
the template's new combinations without touching the municipality's own record.

Combinations keep their origin. The effective shape reports, for each combination, which shape
in the chain declared it, so a client can tell a BISKO row from a local extension. A combination
is identified by its category tuple when shapes are unioned: if a parent later declares a tuple
a child already added, the parent's entry wins and the child's becomes redundant, which is
reported but harmless.


## Ownership

A framework template declares two kinds of shape.

```yaml
# The standard: what BISKO defines and requires.
- id: bisko/end_energy
  dimensions: [sector, energy_carrier]
  combinations: [...]
  required: [...]

# The extension point the model refers to: the standard, closed, extended per municipality.
- id: end_energy
  inherits: [bisko/end_energy]
  closed: true
  owner: instance
```

- **A shape with the default owner (`framework`) is read-only** in dependent instances, like
  inherited nodes (see [template inheritance](template-inheritance.md)). Local declarations
  cannot shadow template identities, as for every other declaration of the instance spec.
- **A shape with `owner: instance` is the municipality's.** When an instance starts following the
  template (on conversion and on upgrade, if it is missing), it gets its own record of the
  shape, carrying the template's `inherits` and `closed` as written. That record cannot be
  removed. For now its `dimensions`, `inherits` and `closed` cannot be changed either; what the
  municipality edits are its own combinations.
- **The template's nodes and datasets refer to the extension point**, not to the standard:
  `final_energy_use` and `kommune/endenergieverbrauch` both refer to `end_energy`. Each
  municipality's references then resolve to its own record, so its additions are accepted
  everywhere at once and nothing is re-declared per node.

Because the municipality's record holds the `inherits` and not a copy of the combinations, the
standard's content must live in the standard shape. A combination written directly into the
template's `end_energy` would reach no municipality, since each one has its own record of that
shape. The template's extension points carry only `inherits` and `closed`.

Editable `inherits` is the intended next step. Shared computation modules (street lighting, or
renewable generation as a common BISKO extension) would be plugged in by adding their shapes to
an extension point's `inherits`, so a municipality could take on a module's combinations
without the template changing. Removing an inherited framework shape must stay impossible then,
which is a rule about the edit and not about the stored data.


## References and enforcement

A shape says what the combinations are. A **reference** says what is checked against them, and
how strictly. The same shape is a hard constraint on a data-entry table and a certification
requirement on the node that consumes it, so enforcement belongs to the reference, not the shape.

### On a dataset

```yaml
datasets:
- id: kommune/endenergieverbrauch
  shape: end_energy
```

A dataset reference uses the shape's **combinations** only:

- they define the **entry form**: the cells an empty year is given, and the rows the data-entry
  UI renders (`categoryDomain` in GraphQL, which keeps its form);
- if the shape is closed, a value outside it is refused when entered (`block_edit`).

The reference is stored on the dataset row (`Dataset.spec['shape']`, beside `default_quality`)
and **resolved on read**, in the dataset's own instance. The effective combinations are never
stored. They cannot be: municipalities share their `kommune/*` schemas with the template, and
each municipality's record of an extension point may add different combinations, so a domain
compiled onto the shared schema could hold only one of them. Resolving on read also means
nothing derived goes stale when a municipality edits its record or upgrades its template.
`datasets.shape_domain.dataset_category_domain()` is the one accessor every reader uses.

This replaces `DatasetSchema.category_domain`. Only BISKO ever declared a domain, and it was used
for exactly this, the entry grid; observed combinations of external datasets live in
`DatasetShapeProfile` and never used it. The column is still read for a dataset without a shape
reference, until production instances have synced theirs; then it goes from `kausal_common`.

A dataset reference does **not** enforce required groups. Whether a required cell is missing is
a question about what the model receives, not about what one table holds: a municipality that
relies on the national defaults legitimately enters nothing, and a framework instance may feed
the same port from a calculation instead of the table. Requirements are checked at the port.

### On an input port

```yaml
- id: final_energy_use
  input_validation:
    data:
      shape: end_energy
      enforcement: block_submission
```

A port reference checks the values the port **receives**, whatever delivers them: a dataset, a
local calculation, or a submodel that replaced the template's binding. It enforces the shape's
required groups and, when the shape is closed, its combinations. It is evaluated with the other
value contracts (`nodes/value_validation.py`), so it surfaces in instance problems and in
`valid_inputs()`, and gates publication or submission according to its enforcement.

Conditions that are about a port's siblings stay on the port, beside the reference:
`required_if_positive`, `combinations_from_positive`, value ranges and `max_rows` are not facts
about a shape. Neither is a value in each year whatever its categories, which is
`required: true` (with `qualifiers:` when the value must also meet them):

```yaml
  input_validation:
    data:
      years: active
      required: true
      min: 1
```

A group with several combinations is satisfied in a year when one of them has values and all
of the group's values meet its qualifiers.

### Where to declare a reference

Where the fact originates, not wherever the data flows. In the BISKO module end energy enters
the graph at three nodes, and the requirement lives at one of them (`final_energy_use`). Nodes
downstream receive node outputs and need no reference. A reference on the entry dataset defines
what can be typed in; a reference on the consuming port defines what the model needs.


## The static check

Whenever the constraint program is compiled (at publication, for a binding edit, in the editor
and in `test_instance`), every binding from a dataset with a shape reference to a port with one
is checked without reading data:

1. Take the dataset's effective combinations.
2. Push them forward through the binding's transformations. Filters restrict the set, assigning
   a dimension adds a fixed coordinate, and a flattening filter drops the dimension and merges
   the combinations that become identical.
3. If the port's shape is closed, every projected combination must belong to it.
4. Every required group of the port's shape must contain at least one projected combination, or
   the data-entry route cannot satisfy it.

The forward direction is always computable for the operations above, which is why the check
projects datasets onto ports and never the other way round: running a filter or an aggregation
backwards is not well defined. A binding whose transformations include an operation that cannot
be projected is reported as unchecked, never passed silently.

The same comparison against a pin's observed `DatasetShapeProfile` answers a different question:
does the data present satisfy the declaration? That is what the runtime port check already does
on delivered values.

When the port's required groups are checked, the projections of every binding into the port are
united first, so two entry tables may share a requirement. A port that also receives a node's
output or an unshaped dataset is not checked for requirements at all: the other route may well
deliver them, and only the runtime check on delivered values can tell.

Findings are constraint conflicts (`outside_shape`, `shape_requirement_unreachable`,
`shape_dimension_missing`, `unknown_shape`), computed while the constraint program is compiled
(`nodes/constraints/shape_check.py`). So they block publication, reject a binding edit that
introduces one, and appear wherever conflicts do: in the editor and in `test_instance`. A
modeller's half-finished draft still syncs. An unprojectable binding is a `shape_unchecked`
notice on the solve result; it blocks nothing, and `test_instance` records it as a warning.

A graph parsed from YAML has no dimension catalogue, so the check, like the other dimension
checks, applies once the instance is synced to the database.


## GraphQL

- An instance exposes its **shapes**, local and inherited, each with `id` (UUID), `identifier`,
  name, `owner`, `closed`, `inherits`, `isEditable`, its own combinations and required groups, and its
  **effective** combinations with each combination's origin shape. A combination exposes its
  categories as dimension and category identities.
- A dataset exposes the shape of its entry form (`shape`), and an input port the shape its
  contract refers to (`shape`) and what a failure of the contract blocks (`contractEnforcement`).
  Both resolve to the same `Shape` as the editor's list, from the request's instance graph.
- The editor's `constraintNotices` lists what the structural checks could not verify, such as an
  unprojectable binding; `constraintConflicts` and `problems` carry the static check's findings.

All of it is on the instance editor (`instance { editor { shapes … } }`), where both the model
editor and the data-entry UI read.

This answers the data-entry UI's question "which combinations are valid for BISKO": the rows of an
entry table are its dataset's effective shape, and each row's origin says whether BISKO defines it
or the municipality added it.


## Conformance at the output surface, never inside the graph

There is no port transformation that keeps only a shape's combinations. Inside the graph a filter
changes what is computed, and the data it removes leaves every downstream total without a trace:
the failure `dbcc899e` fixed for the local factor tables. Mainz's hydrogen would quietly drop out
of the BISKO total.

Conformance is instead a property of a **view**: a node's output queried through `metric_dim`
with a shape, or a BISKO-conformant report. The computation keeps every combination. The view
selects the shape's, and reports what it left out (at least as a remainder total), so a reader
can always see that the conformant figure is not the whole one.


## In the BISKO module

`configs/modules/bisko/model.yaml` declares the standard shapes (`bisko/*`) from the
Methodenpapier and the certification's Prüfprotokoll, and the YAML comments cite them. BISKO has
no matrix of mandatory sector × carrier cells: it requires a value per carrier for the whole
municipality, so end energy is required as one group per carrier that any sector satisfies, with
the grade-A rule of the grid-bound carriers holding for every value of the group. Transport is
required per transport means rather than per carrier. District heating plants and generation
variants are shapes without requirements, because each is one of three permitted routes to the
district heating factor.

Each open standard shape has a closed extension point (`end_energy`, `road_mileage`,
`road_transport_energy`, `other_transport_energy`, `district_heating_plants`), and the
`kommune/*` datasets and the ports that consume them refer to those.


## Open

- **Requirements added by a municipality.** Whether an extension point's owner may add required
  groups as well as combinations. Adding a requirement cannot weaken the standard's, and a Land's
  reporting rules may demand more than BISKO. But it lets a municipality's own declaration block
  its submission. Until decided, a municipality adds combinations only.
- **Converting the existing BISKO instances.** Their records of the closed extension points must
  take the carriers and sectors they use beyond the standard, such as Düsseldorf's sector totals.
- **Editable `inherits`**, for shared computation modules; see *Ownership*.
