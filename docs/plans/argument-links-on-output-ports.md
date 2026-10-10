# Argument nodes attach to output ports, not input ports

*Produced by Claude Opus 5.5 on 2026-10-10.*
*Responsible: Jouni Tuomisto.*

Status: proposal, not agreed. To be built after action hooks reach `main` from
`feat/data-studio-backend` (expected around 2026-10-17). Needs Juha's view on the
open questions before any code is written.

## Summary

An argument node (an objection, a claim, a value statement) documents a dispute about
a number. Today it is wired as an **input** of the node whose number it disputes, and
every arithmetic path filters it out again. This plan moves the link to the target's
**output port**, owned by the argument node. That follows the pattern hooks set: the
thing attached owns the attachment, and the target's calculation never sees it.

It is not a hook. A hook adds numbers; an argument link adds none, and must not touch
cache keys, impact paths or cycle detection.

Reference edges (`reference` tag / `REFERENCE_ROLE`) are **out of scope** and stay on
the input side. See *What stays*.

## How it works today

Since `0e38af08` / `93efc5b6` (merged 2026-10-09):

- An argument node is any node with `quantity: argument`, usually a
  `generic.GenericNode` reading a zeros dataset or a `generic.ConstantNode` with
  `constant: 0`. Its `output_nodes` make it an input of the target.
- The edge gets **no special port role**. On a class migrated to ports, it is bound to
  whatever role the target infers for that port (usually additive). It is then dropped
  at arithmetic time by `Node.iter_computational_input_bindings`, because
  `operands.is_non_computational` checks the source's quantity.
- On legacy classes it is dropped by `operands.resolve_input_nodes` (`GenericNode`
  family) and `Node._drop_non_computational` (`_add_nodes_impl`, `multiply_nodes_pl`,
  `impute_nodes_pl`).
- Explanations skip it in a separate check (`explanations.py`, the terms loop).
- `REFERENCE_ROLE` is assigned only to edges tagged `reference`
  (`instance_graph.py`, port role classification; `instance_loader._setup_runtime_inputs`).
  Argument edges do not go there.

So the exclusion happens in five places, and any new arithmetic path has to remember to
join them. The objections module also edits the input list of nodes it does not own:
`objections.yaml` attaches to `net_emissions`.

Inventory on 2026-10-10: 25 argument nodes in four files.

| File | Nodes | Pattern |
| --- | --- | --- |
| `configs/modules/transportation/congestion_charge.yaml` | 12 | `GenericNode` + `transportation/zeros` |
| `configs/forestry-fi.yaml` | 8 | `GenericNode` + `Argument placeholder` rows of a dataset |
| `configs/modules/moral_argumentation/objections.yaml` | 4 | `ConstantNode`, `constant: 0` |
| `configs/finland-syke.yaml` | 1 | as `forestry-fi` |

Some targets are other argument nodes (`arg_26_revenue_recycling`), i.e. a rebuttal
chain. One argument node has no target at all.

## Design

### The link belongs to the argument node

```yaml
- id: obj_net_emissions_not_ours
  type: argument.ArgumentNode
  name_en: Those are not our emissions
  description_en: <p>…Toulmin register…</p>
  bears_on:
  - node: net_emissions          # or node.port when the target has several outputs
  - node: obj_g1_drop_in_ocean   # another argument: a rebuttal
```

In the spec, as a sibling of `ActionHookDef`:

```python
class ArgumentLinkDef(BaseModel):
    node: NodeRef
    port: UUID | None = None   # the target's output port; omitted for a single output
```

The target addressing (`node`, `port`) is the same as a hook's, so the editor and the
GraphQL resolver can share it. Nothing else is shared: no `from_port`, no
`transformations`, no contribution.

### Argument nodes stop computing

A dedicated node class, `argument.ArgumentNode` (a new node kind, or a class under the
existing simple kind; open question 2), with no inputs, no output metrics and no
`compute()`. That removes the zeros datasets, the `Argument placeholder` dataset rows and
`ConstantNode(0)`. All of these exist only so that a node with nothing to compute can sit
in an input list.

`quantity: argument` stays, as the display marker it already is for the UI.

Objection parameters stay `simple.ValueAction`s feeding evaluation nodes, as in
`argumentation.md` §6. They are computational and are not argument nodes.

### Where the link is visible, and where it is not

| Consumer | Sees argument links? |
| --- | --- |
| Arithmetic, port bindings, shape rules | No: links are not inputs |
| Node cache hash | No: editing an objection's text must not recompute `net_emissions` |
| `context.node_graph` (cycles, impact paths, downstream) | No |
| Explanations | Yes, as a separate "Arguments about this number" list, not a filtered term |
| GraphQL / model editor | Yes, next to hooks, with the same drag-onto-output-port gesture and a distinct edge style |
| Public UI graph | Yes, as today |

Cycle checking for rebuttal chains is a separate, trivial check on the argument graph.

### What goes away

- `operands.is_non_computational`, `drop_non_computational`'s argument branch,
  `Node.iter_computational_input_bindings` and the argument check in
  `Node._drop_non_computational`.
- The argument check in the explanations terms loop.
- `transportation/zeros` as an argument input. Check that nothing else reads it before
  deleting it from DVC.

Order: the removals land in the same change as the migration, not before it.

### What stays

**Reference edges.** A reference edge says "this input should feed this calculation; how
is not known yet". That is a statement about the target's calculation, and its next step
is to become an additive or factor binding on the same port. On migrated classes the role
is already excluded structurally, because no operation resolves it. The legacy tag scan
(`operands.is_reference_edge`) disappears as classes migrate. Moving references to the
output side would turn that promotion into a rewire.

## Migration

1. Hooks merged to `main`, with `ActionHookDef` and its parser in place.
2. Add `ArgumentLinkDef`, `ArgumentNode`, the `bears_on` parser, the explanation list and
   the cycle check. Tests: an argument linked to a node with dimensions and to a framework
   node; a rebuttal chain; a cycle refused; the target's cache hash unchanged when an
   argument is added or edited.
3. Convert the 25 nodes in one change. The loader refuses `quantity: argument` with
   `output_nodes`, and the error message names `bears_on`. A compatibility path is not
   worth having for four files (principle 5 would put it behind `from_yaml_config` if it
   were).
4. Verify per affected instance (`equalia`, `mainz-dev`, the congestion-charge and forestry
   instances): every computational node's output is identical before and after. The
   floating-point churn noted in `argumentation.md` §6 is already behind us, so expect
   exact equality.
5. DB-sourced instances among them: `sync_instance_to_db`, then publish (see
   `docs/trailhead/tools.md`).
6. GraphQL and editor support, built together with the hooks editor work so that the
   output-port drop target is designed once.
7. Update `argumentation.md` §6 and the conventions header of `objections.yaml`.

## Open questions

1. **Naming.** `bears_on` / `ArgumentLinkDef` are placeholders. "Annotation" is too generic
   (node descriptions are annotations too); "argument link" says what it is.
2. **Node kind or class.** A new `NodeKind.ARGUMENT` makes "does not compute" a
   construction-time fact (principle 3). A class under an existing kind is less work, but
   it leaves `compute()` reachable.
3. **Shared GraphQL shape with hooks.** One interface ("attached to an output port") with
   two implementations, or two unrelated types that share only the target fields. I lean
   towards two types, because the hook's contribution semantics should not leak into a
   type that has none.
4. **Should an argument be able to bear on an input port or an edge?** Some objections
   dispute an assumption, such as an emission factor's source, rather than a result. Today
   that is expressed by targeting the node that holds the assumption, which seems
   sufficient. Leave it out of v1.
