# Fault Tolerance in the Model Graph

## Overview

The model editor (Trailhead) lets people build a model instance
incrementally: dragging in nodes, wiring edges, adjusting parameters.
During this work the graph is routinely in a broken state — a node that
isn't connected yet, a node whose `compute()` raises because a required
input is missing, a cost subgraph that fails while the emissions
subgraph is fine.

The default engine is fail-fast: any exception during instance
construction or node computation aborts the whole instance (see
`InstanceLoader` and `Node._get_output_pl`). That is correct for
production, but useless for an editor — one broken node would blank the
entire model the user is trying to fix.

This document describes an opt-in **tolerant mode** for draft models: a
node that fails to construct or compute is quarantined, the healthy
remainder of the graph still computes, and every failure is surfaced to
the editor with enough attribution to point at the root cause.

The guiding constraint: **a tolerant graph must never present a
wrong-but-plausible number as authoritative.** A red node is honest; a
silently-undercounted emissions figure is a correctness bug. Tolerance
is therefore always visible (status is reported) and always unpublishable
(see Scope).


## Scope and guarantees

- Tolerant behavior is **opt-in**, gated on a flag on `Context`. The
  default computation path stays fail-fast, unchanged.
- It applies to **draft models only**. A model snapshot that has any
  node in a non-OK status **cannot be published**. This is what makes a
  partial result safe: it is never the basis of a published scenario.
- Node graphs are rebuilt per GraphQL request — `Context` and the `Node`
  objects are ephemeral. Status therefore lives for the duration of one
  request and needs no cross-request invalidation.


## Node status

Every node carries a status:

```python
class NodeStatus(Enum):
    OK          # computed (or will compute) cleanly
    INCOMPLETE  # not enough inputs wired yet — mid-construction, not an error
    DEGRADED    # produced output, but from a partial input set
    FAILED      # construction or compute failed; no usable output
```

`node.status is None` means "not determined yet" — the node has neither
been validated nor pulled.

| Status | Produces output? | Meaning |
|---|---|---|
| `None` | — | Not yet evaluated this request |
| `OK` | yes | Healthy |
| `INCOMPLETE` | empty (self-report only) | No / insufficient inputs wired |
| `DEGRADED` | yes (partial) | Tolerated a missing/failed input |
| `FAILED` | no | Own construction or compute failed |

Within a single request, status only ever moves *toward* failure
(`OK → DEGRADED → FAILED`), never back. Because each node computes at
most once per request (results are cached) and a fresh graph is built for
the next request, this monotonicity is essentially automatic; the setter
enforces it as cheap insurance against an in-request double-set. A node
fixed by the user "recovers" simply because the next request builds a new
graph and re-evaluates from `None`.


## When status is set

Status is determined at the cheapest point that can determine it, in
three tiers:

1. **Construction time (metadata only).** Anything provable without
   computing is stamped during instance load: missing required
   parameters, incompatible units, dangling edge targets, dimension
   requirements that no upstream can satisfy, and the unwired
   (`INCOMPLETE`) case. This is the most valuable tier — it costs nothing
   at compute time and catches the "just dropped a node in" case, the
   most common ephemeral failure. The dimension-constraint propagation
   described in [`dimension-constraints.md`](dimension-constraints.md) is
   the detector for the shape-related failures here.

   This tier lands together with the planned Spec-based graph
   construction that will replace the current YAML/`InstanceLoader` path
   (see the implementation plan). Until then, the unwired (`INCOMPLETE`)
   case is caught at compute time — a tolerant `AdditiveNode` with no
   available inputs marks itself `INCOMPLETE` — and richer metadata
   validation is deferred rather than retrofitted onto `InstanceLoader`.

2. **Compute time.** A node whose own `compute()` raises marks **its own**
   status `FAILED` in the `except` block of `get_output_pl`, distinguishing
   a node's own failure from an inherited cascade via
   `NodeError.event_chain[0].node` (the origin). The status is memoized: a
   subsequent `get_output_pl` for a node already `FAILED` (whether from
   tier 1 or tier 2) short-circuits and re-signals failure **without
   recomputing**. This matters for performance — a failed node consumed
   by N downstream nodes fails once, not N times.

   The memoization is **intentionally not scenario/parameter-scoped**:
   `status` is a single per-object field, while the output cache is
   scenario-keyed. So a node that fails under *any* scenario evaluated in
   a request stays `FAILED` for the rest of that request, even if it would
   succeed under a different scenario pulled later (e.g. baseline). This is
   deliberate — a model under construction that breaks in one scenario
   needs the developer's attention regardless, and scenario-scoping the
   status would add needless complexity for no editor benefit.

3. **Query time.** The editor reads each node's own `status` and `errors`
   directly (see Reporting). Root-vs-cascade is derivable client-side
   without a server-side walk: a *root* failure is a `FAILED` node with a
   non-empty `errors` list; a *cascade* is a `FAILED` node with empty
   `errors` (the real error lives on the upstream root). So the editor can
   avoid lighting up forty nodes red without the server computing an
   effective-status DAG walk.


## Propagation: ports and edges

Failure attribution lives at the **port**, not only the node. This
mirrors the dimension architecture, and gives a clean duality:

> Dimension **requirements** propagate *upstream* through ports and
> edges; input **availability** propagates *downstream* through the same
> ports and edges. Same topology, opposite directions.

- **Port availability** (derived, consumer-side): an input port is
  *unavailable* iff its upstream node is `FAILED` or `INCOMPLETE`. A
  `DEGRADED` upstream is still *available* — it produced partial output;
  the consumer ingests it and inherits `DEGRADED` via the query-time
  walk.
- **Edge status** (derived): the upstream node's status seen through the
  edge. Nothing is stored on the edge — it is derived for rendering only.

`InputPortDef.multi` (the multi-connection additive port) and
`OutputPortDef.dimensions` (the declared output shape) are the spec-level
anchors for this; see `nodes/defs/port_def.py`.


## Tolerant computation

The core rule for a tolerant consumer is **skip, don't sum**: an
unavailable input is dropped from the computation entirely, exactly as if
its edge did not exist — it is never ingested as an empty frame. This
avoids dimension-alignment errors and keeps the partial result
well-formed.

### AdditiveNode

`AdditiveNode` is the first (and, for now, only) node to opt into
tolerance. The choice is principled, not arbitrary: its documented
contract is already *"Missing values are assumed to be zero."* Treating a
failed or unavailable input as a zero contribution is an **extension of
behavior the node already promises**, not a new semantic. This is exactly
why tolerance must *not* be generalized to nodes where missing ≠ zero —
`SubtractiveNode`, division, or any node whose output changes
non-trivially when an input vanishes would silently produce a
*differently* wrong number with no contract to justify it.

When a tolerant `AdditiveNode` skips one or more inputs, it marks its own
status `DEGRADED`. The skipped inputs have already marked *themselves*
`FAILED`/`INCOMPLETE`, so the consumer does not record anyone else's
status — it only records its own.

### add_nodes_pl

The skip logic lives in the single existing input-fetch loop — one
`get_output_pl` call per input, no probe pass. To avoid changing the
return signature of `add_nodes_pl` (which has many callers), the body
moves into a private implementation that always returns a count of
skipped inputs, with two thin wrappers over it:

```python
def _add_nodes_impl(
    self, df, nodes, ..., tolerate_failures: bool,
) -> tuple[ppl.PathsDataFrame, int]:
    ...
    skipped = 0
    for node, mult in pairs:          # multipliers paired up front
        try:
            node_df = node.get_output_pl(self, metric=metric)
        except NodeError:
            if not tolerate_failures:
                raise
            skipped += 1              # node self-marked; contributes zero
            continue
        node_outputs.append((node, node_df, mult))
    ...
    return df, skipped

def add_nodes_pl(self, df, nodes, ...) -> ppl.PathsDataFrame:
    out, _ = self._add_nodes_impl(df, nodes, ..., tolerate_failures=False)
    return out

def add_nodes_tolerant(self, df, nodes, ...) -> tuple[ppl.PathsDataFrame, int]:
    return self._add_nodes_impl(df, nodes, ..., tolerate_failures=True)
```

Every current caller keeps `add_nodes_pl`'s `PathsDataFrame` return
unchanged and stays fail-fast (`tolerate_failures=False`).
`AdditiveNode` calls `add_nodes_tolerant` (only when the `Context` is in
tolerant/draft mode) and uses the skip count to set its own status.

Enabling the skip forces one cleanup: `node_multipliers` is currently
popped positionally in lockstep with the node list. Skipping a node
mid-loop would silently misalign multipliers, so the multipliers must be
**paired with their nodes up front** (`zip(nodes, multipliers)`) rather
than popped. This is a worthwhile fix regardless — the positional pop is
a latent bug.

If `df is None` and *every* input was skipped, the node falls through to
the `INCOMPLETE` empty-output path rather than popping an empty list.

### Empty output for INCOMPLETE nodes

An `INCOMPLETE` node's `get_output` returns an empty frame. Its shape is
the node's **declared** `OutputPortDef.dimensions` when present (empty
rows, correct shape), falling back to dimensionless (year + value only)
for a transparent node with no declared output dimensions — consistent
with the additive rule that output dims are the union of input dims, which
for zero inputs is empty.

This empty output is a **self-report only** — what the editor shows when
previewing that node directly. It is never an operand in a downstream
sum, because tolerant consumers skip unavailable ports. So the dimension
machinery never has to reconcile an empty/dimensionless frame against a
dimensioned sibling.


## Reporting

`NodeEditorFields` (`nodes/graphql/types/node.py`) gains two fields for
the editor:

- `status(compute: Boolean! = false): NodeStatus` — the node's status,
  `null` until evaluated.
- `errors(compute: Boolean! = false): [NodeError!]!` — the structured
  problems recorded *at this* node. Empty for an OK node, and empty for a
  node only affected via a cascade (its status reflects an upstream break,
  but the error itself lives on the upstream node).

Both fields take a `compute` argument. With `compute: false` (the
default) the resolver just reads what's already there — cheap, so the
editor can fetch statuses for the whole graph in one fast query.
Init-phase failures (metadata validation, once tier 1 lands) are known
without computing and show up immediately; compute-phase status stays
`null` until determined. With `compute: true`, the resolver runs
`get_output_pl()` for that node if its status is still unknown (failures
are swallowed — the node's status/errors are populated as a side effect).
The UI uses this for a node-detail view, or as a deferred follow-up query
("compute statuses for all nodes") after the main view has rendered, with
a spinner while results stream in. Computing a node also stamps statuses
on its whole upstream cone as a side effect, since pulling it pulls them.

The error type carries a few structured fields alongside the free-form
message, so the editor can filter/group without parsing strings:

```graphql
enum NodeErrorPhase {
  INITIALIZATION   # raised while constructing the node / wiring edges
  COMPUTATION      # raised from compute()
}

type NodeError {
  phase: NodeErrorPhase!
  message: String!
}
```

A node can hold several errors — initialization may surface multiple
metadata-validation failures at once, whereas a compute failure yields a
single entry. (Metadata validation reports under `INITIALIZATION`; a
separate phase can be split out later if the editor needs the
distinction.)

Cascade attribution — which upstream node is the root cause of a blocked
node — is recovered by walking ancestors to the nearest nodes that carry
their own `errors`. The existing `NodeError` event chain
(`add_node_event`) already records this path, so an `originNodeId` field
can be added to `NodeError` later to expose it directly if the walk
proves insufficient.

The `DEGRADED` status itself distinguishes *this node tolerated a failed
input* (set by the node from a non-zero skip count) from
*inherited-degraded-from-upstream* (a node that is OK on its own but
whose effective status is dragged down by the ancestor-walk) — the
former is the more actionable badge, pointing at this node's own wiring.


## Alternatives considered

**Drop the failed node's output edges, leave everything else.** The
original framing. Rejected as the primary mechanism: at compute time it
doesn't rescue downstream (a node that lost a required input fails anyway,
just differently), and for nodes that tolerate missing inputs it produces
a *successful-but-wrong* number with no signal. Edge removal also
destroys the declared topology the editor wants to draw. Replaced by node
status + skip-don't-sum + query-time propagation, which keeps the
topology and never fabricates a clean-looking wrong result.

**Make tolerance the default compute path.** Rejected. A graph that
silently amputates failing nodes and reports whatever's left is a
correctness hazard the moment it backs a city's published scenario.
Tolerance is gated and draft-only.

**Generalize tolerance to all nodes.** Rejected. Only nodes whose
contract is "missing input = zero contribution" (`AdditiveNode`) can drop
an input without changing semantics. Applying it to subtraction,
division, etc. would silently corrupt results.

**Key status to the node hash (cache-invalidation scope).** Considered
when we thought a parameter change might flip a node's status within a
live graph. Unnecessary: `Context`/`Node` are rebuilt per GraphQL request,
so status is naturally request-scoped and a fixed node recovers via the
next request's fresh graph. Object-lifetime status is sufficient.

**Pre-filter inputs by probing each one before summing.** Rejected: even
with aggressive caching it means two `get_output` calls per input. The
single try/except in the existing `add_nodes_pl` loop achieves the same
with one call.

**Store status on edges.** Rejected. Edge/port status is a pure function
of the upstream node's status; derive it for rendering rather than storing
and maintaining it (same discipline as removing the `side` field in the
dimension work).


---

## Implementation plan (temporary — remove once landed)

Status: steps 1–6 **landed** (engine + GraphQL reporting, with tests in
`nodes/tests/test_fault_tolerance.py` and `test_model_editor.py`). Steps 7
and the two wiring items below are **not yet done**.

1. **`NodeStatus` enum + `Node.status` field.** Define `NodeStatus`
   (`OK`, `INCOMPLETE`, `DEGRADED`, `FAILED`) and the `NodeErrorPhase`
   enum + `NodeStatusError` record in `nodes/node.py` (small enough not to
   warrant a separate module). Add `Node.status: NodeStatus | None = None`
   and `Node.status_errors: list[NodeStatusError]`, plus a `mark_status`
   helper that only moves toward failure (severity-ordered max), never
   back, within the object's life.

2. **Tolerant flag on `Context`.** A single boolean, default `False`,
   threaded from the draft-editor entry point. Everything below is dormant
   unless it is set.

3. **Compute-time status (tier 2).** In `Node._get_output_pl`: on the
   existing `except`, stamp `self.status = FAILED` before re-raising; at
   the top, short-circuit if `status is FAILED`/`INCOMPLETE` (memoization,
   no recompute). On success, stamp `OK`.

4. **`add_nodes_pl` split.** Move the body into
   `_add_nodes_impl(..., tolerate_failures) -> (df, skip_count)`; keep
   `add_nodes_pl -> PathsDataFrame` and add
   `add_nodes_tolerant -> (df, skip_count)` as wrappers. Pair multipliers
   with nodes up front; skip inputs that raise `NodeError` when tolerant.
   Verify `SubtractiveNode`/`SectorEmissions` and other callers are
   unchanged.

5. **`AdditiveNode` opt-in + empty output.** In tolerant mode, call
   `add_nodes_tolerant`; mark `DEGRADED` when the skip count is > 0; mark
   `INCOMPLETE` and emit the declared/dimensionless empty frame when it
   has no available inputs. This is what catches the unwired-node case
   until tier 1 exists.

6. **Reporting via GraphQL.** `NodeStatus` and `NodeErrorPhase` are
   registered as Strawberry enums (`@sb.enum` directly on the Python
   enums). A `NodeError` GraphQL type (`phase`, `message`) is exposed, and
   `NodeEditorFields` gained `status(compute)` and `errors(compute)` (see
   Reporting). No server-side DAG walk: root-vs-cascade is derivable
   client-side from `(status, errors)` as described under "Query time".
   *(Landed.)*

7. **Publish guard.** Block publishing a snapshot when any node is non-OK.
   *(Not yet done.)*

**Activation (landed).** The UI opts in per request via a
`tolerateNodeFailures` flag on the `@context` input (and the `@instance`
directive), defaulting to `false`. It is stashed on `PathsGraphQLContext`
and threaded through `InstanceConfig.enter_instance_context(...)`, which
sets `context.tolerate_node_failures` explicitly on every entry (so the
flag never leaks across requests that reuse a cached instance). It is
**not** yet auto-derived from `source == DRAFT`: all current public
instances are de-facto draft (no published snapshots exist yet), so a
DRAFT-implies-tolerant default would silently make every instance
tolerant. Once the publishing workflow lands, the default can become
`source == DRAFT`.

**Drive computation to populate status (landed).** Status is `None` until
a node is pulled, so the `status`/`errors` fields take a `compute`
argument (see Reporting): `false` for a fast read of the whole graph,
`true` to run `get_output_pl()` on demand (node-detail view, or a deferred
"compute all" follow-up query). No separate compute endpoint is needed.

**Deferred — not part of this work:**

- **Construction-time status (tier 1).** Do **not** modify
  `InstanceLoader`; it is slated for replacement by graph construction
  directly from the Spec definitions, eliminating the YAML dict structure
  altogether. Per-node construction tolerance and metadata validation
  (units, dangling edges, dimension constraints) land as part of *that*
  work, against the new loader. Before it exists, only tolerant
  `AdditiveNode`s self-identify `INCOMPLETE`; other unwired nodes simply
  surface as `FAILED` at compute time — still flagged, still
  unpublishable.

- **`get_input(port)` migration.** When the port-based input accessor
  lands, move the skip-don't-sum logic out of `add_nodes_pl` into
  `get_input`, generalize it to any tolerant node, and delete the
  `add_nodes_pl` special-casing.
