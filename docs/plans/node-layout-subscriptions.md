# Real-time node-layout propagation

## Status

Deferred follow-up to the first shared `NodeLayout` persistence slice. The
database model, snapshot representation, GraphQL reads, and
`updateNodeLayouts` mutation land first; this document preserves the intended
subscription design.

## Goal

When one model-editor session moves one or more cards, every other open editor
for the same instance should receive and apply the committed positions. Keep
the first real-time feature narrowly limited to layout metadata, but choose
transport, authorization, ordering, and reconnect behavior that later model
edit events can reuse.

## Proposed schema

```graphql
type Subscription {
  nodeLayoutsUpdated(instanceId: ID!): NodeLayoutsUpdated!
}

type NodeLayoutsUpdated {
  instanceId: ID!
  layouts: [NodeLayout!]!
  clientMutationId: UUID
}
```

Add an optional `clientMutationId` to `updateNodeLayouts`. Return it in both
the mutation result and subscription event. Receiving one's own event is
idempotent, but the identifier lets the originating client recognize an
acknowledgement and is useful for diagnostics.

User identity is already available through each layout's
`lastModifiedBy`. Add a separate event-level actor only if the UI needs it
without selecting the full modifier object on every layout.

## Backend event path

1. Authorize `nodeLayoutsUpdated(instanceId)` for users who can enter the
   instance model editor.
2. Subscribe the socket to an instance-specific Channels group, for example
   `instance.node_layout.<instance uuid>`. Do not use the existing global
   `instance_change` group.
3. `updateNodeLayouts` commits its transactional batch upsert.
4. Register the `group_send` with `transaction.on_commit()`. Never publish an
   event for a transaction that later rolls back.
5. Include the authoritative rows returned by the database in the event, not
   the unvalidated mutation input.

`clearNodeLayouts` needs a distinct reset event (or an explicit `reset` flag
on the same event type); an empty `layouts` list alone must not be ambiguous
with an empty update batch.

The existing `availableInstances` subscription in
`nodes/graphql/operations.py` demonstrates the Strawberry/Channels listener
pattern, but it rechecks broad view access for every global event. Layout
events should instead use the per-instance group and editor-specific access.

## Ordering and conflicts

The first version can use last committed write wins for the same node. Writes
to different nodes do not conflict because each card has its own row.

Channels delivery order alone is not a sufficient stale-event guard once
multiple application workers and reconnects are involved. Before enabling
subscriptions, add a monotonically increasing per-layout version, incremented
while the row is locked. Return it from reads, mutations, and events. A client
applies an event only when its version is newer than the position it already
holds.

An event arriving for a card currently under the local pointer must not move
that card mid-drag. Queue the newest remote version for that card and reconcile
it after local drag-stop. Other cards can update immediately through React
Flow's imperative node-state setter; do not route remote movements through
shell-owned React state or node inspection.

## Reconnect recovery

Subscriptions are notifications, not a durable event log. On initial connect
and after every reconnect, fetch `InstanceEditor.nodeLayouts` and replace the
client's position snapshot before resuming event application. The bulk field
exists specifically for this inexpensive recovery path.

If a mutation succeeds while the socket is disconnected, its HTTP response is
still authoritative for the originating session. Other sessions converge on
their next reconnect snapshot.

## Frontend transport seam

The backend already exposes Strawberry subscriptions at `/v1/graphql/`, but
the Paths UI Apollo client currently terminates every operation at an HTTP
link. Browser HTTP GraphQL also goes through the Next.js `/api/graphql` proxy,
which performs server-side authentication work; a browser WebSocket cannot be
assumed to inherit that path or credential flow.

Before UI implementation, decide and verify one deployment-compatible route:

- expose a same-origin WebSocket route that the ingress proxies directly to
  the Paths backend and authenticate it with an appropriate cookie; or
- give the browser a short-lived credential that `graphql-ws` sends in
  `connectionParams`, without exposing the long-lived server-side token.

Then add a browser-only `GraphQLWsLink` and split Apollo operations by
operation type. SSR/RSC operations must remain on the existing HTTP link.
Connection retry should trigger the bulk recovery query above after the
subscription is re-established.

## Validation

- Backend subscription test: unauthorized users cannot subscribe; two
  instance groups do not leak events; rollback emits nothing.
- Ordering test: a stale version is ignored after a newer version.
- Browser test with two contexts: dragging in A moves the card in B after
  release, does not change either details pane, and does not move a card under
  B's active pointer.
- Reconnect test: disconnect B, make multiple moves in A, reconnect B, and
  confirm the bulk snapshot converges without replay support.
