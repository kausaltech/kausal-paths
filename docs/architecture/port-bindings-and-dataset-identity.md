# Port bindings and dataset identity

This note captures the current shape of the node-port binding model, why it is
split the way it is, and the phased plan for dataset identity while the system
straddles legacy DVC-backed datasets and database-backed dataset objects.

## Current architecture

There are three distinct layers:

1. Port contracts
2. Binding storage
3. Editor/query projections

### Port contracts

Port contracts live in the Pydantic node spec models:

- `InputPortDef`
- `OutputPortDef`

These define what a node exposes or accepts:

- port id
- quantity
- unit
- dimensional requirements
- multiplicity (`multi`)

They do not store graph wiring.

### Binding storage

Bindings are currently authoritative in relational models:

- `NodeEdge`: output port of one node to input port of another
- `DatasetPort`: dataset metric to input port of a node

This is intentional. Bindings are graph relations, not intrinsic properties of
one node spec.

The important consequence is:

- `NodeSpec` defines interface
- `NodeEdge` and `DatasetPort` define actual connections

`NodeDataset` is no longer a good authoritative representation of dataset
attachments because it does not carry the input-port reference. It may still be
useful as legacy or derived metadata, but not as the canonical wiring model.

### Editor/query projections

For the editor and GraphQL, bindings are often most useful grouped under ports.
We do not persist that nested shape. Instead, we project it from the relational
rows.

`NodeConfigQuerySet.annotate_ports()` annotates each `NodeConfig` with two JSON
arrays:

- edge bindings touching the node
- dataset-port bindings on the node

The corresponding `NodeConfig.port_edge_bindings` and
`NodeConfig.port_dataset_bindings` accessors intentionally raise if the
annotation was not requested. That keeps consumers honest and avoids silently
falling back to empty lists.

This gives us:

- one authoritative storage model
- one efficient query for editor-facing projections
- no duplicated source of truth between DB rows and JSON blobs

## Port compatibility model

The current direction is:

- no built-in distinction between "node-only" and "dataset-only" input ports
- `multi=False`: at most one binding total on the port
- `multi=True`: multiple edges, multiple dataset bindings, or a mix of both

Compatibility is determined by:

- quantity compatibility
- dimension compatibility
- unit dimensionality compatibility

The database no longer enforces single-binding uniqueness at the port level.
That is now an application-level validation concern because multiplicity depends
on the input-port spec.

## Dataset identity: current tension

The binding object has two different identity needs:

1. a globally unique binding identifier for the GraphQL client
2. a stable identifier for the external dataset/metric the binding refers to

Those are not the same thing.

For the binding itself, the right identifier is `DatasetPort.uuid`.

For externally sourced datasets, the useful stable identifiers are currently:

- `Dataset.identifier`
- `DatasetMetric.name`

Those map naturally to the dataset repository identifier and the external metric
identifier used when loading DVC-backed datasets into DB objects.

## Current GraphQL vocabulary

GraphQL should expose these concepts explicitly:

- `DatasetPortType.id`: globally unique id of the binding
- `DatasetPortType.externalDatasetId`: stable external dataset identifier
- `DatasetPortType.externalMetricId`: stable external metric identifier

The `external*` prefix is deliberate. It makes clear that these are not DB
primary keys or GraphQL object ids.

## Phased plan

### Phase 1: explicit external identifiers

Keep the existing `Dataset` and `DatasetMetric` foreign keys on `DatasetPort`,
but expose the external identity vocabulary clearly in the GraphQL layer and in
editor-facing binding projections.

This phase includes:

- renaming ambiguous `datasetId` / `metricId` fields to `externalDatasetId` /
  `externalMetricId`
- adding descriptions to the exposed schema fields
- using `Dataset.identifier` and `DatasetMetric.name` as the external identity
  values

### Phase 2: create placeholder DB objects for external datasets

Create `Dataset` objects for DVC-backed datasets systematically and mark them as
external-backed.

This gives us:

- one object model for datasets
- one relational path for `DatasetPort`
- fewer nullable identity branches

At that point, the system can continue exposing the external identifiers as
useful metadata, but the underlying object graph becomes uniform.

### Phase 3: collapse the transitional vocabulary

Once all datasets that participate in bindings have proper DB objects and the
client can reliably navigate object identity through those objects, we can
reassess whether the explicit `external*` fields still need to exist.

The likely end state is:

- DB object identity is primary
- external identifiers remain only if they still provide real user or tooling
  value
- transitional compatibility fields can be removed

## Open design pressure

Two questions are still intentionally left open:

1. Whether some input ports will eventually need explicit source-kind
   restrictions (`node`, `dataset`, or both)
2. Whether `NodeDataset` should remain as a derived/indexing convenience or be
   removed entirely from the authoritative path

For now, the design bias is to avoid source-kind restrictions until a real
semantic need appears, and to keep the graph wiring authority in `NodeEdge` and
`DatasetPort`.
