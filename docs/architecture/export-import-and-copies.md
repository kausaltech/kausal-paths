# Export, import and copies

An `InstanceExport` is an instance's model as one document: its authoring snapshot,
its dataset bodies, its framework membership, and the template it inherits from when it
has one (see [template-inheritance.md](template-inheritance.md)). Two operations read it,
and the difference between them is the uuids.

- **An import reproduces the instance.** Everything the export defines keeps its uuid:
  instance, nodes, ports, bindings, dimensions, categories, datasets, schemas, metrics,
  data points, comments, sources. A model developer can import a deployment's instance
  locally, edit it, and export it back, and the receiving deployment sees the same
  entities. The contract is `export(import(x)) == x` over the instance snapshot, the
  dataset bodies, the membership and the template; pages, publication state (an import
  is an unpublished draft) and the document's own provenance are outside it.
- **A copy is the instance under new uuids.** It is the export rekeyed, then imported:
  `import_instance_copy` (used by `copy_instance` and by creating a framework instance from
  its template). The copy refers to the same template, framework and other shared entities
  as its source, and records the source in `copy_of`.

## What each uuid is

Every uuid field in a snapshot says, in its type, what it is (`paths.uuid_kinds`):

| Kind | Example | Copy | Import |
|---|---|---|---|
| `Identity` | `NodeSnapshot.uuid: NodeId` | new uuid | must not exist here |
| `Ref` | `ActionConfig.parent: NodeRef` | follows its target if copied, else kept | target must exist here, unless bundled |
| `Provenance` | `NodeSnapshot.copy_of: NodeCopyOf` | set to the source | kept |
| `Token` | `InstanceExport.draft_head_token` | dropped | not used |

Identities are declared in `paths.identifiers`, references and provenance in `paths.refs`;
`*Ref` always means a uuid, and `*IdentifierRef` a reference by identifier. A test keeps
every uuid reachable from `InstanceExport` marked (`unmarked_uuid_fields`), so a new uuid
field cannot be added without saying what it is.

What a document owns is mostly where the field sits, with two exceptions the models
declare through hooks on the walk (`paths.rekey`):

- **A framework's entities.** Its dimensions and the schemas datasets in several
  instances share appear in an instance's own catalog. `DimensionMeta.scope` and
  `DatasetMeta.schema_scope` say so; a framework schema's metrics are the framework's too.
- **Overrides of the template.** An instance closes a template's shape, or replaces one of
  its bindings, under the template's own uuid. `InstanceExport.__rekey_foreign__` names
  the bundled template, whose identities are someone else's wherever they appear, so a
  copy overrides the same entity. Rekeying refuses a template-built instance without its
  template bundled.

`paths.rekey.survey()` makes the same walk without changing anything, and reports what
the document owns and what it refers to outside itself; the import's checks use it.

## Import rules

`import_instance_export` (behind `paths-devtool instance import`) creates the row under the
document's own uuid and runs `import_instance`, which refuses before writing anything when:

- an entity the export owns is already here (a copy of something here must be rekeyed);
- something it refers to without bundling it is not here: a framework, a framework
  dimension (with the same category uuids and identifiers), a framework schema, a template
  revision it does not carry;
- the instance is already here, unless `replace` is given. Replacing deletes it first and
  points the references the deletion would null (the framework whose template it is,
  copies made from it, users who selected it) at the new row. An instance others inherit
  from is never replaced: their pins name revisions of its row.

A framework member's template is found by content hash, not installed: the framework owns
it. An instance outside any framework brings its template along and installs it when it is
missing. Importing a framework itself is not supported yet.

## Order

An export and its import must list everything alike, so nothing in a snapshot may be
ordered by primary key: datasets go by identifier and uuid, bindings by node uuid, the
node's declared port order and position, override sets by node and port uuid. At runtime a
node takes a role's inputs in its declared port order too (`Node.bind_runtime_inputs`); a
binding's uuid, which a copy changes, must never decide anything.
