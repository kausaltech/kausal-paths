# GraphQL Mutations

## The `gql.mutation` decorator

Use `gql.mutation` (from `paths.gql`) instead of bare `sb.mutation` for
mutations that modify data. It wraps the resolver in a Django
`transaction.atomic()` and integrates with `strawberry-django`'s error
handling via `strawberry_django.mutation(handle_django_errors=True, ...)`:

```python
from paths import gql

@gql.mutation(description='Update an existing node', graphql_type=AnyNodeType)
@staticmethod
def update_node(info: gql.Info, ...) -> Node:
    ...
```

### Automatic transaction

The resolver body runs inside `transaction.atomic()`. You do not need to
open your own transaction unless you need nested savepoints.

### Return type and error union

If the mutation has a success return type, `gql.mutation` automatically
adds `OperationInfo` to the GraphQL result shape. The generated schema
contains an auto-named payload union based on the field name. A resolver
returning `Node` can produce a schema type like:

```graphql
union UpdateNodePayload = ActionNode | Node | OperationInfo
```

On success the resolver returns the domain object directly (no wrapper).
On validation failure the framework catches `ValidationError` and
returns an `OperationInfo` with messages — the resolver does not need to
handle this itself.

If `graphql_type=...` points at a union, the final GraphQL payload union
is flattened. For example, `graphql_type=AnyNodeType` where
`AnyNodeType = ActionNode | Node` becomes:

```graphql
union CreateNodePayload = ActionNode | Node | OperationInfo
```

The exact payload name and members are easiest to verify from the
generated schema.

### Mutations with no success payload

If you do not provide `graphql_type=...` and the resolver's return
annotation is `None`, `gql.mutation` treats the field as a command-style
mutation. In that case the schema exposes `OperationInfo` directly
instead of generating a success/error union:

```graphql
deleteNode(nodeId: ID!): OperationInfo
```

In practice, a successful delete commonly returns `null`, while handled
validation errors return `OperationInfo` with messages. If you want a
non-null success payload, declare it explicitly with `graphql_type=...`
and return that shape from the resolver.

### graphql_type override

When the Python return type doesn't map directly to the desired GraphQL
type, pass `graphql_type=...`:

```python
# Python returns Node, but GraphQL should expose the AnyNodeType union
@gql.mutation(graphql_type=AnyNodeType)
def create_node(...) -> Node: ...
```


## Input types

### Create inputs — `@pydantic_input`

For create mutations, derive the input from the Pydantic spec model
using `@pydantic_input`. Fields with `auto` pull their type from the
Pydantic model:

```python
@pydantic_input(model=NodeSpec, name='CreateNodeInput')
class CreateNodeInput:
    identifier: sb.ID
    name: auto
    kind: auto = NodeKind.FORMULA
    ...
```

Call `.to_pydantic()` on the Strawberry type to get the Pydantic model
instance.

### Update inputs — `Maybe` fields

For partial updates, use plain `@sb.input` with `Maybe[T]` fields.
`Maybe` is Strawberry's way of distinguishing "not provided" from
"provided as null":

```python
@sb.input
class UpdateNodeInput:
    name: Maybe[str]
    color: Maybe[str]
    is_visible: Maybe[bool]
```

Check with `is_maybe_set()` (a `TypeGuard`) and unwrap with `.value`:

```python
def is_maybe_set[T](maybe: Some[T] | None) -> TypeGuard[Some[T]]:
    return maybe is not None and maybe is not sb.UNSET

if is_maybe_set(input.name):
    spec.name = input.name.value
```

### OneOf inputs

For discriminated unions (like node type configs), use `@sb.input(one_of=True)`:

```python
@sb.input(one_of=True)
class NodeConfigInput:
    formula: Maybe[FormulaConfigInput]
    simple: Maybe[SimpleConfigInput]
    action: Maybe[ActionConfigInput]
```


## Mutation patterns by operation

### Create

- Return the created domain object (e.g. `Node`).
- The framework unions it with `OperationInfo` automatically.
- Check permissions with `Model.gql_create_allowed(info, parent)`.

```python
@gql.mutation(graphql_type=AnyNodeType)
def create_node(info, root, input: CreateNodeInput) -> Node:
    ic = root.instance
    ...
    nc = ic.nodes.create(identifier=input.identifier, ...)
    return _resolve_runtime_node(ic, nc.pk)
```

### Update

- Return the updated domain object.
- Use `get_or_error()` to fetch and validate existence.
- Apply partial updates from `Maybe` fields.

```python
@gql.mutation(graphql_type=AnyNodeType)
def update_node(info, root, node_id: sb.ID, input: UpdateNodeInput) -> Node:
    nc = get_or_error(info, ic.nodes.get_queryset(), id=node_id)
    ...
    return _resolve_runtime_node(nc.instance, nc.pk)
```

### Delete

- Return `None` for command-style deletes unless you need an explicit
  success payload.
- With no explicit `graphql_type`, the schema exposes `OperationInfo`
  directly, and successful deletes may return `null`.
- Check object-level permission with `nc.gql_action_allowed(info, 'delete')`.

```python
@gql.mutation(description='Deletes a node from the model')
def delete_node(root, info, node_id: sb.ID) -> None:
    nc = get_or_error(info, ic.nodes.get_queryset(), id=node_id)
    if not nc.gql_action_allowed(info, 'delete'):
        raise PermissionDeniedError(info, 'Permission denied for delete')
    nc.delete()
```


## Generated schema

When in doubt, inspect the generated schema instead of inferring the
final GraphQL shape from Python annotations alone:

```bash
mise export-graphql-schema
```

This writes:

- `__generated__/schema.graphql`
- `__generated__/schema-test-mode.graphql`

This is the most reliable way to confirm:

- auto-generated payload union names such as `CreateNodePayload`
- whether a mutation returns a union or plain `OperationInfo`
- the final field names after Strawberry naming/casing
- the exact nested shapes exposed to clients and tests


## Nested mutation namespaces

Group related mutations under a namespace type to share authorization
and context:

```python
@sb.type
class InstanceEditorMutation:
    instance: sb.Private[InstanceConfig]
    type Me = InstanceEditorMutation

    @gql.mutation(...)
    @staticmethod
    def create_node(info, root: sb.Parent[Me], input: CreateNodeInput) -> Node:
        ic = root.instance  # shared instance, already authorized
        ...
```

The parent mutation field performs the authorization check once:

```python
@sb.type
class ModelEditorMutation:
    @sb.field
    def instance_editor(info, instance_id: sb.ID) -> InstanceEditorMutation:
        ic = _get_instance_config(info, instance_id)
        if not ic.gql_action_allowed(info, 'change'):
            raise PermissionDeniedError(info, 'Model editor access denied')
        return InstanceEditorMutation(instance=ic)
```

Child mutations access the parent via `root: sb.Parent[Me]` and read
`root.instance` without repeating the auth check. Individual mutations
can add finer-grained checks (e.g. `gql_create_allowed`,
`gql_action_allowed('delete')`).


## Error handling

| Error type | When to use | GraphQL result |
|---|---|---|
| `GraphQLValidationError(info, msg)` | Input fails business rules | `OperationInfo` with message |
| `PermissionDeniedError(info, msg)` | Authorization failure | GraphQL error in `errors` |
| `GraphQLError(msg)` | Hard errors (not found, invalid state) | GraphQL error in `errors` |
| Django `ValidationError` | Field-level validation | Caught automatically, becomes `OperationInfo` |


## ClusterableModel caveat

`NodeConfig` inherits from Wagtail's `ClusterableModel`, whose
`save()` can silently revert changes to modeltrans `i18n` fields. When
updating fields that live both on the model and in the `spec` JSON,
use `queryset.update()` instead of `instance.save()`:

```python
updates = {'spec': spec}
if is_maybe_set(input.name):
    spec.name = input.name.value
    updates['name'] = input.name.value  # also update the model column
NodeConfig.objects.filter(pk=nc.pk).update(**updates)
```

This bypasses the `ClusterableModel.save()` chain while ensuring both
the spec JSON and the model columns stay in sync.
