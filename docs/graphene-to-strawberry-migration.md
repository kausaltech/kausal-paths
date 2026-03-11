# Graphene to Strawberry Migration Guide

This document captures the architecture, interop mechanisms, and migration patterns
needed when converting Graphene `ObjectType` classes to Strawberry `@sb.type` classes.

## Interop Layer

The codebase has a custom bridge that allows Graphene and Strawberry types to coexist
in a single GraphQL schema. Understanding this bridge is essential for incremental migration.

### Core files

- `kausal_common/graphene/merged_schema.py` — the active bridge (three cooperating classes)
- `kausal_common/strawberry/registry.py` — `register_strawberry_type` decorator and `strawberry_types` set
- `paths/schema.py` — top-level schema assembly where both worlds merge

### How the bridge works

```
@sb.type Query(GrapheneQuery, SBQuery)
    │
    ▼
GrapheneStrawberrySchema (extends FederationSchema)
    │ injects custom converter via schema_converter property setter hack
    ▼
UnifiedGraphQLConverter (extends GraphQLCoreConverter)
    │ owns
    ▼
UnifiedGrapheneTypeMap (extends Graphene's TypeMap)
```

**`UnifiedGrapheneTypeMap.add_type()`** is the routing hub. When it encounters:
- A pure Strawberry type → delegates to `sb_converter.from_type()`
- A Graphene type → calls Graphene's `super().add_type()`, then back-registers the
  result into Strawberry's `type_map` via `_add_strawberry_type()` (attaching a
  synthetic `__strawberry_definition__` to the class)

**`UnifiedGraphQLConverter.from_type()`** intercepts Graphene-named types and routes
them through `add_graphene_type()` → `graphene_type_map.add_type()`.

**`UnifiedGraphQLConverter.get_graphql_fields()`** handles dual-nature types (classes
that are both `@sb.type` and `graphene.ObjectType` subclasses). It merges Strawberry
fields from `super().get_graphql_fields()` with Graphene fields from
`graphene_type_map.create_fields_for_type()`, raising on name collision.

### What this means for migration

- A Strawberry field annotation can reference a Graphene type directly — the converter's
  `from_type()` will route it through the Graphene path automatically.
- A Graphene `graphene.Field()` can point at a Strawberry type — `add_type()` detects
  the Strawberry definition and delegates to the Strawberry converter.
- Types can be migrated one at a time without breaking the schema.

## Schema Assembly

In `paths/schema.py`:

```python
# Graphene root types
class GrapheneQuery(NodesQuery, ParamsQuery, PagesQuery, ...):
    class Meta:
        name = 'Query'

# Strawberry root types (merged from multiple @sb.type classes)
SBQuery = merge_types('Query', (SBNodesQuery,))
SBMutation = merge_types('Mutation', tuple(SB_MUTATION_TYPES))
Subscription = merge_types('Subscription', (NodesSubscription,))

# Unified root types: inherit from BOTH
@sb.type
class Query(GrapheneQuery, SBQuery):
    pass

@sb.type
class Mutation(GrapheneMutations, SBMutation):
    pass

# Schema construction
schema = PathsSchema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
    types=all_types,       # includes strawberry_types set + grapple models
    directives=[...],
)
```

The `types=` argument ensures types not reachable by graph traversal from
Query/Mutation/Subscription are included. This is where `register_strawberry_type`
feeds into.

## Type Registry

`kausal_common/strawberry/registry.py` provides `@register_strawberry_type`, which adds
a type to the `strawberry_types` set. This set is passed to the schema's `types=` argument.

**When to use it:** Only when a type isn't discoverable by walking the field graph from
the root types. If a type is referenced (directly or transitively) from a Strawberry
field annotation on Query/Mutation/Subscription, it will be discovered automatically.
Some existing `@register_strawberry_type` usage may be legacy leftovers from before the
unified converter handled cross-references properly.

## Migration Patterns

### Pattern 1: Simple ObjectType → @sb.type

Graphene:
```python
class FooType(graphene.ObjectType[Foo]):
    id = graphene.ID(required=True)
    name = graphene.String(required=True)
    count = graphene.Int()
```

Strawberry:
```python
@sb.type
class FooType:
    id: sb.ID
    name: str
    count: int | None
```

Strawberry uses `getattr(root, field_name)` as the default resolver, same as Graphene's
`ObjectType[T]` pattern. If the root object's attribute names match the field names,
no explicit resolvers are needed.

**Important:** `@sb.type` creates a dataclass with keyword-only `__init__`. If the type
is constructed directly in code (e.g. `YearlyValue(year, value)`), you must switch to
keyword arguments: `YearlyValue(year=year, value=value)`.

### Pattern 2: Fields with custom resolvers

Graphene:
```python
class FooType(graphene.ObjectType[Foo]):
    display_name = graphene.String(required=True)

    @staticmethod
    def resolve_display_name(root: Foo, info) -> str:
        return root.config.name_i18n
```

Strawberry:
```python
@sb.type
class FooType:
    @sb.field
    @staticmethod
    def display_name(root: Foo) -> str:
        return root.config.name_i18n
```

The `info` parameter can be omitted if the resolver doesn't need it. If it does,
use `info: gql.Info`.

Alternative using `sb.Parent` (when the resolver is an instance method):
```python
@sb.type
class FooType:
    @sb.field
    def display_name(self, parent: sb.Parent[Foo]) -> str:
        return parent.config.name_i18n
```

### Pattern 3: Fields with GraphQL arguments

Graphene:
```python
class FooType(graphene.ObjectType):
    items = graphene.List(
        graphene.NonNull(ItemType),
        required=True,
        category=graphene.String(required=False),
    )

    @staticmethod
    def resolve_items(root, info, category: str | None = None):
        ...
```

Strawberry:
```python
@sb.type
class FooType:
    @sb.field
    @staticmethod
    def items(root: Foo, info: gql.Info, category: str | None = None) -> list[ItemType]:
        ...
```

In Strawberry, field arguments come from the resolver's Python signature. The interop
layer handles this transparently.

### Pattern 4: Referencing Graphene types from Strawberry fields

When a field's return type is a Graphene `ObjectType` that hasn't been migrated yet,
use a plain type annotation. The unified converter's `from_type()` will detect the
Graphene type and route it through the Graphene path.

```python
@sb.type
class FooType:
    # BarType is still a graphene.ObjectType — this works because the
    # converter routes Graphene types through add_graphene_type()
    bar: BarType
```

### Pattern 5: Referencing dynamically generated types (e.g. grapple)

When a type is generated at runtime (e.g. by grapple's `registry.pages[Model]`),
use the `grapple_field` decorator from `kausal_common.strawberry.grapple`. It inspects
the field's type annotation at schema build time, detects Wagtail `Page` subclasses,
and swaps them for the corresponding grapple-registered Graphene type automatically:

```python
from kausal_common.strawberry.grapple import grapple_field

@sb.type
class FooType:
    @grapple_field
    @staticmethod
    def action_list_page(root: Foo) -> ActionListPage | None:
        return root.config.action_list_page
```

`grapple_field` is a drop-in replacement for `sb.field` that adds the
`GrappleRegistryType` field extension. It accepts all the same keyword arguments
as `sb.field`.

Note: `sb.field(graphql_type=...)` does **not** accept callables — it expects concrete
types. The `grapple_field` extension was created specifically to handle the dynamic
type lookup that grapple requires.

### Pattern 6: Wagtail StreamField / non-Strawberry interface types

For fields returning types like Wagtail's `StreamValue` where the GraphQL type is a
Graphene interface (`StreamFieldInterface` from grapple), use `sb.field(graphql_type=...)`
explicitly:

```python
from grapple.types.streamfield import StreamFieldInterface

@sb.type
class FooType:
    @sb.field(graphql_type=list[StreamFieldInterface] | None)
    @staticmethod
    def intro_content(root: Foo, info: gql.Info) -> StreamValue:
        return root.config.site_content.intro_content
```

This approach is explicit and doesn't require interop layer changes. If stream fields
appear frequently, a future improvement would be to teach `UnifiedGraphQLConverter.from_type()`
to map `StreamValue` → `StreamFieldInterface` automatically.

### Pattern 7: Private attributes and internal state

Use `sb.Private[T]` for fields that should not appear in the GraphQL schema but are
needed by resolvers:

```python
@sb.type
class InstanceGoalEntry:
    id: sb.ID
    label: str | None
    _goal: sb.Private[NodeGoalsEntry]

    @sb.field
    def values(self) -> list[InstanceYearlyGoalType]:
        return [InstanceYearlyGoalType.from_pydantic(x) for x in self._goal.get_actual()]
```

`sb.Private` fields are included in the dataclass `__init__` but excluded from the
schema. They replace the Graphene pattern of setting attributes on instances after
construction (`out._goal = goal`).

### Pattern 8: Forward/circular references to types

When a Strawberry type needs to reference a Graphene type defined later in the same
file (or in a circular import), use `sb.field(graphql_type=...)` with `Annotated` and
`sb.lazy`:

```python
@sb.type
class InstanceGoalEntry:
    outcome_node: Node = sb.field(graphql_type=Annotated['NodeType', sb.lazy('nodes.schema')])
```

The Python type annotation (`Node`) reflects the actual runtime type, while
`graphql_type` tells Strawberry which GraphQL type to use. `sb.lazy` defers resolution
until schema build time, avoiding circular import issues.

### Pattern 9: Mismatched Python and GraphQL return types

When a resolver returns a Python object that maps to a different Strawberry type, use
`sb.field(graphql_type=...)` with the Strawberry type and annotate the return with the
Python type:

```python
@sb.field(graphql_type=UnitType)
def unit(self) -> Unit:
    # Returns a Unit object; Strawberry resolves it as UnitType
    return df.get_unit(column_id)
```

### Pattern 10: Enum handling

The preferred approach is to decorate the source enum with `@sb.enum` and use it
directly in type annotations. This eliminates the need for `graphene.Enum.from_enum()`
wrapper variables entirely.

```python
# At the source definition (e.g. nodes/scenario.py)
@sb.enum
class ScenarioKind(enum.Enum):
    BASELINE = 'baseline'
    DEFAULT = 'default'
    CUSTOM = 'custom'

# Use directly in Strawberry type annotations
@sb.type
class ScenarioType:
    kind: ScenarioKind | None
```

When migrating, find the `graphene.Enum.from_enum(SomeEnum)` variable in the schema
file, add `@sb.enum` to the source enum class, then replace references to the Graphene
wrapper with the original enum. The Graphene wrapper can be removed once no remaining
Graphene fields reference it.

**Note:** If the enum is still referenced by unconverted Graphene fields (via
`graphene.List(graphene.NonNull(SomeGrapheneEnum))`), keep the
`graphene.Enum.from_enum()` wrapper alongside the `@sb.enum` decorator until those
fields are migrated too — both can coexist.

## Graphene → Strawberry Type Mapping Reference

| Graphene | Strawberry |
|----------|------------|
| `graphene.String(required=True)` | `str` |
| `graphene.String()` | `str \| None` |
| `graphene.Int(required=True)` | `int` |
| `graphene.Int()` | `int \| None` |
| `graphene.Float(required=True)` | `float` |
| `graphene.Float()` | `float \| None` |
| `graphene.Boolean(required=True)` | `bool` |
| `graphene.Boolean()` | `bool \| None` |
| `graphene.ID(required=True)` | `sb.ID` |
| `graphene.ID()` | `sb.ID \| None` |
| `graphene.List(graphene.NonNull(T), required=True)` | `list[T]` |
| `graphene.List(graphene.NonNull(T))` | `list[T] \| None` |
| `graphene.Field(T, required=True)` | `T` |
| `graphene.Field(T)` | `T \| None` |
| `graphene.ObjectType[Model]` | `@sb.type` with `getattr`-based resolution |
| `graphene.Enum.from_enum(E)` | Add `@sb.enum` to source enum `E`, use `E` directly |

## Existing Strawberry Types (reference examples)

### Pydantic-backed types (`nodes/schema.py`)

Used when there's already a Pydantic model:

```python
@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricCategory)
class MetricDimensionCategoryType:
    id: sb.ID
    label: strawberry.auto
    color: strawberry.auto
```

### Plain @sb.type with custom resolvers (`paths/graphql_types.py`)

```python
@register_strawberry_type
@sb.type(name='UnitType')
class UnitType:
    @sb.field
    def short(self, parent: sb.Parent[Unit]) -> str:
        return format_unit(parent, html=False)
```

### sb.field(graphql_type=...) for Graphene return types (`nodes/schema.py`)

```python
@sb.type
class SBQuery(Query):
    @sb.field(graphql_type=list[NormalizationType])
    @staticmethod
    def active_normalizations(info: gql.Info) -> list[Normalization]:
        ...
```

## Migration Checklist

When converting a Graphene `ObjectType` to Strawberry:

1. **Read the existing type** — understand all fields, resolvers, and arguments.
2. **Identify dependencies** — which other types does it reference? Are they Graphene or Strawberry?
3. **Choose the pattern** — plain `@sb.type` for most cases; `@sb.experimental.pydantic.type`
   if backed by a Pydantic model.
4. **Convert fields** — use the type mapping table. `required=True` → non-optional;
   no `required` or `required=False` → `| None`.
5. **Convert resolvers** — `resolve_foo(root, info, ...)` → `@sb.field` method or
   `@staticmethod` with `root` parameter.
6. **Handle special cases**:
   - Enums → add `@sb.enum` at source, remove `graphene.Enum.from_enum()` wrapper
   - Dynamic types (grapple) → use `grapple_field` decorator
   - StreamField → `sb.field(graphql_type=list[StreamFieldInterface] | None)`
   - Graphene type references → plain annotations (bridge handles it)
   - Direct construction → switch from positional to keyword arguments
7. **Check if `@register_strawberry_type` is needed** — only if the type isn't
   reachable from root types through Strawberry field traversal.
8. **Update references** — Graphene `graphene.Field(OldType)` can point at the new
   Strawberry type directly (bridge handles it). Other Strawberry types can use
   plain annotations.
9. **Verify schema** — export the schema and diff against production to catch
   unintended changes:
   ```bash
   # Export the local schema
   python manage.py export_schema paths.schema > schema.graphql

   # Diff against the production API
   pnpx @graphql-inspector/cli diff https://api.paths.kausal.dev/v1/graphql/ schema.graphql
   ```
   Review InstanceType-specific changes with `| grep "TypeName\."`. Narrowing
   nullability (e.g. `String` → `String!`) is non-breaking; widening or removing
   fields is breaking. The voyager at `/v1/graphql/docs/` is also useful for
   visual inspection.
