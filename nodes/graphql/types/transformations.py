"""
GraphQL surface for a port binding's transformations.

One union on the read side, shared by dataset bindings and edges — the schema
should not have two vocabularies for the same idea. On the write side there is
one ``oneOf`` input per binding kind, so the field list *is* the applicability
contract: the editor learns what an edge may carry from introspection, and an
inapplicable transformation cannot even be expressed.
``unsupported_transformations_for_binding`` stays as server-side defense.

Editing is whole-list replacement: transformations have no identity of their
own, so there is nothing for a granular add/remove/reorder API to address, and
replacement is idempotent with the instance version token supplying optimistic
locking. That does mean a client has to send back the transformations it doesn't
itself understand — ``isSystemManaged`` marks those, and ``bindDataset``
generates a working list server-side so nothing needs to know about them just to
create a binding.
"""

from typing import TYPE_CHECKING, Annotated

import strawberry as sb
from strawberry import UNSET, Maybe, auto

from kausal_common.strawberry.pydantic import StrawberryPydanticType, pydantic_input, pydantic_type

from nodes.defs.transform_def import (
    AssignCategoryTransformation,
    AssignDimensionOp,
    DropNullsOp,
    EnsureUnitOp,
    FilterColumnOp,
    FilterDimensionOp,
    FilterTemporalOp,
    FlattenTransformation,
    IndexTemporalOp,
    PortTransformOp,
    RemapLegacyYearsOp,
    RenameColumnOp,
    RenameItemOp,
    SelectCategoriesTransformation,
    SelectMetricOp,
    SetForecastFromOp,
    TagOperationOp,
    modernized_transformations,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# --- Output types -----------------------------------------------------------


@pydantic_type(model=FilterDimensionOp)
class FilterDimensionType(StrawberryPydanticType[FilterDimensionOp]):
    dimension: auto
    groups: auto
    categories: auto
    exclude: auto
    flatten: auto


@pydantic_type(model=AssignDimensionOp)
class AssignDimensionType(StrawberryPydanticType[AssignDimensionOp]):
    dimension: auto
    category: auto


@pydantic_type(model=DropNullsOp)
class DropNullsType(StrawberryPydanticType[DropNullsOp]):
    kind: str = sb.field(description='Discriminator; the type name identifies this transformation too.')


@pydantic_type(model=FilterTemporalOp)
class FilterTemporalType(StrawberryPydanticType[FilterTemporalOp]):
    min_year: auto
    max_year: auto


@pydantic_type(model=FilterColumnOp)
class FilterColumnType(StrawberryPydanticType[FilterColumnOp]):
    column: auto
    value: auto
    values: auto
    ref: auto
    drop_col: auto
    exclude: auto
    flatten: auto


@pydantic_type(model=RenameColumnOp)
class RenameColumnType(StrawberryPydanticType[RenameColumnOp]):
    column: auto
    new_name: auto


@pydantic_type(model=RenameItemOp)
class RenameItemType(StrawberryPydanticType[RenameItemOp]):
    column: auto
    old_item: auto
    new_item: auto


@pydantic_type(model=SetForecastFromOp)
class SetForecastFromType(StrawberryPydanticType[SetForecastFromOp]):
    year: auto


@pydantic_type(model=EnsureUnitOp)
class EnsureUnitType(StrawberryPydanticType[EnsureUnitOp]):
    unit: auto


@pydantic_type(model=SelectMetricOp)
class SelectMetricType(StrawberryPydanticType[SelectMetricOp]):
    kind: str = sb.field(description='Discriminator; the type name identifies this transformation too.')


@pydantic_type(model=IndexTemporalOp)
class IndexTemporalType(StrawberryPydanticType[IndexTemporalOp]):
    kind: str = sb.field(description='Discriminator; the type name identifies this transformation too.')


@pydantic_type(model=RemapLegacyYearsOp)
class RemapLegacyYearsType(StrawberryPydanticType[RemapLegacyYearsOp]):
    kind: str = sb.field(description='Discriminator; the type name identifies this transformation too.')


@pydantic_type(model=TagOperationOp)
class TagOperationType(StrawberryPydanticType[TagOperationOp]):
    tag: auto


@pydantic_type(model=SelectCategoriesTransformation)
class SelectCategoriesType(StrawberryPydanticType[SelectCategoriesTransformation]):
    dimension: auto
    categories: auto
    flatten: auto
    exclude: auto


@pydantic_type(model=AssignCategoryTransformation)
class AssignCategoryType(StrawberryPydanticType[AssignCategoryTransformation]):
    dimension: auto
    category: auto


@pydantic_type(model=FlattenTransformation)
class FlattenType(StrawberryPydanticType[FlattenTransformation]):
    dimension: auto


PortTransformationType = Annotated[
    FilterDimensionType
    | AssignDimensionType
    | DropNullsType
    | FilterTemporalType
    | FilterColumnType
    | RenameColumnType
    | RenameItemType
    | SetForecastFromType
    | EnsureUnitType
    | SelectMetricType
    | IndexTemporalType
    | RemapLegacyYearsType
    | TagOperationType
    | SelectCategoriesType
    | AssignCategoryType
    | FlattenType,
    sb.union('PortTransformationUnion'),
]


SYSTEM_MANAGED_KINDS = frozenset({'select_metric', 'index_temporal', 'remap_legacy_years'})
"""
Transformations a client should preserve rather than author.

They mark where stages of the old hardcoded loading sequence happen, and they
are generated when a binding is created. They disappear as the legacy shapes
they describe do; see `docs/architecture/dimension-constraints.md`.
"""


def is_system_managed(transformation: PortTransformOp) -> bool:
    return transformation.kind in SYSTEM_MANAGED_KINDS


# --- Input types ------------------------------------------------------------


@pydantic_input(model=FilterDimensionOp)
class FilterDimensionInput(StrawberryPydanticType[FilterDimensionOp]):
    dimension: auto
    groups: auto
    categories: auto
    exclude: auto
    flatten: auto


@pydantic_input(model=AssignDimensionOp)
class AssignDimensionInput(StrawberryPydanticType[AssignDimensionOp]):
    dimension: auto
    category: auto


@pydantic_input(model=FilterTemporalOp)
class FilterTemporalInput(StrawberryPydanticType[FilterTemporalOp]):
    min_year: auto
    max_year: auto


@pydantic_input(model=FilterColumnOp)
class FilterColumnInput(StrawberryPydanticType[FilterColumnOp]):
    column: auto
    value: auto
    values: auto
    ref: auto
    drop_col: auto
    exclude: auto
    flatten: auto


@pydantic_input(model=RenameColumnOp)
class RenameColumnInput(StrawberryPydanticType[RenameColumnOp]):
    column: auto
    new_name: auto


@pydantic_input(model=RenameItemOp)
class RenameItemInput(StrawberryPydanticType[RenameItemOp]):
    column: auto
    old_item: auto
    new_item: auto


@pydantic_input(model=SetForecastFromOp)
class SetForecastFromInput(StrawberryPydanticType[SetForecastFromOp]):
    year: auto


@pydantic_input(model=EnsureUnitOp)
class EnsureUnitInput(StrawberryPydanticType[EnsureUnitOp]):
    unit: str = sb.field(description='Unit expression, e.g. `kt/a`. Parsed on the way in.')


@pydantic_input(model=TagOperationOp)
class TagOperationInput(StrawberryPydanticType[TagOperationOp]):
    tag: auto


@pydantic_input(model=SelectCategoriesTransformation)
class SelectCategoriesInput(StrawberryPydanticType[SelectCategoriesTransformation]):
    dimension: auto
    categories: auto
    flatten: auto
    exclude: auto


@pydantic_input(model=AssignCategoryTransformation)
class AssignCategoryInput(StrawberryPydanticType[AssignCategoryTransformation]):
    dimension: auto
    category: auto


@pydantic_input(model=FlattenTransformation)
class FlattenInput(StrawberryPydanticType[FlattenTransformation]):
    dimension: auto


@sb.input(
    one_of=True,
    description='Exactly one transformation of a dataset binding. Order in the containing list is execution order.',
)
class DatasetTransformationInput:
    filter_dimension: Maybe[FilterDimensionInput]
    assign_dimension: Maybe[AssignDimensionInput]
    drop_nulls: Maybe[bool]
    filter_temporal: Maybe[FilterTemporalInput]
    filter_column: Maybe[FilterColumnInput]
    rename_column: Maybe[RenameColumnInput]
    rename_item: Maybe[RenameItemInput]
    set_forecast_from: Maybe[SetForecastFromInput]
    ensure_unit: Maybe[EnsureUnitInput]
    select_metric: Maybe[bool]
    index_temporal: Maybe[bool]
    remap_legacy_years: Maybe[bool]
    tag_operation: Maybe[TagOperationInput]


@sb.input(
    one_of=True,
    description=(
        'Exactly one transformation of an edge binding. Order in the containing list is '
        'execution order. Only the dimension-reshaping transformations are accepted until '
        'edges execute the shared transform pipeline.'
    ),
)
class EdgeTransformationInput:
    filter_dimension: Maybe[FilterDimensionInput]
    assign_dimension: Maybe[AssignDimensionInput]
    select_categories: Maybe[SelectCategoriesInput] = sb.field(
        default=UNSET,
        deprecation_reason='Use filterDimension instead.',
    )
    assign_category: Maybe[AssignCategoryInput] = sb.field(
        default=UNSET,
        deprecation_reason='Use assignDimension instead.',
    )
    flatten: Maybe[FlattenInput] = sb.field(
        default=UNSET,
        deprecation_reason='A port shape declaration, not a transformation; it moves onto the input port.',
    )


_DATASET_INPUT_FIELDS = (
    'filter_dimension',
    'assign_dimension',
    'drop_nulls',
    'filter_temporal',
    'filter_column',
    'rename_column',
    'rename_item',
    'set_forecast_from',
    'ensure_unit',
    'select_metric',
    'index_temporal',
    'remap_legacy_years',
    'tag_operation',
)

_EDGE_INPUT_FIELDS = (
    'filter_dimension',
    'assign_dimension',
    'select_categories',
    'assign_category',
    'flatten',
)


_PARAMETERLESS: dict[str, Callable[[], PortTransformOp]] = {
    'drop_nulls': DropNullsOp,
    'select_metric': SelectMetricOp,
    'index_temporal': IndexTemporalOp,
    'remap_legacy_years': RemapLegacyYearsOp,
}
"""Transformations with nothing to configure; given as ``true`` rather than an empty object."""


def _transformation_from_one_of(value: object, field_names: tuple[str, ...]) -> PortTransformOp:
    """Convert one ``oneOf`` input entry into its pydantic transformation."""
    for field_name in field_names:
        entry = getattr(value, field_name, None)
        if entry is None or entry is UNSET:
            continue
        # `Maybe` fields arrive wrapped, so that "absent" and "explicitly null"
        # stay distinguishable.
        given = entry.value if hasattr(entry, 'value') else entry
        if given is None:
            continue
        parameterless = _PARAMETERLESS.get(field_name)
        if parameterless is not None:
            if given is not True:
                raise ValueError(f'{field_name} takes `true`; it has nothing to configure')
            return parameterless()
        return given.to_pydantic()
    raise ValueError('No transformation given; exactly one field must be set')


def dataset_transformations_from_input(entries: list[DatasetTransformationInput]) -> list[PortTransformOp]:
    return [_transformation_from_one_of(entry, _DATASET_INPUT_FIELDS) for entry in entries]


def edge_transformations_from_input(entries: list[EdgeTransformationInput]) -> list[PortTransformOp]:
    """Convert edge input entries, rewriting the deprecated legacy kinds into the current vocabulary."""
    return modernized_transformations([_transformation_from_one_of(entry, _EDGE_INPUT_FIELDS) for entry in entries])
