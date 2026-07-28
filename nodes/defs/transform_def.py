"""
Transform transformations carried by a port binding.

This is the target vocabulary for reshaping data as it arrives at a node
input port, shared by dataset bindings and (eventually) edges. See
`docs/architecture/dimension-constraints.md`.

Two things deliberately absent:

* **Metric selection.** A binding names the single metric it carries (a
  ``DatasetMetric``, or the source's output port). Selecting a column is
  part of the binding's source reference, not an operation.
* **A separate "flatten" op.** Flattening is ``filter_dimension`` with
  ``flatten=True``. The legacy ``FlattenTransformation`` is a port shape
  declaration rather than an operation, and belongs on ``InputPortDef``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from paths.refs import DimensionCategoryRef, DimensionRef

from nodes.units import Unit

if TYPE_CHECKING:
    from collections.abc import Sequence

type PortBindingKind = Literal['edge', 'dataset']

EDGE_AND_DATASET: frozenset[PortBindingKind] = frozenset({'edge', 'dataset'})
DATASET_ONLY: frozenset[PortBindingKind] = frozenset({'dataset'})


class PortTransformOpBase(BaseModel):
    """Common base for port transformations."""

    model_config = ConfigDict(extra='forbid')

    applies_to: ClassVar[frozenset[PortBindingKind]] = EDGE_AND_DATASET
    """Binding kinds that may carry this operation."""

    @model_serializer(mode='wrap')
    def _serialize_with_kind(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """
        Keep the discriminator even under ``exclude_defaults``.

        ``kind`` always equals its default, so it is the first thing dropped —
        and without it the union cannot be deserialized. Parameterless ops would
        serialize to ``{}``.
        """
        data = handler(self)
        data['kind'] = getattr(self, 'kind')  # noqa: B009  (declared on the subclasses)
        return data


class FilterDimensionOp(PortTransformOpBase):
    """
    Keep or exclude categories within a dimension, optionally flattening after.

    ``flatten`` sums over the dimension, so it is only valid for structural
    dimensions — ensemble and decomposition axes need explicit reducers.
    """

    kind: Literal['filter_dimension'] = 'filter_dimension'
    dimension: DimensionRef
    groups: list[str] = Field(default_factory=list)
    categories: list[DimensionCategoryRef] = Field(default_factory=list)
    exclude: bool = False
    flatten: bool = False


class AssignDimensionOp(PortTransformOpBase):
    """Tag every row with a fixed category in a dimension the input doesn't have."""

    kind: Literal['assign_dimension'] = 'assign_dimension'
    dimension: DimensionRef
    category: DimensionCategoryRef


class DropNullsOp(PortTransformOpBase):
    """Drop rows with null values."""

    kind: Literal['drop_nulls'] = 'drop_nulls'


class FilterTemporalOp(PortTransformOpBase):
    """
    Limit the temporal extent of the input.

    ``min_year`` / ``max_year`` are the yearly specialization; a resolution
    field is added when sub-yearly data appears.
    """

    kind: Literal['filter_temporal'] = 'filter_temporal'
    min_year: int | None = None
    max_year: int | None = None


class FilterColumnOp(PortTransformOpBase):
    """
    Filter on a raw column that is not (yet) a modelled dimension.

    Legacy: wide DVC datasets use human-readable column labels that never
    became dimensions. Prefer ``filter_dimension`` for anything modelled.
    """

    applies_to = DATASET_ONLY

    kind: Literal['filter_column'] = 'filter_column'
    column: str
    value: str | None = None
    values: list[str] = Field(default_factory=list)
    ref: str | None = None
    drop_col: bool = True
    exclude: bool = False
    flatten: bool = False


class RenameColumnOp(PortTransformOpBase):
    """Rename a raw column before anything else looks at it. Legacy."""

    applies_to = DATASET_ONLY

    kind: Literal['rename_column'] = 'rename_column'
    column: str
    new_name: str | None = None


class RenameItemOp(PortTransformOpBase):
    """Rename a category value within a column. Legacy."""

    applies_to = DATASET_ONLY

    kind: Literal['rename_item'] = 'rename_item'
    column: str
    old_item: str
    new_item: str


class SetForecastFromOp(PortTransformOpBase):
    """
    Mark values from ``year`` onwards as forecast.

    This sets the metric's forecast *qualifier*; it does not add a column.
    See `docs/architecture/metric-dataframe.md`.
    """

    applies_to = DATASET_ONLY

    kind: Literal['set_forecast_from'] = 'set_forecast_from'
    year: int


class EnsureUnitOp(PortTransformOpBase):
    """
    Coerce the metric columns to a unit.

    Positional: coercion before or after a flatten is not the same thing.
    Converts columns that already carry a unit and forces the unit on ones
    that don't — a distinction that disappears once a port carries exactly
    one metric and the frame states its unit.
    """

    kind: Literal['ensure_unit'] = 'ensure_unit'
    unit: Unit


# --- Legacy stage markers ---------------------------------------------------
#
# These exist so the pipeline is the *complete* recipe and the executor can be
# a single literal pass over it. Each one marks where a stage of the old
# hardcoded loading sequence happened, and each is expected to disappear:
# `select_metric` and `index_temporal` once a port carries one metric over a
# frame that states its own temporal axis, `remap_legacy_years` with the DVC
# datasets that need it, `tag_operation` with the tag mechanism itself.


class SelectMetricOp(PortTransformOpBase):
    """
    Marker for where the binding's metric is picked out of a wide frame.

    Carries no parameters: which metric is selected is the binding's source
    reference, so there is exactly one place to edit it. This op only says
    *when* the selection happens relative to the other transformations, which
    matters because renames may have to run first to make the column
    addressable.
    """

    applies_to = DATASET_ONLY

    kind: Literal['select_metric'] = 'select_metric'


class IndexTemporalOp(PortTransformOpBase):
    """Marker for adding the temporal column to the frame's index."""

    applies_to = DATASET_ONLY

    kind: Literal['index_temporal'] = 'index_temporal'


class RemapLegacyYearsOp(PortTransformOpBase):
    """
    Marker for remapping placeholder year numbers to real years.

    Legacy DVC datasets encode the reference year as 0 or 1 and the target
    year as 100 or 101. Parameterless: the years come from the instance.
    """

    applies_to = DATASET_ONLY

    kind: Literal['remap_legacy_years'] = 'remap_legacy_years'


class TagOperationOp(PortTransformOpBase):
    """
    Apply a named dataframe operation registered under a tag.

    Makes the operation half of the legacy ``tags`` mechanism explicit and
    ordered. Tags used to *select* a dataset rather than transform it stay
    on the binding.
    """

    applies_to = DATASET_ONLY

    kind: Literal['tag_operation'] = 'tag_operation'
    tag: str


type PortTransformOp = Annotated[
    FilterDimensionOp
    | AssignDimensionOp
    | DropNullsOp
    | FilterTemporalOp
    | FilterColumnOp
    | RenameColumnOp
    | RenameItemOp
    | SetForecastFromOp
    | EnsureUnitOp
    | SelectMetricOp
    | IndexTemporalOp
    | RemapLegacyYearsOp
    | TagOperationOp,
    Field(discriminator='kind'),
]


def forecast_from_transformations(transformations: Sequence[PortTransformOp]) -> int | None:
    """Return the year the pipeline starts marking values as forecast, if any."""
    for op in transformations:
        if isinstance(op, SetForecastFromOp):
            return op.year
    return None


def unit_from_transformations(transformations: Sequence[PortTransformOp]) -> Unit | None:
    """Return the unit the pipeline ends up coercing to, if any."""
    for op in reversed(transformations):
        if isinstance(op, EnsureUnitOp):
            return op.unit
    return None


def with_forecast_from(transformations: Sequence[PortTransformOp], year: int) -> list[PortTransformOp]:
    """
    Insert forecast synthesis into a pipeline that has none.

    Placed where the converter puts it — after the temporal stages, before the
    other filters — so a default inherited from the dataset behaves exactly like
    one declared on the binding.
    """
    ops = list(transformations)
    if any(isinstance(op, SetForecastFromOp) for op in ops):
        return ops
    anchors = ('index_temporal', 'remap_legacy_years', 'select_metric')
    insert_at = 0
    for index, op in enumerate(ops):
        if op.kind in anchors:
            insert_at = index + 1
    ops.insert(insert_at, SetForecastFromOp(year=year))
    return ops


def without_transformations(transformations: Sequence[PortTransformOp], *kinds: str) -> list[PortTransformOp]:
    """Return the pipeline with every operation of the given kinds removed."""
    return [op for op in transformations if op.kind not in kinds]


def unsupported_transformations_for_binding(
    transformations: Sequence[PortTransformOp],
    binding_kind: PortBindingKind,
) -> list[PortTransformOp]:
    """Return the transformations that the given binding kind may not carry."""
    return [op for op in transformations if binding_kind not in type(op).applies_to]
