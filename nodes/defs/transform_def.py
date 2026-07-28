"""
Transform operations carried by a port binding.

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

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from paths.refs import DimensionCategoryRef, DimensionRef

if TYPE_CHECKING:
    from collections.abc import Sequence

type PortBindingKind = Literal['edge', 'dataset']

EDGE_AND_DATASET: frozenset[PortBindingKind] = frozenset({'edge', 'dataset'})
DATASET_ONLY: frozenset[PortBindingKind] = frozenset({'dataset'})


class PortTransformOpBase(BaseModel):
    """Common base for port transform operations."""

    model_config = ConfigDict(extra='forbid')

    applies_to: ClassVar[frozenset[PortBindingKind]] = EDGE_AND_DATASET
    """Binding kinds that may carry this operation."""


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


class InterpolateOp(PortTransformOpBase):
    """Linearly interpolate gaps, setting the interpolation qualifier."""

    applies_to = DATASET_ONLY

    kind: Literal['interpolate'] = 'interpolate'


type PortTransformOp = Annotated[
    FilterDimensionOp
    | AssignDimensionOp
    | DropNullsOp
    | FilterTemporalOp
    | FilterColumnOp
    | RenameColumnOp
    | RenameItemOp
    | SetForecastFromOp
    | InterpolateOp,
    Field(discriminator='kind'),
]


class PortTransformPipeline(BaseModel):
    """An ordered pipeline of transform operations."""

    model_config = ConfigDict(extra='forbid')

    operations: list[PortTransformOp] = Field(default_factory=list)


def unsupported_ops_for_binding(
    operations: Sequence[PortTransformOp],
    binding_kind: PortBindingKind,
) -> list[PortTransformOp]:
    """Return the operations that the given binding kind may not carry."""
    return [op for op in operations if binding_kind not in type(op).applies_to]
