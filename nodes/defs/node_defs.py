from __future__ import annotations

import uuid
from enum import StrEnum
from functools import cached_property
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString, TranslatedString

from paths.identifiers import DatasetIdentifier, MetricIdentifier, MixedCaseIdentifier
from paths.refs import ActionGroupRef, DimensionRef, NodeRef, QuantityKindRef

from nodes.constants import DecisionLevel
from nodes.goals import NodeGoals
from nodes.units import Unit
from nodes.visualizations import NodeVisualizations
from params.discover import AnyParameter

from .port_def import InputPortDef, OutputPortDef
from .transform_def import (
    AssignDimensionOp,
    DropNullsOp,
    FilterColumnOp,
    FilterDimensionOp,
    FilterTemporalOp,
    InterpolateOp,
    PortTransformOp,
    PortTransformPipeline,
    RenameColumnOp,
    RenameItemOp,
    SetForecastFromOp,
)


class ColumnDatasetFilterDef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    column: str
    value: str | None = None
    values: list[str] = Field(default_factory=list)
    ref: str | None = None
    drop_col: bool = True
    exclude: bool = False
    flatten: bool = False

    @model_validator(mode='after')
    def validate_model(self) -> ColumnDatasetFilterDef:
        if sum([bool(self.value), bool(self.values), bool(self.ref)]) > 1:
            raise ValueError('Cannot specify multiple filter criteria for the same column')
        return self


class DimensionDatasetFilterDef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    dimension: str
    groups: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    assign_category: str | None = None
    flatten: bool = False


class RenameItemDatasetFilterDef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    rename_item: str
    value: str


class RenameColumnDatasetFilterDef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    rename_col: str
    value: str | None = None


type InputDatasetFilterDef = (
    ColumnDatasetFilterDef | DimensionDatasetFilterDef | RenameItemDatasetFilterDef | RenameColumnDatasetFilterDef
)


def input_dataset_filter_to_ops(filter_def: InputDatasetFilterDef) -> list[PortTransformOp]:
    """
    Convert one legacy input-dataset filter into port transform operations.

    A legacy dimension filter conflates filtering and assignment, so it can
    expand into two ops. The runtime applies the branches in the order
    reproduced here: filter (by group or category) or assign, then flatten.
    """
    if isinstance(filter_def, ColumnDatasetFilterDef):
        return [FilterColumnOp(**filter_def.model_dump())]
    if isinstance(filter_def, DimensionDatasetFilterDef):
        ops: list[PortTransformOp] = []
        if filter_def.assign_category is not None and not filter_def.groups and not filter_def.categories:
            ops.append(AssignDimensionOp(dimension=filter_def.dimension, category=filter_def.assign_category))
            if filter_def.flatten:
                ops.append(FilterDimensionOp(dimension=filter_def.dimension, flatten=True))
            return ops
        if filter_def.assign_category is not None:
            raise ValueError(
                f'Dimension filter for {filter_def.dimension!r} both selects and assigns categories;'
                + ' the runtime ignores the assignment, so the intent is unclear'
            )
        return [
            FilterDimensionOp(
                dimension=filter_def.dimension,
                groups=filter_def.groups,
                categories=filter_def.categories,
                flatten=filter_def.flatten,
            )
        ]
    if isinstance(filter_def, RenameColumnDatasetFilterDef):
        return [RenameColumnOp(column=filter_def.rename_col, new_name=filter_def.value)]
    assert isinstance(filter_def, RenameItemDatasetFilterDef)
    col, old_item = filter_def.rename_item.split('|', 1)
    return [RenameItemOp(column=col, old_item=old_item, new_item=filter_def.value)]


class InputDatasetDef(I18nBaseModel):
    """Definition of an input dataset attached to a node."""

    model_config = ConfigDict(extra='forbid')

    id: DatasetIdentifier
    tags: list[str] = Field(default_factory=list)
    interpolate: bool = False
    input_dataset: str | None = None
    """DVC dataset identifier override (when different from ``id``)."""
    column: str | None = None
    forecast_from: int | None = None
    filters: list[InputDatasetFilterDef] = Field(default_factory=list)
    dropna: bool | None = None
    min_year: int | None = None
    max_year: int | None = None
    unit: Unit | None = None
    output_dimensions: list[DimensionRef] | None = None

    def to_transform_pipeline(self) -> PortTransformPipeline:
        """
        Convert the legacy flat fields into an ordered pipeline.

        The order reproduces what the runtime actually does, so that executing
        the pipeline literally is equivalent to the current behaviour:

        1. ``rename_column`` — applied before anything else looks at columns
           (``_filter_and_process_df``)
        2. metric selection — *not* an op; it is the binding's source reference
        3. ``set_forecast_from`` — forecast synthesis precedes the other filters
        4. the remaining filters, in their declared order (``_filter_df``)
        5. ``filter_temporal`` then ``drop_nulls`` (``_process_output``)
        6. ``interpolate`` — last, in ``post_process``

        Unit coercion is not an op either; it belongs to the binding.
        """
        operations: list[PortTransformOp] = []
        rename_cols = [f for f in self.filters if isinstance(f, RenameColumnDatasetFilterDef)]
        other_filters = [f for f in self.filters if not isinstance(f, RenameColumnDatasetFilterDef)]
        for filter_def in rename_cols:
            operations.extend(input_dataset_filter_to_ops(filter_def))
        if self.forecast_from is not None:
            operations.append(SetForecastFromOp(year=self.forecast_from))
        for filter_def in other_filters:
            operations.extend(input_dataset_filter_to_ops(filter_def))
        if self.min_year is not None or self.max_year is not None:
            operations.append(FilterTemporalOp(min_year=self.min_year, max_year=self.max_year))
        if self.dropna:
            operations.append(DropNullsOp())
        if self.interpolate:
            operations.append(InterpolateOp())
        return PortTransformPipeline(operations=operations)


class DatasetPortSpec(I18nBaseModel):
    """Computation semantics for a dataset-to-input-port binding."""

    model_config = ConfigDict(extra='forbid')

    tags: list[str] = Field(default_factory=list)
    input_dataset: str | None = None
    """DVC dataset identifier override (when different from the bound dataset)."""
    column: str | None = None
    """
    Column the original binding requested, or None for column-less bindings
    (e.g. multi-metric action datasets and legacy wide DVC datasets where the
    node consumes the full frame). The bound ``DatasetMetric`` alone isn't
    enough — it records which metric the port targets for editor purposes,
    but round-trip serialization needs the original column intent. Kept as
    an unconstrained ``str`` to mirror ``InputDatasetDef.column``; legacy
    wide DVC datasets use human-readable column labels with spaces (e.g.
    "Trucks and lorries").
    """
    forecast_from: int | None = None
    filters: list[InputDatasetFilterDef] = Field(default_factory=list)
    dropna: bool | None = None
    min_year: int | None = None
    max_year: int | None = None
    interpolate: bool = False
    unit: Unit | None = None
    output_dimensions: list[DimensionRef] | None = None
    """
    Dimensions the binding claims to produce.

    Authored override of what the dataset schema plus the transform pipeline
    should derive. Read-only over the API and slated for removal; see
    `docs/architecture/dimension-constraints.md`.
    """

    @classmethod
    def from_input_dataset(cls, ds_def: InputDatasetDef) -> DatasetPortSpec:
        return cls(
            tags=ds_def.tags,
            input_dataset=ds_def.input_dataset,
            column=ds_def.column,
            forecast_from=ds_def.forecast_from,
            filters=ds_def.filters,
            interpolate=ds_def.interpolate,
            dropna=ds_def.dropna,
            min_year=ds_def.min_year,
            max_year=ds_def.max_year,
            unit=ds_def.unit,
            output_dimensions=ds_def.output_dimensions,
        )

    def to_input_dataset(self, *, id: DatasetIdentifier) -> InputDatasetDef:
        return InputDatasetDef(
            id=id,
            tags=self.tags,
            input_dataset=self.input_dataset,
            column=self.column,
            forecast_from=self.forecast_from,
            filters=self.filters,
            dropna=self.dropna,
            min_year=self.min_year,
            max_year=self.max_year,
            interpolate=self.interpolate,
            unit=self.unit,
            output_dimensions=self.output_dimensions,
        )


class OutputMetricDef(I18nBaseModel):
    """A single output metric produced by a node."""

    id: MetricIdentifier
    label: I18nString | None = None
    unit: Unit
    quantity: QuantityKindRef | None = None
    column_id: MixedCaseIdentifier | None = None
    """DataFrame column name. When None, the loader infers it from context."""


class NodeKind(StrEnum):
    FORMULA = 'formula'
    PIPELINE = 'pipeline'
    ACTION = 'action'
    SIMPLE = 'simple'


class FormulaConfig(BaseModel):
    """Type-specific config for formula nodes."""

    kind: Literal[NodeKind.FORMULA] = NodeKind.FORMULA
    formula: str


class PipelineOperation(BaseModel):
    """A single operation in a pipeline."""

    # FIXME
    operation: str


class PipelineConfig(BaseModel):
    """Type-specific config for pipeline nodes."""

    kind: Literal[NodeKind.PIPELINE] = NodeKind.PIPELINE
    operations: list[PipelineOperation] = Field(default_factory=list)


class ActionConfig(BaseModel):
    """Type-specific config for action nodes."""

    kind: Literal[NodeKind.ACTION] = NodeKind.ACTION
    node_class: str
    decision_level: DecisionLevel | None = None
    group: ActionGroupRef | None = None
    parent: NodeRef | None = None
    no_effect_value: float | None = None


class SimpleConfig(BaseModel):
    """Type-specific config for nodes that are fully defined by their Python class."""

    kind: Literal[NodeKind.SIMPLE] = NodeKind.SIMPLE
    node_class: str


TypeConfig = Annotated[
    FormulaConfig | ActionConfig | SimpleConfig | PipelineConfig,
    Field(discriminator='kind'),
]


class NodeSpecExtra(BaseModel):
    """
    Attic for legacy node config fields.

    These fields are passed through to the InstanceLoader config dict
    but are not part of the long-term NodeSpec schema. Each field here
    is a candidate for removal once we stop relying on the corresponding
    YAML-era feature. Durable node semantics should be modeled directly on
    NodeSpec; the desired end state is for this class to be empty.
    """

    historical_values: list[tuple[int, float]] | None = None
    forecast_values: list[tuple[int, float]] | None = None
    input_dataset_processors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # Catch-all for anything else the node config had
    other: dict[str, Any] = Field(default_factory=dict)


class NodeSpec(I18nBaseModel):
    """Computation schema for a node, stored as a SchemaField on NodeConfig."""

    uuid: UUID = Field(default_factory=uuid.uuid4)
    identifier: str = ''
    name: I18nString = ''
    short_name: str | TranslatedString | None = None
    """
    Short display label used when this node stands in for a category, for example
    when an additive node disaggregates its output by input node.
    """
    description: I18nString | None = None
    """Short description for the node (in markdown format)"""
    kind: NodeKind = NodeKind.FORMULA
    color: str | None = None
    order: int | None = None
    is_visible: bool = True

    type_config: TypeConfig = Field(default_factory=lambda: SimpleConfig(node_class='simple.AdditiveNode'))

    # Inputs
    input_ports: list[InputPortDef] = Field(default_factory=list)

    # Outputs
    output_ports: list[OutputPortDef] = Field(default_factory=list)

    # Dimensions
    input_dimensions: list[str] = Field(default_factory=list)
    output_dimensions: list[str] = Field(default_factory=list)

    # Computation
    pipeline: list[dict[str, object]] | None = None
    params: list[AnyParameter] = Field(default_factory=list)
    goals: NodeGoals = Field(default_factory=NodeGoals)
    visualizations: NodeVisualizations = Field(default_factory=NodeVisualizations)
    allow_nulls: bool = False
    node_group: str | None = None

    # Node behaviour flags
    is_outcome: bool = False
    # TODO: Replace with a pipeline operation that clips years.
    minimum_year: int | None = None

    # Legacy fields — see NodeSpecExtra docstring
    extra: NodeSpecExtra = NodeSpecExtra()

    @cached_property
    def output_port_by_id(self) -> dict[UUID, OutputPortDef]:
        return {port.id: port for port in self.output_ports}

    @cached_property
    def input_port_by_id(self) -> dict[UUID, InputPortDef]:
        return {port.id: port for port in self.input_ports}

    @cached_property
    def input_port_by_identifier(self) -> dict[str, InputPortDef]:
        """Input ports that have a human-readable identifier, keyed by it."""
        return {port.identifier: port for port in self.input_ports if port.identifier is not None}

    @cached_property
    def output_port_by_identifier(self) -> dict[str, OutputPortDef]:
        """Output ports that have a human-readable identifier, keyed by it."""
        return {port.identifier: port for port in self.output_ports if port.identifier is not None}

    @model_validator(mode='after')
    def validate_port_identifiers(self) -> NodeSpec:
        """Port identifiers must be unique within their direction, since they name the port."""
        for direction, ports in (('input', self.input_ports), ('output', self.output_ports)):
            seen: set[str] = set()
            for port in ports:
                if port.identifier is None:
                    continue
                if port.identifier in seen:
                    raise ValueError(f'Duplicate {direction} port identifier {port.identifier!r} on node {self.identifier!r}')
                seen.add(port.identifier)
        return self
