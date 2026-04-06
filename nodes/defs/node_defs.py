from __future__ import annotations

import uuid
from enum import StrEnum
from functools import cached_property
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from paths.identifiers import DatasetIdentifier, MetricIdentifier, MixedCaseIdentifier
from paths.refs import ActionGroupRef, DimensionRef, NodeRef, QuantityKindRef

from nodes.constants import DecisionLevel
from nodes.goals import NodeGoals
from nodes.units import Unit
from nodes.visualizations import NodeVisualizations
from params.discover import AnyParameter

from .port_def import InputPortDef, OutputPortDef


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


class SelectColumnDatasetTransformOp(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['select_column'] = 'select_column'
    column: str


class FilterColumnDatasetTransformOp(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['filter_column'] = 'filter_column'
    column: str
    value: str | None = None
    values: list[str] = Field(default_factory=list)
    ref: str | None = None
    drop_col: bool = True
    exclude: bool = False
    flatten: bool = False


class FilterDimensionDatasetTransformOp(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['filter_dimension'] = 'filter_dimension'
    dimension: str
    groups: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    assign_category: str | None = None
    flatten: bool = False


class RenameItemDatasetTransformOp(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['rename_item'] = 'rename_item'
    column: str
    old_item: str
    new_item: str


class RenameColumnDatasetTransformOp(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['rename_column'] = 'rename_column'
    column: str
    new_name: str | None = None


class DropNullsDatasetTransformOp(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['drop_nulls'] = 'drop_nulls'


class LimitYearsDatasetTransformOp(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['limit_years'] = 'limit_years'
    min_year: int | None = None
    max_year: int | None = None


class ForecastFromDatasetTransformOp(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['forecast_from'] = 'forecast_from'
    year: int


type DatasetTransformOp = (
    SelectColumnDatasetTransformOp
    | FilterColumnDatasetTransformOp
    | FilterDimensionDatasetTransformOp
    | RenameItemDatasetTransformOp
    | RenameColumnDatasetTransformOp
    | DropNullsDatasetTransformOp
    | LimitYearsDatasetTransformOp
    | ForecastFromDatasetTransformOp
)


class DatasetTransformPipelineDef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    operations: list[DatasetTransformOp] = Field(default_factory=list)


def input_dataset_filter_to_transform_op(filter_def: InputDatasetFilterDef) -> DatasetTransformOp:
    if isinstance(filter_def, ColumnDatasetFilterDef):
        return FilterColumnDatasetTransformOp(**filter_def.model_dump())
    if isinstance(filter_def, DimensionDatasetFilterDef):
        return FilterDimensionDatasetTransformOp(**filter_def.model_dump())
    if isinstance(filter_def, RenameColumnDatasetFilterDef):
        return RenameColumnDatasetTransformOp(column=filter_def.rename_col, new_name=filter_def.value)
    assert isinstance(filter_def, RenameItemDatasetFilterDef)
    col, old_item = filter_def.rename_item.split('|', 1)
    return RenameItemDatasetTransformOp(column=col, old_item=old_item, new_item=filter_def.value)


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

    def to_transform_pipeline(self) -> DatasetTransformPipelineDef:
        operations: list[DatasetTransformOp] = []
        if self.column is not None:
            operations.append(SelectColumnDatasetTransformOp(column=self.column))
        operations.extend(input_dataset_filter_to_transform_op(filter_def) for filter_def in self.filters)
        if self.forecast_from is not None:
            operations.append(ForecastFromDatasetTransformOp(year=self.forecast_from))
        if self.dropna:
            operations.append(DropNullsDatasetTransformOp())
        if self.min_year is not None or self.max_year is not None:
            operations.append(LimitYearsDatasetTransformOp(min_year=self.min_year, max_year=self.max_year))
        return DatasetTransformPipelineDef(operations=operations)


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
    YAML-era feature.
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

    # Datasets and dimensions
    input_datasets: list[InputDatasetDef] = Field(default_factory=list)
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
