from __future__ import annotations

import uuid
from enum import StrEnum
from functools import cached_property
from typing import Annotated, Any, Literal, cast
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
    EnsureUnitOp,
    FilterColumnOp,
    FilterDimensionOp,
    FilterTemporalOp,
    IndexTemporalOp,
    PortTransformOp,
    RemapLegacyYearsOp,
    RenameColumnOp,
    RenameItemOp,
    SelectMetricOp,
    SetForecastFromOp,
    TagOperationOp,
    forecast_from_transformations,
    unit_from_transformations,
    without_transformations,
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
    Convert one legacy input-dataset filter into port transformations.

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
    transformations: list[PortTransformOp] | None = None
    """
    The transform pipeline, when the definition came from the DB.

    Authoritative when set: the flat fields above are the YAML-era shape, and
    ``to_transformations()`` converts them into this. Keeping both would mean
    two sources of truth for the same semantics, so only one is ever read —
    see ``DatasetWithFilters.kwargs_from_def``.
    """

    def to_transformations(self) -> list[PortTransformOp]:
        """
        Convert the legacy flat fields into the complete ordered pipeline.

        The order reproduces exactly what ``DatasetWithFilters`` used to do in
        hardcoded sequence, so that executing the pipeline literally is
        equivalent to the previous behaviour:

        1. ``rename_column`` — before anything else looks at columns; renames
           can be what makes the selected column (or ``Year``) addressable
        2. ``select_metric`` — alias the bound column to ``Value``, drop its
           nulls, narrow the frame
        3. ``index_temporal`` — put ``Year`` in the index
        4. ``remap_legacy_years`` — placeholder year numbers to real years
        5. ``set_forecast_from`` — forecast synthesis, before the other filters
        6. the remaining filters, in their declared order
        7. ``tag_operation`` — registered dataframe operations, in tag order
        8. ``filter_temporal``, ``drop_nulls``, ``ensure_unit``

        ``interpolate`` is deliberately absent: it applies to datasets that
        have no pipeline at all (``FixedDataset``, ``JSONDataset``), and
        ``GenericDataset`` interpolates at its own point in loading. It stays a
        binding field until those cases are gone.
        """
        transformations: list[PortTransformOp] = []
        rename_cols = [f for f in self.filters if isinstance(f, RenameColumnDatasetFilterDef)]
        other_filters = [f for f in self.filters if not isinstance(f, RenameColumnDatasetFilterDef)]
        for filter_def in rename_cols:
            transformations.extend(input_dataset_filter_to_ops(filter_def))
        if self.column is not None:
            transformations.append(SelectMetricOp())
        transformations.append(IndexTemporalOp())
        transformations.append(RemapLegacyYearsOp())
        if self.forecast_from is not None:
            transformations.append(SetForecastFromOp(year=self.forecast_from))
        for filter_def in other_filters:
            transformations.extend(input_dataset_filter_to_ops(filter_def))
        transformations.extend(TagOperationOp(tag=tag) for tag in self.tags)
        if self.min_year is not None or self.max_year is not None:
            transformations.append(FilterTemporalOp(min_year=self.min_year, max_year=self.max_year))
        if self.dropna:
            transformations.append(DropNullsOp())
        if self.unit is not None:
            transformations.append(EnsureUnitOp(unit=self.unit))
        return transformations


LEGACY_DATASET_SPEC_FIELDS = ('column', 'filters', 'forecast_from', 'dropna', 'min_year', 'max_year', 'unit')
"""The flat fields ``DatasetPortSpec`` used to store before it stored a pipeline."""


def legacy_dataset_spec_to_transformations(data: dict[str, Any]) -> dict[str, Any]:
    """
    Rewrite a stored spec that still has the flat filter fields into a pipeline.

    The migration normalizes storage, but this also runs on read so that rows
    written before it — or by a replica still running older code — convert
    correctly instead of failing validation or silently losing their filters.
    """
    ds_def = InputDatasetDef.model_validate({
        'id': 'placeholder',  # the binding's dataset supplies the real id
        **{key: value for key, value in data.items() if key in LEGACY_DATASET_SPEC_FIELDS and value is not None},
        'tags': data.get('tags') or [],
        'interpolate': data.get('interpolate', False),
        'input_dataset': data.get('input_dataset'),
        'output_dimensions': data.get('output_dimensions'),
    })
    return {
        'transformations': [op.model_dump(mode='json') for op in ds_def.to_transformations()],
        'column': ds_def.column,
        'tags': ds_def.tags,
        'input_dataset': ds_def.input_dataset,
        'interpolate': ds_def.interpolate,
        'output_dimensions': ds_def.output_dimensions,
    }


class DatasetPortSpec(I18nBaseModel):
    """Computation semantics for a dataset-to-input-port binding."""

    model_config = ConfigDict(extra='forbid')

    transformations: list[PortTransformOp] = Field(default_factory=list)
    """
    The transform pipeline, in execution order.

    This is the whole of what the binding does to its data. The YAML-era flat
    fields (``column``, ``filters``, ``forecast_from``, ``dropna``,
    ``min_year``, ``max_year``, ``unit``) are deliberately absent: they said the
    same things less precisely, and keeping both would leave two sources of
    truth. ``InputDatasetDef`` still has them, because that is where YAML
    enters.
    """

    column: str | None = None
    """
    The metric column this binding selects, or None when it consumes the frame whole.

    Where the selection happens is the ``select_metric`` operation; *what* is
    selected is this, which the bound ``DatasetMetric`` will replace once a port
    carries exactly one metric. Unconstrained ``str`` because legacy wide DVC
    datasets use human-readable labels with spaces (e.g. "Trucks and lorries").
    """

    tags: list[str] = Field(default_factory=list)
    """
    Tags, which pick this dataset out at runtime (``get_input_dataset_pl(tag=...)``).

    Only selection now: the transforming half of the tag mechanism is explicit
    as ``tag_operation`` entries in ``transformations``. A tag can legitimately be
    both, so the list keeps every tag and the pipeline carries the ones that
    name a registered dataframe operation.
    """

    input_dataset: str | None = None
    """DVC dataset identifier override (when different from the bound dataset)."""

    interpolate: bool = False
    """
    Fill year gaps by linear interpolation.

    Not an operation, unlike everything else here: interpolation also applies to
    datasets that have no pipeline at all (``FixedDataset``, ``JSONDataset``),
    and ``GenericDataset`` interpolates at its own point during loading. It
    becomes a positional op once those cases are gone.
    """

    output_dimensions: list[DimensionRef] | None = None
    """
    Dimensions the binding claims to produce.

    Authored override of what the dataset schema plus the transform pipeline
    should derive. Read-only over the API and slated for removal; see
    `docs/architecture/dimension-constraints.md`.
    """

    @model_validator(mode='before')
    @classmethod
    def _convert_legacy_shapes(cls, data: Any) -> Any:
        """
        Accept the two spec shapes this field has had before.

        First it stored flat filter fields; then it stored the pipeline under
        ``operations``, before that name was reserved for a node's own
        computation. Rows in either shape convert on read, so deploy order never
        matters; both conversions go away when bindings move to their own table.
        """
        if not isinstance(data, dict) or 'transformations' in data:
            return data
        if 'operations' in data:
            renamed = {key: value for key, value in data.items() if key != 'operations'}
            renamed['transformations'] = data['operations']
            return renamed
        if not any(key in data for key in LEGACY_DATASET_SPEC_FIELDS):
            return data
        return legacy_dataset_spec_to_transformations(cast('dict[str, Any]', data))

    @property
    def forecast_from(self) -> int | None:
        """The year the pipeline starts marking values as forecast. Derived, not stored."""
        return forecast_from_transformations(self.transformations)

    @property
    def unit(self) -> Unit | None:
        """The unit the pipeline coerces to. Derived, not stored."""
        return unit_from_transformations(self.transformations)

    def without_forecast_from(self) -> DatasetPortSpec:
        """Return a copy with forecast synthesis removed, so the dataset default applies."""
        return self.model_copy(update={'transformations': without_transformations(self.transformations, 'set_forecast_from')})

    @classmethod
    def from_input_dataset(cls, ds_def: InputDatasetDef) -> DatasetPortSpec:
        transformations = ds_def.transformations if ds_def.transformations is not None else ds_def.to_transformations()
        return cls(
            transformations=list(transformations),
            column=ds_def.column,
            tags=list(ds_def.tags),
            input_dataset=ds_def.input_dataset,
            interpolate=ds_def.interpolate,
            output_dimensions=ds_def.output_dimensions,
        )

    def to_input_dataset(self, *, id: DatasetIdentifier) -> InputDatasetDef:
        return InputDatasetDef(
            id=id,
            transformations=list(self.transformations),
            column=self.column,
            unit=self.unit,
            tags=self.tags,
            input_dataset=self.input_dataset,
            interpolate=self.interpolate,
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
    """The node's own computation. Not to be confused with a binding's transformations."""


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
