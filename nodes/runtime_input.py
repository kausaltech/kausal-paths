"""Request-local adapters from graph input bindings to computation values."""

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from common.polars import PathsDataFrame
    from nodes.datasets import Dataset
    from nodes.defs.binding_def import DatasetBindingDef, EdgeBindingDef, PortBindingDef
    from nodes.node import Node


def _load_dataset_value(definition: DatasetBindingDef, source: Dataset) -> PathsDataFrame:
    df = source.get_copy()
    metric = definition.external_metric_id
    if metric is not None and metric in df.metric_cols and len(df.metric_cols) > 1:
        from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN

        columns = [*df.primary_keys, metric]
        if FORECAST_COLUMN in df.columns:
            columns.append(FORECAST_COLUMN)
        df = df.select(columns).rename({metric: VALUE_COLUMN})
    return df


def _load_node_value(definition: EdgeBindingDef, source: Node, target: Node) -> PathsDataFrame:  # noqa: C901
    import polars as pl

    from nodes.constants import FORECAST_COLUMN, NODE_COLUMN, VALUE_COLUMN
    from nodes.defs.transform_def import FlattenTransformation
    from nodes.exceptions import NodeError
    from nodes.transforms import PipelineEnv, apply_port_transformations

    df = source.get_output_pl()
    if NODE_COLUMN in df.columns:
        df = df.filter(pl.col(NODE_COLUMN) == target.id).drop(NODE_COLUMN)

    column = definition.source_port.column_id
    if column is not None and column in df.metric_cols and (column != VALUE_COLUMN or len(df.metric_cols) > 1):
        columns = [*df.primary_keys, column]
        if FORECAST_COLUMN in df.columns:
            columns.append(FORECAST_COLUMN)
        df = df.select(columns).rename({column: VALUE_COLUMN})
    if len(df.metric_cols) != 1:
        raise NodeError(source, f'Binding {definition.id} does not select exactly one output metric')
    metric = df.metric_cols[0]
    if metric != VALUE_COLUMN:
        df = df.rename({metric: VALUE_COLUMN})
    df = df.filter(pl.col(VALUE_COLUMN).is_not_null())

    for dimension in list(df.dim_ids):
        if df[dimension].null_count() == len(df):
            df = df.drop(dimension)

    operations = [op for op in definition.transformations if not isinstance(op, FlattenTransformation)]
    if operations:
        df = apply_port_transformations(df, operations, PipelineEnv(context=source.context, node=source))

    for tag in definition.tags:
        if tag == 'ignore_content':
            df = df.paths._ignore_content(df, target)
        elif df.paths.has_operation(tag):
            df = df.paths.get_operation(tag)(df, source.context)

    expected_dimensions = {str(dimension) for dimension in definition.target_port.required_dimensions}
    if expected_dimensions and set(df.dim_ids) != expected_dimensions:
        raise NodeError(
            source,
            f'Binding {definition.id} produced dimensions {set(df.dim_ids)}, expected {expected_dimensions}',
        )
    return df


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeInputBinding:
    """
    One immutable binding definition paired with a request-local value loader.

    Runtime nodes and datasets deliberately live here rather than on the cached
    ``InstanceGraph`` models.  ``definition`` is retained for diagnostics and
    cache identity; callers address the binding through ``port_role`` only.
    """

    id: UUID
    port_role: str
    position: int
    source_kind: Literal['node', 'dataset']
    value_loader: Callable[[], PathsDataFrame]
    source_id: str | None = None
    definition: PortBindingDef | None = None

    def get_value(self) -> PathsDataFrame:
        return self.value_loader()

    @classmethod
    def from_graph_binding(
        cls,
        definition: PortBindingDef,
        *,
        port_role: str,
        source: object,
        target: object,
    ) -> RuntimeInputBinding:
        """Pair a graph definition with its request-local runtime source."""
        from nodes.datasets import Dataset
        from nodes.defs.binding_def import DatasetBindingDef, EdgeBindingDef
        from nodes.node import Node

        if isinstance(definition, DatasetBindingDef):
            if not isinstance(source, Dataset):
                raise TypeError(f'Dataset binding {definition.id} has a non-dataset runtime source')

            return cls(
                id=definition.id,
                port_role=port_role,
                position=definition.position,
                source_kind='dataset',
                source_id=source.id,
                definition=definition,
                value_loader=partial(_load_dataset_value, definition, source),
            )

        if not isinstance(definition, EdgeBindingDef):
            raise TypeError(f'Unsupported runtime binding definition {type(definition).__name__}')
        if not isinstance(source, Node) or not isinstance(target, Node):
            raise TypeError(f'Edge binding {definition.id} requires runtime node endpoints')

        return cls(
            id=definition.id,
            port_role=port_role,
            position=definition.position,
            source_kind='node',
            source_id=source.id,
            definition=definition,
            value_loader=partial(_load_node_value, definition, source, target),
        )
