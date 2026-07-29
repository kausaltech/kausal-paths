"""
Mutations for editing what is bound to a node's input ports.

A dataset *binding* is what the editor manipulates, and it is not always one
row: a binding that names no metric expands to one ``DatasetPort`` per metric
the dataset exposes. So mutations resolve a binding from any of its rows' uuids
and then write the whole group, which is also why divergent transformations
between rows of one binding cannot arise through this API. An edge binding is
always exactly one ``NodeEdge`` row.

``bindingEditor`` resolves either kind from one id namespace, but updates are
kind-typed mutations with separate input types: the ``oneOf`` field list is the
applicability contract, so the editor learns what an edge may carry from
introspection rather than from validation errors.

``dataset_index`` is deliberately not part of the surface. It is the position
the YAML sync observed, and it becomes derivable once nodes stop indexing into
``input_dataset_instances``; addressing bindings by uuid survives that.
"""

from typing import TYPE_CHECKING, Annotated, cast

import strawberry as sb
from graphql import GraphQLError
from strawberry import Maybe

from kausal_common.strawberry.errors import GraphQLValidationError

from paths import gql

from nodes.defs.transform_def import (
    PortTransformOp,
    SelectMetricOp,
    modernized_transformations,
    unsupported_transformations_for_binding,
)
from nodes.graphql.types.graph import DatasetPortType, NodeEdgeType
from nodes.graphql.types.transformations import (
    DatasetTransformationInput,
    EdgeTransformationInput,
    dataset_transformations_from_input,
    edge_transformations_from_input,
)
from nodes.models import DatasetPort, NodeEdge

if TYPE_CHECKING:
    from uuid import UUID

    from kausal_common.datasets.models import Dataset as DatasetModel, DatasetMetric

    from nodes.defs.node_defs import DatasetPortSpec
    from nodes.defs.transform_def import EdgeTransformOp
    from nodes.models import InstanceConfig, NodeConfig


@sb.input(description='Bind a dataset metric to an existing input port on a node.')
class BindDatasetInput:
    # A create-style input: every field resolves to a given value or a default,
    # so null means the same as omitted and Maybe would be the wrong tool —
    # there is no existing value for an absent field to leave untouched.
    port_id: sb.ID = sb.field(description='Input port to bind to. The port must already exist.')
    dataset_id: sb.ID = sb.field(description='UUID or identifier of the dataset to bind.')
    metric_id: sb.ID | None = sb.field(
        default=None,
        description='Dataset metric this binding carries. May be omitted only when the dataset exposes exactly one metric.',
    )
    transformations: list[DatasetTransformationInput] | None = sb.field(
        default=None,
        description=(
            'Transformations to apply. When omitted, a working default list is generated; '
            'an explicit empty list means none, which a metric-named binding rejects.'
        ),
    )
    replace: bool = sb.field(
        default=False,
        description=(
            'Atomically displace whatever occupies the port — an edge or a dataset binding — '
            'instead of rejecting the bind. Validation runs first, so a rejected bind leaves '
            'the old binding untouched. Not valid for `multi` ports; delete a specific binding there.'
        ),
    )


@sb.input(description='Change what a dataset binding carries or does.')
class UpdateDatasetBindingInput:
    metric_id: Maybe[sb.ID]
    transformations: Maybe[list[DatasetTransformationInput]] = sb.field(
        description='Replaces the whole list; order is execution order.',
    )
    tags: Maybe[list[str]]


@sb.input(description='Change what an edge binding does.')
class UpdateEdgeBindingInput:
    transformations: Maybe[list[EdgeTransformationInput]] = sb.field(
        description='Replaces the whole list; order is execution order.',
    )
    tags: Maybe[list[str]]


def _dataset_transformations(info: gql.Info, entries: list[DatasetTransformationInput]) -> list[PortTransformOp]:
    try:
        return dataset_transformations_from_input(entries)
    except ValueError as e:
        raise GraphQLValidationError(info, str(e)) from None


def _edge_transformations(info: gql.Info, entries: list[EdgeTransformationInput]) -> list[PortTransformOp]:
    try:
        return edge_transformations_from_input(entries)
    except ValueError as e:
        raise GraphQLValidationError(info, str(e)) from None


def _validate_transformations(
    info: gql.Info,
    transformations: list[PortTransformOp],
    *,
    metric_column: str | None,
) -> None:
    """
    Check a list can be executed as a dataset binding before it is stored.

    Deliberately not checked: whether a dimension is present in the frame at the
    point a transformation touches it. That depends on the preceding
    transformations and on the dataset's actual content, so only the runtime
    knows it.
    """
    unsupported = unsupported_transformations_for_binding(transformations, 'dataset')
    if unsupported:
        kinds = ', '.join(sorted({op.kind for op in unsupported}))
        raise GraphQLValidationError(info, f'Transformations not valid for a dataset binding: {kinds}')

    selects = [op for op in transformations if isinstance(op, SelectMetricOp)]
    if metric_column is not None and not selects:
        raise GraphQLValidationError(
            info,
            'This binding selects a metric column, so its transformations must include `selectMetric`',
        )
    if len(selects) > 1:
        raise GraphQLValidationError(info, 'A binding can select its metric only once')


def _resolve_dataset(info: gql.Info, ic: InstanceConfig, dataset_id: str) -> DatasetModel:
    from kausal_common.datasets.models import Dataset

    qs = Dataset.objects.get_queryset().for_instance_config(ic).select_related('schema')
    dataset = qs.filter(uuid=dataset_id).first() if _looks_like_uuid(dataset_id) else qs.filter(identifier=dataset_id).first()
    if dataset is None:
        raise GraphQLValidationError(info, f'Dataset "{dataset_id}" not found in this instance')
    if dataset.schema is None:
        raise GraphQLValidationError(info, f'Dataset "{dataset_id}" has no schema, so it has no metrics to bind')
    return dataset


def _looks_like_uuid(value: str) -> bool:
    from uuid import UUID as _UUID

    try:
        _UUID(value)
    except ValueError:
        return False
    return True


def _resolve_metric(info: gql.Info, dataset: DatasetModel, metric_id: str) -> DatasetMetric:
    from kausal_common.datasets.models import DatasetMetric

    assert dataset.schema is not None
    qs = DatasetMetric.objects.filter(schema=dataset.schema)
    metric = qs.filter(uuid=metric_id).first() if _looks_like_uuid(metric_id) else qs.filter(name=metric_id).first()
    if metric is None:
        raise GraphQLValidationError(info, f'Metric "{metric_id}" not found in dataset "{dataset.identifier or dataset.uuid}"')
    return metric


def _resolve_port(info: gql.Info, nc: NodeConfig, port_id: str) -> UUID:
    from uuid import UUID as _UUID

    from nodes.graphql.editor import _get_input_port

    try:
        parsed = _UUID(port_id)
    except ValueError:
        spec = nc.spec
        assert spec is not None
        named = spec.input_port_by_identifier.get(port_id)
        if named is None:
            raise GraphQLValidationError(info, f'Input port "{port_id}" not found on node "{nc.identifier}"') from None
        return named.id
    if _get_input_port(nc, parsed) is None:
        raise GraphQLValidationError(info, f'Input port "{port_id}" does not exist on node "{nc.identifier}"')
    return parsed


def _check_port_has_capacity(info: gql.Info, nc: NodeConfig, port_id: UUID) -> None:
    """Reject the binding if the port is already occupied and not declared ``multi``."""
    from nodes.graphql.editor import _get_input_port
    from nodes.models import DatasetPort, NodeEdge

    port = _get_input_port(nc, port_id)
    assert port is not None
    if port.multi:
        return
    if NodeEdge.objects.filter(to_node=nc, to_port=port_id).exists():
        raise GraphQLValidationError(info, f'Input port "{port_id}" already has an edge bound to it')
    if DatasetPort.objects.filter(node=nc, port_id=port_id).exists():
        raise GraphQLValidationError(info, f'Input port "{port_id}" already has a dataset bound to it')


def _port_occupants(info: gql.Info, nc: NodeConfig, port_id: UUID) -> tuple[list[NodeEdge], list[DatasetPort]]:
    """
    Return what a replacing bind displaces from the port.

    Only for non-``multi`` ports, where "replace" is unambiguous: the port holds
    exactly one binding. On a ``multi`` port the caller must delete a specific
    binding instead. Dataset rows are collected per port, not per binding group:
    a fanned-out column-less binding spans ports, and replacing this port must
    not unbind its siblings.
    """
    from nodes.graphql.editor import _get_input_port

    port = _get_input_port(nc, port_id)
    assert port is not None
    if port.multi:
        raise GraphQLValidationError(
            info,
            f'Input port "{port_id}" accepts multiple bindings, so `replace` is ambiguous; delete a specific binding instead',
        )
    edges = list(NodeEdge.objects.filter(to_node=nc, to_port=port_id))
    rows = list(DatasetPort.objects.filter(node=nc, port_id=port_id))
    return edges, rows


def _check_metric_fits_port(info: gql.Info, nc: NodeConfig, port_id: UUID, metric: DatasetMetric) -> None:
    """Reject a binding whose metric cannot supply what the port declares."""
    from nodes.graphql.editor import _get_input_port
    from nodes.units import unit_registry

    port = _get_input_port(nc, port_id)
    assert port is not None
    if port.unit is None or not metric.unit:
        return
    try:
        metric_unit = unit_registry.parse_units(metric.unit)
    except Exception:
        raise GraphQLValidationError(info, f'Metric "{metric.name}" has an unparseable unit: {metric.unit}') from None
    if metric_unit.dimensionality != port.unit.dimensionality:
        raise GraphQLValidationError(
            info,
            f'Metric unit {metric_unit} is not compatible with port unit {port.unit}',
        )


def _default_transformations(metric_column: str | None) -> list[PortTransformOp]:
    """Build the list a freshly created binding needs to load correctly."""
    from nodes.defs.node_defs import InputDatasetDef

    return InputDatasetDef(id='placeholder', column=metric_column).to_transformations()


def _binding_rows(ic: InstanceConfig, binding_id: str) -> list[DatasetPort]:
    """
    Return every row of the binding one of whose rows has this uuid.

    A binding is identified by any of its rows because a column-less binding
    fans out to one row per metric; they share a ``dataset_index``.
    """
    from nodes.models import DatasetPort

    anchor = DatasetPort.objects.filter(instance=ic, uuid=binding_id).select_related('node', 'dataset', 'metric').first()
    if anchor is None:
        return []
    return list(
        DatasetPort.objects
        .filter(
            instance=ic,
            node=anchor.node,
            dataset=anchor.dataset,
            dataset_index=anchor.dataset_index,
        )
        .select_related('node', 'dataset', 'metric')
        .order_by('metric__order', 'port_id')
    )


def _next_dataset_index(nc: NodeConfig) -> int:
    from django.db.models import Max

    from nodes.models import DatasetPort

    highest = DatasetPort.objects.filter(node=nc).aggregate(highest=Max('dataset_index'))['highest']
    return 0 if highest is None else highest + 1


def _spec_for(
    *,
    transformations: list[PortTransformOp],
    metric_column: str | None,
    tags: list[str],
    previous: DatasetPortSpec | None = None,
) -> DatasetPortSpec:
    from nodes.defs.node_defs import DatasetPortSpec

    if previous is not None:
        return previous.model_copy(
            update={'transformations': transformations, 'column': metric_column, 'tags': tags},
        )
    return DatasetPortSpec(transformations=transformations, column=metric_column, tags=tags)


@sb.type(description='Edit one input-port binding, dataset or edge.')
class PortBindingEditorMutation:
    instance: sb.Private['InstanceConfig']
    rows: sb.Private[list[DatasetPort]]
    edge: sb.Private['NodeEdge | None']
    type Me = PortBindingEditorMutation

    @gql.mutation(
        description='Change the metric, transformations or tags of this dataset binding.',
        graphql_type=Annotated['DatasetPortType', sb.lazy('nodes.graphql.types.graph')],
    )
    @staticmethod
    def update_dataset_binding(info: gql.Info, root: sb.Parent[Me], input: UpdateDatasetBindingInput) -> DatasetPortType:
        from nodes.change_ops import gql_change_operation, record_change
        from nodes.graphql.editor import is_maybe_set

        if root.edge is not None:
            raise GraphQLValidationError(info, 'This binding is an edge; use updateEdgeBinding')

        rows: list[DatasetPort] = root.rows
        first = rows[0]
        spec = first.spec

        metric_column = spec.column
        metric = None
        if is_maybe_set(input.metric_id):
            if len(rows) > 1:
                raise GraphQLValidationError(
                    info,
                    'This binding fans out to one row per metric of the dataset, so its metric cannot be changed',
                )
            metric = _resolve_metric(info, first.dataset, str(input.metric_id.value))
            # The column follows the new metric; keeping the old column would
            # select what the previous metric carried.
            metric_column = metric.name or None
            _check_metric_fits_port(info, first.node, first.port_id, metric)

        transformations = list(spec.transformations)
        if is_maybe_set(input.transformations):
            transformations = _dataset_transformations(info, input.transformations.value or [])

        tags = list(spec.tags)
        if is_maybe_set(input.tags):
            tags = list(input.tags.value or [])

        _validate_transformations(info, transformations, metric_column=metric_column)

        with gql_change_operation(info, root.instance, action='node.dataset_binding.update'):
            for row in rows:
                before = row.serializable_data()
                row.spec = _spec_for(
                    transformations=transformations,
                    metric_column=metric_column,
                    tags=tags,
                    previous=row.spec,
                )
                if metric is not None:
                    row.metric = metric
                row.save(update_fields=['spec', 'metric'] if metric is not None else ['spec'])
                record_change(
                    row,
                    action='node.dataset_binding.update',
                    before=before,
                    after=row.serializable_data(),
                )

        return _to_gql(rows[0])

    @gql.mutation(
        description='Change the transformations or tags of this edge binding.',
        graphql_type=Annotated['NodeEdgeType', sb.lazy('nodes.graphql.types.graph')],
    )
    @staticmethod
    def update_edge_binding(info: gql.Info, root: sb.Parent[Me], input: UpdateEdgeBindingInput) -> NodeEdgeType:
        from nodes.change_ops import gql_change_operation, record_change
        from nodes.graphql.editor import is_maybe_set
        from nodes.graphql.types.graph import NodeEdgeType

        edge = root.edge
        if edge is None:
            raise GraphQLValidationError(info, 'This binding is a dataset binding; use updateDatasetBinding')

        # Unrelated updates converge the stored list on the current vocabulary too.
        transformations = modernized_transformations(list(edge.transformations))
        if is_maybe_set(input.transformations):
            transformations = _edge_transformations(info, input.transformations.value or [])

        unsupported = unsupported_transformations_for_binding(transformations, 'edge')
        if unsupported:
            kinds = ', '.join(sorted({op.kind for op in unsupported}))
            raise GraphQLValidationError(info, f'Transformations not valid for an edge binding: {kinds}')

        tags = list(edge.tags or [])
        if is_maybe_set(input.tags):
            tags = list(input.tags.value or [])

        with gql_change_operation(info, root.instance, action='edge.update'):
            before = edge.serializable_data()
            # The applicability check above narrowed the list to EdgeTransformOp members.
            edge.transformations = cast('list[EdgeTransformOp]', transformations)
            edge.tags = tags
            edge.save(update_fields=['transformations', 'tags'])
            record_change(edge, action='edge.update', before=before, after=edge.serializable_data())

        return NodeEdgeType.from_node_edge(edge)

    @gql.mutation(description='Remove this binding, leaving the input port in place.')
    @staticmethod
    def delete_binding(info: gql.Info, root: sb.Parent[Me]) -> None:
        from nodes.change_ops import gql_change_operation, record_change

        if root.edge is not None:
            with gql_change_operation(info, root.instance, action='edge.delete'):
                record_change(root.edge, action='edge.delete', before=root.edge.serializable_data(), after=None)
                root.edge.delete()
            return

        with gql_change_operation(info, root.instance, action='node.dataset_binding.delete'):
            for row in root.rows:
                record_change(
                    row,
                    action='node.dataset_binding.delete',
                    before=row.serializable_data(),
                    after=None,
                )
                row.delete()


def _to_gql(row: DatasetPort) -> DatasetPortType:
    """Build the read type for a binding row, matching the instance-level resolver."""
    from datasets.graphql.types import DatasetType
    from nodes.graphql.types.graph import DatasetMetricRefType, DatasetPortType, NodePortRef, _external_dataset_id_from_dataset

    port = DatasetPortType(
        id=sb.ID(str(row.uuid)),
        uuid=row.uuid,
        port_ref=NodePortRef(node_id=sb.ID(str(row.node.identifier)), port_id=row.port_id),
        metric=DatasetMetricRefType.from_model(row.metric),
        external_dataset_id=_external_dataset_id_from_dataset(row.dataset),
        external_metric_id=row.metric.name,
        tags=list(row.spec.tags),
    )
    port._dataset = DatasetType.from_model(row.dataset)
    port._transformations = list(row.spec.transformations)
    if port._dataset is not None:
        port._dataset._forecast_from = row.spec.forecast_from
    return port


def bind_dataset(info: gql.Info, ic: InstanceConfig, nc: NodeConfig, input: BindDatasetInput) -> DatasetPortType:
    """Create a dataset binding on an existing input port."""
    from nodes.change_ops import gql_change_operation, record_change
    from nodes.models import DatasetPort

    port_id = _resolve_port(info, nc, str(input.port_id))
    displaced_edges: list[NodeEdge] = []
    displaced_rows: list[DatasetPort] = []
    if input.replace:
        displaced_edges, displaced_rows = _port_occupants(info, nc, port_id)
    else:
        _check_port_has_capacity(info, nc, port_id)
    dataset = _resolve_dataset(info, ic, str(input.dataset_id))

    metric = None
    metric_column: str | None = None
    if input.metric_id is not None:
        metric = _resolve_metric(info, dataset, str(input.metric_id))
        metric_column = metric.name
    else:
        metric = _sole_metric_or_error(info, dataset)
    _check_metric_fits_port(info, nc, port_id, metric)

    if input.transformations is not None:
        transformations = _dataset_transformations(info, input.transformations)
    else:
        transformations = _default_transformations(metric_column)
    _validate_transformations(info, transformations, metric_column=metric_column)

    replacing = bool(displaced_edges or displaced_rows)
    action = 'node.dataset_binding.replace' if replacing else 'node.dataset_binding.create'
    with gql_change_operation(info, ic, action=action):
        # All validation has passed; only now may the old binding go, so a
        # rejected bind never leaves the port unbound.
        for edge in displaced_edges:
            record_change(edge, action='edge.delete', before=edge.serializable_data(), after=None)
            edge.delete()
        for displaced in displaced_rows:
            record_change(
                displaced,
                action='node.dataset_binding.delete',
                before=displaced.serializable_data(),
                after=None,
            )
            displaced.delete()
        row = DatasetPort.objects.create(
            instance=ic,
            node=nc,
            port_id=port_id,
            dataset=dataset,
            metric=metric,
            spec=_spec_for(transformations=transformations, metric_column=metric_column, tags=[]),
            dataset_index=_next_dataset_index(nc),
        )
        record_change(row, action='node.dataset_binding.create', before=None, after=row.serializable_data())

    return _to_gql(row)


def _sole_metric_or_error(info: gql.Info, dataset: DatasetModel) -> DatasetMetric:
    """Return the dataset's only metric, or demand an explicit one: fan-out is not built yet."""
    from kausal_common.datasets.models import DatasetMetric

    assert dataset.schema is not None
    metrics = list(DatasetMetric.objects.filter(schema=dataset.schema).order_by('order'))
    if len(metrics) == 1:
        return metrics[0]
    raise GraphQLValidationError(
        info,
        f'Dataset "{dataset.identifier or dataset.uuid}" exposes {len(metrics)} metrics, so `metricId` is required',
    )


def binding_editor(info: gql.Info, ic: InstanceConfig, binding_id: sb.ID) -> PortBindingEditorMutation:
    rows = _binding_rows(ic, str(binding_id))
    if rows:
        return PortBindingEditorMutation(instance=ic, rows=rows, edge=None)
    edge = None
    if _looks_like_uuid(str(binding_id)):
        edge = NodeEdge.objects.filter(instance=ic, uuid=str(binding_id)).select_related('from_node', 'to_node').first()
    if edge is not None:
        return PortBindingEditorMutation(instance=ic, rows=[], edge=edge)
    raise GraphQLError(f'Binding "{binding_id}" not found')
