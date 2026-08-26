"""
Mutations for editing what is bound to a node's input ports.

A dataset *binding* is what the editor manipulates, and it is not always one
row: a binding that names no metric expands to one ``NodeInputPortBinding``
per metric the dataset exposes (rows sharing a ``dataset_index``). So
mutations resolve a binding from any of its rows' uuids and then write the
whole group, which is also why divergent transformations between rows of one
binding cannot arise through this API. An edge binding is always exactly one
row.

``bindingEditor`` resolves either kind from one id namespace, but updates are
kind-typed mutations with separate input types: the ``oneOf`` field list is the
applicability contract, so the editor learns what an edge may carry from
introspection rather than from validation errors.

``dataset_index`` is deliberately not part of the surface. It is the position
the YAML sync observed, and it becomes derivable once nodes stop indexing into
``input_dataset_instances``; addressing bindings by uuid survives that.
"""

from typing import TYPE_CHECKING, Annotated

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
from nodes.graphql.types.constraints import ConstraintViolationsType
from nodes.graphql.types.graph import DatasetPortType, NodeEdgeType
from nodes.graphql.types.transformations import (
    DatasetTransformationInput,
    EdgeTransformationInput,
    dataset_transformations_from_input,
    edge_transformations_from_input,
)
from nodes.models import NodeInputPortBinding

if TYPE_CHECKING:
    from uuid import UUID

    from kausal_common.datasets.models import Dataset as DatasetModel, DatasetMetric

    from nodes.defs.binding_def import DatasetBindingDef
    from nodes.defs.graph import DatasetMeta
    from nodes.defs.node_defs import DatasetPortSpec
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

    port = _get_input_port(nc, port_id)
    assert port is not None
    if port.multi:
        return
    occupant = NodeInputPortBinding.objects.filter(node=nc, port_id=port_id).first()
    if occupant is not None:
        kind = 'an edge' if occupant.source_node_id is not None else 'a dataset'
        raise GraphQLValidationError(info, f'Input port "{port_id}" already has {kind} bound to it')


def _port_occupants(info: gql.Info, nc: NodeConfig, port_id: UUID) -> list[NodeInputPortBinding]:
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
    return list(NodeInputPortBinding.objects.filter(node=nc, port_id=port_id).select_related('source_node'))


def _default_transformations(metric_column: str | None) -> list[PortTransformOp]:
    """Build the list a freshly created binding needs to load correctly."""
    from nodes.defs.node_defs import InputDatasetDef

    return InputDatasetDef(id='placeholder', column=metric_column).to_transformations()


def _dataset_binding_rows(ic: InstanceConfig, anchor: NodeInputPortBinding) -> list[NodeInputPortBinding]:
    """
    Return every row of the binding one of whose rows is the anchor.

    A binding is identified by any of its rows because a column-less binding
    fans out to one row per metric; they share a ``dataset_index``.
    """
    return list(
        NodeInputPortBinding.objects
        .filter(
            instance=ic,
            node=anchor.node,
            dataset=anchor.dataset,
            dataset_index=anchor.dataset_index,
        )
        .select_related('node', 'dataset', 'metric')
        .order_by('metric__order', 'port_id')
    )


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


def _check_dataset_binding_rewrite(
    info: gql.Info,
    ic: InstanceConfig,
    rows: list[NodeInputPortBinding],
    *,
    metric: DatasetMetric | None,
    transformations: list[PortTransformOp],
    tags: list[str],
) -> ConstraintViolationsType | None:
    """Validate replacing every row of a dataset binding with its rewritten form."""
    from nodes.constraints.validation import BindingChange
    from nodes.graphql.constraint_checks import check_binding_change, dataset_candidate, require_draft_graph

    graph = require_draft_graph(info, ic)
    add_bindings: list[DatasetBindingDef] = []
    add_datasets: tuple[DatasetMeta, ...] = ()
    for row in rows:
        assert row.dataset is not None
        assert row.metric is not None
        current = graph.binding_by_id.get(row.uuid)
        candidate, additions = dataset_candidate(
            graph,
            nc=row.node,
            port_id=row.port_id,
            dataset=row.dataset,
            metric=metric if metric is not None else row.metric,
            transformations=transformations,
            tags=tags,
            binding_id=row.uuid,
            position=current.position if current is not None else None,
            primary_language=ic.primary_language,
        )
        add_bindings.append(candidate)
        add_datasets = add_datasets or additions
    change = BindingChange(
        add_bindings=tuple(add_bindings),
        remove_binding_ids=frozenset(row.uuid for row in rows),
        add_datasets=add_datasets,
    )
    return check_binding_change(info, ic, change)


@sb.type(description='Edit one input-port binding, dataset or edge.')
class PortBindingEditorMutation:
    instance: sb.Private['InstanceConfig']
    rows: sb.Private[list[NodeInputPortBinding]]
    edge: sb.Private['NodeInputPortBinding | None']
    type Me = PortBindingEditorMutation

    @gql.mutation(
        description='Change the metric, transformations or tags of this dataset binding.',
        graphql_type=Annotated['DatasetPortType', sb.lazy('nodes.graphql.types.graph')] | ConstraintViolationsType,
    )
    @staticmethod
    def update_dataset_binding(
        info: gql.Info,
        root: sb.Parent[Me],
        input: UpdateDatasetBindingInput,
    ) -> DatasetPortType | ConstraintViolationsType:
        from nodes.change_ops import gql_change_operation, record_change
        from nodes.graphql.editor import is_maybe_set

        if root.edge is not None:
            raise GraphQLValidationError(info, 'This binding is an edge; use updateEdgeBinding')

        rows: list[NodeInputPortBinding] = root.rows
        first = rows[0]
        assert first.dataset is not None
        spec = first.dataset_spec

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

        transformations = list(spec.transformations)
        if is_maybe_set(input.transformations):
            transformations = _dataset_transformations(info, input.transformations.value or [])
            # Temporal filling is already authoritative in the stored recipe,
            # but remains hidden from GraphQL until independently deployed old
            # model editors understand the new union members.
            from nodes.defs.transform_def import preserve_temporal_fill_transformations

            transformations = preserve_temporal_fill_transformations(transformations, spec.transformations)

        tags = list(spec.tags)
        if is_maybe_set(input.tags):
            tags = list(input.tags.value or [])

        _validate_transformations(info, transformations, metric_column=metric_column)

        violations = _check_dataset_binding_rewrite(
            info,
            root.instance,
            rows,
            metric=metric,
            transformations=transformations,
            tags=tags,
        )
        if violations is not None:
            return violations

        with gql_change_operation(info, root.instance, action='node.dataset_binding.update'):
            for row in rows:
                before = row.serializable_data()
                row.dataset_spec = _spec_for(
                    transformations=transformations,
                    metric_column=metric_column,
                    tags=tags,
                    previous=row.dataset_spec,
                )
                row.transformations = list(transformations)
                row.tags = list(tags)
                update_fields = ['dataset_spec', 'transformations', 'tags']
                if metric is not None:
                    row.metric = metric
                    update_fields.append('metric')
                row.save(update_fields=update_fields)
                record_change(
                    row,
                    action='node.dataset_binding.update',
                    before=before,
                    after=row.serializable_data(),
                )

        return _to_gql(rows[0])

    @gql.mutation(
        description='Change the transformations or tags of this edge binding.',
        graphql_type=Annotated['NodeEdgeType', sb.lazy('nodes.graphql.types.graph')] | ConstraintViolationsType,
    )
    @staticmethod
    def update_edge_binding(
        info: gql.Info,
        root: sb.Parent[Me],
        input: UpdateEdgeBindingInput,
    ) -> NodeEdgeType | ConstraintViolationsType:
        from nodes.change_ops import gql_change_operation, record_change
        from nodes.constraints.validation import BindingChange
        from nodes.graphql.constraint_checks import check_binding_change, edge_candidate, require_draft_graph
        from nodes.graphql.editor import is_maybe_set
        from nodes.graphql.types.graph import NodeEdgeType

        edge = root.edge
        if edge is None:
            raise GraphQLValidationError(info, 'This binding is a dataset binding; use updateDatasetBinding')
        assert edge.source_node is not None
        assert edge.source_port_id is not None

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

        graph = require_draft_graph(info, root.instance)
        current = graph.binding_by_id.get(edge.uuid)
        candidate = edge_candidate(
            graph,
            from_node=edge.source_node,
            from_port=edge.source_port_id,
            to_node=edge.node,
            to_port=edge.port_id,
            transformations=transformations,
            tags=tags,
            binding_id=edge.uuid,
            position=current.position if current is not None else None,
        )
        change = BindingChange(add_bindings=(candidate,), remove_binding_ids=frozenset({edge.uuid}))
        violations = check_binding_change(info, root.instance, change)
        if violations is not None:
            return violations

        with gql_change_operation(info, root.instance, action='edge.update'):
            before = edge.serializable_data()
            edge.transformations = transformations
            edge.tags = tags
            edge.save(update_fields=['transformations', 'tags'])
            record_change(edge, action='edge.update', before=before, after=edge.serializable_data())

        return NodeEdgeType.from_input_binding(edge)

    @gql.mutation(description='Remove this binding, leaving the input port in place.')
    @staticmethod
    def delete_binding(info: gql.Info, root: sb.Parent[Me]) -> None:
        from nodes.change_ops import gql_change_operation, record_change
        from nodes.input_bindings import compact_port_positions

        if root.edge is not None:
            with gql_change_operation(info, root.instance, action='edge.delete'):
                record_change(root.edge, action='edge.delete', before=root.edge.serializable_data(), after=None)
                target_node = root.edge.node
                port_id = root.edge.port_id
                root.edge.delete()
                compact_port_positions(target_node, [port_id])
            return

        with gql_change_operation(info, root.instance, action='node.dataset_binding.delete'):
            target_node = root.rows[0].node
            port_ids = [row.port_id for row in root.rows]
            for row in root.rows:
                record_change(
                    row,
                    action='node.dataset_binding.delete',
                    before=row.serializable_data(),
                    after=None,
                )
                row.delete()
            compact_port_positions(target_node, port_ids)


def _to_gql(row: NodeInputPortBinding) -> DatasetPortType:
    """Build the read type for a dataset binding row, matching the instance-level resolver."""
    from datasets.graphql.types import DatasetType
    from nodes.graphql.types.graph import DatasetMetricRefType, DatasetPortType, NodePortRef, _external_dataset_id_from_dataset

    assert row.dataset is not None
    assert row.metric is not None
    port = DatasetPortType(
        id=sb.ID(str(row.uuid)),
        uuid=row.uuid,
        port_ref=NodePortRef(
            node_uuid=row.node.uuid,
            node_id=sb.ID(str(row.node.identifier)),
            port_id=row.port_id,
        ),
        metric=DatasetMetricRefType.from_model(row.metric),
        external_dataset_id=_external_dataset_id_from_dataset(row.dataset),
        external_metric_id=row.metric.name,
        tags=list(row.dataset_spec.tags),
    )
    port._dataset = DatasetType.from_model(row.dataset)
    port._transformations = list(row.dataset_spec.transformations)
    if port._dataset is not None:
        port._dataset._forecast_from = row.dataset_spec.forecast_from
    return port


def bind_dataset(
    info: gql.Info,
    ic: InstanceConfig,
    nc: NodeConfig,
    input: BindDatasetInput,
) -> DatasetPortType | ConstraintViolationsType:
    """Create a dataset binding on an existing input port."""
    from nodes.change_ops import gql_change_operation, record_change
    from nodes.constraints.validation import BindingChange
    from nodes.graphql.constraint_checks import check_binding_change, dataset_candidate, require_draft_graph
    from nodes.input_bindings import next_dataset_index, next_port_position

    nc.ensure_gql_action_allowed(info, 'change')
    port_id = _resolve_port(info, nc, str(input.port_id))
    displaced: list[NodeInputPortBinding] = []
    if input.replace:
        displaced = _port_occupants(info, nc, port_id)
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

    if input.transformations is not None:
        transformations = _dataset_transformations(info, input.transformations)
    else:
        transformations = _default_transformations(metric_column)
    _validate_transformations(info, transformations, metric_column=metric_column)

    graph = require_draft_graph(info, ic)
    candidate, catalog_additions = dataset_candidate(
        graph,
        nc=nc,
        port_id=port_id,
        dataset=dataset,
        metric=metric,
        transformations=transformations,
        primary_language=ic.primary_language,
    )
    change = BindingChange(
        add_bindings=(candidate,),
        remove_binding_ids=frozenset(binding.uuid for binding in displaced),
        add_datasets=catalog_additions,
    )
    violations = check_binding_change(info, ic, change)
    if violations is not None:
        # Validation failed before any write: a rejected bind leaves the
        # graph — including a would-be displaced binding — untouched.
        return violations

    action = 'node.dataset_binding.replace' if displaced else 'node.dataset_binding.create'
    with gql_change_operation(info, ic, action=action):
        # All validation has passed; only now may the old binding go, so a
        # rejected bind never leaves the port unbound.
        for binding in displaced:
            delete_action = 'edge.delete' if binding.source_node_id is not None else 'node.dataset_binding.delete'
            record_change(binding, action=delete_action, before=binding.serializable_data(), after=None)
            binding.delete()
        spec = _spec_for(transformations=transformations, metric_column=metric_column, tags=[])
        row = NodeInputPortBinding.objects.create(
            instance=ic,
            node=nc,
            port_id=port_id,
            position=next_port_position(nc, port_id),
            dataset=dataset,
            metric=metric,
            transformations=list(spec.transformations),
            tags=list(spec.tags),
            dataset_spec=spec,
            dataset_index=next_dataset_index(nc),
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
    anchor = None
    if _looks_like_uuid(str(binding_id)):
        anchor = (
            NodeInputPortBinding.objects
            .filter(instance=ic, uuid=str(binding_id))
            .select_related('node', 'source_node', 'dataset', 'metric')
            .first()
        )
    if anchor is None:
        raise GraphQLError(f'Binding "{binding_id}" not found')
    anchor.node.ensure_gql_action_allowed(info, 'change')
    if anchor.source_node_id is not None:
        return PortBindingEditorMutation(instance=ic, rows=[], edge=anchor)
    return PortBindingEditorMutation(instance=ic, rows=_dataset_binding_rows(ic, anchor), edge=None)
