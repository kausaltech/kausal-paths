"""
Giving input ports a dataset of their own (`NodeEditor.createPortDataset`).

Step 2 of docs/plans/node-owned-datasets.md: the dataset takes its shape from
the ports, belongs to the node, and each port is bound to its metric. The
action wizard (`nodes.graphql.action_wizard`) uses the same primitive,
`nodes.owned_datasets.create_owned_dataset`.
"""

from typing import TYPE_CHECKING, Any

from django.db import transaction

from kausal_common.strawberry.errors import GraphQLValidationError
from kausal_common.users import user_or_none

from datasets.snapshot import metric_column_id
from nodes.change_ops import gql_change_operation
from nodes.constraints.values import PortValue
from nodes.defs.node_defs import InputDatasetDef
from nodes.graphql.bindings import bind_dataset_metric, port_occupants
from nodes.graphql.constraint_checks import require_draft_graph
from nodes.graphql.types.constraints import ConstraintViolationsType
from nodes.models import PreferredInstanceSource
from nodes.owned_datasets import OwnedDataset, OwnedDatasetError, OwnedMetric, create_owned_dataset

if TYPE_CHECKING:
    from uuid import UUID

    from paths import gql

    from nodes.defs.port_def import InputPortDef
    from nodes.instance_graph import InstanceGraph, NodeMeta
    from nodes.models import InstanceConfig, NodeConfig


class _Refused(Exception):  # noqa: N818
    def __init__(self, violations: ConstraintViolationsType) -> None:
        self.violations = violations


def _port_dimensions(
    graph: InstanceGraph, info: gql.Info, ic: InstanceConfig, node: NodeMeta, port: InputPortDef
) -> frozenset[UUID]:
    """Return the dimensions of what an input port receives: the solver's answer, else its node's declaration."""
    solve = info.context.require_constraint_solve(ic, source=PreferredInstanceSource.DRAFT)
    shape = solve.shapes.get(PortValue(node.id, port.id, 'input'))
    if shape is not None and shape.dimensions is not None:
        return shape.dimensions
    paired = node.spec.output_port_by_id.get(port.paired_output_port_id) if port.paired_output_port_id else None
    identifiers = (paired.dimensions if paired is not None else None) or node.spec.input_dimensions
    dimensions = [graph.dimension_by_identifier.get(identifier) for identifier in identifiers]
    if any(dimension is None for dimension in dimensions):
        raise GraphQLValidationError(info, f'The dimensions of port {port.identifier or port.id} are not known')
    return frozenset(dimension.id for dimension in dimensions if dimension is not None)


def _label(node: NodeMeta, port: InputPortDef) -> str:
    paired = node.spec.output_port_by_id.get(port.paired_output_port_id) if port.paired_output_port_id else None
    for value in (port.label, paired.label if paired else None, port.identifier, paired.identifier if paired else None):
        if value:
            return str(value)
    return str(node.name or node.identifier)


def _free_ports(info: gql.Info, nc: NodeConfig, node: NodeMeta, port_ids: list[UUID]) -> list[InputPortDef]:
    """Resolve the named input ports, refusing an unknown, unitless or already bound one."""
    if not port_ids:
        raise GraphQLValidationError(info, 'Name at least one input port of the node')
    if len(set(port_ids)) != len(port_ids):
        raise GraphQLValidationError(info, 'Each port may be named once')
    ports = []
    for port_id in port_ids:
        port = node.spec.input_port_by_id.get(port_id)
        if port is None:
            raise GraphQLValidationError(info, f'The node has no input port {port_id}')
        if port.unit is None:
            raise GraphQLValidationError(info, f'Port {port.identifier or port.id} has no unit')
        if port_occupants(info, nc, port_id):
            raise GraphQLValidationError(info, f'Port {port.identifier or port.id} is already bound')
        ports.append(port)
    return ports


def _bind_ports(info: gql.Info, ic: InstanceConfig, nc: NodeConfig, owned: OwnedDataset, ports: list[InputPortDef]) -> None:
    """Bind each port to its metric of `owned`; a constraint refusal raises `_Refused`."""
    for port in ports:
        metric = owned.metrics[port.id]
        result = bind_dataset_metric(
            info,
            ic,
            nc,
            port_id=port.id,
            dataset=owned.dataset,
            metric=metric,
            transformations=InputDatasetDef(id='placeholder', column=metric_column_id(metric)).to_transformations(),
            displaced=[],
            replace=False,
        )
        if isinstance(result, ConstraintViolationsType):
            raise _Refused(result)


def create_port_dataset(info: gql.Info, ic: InstanceConfig, nc: NodeConfig, port_ids: list[UUID], name: str | None) -> Any:
    """Create a dataset owned by `nc` with one metric per input port, and bind each port to its metric."""
    from datasets.graphql.types import DatasetType

    nc.ensure_gql_action_allowed(info, 'change')
    graph = require_draft_graph(info, ic)
    node = graph.node_by_id.get(nc.uuid)
    if node is None:
        raise GraphQLValidationError(info, 'Node not found')
    ports = _free_ports(info, nc, node, port_ids)
    dimension_sets = {_port_dimensions(graph, info, ic, node, port) for port in ports}
    if len(dimension_sets) != 1:
        raise GraphQLValidationError(info, 'The ports of one dataset must have the same dimensions')
    order = {dimension.id: index for index, dimension in enumerate(graph.dimensions)}
    dimension_ids = sorted(dimension_sets.pop(), key=order.__getitem__)

    try:
        with transaction.atomic(), gql_change_operation(info, ic, action='node.dataset.create'):
            try:
                owned = create_owned_dataset(
                    nc,
                    name=name or str(node.name or node.identifier),
                    metrics=[
                        OwnedMetric(port_id=port.id, label=_label(node, port), unit=port.unit, quantity=port.quantity)
                        for port in ports
                        if port.unit is not None
                    ],
                    dimension_ids=dimension_ids,
                    user=user_or_none(info.context.user),
                )
            except OwnedDatasetError as exc:
                raise GraphQLValidationError(info, str(exc)) from exc
            _bind_ports(info, ic, nc, owned, ports)
    except _Refused as refused:
        return refused.violations
    return DatasetType.from_model(owned.dataset)
