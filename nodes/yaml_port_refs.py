"""Compatibility catalog for preserving port UUIDs while parsing YAML."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from uuid import UUID

    from nodes.models import InstanceConfig, NodeConfig


class AmbiguousYamlPortReferenceError(ValueError):
    """A legacy YAML structure matches more than one persisted port UUID."""


def _add(index: dict[tuple[object, ...], set[UUID]], key: tuple[object, ...], port_id: UUID) -> None:
    index.setdefault(key, set()).add(port_id)


def _one(ids: Iterable[UUID], *, description: str) -> UUID | None:
    matches = set(ids)
    if not matches:
        return None
    if len(matches) > 1:
        rendered = ', '.join(str(value) for value in sorted(matches, key=str))
        raise AmbiguousYamlPortReferenceError(f'Ambiguous persisted ports for {description}: {rendered}')
    return next(iter(matches))


@dataclass(frozen=True)
class YamlPortReferenceCatalog:
    """
    Resolve legacy YAML structural selectors to already-persisted port UUIDs.

    This adapter is intentionally limited to YAML parsing. DB drafts and
    published snapshots already carry UUID references and must not pass
    through identifier matching.
    """

    output_ports: dict[tuple[object, ...], set[UUID]] = field(default_factory=dict)
    input_roles: dict[tuple[object, ...], set[UUID]] = field(default_factory=dict)
    edge_ports: dict[tuple[object, ...], set[UUID]] = field(default_factory=dict)
    dataset_ports: dict[tuple[object, ...], set[UUID]] = field(default_factory=dict)
    dataset_groups: dict[tuple[object, ...], set[UUID]] = field(default_factory=dict)

    def output_port_id(self, node_id: UUID, selectors: Iterable[str], fallback: UUID) -> UUID:
        selectors = tuple(selectors)
        matches: set[UUID] = set()
        for selector in selectors:
            matches.update(self.output_ports.get((node_id, selector), ()))
        return _one(matches, description=f'output port {node_id}:{selectors}') or fallback

    def input_role_id(self, node_id: UUID, role: str, fallback: UUID) -> UUID:
        return _one(self.input_roles.get((node_id, role), ()), description=f'input role {node_id}:{role}') or fallback

    def edge_port_id(self, target_node_id: UUID, source_node_id: UUID, source_port_id: UUID, fallback: UUID) -> UUID:
        key = (target_node_id, source_node_id, source_port_id)
        return _one(self.edge_ports.get(key, ()), description=f'edge input {key}') or fallback

    def dataset_port_id(
        self,
        node_id: UUID,
        dataset_id: str,
        dataset_index: int,
        column: str,
        fallback: UUID,
        *,
        allow_group_fallback: bool = False,
        fail_on_ambiguous: bool = False,
    ) -> UUID:
        exact = self.dataset_ports.get((node_id, dataset_id, dataset_index, column), ())
        resolved = _one(exact, description=f'dataset input {node_id}:{dataset_id}[{dataset_index}]:{column}')
        if resolved is not None:
            return resolved
        if not allow_group_fallback:
            return fallback

        group = self.dataset_groups.get((node_id, dataset_id, dataset_index), set())
        if fallback in group:
            return fallback
        if len(group) == 1:
            return next(iter(group))
        if group and fail_on_ambiguous:
            _one(group, description=f'anonymous dataset inputs {node_id}:{dataset_id}[{dataset_index}]')
        return fallback


def build_yaml_port_reference_catalog(instance: InstanceConfig) -> YamlPortReferenceCatalog:
    """Read the persisted mirror before YAML sync rewrites it."""
    from nodes.models import DatasetPort, NodeConfig, NodeEdge

    output_ports: dict[tuple[object, ...], set[UUID]] = {}
    input_roles: dict[tuple[object, ...], set[UUID]] = {}
    edge_ports: dict[tuple[object, ...], set[UUID]] = {}
    dataset_ports: dict[tuple[object, ...], set[UUID]] = {}
    dataset_groups: dict[tuple[object, ...], set[UUID]] = defaultdict(set)

    nodes = list(NodeConfig.objects.filter(instance=instance).active().with_spec())
    specs = {node.uuid: node.spec for node in nodes if node.spec is not None}
    _index_spec_ports(nodes, output_ports=output_ports, input_roles=input_roles)

    edges = NodeEdge.objects.filter(instance=instance).select_related('from_node', 'to_node')
    for edge in edges:
        _add(edge_ports, (edge.to_node.uuid, edge.from_node.uuid, edge.from_port), edge.to_port)

    ports = DatasetPort.objects.filter(instance=instance).select_related('node', 'dataset', 'metric')
    for binding in ports:
        dataset_id = binding.dataset.identifier or str(binding.dataset.uuid)
        group_key = (binding.node.uuid, dataset_id, binding.dataset_index)
        dataset_groups[group_key].add(binding.port_id)

        spec = specs.get(binding.node.uuid)
        target_port = spec.input_port_by_id.get(binding.port_id) if spec is not None else None
        selectors = {
            binding.spec.column,
            binding.metric.name,
            str(target_port.identifier) if target_port is not None and target_port.identifier is not None else None,
        }
        for selector in selectors - {None}:
            _add(dataset_ports, (*group_key, selector), binding.port_id)

    return YamlPortReferenceCatalog(
        output_ports=output_ports,
        input_roles=input_roles,
        edge_ports=edge_ports,
        dataset_ports=dataset_ports,
        dataset_groups=dict(dataset_groups),
    )


def _index_spec_ports(
    nodes: Iterable[NodeConfig],
    *,
    output_ports: dict[tuple[object, ...], set[UUID]],
    input_roles: dict[tuple[object, ...], set[UUID]],
) -> None:
    for node in nodes:
        spec = node.spec
        if spec is None:
            continue
        for port in spec.output_ports:
            for selector in (port.identifier, port.column_id):
                if selector is not None:
                    _add(output_ports, (node.uuid, str(selector)), port.id)
        for port in spec.input_ports:
            if port.identifier is not None:
                _add(input_roles, (node.uuid, str(port.identifier)), port.id)
