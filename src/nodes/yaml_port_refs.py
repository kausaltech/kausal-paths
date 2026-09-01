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
    """Read the persisted bindings before YAML sync rewrites them."""
    from nodes.models import NodeConfig, NodeInputPortBinding

    output_ports: dict[tuple[object, ...], set[UUID]] = {}
    input_roles: dict[tuple[object, ...], set[UUID]] = {}
    edge_ports: dict[tuple[object, ...], set[UUID]] = {}
    dataset_ports: dict[tuple[object, ...], set[UUID]] = {}
    dataset_groups: dict[tuple[object, ...], set[UUID]] = defaultdict(set)

    nodes = list(NodeConfig.objects.filter(instance=instance).active().with_spec())
    specs = {node.uuid: node.spec for node in nodes if node.spec is not None}
    _index_spec_ports(nodes, output_ports=output_ports, input_roles=input_roles)

    from nodes.instance_serialization import InputBindingSnapshot, group_unified_dataset_bindings

    rows = (
        NodeInputPortBinding.objects
        .filter(instance=instance)
        .select_related('node', 'source_node', 'dataset', 'metric')
        .order_by('node_id', 'port_id', 'position')
    )
    dataset_row_snapshots: list[tuple[InputBindingSnapshot, int]] = []
    for binding in rows:
        if binding.source_node is not None:
            _add(edge_ports, (binding.node.uuid, binding.source_node.uuid, binding.source_port_id), binding.port_id)
            continue
        snap = InputBindingSnapshot.from_model(binding)
        dataset_row_snapshots.append((snap, snap.position))

    for node_uuid, node_groups in group_unified_dataset_bindings(dataset_row_snapshots, specs).items():
        for ordinal, (group_spec, dataset_id, group_rows) in enumerate(node_groups):
            group_key = (node_uuid, dataset_id, ordinal)
            node_spec = specs.get(node_uuid)
            for row in group_rows:
                dataset_groups[group_key].add(row.port_id)
                row_source = row.dataset_source
                assert row_source is not None
                target_port = node_spec.input_port_by_id.get(row.port_id) if node_spec is not None else None
                selectors = {
                    group_spec.column,
                    row_source.metric,
                    str(target_port.identifier) if target_port is not None and target_port.identifier is not None else None,
                }
                for selector in selectors - {None}:
                    _add(dataset_ports, (*group_key, selector), row.port_id)

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
