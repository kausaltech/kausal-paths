"""Expand legacy shared operands at stored-spec upgrade boundaries."""

from collections import defaultdict
from typing import TYPE_CHECKING
from uuid import uuid5

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from nodes.defs.node_defs import NodeSpec
    from nodes.instance_serialization import InputBindingSnapshot


def split_legacy_edge_roles(
    spec: NodeSpec, bindings: Sequence[InputBindingSnapshot], roles: tuple[str, ...]
) -> tuple[NodeSpec, list[InputBindingSnapshot]]:
    """Give distinct operands explicit ports while retaining the first port and binding identities."""
    by_port: dict[UUID, list[InputBindingSnapshot]] = defaultdict(list)
    for binding in bindings:
        by_port[binding.port_id].append(binding)
    ports = []
    replacements: dict[UUID, list[InputBindingSnapshot]] = {}
    for port in spec.input_ports:
        candidates = by_port[port.id]
        if port.role is not None or port.identifier in roles or len(candidates) != 1:
            ports.append(port)
            continue
        binding = candidates[0]
        selected = [role for role in roles if role in binding.tags] if binding.node_source is not None else []
        if len(selected) < 2:
            ports.append(port)
            continue
        expanded = []
        for index, role in enumerate(selected):
            port_id = port.id if index == 0 else uuid5(port.id, f'legacy-edge-role:{role}')
            binding_id = binding.uuid if index == 0 else uuid5(binding.uuid or port.id, f'legacy-edge-role:{role}')
            identifier = port.identifier if index == 0 or port.identifier is None else f'{port.identifier}_{role}'
            ports.append(port.model_copy(update={'id': port_id, 'role': role, 'identifier': identifier}))
            expanded.append(binding.model_copy(update={'uuid': binding_id, 'port_id': port_id}))
        replacements[port.id] = expanded
    if not replacements:
        return spec, list(bindings)
    upgraded = [item for binding in bindings for item in replacements.get(binding.port_id, [binding])]
    return spec.model_copy(update={'input_ports': ports}), upgraded
