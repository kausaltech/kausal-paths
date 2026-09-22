"""Small input adapters shared by the node and binding mutation APIs."""

from typing import TYPE_CHECKING, TypeGuard

import strawberry as sb

if TYPE_CHECKING:
    from uuid import UUID

    from strawberry import Some

    from nodes.defs.port_def import InputPortDef, OutputPortDef
    from nodes.instance_graph import NodeMeta
    from nodes.models import NodeConfig


def _get_output_port(nc: NodeConfig | NodeMeta, port_id: UUID) -> OutputPortDef | None:
    assert nc.spec is not None
    for port in nc.spec.output_ports:
        if port.id == port_id:
            return port
    return None


def _get_input_port(nc: NodeConfig, port_id: UUID) -> InputPortDef | None:
    assert nc.spec is not None
    for port in nc.spec.input_ports:
        if port.id == port_id:
            return port
    return None


def is_maybe_set[T](maybe: Some[T] | None) -> TypeGuard[Some[T]]:
    return maybe is not None and maybe is not sb.UNSET
