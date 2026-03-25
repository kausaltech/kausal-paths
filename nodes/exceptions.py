from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from params import Parameter

    from .node import Node


type NodeErrorCode = Literal['hash_error', 'computation_error', 'no_default_unit_error']


@dataclass
class NodeEvent:
    node: Node
    event: str | None = None

    def __str__(self):
        if self.event:
            return '%s: %s' % (self.node.id, self.event)
        return '%s' % self.node.id


class NodeError(Exception):
    error_code: ClassVar[NodeErrorCode | None] = None
    event_chain: list[NodeEvent] = []

    def __init__(self, node: Node, msg: str, *args, **kwargs):
        self.event_chain = [NodeEvent(node, msg)]
        msg_with_id = 'Node %s: %s' % (node.id, msg)
        super().__init__(msg_with_id, *args, **kwargs)

    def add_node_event(self, node: Node, event: str | None = None):
        self.event_chain.append(NodeEvent(node, event))

    def get_event_chain(self) -> str:
        return ' -> '.join([str(event) for event in reversed(self.event_chain)])

    def __str__(self):
        msg = super().__str__()
        if len(self.event_chain) > 1:
            msg += '\nEvent chain: %s' % self.get_event_chain()
        return msg


class NodeHashingError(NodeError):
    error_code = 'hash_error'


class NodeComputationError(NodeError):
    error_code = 'computation_error'


class NodeMissingDefaultUnitError(NodeError):
    error_code = 'no_default_unit_error'

    def __init__(self, node: Node):
        msg = 'Node does not have a default unit'
        super().__init__(node, msg)


class ParameterError(Exception):
    def __init__(self, param: Parameter, msg: str):
        super().__init__(f'[{param.global_id}: {msg}')
