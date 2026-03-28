from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    import rich.repr

    from params import Parameter

    from .node import Node


type NodeErrorCode = Literal['hash_error', 'computation_error', 'no_default_unit_error']


@dataclass
class NodeEvent:
    node: Node
    event: str | None = None
    target_node: Node | None = None
    cause: BaseException | None = None

    def __str__(self):
        s = self.node.id
        if self.event:
            s += ': %s' % self.event
        if self.target_node:
            s += ' (target: %s)' % self.target_node.id
        return s

    def __rich_repr__(self) -> rich.repr.Result:
        yield 'node', self.node
        yield 'event', self.event, None
        yield 'target_node', self.target_node, None


class NodeError(Exception):
    error_code: ClassVar[NodeErrorCode | None] = None
    event_chain: list[NodeEvent] = []

    def __init__(self, node: Node, msg: str, *args, event: str | None = None, target_node: Node | None = None, **kwargs):
        self.event_chain.append(NodeEvent(node, event, target_node))
        msg_with_id = 'Node %s: %s' % (node.id, msg)
        super().__init__(msg_with_id, *args, **kwargs)

    def add_node_event(self, node: Node, event: str | None = None, target_node: Node | None = None):
        self.event_chain.append(NodeEvent(node, event, target_node))

    def get_event_chain(self) -> str:
        return ' -> '.join([str(event) for event in reversed(self.event_chain)])

    def __rich_repr__(self) -> rich.repr.Result:
        yield 'code', self.error_code, None
        if self.__cause__:
            yield 'cause', str(self.__cause__)
        yield 'event_chain', self.event_chain

    # def __str__(self):
    #     msg = super().__str__()
    #     if len(self.event_chain) > 1:
    #         msg += '\nEvent chain: %s' % self.get_event_chain()
    #     return msg


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
