from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from .node import Node


type NodeErrorCode = Literal['hash_error', 'computation_error', 'no_default_unit_error']

class NodeError(Exception):
    error_code: ClassVar[NodeErrorCode | None] = None
    node_paths: list[str] = []

    def __init__(self, node: Node, msg: str, *args, **kwargs):
        self.node_paths = []
        msg = '[%s] %s' % (str(node), msg)
        super().__init__(msg, *args, **kwargs)

    def add_node(self, node: Node):
        self.node_paths.append(node.id)

    def get_dependency_path(self):
        return ' -> '.join(reversed(self.node_paths))

    def __str__(self):
        msg = super().__str__()
        if self.node_paths:
            msg += '\nNode dependency path: %s' % self.get_dependency_path()
        return msg

class NodeHashingError(NodeError):
    error_code = 'hash_error'


class NodeComputationError(NodeError):
    error_code = 'computation_error'


class NodeMissingDefaultUnitError(NodeError):
    error_code = 'no_default_unit_error'

    def __init__(self, node: Node):
        msg = "Node does not have a default unit"
        super().__init__(node, msg)
