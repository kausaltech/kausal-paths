from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .node import Node


class NodeError(Exception):
    node_paths: list[str] = []

    def __init__(self, node: Node, msg: str, *args, **kwargs):
        self.node_paths = []
        msg = '[%s] %s' % (str(node), msg)
        super().__init__(msg, *args, **kwargs)

    def add_node(self, node: Node):
        self.node_paths.append(node.id)

    def __str__(self):
        msg = super().__str__()
        if self.node_paths:
            msg += '\nNode dependency path: %s' % ' -> '.join(reversed(self.node_paths))
        return msg

class NodeHashingError(NodeError):
    pass


class NodeComputationError(NodeError):
    pass
