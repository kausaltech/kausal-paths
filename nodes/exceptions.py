from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .node import Node


class NodeError(Exception):
    def __init__(self, node: Node, msg: str, *args, **kwargs):
        msg = '[%s] %s' % (str(node), msg)
        super().__init__(msg, *args, **kwargs)
