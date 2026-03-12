import sys
from typing import TYPE_CHECKING

import orjson

from kausal_common.logging.errors import print_exception

from nodes.exceptions import NodeError

if TYPE_CHECKING:
    from nodes.units import Unit


def hash_unit(unit: Unit) -> bytes:
    """Generate a byte hash of a unit in a performant manner."""
    h = getattr(unit, '_paths_hash', None)
    if h is not None:
        return h
    h = orjson.dumps(dict(unit._units))
    unit._paths_hash = h
    return h


def install_node_error_handler():
    from rich import print

    old_excepthook = sys.excepthook

    def node_error_handler(exc_type, exc_value, exc_traceback) -> None:
        if isinstance(exc_value, NodeError):
            if cause := exc_value.__cause__:
                print_exception(cause)
            print(exc_value)
            exit(1)
        else:
            old_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = node_error_handler
