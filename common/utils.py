from typing import TYPE_CHECKING

import orjson

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
