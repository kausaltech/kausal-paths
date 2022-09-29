import orjson
import pint


def hash_unit(unit: pint.Unit) -> bytes:
    """Generate a byte hash of a unit in a performant manner."""
    h = getattr(unit, '_paths_hash', None)
    if h is not None:
        return h
    h = orjson.dumps(dict(unit._units))
    setattr(unit, '_paths_hash', h)
    return h

