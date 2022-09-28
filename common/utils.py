import json
import pint


def hash_unit(unit: pint.Unit) -> bytes:
    """Generate a byte hash of a unit in a performant manner."""
    return hash(unit).to_bytes(8, 'little', signed=True)
