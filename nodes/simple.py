from .base import Node, _


EMISSION_UNIT = 'kg'


class SectorEmissions(Node):
    """Simple addition of subsector emissions"""

    units = {
        'Emissions': EMISSION_UNIT
    }

    def compute(self):
        pass
