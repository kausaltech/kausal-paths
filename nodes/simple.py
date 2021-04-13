from .base import Node, Metric, TimeSeriesVariable, _


EMISSION_UNIT = 'kg'


class SectorEmissions(Node):
    variables = [
        TimeSeriesVariable('emission_reductions', _('Emission reductions (in CO2e)'), EMISSION_UNIT)
    ]
    output_metrics = [
        Metric('emissions', _('Emissions'), unit=EMISSION_UNIT),
    ]
