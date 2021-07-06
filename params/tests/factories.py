from factory import Factory, Sequence, SubFactory

from common.i18n import TranslatedString
from params.param import BoolParameter, NumberParameter, Parameter


class ParameterFactory(Factory):
    class Meta:
        model = Parameter

    local_id = Sequence(lambda i: f'param{i}')
    label = TranslatedString("Parameter label")
    description = TranslatedString("Parameter description")
    is_customizable = None


class NumberParameterFactory(ParameterFactory):
    class Meta:
        model = NumberParameter

    min_value = 1.23
    max_value = 12345.67
    step = 0.01
    unit = None  # TODO: Pint unit


class BoolParameterFactory(ParameterFactory):
    class Meta:
        model = BoolParameter
