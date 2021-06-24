from factory import Factory, Sequence

from common.i18n import TranslatedString
from params.param import NumberParameter, Parameter


class ParameterFactory(Factory):
    class Meta:
        model = Parameter

    id = Sequence(lambda i: f'param{i}')
    label = TranslatedString("Parameter label")
    description = TranslatedString("Parameter description")
    node = None
    is_customized = False
    is_customizable = None


class NumberParameterFactory(ParameterFactory):
    class Meta:
        model = NumberParameter

    value = None
    min_value = None
    max_value = None
    step = None
    unit = None
