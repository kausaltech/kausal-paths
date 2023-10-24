from factory import Factory, Sequence, SubFactory

from common.i18n import TranslatedString
from nodes.context import unit_registry
from nodes.tests.factories import ContextFactory
from params.param import BoolParameter, NumberParameter, Parameter, StringParameter


class ParameterFactory(Factory[Parameter]):
    class Meta:
        model = Parameter

    local_id = Sequence(lambda i: f'param{i}')
    label = TranslatedString("Parameter label")
    description = TranslatedString("Parameter description")
    is_customizable = True
    is_visible = True
    context = SubFactory(ContextFactory)


class NumberParameterFactory(ParameterFactory):
    class Meta:
        model = NumberParameter

    min_value = 1.23
    max_value = 12345.67
    step = 0.01
    unit = unit_registry('kt').units


class BoolParameterFactory(ParameterFactory):
    class Meta:
        model = BoolParameter


class StringParameterFactory(ParameterFactory):
    class Meta:
        model = StringParameter
