from typing import TYPE_CHECKING, Any

from factory import Factory, Sequence, SubFactory

from kausal_common.i18n.pydantic import TranslatedString

from nodes.tests.factories import ContextFactory
from nodes.units import unit_registry
from params.param import BoolParameter, NumberParameter, Parameter, StringParameter

if TYPE_CHECKING:
    from nodes.context import Context


class ParameterFactory[P: Parameter[Any] = Parameter[Any]](Factory[P]):
    class Meta:
        model = Parameter

    local_id = Sequence(lambda i: f'param{i}')
    label = TranslatedString('Parameter label')
    description = TranslatedString('Parameter description')
    is_customizable = True
    is_visible = True
    context: SubFactory[Parameter[Any], Context] = SubFactory(ContextFactory)


class NumberParameterFactory(ParameterFactory[NumberParameter]):
    class Meta:
        model = NumberParameter

    min_value = 1.23
    max_value = 12345.67
    step = 0.01
    unit = unit_registry('kt').units


class BoolParameterFactory(ParameterFactory[BoolParameter]):
    class Meta:
        model = BoolParameter


class StringParameterFactory(ParameterFactory[StringParameter]):
    class Meta:
        model = StringParameter
