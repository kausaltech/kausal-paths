from typing import TYPE_CHECKING, Any

from factory import Factory, Sequence, SubFactory

from kausal_common.i18n.pydantic import TranslatedString

from nodes.tests.factories import ContextFactory
from nodes.units import unit_registry
from params.base import Parameter
from params.param import BoolParameter, NumberParameter, StringParameter

if TYPE_CHECKING:
    from nodes.context import Context


class ParameterFactory[P: Parameter[Any]](Factory[P]):
    class Meta:
        model = Parameter
        abstract = True

    local_id = Sequence(lambda i: f'param{i}')
    label = TranslatedString('Parameter label', default_language='en')
    description = TranslatedString('Parameter description', default_language='en')
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
