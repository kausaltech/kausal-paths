from __future__ import annotations
import typing
from dataclasses import dataclass
from common.i18n import TranslatedString

if typing.TYPE_CHECKING:
    from nodes import Node


class ValidationError(Exception):
    def __init__(self, param: Parameter, msg: str = None):
        if msg is not None:
            msg_str = ': %s' % msg
        else:
            msg_str = ''
        super().__init__("[Param %s]: Parameter validation failed%s" % (param.id, msg_str))


@dataclass
class Parameter:
    id: str
    description: TranslatedString = None
    # Set if this parameter is bound to a specific node
    node: Node = None
    is_customized: bool = False
    is_customizable: bool = None

    def set(self, value: typing.Any):
        self.value = self.clean(value)

    def get(self):
        return self.value


@dataclass
class NumberParameter(Parameter):
    value: float = None
    min_value: float = None
    max_value: float = None
    step: float = None
    unit: str = None

    def __post_init__(self):
        if self.unit is not None and isinstance(self.unit, str):
            from nodes.context import unit_registry
            self.unit = unit_registry(self.unit).units

    def clean(self, value: float):
        # Avoid converting, e.g., bool to float
        if not isinstance(value, (int, float)):
            raise ValidationError(self)
        try:
            value = float(value)
        except ValueError:
            raise ValidationError(self)
        if self.min_value is not None and value < self.min_value:
            raise ValidationError(self, 'Below min_value')
        if self.max_value is not None and value > self.max_value:
            raise ValidationError(self, 'Above max_value')
        return value


@dataclass
class PercentageParameter(NumberParameter):
    unit = '%'


@dataclass
class BoolParameter(Parameter):
    value: bool = None

    def clean(self, value: bool):
        # Avoid converting non-bool to bool
        if not isinstance(value, bool):
            raise ValidationError(self)
        return value


@dataclass
class StringParameter(Parameter):
    value: str = None

    def clean(self, value: str):
        if not isinstance(value, str):
            raise ValidationError(self)
        return value
