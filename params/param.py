from __future__ import annotations
import typing
from dataclasses import dataclass
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
    description: str = None
    # Set if this parameter is bound to a specific node
    node: Node = None
    is_customized: bool = False

    def reset(self):
        self.value = self.default_value

    def set(self, value: typing.Any):
        self.value = self.clean(value)

    def get(self):
        return self.value


@dataclass
class NumberParameter(Parameter):
    value: float = None
    default_value: float = None
    min_value: float = None
    max_value: float = None

    def clean(self, value: float):
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
class BoolParameter(Parameter):
    value: bool = None
    default_value: bool = None

    def clean(self, value: bool):
        try:
            value = bool(value)
        except ValueError:
            raise ValidationError(self)
        return value


@dataclass
class StringParameter(Parameter):
    value: str = None
    default_value: str = None

    def clean(self, value: str):
        if not isinstance(value, str):
            raise ValidationError(self)
        return value
