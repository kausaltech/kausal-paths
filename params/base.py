from __future__ import annotations
import typing
from dataclasses import dataclass
if typing.TYPE_CHECKING:
    from nodes import Node


class ValidationError(Exception):
    def __init__(self, param, *args, **kwargs):
        if not args and not kwargs:
            super().__init__("[Param %s]: Parameter validation failed" % param.id)
        else:
            super().__init__(*args, **kwargs)


@dataclass
class Parameter:
    id: str
    description: str = None
    # Set if this parameter is bound to a specific node
    node: Node = None

    def reset(self):
        self.value = self.default_value

    def set(self, value: typing.Any):
        self.value = self.clean(value)

    def get(self, session_params=None):
        if session_params is None:
            return self.value
        return session_params.get(self.id, self.value)

    def is_customized(self, session_params=None):
        if session_params is None:
            return False
        return self.id in session_params


@dataclass
class NumberParameter(Parameter):
    value: float = None
    default_value: float = None

    def clean(self, value: float):
        try:
            value = float(value)
        except ValueError:
            raise ValidationError(self)
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
