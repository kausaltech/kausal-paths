from __future__ import annotations

import hashlib
from typing import Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

import orjson
from common.i18n import TranslatedString
if TYPE_CHECKING:
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
    node: Optional[Node] = None
    is_customized: bool = False
    is_customizable: Optional[bool] = None

    def set(self, value: Any):
        self.value = self.clean(value)

    def get(self) -> Any:
        return self.value

    def calculate_hash(self) -> bytes:
        s = orjson.dumps({'id': self.id, 'value': self.value})
        h = hashlib.md5(s)
        return h.digest()

    def clean(self, value: Any) -> Any:
        raise NotImplementedError('Implement in subclass')


@dataclass
class NumberParameter(Parameter):
    value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    unit: Optional[str] = None

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
    value: Optional[bool] = None

    def clean(self, value: bool):
        # Avoid converting non-bool to bool
        if not isinstance(value, bool):
            raise ValidationError(self)
        return value


@dataclass
class StringParameter(Parameter):
    value: Optional[str] = None

    def clean(self, value: str):
        if not isinstance(value, str):
            raise ValidationError(self)
        return value
