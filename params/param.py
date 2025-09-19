from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import InitVar, asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, cast

from pydantic import BaseModel

from common.polars import from_pandas
from nodes.units import Quantity, Unit

if TYPE_CHECKING:
    import pandas as pd

    from common.i18n import I18nString
    from nodes.context import Context
    from nodes.dimensions import Dimension
    from nodes.node import Node, NodeMetric
    from nodes.scenario import Scenario


class ValidationError(Exception):
    def __init__(self, param: Parameter, msg: str = ''):
        if msg is not None:
            msg_str = ': %s' % msg
        else:
            msg_str = ''
        super().__init__('[Param %s]: Parameter validation failed%s' % (param.local_id, msg_str))


V = TypeVar('V')


@dataclass
class Parameter(Generic[V]):
    local_id: str  # not globally unique but locally, relative to the parameter's node (if it has one)
    context: Context | None = field(repr=False, hash=False, default=None)
    'The context to which this parameter is bound'

    label: I18nString | None = None
    description: I18nString | None = None

    node: Node | None = None
    'Set if this parameter is bound to a specific node'

    subscription_nodes: list[Node] = field(default_factory=list)
    'Nodes that should be notified when the parameter changes value'

    subscription_params: list[Parameter] = field(default_factory=list)
    'Parametres that should be notified when the parameter changes value'

    is_customized: bool = False
    is_customizable: bool = True
    is_visible: bool | None = None

    def __post_init__(self):
        assert '.' not in self.local_id
        self._hash = None
        self._follows_scenario: Scenario | None = None
        if self.is_visible is None:
            if self.is_customizable:
                self.is_visible = True
            else:
                self.is_visible = False

    def copy(self) -> Self:
        fields = asdict(self)
        return type(self)(**fields)

    def notify_change(self) -> None:
        self._hash = None
        if self.node:
            self.node.notify_parameter_change(self)
        for node in self.subscription_nodes:
            node.notify_parameter_change(self)
        for param in self.subscription_params:
            param.notify_change()

    @contextmanager
    def override(self, value: V):
        prev_val = self.value
        self.set(value)
        yield
        self.set(prev_val)

    def set(self, value: V, notify: bool = True):
        prev_val = getattr(self, 'value', None)
        self.value = self.clean(value)
        if notify and not self.is_value_equal(prev_val):
            self.notify_change()

    def reset_to_scenario_setting(self, scenario: Scenario, value: V):
        self.set(value)
        self._follows_scenario = scenario
        self.is_customized = False

    def get(self) -> V:
        return self.value

    def serialize_value(self) -> Any:
        if isinstance(self.value, BaseModel):
            return self.value.model_dump(mode='json')
        return self.value

    def is_value_equal(self, value: Any) -> bool:
        return self.value == value

    def calculate_hash(self) -> str:
        h = getattr(self, '_hash', None)
        if h is not None:
            return h
        if isinstance(self.value, str):
            v = self.value.encode('unicode_escape').decode('ascii')
        elif isinstance(self.value, bool | int | float):
            v = str(self.value)
        else:
            v = json.dumps({'value': self.serialize_value()}, ensure_ascii=True)

        h = '%s:%s' % (self.global_id, v)
        self._hash = h
        return h

    def clean(self, value: Any) -> Any:
        raise NotImplementedError('Implement in subclass')

    @property
    def global_id(self):
        if self.node is None:
            return self.local_id
        return f'{self.node.id}.{self.local_id}'

    def set_node(self, node: Node):
        if self.node is not None:
            msg = f'Node for parameter {self.global_id} already set'
            raise Exception(msg)
        self.node = node

    def has_unit(self) -> bool:
        if isinstance(self, ParameterWithUnit) and self.unit is not None:
            return True
        return False

    def get_unit(self) -> Unit:
        if not self.has_unit():
            msg = f'Parameter {self.global_id} does not have units'
            raise Exception(msg)
        return self.unit  # type: ignore


@dataclass
class ReferenceParameter(Parameter):
    """
    Parameter that is a reference to another parameter.

    This parameter cannot be changed.
    """

    target: Parameter | None = None
    _target: Parameter = field(init=False)
    is_customizable: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.target is not None
        self._target = self.target
        self.target.subscription_params.append(self)

    @property
    def unit(self) -> Unit | None:
        assert isinstance(self._target, ParameterWithUnit)
        return self._target.unit

    @property
    def value(self) -> Any:
        return self._target.value

    def has_unit(self) -> bool:
        return self._target.has_unit()

    def get_unit(self) -> Unit:
        return self._target.get_unit()

    def clean(self, value: Any) -> Any:
        raise NotImplementedError()


@dataclass
class ParameterWithUnit:
    unit: Unit | None = None
    unit_str: InitVar[str | None] = None

    def _init_unit(self, unit_str: str | None = None) -> None:
        if hasattr(self, 'unit_str') and unit_str is None:
            unit_str = self.unit_str  # type: ignore

        if unit_str is not None:
            from nodes.context import unit_registry

            self.unit = unit_registry.parse_units(unit_str)

        if self.unit is not None and not isinstance(self.unit, Unit):
            raise Exception('str given for unit for parameter %s' % self.local_id)  # type: ignore


@dataclass
class NumberParameter(ParameterWithUnit, Parameter[float]):
    value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None

    def __post_init__(self, unit_str: str | None = None):
        self._init_unit(unit_str)
        if self.min_value is not None:
            self.min_value = float(self.min_value)
        if self.max_value is not None:
            self.max_value = float(self.max_value)
        if self.step is not None:
            self.step = float(self.step)
        super().__post_init__()

    def clean(self, value: float | Quantity) -> float:
        # Store unit first if available
        if isinstance(value, Quantity):
            if self.unit is not None:
                assert isinstance(self.unit, Quantity)
                assert self.unit.is_compatible_with(value.units)
            value = value.m

        # Avoid converting, e.g., bool to float
        if not isinstance(value, int | float | str):
            raise ValidationError(self)
        try:
            value = float(value)
        except ValueError:
            raise ValidationError(self) from None

        if self.min_value is not None:
            self.min_value = float(self.min_value)
            if value < self.min_value:
                raise ValidationError(self, 'Below min_value')
        if self.max_value is not None:
            self.max_value = float(self.max_value)
            if value > self.max_value:
                raise ValidationError(self, 'Above max_value')

        return value

    def set(self, value: Quantity | float) -> None:  # type: ignore[override]
        if isinstance(value, Quantity):
            unit = value.units
            value = value.m
        else:
            unit = None
        super().set(value)  # type: ignore
        if unit is not None:
            self.unit = cast('Unit', unit)


@dataclass
class DatasetParameter(Parameter):
    """Multi-dimensional time-series."""

    dimensions: list[Dimension] = field(default_factory=list)
    metrics: list[NodeMetric] = field(default_factory=list)
    value: pd.DataFrame | None = None

    def __post_init__(self):
        if not self.metrics:
            raise Exception('Must have at least one metric')
        super().__post_init__()

    def is_value_equal(self, value: pd.DataFrame) -> bool:
        assert self.value is not None
        return self.value.equals(value)

    def serialize_value(self) -> dict[str, Any]:
        from nodes.datasets import JSONDataset

        assert self.value is not None
        return JSONDataset.serialize_df(from_pandas(self.value))

    def clean(self, value: dict) -> Any:
        if not isinstance(value, dict):
            raise ValidationError(self, 'Must get a dict as value')

        return super().clean(value)


@dataclass
class PercentageParameter(NumberParameter):
    unit_str = '%'


@dataclass
class BoolParameter(Parameter[bool]):
    value: bool | None = None

    def clean(self, value: bool):
        # Avoid converting non-bool to bool
        if not isinstance(value, bool):
            raise ValidationError(self)
        return value


@dataclass
class StringParameter(Parameter[str]):
    value: str | None = None

    def clean(self, value: str):
        if not isinstance(value, str):
            raise ValidationError(self)
        return value


param_type_registry: set[type[Parameter]] = set()


def register_parameter_type(cls: type[Parameter]):
    if cls in param_type_registry:
        raise Exception('Parameter class %s already registered', str(cls))
    param_type_registry.add(cls)
