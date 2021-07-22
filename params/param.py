from __future__ import annotations

import hashlib
import orjson
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

from common.i18n import TranslatedString

if TYPE_CHECKING:
    from nodes import Node


class ValidationError(Exception):
    def __init__(self, param: Parameter, msg: str = None):
        if msg is not None:
            msg_str = ': %s' % msg
        else:
            msg_str = ''
        super().__init__("[Param %s]: Parameter validation failed%s" % (param.local_id, msg_str))


@dataclass
class Parameter:
    local_id: str  # not globally unique but locally, relative to the parameter's node (if it has one)
    label: Optional[TranslatedString] = None
    description: Optional[TranslatedString] = None
    # Set if this parameter is bound to a specific node
    node: Optional[Node] = None
    is_customized: bool = False
    is_customizable: bool = True
    # Maps a scenario ID to the value of this parameter in that scenario
    scenario_settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert '.' not in self.local_id

    def set(self, value: Any, set_customized=True):
        self.value = self.clean(value)
        if set_customized:
            self.is_customized = True

    @contextmanager
    def temp_set(self, value: Any):
        old_value = self.value
        self.set(value, set_customized=False)
        yield
        self.set(old_value, set_customized=False)

    def reset_to_scenario_setting(self, scenario):
        if scenario.id in self.scenario_settings:
            setting = self.scenario_settings[scenario.id]
            self.set(setting, set_customized=False)
            self.is_customized = False

    def get(self) -> Any:
        return self.value

    def calculate_hash(self) -> bytes:
        s = orjson.dumps({'id': self.global_id, 'value': self.value})
        h = hashlib.md5(s)
        return h.digest()

    def clean(self, value: Any) -> Any:
        raise NotImplementedError('Implement in subclass')

    def add_scenario_setting(self, scenario, value):
        """
        Add the given value as the setting for the given scenario.

        `scenario` can be an instance of `Scenario` or a string that is a scenario ID.
        """
        from nodes.scenario import Scenario
        if isinstance(scenario, Scenario):
            scenario_id = scenario.id
        else:
            scenario_id = scenario

        if scenario_id in self.scenario_settings:
            raise Exception(f"Setting for parameter {self.global_id} in scenario {scenario_id} already exists")
        self.scenario_settings[scenario_id] = value

    def get_scenario_setting(self, scenario):
        return self.scenario_settings.get(scenario.id)

    @property
    def global_id(self):
        if self.node is None:
            return self.local_id
        return f'{self.node.id}.{self.local_id}'

    def set_node(self, node):
        if self.node is not None:
            raise Exception(f"Node for parameter {self.global_id} already set")
        self.node = node


@dataclass
class NumberParameter(Parameter):
    value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    unit: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.unit is not None and isinstance(self.unit, str):
            from nodes.context import unit_registry
            self.unit = unit_registry(self.unit).units

    def clean(self, value: float):
        # Avoid converting, e.g., bool to float
        if not isinstance(value, (int, float, str)):
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
