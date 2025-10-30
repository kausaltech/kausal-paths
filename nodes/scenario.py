from __future__ import annotations

from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from common.i18n import I18nString
    from params import Parameter
    from params.storage import SettingStorage

    from .context import Context


class ScenarioKind(Enum):
    DEFAULT = 'default'
    BASELINE = 'baseline'
    CUSTOM = 'custom'
    PROGRESS_TRACKING = 'progress_tracking'
    COPY_TO_CUSTOM = 'copy_to_custom'


@dataclass
class Scenario:
    context: Context
    id: str
    name: I18nString

    _: KW_ONLY

    kind: ScenarioKind | None = None
    all_actions_enabled: bool = False
    is_selectable: bool = True
    param_values: dict[str, Any] = field(default_factory=dict)
    actual_historical_years: list[int] | None = None

    @property
    def default(self) -> bool:
        return self.kind == ScenarioKind.DEFAULT

    def get_param_values(self) -> Iterable[tuple[Parameter, Any]]:
        for param_id, val in self.param_values.items():
            param = self.context.get_parameter(param_id)
            yield param, val

    @contextmanager
    def override(self, set_active: bool = False) -> Generator[None, None, None]:
        old_vals: dict[str, Any] = {}

        old_scenario = self.context.active_scenario

        for param, _ in self.get_param_values():
            old_vals[param.global_id] = param.value

        self.activate()
        if set_active:
            self.context.active_scenario = self

        yield

        if set_active:
            self.context.active_scenario = old_scenario

        for param_id, val in old_vals.items():
            param = self.context.get_parameter(param_id)
            param.set(val)

    def activate(self):
        """Reset each parameter in the context to its setting for this scenario if it has one."""

        for param, val in self.get_param_values():
            param.reset_to_scenario_setting(self, val)

    def add_parameter(self, param: Parameter, value: Any):
        assert param.global_id not in self.param_values
        self.param_values[param.global_id] = value

    def has_parameter(self, param: Parameter):
        return param.global_id in self.param_values

    def get_parameter_value(self, param: Parameter):
        return self.param_values[param.global_id]

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        instance = self.context.instance if self.context is not None else None
        return "Scenario(id=%s, name='%s', instance=%s)" % (
            self.id,
            str(self.name),
            instance.id if instance is not None else None,
        )


@dataclass
class CustomScenario(Scenario):
    _: KW_ONLY

    base_scenario: Scenario
    kind: ScenarioKind | None = ScenarioKind.CUSTOM
    storage: SettingStorage = field(init=False)

    def set_storage(self, storage: SettingStorage):
        self.storage = storage

    def reset(self):
        self.storage.reset()
        self.base_scenario.activate()

    def get_param_values(self) -> Iterable[tuple[Parameter, Any]]:
        params = list(self.storage.get_customized_param_values().items())
        for param_id, val in params:
            param = self.context.get_parameter(param_id, required=False)
            is_valid = True
            if param is None:
                # The parameter might be stale (e.g. set with an older version of the backend)
                self.context.log.error('parameter %s not found in context' % param_id)
                is_valid = False
            else:
                try:
                    cleaned_val = param.clean(val)
                except Exception:
                    self.context.log.error('parameter %s has invalid value: %s', param_id, val)
                    is_valid = False
            if not is_valid:
                self.storage.reset_param(param_id)
                continue
            assert param is not None
            yield param, cleaned_val

    def activate(self):
        self.base_scenario.activate()
        for param, val in self.get_param_values():
            if not param.is_value_equal(val):
                param.set(val)
                param.is_customized = True
            else:
                self.context.log.warning('parameter %s was set to default value (%s)', param.global_id, val)
                self.storage.reset_param(param.global_id)

    def copy_from_scenario(self, source_scenario: Scenario):
        """Copy all parameter values from source scenario to this custom scenario's storage."""

        if source_scenario.kind == ScenarioKind.CUSTOM:
            print('Nothing to copy-----------------------------')
            return

        print('Reset storage for ++++++++++++++', self.id)
        self.storage.reset()

        for param, val in source_scenario.get_param_values():
            print(param.global_id, val)
            self.storage.set_param(param.global_id, val)

        self.activate()
