from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from collections.abc import Iterable

    from common.i18n import TranslatedString
    from params import Parameter
    from params.storage import SettingStorage

    from .context import Context

logger = logging.getLogger(__name__)


class Scenario:
    id: str
    name: TranslatedString
    context: Context

    default: bool = False
    all_actions_enabled: bool = False
    param_values: dict[str, Any]

    def __init__(
        self,
        context: Context,
        id: str,
        name: TranslatedString,
        default: bool = False,
        all_actions_enabled: bool = False,
    ):
        self.id = id
        self.context = context
        self.name = name
        self.default = default
        self.all_actions_enabled = all_actions_enabled
        self.param_values = {}

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

    def add_parameter(self, param: Parameter, value: Any):  # noqa: ANN401
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


class CustomScenario(Scenario):
    base_scenario: Scenario
    storage: SettingStorage

    def __init__(self, *args, base_scenario: Scenario, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_scenario = base_scenario

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
                    val = param.clean(val)
                except Exception:
                    self.context.log.error('parameter %s has invalid value: %s', param_id, val)
                    is_valid = False
            if not is_valid:
                self.storage.reset_param(param_id)
                continue
            assert param is not None
            yield param, val

    def activate(self):
        self.base_scenario.activate()
        for param, val in self.get_param_values():
            if not param.is_value_equal(val):
                param.set(val)
                param.is_customized = True
            else:
                self.context.log.warning('parameter %s was set to default value (%s)', param.global_id, val)
                self.storage.reset_param(param.global_id)
