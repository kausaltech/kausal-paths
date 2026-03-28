from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any

import strawberry as sb
from pydantic import ConfigDict, Field, PrivateAttr

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from paths.identifiers import ParameterGlobalId, ScenarioIdentifier

from params.storage import SettingStorage

from .context import Context

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from params import Parameter


@sb.enum
class ScenarioKind(Enum):
    DEFAULT = 'default'
    BASELINE = 'baseline'
    CUSTOM = 'custom'
    PROGRESS_TRACKING = 'progress_tracking'


class Scenario(I18nBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: ScenarioIdentifier
    name: I18nString
    description: I18nString | None = None
    kind: ScenarioKind | None = None
    all_actions_enabled: bool = False
    is_selectable: bool = True
    param_values: dict[ParameterGlobalId, Any] = Field(default_factory=dict)
    actual_historical_years: list[int] | None = None

    _context: Context | None = PrivateAttr(default=None)

    @property
    def default(self) -> bool:
        return self.kind == ScenarioKind.DEFAULT

    @property
    def context(self) -> Context:
        if self._context is None:
            raise RuntimeError('Context is not set')
        return self._context

    def get_param_values(self) -> Iterable[tuple[Parameter, Any]]:
        for param_id, val in self.param_values.items():
            param = self.context.get_parameter(param_id)
            yield param, val

    @contextmanager
    def override(self, set_active: bool = False) -> Generator[None]:
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
        instance = self.context.instance if self._context is not None else None
        return "Scenario(id=%s, name='%s', instance=%s)" % (
            self.id,
            str(self.name),
            instance.id if instance is not None else None,
        )


class CustomScenario(Scenario):
    base_scenario: Scenario
    kind: ScenarioKind | None = ScenarioKind.CUSTOM
    _storage: SettingStorage = PrivateAttr(init=False)

    def set_storage(self, storage: SettingStorage):
        self._storage = storage

    def reset(self):
        self._storage.reset()
        self.base_scenario.activate()

    def get_param_values(self) -> Iterable[tuple[Parameter, Any]]:
        params = list(self._storage.get_customized_param_values().items())
        for param_id, val in params:
            param = self.context.get_parameter(param_id, required=False)
            is_valid = True
            cleaned_val = None
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
                self._storage.reset_param(param_id)
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
                self._storage.reset_param(param.global_id)
