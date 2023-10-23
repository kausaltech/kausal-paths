from __future__ import annotations

import logging
from dataclasses import dataclass, field, InitVar
from typing import TYPE_CHECKING, Any, List, Optional

import sentry_sdk

from common.i18n import TranslatedString
from nodes.node import Node

if TYPE_CHECKING:
    from .context import Context
    from params.storage import SettingStorage
    from params import Parameter

logger = logging.getLogger(__name__)


class Scenario:
    id: str
    name: TranslatedString
    context: Context

    default: bool = False
    all_actions_enabled: bool = False
    param_values: dict[str, Any]

    def __init__(
        self, context: Context, id: str, name: TranslatedString, default: bool = False,
        all_actions_enabled: bool = False,
    ):
        self.id = id
        self.context = context
        self.name = name
        self.default = default
        self.all_actions_enabled = all_actions_enabled
        self.param_values = {}

    def activate(self, context: Context):
        """Resets each parameter in the context to its setting for this scenario if it has one."""
        for param_id, val in self.param_values.items():
            param = context.get_parameter(param_id)
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
        return "Scenario(id=%s, name='%s', instance=%s)" % (self.id, str(self.name), self.context.instance.id)

class CustomScenario(Scenario):
    base_scenario: Scenario
    storage: SettingStorage

    def __init__(
        self, *args, base_scenario: Scenario, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.base_scenario = base_scenario

    def set_storage(self, storage: SettingStorage):
        self.storage = storage

    def reset(self, context: Context):
        self.storage.reset()
        self.base_scenario.activate(context)

    def activate(self, context: Context):
        self.base_scenario.activate(context)
        params = self.storage.get_customized_param_values()
        for param_id, val in list(params.items()):
            param = context.get_parameter(param_id, required=False)
            is_valid = True
            if param is None:
                # The parameter might be stale (e.g. set with an older version of the backend)
                logger.error('parameter %s not found in context', param_id)
                is_valid = False
            else:
                try:
                    val = param.clean(val)
                except Exception as e:
                    logger.error('parameter %s has invalid value: %s', param_id, val)
                    is_valid = False
                    sentry_sdk.capture_exception(e)

            if not is_valid:
                self.reset(context)
                return

            assert param is not None
            if not param.is_value_equal(val):
                param.set(val)
                param.is_customized = True
