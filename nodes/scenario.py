from __future__ import annotations

import logging
from dataclasses import dataclass, field, InitVar
from typing import TYPE_CHECKING, List, Optional

import sentry_sdk

from common.i18n import TranslatedString
from nodes.node import Node

if TYPE_CHECKING:
    from .context import Context
    from params.storage import SettingStorage

logger = logging.getLogger(__name__)


class Scenario:
    id: str
    name: TranslatedString

    default: bool = False
    all_actions_enabled: bool = False
    # Nodes that will be notified of this scenario's creation
    notified_nodes: List[Node] = field(default_factory=list)

    def __init__(
        self, id: str, name: TranslatedString, default: bool = False,
        all_actions_enabled: bool = False, notified_nodes: Optional[List[Node]] = None,
    ):
        self.id = id
        self.name = name
        self.default = default
        self.all_actions_enabled = all_actions_enabled
        self.notified_nodes = notified_nodes or []
        for node in self.notified_nodes:
            node.on_scenario_created(self)

    def activate(self, context: Context):
        """Resets each parameter in the context to its setting for this scenario if it has one."""
        for param in context.get_all_parameters():
            param.reset_to_scenario_setting(self)

    def __str__(self) -> str:
        return self.id


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
            param.set(val)
            param.is_customized = True
