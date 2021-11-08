from __future__ import annotations

import logging
from dataclasses import dataclass, field, InitVar
from typing import TYPE_CHECKING, List, Optional

import sentry_sdk

from common.i18n import TranslatedString
from nodes.node import Node

if TYPE_CHECKING:
    from .context import Context
    from django.contrib.sessions.backends.base import SessionBase

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    id: str
    name: TranslatedString

    default: bool = False
    all_actions_enabled: bool = False
    # Nodes that will be notified of this scenario's creation
    notified_nodes: List[Node] = field(default_factory=list)

    def __post_init__(self):
        for node in self.notified_nodes:
            node.on_scenario_created(self)

    def activate(self, context):
        """Resets each parameter in the context to its setting for this scenario if it has one."""
        for param in context.get_all_parameters():
            param.reset_to_scenario_setting(self)


@dataclass
class CustomScenario(Scenario):
    base_scenario: Optional[Scenario] = None
    session: 'Optional[SessionBase]' = None

    def set_session(self, session):
        self.session = session

    def reset(self, context: Context):
        self.session['settings'] = {}
        self.session.modified = True
        self.base_scenario.activate(context)

    def activate(self, context: Context):
        assert self.base_scenario is not None
        assert self.session is not None

        self.base_scenario.activate(context)
        settings = self.session.get('settings', {})
        for param_id, val in list(settings.items()):
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
