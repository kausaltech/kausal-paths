import logging
from dataclasses import dataclass, InitVar
from typing import Any, Dict, List

import sentry_sdk

from .context import Context
from common.i18n import TranslatedString
from nodes.node import Node
from params import Parameter


logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    id: str
    name: TranslatedString
    context: Context

    default: bool = False
    # Dict of params and their values in the scenario
    params: Dict[str, Any] = None
    all_actions_enabled: bool = False
    # Nodes that will be notified of this scenario's creation
    nodes: InitVar[List[Node]] = []

    def __post_init__(self, nodes):
        self.params = {}
        for node in nodes:
            node.on_scenario_created(self)

    def activate(self):
        for param_id, val in self.params.items():
            param = self.context.get_param(param_id)
            param.set(val)
            param.is_customized = False

    def is_parameter_customized(self, param: Parameter):
        # Parameters can be customized only in the CustomScenario
        return False


@dataclass
class CustomScenario(Scenario):
    base_scenario: Scenario = None
    session = None

    def set_session(self, session):
        self.session = session

    def is_parameter_customized(self, param: Parameter):
        params = self.session.params
        if param.id in params:
            return True
        return False

    def activate(self):
        assert self.base_scenario.context == self.context
        self.base_scenario.activate()
        params = self.session.get('params', {})
        for param_id, val in list(params.items()):
            param = self.context.params.get(param_id)
            is_valid = True
            if param is None:
                # The parameter might be stale (e.g. set with an older version of the backend)
                logger.error('parameter %s not found in context' % param_id)
                is_valid = False
            else:
                try:
                    val = param.clean(val)
                except Exception as e:
                    logger.error('parameter %s has invalid value: %s' % (param_id, val))
                    is_valid = False
                    sentry_sdk.capture_exception(e)

            if not is_valid:
                del params[param_id]
                self.session.modified = True
                continue

            param.set(val)
            param.is_customized = True
