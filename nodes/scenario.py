import logging
from typing import List, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from common.i18n import TranslatedString
from params import Parameter
if TYPE_CHECKING:
    from .context import Context


logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    id: str
    name: TranslatedString
    default: bool = False
    # Dict of params and their values in the scenario
    params: List[Tuple[Parameter, Any]] = None

    def __post_init__(self):
        self.params = []

    def activate(self, context):
        for param, val in self.params:
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

    def activate(self, context):
        self.base_scenario.activate()
        params = self.session.params
        for param_id, val in list(params.items()):
            param = context.params.get(param_id)
            if param is None:
                # The parameter might be stale (e.g. set with an older version of the backend)
                logger.warn('parameter %s not found in context' % param_id)
                del params[param_id]
                self.session.modified = True
                continue
            param.set(val)
            param.is_customized = True
