import json
import logging
import sentry_sdk

from common.i18n import TranslatedString


logger = logging.getLogger(__name__)


class Scenario:
    def __init__(self, id: str, name: TranslatedString, default=False, all_actions_enabled=False, notified_nodes=None):
        if notified_nodes is None:
            notified_nodes = []

        self.id = id
        self.name = name
        self.default = default
        self.all_actions_enabled = all_actions_enabled

        for node in notified_nodes:
            node.on_scenario_created(self)

    def activate(self, context, session=None):
        """Resets each parameter in the context to its setting for this scenario if it has one."""
        for param in context.get_all_parameters():
            param.reset_to_scenario_setting(self)

    def export(self, name, context, session):
        customized_parameters = [p for p in context.get_all_parameters() if p.is_customized]
        return ScenarioExport(self, name, customized_parameters, context, session)


class SessionSettingsScenario(Scenario):
    """
    Activating a SessionSettingsScenario first activates a base scenario and then sets parameters to values given in the
    user's session.
    """
    def __init__(
        self, id: str, name: TranslatedString, base_scenario: Scenario, settings_getter=None, notified_nodes=None
    ):
        """
        `settings_getter` is a function mapping a session to the parameter settings. If `None`, uses the dict key
        `'settings'`.
        """
        if settings_getter is None:
            def settings_getter(session):
                return session.get('settings', {})

        super().__init__(id, name, default=False, all_actions_enabled=False, notified_nodes=notified_nodes)
        self.base_scenario = base_scenario
        self.settings_getter = settings_getter

    def activate(self, context, session=None):
        if session is None:
            raise ValueError("SessionSettingsScenario cannot be activated without a session")

        self.base_scenario.activate(context, session)
        settings = self.settings_getter(session)
        for param_id, val in list(settings.items()):
            param = context.get_parameter(param_id, required=False)
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
                del settings[param_id]
                session.modified = True
                continue

            param.set(val)


class ScenarioExport:
    def __init__(self, scenario, name, parameters, context, session):
        """
        Export the values of all the given parameters, whether they are customized or not in the scenario.

        Filter them before if you only care about customized parameters.
        """
        self.name = name
        with context.temp_activate_scenario(scenario, session):
            self.settings = {p.global_id: p.value for p in parameters}

    @property
    def json_filename(self):
        return f'{self.name}.json'

    def to_json(self):
        result = {
            'name': self.name,
            'settings': self.settings,
        }
        return json.dumps(result)
