from __future__ import annotations
import typing
from typing import Any, Optional

if typing.TYPE_CHECKING:
    from django.contrib.sessions.backends.base import SessionBase

    from nodes.instance import Instance


class SettingStorage:
    def reset(self):
        """Resets all customized settings to their default values."""
        raise NotImplementedError()

    def set_param(self, id: str, val: Any):
        """Sets a parameter value."""
        raise NotImplementedError()

    def reset_param(self, id: str):
        """Resets a parameter to its default value."""
        raise NotImplementedError()

    def get_customized_param_values(self) -> dict[str, Any]:
        """Returns ids of all currently customized parameters with their values."""
        raise NotImplementedError()

    def set_active_scenario(self, id: str):
        """Mark the supplied scenario as active."""
        raise NotImplementedError()

    def get_active_scenario(self) -> Optional[str]:
        """Returns the scenario currently marked as active."""
        raise NotImplementedError()


class SessionStorage(SettingStorage):
    session: SessionBase
    instance: Instance

    def __init__(self, instance: Instance, session: SessionBase):
        self.session = session
        self.instance = instance

    def reset(self):
        self.session[self.instance.id] = {}

    @property
    def _instance_settings(self):
        return self.session.setdefault(self.instance.id, {})

    @property
    def _instance_params(self) -> dict[str, Any]:
        return self._instance_settings.setdefault('params', {})

    def set_param(self, id: str, val: Any):
        self._instance_params[id] = val
        self.session.modified = True

    def reset_param(self, id: str):
        if id in self._instance_params:
            del self._instance_params[id]
            self.session.modified = True

    def get_customized_param_values(self) -> dict[str, Any]:
        return self._instance_params.copy()

    def set_active_scenario(self, id: Optional[str]):
        self._instance_settings['active_scenario'] = id
        self.session.modified = True

    def get_active_scenario(self) -> Optional[str]:
        return self._instance_settings.get('active_scenario')
