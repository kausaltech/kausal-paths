from __future__ import annotations

import hashlib
import json
import typing
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from loguru import logger

if typing.TYPE_CHECKING:
    from django.contrib.sessions.backends.base import SessionBase

    from nodes.instance import Instance


class SettingStorage:
    def reset(self):
        """Reset all customized settings to their default values."""
        raise NotImplementedError()

    def set_param(self, id: str, val: Any):
        """Set a parameter value."""
        raise NotImplementedError()

    def reset_param(self, id: str):
        """Reset a parameter to its default value."""
        raise NotImplementedError()

    def set_option(self, id: str, val: Any):
        """Set a global option."""
        raise NotImplementedError()

    def reset_option(self, id: str):
        """Reset a global option to its default value."""
        raise NotImplementedError()

    def get_customized_param_values(self) -> dict[str, Any]:
        """Return ids of all currently customized parameters with their values."""
        raise NotImplementedError()

    def set_active_scenario(self, id: str | None):
        """Mark the supplied scenario as active."""
        raise NotImplementedError()

    def get_active_scenario(self) -> str | None:
        """Return the scenario currently marked as active."""
        raise NotImplementedError()


class InstanceData(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    active_scenario: str | None = None


class SessionStorage(SettingStorage):
    session: SessionBase
    instance: Instance
    data: InstanceData

    def __init__(self, instance: Instance, session: SessionBase):
        self.session = session
        self.instance = instance
        self.data = self.get_instance_settings(session, instance.id)
        self.log = logger.bind(session=session.session_key)

    def reset(self):
        self.session[self.instance.id] = {}

    @classmethod
    def get_instance_settings(cls, session: SessionBase, instance_id: str) -> InstanceData:
        settings = session.get(instance_id, None)
        log = logger.bind(session=session.session_key)
        if settings is None:
            return InstanceData()
        try:
            return InstanceData.model_validate(settings)
        except ValidationError as e:
            log.error('invalid settings: %s' % e)
            data = InstanceData()
            session[instance_id] = data.model_dump()
            return data

    def mark_modified(self):
        self.session[self.instance.id] = self.data.model_dump()

    def set_instance_option(self, id: str, val: Any):
        if self.data.options.get(id) == val:
            return
        self.data.options[id] = val
        self.mark_modified()

    def set_param(self, id: str, val: Any):
        if self.data.params.get(id) == val:
            return
        self.data.params[id] = val
        self.mark_modified()

    def reset_param(self, id: str):
        if id not in self.data.params:
            return
        del self.data.params[id]
        self.mark_modified()

    def set_option(self, id: str, val: Any):
        if id in self.data.options and self.data.options[id] == val:
            return
        self.data.options[id] = val
        self.mark_modified()

    def has_option(self, id: str) -> bool:
        return id in self.data.options

    def get_option(self, id: str) -> Any:
        return self.data.options.get(id)

    def reset_option(self, id: str):
        if id not in self.data.options:
            return
        del self.data.options[id]
        self.mark_modified()

    def get_customized_param_values(self) -> dict[str, Any]:
        return self.data.params.copy()

    def set_active_scenario(self, id: str | None):
        if self.data.active_scenario == id:
            return
        self.data.active_scenario = id
        self.mark_modified()

    def get_active_scenario(self) -> str | None:
        return self.data.active_scenario

    @classmethod
    def get_cache_key(cls, session: SessionBase, instance_id: str) -> str | None:
        data = cls.get_instance_settings(session, instance_id)
        if data.active_scenario and data.active_scenario != 'default':
            return None

        opts = data.options
        if not opts:
            return ''

        s = hashlib.md5(
            json.dumps(opts, sort_keys=True, ensure_ascii=True).encode('ascii'),
            usedforsecurity=False,
        ).hexdigest()
        return s
