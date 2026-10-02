"""Persistence boundaries for local instance definitions."""

from typing import Any

from django.db.models.expressions import Value

from django_pydantic_field.fields import PydanticSchemaField

from nodes.defs.instance_defs import InstanceModelSpec


class InstanceSpecField(PydanticSchemaField[InstanceModelSpec | None]):
    """Never persist a runtime composition as an instance's local definition."""

    def get_prep_value(self, value: Any) -> Any:
        candidate = value.value if isinstance(value, Value) else value
        if isinstance(candidate, InstanceModelSpec) and candidate.is_composed:
            raise ValueError('Cannot store a composed spec in InstanceConfig.spec')
        return super().get_prep_value(value)
