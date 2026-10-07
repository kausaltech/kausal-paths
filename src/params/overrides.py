"""
Per-operation model overrides: a scenario, parameter values and a normalizer for one runtime.

`InstanceType.model(...)` takes these as arguments and gets a runtime of its own, next
to the request's default one. They are the visitor's edits without the session: the
runtime starts from the visitor's stored settings and applies the overrides the way
`setParameter` would, into an in-memory storage that is never written back. So the
custom scenario is what carries the parameter values, exactly as for a visitor who set
them by hand -- a diff plus a base, as `docs/architecture/scenarios.md` describes --
and `activeScenario` says *custom* whenever a parameter is overridden.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .param import (
    BoolParameter,
    ChoiceParameter,
    NumberParameter,
    StringParameter,
    ValidationError as ParameterValidationError,
)
from .storage import InstanceData, InstanceDataStorage

if TYPE_CHECKING:
    from nodes.context import Context

    from .base import Parameter


class InvalidModelOverrideError(ValueError):
    """An override names something the model does not have, or a value it does not accept."""


VALUE_FIELDS: dict[type[Parameter[Any, Any]], str] = {
    NumberParameter: 'numberValue',
    BoolParameter: 'boolValue',
    StringParameter: 'stringValue',
    ChoiceParameter: 'stringValue',
}


def parameter_value_from_fields(
    param: Parameter[Any, Any],
    *,
    number_value: float | None,
    bool_value: bool | None,
    string_value: str | None,
) -> Any:
    """
    Pick and clean the one value field that matches the parameter's type.

    Shared by `setParameter` and the model overrides, which take values in the same
    three typed fields.
    """
    values = {'numberValue': number_value, 'boolValue': bool_value, 'stringValue': string_value}
    attr_name = next((name for klass, name in VALUE_FIELDS.items() if isinstance(param, klass)), None)
    if attr_name is None:
        msg = f'Unsupported parameter class: {type(param).__name__}'
        raise InvalidModelOverrideError(msg)

    value = values.pop(attr_name)
    if value is None:
        raise InvalidModelOverrideError(f"You must specify '{attr_name}' for '{param.global_id}'")
    if any(other is not None for other in values.values()):
        raise InvalidModelOverrideError('Only one type of value allowed')

    try:
        return param.clean(value)
    except ParameterValidationError as e:
        raise InvalidModelOverrideError(str(e)) from e


@dataclass(frozen=True)
class ParameterOverride:
    id: str
    number_value: float | None = None
    bool_value: bool | None = None
    string_value: str | None = None


@dataclass(frozen=True)
class ModelOverrides:
    """
    What one `model(...)` field changes about its runtime. Hashable: it is part of the runtime key.

    `normalizer` applies only when `override_normalizer` is set, so that "no
    normalization" (`None`) can be asked for as well as a particular one.
    """

    scenario: str | None = None
    parameters: tuple[ParameterOverride, ...] = ()
    normalizer: str | None = None
    override_normalizer: bool = False

    def __post_init__(self) -> None:
        ids = [override.id for override in self.parameters]
        if len(ids) != len(set(ids)):
            raise InvalidModelOverrideError('A parameter may be overridden only once')

    @property
    def is_empty(self) -> bool:
        return self.scenario is None and not self.parameters and not self.override_normalizer

    def storage_for(self, context: Context, session: InstanceData) -> InstanceDataStorage:
        """
        Return the settings of a runtime that starts from `session` and applies these overrides.

        Follows the branch rule of `setParameter`: overriding a parameter while a named
        scenario is active starts a custom scenario based on that one, discarding the
        session's stored edits; overriding while the custom scenario is already active
        adds to its edits.
        """
        data = session.model_copy(deep=True)
        custom = context.custom_scenario
        if self.scenario is not None:
            if self.scenario not in context.scenarios:
                raise InvalidModelOverrideError(f"Scenario '{self.scenario}' not found")
            data.active_scenario = self.scenario
        if self.parameters:
            values = self._cleaned_parameter_values(context)
            active = data.active_scenario
            if active != custom.id:
                data.params = {}
                data.custom_base = active
            data.params.update(values)
            data.active_scenario = custom.id
        if self.override_normalizer:
            if self.normalizer is not None and self.normalizer not in context.normalizations:
                raise InvalidModelOverrideError(f"Normalization '{self.normalizer}' not found")
            data.options['normalizer'] = self.normalizer
        return InstanceDataStorage(data)

    def _cleaned_parameter_values(self, context: Context) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for override in self.parameters:
            param = context.get_parameter(override.id, required=False)
            if param is None:
                raise InvalidModelOverrideError(f'Parameter {override.id} does not exist')
            if not param.is_customizable:
                raise InvalidModelOverrideError(f'Parameter {override.id} is not customizable')
            values[param.global_id] = parameter_value_from_fields(
                param,
                number_value=override.number_value,
                bool_value=override.bool_value,
                string_value=override.string_value,
            )
        return values
