from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PrivateAttr, model_validator

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from paths.identifiers import DimensionIdentifier, Identifier, ParameterGlobalId

from nodes.exceptions import ParameterError
from nodes.units import Quantity

from .base import Parameter, ParameterWithUnit, parameter

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.scenario import Scenario
    from nodes.units import Unit


class ValidationError(Exception):
    def __init__(self, param: Parameter[Any], msg: str = ''):
        if msg is not None:  # pyright: ignore[reportUnnecessaryComparison]
            msg_str = ': %s' % msg
        else:
            msg_str = ''
        super().__init__('[Param %s]: Parameter validation failed%s' % (param.local_id, msg_str))


@parameter
class ReferenceParameter[ValueT = Any, SetValueT = ValueT](Parameter[ValueT, SetValueT]):
    """
    Parameter that is a reference to another parameter.

    This parameter cannot be changed.
    """

    type: Literal['reference'] = 'reference'
    target_id: ParameterGlobalId
    is_customizable: bool = False

    _target: Parameter[ValueT, SetValueT] | None = PrivateAttr(default=None)

    def model_post_init(self, /, __context: Any) -> None:
        super().model_post_init(__context)
        assert self.is_customizable is False

    @property
    def target(self) -> Parameter[ValueT, SetValueT]:
        assert self._target is not None
        return self._target

    @property
    def unit(self) -> Unit | None:
        assert isinstance(self.target, ParameterWithUnit)
        return self.target.unit

    @property
    def value(self) -> Any:
        return self.target.value

    def has_unit(self) -> bool:
        return self.target.has_unit()

    def get_unit(self) -> Unit:
        return self.target.get_unit()

    def reset_to_scenario_setting(self, scenario: Scenario, value: SetValueT):
        return

    def calculate_hash(self) -> str:
        return self.target.calculate_hash()

    def set_context(self, context: Context):
        super().set_context(context)
        target = context.global_parameters.get(self.target_id)
        if target is None:
            raise Exception(f'ReferenceParameter {self.global_id} target parameter {self.target_id} not found')
        self._target = target
        target._subscription_params.append(self)

    def clean(self, value: Any) -> Any:
        raise NotImplementedError()


@parameter
class NumberParameter(ParameterWithUnit[float, float | Quantity]):
    type: Literal['number'] = 'number'
    value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None

    def model_post_init(self, /, __context: Any) -> None:
        if self.min_value is not None:
            self.min_value = float(self.min_value)
        if self.max_value is not None:
            self.max_value = float(self.max_value)
        if self.step is not None:
            self.step = float(self.step)
        super().model_post_init(__context)

    def clean(self, value: float | Quantity) -> float:
        if isinstance(value, Quantity):
            if self.unit is not None:
                assert isinstance(self.unit, Quantity)
                assert self.unit.is_compatible_with(value.units)
            unit_value = value.to(self.unit)
            value = unit_value.m

        # Avoid converting, e.g., bool to float
        if not isinstance(value, int | float | str):
            raise ValidationError(self, 'Invalid value type: %s' % type(value))
        try:
            value = float(value)
        except ValueError:
            raise ValidationError(self) from None

        if self.min_value is not None:
            self.min_value = float(self.min_value)
            if value < self.min_value:
                raise ValidationError(self, 'Below min_value')
        if self.max_value is not None:
            self.max_value = float(self.max_value)
            if value > self.max_value:
                raise ValidationError(self, 'Above max_value')

        return value


@parameter
class BoolParameter(Parameter[bool]):
    type: Literal['bool'] = 'bool'
    value: bool | None = None

    def clean(self, value: bool):
        # Avoid converting non-bool to bool
        if not isinstance(value, bool):
            raise ValidationError(self)
        return value


@parameter
class StringParameter(Parameter[str]):
    type: Literal['string'] = 'string'
    value: str | None = None

    def clean(self, value: str):
        if not isinstance(value, str):
            raise ValidationError(self)
        return value


class ParameterChoice(I18nBaseModel):
    """One of the values a `ChoiceParameter` can take."""

    id: Identifier
    label: I18nString | None = None


@parameter
class ChoiceParameter(Parameter[str]):
    """A parameter whose value is one of a fixed set of choices, identified by id."""

    type: Literal['choice'] = 'choice'
    value: str | None = None
    choices: list[ParameterChoice] = Field(default_factory=list)

    def get_choices(self) -> list[ParameterChoice] | None:
        """Return the choices, or None while they cannot be known (before the context is set)."""
        return self.choices

    def clean(self, value: str) -> str:
        if not isinstance(value, str):
            raise ValidationError(self, 'Invalid value type: %s' % type(value))
        choices = self.get_choices()
        if choices is not None and value not in {choice.id for choice in choices}:
            raise ValidationError(self, "'%s' is not one of %s" % (value, ', '.join(choice.id for choice in choices)))
        return value


@parameter
class DimensionCategoryParameter(ChoiceParameter):
    """A choice among the categories of a dimension; the choices are the dimension's own."""

    type: Literal['dimension_category'] = 'dimension_category'  # type: ignore[assignment]
    dimension: DimensionIdentifier

    @model_validator(mode='after')
    def _no_own_choices(self) -> DimensionCategoryParameter:
        if self.choices:
            raise ValueError('A dimension category parameter takes its choices from the dimension')
        return self

    def get_choices(self) -> list[ParameterChoice] | None:
        if self.context is None:
            return None
        dimension = self.context.dimensions.get(self.dimension)
        if dimension is None:
            raise ParameterError(self, "Dimension '%s' does not exist" % self.dimension)
        return [ParameterChoice(id=cat.id, label=cat.label) for cat in dimension.categories]

    def set_context(self, context: Context):
        super().set_context(context)
        if self.value is not None:
            # Checked now: the dimension is only known once the context is.
            self.clean(self.value)
