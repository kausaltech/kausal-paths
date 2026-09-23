from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from .base import BinaryOperationSpec, InputOperationSpec, OperationInput, VariadicOperationSpec


class IdentityOperationSpec(InputOperationSpec):
    """Pass through a single input unchanged."""

    kind: Literal['identity'] = 'identity'


class AddOperationSpec(VariadicOperationSpec):
    """Add one or more operands to the primary input."""

    kind: Literal['add'] = 'add'


class SubtractOperationSpec(VariadicOperationSpec):
    """Subtract one or more operands from the primary input."""

    kind: Literal['subtract'] = 'subtract'


class MultiplyOperationSpec(VariadicOperationSpec):
    """Multiply the primary input by one or more operands."""

    kind: Literal['multiply'] = 'multiply'


class DivideOperationSpec(BinaryOperationSpec):
    """Divide the primary input by a secondary operand."""

    kind: Literal['divide'] = 'divide'


class ClipOperationSpec(InputOperationSpec):
    """Clamp values to a lower and/or upper bound."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    kind: Literal['clip'] = 'clip'
    min_value: OperationInput | None = Field(default=None, alias='min')
    max_value: OperationInput | None = Field(default=None, alias='max')

    @model_validator(mode='after')
    def _validate_bounds(self) -> ClipOperationSpec:
        if self.min_value is None and self.max_value is None:
            raise ValueError('Clip operations must define at least one bound')
        return self
