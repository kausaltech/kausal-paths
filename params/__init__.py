from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .base import Parameter, ParameterWithUnit
from .param import (
    BoolParameter,
    NumberParameter,
    PercentageParameter,
    ReferenceParameter,
    StringParameter,
    ValidationError,
)
from .registry import register_parameter_type

type AnyParameterType = NumberParameter | BoolParameter | StringParameter


# Static discriminated union of parameter types for use in defs schemas.
# Action-specific parameter types (ShiftParameter, ReduceParameter) are
# registered dynamically but added here for serialization support.
AnyParameter = Annotated[
    AnyParameterType,
    Field(discriminator='type'),
]

__all__ = [
    'AnyParameter',
    'BoolParameter',
    'NumberParameter',
    'Parameter',
    'ParameterWithUnit',
    'PercentageParameter',
    'ReferenceParameter',
    'StringParameter',
    'ValidationError',
    'register_parameter_type',
]
