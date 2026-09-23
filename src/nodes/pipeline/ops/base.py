from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from paths.identifiers import (
    DatasetIdentifier,
    Identifier,
    NodeIdentifier,
    NodePortIdentifier,
    ParameterGlobalId,
    ParameterLocalId,
)

from nodes.units import Unit


class PortInputRef(BaseModel):
    """Reference to a node input port."""

    port: NodePortIdentifier


class IntermediateInputRef(BaseModel):
    """Reference to a named intermediate result from an earlier pipeline step."""

    ref: Identifier


class DatasetInputRef(BaseModel):
    """Reference to an input dataset attached to the node."""

    dataset: DatasetIdentifier


class ParameterInputRef(BaseModel):
    """
    Reference to a parameter value from the active scenario.

    The identifier may refer to either a node-local parameter or a global
    parameter. Resolution rules belong to the loader/runtime, not the schema.
    """

    parameter: ParameterLocalId | ParameterGlobalId


class HookBaseInputRef(BaseModel):
    """
    The un-hooked value of a node an action acts on (see ``nodes.hooks``).

    Only meaningful in an action's pipeline: it reads the target's own output,
    before any action's contribution is added.
    """

    hook_base: NodeIdentifier


class ScalarValue(BaseModel):
    """
    Constant scalar input.

    Numeric literals must be explicit about dimensionality: either provide a
    unit or mark the value as dimensionless.
    """

    value: float
    unit: Unit | None = None
    dimensionless: bool = False

    @model_validator(mode='after')
    def _validate_dimensionality(self) -> ScalarValue:
        if self.dimensionless and self.unit is not None:
            raise ValueError('Dimensionless scalar values cannot define a unit')
        if not self.dimensionless and self.unit is None:
            raise ValueError('Scalar values must define a unit or set dimensionless=true')
        return self


OperationInput = PortInputRef | IntermediateInputRef | DatasetInputRef | ParameterInputRef | HookBaseInputRef | ScalarValue


class ComparisonOperator(StrEnum):
    EQ = 'eq'
    NE = 'ne'
    GT = 'gt'
    GTE = 'gte'
    LT = 'lt'
    LTE = 'lte'


class TruthyCondition(BaseModel):
    """Execute when the referenced input evaluates to true."""

    kind: Literal['truthy'] = 'truthy'
    input: OperationInput


class ComparisonCondition(BaseModel):
    """Execute when two inputs satisfy a simple comparison."""

    kind: Literal['compare'] = 'compare'
    left: OperationInput
    op: ComparisonOperator = ComparisonOperator.EQ
    right: OperationInput


OperationCondition = Annotated[TruthyCondition | ComparisonCondition, Field(discriminator='kind')]


ANONYMOUS_STEP_PREFIX = '_step_'


def step_key(operation: OperationSpec, index: int) -> str:
    """Return the key later steps use to refer to the result of ``operation``, the ``index``-th step."""
    return operation.result_id or f'{ANONYMOUS_STEP_PREFIX}{index}'


class OperationSpec(BaseModel):
    """
    Base schema for a pipeline operation.

    Concrete operations should narrow ``kind`` to a literal and add their own
    operation-specific fields.
    """

    kind: str
    result_id: Identifier | None = None
    """
    The step's name. A step without one is anonymous: later steps refer to it
    as ``_step_<index>``, and a formula writes it inline.
    """
    description: str | None = None
    """The modeller's explanation of the step; a formula writes it as a comment."""
    only_if: OperationCondition | None = None
    skip_if: OperationCondition | None = None


class InputOperationSpec(OperationSpec):
    """Base class for operations with one primary input."""

    input: OperationInput


class BinaryOperationSpec(InputOperationSpec):
    """Base class for operations with one primary input and one secondary operand."""

    other: OperationInput


class VariadicOperationSpec(InputOperationSpec):
    """Base class for operations with one primary input and additional operands."""

    values: list[OperationInput] = Field(default_factory=list, min_length=1)


class MultiInputOperationSpec(OperationSpec):
    """Base class for operations that consume a list of peer inputs."""

    inputs: list[OperationInput] = Field(default_factory=list, min_length=1)
