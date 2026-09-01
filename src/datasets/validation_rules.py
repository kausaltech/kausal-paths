"""
Schema of the dataset metric validation rules, with their GraphQL projections.

The Pydantic rule union is the single authority for the rule shape: the
storage rows (``kausal_common.datasets.models.DatasetMetricValidationRule``)
hold its JSON dump, snapshots parse it strictly, and the GraphQL types are
derived from the models via ``kausal_common.strawberry.pydantic`` — declared
inline as nested ``ObjectType``/``InputType`` stubs so the projection lives
next to the fields it projects.

This module stays free of Django, polars and the evaluation machinery: the
rule union is referenced from the instance-graph catalog
(``nodes.defs.graph.DatasetMetricMeta``) and from dataset snapshots. The
evaluator lives in ``datasets.validation``.

Enforcement semantics:

* ``block_edit`` — a mutation that introduces *new* violations of the rule
  is rejected (baseline-diff: pre-existing violations never block an
  unrelated edit, mirroring ``validate_binding_change``).
* ``block_publish`` — edits are allowed and the violations surface in the
  UI, but the instance publication gate refuses while any remain.

Both tiers block publication; the tiers differ only at edit time.
"""

from enum import Enum as PyEnum
from typing import Annotated, Any, Literal, assert_never
from uuid import UUID

import strawberry as sb
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator
from strawberry import auto

from kausal_common.strawberry.pydantic import (
    StrawberryPydanticType,
    graphql_types,
    pydantic_type,
    register_type_conversion,
)

type Enforcement = Literal['block_edit', 'block_publish']


@sb.enum(
    name='DatasetRuleEnforcement',
    description=(
        'BLOCK_EDIT rules reject mutations that introduce new violations; '
        'BLOCK_PUBLISH rules allow edits but block publication while violations remain.'
    ),
)
class DatasetRuleEnforcement(PyEnum):
    BLOCK_EDIT = 'block_edit'
    BLOCK_PUBLISH = 'block_publish'


# The GraphQL projections carry ``enforcement`` as the enum; the conversions
# swap it in for both the alias and the narrowed literal ``NoGapsRule`` uses.
register_type_conversion(Enforcement, DatasetRuleEnforcement)
register_type_conversion(Literal['block_publish'], DatasetRuleEnforcement)


class _RuleBase(BaseModel):
    model_config = ConfigDict(extra='forbid')

    enforcement: Enforcement

    @field_validator('enforcement', mode='before')
    @classmethod
    def _enforcement_from_enum(cls, value: object) -> object:
        # Accept the GraphQL enum (or any value-compatible enum), so the
        # derived input types round-trip through ``to_pydantic``.
        if isinstance(value, PyEnum):
            return value.value
        return value


@pydantic_type(_RuleBase, is_interface=True, name='ValidationRule')
class ValidationRuleGQLInterface(StrawberryPydanticType[_RuleBase]):
    enforcement: auto


@graphql_types
class ValueRangeRule(_RuleBase):
    """Every non-null value of the metric must lie within the given bounds."""

    kind: Literal['value_range'] = 'value_range'
    min: float | None = None
    max: float | None = None
    exclusive_min: bool = False
    exclusive_max: bool = False

    @model_validator(mode='after')
    def _require_a_bound(self) -> ValueRangeRule:
        if self.min is None and self.max is None:
            raise ValueError('value_range rule requires at least one of min/max')
        if self.exclusive_min and self.min is None:
            raise ValueError('value_range exclusive_min requires min')
        if self.exclusive_max and self.max is None:
            raise ValueError('value_range exclusive_max requires max')
        if (
            self.min is not None
            and self.max is not None
            and (self.min > self.max or (self.min == self.max and (self.exclusive_min or self.exclusive_max)))
        ):
            raise ValueError('value_range bounds define an empty range')
        return self

    class ObjectType(ValidationRuleGQLInterface):
        min: auto
        max: auto
        exclusive_min: auto
        exclusive_max: auto

    class InputType(StrawberryPydanticType['ValueRangeRule']):
        enforcement: auto
        min: auto
        max: auto
        exclusive_min: auto
        exclusive_max: auto


@graphql_types
class DimensionSumRule(_RuleBase):
    """
    Values must sum to ``target`` over the named dimension's categories.

    The sum is taken within each (year x remaining dimensions) group over
    the categories that have a non-null value; ``tolerance`` is absolute.
    """

    kind: Literal['dimension_sum'] = 'dimension_sum'
    dimension: str
    target: float = 1.0
    tolerance: float = 1e-6

    class ObjectType(ValidationRuleGQLInterface):
        dimension: auto
        target: auto
        tolerance: auto

    class InputType(StrawberryPydanticType['DimensionSumRule']):
        enforcement: auto
        dimension: auto
        target: auto
        tolerance: auto


@graphql_types
class NoGapsRule(_RuleBase):
    """
    Observed-union completeness for the metric.

    Every dimension-category combination that has a value in *any* year
    must have a non-null value in *every* year where the metric has data.
    An explicit 0 passes; a missing row or an explicit null is a gap.

    Whole-dataset invariants cannot gate individual edits (a dataset being
    entered incrementally is always incomplete mid-entry), so this rule is
    ``block_publish`` only.
    """

    kind: Literal['no_gaps'] = 'no_gaps'
    enforcement: Literal['block_publish'] = 'block_publish'

    class ObjectType(ValidationRuleGQLInterface):
        enforcement: auto

    class InputType(StrawberryPydanticType['NoGapsRule']):
        enforcement: auto


@graphql_types
class RequiredCombinationGroup(BaseModel):
    """A named requirement satisfied by any one of its category combinations."""

    model_config = ConfigDict(extra='forbid')

    id: str
    combinations: list[UUID] = Field(min_length=1)

    class ObjectType(StrawberryPydanticType['RequiredCombinationGroup']):
        id: auto
        combinations: auto

    class InputType(StrawberryPydanticType['RequiredCombinationGroup']):
        id: auto
        combinations: auto


@graphql_types
class RequiredCombinationsRule(_RuleBase):
    """Every named group must have a value in at least one allowed category tuple."""

    kind: Literal['required_combinations'] = 'required_combinations'
    groups: list[RequiredCombinationGroup] = Field(min_length=1)

    class ObjectType(ValidationRuleGQLInterface):
        groups: auto

    class InputType(StrawberryPydanticType['RequiredCombinationsRule']):
        enforcement: auto
        groups: auto


@graphql_types
class AllowedCombinationsRule(_RuleBase):
    """Populated category tuples must belong to a closed schema domain."""

    kind: Literal['allowed_combinations'] = 'allowed_combinations'

    class ObjectType(ValidationRuleGQLInterface):
        enforcement: auto

    class InputType(StrawberryPydanticType['AllowedCombinationsRule']):
        enforcement: auto


type ValidationRule = Annotated[
    ValueRangeRule | DimensionSumRule | NoGapsRule | RequiredCombinationsRule | AllowedCombinationsRule,
    Field(discriminator='kind'),
]

validation_rule_adapter: TypeAdapter[ValidationRule] = TypeAdapter(ValidationRule)
rule_list_adapter: TypeAdapter[list[ValidationRule]] = TypeAdapter(list[ValidationRule])


def rule_to_gql(rule: ValidationRule) -> ValidationRuleGQLInterface:
    """Project one rule onto its concrete GraphQL object type."""
    match rule:
        case ValueRangeRule():
            return ValueRangeRule.ObjectType.from_pydantic(rule)
        case DimensionSumRule():
            return DimensionSumRule.ObjectType.from_pydantic(rule)
        case NoGapsRule():
            return NoGapsRule.ObjectType.from_pydantic(rule)
        case RequiredCombinationsRule():
            return RequiredCombinationsRule.ObjectType.from_pydantic(rule)
        case AllowedCombinationsRule():
            return AllowedCombinationsRule.ObjectType.from_pydantic(rule)
        case _:
            assert_never(rule)


@sb.input(
    one_of=True,
    name='ValidationRuleSpecInput',
    description='One validation rule; exactly one variant field must be set.',
)
class ValidationRuleSpecInput:
    value_range: ValueRangeRule.InputType | None = sb.UNSET  # type: ignore[valid-type]
    dimension_sum: DimensionSumRule.InputType | None = sb.UNSET  # type: ignore[valid-type]
    no_gaps: NoGapsRule.InputType | None = sb.UNSET  # type: ignore[valid-type]
    required_combinations: RequiredCombinationsRule.InputType | None = sb.UNSET  # type: ignore[valid-type]
    allowed_combinations: AllowedCombinationsRule.InputType | None = sb.UNSET  # type: ignore[valid-type]

    def to_rule(self) -> ValidationRule:
        """Convert to the Pydantic rule; raises ``pydantic.ValidationError`` on invalid field values."""
        variants: list[Any] = [
            variant
            for variant in (
                self.value_range,
                self.dimension_sum,
                self.no_gaps,
                self.required_combinations,
                self.allowed_combinations,
            )
            if variant is not sb.UNSET and variant is not None
        ]
        if len(variants) != 1:
            # The GraphQL layer enforces @oneOf on real requests; guard
            # against programmatic construction anyway.
            msg = f'Exactly one rule variant must be set, got {len(variants)}'
            raise ValueError(msg)
        return variants[0].to_pydantic()
