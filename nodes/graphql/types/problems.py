"""
The shared problem surface of the model editor.

``InstanceProblem`` is the GraphQL interface implemented by everything that
can stand between a draft and publication: structural constraint conflicts
(see ``constraints.py``) and dataset validation-rule violations (evaluated by
``datasets.validation`` and persisted on ``DatasetMaterialization``). Clients
render a generic problem list from the interface fields and use fragments
for the type-specific coordinates.
"""

from enum import Enum
from typing import TYPE_CHECKING, Self
from uuid import UUID

import strawberry as sb

from datasets.validation_rules import DatasetRuleEnforcement

if TYPE_CHECKING:
    from collections.abc import Iterable

    from datasets.validation import RuleViolation


@sb.enum(name='ProblemSeverity', description='How a problem is presented; every problem blocks publication.')
class ProblemSeverity(Enum):
    ERROR = 'error'
    WARNING = 'warning'


@sb.interface(name='InstanceProblem', description='One problem that blocks publication of the instance draft.')
class InstanceProblemInterface:
    code: str = sb.field(description='Machine-readable problem kind.')
    message: str = sb.field(description='Untranslated human-readable fallback.')
    severity: ProblemSeverity


@sb.type(name='DatasetDimensionCoordinate', description='One dimension coordinate of a violation.')
class DatasetDimensionCoordinateType:
    dimension: str = sb.field(description='Dimension column identifier in the dataset.')
    category: str = sb.field(description='Category identifier within the dimension.')


@sb.type(
    name='DatasetValidationViolation',
    description='One located violation of a dataset metric validation rule.',
)
class DatasetValidationViolationType(InstanceProblemInterface):
    enforcement: DatasetRuleEnforcement
    rule_uuid: UUID
    metric_uuid: UUID
    metric: str = sb.field(description='Metric column identifier.')
    dataset_uuid: UUID | None
    dataset_identifier: str | None
    years: list[int] = sb.field(description='Affected years; empty for dataset-wide problems.')
    coordinates: list[DatasetDimensionCoordinateType]

    @classmethod
    def from_violation(cls, violation: RuleViolation) -> Self:
        return cls(
            code=violation.kind,
            message=violation.message,
            severity=ProblemSeverity.ERROR if violation.enforcement == 'block_edit' else ProblemSeverity.WARNING,
            enforcement=DatasetRuleEnforcement(violation.enforcement),
            rule_uuid=violation.rule_uuid,
            metric_uuid=violation.metric_uuid,
            metric=violation.metric,
            dataset_uuid=violation.dataset_uuid,
            dataset_identifier=violation.dataset_identifier,
            years=list(violation.years),
            coordinates=[
                DatasetDimensionCoordinateType(dimension=dimension, category=category)
                for dimension, category in violation.categories.items()
            ],
        )


@sb.type(
    name='DatasetValidationViolations',
    description=(
        'Publication was refused because datasets bound to the instance carry '
        'these validation-rule violations. Nothing was published.'
    ),
)
class DatasetValidationViolationsType:
    violations: list[DatasetValidationViolationType]

    @classmethod
    def from_violations(cls, violations: Iterable[RuleViolation]) -> Self:
        return cls(violations=[DatasetValidationViolationType.from_violation(violation) for violation in violations])
