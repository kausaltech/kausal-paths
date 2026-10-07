"""
The shared problem surface of the model editor.

``InstanceProblem`` covers publication blockers. ``DatasetFinding`` shares
the cell locator between validation violations and advisory plausibility
findings; a plausibility finding is never an ``InstanceProblem``.
"""

from enum import Enum
from typing import TYPE_CHECKING, Self
from uuid import UUID

import strawberry as sb

from paths.graphql_types import UnitType

from datasets.models import PlausibilityAggregation, PlausibilityReference
from nodes.units import Unit, unit_registry

if TYPE_CHECKING:
    from collections.abc import Iterable

    from datasets.coordinates import DatasetCoordinate
    from datasets.plausibility import PlausibilityAttribution, PlausibilityFinding
    from datasets.validation import RuleViolation
    from nodes.data_entry import EntryProblem
    from nodes.value_validation import ValueValidationViolation


@sb.enum(name='ProblemSeverity', description='How a problem or advisory finding is presented.')
class ProblemSeverity(Enum):
    ERROR = 'error'
    WARNING = 'warning'


@sb.enum(name='ProblemEnforcement')
class ProblemEnforcement(Enum):
    BLOCK_EDIT = 'block_edit'
    BLOCK_PUBLISH = 'block_publish'
    BLOCK_SUBMISSION = 'block_submission'


@sb.interface(name='InstanceProblem', description='One problem that blocks publication or submission of the instance draft.')
class InstanceProblemInterface:
    code: str = sb.field(description='Machine-readable problem kind.')
    message: str = sb.field(description='Untranslated human-readable fallback.')
    severity: ProblemSeverity
    enforcement: ProblemEnforcement


@sb.type(name='DataEntryDefinitionProblem', description='An invalid data-entry layout; drafts remain editable.')
class DataEntryDefinitionProblemType(InstanceProblemInterface):
    enforcement: ProblemEnforcement
    section_id: UUID | None
    node_id: UUID | None
    port_id: UUID | None
    dataset_id: UUID | None

    @classmethod
    def from_problem(cls, problem: EntryProblem) -> Self:
        return cls(
            code=problem.code,
            message=problem.message,
            severity=ProblemSeverity.ERROR,
            enforcement=ProblemEnforcement.BLOCK_PUBLISH,
            section_id=problem.section_id,
            node_id=problem.node_id,
            port_id=problem.port_id,
            dataset_id=problem.dataset_id,
        )


@sb.type(name='DataEntryDefinitionProblems')
class DataEntryDefinitionProblemsType:
    problems: list[DataEntryDefinitionProblemType]

    @classmethod
    def from_problems(cls, problems: list[EntryProblem]) -> Self:
        return cls(problems=[DataEntryDefinitionProblemType.from_problem(problem) for problem in problems])


@sb.type(name='NodeValueCoordinate')
class NodeValueCoordinateType:
    dimension: str
    category: str


@sb.type(
    name='NodeValueValidationViolation', description='A delivered node or dataset value violates its consuming port contract.'
)
class NodeValueValidationViolationType(InstanceProblemInterface):
    node_uuid: UUID
    port_uuid: UUID
    binding_uuid: UUID | None
    years: list[int]
    categories: list[NodeValueCoordinateType]
    enforcement: ProblemEnforcement

    @classmethod
    def from_violation(cls, violation: ValueValidationViolation) -> Self:
        return cls(
            code=violation.code,
            message=violation.message,
            severity=ProblemSeverity.ERROR if violation.enforcement == 'block_publish' else ProblemSeverity.WARNING,
            enforcement=ProblemEnforcement(violation.enforcement),
            node_uuid=violation.node_uuid,
            port_uuid=violation.port_uuid,
            binding_uuid=violation.binding_uuid,
            years=violation.years,
            categories=[
                NodeValueCoordinateType(dimension=dimension, category=category)
                for dimension, category in violation.categories.items()
            ],
        )


@sb.type(name='DatasetDimensionCoordinate', description='One dimension and category in a dataset finding.')
class DatasetDimensionCoordinateType:
    dimension_uuid: UUID
    category_uuid: UUID
    dimension: str = sb.field(deprecation_reason='Use dimensionUuid.')
    category: str = sb.field(deprecation_reason='Use categoryUuid.')

    @classmethod
    def from_coordinate(cls, coordinate: DatasetCoordinate) -> Self:
        return cls(
            dimension_uuid=coordinate.dimension_uuid,
            category_uuid=coordinate.category_uuid,
            dimension=coordinate.dimension,
            category=coordinate.category,
        )


@sb.interface(name='DatasetFinding', description='The dataset cell or cells located by a finding.')
class DatasetFindingInterface:
    rule_uuid: UUID
    metric_uuid: UUID
    metric: str = sb.field(description='Metric column identifier.')
    dataset_uuid: UUID | None
    dataset_identifier: str | None
    years: list[int] = sb.field(description='Affected years; empty for dataset-wide findings.')
    coordinates: list[DatasetDimensionCoordinateType]
    combination_ids: list[UUID] = sb.field(description='Schema category combinations involved in the finding.')


@sb.type(
    name='DatasetValidationViolation',
    description='One located violation of a dataset metric validation rule.',
)
class DatasetValidationViolationType(InstanceProblemInterface, DatasetFindingInterface):
    enforcement: ProblemEnforcement
    requirement_group: str | None = sb.field(description='Named required-combination group, when applicable.')

    @classmethod
    def from_violation(cls, violation: RuleViolation) -> Self:
        return cls(
            code=violation.kind,
            message=violation.message,
            severity=ProblemSeverity.ERROR if violation.enforcement == 'block_edit' else ProblemSeverity.WARNING,
            enforcement=ProblemEnforcement(violation.enforcement),
            rule_uuid=violation.rule_uuid,
            metric_uuid=violation.metric_uuid,
            metric=violation.metric,
            dataset_uuid=violation.dataset_uuid,
            dataset_identifier=violation.dataset_identifier,
            years=list(violation.years),
            coordinates=[DatasetDimensionCoordinateType.from_coordinate(coordinate) for coordinate in violation.coordinates],
            combination_ids=list(violation.combination_ids),
            requirement_group=violation.requirement_group,
        )


@sb.type(
    name='PlausibilityAttribution',
    description=(
        'The one cell that explains a sum finding: with its value in comparedYear, the sum alone would be back '
        'in range. Absent when no cell, or more than one, does so on its own.'
    ),
)
class PlausibilityAttributionType:
    year: int = sb.field(
        description=(
            "The year whose value is suspect. Usually the finding's year; for a spike it is the year before, "
            'and the finding is the return to normal.'
        )
    )
    compared_year: int
    coordinates: list[DatasetDimensionCoordinateType]
    combination_ids: list[UUID]
    message: str = sb.field(description='Untranslated human-readable fallback.')

    @classmethod
    def from_attribution(cls, attribution: PlausibilityAttribution) -> Self:
        return cls(
            year=attribution.year,
            compared_year=attribution.compared_year,
            coordinates=[DatasetDimensionCoordinateType.from_coordinate(coordinate) for coordinate in attribution.coordinates],
            combination_ids=list(attribution.combination_ids),
            message=attribution.message,
        )


@sb.type(name='DatasetPlausibilityFinding', description='A non-blocking observation outside a reference range.')
class DatasetPlausibilityFindingType(DatasetFindingInterface):
    code: str
    message: str
    severity: ProblemSeverity
    aggregation: PlausibilityAggregation
    reference: PlausibilityReference
    selection: list[DatasetDimensionCoordinateType] = sb.field(
        description='For a sum: every selected dimension and category. Empty for a single cell.'
    )
    component_count: int = sb.field(description='How many cells the checked value covers.')
    complete: bool = sb.field(
        description='False for a sum with empty cells; such a sum is only reported for exceeding the upper bound.'
    )
    observed: float = sb.field(description="The cell value or sum, in the metric's unit.")
    normalized: float = sb.field(description='The quantity compared with the bounds, in `unit`.')
    denominator_value: int | None
    reference_year: int | None = sb.field(description='For a previous-year reference: the year compared with.')
    reference_value: float | None = sb.field(description='For a previous-year reference: the value in that year.')
    lower: float
    upper: float
    unit: Unit = sb.field(graphql_type=UnitType)
    source_identifier: str
    source_url: str
    source_revision: str
    rule_revision: int
    is_example: bool
    attribution: PlausibilityAttributionType | None = sb.field(
        description=(
            'For a sum: the one cell that explains it, when there is one. For a cell: set only when the finding is '
            'the return from a spike, and names the spike year. Group findings by it.'
        )
    )

    @classmethod
    def from_finding(cls, finding: PlausibilityFinding) -> Self:
        return cls(
            code='outside_reference_range',
            message=finding.message,
            severity=ProblemSeverity.WARNING,
            rule_uuid=finding.rule_uuid,
            metric_uuid=finding.metric_uuid,
            metric=finding.metric,
            dataset_uuid=finding.dataset_uuid,
            dataset_identifier=finding.dataset_identifier,
            years=finding.years,
            coordinates=[DatasetDimensionCoordinateType.from_coordinate(coordinate) for coordinate in finding.coordinates],
            combination_ids=finding.combination_ids,
            aggregation=PlausibilityAggregation(finding.aggregation),
            reference=PlausibilityReference(finding.reference),
            selection=[DatasetDimensionCoordinateType.from_coordinate(coordinate) for coordinate in finding.selection],
            component_count=finding.component_count,
            complete=finding.complete,
            observed=float(finding.observed),
            normalized=float(finding.normalized),
            denominator_value=finding.denominator_value,
            reference_year=finding.reference_year,
            reference_value=finding.reference_value,
            lower=float(finding.lower),
            upper=float(finding.upper),
            unit=unit_registry.parse_units(finding.unit),
            source_identifier=finding.source_identifier,
            source_url=finding.source_url,
            source_revision=finding.source_revision,
            rule_revision=finding.rule_revision,
            is_example=finding.is_example,
            attribution=PlausibilityAttributionType.from_attribution(finding.attribution) if finding.attribution else None,
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


@sb.type(
    name='NodeValueValidationViolations', description='Publication was refused because delivered values violate input contracts.'
)
class NodeValueValidationViolationsType:
    violations: list[NodeValueValidationViolationType]

    @classmethod
    def from_violations(cls, violations: list[ValueValidationViolation]) -> Self:
        return cls(violations=[NodeValueValidationViolationType.from_violation(v) for v in violations])
