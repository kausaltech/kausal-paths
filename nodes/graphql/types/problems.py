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
    dimension_label: str = sb.field(
        description="The dimension's label in the active language; falls back to the identifier when unresolvable."
    )
    category_label: str = sb.field(
        description="The category's label in the active language; falls back to the identifier when unresolvable."
    )


#: ``(dataset_uuid, dimension column, category identifier) -> (dimension label, category label)``.
type CoordinateLabels = dict[tuple[UUID | None, str, str], tuple[str, str]]


def build_coordinate_labels(violations: Iterable[RuleViolation]) -> CoordinateLabels:
    """
    Resolve localized labels for the dimension coordinates named by ``violations``.

    Violations are persisted with identifiers only: a materialization is shared
    across users and languages, and ``RuleViolation.key`` diffs on the identifier
    coordinates at edit time. Labels are therefore a presentation concern resolved
    here, in the active language, rather than baked into the stored payload.

    One query pass over the datasets involved, so a violation list costs a fixed
    number of queries rather than one per coordinate.
    """
    from kausal_common.datasets.models import Dataset, DatasetSchemaDimension, DimensionScope

    dataset_uuids = {violation.dataset_uuid for violation in violations if violation.dataset_uuid is not None}
    if not dataset_uuids:
        return {}
    datasets = list(
        Dataset.objects
        .filter(uuid__in=dataset_uuids)
        .select_related('schema')
        .only('uuid', 'schema', 'scope_content_type', 'scope_id')
    )
    labels: CoordinateLabels = {}
    for dataset in datasets:
        schema = dataset.schema
        if schema is None or dataset.scope_id is None:
            continue
        # The dataframe column is DatasetSchemaDimension.column_name when set and the
        # scoped dimension identifier otherwise -- the same rule the evaluator applies
        # when it records the coordinate.
        scopes = {
            scope.dimension_id: scope
            for scope in DimensionScope.objects
            .filter(
                scope_content_type=dataset.scope_content_type,
                scope_id=dataset.scope_id,
                dimension_id__in=schema.dimensions.values_list('dimension_id', flat=True),
            )
            .select_related('dimension')
            .prefetch_related('dimension__categories')
        }
        for schema_dimension in DatasetSchemaDimension.objects.filter(schema=schema).only('dimension_id', 'column_name'):
            scope = scopes.get(schema_dimension.dimension_id)
            if scope is None:
                continue
            column = schema_dimension.column_name or scope.identifier
            if not column:
                continue
            dimension_label = scope.dimension.name_i18n or column
            for category in scope.dimension.categories.all():
                if category.identifier is None:
                    continue
                labels[(dataset.uuid, column, category.identifier)] = (
                    dimension_label,
                    category.label_i18n or category.identifier,
                )
    return labels


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
    combination_ids: list[UUID] = sb.field(description='Schema category combinations involved in the violation.')
    requirement_group: str | None = sb.field(description='Named required-combination group, when applicable.')

    @classmethod
    def from_violation(cls, violation: RuleViolation, labels: CoordinateLabels | None = None) -> Self:
        labels = labels if labels is not None else {}

        def coordinate(dimension: str, category: str) -> DatasetDimensionCoordinateType:
            dimension_label, category_label = labels.get((violation.dataset_uuid, dimension, category), (dimension, category))
            return DatasetDimensionCoordinateType(
                dimension=dimension,
                category=category,
                dimension_label=dimension_label,
                category_label=category_label,
            )

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
            coordinates=[coordinate(dimension, category) for dimension, category in violation.categories.items()],
            combination_ids=list(violation.combination_ids),
            requirement_group=violation.requirement_group,
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
        found = list(violations)
        labels = build_coordinate_labels(found)
        return cls(violations=[DatasetValidationViolationType.from_violation(violation, labels) for violation in found])
