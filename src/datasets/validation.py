"""
Evaluation of dataset metric validation rules.

The rule schema (a Pydantic discriminated union on ``kind``) lives in
``datasets.validation_rules``; this module owns the evaluator that runs
the rules against a dataset's full contents, the violation payloads
persisted on ``DatasetMaterialization``, and the errors the edit and
publication boundaries raise.

Stored rule blobs are parsed strictly: every write path validates through
``validation_rule_adapter`` before persisting, so an unparseable row is a
bug and fails loudly instead of degrading into a violation.
"""

import hashlib
import json
from typing import TYPE_CHECKING, Any, TypedDict, cast
from uuid import UUID

from django.contrib.contenttypes.models import ContentType
from django.contrib.postgres.expressions import ArraySubquery
from django.core.exceptions import ValidationError as DjangoValidationError
from django.db.models import F, OuterRef
from django.db.models.functions import JSONObject
from pydantic import BaseModel, Field, TypeAdapter

import polars as pl

from kausal_common.datasets.models import DatasetMetricValidationRule, DatasetSchema

from nodes.constants import YEAR_COLUMN

from .coordinates import DatasetCoordinate, DatasetCoordinateIndex
from .shape_domain import dataset_category_domain
from .validation_rules import (
    AllowedCombinationsRule,
    DimensionSumRule,
    Enforcement,
    NoGapsRule,
    RequiredCombinationsRule,
    ValidationRule,
    ValueRangeRule,
    validation_rule_adapter,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from django_stubs_ext import WithAnnotations

    from kausal_common.datasets.category_domain import DatasetCategoryDomain
    from kausal_common.datasets.models import Dataset

#: Cap on located violations reported per rule; the remainder is summarized.
MAX_VIOLATIONS_PER_RULE = 200

#: ``RuleViolation.kind`` used when a valid rule cannot be applied to the
#: dataset (e.g. it references a dimension the data no longer has).
INVALID_RULE_KIND = 'invalid_rule'


class RuleViolation(BaseModel):
    """
    One located violation of a metric validation rule.

    ``categories`` maps dimension column ids to category identifiers, as
    they appear in the dataset's dataframe; ``metric`` is the metric's
    column identifier. ``message`` is an untranslated fallback — clients
    should render from the structured fields.
    """

    rule_uuid: UUID
    kind: str
    enforcement: Enforcement
    metric_uuid: UUID
    metric: str
    dataset_uuid: UUID | None = None
    dataset_identifier: str | None = None
    years: list[int] = Field(default_factory=list)
    categories: dict[str, str] = Field(default_factory=dict)
    coordinates: list[DatasetCoordinate] = Field(default_factory=list)
    combination_ids: list[UUID] = Field(default_factory=list)
    requirement_group: str | None = None
    message: str

    @property
    def key(self) -> tuple[str, str, str, str | None, tuple[str, ...], tuple[tuple[str, str], ...], tuple[int, ...]]:
        """Stable identity for baseline-diffing violation sets across an edit."""
        return (
            str(self.rule_uuid),
            str(self.metric_uuid),
            self.kind,
            self.requirement_group,
            tuple(str(combination_id) for combination_id in self.combination_ids),
            tuple(sorted(self.categories.items())),
            tuple(self.years),
        )


violation_list_adapter: TypeAdapter[list[RuleViolation]] = TypeAdapter(list[RuleViolation])


def dump_violations(violations: list[RuleViolation]) -> list[dict[str, Any]]:
    return violation_list_adapter.dump_python(violations, mode='json')


def load_violations(data: object) -> list[RuleViolation]:
    if not data:
        return []
    return violation_list_adapter.validate_python(data)


def new_blocking_violations(baseline: list[RuleViolation], current: list[RuleViolation]) -> list[RuleViolation]:
    """Violations in ``current`` with ``block_edit`` enforcement that ``baseline`` did not have."""
    baseline_keys = {violation.key for violation in baseline}
    return [v for v in current if v.enforcement == 'block_edit' and v.key not in baseline_keys]


class DatasetValidationError(DjangoValidationError):
    """
    An edit introduced new violations of ``block_edit`` rules; nothing was written.

    Subclasses Django's ``ValidationError`` so the GraphQL mutation machinery
    (``handle_django_errors``) and DRF converters handle it out of the box.
    """

    def __init__(self, violations: list[RuleViolation]):
        self.violations = violations
        super().__init__([violation.message for violation in violations])


class InstanceDatasetValidationError(Exception):
    """
    Publication was refused: the instance's bound datasets carry validation-rule violations.

    The parallel of ``InstanceConstraintError`` for data-level problems. The caller
    selects publication or submission enforcement before raising this error.
    """

    def __init__(self, violations: list[RuleViolation]):
        self.violations = tuple(violations)
        preview = '; '.join(violation.message for violation in violations[:5])
        more = f' (+{len(violations) - 5} more)' if len(violations) > 5 else ''
        super().__init__(f'{len(violations)} dataset validation violation(s): {preview}{more}')


class ValidationRuleFingerprint(TypedDict):
    uuid: UUID | str
    rule: object
    metric__uuid: UUID | str
    metric__name: str | None
    metric__label: str


class _SchemaValidationContext(TypedDict):
    current_validation_rules: list[ValidationRuleFingerprint]


def validation_rules_subquery(schema_id_field: str) -> ArraySubquery:
    """Project ordered rule definitions alongside a bulk materialization query."""
    return ArraySubquery(
        DatasetMetricValidationRule.objects
        .filter(metric__schema_id=OuterRef(schema_id_field))
        .order_by('metric__order', 'metric__uuid', 'order', 'uuid')
        .annotate(fingerprint=JSONObject(**{field: F(field) for field in ValidationRuleFingerprint.__annotations__}))
        .values('fingerprint')
    )


def validation_context_hash(rules: Sequence[ValidationRuleFingerprint], domain: DatasetCategoryDomain | None) -> str:
    payload = json.dumps(
        {'rules': rules, 'domain': domain.model_dump(mode='json') if domain is not None else None},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def dataset_validation_hash(dataset: Dataset) -> str:
    """
    Fingerprint current rule definitions independently of dataset modification time.

    Query the database rather than prefetched relations: a shared schema can be
    changed without touching any of its datasets. Include the category domain
    because it defines the meaning of combination rules.
    """
    if dataset.schema_id is None:
        return validation_context_hash([], None)
    schema = cast(
        'WithAnnotations[DatasetSchema, _SchemaValidationContext]',
        DatasetSchema.objects.annotate(current_validation_rules=validation_rules_subquery('pk')).get(pk=dataset.schema_id),
    )
    return validation_context_hash(schema.current_validation_rules, dataset_category_domain(dataset))


def evaluate_dataset_rules(dataset: Dataset) -> list[RuleViolation]:
    """
    Evaluate all validation rules bound to the dataset's metrics.

    Returns the complete violation set for the dataset's current contents.
    """
    schema = dataset.schema
    if schema is None or dataset.is_external_placeholder:
        return []
    rules = list(
        DatasetMetricValidationRule.objects
        .filter(metric__schema=schema)
        .select_related('metric')
        .order_by('metric__order', 'order'),
    )
    if not rules:
        return []

    from nodes.datasets import DBDataset

    ppdf = DBDataset.deserialize_df(dataset)
    dim_cols = [col for col in ppdf.primary_keys if col != YEAR_COLUMN]
    # Work on a plain polars frame: the rules need no unit metadata, and the
    # categorical dimension columns are cast to strings to keep joins simple.
    df = pl.DataFrame({col: ppdf.get_column(col) for col in ppdf.columns})
    if dim_cols:
        df = df.with_columns([pl.col(col).cast(pl.Utf8) for col in dim_cols])
    domain = dataset_category_domain(dataset)
    domain_coordinates = _category_domain_coordinates(dataset, domain)
    domain_is_closed = domain.mode == 'closed'

    violations: list[RuleViolation] = []
    for row in rules:
        metric = row.metric
        column = metric.name or metric.label or str(metric.uuid)
        rule = validation_rule_adapter.validate_python(row.rule)
        violations.extend(
            evaluate_rule(
                rule,
                row.uuid,
                metric.uuid,
                column,
                df,
                dim_cols,
                domain_coordinates,
                domain_is_closed,
            )
        )
    coordinate_index = DatasetCoordinateIndex(dataset) if violations else None
    for found in violations:
        found.dataset_uuid = dataset.uuid
        found.dataset_identifier = dataset.identifier
        assert coordinate_index is not None
        found.coordinates = coordinate_index.resolve(found.categories)
    return violations


def _row_categories(row: dict[str, Any], cols: list[str]) -> dict[str, str]:
    return {col: row[col] for col in cols if row[col] is not None}


def evaluate_rule(
    rule: ValidationRule,
    rule_uuid: UUID,
    metric_uuid: UUID,
    column: str,
    df: pl.DataFrame,
    dim_cols: list[str],
    domain_coordinates: dict[UUID, dict[str, str]],
    domain_is_closed: bool,
) -> list[RuleViolation]:
    if column not in df.columns:
        # The metric has no data points at all; every rule holds vacuously.
        return []

    def violation(**kwargs) -> RuleViolation:
        # The rule's own kind and enforcement are defaults: a rule that cannot be
        # evaluated reports itself as INVALID_RULE_KIND with its own enforcement.
        kwargs.setdefault('kind', rule.kind)
        kwargs.setdefault('enforcement', rule.enforcement)
        return RuleViolation(rule_uuid=rule_uuid, metric_uuid=metric_uuid, metric=column, **kwargs)

    match rule:
        case ValueRangeRule():
            return _eval_value_range(rule, violation, column, df, dim_cols)
        case DimensionSumRule():
            return _eval_dimension_sum(rule, violation, column, df, dim_cols)
        case NoGapsRule():
            return _eval_no_gaps(violation, column, df, dim_cols)
        case RequiredCombinationsRule():
            return _eval_required_combinations(rule, violation, column, df, domain_coordinates)
        case AllowedCombinationsRule():
            return _eval_allowed_combinations(
                violation,
                column,
                df,
                dim_cols,
                domain_coordinates,
                domain_is_closed,
            )


def _category_domain_coordinates(dataset: Dataset, domain: DatasetCategoryDomain | None = None) -> dict[UUID, dict[str, str]]:
    from kausal_common.datasets.models import DatasetSchemaDimension

    from frameworks.catalogue import dimension_scopes

    schema = dataset.schema
    if domain is None:
        domain = dataset_category_domain(dataset)
    if schema is None or not domain.combinations:
        return {}
    instance = dataset.scope_instance
    local_type_id = ContentType.objects.get_for_model(instance).pk
    visible_scopes = list(
        dimension_scopes(instance)
        .filter(dimension_id__in=schema.dimensions.values_list('dimension_id', flat=True))
        .select_related('dimension')
        .prefetch_related('dimension__categories')
    )
    scopes = {
        scope.dimension.pk: scope
        for scope in sorted(visible_scopes, key=lambda scope: scope.scope_content_type_id == local_type_id)
    }
    columns = {
        dimension_id: column_name or (scopes[dimension_id].identifier if dimension_id in scopes else None)
        for dimension_id, column_name in DatasetSchemaDimension.objects.filter(schema=schema).values_list(
            'dimension_id', 'column_name'
        )
    }
    category_identifiers = {
        category.uuid: category.identifier
        for scope in scopes.values()
        for category in scope.dimension.categories.all()
        if category.identifier is not None
    }
    result: dict[UUID, dict[str, str]] = {}
    for combination in domain.combinations:
        coordinates: dict[str, str] = {}
        for dimension_uuid, category_uuid in combination.categories.items():
            dimension_id = next(
                (dimension_id for dimension_id, scope in scopes.items() if scope.dimension.uuid == dimension_uuid),
                None,
            )
            column = columns.get(dimension_id) if dimension_id is not None else None
            category = category_identifiers.get(category_uuid)
            if column is not None and category is not None:
                coordinates[column] = category
        if len(coordinates) == len(combination.categories):
            result[combination.id] = coordinates
    return result


def _eval_required_combinations(
    rule: RequiredCombinationsRule,
    violation: Callable[..., RuleViolation],
    column: str,
    df: pl.DataFrame,
    domain_coordinates: dict[UUID, dict[str, str]],
) -> list[RuleViolation]:
    unknown = [
        combination for group in rule.groups for combination in group.combinations if combination not in domain_coordinates
    ]
    if unknown:
        return [
            violation(
                kind=INVALID_RULE_KIND,
                enforcement='block_publish',
                combination_ids=unknown,
                message='required_combinations rule references combinations outside the dataset schema domain',
            )
        ]
    years = df.select(YEAR_COLUMN).unique().sort(YEAR_COLUMN).get_column(YEAR_COLUMN).to_list()
    present = df.filter(pl.col(column).is_not_null())
    violations: list[RuleViolation] = []
    for group in rule.groups:
        missing_years: list[int] = []
        for year in years:
            candidates = present.filter(pl.col(YEAR_COLUMN) == year)
            satisfied = False
            for combination_id in group.combinations:
                coordinates = domain_coordinates[combination_id]
                matching = candidates
                for dimension, category in coordinates.items():
                    matching = matching.filter(pl.col(dimension).cast(pl.Utf8) == category)
                if not matching.is_empty():
                    satisfied = True
                    break
            if not satisfied:
                missing_years.append(year)
        if missing_years:
            single_coordinates = domain_coordinates[group.combinations[0]] if len(group.combinations) == 1 else {}
            violations.append(
                violation(
                    years=missing_years,
                    categories=single_coordinates,
                    combination_ids=group.combinations,
                    requirement_group=group.id,
                    message=(
                        f'{column} has no value for required category group {group.id} '
                        f'in year(s) {_years_text(missing_years)}; an explicit 0 counts as a value'
                    ),
                )
            )
    return violations


def _eval_allowed_combinations(
    violation: Callable[..., RuleViolation],
    column: str,
    df: pl.DataFrame,
    dim_cols: list[str],
    domain_coordinates: dict[UUID, dict[str, str]],
    domain_is_closed: bool,
) -> list[RuleViolation]:
    if not domain_is_closed:
        return []
    allowed = {tuple(sorted(coordinates.items())) for coordinates in domain_coordinates.values()}
    offending: list[dict[str, Any]] = []
    for row in df.filter(pl.col(column).is_not_null()).sort([YEAR_COLUMN, *dim_cols]).iter_rows(named=True):
        categories = _row_categories(row, dim_cols)
        if tuple(sorted(categories.items())) not in allowed:
            offending.append(row)
    violations = [
        violation(
            years=[row[YEAR_COLUMN]],
            categories=(categories := _row_categories(row, dim_cols)),
            message=f'{column} uses a category combination outside the schema domain{_categories_text(categories)}',
        )
        for row in offending[:MAX_VIOLATIONS_PER_RULE]
    ]
    return _with_overflow(violations, len(offending), violation)


def _eval_value_range(
    rule: ValueRangeRule,
    violation: Callable[..., RuleViolation],
    column: str,
    df: pl.DataFrame,
    dim_cols: list[str],
) -> list[RuleViolation]:
    conditions = []
    if rule.min is not None:
        conditions.append(pl.col(column) <= rule.min if rule.exclusive_min else pl.col(column) < rule.min)
    if rule.max is not None:
        conditions.append(pl.col(column) >= rule.max if rule.exclusive_max else pl.col(column) > rule.max)
    out_of_range = conditions[0] if len(conditions) == 1 else conditions[0] | conditions[1]
    offending = df.filter(pl.col(column).is_not_null() & out_of_range).sort([YEAR_COLUMN, *dim_cols])
    bounds = _range_text(rule)
    violations = [
        violation(
            years=[row[YEAR_COLUMN]],
            categories=(cats := _row_categories(row, dim_cols)),
            message=(
                f'{column} = {row[column]:g} is outside the allowed range ({bounds}) '
                f'in {row[YEAR_COLUMN]}{_categories_text(cats)}'
            ),
        )
        for row in offending.head(MAX_VIOLATIONS_PER_RULE).iter_rows(named=True)
    ]
    return _with_overflow(violations, len(offending), violation)


def _eval_dimension_sum(
    rule: DimensionSumRule,
    violation: Callable[..., RuleViolation],
    column: str,
    df: pl.DataFrame,
    dim_cols: list[str],
) -> list[RuleViolation]:
    if rule.dimension not in dim_cols:
        return [
            violation(
                kind=INVALID_RULE_KIND,
                enforcement='block_publish',
                message=f"dimension_sum rule references dimension '{rule.dimension}', which the dataset does not have",
            )
        ]
    group_cols = [YEAR_COLUMN, *(col for col in dim_cols if col != rule.dimension)]
    sums = df.filter(pl.col(column).is_not_null()).group_by(group_cols).agg(pl.col(column).sum().alias('_sum'))
    offending = sums.filter((pl.col('_sum') - rule.target).abs() > rule.tolerance).sort(group_cols)
    violations = [
        violation(
            years=[row[YEAR_COLUMN]],
            categories=(cats := _row_categories(row, group_cols[1:])),
            message=(
                f'{column} sums to {row["_sum"]:g} over {rule.dimension} '
                f'(expected {rule.target:g} ± {rule.tolerance:g}) '
                f'in {row[YEAR_COLUMN]}{_categories_text(cats)}'
            ),
        )
        for row in offending.head(MAX_VIOLATIONS_PER_RULE).iter_rows(named=True)
    ]
    return _with_overflow(violations, len(offending), violation)


def _eval_no_gaps(
    violation: Callable[..., RuleViolation],
    column: str,
    df: pl.DataFrame,
    dim_cols: list[str],
) -> list[RuleViolation]:
    if not dim_cols:
        # Without dimensions, every year present in the data carries
        # the value that made it present; gaps cannot exist.
        return []
    present = df.filter(pl.col(column).is_not_null()).select([YEAR_COLUMN, *dim_cols])
    if present.is_empty():
        return []
    years = present.select(YEAR_COLUMN).unique()
    combos = present.select(dim_cols).unique()
    expected = combos.join(years, how='cross')
    missing = expected.join(present, on=[YEAR_COLUMN, *dim_cols], how='anti', nulls_equal=True)
    gaps = missing.group_by(dim_cols).agg(pl.col(YEAR_COLUMN).sort().alias('_years')).sort(dim_cols)
    violations = [
        violation(
            years=list(row['_years']),
            categories=(cats := _row_categories(row, dim_cols)),
            message=(
                f'{column} has no value{_categories_text(cats)} '
                f'for year(s) {_years_text(row["_years"])}; an explicit 0 counts as a value'
            ),
        )
        for row in gaps.head(MAX_VIOLATIONS_PER_RULE).iter_rows(named=True)
    ]
    return _with_overflow(violations, len(gaps), violation)


def _with_overflow(
    violations: list[RuleViolation],
    total: int,
    violation: Callable[..., RuleViolation],
) -> list[RuleViolation]:
    if total > MAX_VIOLATIONS_PER_RULE:
        violations.append(
            violation(
                message=f'... and {total - MAX_VIOLATIONS_PER_RULE} further violations of the same rule',
            )
        )
    return violations


def _range_text(rule: ValueRangeRule) -> str:
    if rule.min is not None and rule.max is not None:
        left = '>' if rule.exclusive_min else '≥'
        right = '<' if rule.exclusive_max else '≤'
        return f'{left} {rule.min:g} and {right} {rule.max:g}'
    if rule.min is not None:
        operator = '>' if rule.exclusive_min else '≥'
        return f'{operator} {rule.min:g}'
    operator = '<' if rule.exclusive_max else '≤'
    return f'{operator} {rule.max:g}'


def _categories_text(categories: dict[str, str]) -> str:
    if not categories:
        return ''
    return ' [' + ', '.join(f'{dim}={cat}' for dim, cat in categories.items()) + ']'


def _years_text(years: list[int]) -> str:
    return ', '.join(str(year) for year in years)
