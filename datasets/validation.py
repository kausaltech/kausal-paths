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

from typing import TYPE_CHECKING, Any
from uuid import UUID

from django.core.exceptions import ValidationError as DjangoValidationError
from pydantic import BaseModel, Field, TypeAdapter

import polars as pl

from nodes.constants import YEAR_COLUMN

from .validation_rules import (
    DimensionSumRule,
    Enforcement,
    NoGapsRule,
    ValidationRule,
    ValueRangeRule,
    validation_rule_adapter,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
    message: str

    @property
    def key(self) -> tuple[str, str, tuple[tuple[str, str], ...], tuple[int, ...]]:
        """Stable identity for baseline-diffing violation sets across an edit."""
        return (
            str(self.rule_uuid),
            str(self.metric_uuid),
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

    The parallel of ``InstanceConstraintError`` for data-level problems; both
    enforcement tiers block publication.
    """

    def __init__(self, violations: list[RuleViolation]):
        self.violations = tuple(violations)
        preview = '; '.join(violation.message for violation in violations[:5])
        more = f' (+{len(violations) - 5} more)' if len(violations) > 5 else ''
        super().__init__(f'{len(violations)} dataset validation violation(s): {preview}{more}')


def evaluate_dataset_rules(dataset: Dataset) -> list[RuleViolation]:
    """
    Evaluate all validation rules bound to the dataset's metrics.

    Returns the complete violation set for the dataset's current contents.
    """
    from kausal_common.datasets.models import DatasetMetricValidationRule

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

    violations: list[RuleViolation] = []
    for row in rules:
        metric = row.metric
        column = metric.name or metric.label or str(metric.uuid)
        rule = validation_rule_adapter.validate_python(row.rule)
        violations.extend(_evaluate_rule(rule, row.uuid, metric.uuid, column, df, dim_cols))
    for found in violations:
        found.dataset_uuid = dataset.uuid
        found.dataset_identifier = dataset.identifier
    return violations


def _row_categories(row: dict[str, Any], cols: list[str]) -> dict[str, str]:
    return {col: row[col] for col in cols if row[col] is not None}


def _evaluate_rule(
    rule: ValidationRule,
    rule_uuid: UUID,
    metric_uuid: UUID,
    column: str,
    df: pl.DataFrame,
    dim_cols: list[str],
) -> list[RuleViolation]:
    if column not in df.columns:
        # The metric has no data points at all; every rule holds vacuously.
        return []

    def violation(**kwargs) -> RuleViolation:
        return RuleViolation(
            rule_uuid=rule_uuid,
            kind=rule.kind,
            enforcement=rule.enforcement,
            metric_uuid=metric_uuid,
            metric=column,
            **kwargs,
        )

    match rule:
        case ValueRangeRule():
            return _eval_value_range(rule, violation, column, df, dim_cols)
        case DimensionSumRule():
            return _eval_dimension_sum(rule, violation, column, df, dim_cols)
        case NoGapsRule():
            return _eval_no_gaps(violation, column, df, dim_cols)


def _eval_value_range(
    rule: ValueRangeRule,
    violation: Callable[..., RuleViolation],
    column: str,
    df: pl.DataFrame,
    dim_cols: list[str],
) -> list[RuleViolation]:
    conditions = []
    if rule.min is not None:
        conditions.append(pl.col(column) < rule.min)
    if rule.max is not None:
        conditions.append(pl.col(column) > rule.max)
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
        return f'{rule.min:g} … {rule.max:g}'
    if rule.min is not None:
        return f'≥ {rule.min:g}'
    return f'≤ {rule.max:g}'


def _categories_text(categories: dict[str, str]) -> str:
    if not categories:
        return ''
    return ' [' + ', '.join(f'{dim}={cat}' for dim, cat in categories.items()) + ']'


def _years_text(years: list[int]) -> str:
    return ', '.join(str(year) for year in years)
