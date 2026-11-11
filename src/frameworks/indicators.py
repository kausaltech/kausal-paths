"""
Compute the figures a model declares for a submission, and store them on it.

Views over many instances read the stored figures and never build a runtime.
An open submission's figures are rewritten whenever its instance's draft
changes; a final submission's are computed once more from the revision it pins
and then frozen.

Which nodes feed which figure, and under which parameter values, is declared in
the instance spec (`nodes.defs.indicator_defs`). Nothing here names a node.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from django.db import transaction
from django.utils import timezone

import polars as pl
from loguru import logger

from datasets.plausibility import collect_instance_dataset_plausibility_findings
from frameworks.models import (
    Completeness,
    IndicatorCell,
    IndicatorProvenance,
    IndicatorValue,
    Submission,
    SubmissionIndicators,
    SubmissionKind,
    SubmissionStatus,
)
from frameworks.models.submission import OPEN_STATUSES
from nodes.constants import YEAR_COLUMN
from nodes.exceptions import NodeError
from nodes.instance_loader import InstanceLoader
from nodes.instance_serialization import InstanceSnapshot, build_instance_snapshot
from nodes.value_validation import RequirementCounts, collect_instance_value_violations
from params.overrides import InvalidModelOverrideError, ModelOverrides, ParameterOverride, activate_settings
from params.storage import InstanceData

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from common.polars import PathsDataFrame
    from frameworks.models.indicators import SettingValue
    from nodes.defs.indicator_defs import IndicatorSetting, IndicatorSpec, SubmissionIndicatorsSpec
    from nodes.instance import Instance
    from nodes.models import InstanceConfig


class IndicatorComputationError(Exception):
    """The model could not give the declared figures. The message is shown to the reporter."""


@dataclass
class ComputedIndicators:
    """One year's figures as the runtime gives them; where they came from is added when they are stored."""

    values: dict[str, IndicatorValue] = field(default_factory=dict)
    requirements_met: int = 0
    requirements_applicable: int = 0
    declarations: dict[str, SettingValue | None] = field(default_factory=dict)


def _override(setting: IndicatorSetting) -> ParameterOverride:
    value = setting.value
    if isinstance(value, bool):
        return ParameterOverride(id=setting.parameter, bool_value=value)
    if isinstance(value, str):
        return ParameterOverride(id=setting.parameter, string_value=value)
    return ParameterOverride(id=setting.parameter, number_value=value)


def _effective_settings(spec: SubmissionIndicatorsSpec, indicator: IndicatorSpec | None = None) -> tuple[IndicatorSetting, ...]:
    """Merge the shared settings with an indicator's own; the indicator's value wins."""
    merged = {setting.parameter: setting for setting in spec.settings}
    if indicator is not None:
        merged.update({setting.parameter: setting for setting in indicator.settings})
    return tuple(merged[parameter] for parameter in sorted(merged))


def _activate(instance: Instance, settings: tuple[IndicatorSetting, ...]) -> None:
    """
    Put the runtime under the prescribed settings.

    Starts from blank settings, so a visitor's session never reaches the figures,
    and turns normalization off: a stored figure is an absolute one.
    """
    overrides = ModelOverrides(
        parameters=tuple(_override(setting) for setting in settings), normalizer=None, override_normalizer=True
    )
    context = instance.context
    activate_settings(context, overrides.storage_for(context, InstanceData()))


def _year_value(indicator: IndicatorSpec, output: PathsDataFrame, year: int) -> IndicatorValue:
    metrics = output.metric_cols
    if len(metrics) != 1:
        raise IndicatorComputationError(f'Indicator {indicator.id}: node {indicator.node} has {len(metrics)} metrics, not one')
    metric = metrics[0]
    dimensions = list(indicator.dimensions)
    unknown = [dimension for dimension in dimensions if dimension not in output.dim_ids]
    if unknown:
        raise IndicatorComputationError(f'Indicator {indicator.id}: node {indicator.node} has no dimension {", ".join(unknown)}')
    if indicator.composition != 'sum' and output.dim_ids:
        raise IndicatorComputationError(
            f'Indicator {indicator.id}: node {indicator.node} has dimensions, which only a summed indicator can reduce'
        )
    frame = pl.DataFrame({
        YEAR_COLUMN: output[YEAR_COLUMN],
        metric: output[metric].cast(pl.Float64),
        **{dimension: output[dimension].cast(pl.String) for dimension in dimensions},
    }).filter((pl.col(YEAR_COLUMN) == year) & pl.col(metric).is_not_null() & pl.col(metric).is_finite())
    result = IndicatorValue(
        unit=str(output.get_unit(metric)),
        dimensions=dimensions,
        composition=indicator.composition,
        weight=indicator.weight,
        total=None,
    )
    if frame.is_empty():
        return result
    result.total = float(frame[metric].sum())
    if dimensions:
        cells = frame.group_by(dimensions).agg(pl.col(metric).sum()).sort(dimensions)
        result.cells = [
            IndicatorCell(categories={dimension: row[dimension] for dimension in dimensions}, value=row[metric])
            for row in cells.iter_rows(named=True)
        ]
    return result


def compute_indicators(instance: Instance, spec: SubmissionIndicatorsSpec, years: Sequence[int]) -> dict[int, ComputedIndicators]:
    """
    Compute the declared figures of `instance` for each of `years`.

    The runtime is left under the last settings used, so it must be one built for
    this purpose and not one a request is reading. Any failure of the model raises:
    a year either has all of its figures or none.
    """
    context = instance.context
    computed = {year: ComputedIndicators() for year in years}
    try:
        base = _effective_settings(spec)
        _activate(instance, base)
        counts = RequirementCounts()
        collect_instance_value_violations(instance, counts=counts)
        declarations: dict[str, SettingValue | None] = {}
        for parameter_id in spec.declarations:
            parameter = context.get_parameter(parameter_id, required=False)
            if parameter is None:
                raise IndicatorComputationError(f'Declared parameter {parameter_id} does not exist')
            declarations[parameter_id] = parameter.serialize_value()
        for year, result in computed.items():
            result.requirements_applicable = counts.applicable[year]
            result.requirements_met = counts.applicable[year] - counts.unmet[year]
            result.declarations = dict(declarations)

        active = base
        # The shared settings first, so that most indicators need no second activation.
        for indicator in sorted(spec.indicators, key=lambda item: _effective_settings(spec, item) != base):
            settings = _effective_settings(spec, indicator)
            if settings != active:
                _activate(instance, settings)
                active = settings
            if indicator.node not in context.nodes:
                raise IndicatorComputationError(f'Indicator {indicator.id}: node {indicator.node} does not exist')
            with context.run():
                output = context.get_node(indicator.node).get_output_pl()
            for year, result in computed.items():
                result.values[indicator.id] = _year_value(indicator, output, year)
    except (NodeError, InvalidModelOverrideError) as error:
        cause = f': {error.__cause__}' if error.__cause__ is not None else ''
        raise IndicatorComputationError(f'{error}{cause}') from error
    return computed


def _plausibility_counts(ic: InstanceConfig, years: Iterable[int]) -> dict[int, int]:
    findings = collect_instance_dataset_plausibility_findings(ic)
    return {year: sum(1 for finding in findings if year in finding.years) for year in years}


def _store(submission: Submission, indicators: SubmissionIndicators) -> None:
    # Not `save()`: the figures are derived, so they must not move `last_modified_at`.
    Submission.objects.filter(pk=submission.pk).update(indicators=indicators)
    submission.indicators = indicators


def _as_stored(
    year: int, computed: ComputedIndicators, *, plausibility_findings: int, provenance: IndicatorProvenance
) -> SubmissionIndicators:
    values = computed.values
    for value in values.values():
        if value.total is not None and not math.isfinite(value.total):
            value.total = None
    return SubmissionIndicators(
        status='computed',
        year=year,
        values=values,
        completeness=Completeness(
            met=computed.requirements_met,
            applicable=computed.requirements_applicable,
            open_plausibility_findings=plausibility_findings,
        ),
        declarations=computed.declarations,
        provenance=provenance,
    )


def materialise_open_submissions(ic: InstanceConfig) -> int:
    """
    Rewrite the figures of every open submission of `ic` from its current draft.

    All of them, not only the year that was edited: a parameter or a template
    change moves every year, and one runtime computes them all. A draft that
    cannot be computed is recorded as such and never refuses the edit that made
    it so. Returns how many submissions were written.
    """
    submissions = list(Submission.objects.filter(instance_config=ic, kind=SubmissionKind.INVENTORY, status__in=OPEN_STATUSES))
    if not submissions or ic.config_source != 'database':
        return 0
    snapshot = build_instance_snapshot(ic)
    spec = snapshot.spec.submission_indicators
    if spec is None:
        return 0
    years = sorted({submission.period_start for submission in submissions})
    provenance = IndicatorProvenance(
        computed_at=timezone.now(),
        draft_head_token=ic.draft_head_token,
        template_revision=ic.template_revision_id,
        settings={setting.parameter: setting.value for setting in spec.settings},
    )
    computed: dict[int, ComputedIndicators] = {}
    plausibility: dict[int, int] = {}
    reason: str | None = None
    try:
        # A savepoint, so that a failed query inside the computation cannot poison the caller's transaction.
        with transaction.atomic():
            instance = InstanceLoader.from_snapshot(snapshot, instance_config=ic).instance
            try:
                computed = compute_indicators(instance, spec, years)
            finally:
                instance.clean()
            plausibility = _plausibility_counts(ic, years)
    except IndicatorComputationError as error:
        reason = str(error)
    except Exception as error:
        # The edit that led here is valid on its own terms; losing the figures must not lose the edit.
        logger.exception('Submission indicators of {instance} failed to compute', instance=ic.identifier)
        reason = f'{type(error).__name__}: {error}'
    for submission in submissions:
        year = submission.period_start
        if reason is not None:
            indicators = SubmissionIndicators(status='not_computable', reason=reason, year=year, provenance=provenance)
        else:
            indicators = _as_stored(year, computed[year], plausibility_findings=plausibility[year], provenance=provenance)
        _store(submission, indicators)
    return len(submissions)


def materialise_final_submission(submission: Submission) -> None:
    """
    Compute a final submission's figures from the revision it pins, and freeze them.

    Raises `IndicatorComputationError` when the model cannot give them: a final
    submission without figures must not exist where the model declares figures.
    """
    if submission.status != SubmissionStatus.FINAL or submission.instance_revision is None:
        raise ValueError('Only a final submission has a pinned revision to compute from')
    ic = submission.instance_config
    structured = ((submission.instance_revision.content or {}).get('model_snapshot') or {}).get('structured') or {}
    if not structured:
        return
    snapshot = InstanceSnapshot.from_serialized_data(structured)
    if snapshot.spec.submission_indicators is None:
        # Most models declare no figures; do not build a runtime to find that out.
        return
    instance = ic.create_instance_from_revision(submission.instance_revision)
    if instance is None:
        return
    try:
        snapshot = instance.source_snapshot
        spec = snapshot.spec.submission_indicators if snapshot is not None else None
        if spec is None:
            return
        year = submission.period_start
        computed = compute_indicators(instance, spec, [year])[year]
    finally:
        instance.clean()
    provenance = IndicatorProvenance(
        computed_at=timezone.now(),
        instance_revision=submission.instance_revision_id,
        template_revision=submission.template_revision_id,
        settings={setting.parameter: setting.value for setting in spec.settings},
    )
    plausibility = _plausibility_counts(ic, [year])[year]
    _store(submission, _as_stored(year, computed, plausibility_findings=plausibility, provenance=provenance))


def indicators_are_stale(submission: Submission, ic: InstanceConfig) -> bool:
    """
    Whether an open submission's figures predate the instance's current draft.

    Writes that bypass the change operation (an import, a sync, the admin) leave
    the figures behind; a reader shows them as stale, never as current. A final
    submission's figures are frozen and cannot be stale.
    """
    indicators = submission.indicators
    if indicators is None or submission.status not in OPEN_STATUSES:
        return False
    provenance = indicators.provenance
    return provenance.draft_head_token != ic.draft_head_token or provenance.template_revision != ic.template_revision_id
