"""
Aggregation over a framework's organization tree, computed on read.

An organization's aggregation states, for one kind of submission and one year,
which of the reporting units beneath it have a final submission, and lists its
direct children as members. Nothing is stored: the tree, the instances and the
submissions are the record.

The basis is the published sum: only final submissions count, and a territory's
figures are composed from the figures frozen on them (`frameworks.composition`).
No runtime is built.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING

from django.core.exceptions import PermissionDenied, ValidationError
from django.db.models import Max, Prefetch, Q

from frameworks.composition import Composition, compose
from frameworks.models import (
    FrameworkConfig,
    FrameworkOrganizationRoot,
    Submission,
    SubmissionEvent,
    SubmissionKind,
    SubmissionStatus,
)
from frameworks.models.submission import OPEN_STATUSES
from frameworks.population import unit_populations
from orgs.models import Organization

if TYPE_CHECKING:
    from frameworks.models import Framework, SubmissionIndicators
    from frameworks.organization_access import ViewerScope


class AggregationError(ValidationError):
    pass


class AggregationBasis(Enum):
    PUBLISHED_SUM = 'published_sum'
    """Each unit contributes its final submission, as it was frozen."""


class ExclusionReason(Enum):
    NO_INSTANCE = 'no_instance'
    NO_SUBMISSION = 'no_submission'
    NOT_FINAL = 'not_final'


@dataclass(frozen=True)
class Coverage:
    """How much of a territory's reporting units a published sum would rest on."""

    reporting_units: int
    activated: int | None
    """Units that hold an instance. Null where the figure is withheld."""
    finalised: int | None
    """Units with a final submission. Null where the figure is withheld."""
    population: int | None
    """Reference population of all units; null unless every unit has an observation."""
    partial_population: int | None
    """Reference population of the units that have an observation."""
    finalised_population: int | None
    """Reference population of the finalised units; null if one lacks an observation, or where withheld."""


@dataclass(frozen=True)
class AggregationEntry:
    """One organization in an aggregation: the aggregated organization itself, or one of its members."""

    organization: Organization
    collapsed_from: Organization | None
    """The territory this unit stands for, when that territory holds nothing but the unit."""
    is_reporting_unit: bool
    instance_identifier: str | None
    final_submission: Submission | None
    """The organization's own final submission. For a territory it is shown beside the sum, never added to it."""
    open_status: SubmissionStatus | None
    """Draft or in review, when the organization has an open submission; beside a final one, a correction."""
    exclusion_reason: ExclusionReason | None
    """Why a reporting unit does not count as finalised; null when it does, and for territories."""
    latest_year: int | None
    """The newest year for which the organization has any submission of this kind."""
    coverage: Coverage
    figures: Composition | None
    """
    The figures of the finalised reporting units beneath this organization, composed. The
    organization's own balance is not among them. Null where withheld.
    """
    details_withheld: bool
    """The viewer may see this organization's totals, but not its submissions or members."""
    figures_withheld: bool
    """The territory holds too few reporting units for its totals to be shown to this viewer."""


@dataclass(frozen=True)
class Aggregation:
    kind: SubmissionKind
    year: int
    basis: AggregationBasis
    territorial_reference_date: date | None
    population_source_revision: str | None
    own: AggregationEntry
    members: list[AggregationEntry]
    member_count: int
    members_restricted: bool
    """Only the members on the viewer's own path are listed."""


def existed_in(year: int, prefix: str = '') -> Q:
    """
    Match organizations that existed at the end of `year`.

    A dissolution date is the last day of existence, so a unit dissolved on
    31 December still belongs to that year and its successor, founded on
    1 January, does not.
    """
    reference = date(year, 12, 31)
    founded = Q(**{f'{prefix}founding_date__isnull': True}) | Q(**{f'{prefix}founding_date__lte': reference})
    not_dissolved = Q(**{f'{prefix}dissolution_date__isnull': True}) | Q(**{f'{prefix}dissolution_date__gte': reference})
    return founded & not_dissolved


@dataclass
class _Subtree:
    """Everything one aggregation reads, loaded once for the aggregated organization's subtree."""

    unit_paths: dict[int, str]
    instances: dict[int, str]
    finals: dict[int, int]
    open_statuses: dict[int, SubmissionStatus]
    populations: dict[int, int]
    population_revision: str | None
    indicators: dict[int, SubmissionIndicators]
    """The frozen figures of each finalised reporting unit, by organization."""

    def figures(self, path: str) -> Composition:
        return compose([
            self.indicators[pk]
            for pk, unit_path in self.unit_paths.items()
            if unit_path.startswith(path) and pk in self.indicators
        ])

    def coverage(self, path: str) -> Coverage:
        units = [pk for pk, unit_path in self.unit_paths.items() if unit_path.startswith(path)]
        observed = [self.populations[pk] for pk in units if pk in self.populations]
        finalised = [pk for pk in units if pk in self.finals]
        return Coverage(
            reporting_units=len(units),
            activated=sum(1 for pk in units if pk in self.instances),
            finalised=len(finalised),
            population=sum(observed) if units and len(observed) == len(units) else None,
            partial_population=sum(observed) if observed else None,
            finalised_population=(
                sum(self.populations[pk] for pk in finalised) if all(pk in self.populations for pk in finalised) else None
            ),
        )


def _load_subtree(framework: Framework, organization: Organization, *, kind: SubmissionKind, year: int) -> _Subtree:
    unit_paths = dict(
        Organization.objects
        .filter(framework.reporting_unit_q(), existed_in(year), path__startswith=organization.path)
        .values_list('pk', 'path')
        .distinct()
    )
    in_subtree = Q(instance_config__organization__path__startswith=organization.path)
    instances: dict[int, str] = {}
    for organization_id, identifier in (
        FrameworkConfig.objects
        .filter(in_subtree, framework=framework)
        .order_by('instance_config__identifier')
        .values_list('instance_config__organization_id', 'instance_config__identifier')
    ):
        instances.setdefault(organization_id, identifier)

    # An organization with several instances counts once. The most recently finalised submission stands for it.
    finals: dict[int, int] = {}
    open_statuses: dict[int, SubmissionStatus] = {}
    submissions = (
        Submission.objects
        .filter(
            in_subtree,
            instance_config__framework_config__framework=framework,
            kind=kind,
            period_start=year,
            period_end=year,
            status__in=(*OPEN_STATUSES, SubmissionStatus.FINAL),
        )
        .order_by('finalised_at')
        .values_list('pk', 'instance_config__organization_id', 'status')
    )
    for pk, organization_id, status in submissions:
        if status == SubmissionStatus.FINAL:
            finals[organization_id] = pk
        elif open_statuses.get(organization_id) != SubmissionStatus.IN_REVIEW:
            open_statuses[organization_id] = SubmissionStatus(status)
    populations, revision = unit_populations(framework, year, organization.path)
    by_submission = {
        pk: indicators
        for pk, indicators in Submission.objects.filter(
            pk__in=[pk for organization_id, pk in finals.items() if organization_id in unit_paths]
        ).values_list('pk', 'indicators')
        if indicators is not None
    }
    indicators = {organization_id: by_submission[pk] for organization_id, pk in finals.items() if pk in by_submission}
    return _Subtree(unit_paths, instances, finals, open_statuses, populations, revision, indicators)


def _exclusion_reason(subtree: _Subtree, organization_id: int) -> ExclusionReason | None:
    if organization_id in subtree.finals:
        return None
    if organization_id not in subtree.instances:
        return ExclusionReason.NO_INSTANCE
    if organization_id in subtree.open_statuses:
        return ExclusionReason.NOT_FINAL
    return ExclusionReason.NO_SUBMISSION


def _collapse_single_units(
    members: list[Organization], subtree: _Subtree, year: int
) -> list[tuple[Organization, Organization | None]]:
    """
    Present a territory that holds nothing but one reporting unit as that unit.

    A kreisfreie Stadt is such a territory: a district-level node whose only
    child is the municipality that reports. Listing the territory would offer a
    drill-down into an area that contains only itself.
    """
    candidates = [member for member in members if member.pk not in subtree.unit_paths]
    if not candidates:
        return [(member, None) for member in members]
    child_depth = members[0].depth + 1
    scope = Q(pk__in=[])
    for member in candidates:
        scope |= Q(path__startswith=member.path)
    children: dict[str, list[Organization]] = defaultdict(list)
    for child in Organization.objects.filter(scope, existed_in(year), depth=child_depth).select_related('classification'):
        children[child.path[: len(child.path) - Organization.steplen]].append(child)
    result: list[tuple[Organization, Organization | None]] = []
    for member in members:
        only = children.get(member.path, [])
        if member.pk not in subtree.unit_paths and len(only) == 1 and only[0].pk in subtree.unit_paths:
            result.append((only[0], member))
        else:
            result.append((member, None))
    return result


def _restrict(entry: AggregationEntry, *, minimum_units: int) -> AggregationEntry:
    """Reduce an entry to what a viewer outside it may see: its totals, if it holds enough units to hide each one."""
    coverage = entry.coverage
    too_few = coverage.reporting_units < minimum_units
    # A sum of fewer finalised units than the minimum gives each one away, however large the territory.
    figures = entry.figures if not too_few and (coverage.finalised or 0) >= minimum_units else None
    if too_few:
        coverage = replace(coverage, activated=None, finalised=None, finalised_population=None)
    return replace(
        entry,
        figures=figures,
        instance_identifier=None,
        final_submission=None,
        open_status=None,
        exclusion_reason=None,
        latest_year=None,
        coverage=coverage,
        details_withheld=True,
        figures_withheld=too_few,
    )


def aggregate(
    framework: Framework,
    organization: Organization,
    *,
    year: int,
    scope: ViewerScope,
    kind: SubmissionKind = SubmissionKind.INVENTORY,
    first: int = 100,
    offset: int = 0,
) -> Aggregation:
    """
    Aggregate the reporting units beneath `organization` for one kind and year.

    A viewer whose scope covers the organization gets every member. A viewer who
    only holds something inside it gets its totals and the members on the way to
    what they hold; anyone else is refused.
    """
    if not framework.reporting.units.classes:
        raise AggregationError(f'Framework {framework.identifier} declares no reporting units')
    root = (
        FrameworkOrganizationRoot.objects
        .filter(framework=framework)
        .select_related('organization')
        .order_by('organization__path')
    )
    covering_root = next((r for r in root if organization.path.startswith(r.organization.path)), None)
    if covering_root is None:
        raise PermissionDenied('The organization is outside the framework')
    full = scope.covers(organization.path)
    if not full and not scope.is_above(organization.path):
        raise PermissionDenied('The organization is outside what the viewer may read')

    subtree = _load_subtree(framework, organization, kind=kind, year=year)
    children = list(
        Organization.objects
        .filter(existed_in(year), path__startswith=organization.path, depth=organization.depth + 1)
        .select_related('classification')
        .order_by('path')
    )
    if not full:
        children = [child for child in children if scope.covers(child.path) or scope.is_above(child.path)]
    page = _collapse_single_units(children[offset : offset + first], subtree, year)

    shown = [organization, *(member for member, _ in page)]
    final_ids = [subtree.finals[org.pk] for org in shown if org.pk in subtree.finals]
    finals = {
        submission.pk: submission
        for submission in Submission.objects
        .filter(pk__in=final_ids)
        .select_related('finalised_by', 'created_by', 'supersedes')
        .prefetch_related(Prefetch('events', queryset=SubmissionEvent.objects.select_related('actor')))
    }
    latest_years: dict[int, int] = dict(
        Submission.objects
        .filter(
            instance_config__framework_config__framework=framework,
            instance_config__organization_id__in=[org.pk for org in shown],
            kind=kind,
        )
        .values('instance_config__organization_id')
        .annotate(latest=Max('period_start'))
        .values_list('instance_config__organization_id', 'latest')
    )
    minimum_units = framework.reporting.min_units_for_ancestor_total

    def entry(org: Organization, collapsed_from: Organization | None) -> AggregationEntry:
        is_unit = org.pk in subtree.unit_paths
        result = AggregationEntry(
            organization=org,
            collapsed_from=collapsed_from,
            is_reporting_unit=is_unit,
            instance_identifier=subtree.instances.get(org.pk),
            final_submission=finals.get(subtree.finals.get(org.pk, -1)),
            open_status=subtree.open_statuses.get(org.pk),
            exclusion_reason=_exclusion_reason(subtree, org.pk) if is_unit else None,
            latest_year=latest_years.get(org.pk),
            coverage=subtree.coverage(org.path),
            figures=subtree.figures(org.path),
            details_withheld=False,
            figures_withheld=False,
        )
        return result if scope.covers(org.path) else _restrict(result, minimum_units=minimum_units)

    return Aggregation(
        kind=kind,
        year=year,
        basis=AggregationBasis.PUBLISHED_SUM,
        territorial_reference_date=covering_root.territorial_reference_date,
        population_source_revision=subtree.population_revision,
        own=entry(organization, None),
        members=[entry(member, collapsed_from) for member, collapsed_from in page],
        member_count=len(children),
        members_restricted=not full,
    )
