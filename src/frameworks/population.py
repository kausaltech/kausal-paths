"""Versioned municipal population lookup and territorial rollups."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from django.db import transaction
from django.db.models import Q

from datasets.placeholders import build_dataset_repo
from frameworks.models import Framework, OrganizationPopulation
from orgs.models import Organization, OrganizationIdentifier

if TYPE_CHECKING:
    import polars as pl


SOURCE_DATASET = 'demography/population_lau'


@dataclass(frozen=True)
class PopulationImportResult:
    observations: int
    years: tuple[int, ...]
    source_revision: str


@dataclass(frozen=True)
class PopulationAggregate:
    value: int
    observed: int


@dataclass(frozen=True)
class PopulationYearIndex:
    by_path: dict[str, PopulationAggregate]
    source_revision: str | None


def _municipal_organizations(framework: Framework) -> dict[str, Organization]:
    roots = list(framework.organization_roots.values_list('organization__path', flat=True))
    if not roots:
        raise ValueError('Framework has no organization roots.')
    scope = Q(pk__in=[])
    for path in roots:
        scope |= Q(organization__path__startswith=path)
    identifiers = OrganizationIdentifier.objects.filter(
        scope,
        namespace__identifier='ags',
        organization__classification__identifier__in=('de_municipality', 'de_district_free_city'),
    ).select_related('organization')
    return {identifier.identifier: identifier.organization for identifier in identifiers}


def _validated_population(lau: str, raw_year: float | str | None, raw_value: float | str) -> tuple[int, int]:
    if raw_year is None:
        raise ValueError(f'Missing year for {lau}.')
    try:
        year = int(raw_year)
        value = int(raw_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'Invalid population for {lau} in {raw_year}: {raw_value}') from error
    if year != raw_year or value != raw_value or value < 0:
        raise ValueError(f'Invalid population for {lau} in {raw_year}: {raw_value}')
    return year, value


@transaction.atomic
def replace_population_projection(
    framework: Framework, frame: pl.DataFrame, *, source_revision: str, source_dataset: str = SOURCE_DATASET
) -> PopulationImportResult:
    """Replace one framework's provider projection only after validating the whole input."""
    year_column = 'Year' if 'Year' in frame.columns else 'year'
    required = {'lau', 'population', year_column}
    if not required <= set(frame.columns):
        raise ValueError(f'Population dataset lacks columns: {sorted(required - set(frame.columns))}')
    if not source_revision:
        raise ValueError('Population source revision must be specified.')
    organizations = _municipal_organizations(framework)
    observations: list[OrganizationPopulation] = []
    seen: set[tuple[str, int]] = set()
    years: set[int] = set()
    for lau, raw_year, raw_value in frame.select('lau', year_column, 'population').iter_rows():
        if not isinstance(lau, str) or not lau.startswith('DE_'):
            continue
        ags = lau[3:]
        organization = organizations.get(ags)
        if organization is None or raw_value is None:
            continue
        year, value = _validated_population(lau, raw_year, raw_value)
        key = (ags, year)
        if key in seen:
            raise ValueError(f'Duplicate population for {lau} in {year}')
        seen.add(key)
        years.add(year)
        observations.append(
            OrganizationPopulation(
                framework=framework,
                organization=organization,
                year=year,
                value=value,
                source_dataset=source_dataset,
                source_revision=source_revision,
            )
        )
    if not observations:
        raise ValueError('Population dataset has no observations for this framework.')
    OrganizationPopulation.objects.filter(framework=framework).delete()
    OrganizationPopulation.objects.bulk_create(observations, batch_size=1000)
    return PopulationImportResult(len(observations), tuple(sorted(years)), source_revision)


def refresh_population_from_dvc(framework: Framework) -> PopulationImportResult:
    template = framework.template_instance
    if template is None:
        raise ValueError('Framework has no template instance.')
    repo_spec = template.ensure_spec().dataset_repo
    if repo_spec is None or repo_spec.commit is None:
        raise ValueError('The template needs a pinned dataset repository commit.')
    dataset = build_dataset_repo(repo_spec).load_dataset(SOURCE_DATASET)
    if dataset.df is None:
        raise ValueError(f'{SOURCE_DATASET} has no dataframe.')
    return replace_population_projection(framework, dataset.df, source_revision=repo_spec.commit)


def population_aggregates(framework: Framework, year: int) -> PopulationYearIndex:
    """One query for a year's leaves; attribute each to all of its tree ancestors."""
    totals: dict[str, PopulationAggregate] = {}
    revision: str | None = None
    paths = OrganizationPopulation.objects.filter(framework=framework, year=year).values_list(
        'organization__path', 'value', 'source_revision'
    )
    for path, value, row_revision in paths.iterator():
        if revision is None:
            revision = row_revision
        elif revision != row_revision:
            raise ValueError(f'Mixed population source revisions for {framework.identifier} in {year}.')
        for length in range(Organization.steplen, len(path) + 1, Organization.steplen):
            prefix = path[:length]
            current = totals.get(prefix, PopulationAggregate(0, 0))
            totals[prefix] = PopulationAggregate(current.value + value, current.observed + 1)
    return PopulationYearIndex(totals, revision)
