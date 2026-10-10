"""Copy pinned provider defaults into editable municipal datasets."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from pydantic import BaseModel, Field

import polars as pl

from kausal_common.datasets.models import (
    DataPoint,
    DataPointDimensionCategory,
    Dataset,
    DatasetSourceReference,
    DataSource,
    DimensionScope,
)

from datasets.materialization import refresh_dataset_materialization
from datasets.placeholders import build_dataset_repo
from frameworks.bisko.default_sources import (
    ENERGY_DATASET,
    ENERGY_SOURCE,
    POPULATION_DATASET,
    POPULATION_SOURCE,
    DefaultProvenance,
    EnergyDefault,
    PopulationDefault,
)
from frameworks.evidence import get_evidence, set_evidence
from frameworks.models import DataEvidenceKind, DataQualityLevel
from nodes.template_graph import template_snapshot

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


class ProviderCell(BaseModel):
    value: Decimal
    source_revision: str
    provenance: DefaultProvenance


class ProviderDefaults(BaseModel):
    """Latest provider values; local observations can differ without losing their reference."""

    cells: dict[str, ProviderCell] = Field(default_factory=dict)
    overrides: set[str] = Field(default_factory=set)


def load_activation_sources(instance: InstanceConfig) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    spec = template_snapshot(instance).spec.dataset_repo
    if spec is None or spec.commit is None:
        raise ValueError('Activation defaults require a pinned published template dataset repository.')
    repo = build_dataset_repo(spec)
    missing = [identifier for identifier in (POPULATION_SOURCE, ENERGY_SOURCE) if not repo.has_dataset(identifier)]
    if missing:
        raise ValueError(f'Published template dataset pin lacks activation defaults: {missing}')
    population = repo.load_dataset(POPULATION_SOURCE).df
    energy = repo.load_dataset(ENERGY_SOURCE).df
    if population is None or energy is None:
        raise ValueError('Activation default sources lack dataframes.')
    return population, energy, spec.commit


def _provider_owned(point: DataPoint, previous: ProviderCell | None) -> bool:
    if point.last_modified_by_id is not None:
        return False
    evidence = get_evidence(point)
    if evidence is None:
        return point.value is None and previous is None
    return (
        evidence.kind == DataEvidenceKind.PROVIDER_DEFAULT
        and evidence.last_modified_by_id is None
        and previous is not None
        and point.value == previous.value
    )


def _cite(point: DataPoint, cell: ProviderCell, instance: InstanceConfig) -> None:
    provenance = cell.provenance
    source, _ = DataSource.objects.get_or_create(
        scope_content_type=ContentType.objects.get_for_model(instance),
        scope_id=instance.pk,
        name=f'BISKO provider default: {provenance.source_dataset}',
        edition=provenance.source_edition or cell.source_revision,
        description=provenance.describe(cell.source_revision),
        url=provenance.source_url,
    )
    # This point is still provider-owned. Replace only its previous provider citations.
    point.source_references.filter(data_source__name__startswith='BISKO provider default:').delete()
    DatasetSourceReference.objects.get_or_create(data_point=point, data_source=source)


@transaction.atomic
def seed_population_defaults(
    instance: InstanceConfig,
    dataset: Dataset,
    values: list[PopulationDefault],
    *,
    source_revision: str,
) -> int:
    """Refresh untouched defaults, record new references, and preserve municipal edits."""
    dataset = Dataset.objects.select_for_update().get(pk=dataset.pk)
    if instance.is_locked or dataset.identifier != POPULATION_DATASET or dataset.scope_instance.pk != instance.pk:
        raise ValueError('Population defaults require an unlocked municipal population dataset.')
    schema = dataset.schema
    if schema is None or schema.dimensions.exists():
        raise ValueError('Municipal population requires a dimensionless annual dataset.')
    metric = schema.metrics.get(name='population')
    if metric.unit != 'cap':
        raise ValueError('Population default metric must use cap.')
    state = ProviderDefaults.model_validate((dataset.spec or {}).get('provider_defaults', {}))
    points = {point.date.year: point for point in dataset.data_points.select_related('evidence').filter(metric=metric)}
    changed = 0
    seen: set[int] = set()
    for value in values:
        if value.year in seen:
            raise ValueError(f'Duplicate population default year {value.year}.')
        seen.add(value.year)
        key = str(value.year)
        cell = ProviderCell(value=Decimal(value.value), source_revision=source_revision, provenance=value.provenance)
        point = points.get(value.year)
        previous = state.cells.get(key)
        owned = key not in state.overrides and (point is None or _provider_owned(point, previous))
        if not owned:
            state.overrides.add(key)
        if owned and (point is None or previous != cell or point.value != cell.value):
            if point is None:
                point = DataPoint.objects.create(dataset=dataset, metric=metric, date=date(value.year, 1, 1), value=cell.value)
            else:
                point.value = cell.value
                point.save(update_fields=['value'])
            set_evidence(point, kind=DataEvidenceKind.PROVIDER_DEFAULT, user=None)
            _cite(point, cell, instance)
            changed += 1
        state.cells[key] = cell
    if not values:
        raise ValueError('Municipality has no provider population defaults.')
    forecast_years = [value.year for value in values if value.is_forecast]
    dataset.spec = {
        **(dataset.spec or {}),
        'provider_defaults': state.model_dump(mode='json'),
        'forecast_from': min(forecast_years) if forecast_years else None,
    }
    dataset.save(update_fields=['spec'])
    refresh_dataset_materialization(dataset)
    instance.invalidate_cache()
    return changed


@transaction.atomic
def seed_energy_defaults(  # noqa: C901
    instance: InstanceConfig,
    dataset: Dataset,
    values: list[EnergyDefault],
    *,
    source_revision: str,
) -> int:
    """Seed an untouched input grid once; an excluded municipality has no source rows."""
    dataset = Dataset.objects.select_for_update().get(pk=dataset.pk)
    if instance.is_locked or dataset.identifier != ENERGY_DATASET or dataset.scope_instance.pk != instance.pk:
        raise ValueError('Energy defaults require an unlocked municipal energy dataset.')
    if not values or dataset.data_points.filter(value__isnull=False).exists():
        return 0
    if dataset.data_points.filter(last_modified_by__isnull=False).exists():
        return 0
    schema = dataset.schema
    if schema is None:
        raise ValueError('Energy dataset lacks a schema.')
    metric = schema.metrics.get(name='Value')
    if metric.unit != 'MWh/a':
        raise ValueError('Energy defaults must use MWh/a.')
    dimensions = {
        scope.identifier: scope.dimension
        for scope in DimensionScope.objects.filter(
            dimension__in=schema.dimensions.values('dimension'),
            identifier__in=('sector', 'energy_carrier'),
        ).select_related('dimension')
    }
    if set(dimensions) != {'sector', 'energy_carrier'}:
        raise ValueError('Energy defaults require sector and energy_carrier dimensions.')
    sectors = {category.identifier: category for category in dimensions['sector'].categories.all()}
    carriers = {category.identifier: category for category in dimensions['energy_carrier'].categories.all()}
    quality = DataQualityLevel.objects.get(
        scheme__framework=instance.framework_config.framework,
        scheme__identifier='quality',
        scheme__version='1',
        identifier='C',
    )
    state = ProviderDefaults()
    changed = 0
    for value in values:
        key = f'{value.year}:{value.sector}:{value.energy_carrier}'
        if key in state.cells:
            raise ValueError(f'Duplicate energy default {key}.')
        categories = [sectors[value.sector], carriers[value.energy_carrier]]
        existing = dataset.data_points.filter(metric=metric, date=date(value.year, 1, 1))
        for category in categories:
            existing = existing.filter(dimension_categories=category)
        point = existing.first()
        cell = ProviderCell(value=Decimal(str(value.value)), source_revision=source_revision, provenance=value.provenance)
        if point is None:
            point = DataPoint.objects.create(dataset=dataset, metric=metric, date=date(value.year, 1, 1), value=cell.value)
            DataPointDimensionCategory.objects.bulk_create([
                DataPointDimensionCategory(data_point=point, dimension_category=category) for category in categories
            ])
        else:
            if not _provider_owned(point, None):
                raise ValueError('Energy grid contains municipal assertions; defaults cannot replace them.')
            point.value = cell.value
            point.save(update_fields=['value'])
        set_evidence(point, kind=DataEvidenceKind.PROVIDER_DEFAULT, quality_level=quality, user=None)
        _cite(point, cell, instance)
        state.cells[key] = cell
        changed += 1
    dataset.spec = {**(dataset.spec or {}), 'provider_defaults': state.model_dump(mode='json')}
    dataset.save(update_fields=['spec'])
    refresh_dataset_materialization(dataset)
    instance.invalidate_cache()
    return changed


def seed_activation_defaults(instance: InstanceConfig, local: dict[str, Dataset]) -> None:
    missing = {POPULATION_DATASET, ENERGY_DATASET} - local.keys()
    if missing:
        raise ValueError(f'Upgrade the published BISKO template before activation; missing municipal inputs: {sorted(missing)}')
    population, energy, revision = load_activation_sources(instance)
    organization = instance.organization
    assert organization is not None
    ags = organization.identifiers.get(namespace__identifier='ags').identifier
    values = [
        PopulationDefault(
            ags=ags, year=year, value=value, is_forecast=forecast, provenance=DefaultProvenance.model_validate_json(provenance)
        )
        for year, value, forecast, provenance in population
        .filter(pl.col('ags') == ags)
        .select('Year', 'population', 'Forecast', 'provenance')
        .iter_rows()
    ]
    seed_population_defaults(instance, local[POPULATION_DATASET], values, source_revision=revision)
    cells = [
        EnergyDefault(
            ags=ags,
            year=year,
            sector=sector,
            energy_carrier=carrier,
            value=value,
            provenance=DefaultProvenance.model_validate_json(provenance),
        )
        for year, sector, carrier, value, provenance in energy
        .filter(pl.col('ags') == ags)
        .select('Year', 'sector', 'energy_carrier', 'Value', 'provenance')
        .iter_rows()
    ]
    seed_energy_defaults(instance, local[ENERGY_DATASET], cells, source_revision=revision)
