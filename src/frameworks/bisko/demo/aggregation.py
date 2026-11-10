"""Deterministic synthetic inventories for complete administrative territories."""

import hashlib
import json
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from django.db import transaction
from pydantic import BaseModel, Field

from kausal_common.datasets.models import DataPoint, DataPointDimensionCategory, Dataset
from kausal_common.i18n.pydantic import set_i18n_context

from datasets.materialization import refresh_dataset_materialization
from frameworks import submissions
from frameworks.bisko.activation import activate_bisko_municipality
from frameworks.bisko.default_sources import ENERGY_DATASET
from frameworks.bisko.demo.inventories import configure_demo_definition, demo_grade
from frameworks.models import DataEvidenceKind, DataPointEvidence, DataQualityLevel, FrameworkConfig, SubmissionStatus
from orgs.models import OrganizationIdentifier

if TYPE_CHECKING:
    from frameworks.models import Framework
    from nodes.models import InstanceConfig
    from orgs.models import Organization


DEMO_KEY = 'bisko_aggregation_demo'
DEFAULT_DISTRICTS = ('03451', '12072')
DEFAULT_YEARS = (2022, 2023)
BASE_YEAR = 2023


class DemoRecipe(BaseModel):
    version: int = 1
    seed: int
    years: list[int]
    district: str
    template_revision: int


class DemoRecord(BaseModel):
    recipe: DemoRecipe
    fingerprint: str = Field(min_length=64, max_length=64)


def energy_factor(seed: int, ags: str, sector: str, carrier: str, year: int) -> Decimal:
    """Stable changes around the 2023 estimate; unrelated towns keep distinct trajectories."""
    digest = hashlib.sha256(f'{seed}:{ags}:{sector}:{carrier}'.encode()).digest()
    # Most energy use declines into 2023; some cells grow, and heat pumps grow faster.
    annual = Decimal(int.from_bytes(digest[:4]) % 1001 - 200) / Decimal(10000)
    if carrier in ('environmental_heat', 'solar_thermal'):
        annual = -abs(annual) - Decimal('0.02')
    return (Decimal(1) + annual) ** (BASE_YEAR - year)


def district_municipalities(framework: Framework, ars: str) -> list[Organization]:
    """Select AGS-bearing leaves through the actual imported district tree."""
    try:
        district = (
            OrganizationIdentifier.objects
            .select_related('organization__classification')
            .get(namespace__identifier='ars', identifier=ars)
            .organization
        )
    except OrganizationIdentifier.DoesNotExist as error:
        raise ValueError(f'Unknown district ARS {ars}; import the BKG tree first.') from error
    if district.classification is None or district.classification.identifier != 'de_district':
        raise ValueError(f'{ars} is not a Landkreis.')
    roots = framework.organization_roots.values_list('organization__path', flat=True)
    if not any(district.path.startswith(path) for path in roots):
        raise ValueError(f'{ars} is outside the framework.')
    identifiers = (
        OrganizationIdentifier.objects
        .filter(
            namespace__identifier='ags',
            organization__path__startswith=district.path,
            organization__classification__identifier='de_municipality',
        )
        .select_related('organization')
        .order_by('identifier')
    )
    municipalities = [identifier.organization for identifier in identifiers]
    if not municipalities:
        raise ValueError(f'{ars} contains no municipalities.')
    return municipalities


def inventory_fingerprint(instance: InstanceConfig) -> str:
    """Detect edits to any local input or local model settings before accepting a repeat run."""
    datasets = Dataset.objects.for_instance_config(instance).order_by('uuid')
    content: list[object] = [instance.ensure_spec().model_dump(mode='json')]
    for dataset in datasets:
        points = dataset.data_points.select_related('metric', 'evidence').prefetch_related('dimension_categories')
        cells = []
        for point in points:
            evidence = getattr(point, 'evidence', None)
            cells.append((
                str(point.metric.uuid),
                point.date.isoformat(),
                str(point.value),
                sorted(str(category.uuid) for category in point.dimension_categories.all()),
                evidence.kind if evidence else None,
                evidence.quality_level_id if evidence else None,
                point.last_modified_by_id,
            ))
        content.append((str(dataset.uuid), sorted(cells), {k: v for k, v in (dataset.spec or {}).items() if k != DEMO_KEY}))
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _repeat(instance: InstanceConfig, recipe: DemoRecipe) -> bool:
    dataset = Dataset.objects.for_instance_config(instance).filter(identifier=ENERGY_DATASET).first()
    raw = (dataset.spec or {}).get(DEMO_KEY) if dataset is not None else None
    if raw is None:
        raise ValueError(f'{instance.identifier}: existing instance was not created by this demo seeder.')
    record = DemoRecord.model_validate(raw)
    if record.recipe != recipe:
        raise ValueError(f'{instance.identifier}: demo recipe changed; refusing to overwrite existing inventories.')
    if record.fingerprint != inventory_fingerprint(instance):
        raise ValueError(f'{instance.identifier}: local data or model settings changed since demo generation.')
    finals = instance.submissions.filter(status=SubmissionStatus.FINAL)
    if set(finals.values_list('period_start', flat=True)) != set(recipe.years) or finals.count() != len(recipe.years):
        raise ValueError(f'{instance.identifier}: final submissions differ from the demo recipe.')
    if instance.submissions.exclude(status=SubmissionStatus.FINAL).exists():
        raise ValueError(f'{instance.identifier}: additional submissions exist.')
    return False


def _energy(instance: InstanceConfig, recipe: DemoRecipe, ags: str) -> Dataset:
    dataset = Dataset.objects.for_instance_config(instance).get(identifier=ENERGY_DATASET)
    points = list(
        dataset.data_points
        .filter(date__year=BASE_YEAR, value__isnull=False)
        .select_related('metric')
        .prefetch_related('dimension_categories__dimension')
    )
    if len(points) != 60:
        raise ValueError(f'{instance.identifier}: expected 60 provider energy cells in {BASE_YEAR}, got {len(points)}.')
    levels = {
        level.identifier: level
        for level in DataQualityLevel.objects.filter(
            scheme__framework=instance.framework_config.framework,
            scheme__identifier='quality',
            scheme__version='1',
        )
    }
    if set(levels) != {'A', 'B', 'C', 'D'}:
        raise ValueError('Expected the BISKO A/B/C/D quality scheme.')
    rows: list[DataPoint] = []
    coordinates = []
    grades = []
    for year in recipe.years:
        for point in points:
            categories = list(point.dimension_categories.all())
            names = {category.dimension.name: category.identifier for category in categories}
            sector, carrier = names['Sektoren'], names['Energieträger']
            if sector is None or carrier is None or point.value is None:
                raise ValueError('Provider energy cell lacks coordinates or a value.')
            value = (point.value * energy_factor(recipe.seed, ags, sector, carrier, year)).quantize(Decimal('0.000001'))
            rows.append(DataPoint(dataset=dataset, metric=point.metric, date=date(year, 1, 1), value=value))
            coordinates.append(categories)
            grades.append(levels[demo_grade(sector, carrier)])
    dataset.data_points.all().delete()
    rows = DataPoint.objects.bulk_create(rows)
    DataPointDimensionCategory.objects.bulk_create([
        DataPointDimensionCategory(data_point=point, dimension_category=category)
        for point, categories in zip(rows, coordinates, strict=True)
        for category in categories
    ])
    DataPointEvidence.objects.bulk_create([
        DataPointEvidence(data_point=point, kind=DataEvidenceKind.ESTIMATED, quality_level=grade)
        for point, grade in zip(rows, grades, strict=True)
    ])
    # These generated cells no longer belong to the provider refresh process.
    dataset.spec = {key: value for key, value in (dataset.spec or {}).items() if key != 'provider_defaults'}
    dataset.save(update_fields=['spec'])
    refresh_dataset_materialization(dataset)
    return dataset


@transaction.atomic
def seed_municipality(framework: Framework, municipality: Organization, *, district: str, years: list[int], seed: int) -> bool:
    """Create and genuinely finalise the municipality's inventories, or verify an unchanged repeat."""
    template = framework.template_instance
    if template is None or template.live_revision_id is None:
        raise ValueError('Publish the BISKO template before seeding inventories.')
    recipe = DemoRecipe(seed=seed, years=years, district=district, template_revision=template.live_revision_id)
    existing = FrameworkConfig.objects.filter(framework=framework, instance_config__organization=municipality)
    if existing.count() > 1:
        raise ValueError(f'{municipality.name}: multiple BISKO instances.')
    config = existing.select_related('instance_config').first()
    if config is not None:
        return _repeat(config.instance_config, recipe)
    with set_i18n_context('de', []):
        config, created = activate_bisko_municipality(framework, municipality)
        if not created:
            raise ValueError(f'{municipality.name}: concurrent activation; retry.')
        instance = config.instance_config
        if instance.is_locked:
            raise ValueError(f'{instance.identifier}: instance is locked.')
        ags = municipality.identifiers.get(namespace__identifier='ags').identifier
        energy = _energy(instance, recipe, ags)
        configure_demo_definition(instance, years)
        # Activation opens one submission; discard only that new, empty lifecycle record.
        for submission in instance.submissions.all():
            submissions.discard(submission)
        for year in years:
            submission = submissions.create_submission(instance, period_start=year, user=None)
            submission = submissions.request_review(submission, user=None, note='Synthetic district aggregation demo')
            submissions.finalise(submission, user=None)
        instance.refresh_from_db()
        record = DemoRecord(recipe=recipe, fingerprint=inventory_fingerprint(instance))
        energy.refresh_from_db()
        energy.spec = {**(energy.spec or {}), DEMO_KEY: record.model_dump(mode='json')}
        energy.save(update_fields=['spec'])
    return True
