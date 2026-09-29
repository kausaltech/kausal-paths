"""
Build and load temporary synthetic BISKO demo cells from dashboard Parquet.

Produce the input in kausal-importers:
    uv run kausal-importers germany ksp-dashboards --output /tmp/ksp-dashboards.parquet

Preview in kausal-paths, then add --apply to persist:
    python manage.py load_bisko_demo /tmp/ksp-dashboards.parquet

All demo-specific processing lives here so this command can be removed as one file.
The split uses the local mainz-bisko instance's mainz/final_energy dataset for 2018
as its prior. Rows before 2000 are excluded. Existing matching demo cells are skipped;
different or manually supplied values are rejected. Transport uses the model's
shared ifeu defaults and is not copied into municipal observation datasets here.
The provisioned district and municipal test accounts also receive editor grants
on their state's demo city. Provision the test accounts before running this command.
"""

import hashlib
import math
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

import polars as pl

from kausal_common.datasets.models import (
    DataPoint,
    DataPointDimensionCategory,
    Dataset,
    DatasetSourceReference,
    DataSource,
)
from kausal_common.people.models import ObjectRole

from datasets.materialization import refresh_dataset_materialization
from frameworks.bisko.activation import activate_bisko_municipality
from frameworks.models import DataEvidenceKind, DataPointEvidence, Framework, OrganizationAccessGrant
from nodes.models import InstanceConfig
from orgs.models import OrganizationIdentifier
from users.models import User

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from orgs.models import Organization


TARGET_ARS = ('034040000000', '071370203203', '120630252252')
SOURCE_NAME = 'Synthetic BISKO demo cells from Klimaschutz-Planer dashboards'


SECTORS = {
    'private_households': 'HE_Gesamt',
    'commerce_trade_services': 'GE_Gesamt',
    'industry': 'IE_Gesamt',
    'municipal_facilities': 'KE_Gesamt',
}
CARRIERS = {
    'electricity': ('SUM_Strom',),
    'heating_electricity': ('SUM_Heizstrom',),
    'natural_gas': ('SUM_Erdgas',),
    'heating_oil': ('SUM_Heizoel',),
    'district_heating': ('SUM_Fernwaerme',),
    'local_heating': ('SUM_Nahwaerme',),
    'biomass': ('SUM_Biomasse',),
    'solar_thermal': ('SUM_Solarthermie',),
    'environmental_heat': ('SUM_Umweltwaerme',),
    'hard_coal': ('SUM_Steinkohle',),
    'brown_coal': ('SUM_Braunkohle',),
    'propane': ('SUM_Fluessiggas',),
    'biogas': ('SUM_Biogas',),
    'other_conventional': ('SUM_SonstigeKo',),
    'other_renewables': ('SUM_SonstigeEE',),
}


def mainz_prior() -> dict[tuple[str, str], float]:
    """Read the existing Mainz cell values; the dashboard itself has no cell split."""
    instance = InstanceConfig.objects.get(identifier='mainz-bisko')
    dataset = Dataset.objects.for_instance_config(instance).get(identifier='mainz/final_energy')
    points = dataset.data_points.filter(date=date(2018, 1, 1), metric__name='Value').prefetch_related(
        'dimension_categories__dimension'
    )
    prior: dict[tuple[str, str], float] = {}
    for point in points:
        categories = {category.dimension.name: category.identifier for category in point.dimension_categories.all()}
        sector, carrier = categories.get('Sektoren'), categories.get('Energieträger')
        if sector is None or carrier is None:
            raise ValueError('Mainz 2018 prior cell has no sector or carrier')
        key = (sector, carrier)
        if key in prior or key[0] not in SECTORS or key[1] not in CARRIERS or point.value is None:
            raise ValueError(f'Unexpected Mainz 2018 prior cell {key}')
        prior[key] = float(point.value)
    expected = {(sector, carrier) for sector in SECTORS for carrier in CARRIERS}
    if set(prior) != expected:
        raise ValueError(f'Mainz 2018 prior has {len(prior)} cells, expected {len(expected)}')
    if any(not math.isfinite(value) or value < 0 for value in prior.values()):
        raise ValueError('Mainz 2018 prior contains an invalid value')
    return prior


def dashboard_margins(frame: pl.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    fields = {key: value for key, value in frame.select('field_key', 'value').iter_rows() if value is not None}
    needed = set(SECTORS.values()) | {key for keys in CARRIERS.values() for key in keys}
    if missing := needed - fields.keys():
        raise ValueError(f'Dashboard missing energy fields: {sorted(missing)}')
    if any(not math.isfinite(fields[key]) or fields[key] < 0 for key in needed):
        raise ValueError('Dashboard has an invalid energy value')
    rows = {sector: fields[key] for sector, key in SECTORS.items()}
    columns = {carrier: sum(fields[key] for key in keys) for carrier, keys in CARRIERS.items()}
    row_total, column_total = sum(rows.values()), sum(columns.values())
    if abs(row_total - column_total) > max(0.01, row_total * 1e-8):
        raise ValueError(f'Dashboard margins disagree: sectors={row_total}, carriers={column_total}')
    return rows, columns


def fit_margins(
    prior: dict[tuple[str, str], float], rows: dict[str, float], columns: dict[str, float]
) -> dict[tuple[str, str], float]:
    """Fit margins iteratively with a small positive prior for unseen combinations."""
    if set(rows) != set(SECTORS) or set(columns) != set(CARRIERS):
        raise ValueError('Unexpected sector or carrier marginals')
    total = sum(rows.values())
    if total == 0:
        return {(sector, carrier): 0.0 for sector in rows for carrier in columns}
    if abs(total - sum(columns.values())) > max(0.01, total * 1e-8):
        raise ValueError('Sector and carrier marginals disagree')
    floor = max(sum(prior.values()) * 1e-9 / len(prior), 1e-12)
    cells = {(sector, carrier): max(prior[sector, carrier], floor) for sector in rows for carrier in columns}
    for _ in range(1000):
        for sector, target in rows.items():
            current = sum(cells[sector, carrier] for carrier in columns)
            factor = target / current if target else 0.0
            for carrier in columns:
                cells[sector, carrier] *= factor
        for carrier, target in columns.items():
            current = sum(cells[sector, carrier] for sector in rows)
            factor = target / current if target else 0.0
            for sector in rows:
                cells[sector, carrier] *= factor
        error = max(abs(sum(cells[sector, carrier] for carrier in columns) - target) for sector, target in rows.items())
        if error <= max(1e-5, total * 1e-10):
            return cells
    raise ValueError(f'Margin fitting did not converge; residual={error}')


def build_cells(frame: pl.DataFrame, *, min_year: int) -> pl.DataFrame:
    required = {'ags', 'year', 'field_key', 'unit', 'value'}
    if missing := required - set(frame.columns):
        raise ValueError(f'Dashboard Parquet lacks columns: {sorted(missing)}')
    # The importer currently calls the 12-digit dashboard/ARS key "ags".
    target = frame.filter(pl.col('ags').is_in(TARGET_ARS) & (pl.col('year') >= min_year))
    if set(target['ags']) != set(TARGET_ARS):
        raise ValueError('Dashboard Parquet does not contain all three target cities')
    if target.select('ags', 'year', 'field_key').unique().height != target.height:
        raise ValueError('Dashboard Parquet contains duplicate fields')
    prior = mainz_prior()
    records: list[dict[str, str | int | float]] = []
    needed = set(SECTORS.values()) | {key for keys in CARRIERS.values() for key in keys}
    for (ars, year), group in target.group_by('ags', 'year'):
        if set(group.filter(pl.col('field_key').is_in(needed))['unit']) != {'MWh'}:
            raise ValueError(f'{ars} {year}: dashboard energy fields must use MWh')
        sectors, carriers = dashboard_margins(group)
        cells = fit_margins(prior, sectors, carriers)
        records.extend(
            {
                'ars': ars,
                'year': year,
                'sector': sector,
                'energy_carrier': carrier,
                'value_mwh_per_year': value,
            }
            for (sector, carrier), value in cells.items()
        )
    return pl.DataFrame(records).sort('ars', 'year', 'sector', 'energy_carrier')


def already_loaded(instance: InstanceConfig, dataset: Dataset, city: pl.DataFrame) -> bool:
    valued = (
        dataset.data_points
        .filter(date__year__in=city['year'].unique().to_list())
        .filter(value__isnull=False)
        .prefetch_related('dimension_categories__dimension', 'evidence')
    )
    if valued.exists():
        # Permit a repeat import only when every stored cell is the same estimated demo value.
        incoming = {
            (year, sector, carrier): value
            for year, sector, carrier, value in city.select('year', 'sector', 'energy_carrier', 'value_mwh_per_year').iter_rows()
        }
        stored: dict[tuple[int, str | None, str | None], float] = {}
        for point in valued:
            categories = {category.dimension.name: category.identifier for category in point.dimension_categories.all()}
            key = (point.date.year, categories.get('Sektoren'), categories.get('Energieträger'))
            if key in stored or not hasattr(point, 'evidence') or point.evidence.kind != DataEvidenceKind.ESTIMATED:
                raise ValueError(f'{instance.identifier}: existing cells are not the same demo import')
            assert point.value is not None
            stored[key] = float(point.value)
        if set(stored) != set(incoming) or any(
            not math.isclose(value, incoming[key], abs_tol=1e-6, rel_tol=1e-10) for key, value in stored.items()
        ):
            raise ValueError(f'{instance.identifier}: energy dataset already has different values in target years')
        if not dataset.source_references.filter(data_source__name=SOURCE_NAME).exists():
            raise ValueError(f'{instance.identifier}: existing values lack the demo source')
        return True
    return False


def load_city(instance: InstanceConfig, city: pl.DataFrame, *, source_digest: str) -> int:
    dataset = Dataset.objects.for_instance_config(instance).get(identifier='kommune/endenergieverbrauch')
    if dataset.schema is None:
        raise ValueError(f'{instance.identifier}: energy dataset has no schema')
    years = city['year'].unique().to_list()
    existing = dataset.data_points.filter(date__year__in=years)
    if already_loaded(instance, dataset, city):
        return 0
    if (
        existing.filter(evidence__isnull=False).exists()
        or existing.filter(source_references__isnull=False).exists()
        or existing.filter(comments__isnull=False).exists()
    ):
        raise ValueError(f'{instance.identifier}: empty cells already have evidence, sources or comments')
    # Activation creates blank year-slot cells. Replace only those empty placeholders.
    existing.delete()
    dimensions = {item.dimension.name: item.dimension for item in dataset.schema.dimensions.select_related('dimension')}
    if set(dimensions) != {'Sektoren', 'Energieträger'}:
        raise ValueError(f'{instance.identifier}: unexpected energy dimensions')
    sector_categories = {category.identifier: category for category in dimensions['Sektoren'].categories.all()}
    carrier_categories = {category.identifier: category for category in dimensions['Energieträger'].categories.all()}
    if not set(SECTORS) <= sector_categories.keys() or not set(CARRIERS) <= carrier_categories.keys():
        raise ValueError(f'{instance.identifier}: energy categories are incomplete')
    metric = dataset.schema.metrics.get(name='Value')
    if metric.unit != 'MWh/a':
        raise ValueError(f'{instance.identifier}: expected MWh/a, got {metric.unit}')
    records = list(city.select('year', 'sector', 'energy_carrier', 'value_mwh_per_year').iter_rows())
    points = DataPoint.objects.bulk_create([
        DataPoint(dataset=dataset, metric=metric, date=date(year, 1, 1), value=Decimal(str(value)))
        for year, _sector, _carrier, value in records
    ])
    DataPointDimensionCategory.objects.bulk_create([
        link
        for point, (_year, sector, carrier, _value) in zip(points, records, strict=True)
        for link in (
            DataPointDimensionCategory(data_point=point, dimension_category=sector_categories[sector]),
            DataPointDimensionCategory(data_point=point, dimension_category=carrier_categories[carrier]),
        )
    ])
    DataPointEvidence.objects.bulk_create([
        DataPointEvidence(data_point=point, kind=DataEvidenceKind.ESTIMATED) for point in points
    ])
    content_type = ContentType.objects.get_for_model(instance)
    source = DataSource.objects.create(
        scope_content_type=content_type,
        scope_id=instance.pk,
        name=SOURCE_NAME,
        edition=f'Parquet SHA-256 {source_digest[:16]}',
        authority='Demo data generated locally',
        description=(
            'Synthetic sector-by-carrier cells. Published Klimaschutz-Planer dashboard sector and carrier totals '
            'were fitted using Mainz 2018 as a prior. These are estimates for demonstration, not reported inventory cells. '
            f'Input Parquet SHA-256: {source_digest}'
        ),
        url='https://dashboard-be.klimaschutz-planer.de/v1/communes',
    )
    DatasetSourceReference.objects.create(dataset=dataset, data_source=source)
    refresh_dataset_materialization(dataset)
    return len(points)


def grant_demo_access(framework: Framework, organization: Organization, users: list[User]) -> list[str]:
    """Add city editor access to the provisioned test accounts without replacing their original grants."""
    results: list[str] = []
    for user in users:
        grant, created = OrganizationAccessGrant.objects.get_or_create(
            framework=framework, organization=organization, user=user, defaults={'role': ObjectRole.EDITOR}
        )
        if grant.suspended_at is not None:
            raise CommandError(f'{user.email}: demo-city grant is suspended; reactivate it explicitly.')
        if grant.role not in (ObjectRole.EDITOR, ObjectRole.ADMIN):
            grant.role = ObjectRole.EDITOR
            grant.save(update_fields=['role', 'last_modified_at'])
        results.append(f'{user.email}: {grant.role} on {organization.name}, grant_created={created}')
    return results


class Command(BaseCommand):
    help = 'Load demo energy data and grant test-account access for Osnabrück, Bendorf and Rathenow; dry run by default.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('staging_file', type=Path, help='Parquet produced by kausal-importers germany ksp-dashboards.')
        parser.add_argument('--min-year', type=int, default=2000)
        parser.add_argument('--apply', action='store_true', help='Persist the three demo instances, values and test-user grants.')

    def handle(self, *args: Any, **options: Any) -> None:
        source: Path = options['staging_file']
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        frame = build_cells(pl.read_parquet(source), min_year=options['min_year'])
        framework = Framework.objects.get(identifier='bisko')
        identifiers = OrganizationIdentifier.objects.filter(
            namespace__identifier='ars',
            identifier__in=TARGET_ARS,
            organization__classification__identifier='de_municipality',
        ).select_related('organization')
        organizations = {identifier.identifier: identifier.organization for identifier in identifiers}
        if set(organizations) != set(TARGET_ARS):
            raise CommandError(f'Missing municipal organizations: {sorted(set(TARGET_ARS) - organizations.keys())}')
        results: list[str] = []
        with transaction.atomic():
            test_users: dict[str, list[User]] = {}
            for ars in TARGET_ARS:
                test_users[ars] = []
                for level in ('district-editor', 'municipality-editor'):
                    email = f'bisko-{ars[:2]}-{level}.fake@kausal.tech'
                    try:
                        user = User.objects.select_for_update().get(email__iexact=email)
                    except User.DoesNotExist as error:
                        raise CommandError(f'Missing test account {email}; run provision_bisko_test_accounts first.') from error
                    if not user.is_active:
                        raise CommandError(f'Test account {email} is inactive; reactivate it explicitly.')
                    test_users[ars].append(user)
            for ars in TARGET_ARS:
                city = frame.filter(pl.col('ars') == ars)
                config, created = activate_bisko_municipality(framework, organizations[ars])
                count = load_city(config.instance_config, city, source_digest=source_digest)
                results.append(f'{config.instance_config.identifier}: {count} new estimated cells, created={created}')
                results.extend(grant_demo_access(framework, organizations[ars], test_users[ars]))
            if not options['apply']:
                transaction.set_rollback(True)
        for result in results:
            self.stdout.write(result)
        if not options['apply']:
            self.stdout.write('Dry run: all database changes rolled back.')
