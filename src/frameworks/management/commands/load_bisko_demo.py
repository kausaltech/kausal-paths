"""
Build and load temporary synthetic BISKO demo cells from dashboard Parquet.

Produce the input in kausal-importers:
    uv run kausal-importers germany ksp-dashboards --output /tmp/ksp-dashboards.parquet

Preview in kausal-paths, then add --apply to persist:
    python manage.py load_bisko_demo /tmp/ksp-dashboards.parquet

For a deployment without the reference instance, export the prior locally:
    python manage.py load_bisko_demo --export-prior /tmp/mainz-prior.parquet
Then copy that file to the deployment and use it:
    python manage.py load_bisko_demo /tmp/ksp-dashboards.parquet --prior /tmp/mainz-prior.parquet --apply

Repair definitions of previously loaded demos without the original Parquet:
    python manage.py load_bisko_demo --reconcile-existing
    python manage.py load_bisko_demo --reconcile-existing --apply

Demo commerce and municipal facility cells are disjoint, so their local overlap
parameter is false. Their inventory calendar names only the imported years.
The synthetic estimates carry invented quality grades (`DEMO_GRADES`) so the demo can show
graded data; the grades describe no real source.
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
from django.utils import timezone

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
from frameworks.evidence import quality_schemes_for_dataset
from frameworks.models import DataEvidenceKind, DataPointEvidence, DataQualityLevel, Framework, OrganizationAccessGrant
from nodes.defs.instance_defs import YearsSpec
from nodes.instance_serialization import build_instance_snapshot
from nodes.models import InstanceConfig
from orgs.models import OrganizationIdentifier
from params.base import ParameterOwner
from params.param import BoolParameter
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

DEMO_GRADES = {
    'electricity': 'A',
    'heating_electricity': 'A',
    'natural_gas': 'A',
    'district_heating': 'A',
    'local_heating': 'B',
    'environmental_heat': 'B',
    'heating_oil': 'C',
    'biomass': 'C',
    'solar_thermal': 'C',
    'hard_coal': 'D',
    'brown_coal': 'D',
    'propane': 'D',
    'biogas': 'D',
    'other_conventional': 'D',
    'other_renewables': 'D',
}
"""
Invented grade per carrier, following how such a cell is typically sourced.

Grid-bound carriers come from network operators, which is also what the BISKO grade-A
requirement on them expects; municipal facilities are graded A in every carrier, since the
municipality holds its own bills. A cell's grade does not vary by year.
"""


def demo_grade(sector: str, carrier: str) -> str:
    return 'A' if sector == 'municipal_facilities' else DEMO_GRADES[carrier]


def grade_demo_cells(dataset: Dataset) -> int:
    """Assign the invented grades to the estimated demo cells; the caller refreshes the materialization."""
    levels: dict[str, DataQualityLevel] = {}
    for level in DataQualityLevel.objects.filter(scheme__in=quality_schemes_for_dataset(dataset)):
        if level.identifier in levels:
            raise ValueError(f'{dataset.identifier}: quality level {level.identifier} is ambiguous')
        levels[level.identifier] = level
    if missing := set(DEMO_GRADES.values()) - levels.keys():
        raise ValueError(f'{dataset.identifier}: no quality levels {sorted(missing)}')
    evidence = (
        DataPointEvidence.objects
        .filter(data_point__dataset=dataset, data_point__metric__name='Value', kind=DataEvidenceKind.ESTIMATED)
        .select_related('data_point')
        .prefetch_related('data_point__dimension_categories__dimension')
    )
    now = timezone.now()
    changed: list[DataPointEvidence] = []
    for item in evidence:
        categories = {category.dimension.name: category.identifier for category in item.data_point.dimension_categories.all()}
        sector, carrier = categories.get('Sektoren'), categories.get('Energieträger')
        if sector is None or carrier is None:
            raise ValueError(f'{dataset.identifier}: demo cell has no sector or carrier')
        level = levels[demo_grade(sector, carrier)]
        if item.quality_level_id != level.pk:
            item.quality_level = level
            item.last_modified_at = now
            changed.append(item)
    DataPointEvidence.objects.bulk_update(changed, ['quality_level', 'last_modified_at'])
    return len(changed)


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
    validate_prior(prior)
    return prior


def validate_prior(prior: dict[tuple[str, str], float]) -> None:
    """Require a complete, finite, nonnegative reference grid with some energy."""
    expected = {(sector, carrier) for sector in SECTORS for carrier in CARRIERS}
    if set(prior) != expected:
        raise CommandError(f'Prior must contain exactly the {len(expected)} expected sector/carrier cells')
    if any(not math.isfinite(value) or value < 0 for value in prior.values()):
        raise CommandError('Prior contains a negative or nonfinite value')
    if sum(prior.values()) <= 0:
        raise CommandError('Prior must contain some positive energy')


def read_prior(path: Path) -> dict[tuple[str, str], float]:
    """Read the exported Mainz 2018 grid without accessing the reference instance."""
    if path.suffix.lower() == '.csv':
        frame = pl.read_csv(path)
    elif path.suffix.lower() == '.parquet':
        frame = pl.read_parquet(path)
    else:
        raise CommandError('Prior file must be .csv or .parquet')
    required = {'sector', 'energy_carrier', 'value_mwh_per_year', 'year', 'unit'}
    if missing := required - set(frame.columns):
        raise CommandError(f'Prior file lacks columns: {sorted(missing)}')
    if set(frame['year']) != {2018} or set(frame['unit']) != {'MWh/a'}:
        raise CommandError('Prior file must contain the 2018 reference grid in MWh/a')
    prior: dict[tuple[str, str], float] = {}
    for sector, carrier, value in frame.select('sector', 'energy_carrier', 'value_mwh_per_year').iter_rows():
        key = (sector, carrier)
        if key in prior or value is None:
            raise CommandError(f'Duplicate or empty prior cell: {key}')
        prior[key] = float(value)
    validate_prior(prior)
    return prior


def export_prior(path: Path) -> None:
    """Export only the reference grid; leave demo instances and accounts untouched."""
    if path.suffix.lower() not in ('.csv', '.parquet'):
        raise CommandError('Export file must be .csv or .parquet')
    if path.exists():
        raise CommandError(f'Refusing to overwrite {path}')
    prior = mainz_prior()
    frame = pl.DataFrame([
        {'sector': sector, 'energy_carrier': carrier, 'value_mwh_per_year': value, 'year': 2018, 'unit': 'MWh/a'}
        for (sector, carrier), value in sorted(prior.items())
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == '.csv':
        frame.write_csv(path)
    else:
        frame.write_parquet(path)


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


def build_cells(frame: pl.DataFrame, *, min_year: int, prior: dict[tuple[str, str], float] | None = None) -> pl.DataFrame:
    required = {'ags', 'year', 'field_key', 'unit', 'value'}
    if missing := required - set(frame.columns):
        raise ValueError(f'Dashboard Parquet lacks columns: {sorted(missing)}')
    # The importer currently calls the 12-digit dashboard/ARS key "ags".
    target = frame.filter(pl.col('ags').is_in(TARGET_ARS) & (pl.col('year') >= min_year))
    if set(target['ags']) != set(TARGET_ARS):
        raise ValueError('Dashboard Parquet does not contain all three target cities')
    if target.select('ags', 'year', 'field_key').unique().height != target.height:
        raise ValueError('Dashboard Parquet contains duplicate fields')
    if prior is None:
        prior = mainz_prior()
    validate_prior(prior)
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


def load_city(instance: InstanceConfig, city: pl.DataFrame, *, source_digest: str, prior_digest: str | None = None) -> int:
    dataset = Dataset.objects.for_instance_config(instance).get(identifier='kommune/endenergieverbrauch')
    if dataset.schema is None:
        raise ValueError(f'{instance.identifier}: energy dataset has no schema')
    years = city['year'].unique().to_list()
    existing = dataset.data_points.filter(date__year__in=years)
    if already_loaded(instance, dataset, city):
        configure_demo_definition(instance, sorted(set(years)))
        if grade_demo_cells(dataset):
            refresh_dataset_materialization(dataset)
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
    grade_demo_cells(dataset)
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
            f'Input Parquet SHA-256: {source_digest}. '
            + (f'Prior file SHA-256: {prior_digest}' if prior_digest else 'Prior read from local mainz/final_energy 2018.')
        ),
        url='https://dashboard-be.klimaschutz-planer.de/v1/communes',
    )
    DatasetSourceReference.objects.create(dataset=dataset, data_source=source)
    refresh_dataset_materialization(dataset)
    configure_demo_definition(instance, sorted(set(years)))
    return len(points)


def configure_demo_definition(instance: InstanceConfig, inventory_years: list[int]) -> None:
    """Declare the demo's separate sectors and actual inventory calendar without assigning quality grades."""
    if instance.is_locked:
        raise ValueError(f'{instance.identifier}: instance is locked')
    if not inventory_years:
        raise ValueError(f'{instance.identifier}: demo has no inventory years')
    effective = build_instance_snapshot(instance)
    identifier = 'municipal_facilities_included_in_commerce'
    parameter = next((item for item in effective.spec.params if item.local_id == identifier), None)
    if not isinstance(parameter, BoolParameter) or parameter.owner != ParameterOwner.INSTANCE:
        raise ValueError(f'{instance.identifier}: demo requires an instance-owned {identifier} parameter')
    default = next((scenario for scenario in effective.spec.scenarios if scenario.default), None)
    if default is None:
        raise ValueError(f'{instance.identifier}: no default scenario')
    spec = instance.ensure_spec().model_copy(deep=True)
    local = spec.local_scenario(default.id)
    local.param_values[identifier] = parameter.clean(value=False)
    local.parameter_types[identifier] = parameter.type
    years = spec.years.model_dump(mode='python')
    first, last = min(inventory_years), max(inventory_years)
    years.update(min_historical=first, max_historical=last, skipped=sorted(set(range(first, last + 1)) - set(inventory_years)))
    if years['reference'] not in inventory_years:
        years['reference'] = first
    spec.years = YearsSpec.model_validate(years)
    if spec.model_dump(mode='json') != instance.ensure_spec().model_dump(mode='json'):
        instance.spec = spec
        instance.save(update_fields=['spec'])
        instance.invalidate_cache()


def existing_demo_years(dataset: Dataset) -> list[int]:
    """Only repair complete synthetic demo grids; other city data needs its own declaration."""
    if not dataset.source_references.filter(data_source__name=SOURCE_NAME).exists():
        raise ValueError(f'{dataset.identifier}: not a synthetic dashboard demo dataset')
    points = dataset.data_points.filter(metric__name='Value', value__isnull=False)
    if points.exclude(evidence__kind=DataEvidenceKind.ESTIMATED).exists():
        raise ValueError(f'{dataset.identifier}: demo contains values without estimated provenance')
    keys = []
    for point in points.prefetch_related('dimension_categories__dimension'):
        categories = {category.dimension.name: category.identifier for category in point.dimension_categories.all()}
        keys.append((point.date.year, categories.get('Sektoren'), categories.get('Energieträger')))
    years = sorted({key[0] for key in keys})
    expected = {(year, sector, carrier) for year in years for sector in SECTORS for carrier in CARRIERS}
    if not years or len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f'{dataset.identifier}: demo sector/carrier grid is incomplete or duplicated')
    return years


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
        parser.add_argument('staging_file', nargs='?', type=Path, help='Parquet from kausal-importers germany ksp-dashboards.')
        parser.add_argument('--prior', type=Path, help='Exported Mainz 2018 CSV or Parquet; bypasses the local Mainz instance.')
        parser.add_argument('--export-prior', type=Path, help='Export the local Mainz 2018 grid to CSV or Parquet and exit.')
        parser.add_argument('--min-year', type=int, default=2000)
        parser.add_argument('--apply', action='store_true', help='Persist the demo changes; otherwise roll them back.')
        parser.add_argument(
            '--reconcile-existing',
            action='store_true',
            help='Repair only the definitions and demo grades of existing synthetic demo instances; no staging file needed.',
        )
        parser.add_argument(
            '--instance',
            action='append',
            default=[],
            help='Limit --reconcile-existing to a named BISKO demo instance; repeat for several.',
        )

    def handle(self, *args: Any, **options: Any) -> None:
        if options['reconcile_existing']:
            if any(options[key] is not None for key in ('staging_file', 'prior', 'export_prior')):
                raise CommandError('--reconcile-existing cannot be combined with input or export files')
            self._reconcile_existing(options['instance'], apply=options['apply'])
            return
        if options['instance']:
            raise CommandError('--instance requires --reconcile-existing')
        if options['export_prior'] is not None:
            if options['staging_file'] is not None or options['prior'] is not None or options['apply']:
                raise CommandError('--export-prior cannot be combined with a staging file, --prior or --apply')
            export_prior(options['export_prior'])
            self.stdout.write(f'Exported 60 Mainz 2018 prior cells to {options["export_prior"]}')
            return
        if options['staging_file'] is None:
            raise CommandError('Provide a dashboard Parquet file, or use --export-prior to export the reference grid')
        self._load_demo(options['staging_file'], options['prior'], min_year=options['min_year'], apply=options['apply'])

    def _reconcile_existing(self, identifiers: list[str], *, apply: bool) -> None:  # noqa: C901
        results = []
        with transaction.atomic():
            instances = InstanceConfig.objects.select_for_update().filter(framework_config__framework__identifier='bisko')
            if identifiers:
                instances = instances.filter(identifier__in=identifiers)
            found = set()
            for instance in instances:
                dataset = Dataset.objects.for_instance_config(instance).filter(identifier='kommune/endenergieverbrauch').first()
                if dataset is None or not dataset.source_references.filter(data_source__name=SOURCE_NAME).exists():
                    if identifiers:
                        raise CommandError(f'{instance.identifier}: not a loaded synthetic dashboard demo')
                    continue
                try:
                    years = existing_demo_years(dataset)
                    configure_demo_definition(instance, years)
                    graded = grade_demo_cells(dataset)
                except ValueError as error:
                    raise CommandError(str(error)) from error
                if graded:
                    refresh_dataset_materialization(dataset)
                found.add(instance.identifier)
                results.append(
                    f'{instance.identifier}: municipal overlap=false; inventory years={years}; '
                    f'reference year={instance.ensure_spec().years.reference}; grades changed on {graded} cells'
                )
            if missing := set(identifiers) - found:
                raise CommandError(f'Not a BISKO demo instance: {", ".join(sorted(missing))}')
            if not found:
                raise CommandError('No loaded synthetic dashboard demos found')
            if not apply:
                transaction.set_rollback(True)
        for result in results:
            self.stdout.write(result)
        if not apply:
            self.stdout.write('Dry run: definition changes rolled back.')

    def _load_demo(self, source: Path, prior_path: Path | None, *, min_year: int, apply: bool) -> None:
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        prior = read_prior(prior_path) if prior_path is not None else mainz_prior()
        prior_digest = hashlib.sha256(prior_path.read_bytes()).hexdigest() if prior_path is not None else None
        frame = build_cells(pl.read_parquet(source), min_year=min_year, prior=prior)
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
                count = load_city(config.instance_config, city, source_digest=source_digest, prior_digest=prior_digest)
                results.append(f'{config.instance_config.identifier}: {count} new estimated cells, created={created}')
                results.extend(grant_demo_access(framework, organizations[ars], test_users[ars]))
            if not apply:
                transaction.set_rollback(True)
        for result in results:
            self.stdout.write(result)
        if not apply:
            self.stdout.write('Dry run: all database changes rolled back.')
