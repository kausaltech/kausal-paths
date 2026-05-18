#!/usr/bin/env python3
"""
Import NZC yearly placeholder data into framework defaults.

Reads either a canonical CSV file or (by default) the DVC dataset
``nzc/placeholders_yearly``, and writes ``MeasureTemplateDefaultDataPoint``
records plus updates ``MeasureTemplate.default_value_scaling``.

Usage:
    # Load from DVC (default):
    python tools/import_nzc_yearly_placeholders.py

    # Load from a specific CSV:
    python tools/import_nzc_yearly_placeholders.py path/to/file.csv

    # Dry run (parse + validate, then roll back DB changes):
    python tools/import_nzc_yearly_placeholders.py --dry-run
"""

import argparse
import csv
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from django.db import transaction

GROUP_ORDER = ('0', '1', '2', '3')
GROUPS = {
    '0': {'renewable_mix': 'low', 'temperature': 'low'},
    '1': {'renewable_mix': 'low', 'temperature': 'high'},
    '2': {'renewable_mix': 'high', 'temperature': 'low'},
    '3': {'renewable_mix': 'high', 'temperature': 'high'},
}

DIMENSIONS = {
    'renewable_mix': {
        'name': 'Renewable mix',
        'categories': {
            'low': 'Low',
            'high': 'High',
        },
    },
    'temperature': {
        'name': 'Temperature',
        'categories': {
            'low': 'Low',
            'high': 'High',
        },
    },
}

POPULATION_MEASURE_TEMPLATE_UUID = '3779efa4-9eb0-4f4b-b5d5-eb510461bed8'
PLACEHOLDER_YEARLY_DATASET_IDENTIFIER = 'nzc/placeholders_yearly'


@dataclass
class ImportResult:
    source_rows: int
    measure_templates: int
    default_data_points: int
    skipped: int
    warnings: list[str]
    missing_measure_templates: list[str]
    dry_run: bool
    configs_inferred: int = 0
    configs_unresolved: int = 0
    configs_without_population: int = 0


def parse_decimal(value: str) -> float | None:
    value = value.strip()
    if value == '' or value.upper() == 'NA':
        return None
    return float(value.replace(',', '.'))


def parse_bool(value: str) -> bool:
    value = value.strip().upper()
    if value == 'TRUE':
        return True
    if value == 'FALSE':
        return False
    msg = f'Expected TRUE/FALSE, got {value!r}'
    raise ValueError(msg)


def get_source_default_value_scaling(measure_template_uuid: str, per_capita: bool) -> str | None:
    if not per_capita:
        return None
    if measure_template_uuid == POPULATION_MEASURE_TEMPLATE_UUID:
        return None
    return 'population'


def read_rows(path: Path) -> list[dict[str, str]]:
    """Load rows from a canonical comma-delimited CSV file."""
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f, delimiter=','))


def load_rows_from_dvc(framework_identifier: str) -> list[dict[str, str]]:
    """
    Load rows from the nzc/placeholders_yearly DVC dataset.

    The DVC dataset must have the canonical CSV structure (columns UUID,
    MeasureID, Unit, PerCapita, Year, 0_ccv … 3_max, Metric, Description).
    Typed values (float, bool, int) are converted to strings so the result
    matches what ``read_rows()`` produces from a CSV file.
    """
    from frameworks.models import Framework

    framework = Framework.objects.get(identifier=framework_identifier)
    config = framework.configs.first()
    if config is None:
        print(f'No FrameworkConfig found for framework {framework_identifier!r}', file=sys.stderr)
        raise SystemExit(1)
    instance = config.instance_config.get_instance()
    repo = instance.context.dataset_repo
    if repo is None:
        print('No dataset repo available for this instance.', file=sys.stderr)
        raise SystemExit(1)
    repo.set_target_commit(None)  # Use the newest commit
    if not repo.has_dataset(PLACEHOLDER_YEARLY_DATASET_IDENTIFIER):
        print(f'Dataset {PLACEHOLDER_YEARLY_DATASET_IDENTIFIER!r} not found in DVC.', file=sys.stderr)
        raise SystemExit(1)

    df = repo.load_dataframe(PLACEHOLDER_YEARLY_DATASET_IDENTIFIER)
    rows: list[dict[str, str]] = []
    for row_dict in df.to_dicts():
        str_row: dict[str, str] = {}
        for col, val in row_dict.items():
            if val is None or (isinstance(val, float) and math.isnan(val)):
                str_row[col] = ''
            elif isinstance(val, bool):
                str_row[col] = 'TRUE' if val else 'FALSE'
            else:
                str_row[col] = str(val)
        rows.append(str_row)
    return rows


def convert(rows: list[dict[str, str]], source_label: str = '') -> dict[str, Any]:
    measures: OrderedDict[str, dict[str, Any]] = OrderedDict()
    default_data_points: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    warnings: list[str] = []

    for source_row_number, row in enumerate(rows, start=2):
        measure = row['MeasureID'].strip()
        year = int(row['Year'])
        uuid = row['UUID'].strip()
        if uuid == '00924063-1dd8-43b1-a118-f3e29edbecf':
            uuid = '00924063-1dd8-43b1-a118-f3e29edbecf7'
        per_capita = parse_bool(row['PerCapita'])

        measure_info = measures.get(measure)
        if measure_info is None:
            scaling = get_source_default_value_scaling(uuid, per_capita)
            measure_info = {
                'source_measure': measure,
                'measure_template_uuid': uuid,
                'default_value_scaling': scaling,
                'source_per_capita': per_capita,
                'source_uuid_variants': [uuid],
            }
            measures[measure] = measure_info
        else:
            scaling = get_source_default_value_scaling(measure_info['measure_template_uuid'], per_capita)
            if uuid not in measure_info['source_uuid_variants']:
                measure_info['source_uuid_variants'].append(uuid)
            if scaling != measure_info['default_value_scaling']:
                msg = 'Measure %s mixes scaling modes: %s and %s' % (
                    measure,
                    measure_info['default_value_scaling'],
                    scaling,
                )
                raise ValueError(msg)

        for group_id, categories in GROUPS.items():
            value = parse_decimal(row[f'{group_id}_ccv'])
            lower_bound = parse_decimal(row[f'{group_id}_min'])
            upper_bound = parse_decimal(row[f'{group_id}_max'])
            if value is None:
                skipped.append({
                    'source_row_number': source_row_number,
                    'source_measure': measure,
                    'year': year,
                    'group': group_id,
                    'reason': 'missing_value',
                })
                continue

            default_data_points.append({
                'source_measure': measure,
                'measure_template_uuid': measure_info['measure_template_uuid'],
                'year': year,
                'value': value,
                'probable_lower_bound': lower_bound,
                'probable_upper_bound': upper_bound,
                'categories': categories,
            })

    warnings.extend(
        'Measure %s has %d source UUIDs; using first UUID %s'
        % (
            measure_info['source_measure'],
            len(measure_info['source_uuid_variants']),
            measure_info['measure_template_uuid'],
        )
        for measure_info in measures.values()
        if len(measure_info['source_uuid_variants']) > 1
    )
    warnings.extend(
        'Measure %s maps to the Population template; ignoring source PerCapita=TRUE for default scaling.'
        % measure_info['source_measure']
        for measure_info in measures.values()
        if measure_info['measure_template_uuid'] == POPULATION_MEASURE_TEMPLATE_UUID and measure_info['source_per_capita']
    )

    return {
        'schema_version': 1,
        'source': {
            'path': source_label,
            'row_count': len(rows),
            'notes': [
                'Source CSV uses semicolon separators and decimal commas.',
                'When a source measure has varying UUIDs across years, the first UUID is used.',
                'Rows with NA central values are skipped.',
            ],
        },
        'dimensions': DIMENSIONS,
        'measure_templates': list(measures.values()),
        'default_data_points': default_data_points,
        'skipped': skipped,
        'warnings': warnings,
    }


def get_or_create_dimensions(framework: Any) -> dict[tuple[str, str], Any]:
    from frameworks.models import FrameworkDimension, FrameworkDimensionCategory

    categories: dict[tuple[str, str], Any] = {}
    for dimension_identifier, dimension_data in DIMENSIONS.items():
        dimension, _created = FrameworkDimension.objects.get_or_create(
            framework=framework,
            identifier=dimension_identifier,
            defaults={'name': dimension_data['name']},
        )
        if dimension.name != dimension_data['name']:
            dimension.name = dimension_data['name']
            dimension.save(update_fields=['name'])

        for category_identifier, category_name in dimension_data['categories'].items():
            category, _created = FrameworkDimensionCategory.objects.get_or_create(
                dimension=dimension,
                name=category_name,
            )
            categories[(dimension_identifier, category_identifier)] = category

    return categories


def get_old_placeholder_rows(framework: Any) -> list[dict[str, Any]]:
    from frameworks.nzc import PLACEHOLDER_DATASET_IDENTIFIER

    config = framework.configs.first()
    if config is None:
        return []
    instance = config.instance_config.get_instance()
    repo = instance.context.dataset_repo
    if repo is None:
        return []
    if not repo.has_dataset(PLACEHOLDER_DATASET_IDENTIFIER):
        return []
    df = repo.load_dataframe(PLACEHOLDER_DATASET_IDENTIFIER)
    return df.to_dicts()


def build_old_placeholder_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    lookup: dict[str, dict[str, float | None]] = {}
    for row in rows:
        uuid = str(row['UUID'])
        values: dict[str, float | None] = {
            'per_capita': bool(row.get('PerCapita')),
        }
        for group_id in GROUP_ORDER:
            raw_value = row.get(group_id)
            values[group_id] = None if raw_value is None else float(raw_value)
        lookup[uuid] = values
    return lookup


def get_effective_default_value_scaling(template: Any, requested_scaling: str | None) -> str | None:
    if requested_scaling != 'population':
        return requested_scaling
    if str(template.uuid) == POPULATION_MEASURE_TEMPLATE_UUID or template.name.strip().lower() == 'population':
        return None
    return requested_scaling


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(actual), abs(expected), 1.0)


def score_group(
    datapoints: list[tuple[str, float]],
    old_lookup: dict[str, dict[str, float | None]],
    group_id: str,
    *,
    per_capita: bool,
) -> tuple[int, float]:
    errors: list[float] = []
    for uuid, actual in datapoints:
        row = old_lookup.get(uuid)
        if row is None or bool(row['per_capita']) != per_capita:
            continue
        expected = row.get(group_id)
        if expected is None:
            continue
        errors.append(relative_error(actual, expected))
    if not errors:
        return 0, float('inf')
    return len(errors), sum(errors) / len(errors)


def infer_population(
    datapoints: list[tuple[str, float]],
    old_lookup: dict[str, dict[str, float | None]],
    group_id: str,
) -> int | None:
    ratios = []
    for uuid, actual in datapoints:
        row = old_lookup.get(uuid)
        if row is None or not row['per_capita']:
            continue
        expected = row.get(group_id)
        if expected is None or expected == 0:
            continue
        ratio = actual / expected
        if ratio > 0:
            ratios.append(ratio)
    if not ratios:
        return None
    return round(median(ratios))


def set_framework_config_categories(config: Any, category_map: dict[tuple[str, str], Any], group_id: str) -> None:
    group = GROUPS[group_id]
    dimension_identifiers = set(DIMENSIONS.keys())
    config.categories.remove(
        *config.categories.filter(dimension__identifier__in=dimension_identifiers),
    )
    config.categories.add(
        category_map[('renewable_mix', group['renewable_mix'])],
        category_map[('temperature', group['temperature'])],
    )


def backfill_framework_config_context(
    framework: Any,
    category_map: dict[tuple[str, str], Any],
    old_lookup: dict[str, dict[str, float | None]],
) -> tuple[int, int, int]:
    from frameworks.models import FrameworkConfig, MeasureDataPoint

    inferred = 0
    unresolved = 0
    without_population = 0
    configs = FrameworkConfig.objects.filter(framework=framework).order_by('pk')

    for config in configs:
        datapoints = [
            (str(uuid), float(default_value))
            for uuid, default_value in MeasureDataPoint.objects.filter(
                measure__framework_config=config,
                year=config.baseline_year,
                default_value__isnull=False,
            ).values_list('measure__measure_template__uuid', 'default_value')
        ]
        scores = {group_id: score_group(datapoints, old_lookup, group_id, per_capita=False) for group_id in GROUP_ORDER}
        viable_scores = {group_id: score for group_id, score in scores.items() if score[0] >= 3}
        if not viable_scores:
            unresolved += 1
            print(
                'Could not infer renewable_mix/temperature for %s: not enough comparable defaults.' % config.organization_name,
                file=sys.stderr,
            )
            continue

        group_id, (match_count, mean_error) = min(
            viable_scores.items(),
            key=lambda item: (item[1][1], -item[1][0]),
        )
        group = GROUPS[group_id]
        population = infer_population(datapoints, old_lookup, group_id)
        if population is None:
            without_population += 1

        set_framework_config_categories(config, category_map, group_id)
        extra = dict(config.extra or {})
        create_context = dict(extra.get('create_context') or {})
        create_context.update({
            'renewable_mix': group['renewable_mix'],
            'temperature': group['temperature'],
        })
        if population is not None:
            create_context['population'] = population
        extra['create_context'] = create_context
        extra['create_context_inference'] = {
            'method': 'baseline_default_value_fingerprint',
            'matched_non_population_defaults': match_count,
            'mean_relative_error': mean_error,
        }
        config.extra = extra
        config.save(update_fields=['extra'])
        inferred += 1

    return inferred, unresolved, without_population


def import_data(
    rows: list[dict[str, str]],
    source_label: str,
    framework_identifier: str,
    dry_run: bool,
) -> ImportResult:
    from frameworks.models import Framework, MeasureTemplate, MeasureTemplateDefaultDataPoint

    data = convert(rows, source_label)
    framework = Framework.objects.get(identifier=framework_identifier)
    categories = get_or_create_dimensions(framework)
    old_placeholder_lookup = build_old_placeholder_lookup(get_old_placeholder_rows(framework))
    measure_templates = data['measure_templates']
    for item in measure_templates:
        if not item['measure_template_uuid']:
            print(item)
    template_uuid_by_source_measure = {
        item['source_measure']: item['measure_template_uuid'] for item in measure_templates if item['measure_template_uuid']
    }
    template_uuids = set(template_uuid_by_source_measure.values())
    templates_by_uuid = {
        str(mt.uuid): mt for mt in MeasureTemplate.objects.filter(section__framework=framework, uuid__in=template_uuids)
    }
    missing_measure_templates = sorted(template_uuids - set(templates_by_uuid.keys()))

    for item in measure_templates:
        template = templates_by_uuid.get(item['measure_template_uuid'])
        if template is None:
            continue
        effective_scaling = get_effective_default_value_scaling(template, item['default_value_scaling'])
        if effective_scaling != item['default_value_scaling']:
            data['warnings'].append(
                'Measure %s maps to the Population template; ignoring source PerCapita=TRUE for default scaling.'
                % item['source_measure']
            )
        if template.default_value_scaling != effective_scaling:
            template.default_value_scaling = effective_scaling
            template.save(update_fields=['default_value_scaling'])

    affected_templates = list(templates_by_uuid.values())
    MeasureTemplateDefaultDataPoint.objects.filter(template__in=affected_templates).delete()

    created = 0
    for item in data['default_data_points']:
        template = templates_by_uuid.get(item['measure_template_uuid'])
        if template is None:
            continue
        ddp = MeasureTemplateDefaultDataPoint.objects.create(
            template=template,
            year=item['year'],
            value=item['value'],
            probable_lower_bound=item['probable_lower_bound'],
            probable_upper_bound=item['probable_upper_bound'],
        )
        ddp.categories.set([
            categories[('renewable_mix', item['categories']['renewable_mix'])],
            categories[('temperature', item['categories']['temperature'])],
        ])
        created += 1

    configs_inferred = 0
    configs_unresolved = 0
    configs_without_population = 0
    if old_placeholder_lookup:
        configs_inferred, configs_unresolved, configs_without_population = backfill_framework_config_context(
            framework,
            categories,
            old_placeholder_lookup,
        )
    else:
        print('Old nzc/placeholders dataset unavailable; skipped FrameworkConfig context backfill.', file=sys.stderr)

    return ImportResult(
        source_rows=data['source']['row_count'],
        measure_templates=len(measure_templates),
        default_data_points=created,
        skipped=len(data['skipped']),
        warnings=data['warnings'],
        missing_measure_templates=missing_measure_templates,
        dry_run=dry_run,
        configs_inferred=configs_inferred,
        configs_unresolved=configs_unresolved,
        configs_without_population=configs_without_population,
    )


def print_result(result: ImportResult) -> None:
    action = 'Would import' if result.dry_run else 'Imported'
    print(
        '%s %d default datapoints for %d source measures from %d CSV rows.'
        % (action, result.default_data_points, result.measure_templates, result.source_rows)
    )
    if result.skipped:
        print('Skipped %d entries with missing central values.' % result.skipped)
    print(
        'Inferred context for %d framework configs; %d unresolved; %d without inferred population.'
        % (result.configs_inferred, result.configs_unresolved, result.configs_without_population)
    )
    if result.missing_measure_templates:
        print(
            'Missing %d referenced MeasureTemplates:' % len(result.missing_measure_templates),
            file=sys.stderr,
        )
        for uuid in result.missing_measure_templates:
            print(f'  {uuid}', file=sys.stderr)
    for warning in result.warnings:
        print(f'warning: {warning}', file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Import NZC yearly placeholder data into framework defaults.',
    )
    parser.add_argument(
        'input',
        nargs='?',
        type=Path,
        default=None,
        help=(
            'Input CSV path (comma-delimited, canonical column names). '
            'Omit to load from DVC dataset %r.' % PLACEHOLDER_YEARLY_DATASET_IDENTIFIER
        ),
    )
    parser.add_argument(
        '--framework',
        default='nzc',
        help='Framework identifier. Defaults to nzc.',
    )
    parser.add_argument('--dry-run', action='store_true', help='Parse and validate, then roll back DB changes.')
    args = parser.parse_args()

    from kausal_common.development.django import init_django

    init_django()

    if args.input is not None:
        rows = read_rows(args.input)
        source_label = str(args.input)
    else:
        rows = load_rows_from_dvc(args.framework)
        source_label = PLACEHOLDER_YEARLY_DATASET_IDENTIFIER

    with transaction.atomic():
        result = import_data(rows, source_label, args.framework, args.dry_run)
        print_result(result)
        if args.dry_run:
            transaction.set_rollback(True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
