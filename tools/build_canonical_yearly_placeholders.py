#!/usr/bin/env python3
"""
Build the canonical NZC yearly placeholders CSV.

Joins five source files into a single authoritative CSV:

  placeholders.csv               → UUID and PerCapita (authoritative; overrides source CSV)
  42_cities_input_9_clusters.csv → Description (from column-header prefix "MID/description")
  nzc_updated_datasets.csv       → Unit and Metric (primary)
  Draft Matias_...defaults.csv   → Unit and Metric (fallback for measures absent from above)
  placeholders_30042026_11_34_PM.csv → yearly values (UUID and PerCapita columns discarded)

Output column order:
  UUID | MeasureID | Unit | PerCapita | Year | <data cols> | Metric | Description

Usage:
    cd /path/to/kausal-paths
    python data/nzc/lucia/build_canonical_yearly_placeholders.py [--output PATH]
"""  # noqa: EXE001

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DATA_NZC = Path(__file__).resolve().parents[3] / 'data' / 'nzc'

PLACEHOLDERS_CSV = DATA_NZC / 'placeholders.csv'
CITIES_CSV = DATA_NZC / '42_cities_input_9_clusters.csv'
UPDATED_DATASETS_CSV = DATA_NZC / 'nzc_updated_datasets.csv'
VALENCIA_CSV = DATA_NZC / 'Draft Matias_ Dataset of 010824 Template Economic Case - Update5 - defaults.csv'
YEARLY_SOURCE_CSV = DATA_NZC / 'placeholders_30042026_11_34_PM.csv'

DEFAULT_OUTPUT = DATA_NZC / 'placeholders_yearly_canonical.csv'

DATA_COLS = [
    '0_ccv',
    '0_min',
    '0_max',
    '1_ccv',
    '1_min',
    '1_max',
    '2_ccv',
    '2_min',
    '2_max',
    '3_ccv',
    '3_min',
    '3_max',
]

OUTPUT_FIELDS = ['UUID', 'MeasureID', 'Unit', 'PerCapita', 'Year'] + DATA_COLS + ['Metric', 'Description']


def parse_eu_float(s: str) -> str:
    """Convert European decimal-comma float string to dot notation, preserving empty/NA as empty."""
    s = s.strip().replace(',', '.')
    if not s or s.upper() in ('NA', 'N/A', 'NULL', 'NONE'):
        return ''
    # Validate it parses; propagate as-is if it does
    float(s)
    return s


def load_uuid_authority(path: Path) -> dict[str, dict]:
    """Return {measure_id: {uuid, per_capita}} from placeholders.csv."""
    result: dict[str, dict] = {}
    with path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mid = row['Measure'].strip()
            result[mid] = {
                'uuid': row['UUID'].strip(),
                'per_capita': row['PerCapita'].strip().lower() == 'true',
            }
    return result


def load_descriptions(path: Path) -> dict[str, str]:
    """Return {measure_id: description} from 42_cities column headers (format: 'MID/description')."""
    with path.open(encoding='utf-8') as f:
        headers = next(csv.reader(f))
    result: dict[str, str] = {}
    for h in headers:
        parts = h.split('/', 1)
        if len(parts) == 2:
            result[parts[0].strip()] = parts[1].strip()
    return result


def load_unit_metric(updated_path: Path, valencia_path: Path) -> dict[str, dict]:
    """Return {uuid: {unit, metric}} from nzc_updated_datasets, with Valencia fallback."""
    result: dict[str, dict] = {}

    # Primary: nzc_updated_datasets.csv
    with updated_path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            uuid = row.get('UUID', '').strip()
            if uuid and uuid not in result:
                result[uuid] = {
                    'unit': row.get('Unit', '').strip(),
                    'metric': row.get('Metric', '').strip(),
                }

    # Fallback: Valencia/defaults file (Sector used as Metric)
    with valencia_path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            uuid = row.get('UUID', '').strip()
            if uuid and uuid not in result:
                result[uuid] = {
                    'unit': row.get('Unit', '').strip(),
                    'metric': row.get('Sector', '').strip(),
                }

    return result


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Output CSV path')
    args = parser.parse_args()

    authority = load_uuid_authority(PLACEHOLDERS_CSV)
    descriptions = load_descriptions(CITIES_CSV)
    unit_metric = load_unit_metric(UPDATED_DATASETS_CSV, VALENCIA_CSV)

    skipped_no_auth: list[str] = []
    skipped_no_meta: list[str] = []
    rows_out: list[dict] = []

    with YEARLY_SOURCE_CSV.open(encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for raw in reader:
            mid = raw['Measure'].strip()

            # UUID and PerCapita come from the authority, not the source CSV
            auth = authority.get(mid)
            if auth is None:
                if mid not in skipped_no_auth:
                    skipped_no_auth.append(mid)
                continue

            uuid = auth['uuid']
            per_capita = auth['per_capita']

            meta = unit_metric.get(uuid)
            if meta is None:
                if mid not in skipped_no_meta:
                    skipped_no_meta.append(mid)
                continue

            rows_out.append({
                'UUID': uuid,
                'MeasureID': mid,
                'Unit': meta['unit'],
                'PerCapita': 'true' if per_capita else 'false',
                'Year': raw['Year'].strip(),
                '0_ccv': parse_eu_float(raw['0_ccv']),
                '0_min': parse_eu_float(raw['0_min_confidence']),
                '0_max': parse_eu_float(raw['0_max_confidence']),
                '1_ccv': parse_eu_float(raw['1_ccv']),
                '1_min': parse_eu_float(raw['1_min_confidence']),
                '1_max': parse_eu_float(raw['1_max_confidence']),
                '2_ccv': parse_eu_float(raw['2_ccv']),
                '2_min': parse_eu_float(raw['2_min_confidence']),
                '2_max': parse_eu_float(raw['2_max_confidence']),
                '3_ccv': parse_eu_float(raw['3_ccv']),
                '3_min': parse_eu_float(raw['3_min_confidence']),
                '3_max': parse_eu_float(raw['3_max_confidence']),
                'Metric': meta['metric'],
                'Description': descriptions.get(mid, ''),
            })

    # Deduplicate: keep last occurrence per (UUID, Year) — matches lucia script behaviour
    seen: dict[tuple, int] = {}
    for i, row in enumerate(rows_out):
        seen[(row['UUID'], row['Year'])] = i
    deduped = [rows_out[i] for i in sorted(seen.values())]

    dropped = len(rows_out) - len(deduped)

    output: Path = args.output
    with output.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(deduped)

    print(f'Written {len(deduped)} rows to {output}')
    if dropped:
        print(f'Dropped {dropped} duplicate (UUID, Year) rows (kept last)')
    if skipped_no_auth:
        print(f'Skipped {len(skipped_no_auth)} measure(s) not in placeholders.csv: {skipped_no_auth}')
    if skipped_no_meta:
        print(f'Skipped {len(skipped_no_meta)} measure(s) with no unit/metric: {skipped_no_meta}')

    # Report PerCapita corrections vs source CSV
    source_true: set[str] = set()
    with YEARLY_SOURCE_CSV.open(encoding='utf-8-sig') as f:
        for raw in csv.DictReader(f, delimiter=';'):
            if raw['PerCapita'].strip().upper() == 'TRUE':
                source_true.add(raw['Measure'].strip())
    auth_true = {mid for mid, a in authority.items() if a['per_capita']}
    corrected = source_true - auth_true
    if corrected:
        print(f'\nPerCapita corrections (source had TRUE, authority says FALSE): {sorted(corrected)}')


if __name__ == '__main__':
    main()
