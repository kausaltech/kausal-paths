from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand
from django.db import transaction

from kausal_common.datasets.sync import DatasetDjangoAdapter, DatasetJSONAdapter

from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from django.core.management.base import CommandParser


class Command(BaseCommand):
    help = 'Synchronize or export datasets'

    dry_run: bool = False
    yes: bool = False

    def add_arguments(self, parser: CommandParser):
        parser.add_argument(
            'action',
            type=str,
            choices=['import', 'export', 'csv'],
            help='Action to perform',
        )
        parser.add_argument(
            'instance',
            type=str,
            help='Instance identifier',
        )
        parser.add_argument(
            'file',
            type=Path,
            nargs='?',
            help='JSON file to read from or write to (not used for csv action)',
        )
        parser.add_argument('--dry-run', '-N', action='store_true', help='Only show the changes that would be made')
        parser.add_argument('--yes', '-y', action='store_true', help='Do not prompt for anything')
        parser.add_argument(
            '--datasets',
            '-d',
            type=str,
            action='append',
            metavar='DATASET_IDENTIFIER',
            help='Limit to specific dataset(s) by identifier (can be repeated)',
        )

    @transaction.atomic
    def handle(self, *_args, **options):
        action = options['action']
        file_path: Path | None = options.get('file')
        self.dry_run = options['dry_run']
        self.yes = options['yes']
        dataset_identifiers: list[str] | None = options.get('datasets')

        instance_identifier = options['instance']

        if action == 'import':
            if not file_path:
                self.stderr.write(self.style.ERROR('file argument is required for import'))
                return
            self.import_data(file_path, instance_identifier, dataset_identifiers)
        elif action == 'export':
            if not file_path:
                self.stderr.write(self.style.ERROR('file argument is required for export'))
                return
            self.export_data(file_path, instance_identifier, dataset_identifiers)
        elif action == 'csv':
            if not file_path:
                self.stderr.write(self.style.ERROR('file argument is required for csv'))
                return
            self.export_csv(file_path, instance_identifier, dataset_identifiers)

    def import_data(self, file_path: Path, instance_identifier: str, _dataset_identifiers: list[str] | None):
        try:
            ic = InstanceConfig.objects.get(identifier=instance_identifier)
        except InstanceConfig.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Instance with identifier '{instance_identifier}' not found"))
            return

        js = DatasetJSONAdapter(file_path)
        js.load()

        dj = DatasetDjangoAdapter(scope=ic, interactive=True)
        if self.yes:
            dj.allow_related_deletion = True
        with dj.start():
            dj.load()
            dj.sync_from(js)
            if self.dry_run:
                self.stderr.write(self.style.WARNING('Dry run requested; no changes done.'))
                dj.rollback()

    def export_data(self, file_path: Path, instance_identifier: str, _dataset_identifiers: list[str] | None):
        try:
            ic = InstanceConfig.objects.get(identifier=instance_identifier)
        except InstanceConfig.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Instance with identifier '{instance_identifier}' not found"))
            return

        dj = DatasetDjangoAdapter(ic)
        with dj.start():
            dj.load()

        if self.dry_run:
            self.stderr.write(self.style.WARNING('Dry run requested; no changes done.'))
            return

        dj.save_json(file_path)
        self.stdout.write(self.style.SUCCESS(f'Successfully exported datasets to {file_path}'))

    @staticmethod
    def _dataset_to_rows(
        dataset: Any,
        schema_dims: list[Any],
        schema_name: str,
    ) -> tuple[list[dict[str, str]], set[int]]:
        """Return (rows, years) for one Dataset, iterating all its metrics."""
        from collections import defaultdict

        from kausal_common.datasets.models import DataPoint

        rows: list[dict[str, str]] = []
        years: set[int] = set()

        for metric in dataset.schema.metrics.order_by('order'):
            dps = (
                DataPoint.objects.filter(dataset=dataset, metric=metric).prefetch_related('dimension_categories').order_by('date')
            )
            grouped: dict[tuple[tuple[str, str], ...], dict[int, str]] = defaultdict(dict)
            for dp in dps:
                cats_by_dim = {
                    cat.dimension_id: (cat.identifier or cat.label_i18n or cat.label) for cat in dp.dimension_categories.all()
                }
                dim_key = tuple((d.name_i18n or d.name, cats_by_dim.get(d.pk, '')) for d in schema_dims)
                year = dp.date.year
                years.add(year)
                grouped[dim_key][year] = str(dp.value) if dp.value is not None else ''

            metric_label = metric.label_i18n or metric.label or metric.name or ''
            for dim_key, year_values in grouped.items():
                row: dict[str, str] = {
                    'Metric': metric_label,
                    'Unit': metric.unit or '',
                    'Quantity': '',
                    'Dataset': schema_name,
                    **dict(dim_key),
                    **{str(yr): val for yr, val in year_values.items()},
                }
                rows.append(row)

        return rows, years

    def export_csv(self, output_path: Path, instance_identifier: str, _dataset_identifiers: list[str] | None):
        import csv

        from django.contrib.contenttypes.models import ContentType

        from kausal_common.datasets.models import Dataset

        try:
            ic = InstanceConfig.objects.get(identifier=instance_identifier)
        except InstanceConfig.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Instance with identifier '{instance_identifier}' not found"))
            return

        ct = ContentType.objects.get_for_model(ic)
        datasets = (
            Dataset.objects
            .filter(scope_content_type=ct, scope_id=ic.pk)
            .select_related('schema')
            .prefetch_related('schema__metrics', 'schema__dimensions__dimension')
        )

        all_rows: list[dict[str, str]] = []
        all_years: set[int] = set()
        dim_col_order: list[str] = []

        for dataset in datasets:
            if dataset.schema is None:
                continue
            schema_name = dataset.schema.name_i18n or dataset.schema.name or str(dataset.schema.uuid)
            schema_dims = [sd.dimension for sd in dataset.schema.dimensions.order_by('order')]
            for d in schema_dims:
                col = d.name_i18n or d.name
                if col not in dim_col_order:
                    dim_col_order.append(col)
            rows, years = self._dataset_to_rows(dataset, schema_dims, schema_name)
            all_rows.extend(rows)
            all_years.update(years)

        if not all_rows:
            self.stderr.write(self.style.WARNING('No data found for this instance.'))
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['Metric', 'Unit', 'Quantity', 'Dataset'] + dim_col_order + [str(y) for y in sorted(all_years)]

        with output_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', restval='')
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

        self.stdout.write(self.style.SUCCESS(f'Exported datasets to {output_path}'))
