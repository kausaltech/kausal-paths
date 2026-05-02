"""
Fix DatasetMetric.name fields.

Management command to fix DatasetMetric.name fields that were incorrectly set to
human-readable labels instead of snake_case DVC column identifiers.

See docs/architecture/dataset-metric-names.md for the full explanation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand

from rich import print

from kausal_common.datasets.models import DatasetMetric

from common import polars as ppl
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset as DBDatasetModel

    from nodes.context import Context


def _normalize_for_matching(name: str) -> str:
    """
    Normalize a metric name or column identifier for fuzzy comparison.

    The main mismatch pattern is that DatasetMetric.name was set to a human-readable
    label (e.g. "1.3 Car pooling") while DVC uses snake_case ("13_car_pooling").
    This function normalizes both sides so they can be compared.

    Transformations applied:
    - Remove decimal points between digits: "1.3" -> "13"  (NZC section numbers)
    - Lowercase
    - Replace any run of non-alphanumeric characters with a single underscore
    - Strip leading/trailing underscores
    """
    name = re.sub(r'(\d)\.(\d)', r'\1\2', name)
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def fix_metrics_for_dataset(
    dataset: DBDatasetModel,
    ctx: Context,
    *,
    dry_run: bool = False,
) -> tuple[int, int, list[str]]:
    """
    Check and fix DatasetMetric.name values for a single dataset.

    Returns (fixed_count, unfixable_count, unfixable_names).
    """
    ds_id = dataset.identifier
    if ds_id is None:
        return 0, 0, []

    try:
        dvc_ds = ctx.load_dvc_dataset(ds_id)
    except Exception as exc:
        print(f"  [yellow]Could not load DVC dataset '{ds_id}': {exc}[/yellow]")
        return 0, 0, []

    df = ppl.from_dvc_dataset(dvc_ds)
    dvc_cols = set(df.get_meta().metric_cols)

    # Build normalized lookup: normalized_form -> original_dvc_col
    normalized_dvc = {_normalize_for_matching(col): col for col in dvc_cols}

    metrics = DatasetMetric.objects.filter(schema=dataset.schema)
    fixed = 0
    unfixable: list[str] = []

    for metric in metrics:
        name = metric.name or ''
        if name in dvc_cols:
            continue  # already correct

        norm = _normalize_for_matching(name)
        matched_col = normalized_dvc.get(norm)
        if matched_col is None:
            unfixable.append(name)
            continue

        if dry_run:
            print(f'    [cyan]Would fix[/cyan]: {name!r} -> {matched_col!r}')
        else:
            old = name
            metric.name = matched_col
            metric.save(update_fields=['name'])
            print(f'    [green]Fixed[/green]: {old!r} -> {matched_col!r}')
        fixed += 1

    return fixed, len(unfixable), unfixable


class Command(BaseCommand):
    help = (
        'Fix DatasetMetric.name fields that were set to human-readable labels '
        'instead of snake_case DVC column identifiers. '
        'See docs/architecture/dataset-metric-names.md for details.'
    )

    def add_arguments(self, parser):
        parser.add_argument('instances', metavar='INSTANCE_ID', type=str, nargs='*')
        parser.add_argument('--all', action='store_true', help='Fix all instances')
        parser.add_argument('--dry-run', action='store_true', help='Show what would change without making changes')

    def handle(self, **options):  # noqa: C901, PLR0912
        dry_run: bool = options['dry_run']
        instance_ids: list[str] = options['instances']
        do_all: bool = options['all']

        if not instance_ids and not do_all:
            self.stderr.write('Specify instance ID(s) or use --all.')
            return

        if do_all:
            ics = list(InstanceConfig.objects.all())
        else:
            ics = []
            for iid in instance_ids:
                try:
                    ics.append(InstanceConfig.objects.get(identifier=iid))
                except InstanceConfig.DoesNotExist:
                    self.stderr.write(f"Instance '{iid}' not found.")
                    return

        if dry_run:
            print('[yellow]Dry run — no changes will be written.[/yellow]')

        total_fixed = 0
        total_unfixable = 0

        for ic in ics:
            print(f'\n[bold]Instance: {ic.identifier}[/bold]')
            try:
                ctx = ic.get_instance().context
            except Exception as exc:
                print(f'  [red]Could not load instance: {exc}[/red]')
                continue

            from kausal_common.datasets.models import Dataset as DBDatasetModel

            datasets = list(DBDatasetModel.objects.get_queryset().for_instance_config(ic).select_related('schema'))
            if not datasets:
                print('  No datasets found.')
                continue

            for dataset in datasets:
                print(f'  Dataset: {dataset.identifier}')
                fixed, unfixable_count, unfixable_names = fix_metrics_for_dataset(dataset, ctx, dry_run=dry_run)
                total_fixed += fixed
                total_unfixable += unfixable_count
                if fixed == 0 and unfixable_count == 0:
                    print('    All metric names OK.')
                if unfixable_names:
                    print(f'    [red]Could not match {unfixable_count} metric(s) to DVC columns:[/red]')
                    for name in unfixable_names:
                        print(f'      - {name!r}')

        verb = 'Would fix' if dry_run else 'Fixed'
        print(f'\n[bold]{verb} {total_fixed} metric name(s). {total_unfixable} could not be matched automatically.[/bold]')
        if total_unfixable:
            print(
                '[yellow]Unmatched metrics need manual correction: '
                + 'set DatasetMetric.name to the exact DVC column identifier.[/yellow]'
            )
