"""
Report, per instance, which datasets are out of date with their DVC source and what to do.

``load_dvc_dataset --plan`` answers this for datasets you can already name. The problem is
naming them: ``--all`` enumerates ``ctx.get_all_dvc_dataset_ids()``, which only sees datasets
that load as ``DVCDataset``. An instance using ``use_datasets_from_db`` loads every imported
identifier as a ``DBDataset``, so the set is empty and ``--all`` has nothing to do — exactly
the instances where keeping track by hand is hardest.

This command enumerates the other way round: from the DB rows that record where they came
from (``external_ref``), plus whatever the model still declares as DVC-backed. For each it
reports a verdict:

    current       the stored data already matches the pinned commit
    import        out of date, and nothing is in the way
    rename first  a metric was renamed upstream and model bindings still hold the old name
    new           the model declares it but there is no DB row yet
    unreadable    the row claims a DVC source that will not read at the pinned commit
    db only       no DVC source: authored in the admin, so nothing to import

and then prints the commands to run, in the order they have to happen.

    python manage.py dataset_status mainz-bisko
    python manage.py dataset_status bisko mainz-bisko augsburg-bisko --stale-only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand

from rich import print

from kausal_common.datasets.models import Dataset

from common import polars as ppl
from nodes.management.commands.load_dvc_dataset import (
    apply_repo_provenance,
    build_dataset_plan,
    resolve_repo_provenance,
)
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from nodes.context import Context

VERDICT_STYLE = {
    'current': 'dim',
    'import': 'yellow',
    'rename first': 'red',
    'new': 'cyan',
    'unreadable': 'red',
    'db only': 'dim',
}


@dataclass
class DatasetStatus:
    ds_id: str
    verdict: str
    detail: str = ''

    @property
    def is_stale(self) -> bool:
        return self.verdict not in ('current', 'db only')


def candidate_dataset_ids(ic: InstanceConfig, ctx: Context) -> list[str]:
    """
    Every dataset worth checking: what the model declares, plus what has already been imported.

    The union matters. A model-declared id with no row is a dataset never imported; an
    imported row the model no longer declares is one that has fallen out of the model and
    may be stale forever without anyone noticing.
    """
    declared = set(ctx.get_all_dvc_dataset_ids())
    # Every row, not only the ones stamped with an ``external_ref``. A row imported before
    # provenance stamping existed has no stamp and is precisely the kind that goes stale
    # unnoticed; ``status_for`` works out whether it has a DVC source rather than guessing
    # from the stamp. Rows that turn out to be admin-authored are reported as "db only".
    stored = {ds.identifier for ds in Dataset.objects.get_queryset().for_instance_config(ic) if ds.identifier}
    return sorted(declared | stored)


def status_for(ic: InstanceConfig, ctx: Context, ds_id: str) -> DatasetStatus:
    dataset = Dataset.objects.get_queryset().for_instance_config(ic).filter(identifier=ds_id).select_related('schema').first()
    try:
        df = ppl.from_dvc_dataset(ctx.load_dvc_dataset(ds_id))
    except Exception as exc:  # any failure to read is reportable, not fatal
        summary = f'{type(exc).__name__}: {exc}'.split('\n')[0][:100]
        if dataset is not None and not dataset.external_ref:
            # Never came from DVC, so there is nothing to be out of date with.
            return DatasetStatus(ds_id, 'db only', 'no DVC source')
        return DatasetStatus(ds_id, 'unreadable', summary)

    meta = df.get_meta()
    plan = build_dataset_plan(
        ds_id=ds_id,
        dataset=dataset,
        incoming_metric_cols=list(meta.metric_cols),
        incoming_data_points=sum(df[col].drop_nulls().len() for col in meta.metric_cols),
        incoming_commit=None,
    )
    if plan.is_new:
        return DatasetStatus(ds_id, 'new', f'{plan.incoming_data_points} data point(s) waiting')
    if plan.blockers:
        return DatasetStatus(ds_id, 'rename first', plan.blockers[0])

    assert dataset is not None
    current_commit = (dataset.external_ref or {}).get('commit')
    pinned = ctx.dataset_repo_spec.commit if ctx.dataset_repo_spec else None
    changes = []
    if pinned and current_commit != pinned:
        changes.append(f'commit {current_commit or "unrecorded"} -> {pinned}')
    if plan.current_data_points != plan.incoming_data_points:
        changes.append(f'{plan.current_data_points} -> {plan.incoming_data_points} data points')
    if plan.added_metrics:
        changes.append(f'metrics add {", ".join(plan.added_metrics)}')
    if not changes:
        return DatasetStatus(ds_id, 'current')
    return DatasetStatus(ds_id, 'import', '; '.join(changes))


class Command(BaseCommand):
    help = "Report which of an instance's datasets are out of date with their DVC source"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instances', type=str, nargs='+', help='Instance identifiers')
        parser.add_argument('--stale-only', action='store_true', help='Hide datasets that are already current')
        parser.add_argument(
            '--repo-from',
            choices=['auto', 'yaml', 'db'],
            default='auto',
            help="Which config's DVC pin to compare against (default: auto)",
        )

    def _report_instance(self, instance_id: str, repo_from: str, stale_only: bool) -> list[DatasetStatus]:
        ic = InstanceConfig.objects.get(identifier=instance_id)
        ctx = ic.get_instance().context
        provenance = resolve_repo_provenance(ic, ctx, repo_from)
        apply_repo_provenance(ctx, provenance)

        pin = ctx.dataset_repo_spec.commit if ctx.dataset_repo_spec else None
        print(f'\n[bold]{instance_id}[/bold]  (config_source={ic.config_source}, pin={pin or "unset"})')

        statuses = [status_for(ic, ctx, ds_id) for ds_id in candidate_dataset_ids(ic, ctx)]
        shown = [s for s in statuses if s.is_stale or not stale_only]
        if not shown:
            print('  [dim]all datasets current[/dim]')
        for status in shown:
            style = VERDICT_STYLE.get(status.verdict, '')
            detail = f'  [dim]{status.detail}[/dim]' if status.detail else ''
            print(f'  [{style}]{status.verdict:<13}[/{style}] {status.ds_id}{detail}')
        return statuses

    def handle(self, *args: Any, **options: Any) -> None:
        by_instance: dict[str, list[DatasetStatus]] = {}
        for instance_id in options['instances']:
            by_instance[instance_id] = self._report_instance(instance_id, options['repo_from'], options['stale_only'])

        print('\n[bold]What to run[/bold]')
        nothing_to_do = True
        for instance_id, statuses in by_instance.items():
            renames = [s.ds_id for s in statuses if s.verdict == 'rename first']
            imports = [s.ds_id for s in statuses if s.verdict in ('import', 'new')]
            if not renames and not imports:
                continue
            nothing_to_do = False
            print(f'\n  [bold]{instance_id}[/bold]')
            if renames:
                # The rename has to land first: the import refuses while a model binding
                # still holds the old metric name.
                print(f'    python manage.py rename_dataset_metrics {instance_id} {" ".join(renames)} --apply')
            if imports or renames:
                targets = ' '.join(sorted(set(imports) | set(renames)))
                print(f'    python manage.py load_dvc_dataset {instance_id} {targets} --force')
            print(f'    python manage.py sync_instance_to_db {instance_id}')
        if nothing_to_do:
            print('  [green]nothing to do[/green]')
