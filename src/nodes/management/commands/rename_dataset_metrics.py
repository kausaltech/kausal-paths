"""
Rename a dataset's metric rows so they match columns that were renamed in the DVC data.

When a DVC dataset is re-uploaded with a metric column under a new name, importing it wants
to drop the old metric and add the new one. It cannot: ``NodeInputPortBinding.metric``
holds a PROTECTed reference, so ``load_dvc_dataset`` refuses (or, before that refusal
counted every binding, died halfway through).

Deleting the bindings and rebuilding them by re-syncing works but throws away and recreates
identity for no reason. Renaming the metric row in place is better on every count: the
bindings keep pointing at the same row, the metric UUID survives for anything that pinned
it, and the import then sees the metric as *kept* rather than dropped, so the conflict never
arises. The data points come along, and are replaced by the import anyway.

    python manage.py rename_dataset_metrics mainz-bisko bisko/weather_correction
    python manage.py rename_dataset_metrics mainz-bisko --all
    python manage.py rename_dataset_metrics mainz-bisko bisko/energy_shares --rename Value=default --apply

With no ``--rename``, the mapping is inferred by comparing the stored metric names against
the incoming DVC columns, and only when the answer is unambiguous. Nothing is written
without ``--apply``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from rich import print

from kausal_common.datasets.models import Dataset, DatasetMetric

from common import polars as ppl
from nodes.management.commands.load_dvc_dataset import apply_repo_provenance, resolve_repo_provenance
from nodes.models import InstanceConfig, NodeInputPortBinding

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from nodes.context import Context


@dataclass
class RenamePlan:
    """What one dataset needs, and why it may not be possible."""

    ds_id: str
    renames: list[tuple[str, str]] = field(default_factory=list)
    """(old name, new name) pairs, in the order they will be applied."""
    bindings: dict[str, int] = field(default_factory=dict)
    """How many model bindings ride along per renamed metric."""
    unchanged: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    note: str | None = None

    @property
    def is_actionable(self) -> bool:
        return bool(self.renames) and not self.blockers


def _binding_count(metric: DatasetMetric) -> int:
    return NodeInputPortBinding.objects.filter(metric=metric).count()


def _explicit_renames(existing: dict[str, DatasetMetric], explicit: dict[str, str], plan: RenamePlan) -> None:
    for old, new in explicit.items():
        if old not in existing:
            plan.blockers.append(f'no metric named {old!r} on this dataset')
        elif new in existing:
            plan.blockers.append(f'metric {new!r} already exists; renaming {old!r} onto it would collide')
        else:
            plan.renames.append((old, new))


def _inferred_renames(dropped: list[str], added: list[str], plan: RenamePlan) -> None:
    """Infer the mapping, but only where there is exactly one sensible answer."""
    if not dropped and not added:
        plan.note = 'metric names already match the DVC data'
    elif len(dropped) == 1 and len(added) == 1:
        plan.renames.append((dropped[0], added[0]))
    elif dropped and added:
        plan.blockers.append(
            'cannot infer the mapping: %d metric(s) would be dropped (%s) and %d added (%s). '
            'Say which is which with --rename OLD=NEW.'
            % (len(dropped), ', '.join(sorted(dropped)), len(added), ', '.join(sorted(added)))
        )
    elif dropped:
        plan.note = 'metric(s) %s are gone from the DVC data with nothing to rename them to' % ', '.join(sorted(dropped))
    else:
        plan.note = 'metric(s) %s are new; the import will add them' % ', '.join(sorted(added))


def build_rename_plan(
    ds_id: str,
    dataset: Dataset | None,
    incoming_metric_cols: list[str],
    explicit: dict[str, str],
) -> RenamePlan:
    """Work out which metric rows to rename, without touching anything."""
    plan = RenamePlan(ds_id=ds_id)
    if dataset is None:
        plan.note = 'no DB row for this dataset; nothing to rename'
        return plan
    schema = dataset.schema
    if schema is None:
        plan.note = 'dataset has no schema'
        return plan

    existing = {m.name: m for m in schema.metrics.all() if m.name}
    incoming = list(incoming_metric_cols)
    plan.unchanged = sorted(name for name in existing if name in incoming)

    if explicit:
        _explicit_renames(existing, explicit, plan)
    else:
        _inferred_renames(
            [name for name in existing if name not in incoming],
            [name for name in incoming if name not in existing],
            plan,
        )

    # A shared schema means the rename lands on every dataset using it, which is rarely what
    # the operator has in mind when naming one dataset.
    if plan.renames and schema.datasets.count() > 1:
        plan.blockers.append(
            'schema is shared with %d dataset(s); renaming here would rename it for all of them' % schema.datasets.count()
        )

    for old, _ in plan.renames:
        plan.bindings[old] = _binding_count(existing[old])
    return plan


def print_rename_plan(plan: RenamePlan) -> None:
    print(f'[bold]{plan.ds_id}[/bold]')
    if plan.note and not plan.renames:
        print(f'  [dim]{plan.note}[/dim]')
    for old, new in plan.renames:
        rides = plan.bindings.get(old, 0)
        carried = f'{rides} binding(s) follow it' if rides else 'no bindings'
        print(f'  rename       {old!r} [yellow]->[/yellow] {new!r} ({carried})')
    if plan.unchanged:
        print(f'  unchanged    {", ".join(plan.unchanged)}')
    for blocker in plan.blockers:
        print(f'  [red]blocker[/red]      {blocker}')


class Command(BaseCommand):
    help = 'Rename dataset metric rows to match columns renamed in the DVC data, keeping model bindings intact'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instance', type=str, help='Instance identifier')
        parser.add_argument('datasets', type=str, nargs='*', help='Dataset identifiers; omit with --all for every one')
        parser.add_argument('--all', action='store_true', help='Every DVC dataset the instance declares')
        parser.add_argument(
            '--rename',
            action='append',
            default=[],
            metavar='OLD=NEW',
            help='Explicit mapping, repeatable. Without it the mapping is inferred when unambiguous.',
        )
        parser.add_argument(
            '--repo-from',
            choices=['auto', 'yaml', 'db'],
            default='auto',
            help="Which config's DVC pin to read the incoming data from (default: auto)",
        )
        parser.add_argument('--apply', action='store_true', help='Write the renames (default is a plan only)')

    def _incoming_metric_cols(self, ctx: Context, ds_id: str) -> list[str]:
        df = ppl.from_dvc_dataset(ctx.load_dvc_dataset(ds_id))
        return list(df.get_meta().metric_cols)

    def _parse_renames(self, items: list[str]) -> dict[str, str]:
        explicit: dict[str, str] = {}
        for item in items:
            if '=' not in item:
                raise CommandError(f'--rename expects OLD=NEW, got {item!r}')
            old, new = item.split('=', 1)
            explicit[old.strip()] = new.strip()
        return explicit

    def _dataset_for(self, ic: InstanceConfig, ds_id: str) -> Dataset | None:
        return Dataset.objects.get_queryset().for_instance_config(ic).filter(identifier=ds_id).select_related('schema').first()

    def _apply(self, ic: InstanceConfig, plans: list[RenamePlan]) -> None:
        with transaction.atomic():
            for plan in plans:
                dataset = self._dataset_for(ic, plan.ds_id)
                assert dataset is not None, 'a plan with renames always has a dataset'
                for old, new in plan.renames:
                    metric = DatasetMetric.objects.get(schema=dataset.schema, name=old)
                    metric.name = new
                    metric.save(update_fields=['name'])
                    print(f'{plan.ds_id}: renamed {old!r} -> {new!r}')

    def _dataset_ids(self, ic: InstanceConfig, options: dict[str, Any], explicit: dict[str, str]) -> list[str]:
        ds_ids: list[str] = options['datasets']
        if not ds_ids:
            if not options['all']:
                raise CommandError('Name at least one dataset, or pass --all')
            known = Dataset.objects.get_queryset().for_instance_config(ic).values_list('identifier', flat=True)
            ds_ids = sorted(i for i in known if i)
        if explicit and len(ds_ids) > 1:
            raise CommandError('--rename applies to one dataset at a time; name a single dataset')
        return ds_ids

    def _plans_for(self, ic: InstanceConfig, ctx: Context, ds_ids: list[str], explicit: dict[str, str]) -> list[RenamePlan]:
        plans: list[RenamePlan] = []
        for ds_id in ds_ids:
            try:
                incoming = self._incoming_metric_cols(ctx, ds_id)
            except Exception as exc:
                plan = RenamePlan(ds_id=ds_id)
                plan.blockers.append(f'could not read the DVC data: {type(exc).__name__}: {exc}')
                plans.append(plan)
                continue
            plans.append(build_rename_plan(ds_id, self._dataset_for(ic, ds_id), incoming, explicit))
        return plans

    def handle(self, *args: Any, **options: Any) -> None:
        explicit = self._parse_renames(options['rename'])
        ic = InstanceConfig.objects.get(identifier=options['instance'])
        ctx = ic.get_instance().context
        apply_repo_provenance(ctx, resolve_repo_provenance(ic, ctx, options['repo_from']))

        plans = self._plans_for(ic, ctx, self._dataset_ids(ic, options, explicit), explicit)
        actionable = [p for p in plans if p.is_actionable]
        blocked = [p for p in plans if p.blockers]
        for plan in plans:
            if plan.renames or plan.blockers:
                print_rename_plan(plan)

        if not actionable:
            print('\n[yellow]Nothing to rename.[/yellow]')
            if blocked:
                raise CommandError(f'{len(blocked)} dataset(s) blocked; see above')
            return

        total_bindings = sum(sum(p.bindings.values()) for p in actionable)
        if not options['apply']:
            print(
                f'\n[bold]Plan:[/bold] {len(actionable)} dataset(s), '
                f'{sum(len(p.renames) for p in actionable)} metric(s), {total_bindings} binding(s) preserved. '
                'Re-run with --apply to write.'
            )
            if blocked:
                raise CommandError(f'{len(blocked)} dataset(s) blocked; see above')
            return

        self._apply(ic, actionable)

        print(
            f'\n[green]Renamed {sum(len(p.renames) for p in actionable)} metric(s) '
            f'across {len(actionable)} dataset(s); {total_bindings} binding(s) kept.[/green] '
            'Re-run load_dvc_dataset to bring in the data.'
        )
        if blocked:
            raise CommandError(f'{len(blocked)} dataset(s) blocked; see above')
