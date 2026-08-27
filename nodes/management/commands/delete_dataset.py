"""
Delete a dataset row and its data points from the database, with the checks that make it safe.

There was no command for this, so it was done from ``shell_plus``, which is the wrong place
for it: a bare ``Dataset.objects.filter(identifier=...).delete()`` takes every city's row of
that name, leaves the schema behind as an orphan, and either raises ``ProtectedError``
halfway through or silently removes a dataset the model still binds. In production none of
those are recoverable by re-running something.

    python manage.py delete_dataset mainz-bisko bisko/energy_costs            # plan only
    python manage.py delete_dataset mainz-bisko bisko/energy_costs --apply --dump-to /tmp/x
    python manage.py delete_dataset mainz-bisko a b c --apply --no-dump

Nothing is written without ``--apply``, and ``--apply`` refuses unless you have chosen
either ``--dump-to DIR`` or ``--no-dump``: a delete here destroys data points, and for a
dataset with no DVC source there is no other copy anywhere.

## What it refuses, and why

A refusal stops the **whole set**, as in ``rename_dataset``: a half-deleted group is harder
to reason about than one that has not moved.

``ambiguous``
    The identifier resolves to more than one row visible to this instance. Never guess
    which.

``foreign scope``
    The row's scope is not the named instance -- it belongs to a framework or another
    instance and is merely *visible* here. Deleting it would remove it from every scope
    holder at once.

``published revision``
    An ``InstanceRevisionDatasetPin`` names it. The pin records what a published revision
    computed from, so deleting the row falsifies history. **No override**: publish a new
    revision instead.

``model still binds it``
    A ``NodeInputPortBinding``, ``DatasetPort`` or ``NodeDataset`` points at the row. These
    are the authoritative signal and the only one that survives ``dataset_replacements``:
    the loaded dataset object carries the *module's* declared id (``kommune/...``) while the
    binding points at the row the replacement resolved to (``mainz/...``), so a scan of node
    dataset ids reports such a row as unused when three bindings hold it.
    ``--clear-bindings`` overrides, and should be used only together with removing the
    declaration from ``configs/`` -- otherwise the next ``sync_instance_to_db`` rebuilds the
    bindings (``reconcile_input_bindings`` recreates the set from the spec on every run) and
    re-creates the row as an empty placeholder.

``still named in configs``
    A textual scan of ``configs/`` found the identifier. Reported as a refusal because a
    deleted-but-declared dataset comes back on the next sync, empty, and an empty row that
    the model binds is the failure mode that took 700 kt out of the Mainz balance without
    an error. ``--ignore-configs`` overrides -- legitimate when the hit is another city's
    config or a module this instance does not include.

## What it reports rather than refuses

**Recoverability.** A row stamped with ``external_ref['commit']`` whose DVC path still reads
can be restored with ``load_dvc_dataset``. One without cannot: it was authored in the admin
or its source has gone, and the database is the only copy. The plan says which, per row,
because it changes what a mistake costs.

**The schema.** ``Dataset.schema`` is ``PROTECT``, so deleting the dataset leaves the schema
with its metrics and dimensions behind. Where the schema serves only this dataset it is
deleted too; where it is shared it is kept, and the plan says so.

Data points, their comments and their source references all ``CASCADE`` off the dataset.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from rich.console import Console

from kausal_common.datasets.models import DataPointComment, Dataset, DatasetSchema

from nodes.models import (
    DatasetPort,
    InstanceConfig,
    InstanceRevisionDatasetPin,
    NodeDataset,
    NodeInputPortBinding,
)

if TYPE_CHECKING:
    from argparse import ArgumentParser

console = Console()
print = console.print

CONFIG_ROOT = Path('configs')

# Dimension names here are mostly German, and this repo transliterates rather than strips:
# identifiers read `fernwaerme`, not `fernwrme`. Without this map an `Energieträger` column
# header slugs to `energietr_ger`.
UMLAUTS = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss', 'å': 'aa', 'é': 'e'}


@dataclass
class DeletePlan:
    """What deleting one identifier would do, and every reason it may be refused."""

    identifier: str
    dataset: Dataset | None = None
    data_points: int = 0
    schema: DatasetSchema | None = None
    schema_shared_by: int = 0
    bindings: dict[str, int] = field(default_factory=dict)
    pins: int = 0
    config_hits: dict[str, int] = field(default_factory=dict)
    config_mentions: dict[str, int] = field(default_factory=dict)
    recoverable_from: str | None = None
    blockers: list[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return self.dataset is not None and not self.blockers

    @property
    def deletes_schema(self) -> bool:
        return self.schema is not None and self.schema_shared_by == 1


def config_references(identifier: str) -> tuple[dict[str, int], dict[str, int]]:
    """
    Find the identifier in `configs/`, separating declarations from comment mentions.

    `rename_dataset.config_references` counts raw occurrences, which is right for its job --
    it is telling an operator what is left to edit by hand. Here the count decides a
    refusal, and a mention in a comment is not a declaration: this file's own explanation of
    *why* a `dataset_replacements` entry was removed names the identifier, and counting that
    as a live reference would refuse a delete on the strength of its own documentation.

    Split rather than stripped, and comments counted rather than discarded. Stripping
    everything after a `#` would silently under-count an identifier inside a quoted string,
    and under-counting is the unsafe direction here -- it would let a still-declared dataset
    be deleted. So both are reported, and only the declaration side blocks.
    """
    if not CONFIG_ROOT.is_dir():
        return {}, {}
    declarations: dict[str, int] = {}
    mentions: dict[str, int] = {}
    for path in sorted(CONFIG_ROOT.rglob('*.yaml')):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        decl = sum(line.count(identifier) for line in lines if not line.lstrip().startswith('#'))
        ment = sum(line.count(identifier) for line in lines if line.lstrip().startswith('#'))
        if decl:
            declarations[str(path)] = decl
        if ment:
            mentions[str(path)] = ment
    return declarations, mentions


def recoverability(dataset: Dataset) -> str | None:
    """Return the commit a row could be re-imported from, or None if this database is the only copy."""
    ref = dataset.external_ref or {}
    commit = ref.get('commit')
    if not commit:
        return None
    path = ref.get('dataset_id') or dataset.identifier
    return f'{path} @ {commit[:7]}'


def build_plan(
    ic: InstanceConfig,
    identifier: str,
    *,
    allow_missing: bool,
    clear_bindings: bool,
    ignore_configs: bool,
) -> DeletePlan:
    plan = DeletePlan(identifier=identifier)

    matches = list(Dataset.objects.get_queryset().for_instance_config(ic).filter(identifier=identifier).select_related('schema'))
    if not matches:
        if not allow_missing:
            plan.blockers.append('no dataset with this identifier is visible to this instance')
        return plan
    if len(matches) > 1:
        pks = ', '.join(str(d.pk) for d in matches)
        plan.blockers.append(f'ambiguous: {len(matches)} rows visible here (pks {pks})')
        return plan

    dataset = matches[0]
    plan.dataset = dataset
    plan.data_points = dataset.data_points.count()
    plan.schema = dataset.schema
    plan.schema_shared_by = dataset.schema.datasets.count() if dataset.schema else 0
    plan.recoverable_from = recoverability(dataset)

    # The scope is a generic FK. Where it is not this instance the row is shared, and a
    # delete would remove it from every holder -- refuse rather than surprise the others.
    scope = dataset.scope
    if not (isinstance(scope, InstanceConfig) and scope.pk == ic.pk):
        plan.blockers.append(f'foreign scope: owned by {scope!r}, not by {ic.identifier}')

    plan.pins = InstanceRevisionDatasetPin.objects.filter(dataset=dataset).count()
    if plan.pins:
        # Deliberately not overridable. A pin is a record of what a published revision
        # used; removing the row it names makes that record a lie.
        plan.blockers.append(f'published revision pins it ({plan.pins}) -- publish a new revision instead')

    plan.bindings = {
        name: count
        for name, count in (
            ('NodeInputPortBinding', NodeInputPortBinding.objects.filter(dataset=dataset).count()),
            ('DatasetPort', DatasetPort.objects.filter(dataset=dataset).count()),
            ('NodeDataset', NodeDataset.objects.filter(dataset=dataset).count()),
        )
        if count
    }
    if plan.bindings and not clear_bindings:
        held = ', '.join(f'{name} {count}' for name, count in plan.bindings.items())
        plan.blockers.append(f'model still binds it ({held}) -- pass --clear-bindings if that is intended')

    plan.config_hits, plan.config_mentions = config_references(identifier)
    if plan.config_hits and not ignore_configs:
        where = ', '.join(sorted(plan.config_hits))
        plan.blockers.append(f'still named in {where} -- the next sync would recreate it; --ignore-configs overrides')

    return plan


def binding_detail(dataset: Dataset) -> list[str]:
    """Name the nodes that bind the dataset, so the operator can judge the override."""
    out = []
    for binding in NodeInputPortBinding.objects.filter(dataset=dataset).select_related('node'):
        node = binding.node.identifier if binding.node else '?'
        out.append(f'{node} ({binding.metric})')
    for port in DatasetPort.objects.filter(dataset=dataset).select_related('node'):
        node = getattr(port, 'node', None)
        out.append(f'port on {getattr(node, "identifier", node)}')
    return out


def _print_missing(plan: DeletePlan) -> None:
    if plan.blockers:
        print(f'[red]refused[/red]  {plan.identifier}')
        for blocker in plan.blockers:
            print(f'          [red]{blocker}[/red]')
    else:
        print(f'[dim]missing[/dim]  {plan.identifier}  (--allow-missing, treated as done)')


def _print_detail(plan: DeletePlan, dataset: Dataset) -> None:
    if plan.recoverable_from:
        print(f'          [green]recoverable[/green] from DVC: {plan.recoverable_from}')
    else:
        print('          [red]NOT recoverable[/red]: no DVC provenance, this database is the only copy')
    if plan.schema is not None:
        if plan.deletes_schema:
            print(f'          schema {plan.schema.pk} serves only this dataset and goes with it')
        else:
            print(f'          schema {plan.schema.pk} is shared by {plan.schema_shared_by} datasets and is kept')
    if plan.bindings:
        held = ', '.join(f'{name} {count}' for name, count in plan.bindings.items())
        nodes = binding_detail(dataset)
        print(f'          bindings: {held}')
        for node in nodes[:6]:
            print(f'            - {node}')
        if len(nodes) > 6:
            print(f'            ... and {len(nodes) - 6} more')
    for path, count in sorted(plan.config_hits.items()):
        print(f'          declared in {path} ({count}x)')
    for path, count in sorted(plan.config_mentions.items()):
        print(f'          [dim]mentioned in a comment in {path} ({count}x) -- not a declaration[/dim]')


def print_plan(plan: DeletePlan) -> None:
    """Report one identifier: what would go, what it costs, and why it may be refused."""
    if plan.dataset is None:
        _print_missing(plan)
        return
    dataset = plan.dataset
    style = 'red' if plan.blockers else 'yellow'
    verdict = 'refused' if plan.blockers else 'delete'
    print(f'[{style}]{verdict:7}[/{style}]  {plan.identifier}  pk={dataset.pk}  points={plan.data_points:,}')
    _print_detail(plan, dataset)
    for blocker in plan.blockers:
        print(f'          [red]{blocker}[/red]')


def dump_dataset(dataset: Dataset, out_dir: Path) -> Path:
    """Write every data point to CSV before it is destroyed, dimensions and comments included."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = dataset.identifier.replace('/', '__') if dataset.identifier else f'dataset-{dataset.pk}'
    path = out_dir / f'{name}.csv'

    points = (
        dataset.data_points.all().select_related('metric').prefetch_related('dimension_categories__dimension').order_by('date')
    )
    # One query for every comment rather than a per-point prefetch: `DataPoint.comments` is
    # a reverse manager the type checker cannot see, and going through the child model is
    # both typed and a query cheaper.
    comments_by_point: dict[int, list[str]] = {}
    comment_rows = DataPointComment.objects.filter(data_point__dataset=dataset).values_list('data_point', 'text')
    for point_id, text in comment_rows:
        if point_id is not None and text:
            comments_by_point.setdefault(point_id, []).append(text)

    # `Dimension` carries only a translated `name`, no identifier, so the column key is a
    # slug of it -- de-duplicated by primary key, because two dimensions can legitimately
    # share a display name and silently merging their columns would corrupt the dump.
    dim_key: dict[int, str] = {}
    dim_names: list[str] = []

    def key_for(dimension: Any) -> str:
        if dimension.pk in dim_key:
            return dim_key[dimension.pk]
        name = (dimension.name or 'dimension').lower()
        for umlaut, replacement in UMLAUTS.items():
            name = name.replace(umlaut, replacement)
        base = re.sub(r'[^0-9a-z]+', '_', name).strip('_') or 'dimension'
        candidate = base
        if candidate in dim_names:
            candidate = f'{base}_{dimension.uuid.hex[:6]}'
        dim_key[dimension.pk] = candidate
        dim_names.append(candidate)
        return candidate

    rows: list[dict[str, Any]] = []
    for point in points:
        row: dict[str, Any] = {
            'date': point.date.isoformat() if point.date else '',
            'metric': str(point.metric),
            'value': point.value,
            'uuid': str(point.uuid),
        }
        for category in point.dimension_categories.all():
            row[key_for(category.dimension)] = category.identifier
        texts = comments_by_point.get(point.pk)
        if texts:
            row['comments'] = ' ;; '.join(texts)
        rows.append(row)

    fields = ['date', 'metric', 'value', *sorted(dim_names), 'comments', 'uuid']
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return path


class Command(BaseCommand):
    help = 'Delete dataset rows and their data points from one instance, with safety checks'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instance', help='Instance identifier that owns the datasets')
        parser.add_argument('identifiers', nargs='+', help='Dataset identifiers to delete')
        parser.add_argument('--apply', action='store_true', help='Delete (default is a plan only)')
        parser.add_argument('--dump-to', metavar='DIR', help='Write the data points to CSV here before deleting')
        parser.add_argument('--no-dump', action='store_true', help='Delete without keeping a copy')
        parser.add_argument(
            '--allow-missing',
            action='store_true',
            help='Treat an identifier with no row as already done instead of an error',
        )
        parser.add_argument(
            '--clear-bindings',
            action='store_true',
            help='Delete the NodeInputPortBinding / DatasetPort / NodeDataset rows that hold it first',
        )
        parser.add_argument(
            '--ignore-configs',
            action='store_true',
            help='Proceed even though configs/ still names the identifier',
        )

    @staticmethod
    def _check_backup_choice(options: dict[str, Any]) -> None:
        """Refuse to delete until the operator has decided about a copy."""
        if options['dump_to'] and options['no_dump']:
            raise CommandError('Choose either --dump-to or --no-dump, not both')
        if options['apply'] and not (options['dump_to'] or options['no_dump']):
            raise CommandError(
                'Deleting destroys data points. Pass --dump-to DIR to keep a copy, or --no-dump to say you do not want one.'
            )

    @staticmethod
    def _summarise(actionable: list[DeletePlan]) -> None:
        points = sum(p.data_points for p in actionable)
        schemas = sum(1 for p in actionable if p.deletes_schema)
        unrecoverable = [p for p in actionable if not p.recoverable_from]
        print(f'\n{len(actionable)} dataset(s), {points:,} data point(s) and {schemas} schema(s) would go')
        if unrecoverable:
            names = ', '.join(p.identifier for p in unrecoverable)
            print(f'[red]{len(unrecoverable)} of them cannot be restored from DVC: {names}[/red]')

    @staticmethod
    def _delete(actionable: list[DeletePlan]) -> None:
        with transaction.atomic():
            for plan in actionable:
                dataset = plan.dataset
                assert dataset is not None
                if plan.bindings:
                    NodeInputPortBinding.objects.filter(dataset=dataset).delete()
                    DatasetPort.objects.filter(dataset=dataset).delete()
                    NodeDataset.objects.filter(dataset=dataset).delete()
                schema = plan.schema
                deletes_schema = plan.deletes_schema
                dataset.delete()
                if deletes_schema and schema is not None and not schema.datasets.exists():
                    schema.delete()
                print(f'deleted {plan.identifier} ({plan.data_points:,} data points)')

    def handle(self, *args: Any, **options: Any) -> None:
        self._check_backup_choice(options)
        apply_changes: bool = options['apply']
        dump_to: str | None = options['dump_to']

        try:
            ic = InstanceConfig.objects.get(identifier=options['instance'])
        except InstanceConfig.DoesNotExist as exc:
            raise CommandError(f'no instance {options["instance"]!r}') from exc

        print(f'\n[bold]{ic.identifier}[/bold]  config_source={ic.config_source}')
        plans = [
            build_plan(
                ic,
                identifier,
                allow_missing=options['allow_missing'],
                clear_bindings=options['clear_bindings'],
                ignore_configs=options['ignore_configs'],
            )
            for identifier in options['identifiers']
        ]
        for plan in plans:
            print_plan(plan)

        blocked = [p for p in plans if p.blockers]
        if blocked:
            raise CommandError(f'{len(blocked)} refused; nothing was deleted. See above.')

        actionable = [p for p in plans if p.is_actionable]
        self._summarise(actionable)

        if not actionable:
            print('[dim]nothing to do[/dim]')
            return
        if not apply_changes:
            print('\n[dim]Plan only. Re-run with --apply (and --dump-to DIR or --no-dump).[/dim]')
            return

        if dump_to:
            out_dir = Path(dump_to)
            dumped = [dump_dataset(p.dataset, out_dir) for p in actionable if p.dataset is not None]
            print(f'\nWrote {len(dumped)} dump file(s) to {out_dir}')

        self._delete(actionable)
        ic.invalidate_cache()
        print('\n[green]Done.[/green] Cache invalidated.')
        if options['clear_bindings']:
            print(
                '[yellow]--clear-bindings was used: remove the declaration from configs/ too, '
                'or the next sync_instance_to_db rebuilds the bindings and re-creates the row empty.[/yellow]'
            )
