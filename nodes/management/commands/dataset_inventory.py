"""
Report what each of an instance's datasets holds, on both sides: the database and DVC.

``dataset_status`` answers "what do I have to run?" and deliberately reduces each dataset
to one verdict. This answers the prior question — *what actually exists, and where* — and
prints the four numbers that question turns out to mean:

    points   how many data points the database row holds, and how many the DVC copy has
    dated    when each side was last written

``load_nodes.py``'s dataset listing is the tool this replaces. It enumerates the datasets
the *runtime* resolved, so anything loading as a ``DBDataset`` — which, on an instance with
``use_datasets_from_db``, is every imported identifier — never appears, and the listing
looks short and reassuring while most of the model is missing from it.

The two point counts are directly comparable: a DB ``DataPoint`` is one (metric, year,
dimension) cell, and the DVC count here is non-null metric cells, which is the same unit.
A DVC count larger than the DB one usually means the row is behind the pin; smaller usually
means a metric was dropped upstream. Either way ``dataset_status`` will say what to do.

The dates are not the same kind of fact, and the column headings say so:

  * ``db written`` is the newest ``last_modified_at`` over the row's data points — when
    the data was last written into this database, i.e. when an import or an admin edit
    last touched it. Not when the data was collected.
  * ``dvc committed`` is the last commit **reachable from the pin** that touched
    ``<identifier>.parquet.dvc``, with its author date. Reachable from the pin, not from
    HEAD, so a dataset pushed after the pin is reported as the pin sees it — which is the
    state the model actually computes on.

``stamp`` is the commit recorded in ``external_ref['commit']`` when the row was imported.
Where it differs from ``dvc committed`` the DB copy came from an older version of the file.
A blank stamp is a row imported before provenance stamping existed, which is the state most
likely to be silently stale.

    python manage.py dataset_inventory mainz-bisko
    python manage.py dataset_inventory mainz-bisko --repo-from yaml
    python manage.py dataset_inventory mainz-bisko --order points
    python manage.py dataset_inventory mainz-bisko --csv /tmp/mainz-datasets.csv
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count, Max

from rich.console import Console
from rich.table import Table

from kausal_common.datasets.models import Dataset

from common import polars as ppl
from nodes.management.commands.dataset_status import candidate_dataset_ids
from nodes.management.commands.load_dvc_dataset import (
    apply_repo_provenance,
    resolve_repo_provenance,
)
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from datetime import datetime

    from nodes.context import Context

# The table is wide by design -- eight columns of facts about one dataset -- and rich
# would otherwise wrap every cell to fit an 80-column terminal, which makes it unreadable
# exactly when it has something to say. A fixed width plus `no_wrap` keeps one dataset on
# one line; the shell can scroll.
console = Console(width=190, soft_wrap=False)

# Absolute path so the subprocess call does not depend on PATH (ruff S607).
GIT = shutil.which('git') or '/usr/bin/git'
print = console.print


@dataclass
class Row:
    """One dataset, as the database has it and as DVC has it."""

    identifier: str
    db_points: int | None = None
    db_written: datetime | None = None
    db_stamp: str | None = None
    dvc_points: int | None = None
    dvc_commit: str | None = None
    dvc_committed: str | None = None
    note: str = ''

    @property
    def where(self) -> str:
        """
        Which sides actually hold data -- a row is not presence.

        An empty DB row must not count as presence: `load_dvc_dataset` creates the row and
        the schema before it can fail to fill it, and a dataset indexed by columns the
        instance has no dimension for (`we_from`, `ags`) leaves exactly that -- a row, a
        schema, bindings, and nothing in them. Reporting that as 'both' would report the
        placeholder as the dataset.

        Whether the DB side is an empty row or no row at all is left to the `db pts`
        column, which prints `0` against `—`. It does not belong here: this column names
        the side the data is on, and both cases have it on the same side.
        """
        db_has = bool(self.db_points)
        dvc_has = bool(self.dvc_points)
        if db_has and dvc_has:
            return 'both'
        if db_has:
            return 'db only'
        if dvc_has:
            return 'dvc only'
        if self.db_points == 0 and self.dvc_points == 0:
            return 'both empty'
        return 'neither'

    @property
    def drift(self) -> int | None:
        """DVC minus DB, in data points; None when only one side exists."""
        if self.db_points is None or self.dvc_points is None:
            return None
        return self.dvc_points - self.db_points


def db_rows(ic: InstanceConfig) -> dict[str, Row]:
    """One pass over the DB for counts, write times and provenance stamps."""
    qs = (
        Dataset.objects
        .get_queryset()
        .for_instance_config(ic)
        .annotate(n_points=Count('data_points', distinct=True), written=Max('data_points__last_modified_at'))
    )
    out: dict[str, Row] = {}
    for ds in qs:
        if not ds.identifier:
            continue
        out[ds.identifier] = Row(
            identifier=ds.identifier,
            db_points=ds.n_points,  # type: ignore[attr-defined]
            db_written=ds.written or ds.last_modified_at,  # type: ignore[attr-defined]
            db_stamp=(ds.external_ref or {}).get('commit'),
        )
    return out


class DvcDates:
    """Last commit reachable from the pin that touched each dataset's ``.dvc`` file."""

    def __init__(self, repo_dir: Path | None, pin: str | None) -> None:
        self.repo_dir = repo_dir
        self.pin = pin
        self._cache: dict[str, tuple[str, str] | None] = {}

    def for_dataset(self, ds_id: str) -> tuple[str, str] | None:
        if self.repo_dir is None or self.pin is None:
            return None
        if ds_id in self._cache:
            return self._cache[ds_id]
        result = self._git_log(f'{ds_id}.parquet.dvc')
        self._cache[ds_id] = result
        return result

    def _git_log(self, path: str) -> tuple[str, str] | None:
        assert self.repo_dir is not None
        try:
            proc = subprocess.run(  # noqa: S603
                [GIT, 'log', '-1', '--format=%h\t%ad', '--date=format:%Y-%m-%d %H:%M', str(self.pin), '--', path],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except OSError, subprocess.SubprocessError:
            return None
        line = proc.stdout.strip()
        if proc.returncode != 0 or not line or '\t' not in line:
            return None
        commit, when = line.split('\t', 1)
        return commit, when


def dvc_repo_dir(ctx: Context) -> Path | None:
    """Find the local git clone dvc_pandas keeps for the instance's dataset repo."""
    repo = getattr(ctx, 'dataset_repo', None)
    for attr in ('local_repo_dir', 'repo_dir', 'cache_dir'):
        value = getattr(repo, attr, None)
        if value:
            return Path(str(value))
    git_repo = getattr(repo, 'git_repo', None)
    working_dir = getattr(git_repo, 'working_dir', None)
    if working_dir:
        return Path(str(working_dir))
    return None


def add_dvc_side(ctx: Context, rows: dict[str, Row], ds_ids: list[str], dates: DvcDates) -> None:
    """Fill in the DVC point count and commit for every identifier that reads."""
    for ds_id in ds_ids:
        row = rows.setdefault(ds_id, Row(identifier=ds_id))
        try:
            df = ppl.from_dvc_dataset(ctx.load_dvc_dataset(ds_id))
        except Exception as exc:  # a dataset that will not read is a finding, not a crash
            if row.db_points is None:
                row.note = f'unreadable: {type(exc).__name__}'
            else:
                row.note = 'no DVC source' if not row.db_stamp else f'unreadable: {type(exc).__name__}'
            continue
        meta = df.get_meta()
        row.dvc_points = sum(df[col].drop_nulls().len() for col in meta.metric_cols)
        found = dates.for_dataset(ds_id)
        if found is not None:
            row.dvc_commit, row.dvc_committed = found


def sort_key(row: Row, order: str) -> Any:
    if order == 'points':
        return (-(row.dvc_points or row.db_points or 0), row.identifier)
    if order == 'drift':
        return (-abs(row.drift or 0), row.identifier)
    if order == 'date':
        return (row.dvc_committed or '', row.identifier)
    return (row.identifier,)


def fmt_int(value: int | None) -> str:
    return '—' if value is None else f'{value:,}'


def fmt_drift(value: int | None) -> str:
    if value is None:
        return ''
    if value == 0:
        return '[dim]0[/dim]'
    colour = 'yellow' if value > 0 else 'red'
    return f'[{colour}]{value:+,}[/{colour}]'


class Command(BaseCommand):
    help = "List an instance's datasets with data-point counts and write dates on both the DB and DVC sides"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instances', type=str, nargs='+', help='Instance identifiers')
        parser.add_argument(
            '--repo-from',
            choices=['auto', 'yaml', 'db'],
            default='auto',
            help="Which config's DVC pin to read (default: auto, i.e. the instance's own config_source)",
        )
        parser.add_argument(
            '--order',
            choices=['name', 'points', 'drift', 'date'],
            default='name',
            help='Sort order (default: name)',
        )
        parser.add_argument('--csv', type=str, help='Also write the table to this path as CSV')

    def _inventory(self, instance_id: str, repo_from: str, order: str) -> list[Row]:
        try:
            ic = InstanceConfig.objects.get(identifier=instance_id)
        except InstanceConfig.DoesNotExist as exc:
            raise CommandError(f'no instance {instance_id!r}') from exc
        ctx = ic.get_instance().context
        provenance = resolve_repo_provenance(ic, ctx, repo_from)
        apply_repo_provenance(ctx, provenance)
        pin = ctx.dataset_repo_spec.commit if ctx.dataset_repo_spec else None

        rows = db_rows(ic)
        ds_ids = candidate_dataset_ids(ic, ctx)
        repo_dir = dvc_repo_dir(ctx)
        dates = DvcDates(repo_dir, pin)
        add_dvc_side(ctx, rows, ds_ids, dates)

        ordered = sorted(rows.values(), key=lambda r: sort_key(r, order))

        print(f'\n[bold]{instance_id}[/bold]  config_source={ic.config_source}  pin={(pin or "unset")[:7]}')
        if repo_dir is None:
            print('  [yellow]DVC commit dates unavailable: could not locate the local dataset repo clone[/yellow]')

        table = Table(box=None, pad_edge=False, header_style='bold', padding=(0, 1))
        table.add_column('dataset', no_wrap=True, min_width=42)
        table.add_column('where', no_wrap=True)
        table.add_column('db pts', justify='right', no_wrap=True)
        table.add_column('dvc pts', justify='right', no_wrap=True)
        table.add_column('drift', justify='right', no_wrap=True)
        table.add_column('db written', no_wrap=True)
        table.add_column('dvc committed', no_wrap=True)
        table.add_column('stamp', no_wrap=True)
        table.add_column('note', no_wrap=True)
        for row in ordered:
            table.add_row(
                row.identifier,
                row.where if row.where != 'both' else '[dim]both[/dim]',
                fmt_int(row.db_points),
                fmt_int(row.dvc_points),
                fmt_drift(row.drift),
                row.db_written.strftime('%Y-%m-%d %H:%M') if row.db_written else '—',
                f'{row.dvc_committed}  {row.dvc_commit}' if row.dvc_committed else '—',
                (row.db_stamp or '')[:7] or '[red]none[/red]',
                row.note,
            )
        print(table)

        counts: dict[str, int] = {}
        for row in ordered:
            counts[row.where] = counts.get(row.where, 0) + 1
        drifted = [r for r in ordered if r.where == 'both' and r.drift]
        unstamped = [r for r in ordered if r.db_points is not None and not r.db_stamp]
        print(f'  {len(ordered)} datasets: ' + ', '.join(f'{n} {where}' for where, n in sorted(counts.items())))
        print(
            f'  {len(drifted)} populated on both sides but differing in count; {len(unstamped)} DB rows carry no provenance stamp'
        )
        placeholders = [r for r in ordered if r.where == 'dvc only' and r.db_points == 0]
        if placeholders:
            print(
                f'  [dim]{len(placeholders)} of the "dvc only" rows have an empty DB row and schema rather than '
                'no row at all (db pts 0, not —) — usually an index column this instance has no dimension for[/dim]'
            )
        return ordered

    def handle(self, *args: Any, **options: Any) -> None:
        all_rows: list[tuple[str, Row]] = []
        for instance_id in options['instances']:
            rows = self._inventory(instance_id, options['repo_from'], options['order'])
            all_rows.extend((instance_id, row) for row in rows)

        csv_path = options.get('csv')
        if not csv_path:
            return
        path = Path(csv_path)
        fields = ['instance', *list(asdict(all_rows[0][1]).keys()), 'where', 'drift'] if all_rows else []
        with path.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            for instance_id, row in all_rows:
                record: dict[str, Any] = {'instance': instance_id, **asdict(row)}
                record['where'] = row.where
                record['drift'] = row.drift
                if row.db_written is not None:
                    record['db_written'] = row.db_written.isoformat()
                writer.writerow(record)
        print(f'\nWrote {len(all_rows)} rows to {path}')
