"""
Compare the node outputs of a database-sourced instance's draft with its published revision.

A sync or an edit changes the draft, but visitors are served the live revision until someone
publishes again (`docs/trailhead/tools.md`, *A sync writes the draft*). The public UI never asks
for the draft, so this is the way to see what a publish would change before doing it:

    python manage.py compare_draft_published mainz-bisko bayreuth-bisko
    python manage.py compare_draft_published mainz-bisko --nodes net_emissions,electricity_emissions
    python manage.py compare_draft_published --all

Every node is compared by default, cell by cell over Year and the node's dimensions, every metric.
A node that differs is reported with the number of cells that changed and the year in which its
total moved most. Nodes that exist on one side only, and nodes that fail on either side, are
listed as such.

An instance with no published revision is skipped: `get_instance(source=PUBLISHED)` falls back to
the draft there, which would compare the draft with itself and report everything identical.
Visitors of such an instance already see the draft.

Both sides compute through the normal node cache, as the site does. The cache key is the node's
hash over its spec and inputs, so a draft and a revision that differ do not share entries.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand, CommandError

import polars as pl

from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN
from nodes.models import InstanceConfig, PreferredInstanceSource

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from common.polars import PathsDataFrame
    from nodes.context import Context


@dataclass
class NodeComparison:
    """How one node's output differs between the published revision and the draft."""

    node_id: str
    problems: list[str] = field(default_factory=list)
    """Differences in shape: dimensions, metrics, units, rows on one side only, forecast flags."""
    changed_cells: int = 0
    largest_year_change: tuple[str, int, float | None, float | None] | None = None
    """(metric, year, published total, draft total) for the year whose total moved most."""

    @property
    def identical(self) -> bool:
        return not self.problems and self.changed_cells == 0


def _differs(a: float | None, b: float | None, rel_tol: float) -> bool:
    if a is None or b is None:
        return a is not b
    return abs(a - b) > rel_tol * max(1.0, abs(a), abs(b))


def _shape_problems(published: PathsDataFrame, draft: PathsDataFrame) -> list[str]:
    """Differences that make a cell-by-cell comparison meaningless: dimensions, metrics, units."""
    if sorted(published.dim_ids) != sorted(draft.dim_ids):
        return [f'dimensions {sorted(published.dim_ids)} -> {sorted(draft.dim_ids)}']
    if sorted(published.metric_cols) != sorted(draft.metric_cols):
        return [f'metrics {sorted(published.metric_cols)} -> {sorted(draft.metric_cols)}']
    return [
        f'unit of {col}: {published.get_unit(col)} -> {draft.get_unit(col)}'
        for col in published.metric_cols
        if published.get_unit(col) != draft.get_unit(col)
    ]


def _plain(df: PathsDataFrame, keys: list[str], values: list[str]) -> pl.DataFrame:
    # Categories compare by id. Two contexts build their own categorical mappings, so the
    # dimension columns are cast to strings before the join.
    return pl.DataFrame(df).select([
        pl.col(YEAR_COLUMN),
        *(pl.col(d).cast(pl.String) for d in keys[1:]),
        *(pl.col(v).cast(pl.Float64) for v in values),
    ])


def _row_problems(a: pl.DataFrame, b: pl.DataFrame, joined: pl.DataFrame, keys: list[str], has_forecast: bool) -> list[str]:
    """Rows present on one side only, and rows whose forecast flag flipped."""
    problems = []
    only_published = a.height - a.join(b, on=keys, how='semi', nulls_equal=True).height
    only_draft = b.height - b.join(a, on=keys, how='semi', nulls_equal=True).height
    if only_published:
        problems.append(f'{only_published} rows only in published')
    if only_draft:
        problems.append(f'{only_draft} rows only in draft')
    if has_forecast:
        flips = joined.filter(pl.col(FORECAST_COLUMN).ne_missing(pl.col(f'{FORECAST_COLUMN}_draft'))).height
        if flips:
            problems.append(f'forecast flag differs in {flips} rows')
    return problems


def _year_totals(df: pl.DataFrame, metric: str) -> dict[int, float | None]:
    return dict(df.group_by(YEAR_COLUMN).agg(pl.col(metric).sum()).iter_rows())


def compare_outputs(node_id: str, published: PathsDataFrame, draft: PathsDataFrame, rel_tol: float = 1e-9) -> NodeComparison:
    """Compare two outputs of the same node cell by cell; a shape difference ends the comparison there."""
    result = NodeComparison(node_id, problems=_shape_problems(published, draft))
    if result.problems:
        return result

    keys = [YEAR_COLUMN, *sorted(published.dim_ids)]
    metrics = list(published.metric_cols)
    has_forecast = FORECAST_COLUMN in published.columns and FORECAST_COLUMN in draft.columns
    values = metrics + ([FORECAST_COLUMN] if has_forecast else [])
    a, b = _plain(published, keys, values), _plain(draft, keys, values)
    joined = a.join(b, on=keys, how='full', suffix='_draft', coalesce=True, nulls_equal=True)
    result.problems = _row_problems(a, b, joined, keys, has_forecast)

    largest = 0.0
    for m in metrics:
        pairs = zip(joined[m].to_list(), joined[f'{m}_draft'].to_list(), strict=True)
        result.changed_cells += sum(1 for x, y in pairs if _differs(x, y, rel_tol))
        totals_a, totals_b = _year_totals(a, m), _year_totals(b, m)
        for year in sorted(set(totals_a) | set(totals_b)):
            x, y = totals_a.get(year), totals_b.get(year)
            change = abs((y or 0.0) - (x or 0.0))
            if _differs(x, y, rel_tol) and (result.largest_year_change is None or change > largest):
                result.largest_year_change = (m, year, x, y)
                largest = change
    return result


class Command(BaseCommand):
    help = "Compare a database-sourced instance's draft with its published revision, node by node"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instances', nargs='*', metavar='INSTANCE_ID')
        parser.add_argument('--all', action='store_true', help='Every database-sourced instance with a published revision')
        parser.add_argument('--nodes', help='Comma-separated node ids; default every node')
        parser.add_argument('--tolerance', type=float, default=1e-9, help='Relative tolerance for a changed value (default 1e-9)')

    def handle(self, *_args, **options) -> None:
        identifiers: list[str] = options['instances']
        if options['all']:
            if identifiers:
                raise CommandError('Give instance ids or --all, not both')
            qs = InstanceConfig.objects.filter(config_source='database', live_revision__isnull=False)
            identifiers = list(qs.order_by('identifier').values_list('identifier', flat=True))
        if not identifiers:
            raise CommandError('Give at least one instance id, or --all')
        node_ids = [n.strip() for n in options['nodes'].split(',')] if options['nodes'] else None

        for identifier in identifiers:
            self.compare_instance(identifier, node_ids, options['tolerance'])

    def compare_instance(self, identifier: str, node_ids: list[str] | None, rel_tol: float) -> None:
        ic = InstanceConfig.objects.filter(identifier=identifier).first()
        if ic is None:
            raise CommandError(f'No instance {identifier!r}')
        self.stdout.write(self.style.MIGRATE_HEADING(f'== {identifier}'))
        if ic.config_source != 'database':
            self.stdout.write('  skipped: yaml-sourced, so it has no published revision and visitors see the YAML')
            return
        if ic.live_revision_id is None:
            self.stdout.write('  skipped: never published, so visitors already see the draft')
            return
        self.stdout.write(f'  live revision {ic.live_revision_id}, unpublished changes: {ic.has_unpublished_changes}')

        published = ic.get_instance(source=PreferredInstanceSource.PUBLISHED).context
        draft = ic.get_instance(source=PreferredInstanceSource.DRAFT).context
        ids = node_ids or sorted(set(published.nodes) | set(draft.nodes))

        identical = 0
        for node_id in ids:
            line = self._compare_node(node_id, published, draft, rel_tol)
            if line is None:
                identical += 1
            else:
                self.stdout.write(line)
        style = self.style.SUCCESS if identical == len(ids) else self.style.WARNING
        self.stdout.write(style(f'  {identical} of {len(ids)} nodes identical'))

    def _compare_node(self, node_id: str, published: Context, draft: Context, rel_tol: float) -> str | None:
        """Return the report line for a node that differs, None for an identical one."""
        missing = [label for label, ctx in (('published', published), ('draft', draft)) if node_id not in ctx.nodes]
        if missing:
            return f'  {node_id}: not in {" or ".join(missing)}'
        outputs: dict[str, PathsDataFrame | str] = {}
        for label, ctx in (('published', published), ('draft', draft)):
            try:
                outputs[label] = ctx.get_node(node_id).get_output_pl()
            except Exception as e:  # a failing node is a finding, not the end of the run
                outputs[label] = f'{type(e).__name__}: {str(e)[:200]}'
        a, b = outputs['published'], outputs['draft']
        if isinstance(a, str) and isinstance(b, str):
            # Failing the same way on both sides is not something a publish changes.
            return None if a == b else f'  {node_id}: fails on both sides, differently: {a} -> {b}'
        if isinstance(a, str):
            return f'  {node_id}: fails in published only: {a}'
        if isinstance(b, str):
            return f'  {node_id}: fails in draft only: {b}'
        cmp = compare_outputs(node_id, a, b, rel_tol)
        if cmp.identical:
            return None
        parts = [*cmp.problems]
        if cmp.changed_cells:
            parts.append(f'{cmp.changed_cells} cells changed')
        if cmp.largest_year_change is not None:
            metric, year, x, y = cmp.largest_year_change
            parts.append(f'largest in {year}: {metric} {_fmt(x)} -> {_fmt(y)}')
        return f'  {node_id}: ' + '; '.join(parts)


def _fmt(value: float | None) -> str:
    return '—' if value is None else f'{value:.6g}'
