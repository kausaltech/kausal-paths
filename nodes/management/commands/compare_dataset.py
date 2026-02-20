from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand, CommandError

from loguru import logger
from rich.console import Console
from rich.table import Table

from kausal_common.datasets.models import Dataset as DBDatasetModel

from common import polars as ppl
from nodes.dataset_diff import RowDiff, compute_row_diff, compute_schema_diff
from nodes.datasets import DBDataset, FixedDataset
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl

    from nodes.context import Context

console = Console()


@dataclass
class DataPointInfo:
    database_pk: int
    last_modified_at: datetime | None
    last_modified_by_name: str | None
    comments: list[dict] = field(default_factory=list)
    source_references: list[dict] = field(default_factory=list)


def enrich_data_points(dp_pks: set[int]) -> dict[int, DataPointInfo]:
    from django.contrib.postgres.expressions import ArraySubquery
    from django.db.models.expressions import F, OuterRef, Value
    from django.db.models.functions import Coalesce, JSONObject

    from kausal_common.datasets.models import DataPoint, DataPointComment, DatasetSourceReference

    if not dp_pks:
        return {}

    comments_qs = DataPointComment.objects_including_soft_deleted.filter(
        data_point=OuterRef('pk'),
    ).values(
        json=JSONObject(
            text=F('text'),
            created_by=Coalesce(F('created_by__first_name'), Value('')),
            created_at=F('created_at'),
            is_soft_deleted=F('is_soft_deleted'),
        ),
    )

    source_refs_qs = DatasetSourceReference.objects.filter(
        data_point=OuterRef('pk'),
    ).values(
        json=JSONObject(
            name=F('data_source__name'),
            authority=Coalesce(F('data_source__authority'), Value('')),
            url=Coalesce(F('data_source__url'), Value('')),
        ),
    )

    dps = DataPoint.objects.filter(pk__in=dp_pks).annotate(
        comments_arr=ArraySubquery(comments_qs),
        source_refs_arr=ArraySubquery(source_refs_qs),
        modifier_name=Coalesce(F('last_modified_by__email'), Value('')),
    )

    result: dict[int, DataPointInfo] = {}
    for dp in dps:
        result[dp.pk] = DataPointInfo(
            database_pk=dp.pk,
            last_modified_at=dp.last_modified_at,
            last_modified_by_name=dp.modifier_name or None,  # type: ignore[attr-defined]
            comments=dp.comments_arr,  # type: ignore[attr-defined]
            source_references=dp.source_refs_arr,  # type: ignore[attr-defined]
        )
    return result


def _collect_dp_pks(*dfs: pl.DataFrame) -> set[int]:
    pks: set[int] = set()
    for df in dfs:
        for col in df.columns:
            if col.startswith('_dp_pk_'):
                pks.update(v for v in df[col].drop_nulls().to_list() if v is not None)
    return pks


def get_config_dataset_ids(ic: InstanceConfig) -> set[str]:
    """Get all dataset IDs referenced in the instance config (excluding fixed datasets)."""
    instance = ic.get_instance()
    ds_ids: set[str] = set()
    for node in instance.context.nodes.values():
        for ds in node.input_dataset_instances:
            if isinstance(ds, FixedDataset):
                continue
            ds_ids.add(ds.id)
    return ds_ids


def get_db_dataset_ids(ic: InstanceConfig) -> set[str]:
    return set(
        DBDatasetModel.objects.for_instance_config(ic)   # type: ignore[arg-type]
        .values_list('identifier', flat=True)
    )


def find_overlapping_datasets() -> None:
    table = Table(title='Datasets existing in both DVC and DB')
    table.add_column('Instance')
    table.add_column('Overlapping dataset IDs')
    has_overlap = False

    ics = InstanceConfig.objects.all().order_by('identifier')
    for ic in ics:
        db_ids = get_db_dataset_ids(ic)
        if not db_ids:
            continue

        try:
            config_ids = get_config_dataset_ids(ic)
        except Exception:
            logger.exception('Failed to load instance %s' % ic.identifier)
            continue

        overlap = config_ids & db_ids
        if overlap:
            has_overlap = True
            table.add_row(ic.identifier, ', '.join(sorted(overlap)))

    if has_overlap:
        console.print(table)
    else:
        console.print('[green]No overlapping datasets found.[/green]')


def print_contents(label: str, df: ppl.PathsDataFrame):
    console.rule(f'{label} — Contents')
    console.print(df)


def diff_schemas(dvc_df: ppl.PathsDataFrame, db_df: ppl.PathsDataFrame) -> bool:
    """Compare schemas of two DataFrames and print differences. Returns True if identical."""
    console.rule('Schema diff')
    sd = compute_schema_diff(dvc_df, db_df)

    if sd.pk_diff is not None:
        console.print(f'[yellow]Primary keys differ: DVC={sd.pk_diff[0]}, DB={sd.pk_diff[1]}[/yellow]')

    if sd.dvc_only_cols:
        console.print(f'[yellow]Columns only in DVC: {sd.dvc_only_cols}[/yellow]')
    db_only_user_cols = [c for c in sd.db_only_cols if not c.startswith('_dp_pk_')]
    if db_only_user_cols:
        console.print(f'[yellow]Columns only in DB: {db_only_user_cols}[/yellow]')

    for col, (dvc_dt, db_dt) in sd.dtype_diffs.items():
        console.print(f'[yellow]Column "{col}" dtype differs: DVC={dvc_dt}, DB={db_dt}[/yellow]')

    for col, (dvc_unit, db_unit) in sd.unit_diffs.items():
        console.print(f'[yellow]Column "{col}" unit differs: DVC={dvc_unit}, DB={db_unit}[/yellow]')

    if sd.identical:
        console.print('[green]Schemas are identical.[/green]')
    return sd.identical


def _split_constant_cols(df: pl.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    """Split cols into (constant, varying) based on whether all rows have the same value."""
    constant: list[str] = []
    varying: list[str] = []
    for col in cols:
        if df[col].n_unique() <= 1:
            constant.append(col)
        else:
            varying.append(col)
    return constant, varying


def _print_constant_cols(df: pl.DataFrame, constant_cols: list[str], header: str) -> None:
    """Print columns that have the same value across all rows."""
    if not constant_cols:
        return
    first_row = df.row(0, named=True)
    labels = [(col, str(first_row[col])) for col in constant_cols]
    max_label = max(len(lbl) for lbl, _ in labels)
    console.print(f'\n[dim]{header}[/dim]')
    for lbl, val in labels:
        console.print(f'  {lbl:>{max_label}} : {val}')


def _split_identical_diff_cols(rd: RowDiff) -> tuple[list[str], list[str]]:
    """Split value_cols into (identical_cols, differing_cols) across all value_diffs rows."""
    identical: list[str] = []
    differing: list[str] = []
    for col in rd.value_cols:
        all_same = all(
            str(row[col]) == str(row[f'{col}_db'])
            for row in rd.value_diffs.iter_rows(named=True)
        )
        if all_same:
            identical.append(col)
        else:
            differing.append(col)
    return identical, differing


def _print_identical_diff_cols(rd: RowDiff, identical_cols: list[str]) -> None:
    """Print columns that have identical DVC/DB values across all differing rows."""
    if not identical_cols:
        return
    first_row = rd.value_diffs.row(0, named=True)
    labels: list[tuple[str, str]] = []
    for col in identical_cols:
        dvc_val = str(first_row[col])
        db_val = str(first_row[f'{col}_db'])
        assert dvc_val == db_val, f'Column {col!r} expected identical but DVC={dvc_val!r} != DB={db_val!r}'
        labels.append((col, dvc_val))
    max_label = max(len(lbl) for lbl, _ in labels)
    console.print('\n[dim]Columns with identical values across all rows listed below:[/dim]')
    for lbl, val in labels:
        console.print(f'  {lbl:>{max_label}} : {val}')


def _get_dp_pk_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('_dp_pk_')]


def _metric_name_from_dp_pk_col(col: str) -> str:
    return col.removeprefix('_dp_pk_')


def _print_dp_info(info: DataPointInfo, label_width: int, *, verbose: bool = False) -> None:
    mod_str = 'unknown'
    if info.last_modified_at is not None:
        mod_str = info.last_modified_at.strftime('%Y-%m-%d %H:%M %Z')
        if info.last_modified_by_name:
            mod_str += f' by {info.last_modified_by_name}'
    if verbose:
        console.print(f'  {"database primary key":>{label_width}} : {info.database_pk}')
    console.print(f'  {"last modified":>{label_width}} : {mod_str}')
    for comment in info.comments:
        deleted = ' [deleted]' if comment.get('is_soft_deleted') else ''
        created_by = comment.get('created_by', '')
        created_at = str(comment.get('created_at', ''))[:16]
        console.print(f'  {"comment":>{label_width}} : "{comment.get("text", "")}" ({created_by}, {created_at}){deleted}')
    for ref in info.source_references:
        parts = [ref.get('name', '')]
        if ref.get('authority'):
            parts.append(f'({ref["authority"]})')
        if ref.get('url'):
            parts.append(ref['url'])
        console.print(f'  {"source":>{label_width}} : {" ".join(parts)}')


def _print_enrichment_for_row(
    row: dict, dp_pk_cols: list[str], dp_info: dict[int, DataPointInfo],
    *, show_missing: bool = False, label_width: int = 16, verbose: bool = False,
) -> None:
    found_any = False
    for col in dp_pk_cols:
        pk = row.get(col)
        if pk is None:
            if show_missing:
                metric = _metric_name_from_dp_pk_col(col)
                console.print(f'  [dim]── {metric} (no DataPoint in DB) ──[/dim]')
            continue
        info = dp_info.get(int(pk))
        if info is None:
            continue
        found_any = True
        console.print(f'  [dim]── {_metric_name_from_dp_pk_col(col)} ──[/dim]')
        _print_dp_info(info, label_width, verbose=verbose)
    if not found_any and not show_missing:
        console.print('  [dim](no DataPoint information available for this row)[/dim]')


def _build_row_table(df: pl.DataFrame, pk_cols: list[str], value_cols: list[str]) -> Table:
    table = Table(show_lines=True)
    for col in pk_cols:
        table.add_column(col, style='bold')
    for col in value_cols:
        table.add_column(col)
    for row in df.iter_rows(named=True):
        cells = [str(row[col]) for col in pk_cols]
        cells.extend(str(row[col]) for col in value_cols)
        table.add_row(*cells)
    return table


def _build_value_diff_table(rd: RowDiff, differing_cols: list[str]) -> Table:
    table = Table(show_lines=True)
    for col in rd.pk_cols:
        table.add_column(col, style='bold')
    for col in differing_cols:
        table.add_column(f'{col} (DVC)')
        table.add_column(f'{col} (DB)')

    for row in rd.value_diffs.iter_rows(named=True):
        cells = [str(row[col]) for col in rd.pk_cols]
        for col in differing_cols:
            dvc_val = str(row[col])
            db_val = str(row[f'{col}_db'])
            if dvc_val != db_val:
                cells.extend([f'[red]{dvc_val}[/red]', f'[red]{db_val}[/red]'])
            else:
                cells.extend([dvc_val, db_val])
        table.add_row(*cells)
    return table


def _print_vertical_rows(
    df: pl.DataFrame, cols: list[str],
    *, dp_info: dict[int, DataPointInfo] | None = None, verbose: bool = False,
) -> None:
    """Print rows of a plain DataFrame in vertical label : value format."""
    dp_pk_cols = _get_dp_pk_cols(df)
    max_label = max(len(c) for c in cols)
    total = len(df)
    for i, row in enumerate(df.iter_rows(named=True)):
        console.rule(f'Row {i + 1}/{total}')
        for col in cols:
            console.print(f'  {col:>{max_label}} : {row[col]}')
        if dp_pk_cols:
            _print_enrichment_for_row(
                row, dp_pk_cols, dp_info or {}, show_missing=False, label_width=max_label, verbose=verbose,
            )


def _print_only_rows(
    df: pl.DataFrame, pk_cols: list[str], value_cols: list[str],
    *, vertical: bool, dp_info: dict[int, DataPointInfo] | None = None, verbose: bool = False,
) -> None:
    """Print rows that exist only in one source, extracting constant columns."""
    present_value_cols = [c for c in value_cols if c in df.columns]
    constant_cols, varying_cols = _split_constant_cols(df, present_value_cols)
    _print_constant_cols(df, constant_cols, 'Columns with identical values across all rows listed below:')
    if vertical:
        _print_vertical_rows(df, pk_cols + varying_cols, dp_info=dp_info, verbose=verbose)
    else:
        console.print(_build_row_table(df, pk_cols, varying_cols))


def _print_vertical_diffs(
    rd: RowDiff, differing_cols: list[str],
    *, dp_info: dict[int, DataPointInfo] | None = None, verbose: bool = False,
) -> None:
    """Print each differing row as a vertical block with PK + differing value columns."""
    dp_pk_cols = _get_dp_pk_cols(rd.value_diffs)
    total = len(rd.value_diffs)
    all_labels: list[str] = list(rd.pk_cols)
    for col in differing_cols:
        all_labels.append(f'{col} (DVC)')
        all_labels.append(f'{col} (DB)')
    max_label = max(len(lbl) for lbl in all_labels)

    for i, row in enumerate(rd.value_diffs.iter_rows(named=True)):
        console.rule(f'Row {i + 1}/{total}')
        for col in rd.pk_cols:
            console.print(f'  {col:>{max_label}} : {row[col]}')
        for col in differing_cols:
            dvc_val = str(row[col])
            db_val = str(row[f'{col}_db'])
            differs = dvc_val != db_val
            dvc_display = f'[red]{dvc_val}[/red]' if differs else dvc_val
            db_display = f'[red]{db_val}[/red]' if differs else db_val
            dvc_label = f'{col} (DVC)'
            db_label = f'{col} (DB)'
            console.print(f'  {dvc_label:>{max_label}} : {dvc_display}')
            console.print(f'  {db_label:>{max_label}} : {db_display}')
        if dp_pk_cols:
            _print_enrichment_for_row(
                row, dp_pk_cols, dp_info or {}, show_missing=False, label_width=max_label, verbose=verbose,
            )


def _maybe_enrich(vertical: bool, rd: RowDiff) -> dict[int, DataPointInfo] | None:
    if not vertical:
        return None
    all_dp_pks = _collect_dp_pks(rd.db_only, rd.value_diffs)
    if not all_dp_pks:
        return None
    try:
        return enrich_data_points(all_dp_pks)
    except Exception as e:
        console.print(f'[yellow]Could not enrich data points: {e}[/yellow]')
        return None


def _print_exclusive_rows(
    rd: RowDiff, *, vertical: bool, dp_info: dict[int, DataPointInfo] | None, verbose: bool = False,
) -> None:
    if len(rd.dvc_only):
        console.print(f'[yellow]{len(rd.dvc_only)} row(s) only in DVC:[/yellow]')
        _print_only_rows(rd.dvc_only, rd.pk_cols, rd.value_cols, vertical=vertical, verbose=verbose)
    if len(rd.db_only):
        console.print(f'[yellow]{len(rd.db_only)} row(s) only in DB:[/yellow]')
        _print_only_rows(rd.db_only, rd.pk_cols, rd.value_cols, vertical=vertical, dp_info=dp_info, verbose=verbose)


def _print_value_diffs(
    rd: RowDiff, *, vertical: bool, dp_info: dict[int, DataPointInfo] | None, verbose: bool = False,
) -> None:
    if not rd.value_cols:
        return

    if len(rd.value_diffs) == 0:
        console.print(f'[green]All {rd.matched_count} matched row(s) have identical values.[/green]')
        return

    console.print('\n\n')
    console.print(
        f'[yellow]{len(rd.value_diffs)} row(s) with value differences (out of {rd.matched_count} matched):[/yellow]'
    )

    identical_cols, differing_cols = _split_identical_diff_cols(rd)
    _print_identical_diff_cols(rd, identical_cols)

    if not differing_cols:
        return

    if vertical:
        _print_vertical_diffs(rd, differing_cols, dp_info=dp_info, verbose=verbose)
    else:
        console.print(_build_value_diff_table(rd, differing_cols))


def diff_rows(
    dvc_df: ppl.PathsDataFrame, db_df: ppl.PathsDataFrame,
    *, vertical: bool = False, dvc_modified_at: datetime | None = None, verbose: bool = False,
):
    """Compare row contents after normalizing order. Print differing rows."""
    console.rule('Row diff')

    if dvc_modified_at is not None:
        console.print(f'DVC commit date: {dvc_modified_at.strftime("%Y-%m-%d %H:%M %Z")}')

    sd = compute_schema_diff(dvc_df, db_df)
    if sd.dvc_row_count != sd.db_row_count:
        console.print(f'[yellow]Row count differs: DVC={sd.dvc_row_count}, DB={sd.db_row_count}[/yellow]')
    else:
        console.print(f'Row count: {sd.dvc_row_count}')

    rd = compute_row_diff(dvc_df, db_df)
    if rd is None:
        common_cols = set(dvc_df.columns) & set(db_df.columns)
        if not common_cols:
            console.print('[red]No common columns to compare.[/red]')
        else:
            console.print('[red]No common primary keys — cannot align rows.[/red]')
        return

    dp_info = _maybe_enrich(vertical, rd)
    _print_exclusive_rows(rd, vertical=vertical, dp_info=dp_info, verbose=verbose)
    _print_value_diffs(rd, vertical=vertical, dp_info=dp_info, verbose=verbose)


def compare_dataset(
    ic: InstanceConfig, ctx: Context, dataset_id: str, *, vertical: bool = False, verbose: bool = False,
) -> None:
    dvc_df = None
    db_df = None
    dvc_modified_at = None

    try:
        dvc_ds = ctx.load_dvc_dataset(dataset_id)
        dvc_df = ppl.from_dvc_dataset(dvc_ds)
        dvc_modified_at = dvc_ds.modified_at
    except Exception as e:
        console.print(f'[red]Could not load dataset from DVC: {e}[/red]')

    try:
        db_obj = DBDatasetModel.objects.for_instance_config(ic).get(identifier=dataset_id)
        db_df = DBDataset.deserialize_df(db_obj, include_data_point_primary_keys=True)
    except DBDatasetModel.DoesNotExist:
        console.print(f'[red]Dataset "{dataset_id}" not found in database for instance "{ic.identifier}"[/red]')
    except Exception as e:
        console.print(f'[red]Could not load dataset from DB: {e}[/red]')

    if dvc_df is None and db_df is None:
        raise CommandError(f'Dataset "{dataset_id}" not found in either DVC or database.')

    if dvc_df is not None and db_df is not None:
        diff_schemas(dvc_df, db_df)
        diff_rows(dvc_df, db_df, vertical=vertical, dvc_modified_at=dvc_modified_at, verbose=verbose)


class Command(BaseCommand):
    help = 'Compare datasets between DVC and database, or find overlapping datasets across all instances'

    def add_arguments(self, parser):
        parser.add_argument('instance_id', type=str, nargs='?', help='InstanceConfig identifier')
        parser.add_argument(
            'dataset_id', type=str, nargs='?',
            help='Dataset identifier (e.g. longmont/greenhouse_gas_emissions_by_subsector)',
        )
        parser.add_argument(
            '--find-overlapping', action='store_true',
            help='Find datasets that exist in both DVC and the database across all instances',
        )
        parser.add_argument(
            '--vertical', action='store_true',
            help='Show differing rows in vertical (row-by-row) format instead of a table',
        )
        parser.add_argument(
            '--verbose', action='store_true',
            help='Show additional details like database primary keys in enrichment info',
        )

    def handle(self, *args, **options):
        if options['find_overlapping']:
            find_overlapping_datasets()
            return

        instance_id = options['instance_id']
        dataset_id = options['dataset_id']
        if not instance_id or not dataset_id:
            raise CommandError('instance_id and dataset_id are required when not using --find-overlapping')

        ic = InstanceConfig.objects.get(identifier=instance_id)
        ctx = ic.get_instance().context
        compare_dataset(ic, ctx, dataset_id, vertical=options['vertical'], verbose=options['verbose'])
