"""
Export one dataset to the CSV upload format, with its provenance intact.

The outbound half of the loop described in ``docs/dataset-round-trip.md``. What comes
out is exactly what ``upload_new_dataset`` reads, so the round trip closes by
construction rather than by two authors agreeing on a format:

    python manage.py export_dataset mainz-bisko mainz/endenergie --out /tmp/x
    python -m tools.upload_new_dataset -i /tmp/x/endenergie.csv -o mainz -d endenergie
        --sources-csv /tmp/x/endenergie_sources.csv
    python manage.py load_dvc_dataset mainz-bisko mainz/endenergie --force

Two files are written, because provenance does not live in the data file: the data
CSV, and the sources registry that its ``Source`` names key into. Exporting only the
first would produce names with no authority, URL or description behind them -- which
is what ``sync_datasets --action csv`` does today, and why its output cannot be
re-imported without losing the provenance.

Not exported, and not silently: see ``report_losses``. Anything this command knows it
cannot carry is printed, so an export is never quietly partial.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand, CommandError

from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    Dataset,
    DatasetSourceReference,
    DataSource,
    DimensionScope,
)

from nodes.constants import (
    COMMENT_SEPARATOR,
    SOURCE_NAME_SEPARATOR,
    SOURCE_TARGET_DATA_POINT,
    SOURCE_TARGET_DATASET,
    YEAR_COLUMN,
)
from nodes.management.commands.load_dvc_dataset import apply_repo_provenance, resolve_repo_provenance, source_target
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from kausal_common.datasets.models import DatasetMetric


# The registry columns `load_sources_registry` understands. Any other column it folds
# into Description as labelled prose, so writing exactly these keeps a re-import from
# growing description text that was never in the description.
REGISTRY_COLUMNS: tuple[str, ...] = ('Name', 'Authority', 'URL', 'Description', 'Edition', 'Target', 'Datasets')

# The dataset-level sidecar's columns. Facts the dataset can only have one of, which is why
# they are not row columns: a per-row column for a dataset-level fact invites two rows to
# disagree about it, which is the mistake dataset-level sources exist to undo.
#
# `external_ref` and `is_external_placeholder` are deliberately absent. They are not lost --
# `load_dvc_dataset` writes them itself, stamping the commit it imported from -- and carrying
# a stale stamp through a round trip would replace a true statement with an old one.
DATASET_COLUMNS: tuple[str, ...] = ('Dataset', 'Identifier', 'Name', 'ForecastFrom', 'TimeResolution', 'IsEditable')


@dataclass
class PointRecord:
    """One data point flattened into the terms the CSV format uses."""

    metric: str
    """The metric's dataframe column name -- the `Metric` cell, not the display label."""

    year: int
    value: Decimal | None
    dims: tuple[tuple[str, str], ...]
    """(column name, category identifier) pairs, sorted by column name."""

    source: str
    comment: str

    @property
    def row_key(self) -> tuple[Any, ...]:
        """What distinguishes one wide-format row from another: everything but the year."""
        return (self.metric, self.dims, self.source, self.comment)

    @property
    def series_key(self) -> tuple[Any, ...]:
        """The value series a row belongs to, provenance aside."""
        return (self.metric, self.dims)


@dataclass
class Export:
    """Everything one dataset contributes to the two output files."""

    dataset_name: str
    """The `Dataset` cell: the last segment of the identifier, which the uploader snake-cases."""

    identifier: str
    records: list[PointRecord] = field(default_factory=list)
    dim_columns: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    quantities: dict[str, str] = field(default_factory=dict)
    sources: dict[str, tuple[DataSource, str]] = field(default_factory=dict)
    """name -> (source row, target). One entry per cited source, at whichever level cites it."""

    attributes: dict[str, str] = field(default_factory=dict)
    """Dataset-level facts, written to the `<name>_dataset.csv` sidecar."""

    losses: list[str] = field(default_factory=list)

    def needs_long_format(self) -> tuple[bool, str | None]:
        """
        Tell whether wide format would split a value series across rows.

        Wide format can carry year-varying provenance -- two rows for one series, each
        holding the years that share a citation -- but the result is sparse and reads as
        though the series were two. Long format puts the year in the row instead, so the
        variation is stated rather than implied. `--format auto` picks on that basis.
        """
        by_series: dict[tuple[Any, ...], set[tuple[str, str]]] = defaultdict(set)
        for record in self.records:
            by_series[record.series_key].add((record.source, record.comment))
        for series, provenances in by_series.items():
            if len(provenances) > 1:
                metric, dims = series
                where = ', '.join(f'{col}={cat}' for col, cat in dims) or 'no dimensions'
                return True, f'{metric} ({where}) has {len(provenances)} source/comment combinations across its years'
        return False, None


def format_value(value: Decimal | None) -> str:
    """
    Render a stored value exactly, without scientific notation.

    `str()` on a Decimal keeps the stored exponent, so a zero stored as `0E-16` is written
    as `0E-16`. That parses back to the same number, but it is unreadable in a file people
    are meant to edit, and it makes a diff between two exports of the same data look like
    a change. Fixed-point formatting with the trailing zeros trimmed gives `0` and
    `150.6699999999999875` -- the same values, spelled the way they were entered.
    """
    if value is None:
        return ''
    text = format(value, 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text or '0'


def resolve_dataset(ic: InstanceConfig, identifier: str) -> Dataset:
    """
    Find the one dataset row this instance sees under `identifier`.

    Ambiguity is refused rather than guessed, as in `delete_dataset`: identifiers repeat
    across cities, and exporting the wrong city's data is a mistake that looks like
    success. A foreign scope is *not* refused -- reading a framework-scoped row that this
    instance merely sees is legitimate -- but it is reported, because the row that comes
    back is then not this instance's to change.
    """
    matches = list(Dataset.objects.get_queryset().for_instance_config(ic).filter(identifier=identifier).select_related('schema'))
    if not matches:
        visible = sorted(
            d.identifier for d in Dataset.objects.get_queryset().for_instance_config(ic).only('identifier') if d.identifier
        )
        listed = ', '.join(visible[:20]) + (' …' if len(visible) > 20 else '')
        raise CommandError(f"No dataset '{identifier}' is visible to {ic.identifier}. Visible: {listed or '(none)'}")
    if len(matches) > 1:
        pks = ', '.join(str(d.pk) for d in matches)
        raise CommandError(f"'{identifier}' is ambiguous here: {len(matches)} rows visible (pks {pks}).")
    return matches[0]


def dimension_columns(dataset: Dataset, ic: InstanceConfig) -> dict[int, str]:
    """
    Map each of the schema's dimensions to the column name it should carry in the CSV.

    In order of preference: the schema's own `column_name`, then the dimension's
    identifier *in this instance's scope*. Never the translated display name, which is
    what `sync_datasets --action csv` writes -- on a German instance that puts a German
    heading on a column the uploader will try to match against an identifier, and the
    round trip breaks in a way that reads as a missing dimension.
    """
    assert dataset.schema is not None
    scope_ct = ContentType.objects.get_for_model(ic)
    # `values_list` rather than attribute access on the model: Django generates the `_id`
    # attribute of a ForeignKey at runtime, and these two models do not declare it, so
    # reading `scope.dimension_id` is invisible to the type checker. Asking the query for
    # the column is both typed and a row of objects cheaper.
    scoped = dict(
        DimensionScope.objects
        .filter(scope_content_type=scope_ct, scope_id=ic.pk)
        .exclude(identifier=None)
        .exclude(identifier='')
        .values_list('dimension_id', 'identifier')
    )
    columns: dict[int, str] = {}
    for dim_pk, column_name, dim_name in dataset.schema.dimensions.order_by('order').values_list(
        'dimension_id', 'column_name', 'dimension__name'
    ):
        columns[dim_pk] = column_name or scoped.get(dim_pk) or dim_name
    return columns


def collect_from_db(dataset: Dataset, ic: InstanceConfig) -> Export:
    """Flatten a `Dataset` row and everything hanging off it into an `Export`."""
    assert dataset.schema is not None
    identifier = dataset.identifier or str(dataset.uuid)
    export = Export(dataset_name=identifier.rsplit('/', maxsplit=1)[-1], identifier=identifier)

    dim_by_id = dimension_columns(dataset, ic)
    export.dim_columns = [
        dim_by_id[dim_pk] for dim_pk in dataset.schema.dimensions.order_by('order').values_list('dimension_id', flat=True)
    ]

    metrics: dict[int, DatasetMetric] = {}
    for metric in dataset.schema.metrics.order_by('order'):
        metrics[metric.pk] = metric
        # `name` is the dataframe column, which is what `Metric` means in the CSV and what
        # `load_dvc_dataset.create_metric` writes back. `label` is for people.
        column = metric.name or metric.label
        export.units[column] = metric.unit or ''
        quantity = (metric.spec or {}).get('quantity')
        if quantity:
            export.quantities[column] = str(quantity)

    comments = comments_by_point(dataset)
    sources, point_sources = sources_by_point(dataset, export)

    points = (
        DataPoint.objects
        .filter(dataset=dataset)
        .select_related('metric')
        .prefetch_related('dimension_categories')
        .order_by('date')
    )
    for point in points:
        metric = metrics.get(point.metric.pk)
        if metric is None:  # a metric outside the schema's own set should not exist
            export.losses.append(f'data point {point.uuid} has metric {point.metric.pk}, which is not in the schema')
            continue
        dims = tuple(
            sorted(
                (dim_by_id.get(cat.dimension_id, str(cat.dimension_id)), cat.identifier or cat.label)
                for cat in point.dimension_categories.all()
            )
        )
        export.records.append(
            PointRecord(
                metric=metric.name or metric.label,
                year=point.date.year,
                value=point.value,
                dims=dims,
                source=SOURCE_NAME_SEPARATOR.join(sorted(point_sources.get(point.pk, ()))),
                comment=COMMENT_SEPARATOR.join(comments.get(point.pk, ())),
            )
        )
    export.sources = sources
    export.attributes = {
        'Dataset': export.dataset_name,
        'Identifier': identifier,
        'Name': dataset.schema.name or '',
        'ForecastFrom': str((dataset.spec or {}).get('forecast_from') or ''),
        'TimeResolution': dataset.schema.time_resolution,
        'IsEditable': 'true' if dataset.schema.is_editable else 'false',
    }
    return export


def collect_from_dvc(ic: InstanceConfig, identifier: str, repo_from: str) -> tuple[Export, str | None]:
    """
    Flatten the DVC copy of a dataset into the same `Export` the database path produces.

    `Context.load_dvc_dataset` resolves the identifier against the instance's pinned
    commit, so nothing here needs a hash -- which is the whole of what `fetch_dataset`
    was for, minus the step of finding the hash by hand. Returns the pin as well, because
    an export from DVC is an export of one commit and the file should not be the only
    record of which.
    """
    from common import polars as ppl

    ctx = ic.get_instance().context
    provenance = resolve_repo_provenance(ic, ctx, repo_from)
    apply_repo_provenance(ctx, provenance)
    pin = ctx.dataset_repo_spec.commit if ctx.dataset_repo_spec else None

    try:
        dvc_ds = ctx.load_dvc_dataset(identifier)
    except Exception as exc:
        raise CommandError(f"Could not read '{identifier}' from DVC at pin {(pin or 'unset')[:7]}: {exc}") from exc

    df = ppl.from_dvc_dataset(dvc_ds)
    meta = df.get_meta()
    dvc_metadata = dvc_ds.meta.metadata or {}
    export = Export(dataset_name=identifier.rsplit('/', maxsplit=1)[-1], identifier=identifier)

    metric_cols = list(meta.units) or [c for c in df.columns if c not in meta.primary_keys]
    export.units = {col: str(meta.units[col]) for col in metric_cols if col in meta.units}
    for entry in dvc_metadata.get('metrics') or []:
        quantity = entry.get('quantity') if isinstance(entry, dict) else None
        column = entry.get('column_id') or entry.get('id') if isinstance(entry, dict) else None
        if column and quantity:
            export.quantities[str(column)] = str(quantity)

    source_col = next((c for c in df.columns if c.lower() == 'source'), None)
    comment_col = next((c for c in df.columns if c.lower() in ('comment', 'description')), None)
    reserved = {c for c in (source_col, comment_col) if c}
    export.dim_columns = [c for c in meta.dim_ids if c not in reserved]

    for row in df.iter_rows(named=True):
        year = row.get(YEAR_COLUMN)
        if year is None:
            continue
        dims = tuple(sorted((col, str(row[col])) for col in export.dim_columns if row.get(col) is not None))
        for column in metric_cols:
            value = row.get(column)
            export.records.append(
                PointRecord(
                    metric=column,
                    year=int(year),
                    value=None if value is None else Decimal(str(value)),
                    dims=dims,
                    source=str(row.get(source_col) or '') if source_col else '',
                    comment=str(row.get(comment_col) or '') if comment_col else '',
                )
            )

    export.sources = sources_from_dvc_metadata(dvc_metadata.get('sources'))
    stored = dvc_metadata.get('dataset') or {}
    name = dvc_metadata.get('name')
    export.attributes = {
        'Dataset': export.dataset_name,
        'Identifier': identifier,
        'Name': str(stored.get('name') or (next(iter(name.values())) if isinstance(name, dict) and name else '') or ''),
        'ForecastFrom': str(stored.get('forecast_from') or ''),
        'TimeResolution': str(stored.get('time_resolution') or 'yearly'),
        'IsEditable': 'false' if stored.get('is_editable') is False else 'true',
    }
    return export, pin


def sources_from_dvc_metadata(sources_meta: Any) -> dict[str, tuple[Any, str]]:
    """
    Rebuild the registry from `metadata['sources']`, which is where DVC keeps it.

    The entries are already the registry's own fields, so this is a rename rather than a
    lookup -- and a `SimpleNamespace` stands in for the `DataSource` row, since there need
    not be one: a dataset can be exported from DVC into an instance that has never
    imported it.
    """
    from types import SimpleNamespace

    registry: dict[str, tuple[Any, str]] = {}
    for entry in sources_meta or []:
        if not isinstance(entry, dict) or not entry.get('name'):
            continue
        name = str(entry['name'])
        registry[name] = (
            SimpleNamespace(
                name=name,
                authority=entry.get('authority'),
                url=entry.get('url'),
                description=entry.get('description'),
                edition=entry.get('edition'),
            ),
            source_target(entry),
        )
    return registry


def comments_by_point(dataset: Dataset) -> dict[int, list[str]]:
    """
    Collect each data point's comment texts, oldest first.

    Through the child model rather than `point.comments`, which is a reverse manager the
    type checker cannot see, and one query instead of a prefetch per point. The default
    manager already excludes soft-deleted comments, which is what we want: a deleted
    comment should not come back through an export.
    """
    out: dict[int, list[str]] = defaultdict(list)
    rows = (
        DataPointComment.objects.filter(data_point__dataset=dataset).order_by('created_at').values_list('data_point_id', 'text')
    )
    for point_id, text in rows:
        if point_id is not None and text:
            out[point_id].append(text)
    return out


def sources_by_point(dataset: Dataset, export: Export) -> tuple[dict[str, tuple[DataSource, str]], dict[int, set[str]]]:
    """
    Collect the dataset's citations at both levels.

    Returns the registry (source name -> row and target) and, separately, which sources
    each data point cites. A source cited at both levels is a state the uploader refuses,
    so it is reported here rather than written into a registry that cannot be re-read.
    """
    registry: dict[str, tuple[DataSource, str]] = {}
    per_point: dict[int, set[str]] = defaultdict(set)
    seen_targets: dict[str, set[str]] = defaultdict(set)

    # The three FK ids as values, for the same reason as in `dimension_columns`:
    # `DatasetSourceReference` declares none of them, so attribute access on them is
    # invisible to the type checker. The sources themselves are then one query by pk.
    references = list(
        DatasetSourceReference.objects.filter(models_q_for_dataset(dataset)).values_list(
            'dataset_id', 'data_point_id', 'data_source_id'
        )
    )
    sources_by_pk = {s.pk: s for s in DataSource.objects.filter(pk__in={ref[2] for ref in references})}

    for dataset_pk, data_point_pk, source_pk in references:
        source = sources_by_pk.get(source_pk)
        if source is None:  # a PROTECTed FK, so this should not be reachable
            continue
        target = SOURCE_TARGET_DATASET if dataset_pk is not None else SOURCE_TARGET_DATA_POINT
        registry[source.name] = (source, target)
        seen_targets[source.name].add(target)
        if data_point_pk is not None:
            per_point[data_point_pk].add(source.name)

    for name, targets in seen_targets.items():
        if len(targets) > 1:
            export.losses.append(
                f"source '{name}' is cited at both levels; the uploader refuses that, so fix the references before re-importing"
            )
    return registry, per_point


def models_q_for_dataset(dataset: Dataset):
    """Build the filter for references belonging to this dataset: its own, and its data points'."""
    from django.db.models import Q

    return Q(dataset=dataset) | Q(data_point__dataset=dataset)


def write_data_csv(export: Export, path: Path, long_format: bool) -> int:
    """Write the data file in the upload format. Returns the number of rows written."""
    has_source = any(r.source for r in export.records)
    has_comment = any(r.comment for r in export.records)
    has_quantity = bool(export.quantities)

    lead = ['Metric', 'Unit']
    if has_quantity:
        lead.append('Quantity')
    lead.append('Dataset')
    trail = ([] if not has_source else ['Source']) + ([] if not has_comment else ['Comment'])

    def base(record: PointRecord) -> dict[str, str]:
        row: dict[str, str] = {
            'Metric': record.metric,
            'Unit': export.units.get(record.metric, ''),
            'Dataset': export.dataset_name,
            **dict(record.dims),
        }
        if has_quantity:
            row['Quantity'] = export.quantities.get(record.metric, '')
        if has_source:
            row['Source'] = record.source
        if has_comment:
            row['Comment'] = record.comment
        return row

    if long_format:
        # `Year` and `Value` next to `Quantity` rather than after the dimensions, so the
        # fact reads as one phrase; the pipeline keys on names, so order is free.
        fields = [*lead[:-1], 'Year', 'Value', lead[-1], *export.dim_columns, *trail]
        rows = [
            {**base(r), 'Year': str(r.year), 'Value': format_value(r.value)}
            for r in sorted(export.records, key=lambda r: (r.metric, r.dims, r.year))
        ]
    else:
        years = sorted({r.year for r in export.records})
        fields = [*lead, *export.dim_columns, *trail, *[str(y) for y in years]]
        grouped: dict[tuple[Any, ...], dict[str, str]] = {}
        for record in sorted(export.records, key=lambda r: (r.metric, r.dims, r.year)):
            row = grouped.setdefault(record.row_key, base(record))
            row[str(record.year)] = format_value(record.value)
        rows = list(grouped.values())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n', restval='', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_sources_csv(export: Export, path: Path) -> int:
    """Write the sources registry keyed by the names the data file's `Source` cells use."""
    if not export.sources:
        return 0
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(REGISTRY_COLUMNS), lineterminator='\n', restval='')
        writer.writeheader()
        for name in sorted(export.sources):
            source, target = export.sources[name]
            writer.writerow({
                'Name': name,
                'Authority': source.authority or '',
                'URL': source.url or '',
                'Description': source.description or '',
                'Edition': source.edition or '',
                'Target': target,
                # Only a dataset-level source takes a restriction; the uploader refuses one
                # on a data_point-targeted row. Naming the dataset is precise rather than
                # relying on an empty cell meaning "all of them".
                'Datasets': export.dataset_name if target == SOURCE_TARGET_DATASET else '',
            })
    return len(export.sources)


def write_dataset_csv(export: Export, path: Path) -> bool:
    """Write the dataset-level sidecar. Returns whether anything was written."""
    if not export.attributes:
        return False
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(DATASET_COLUMNS), lineterminator='\n', restval='', extrasaction='ignore')
        writer.writeheader()
        writer.writerow(export.attributes)
    return True


def report_losses(export: Export, dataset: Dataset) -> list[str]:
    """
    Name what this export cannot carry, so the gap is visible rather than discovered later.

    Dataset-level attributes now travel in the sidecar (see ``DATASET_COLUMNS``), so what
    is left here is the short list of things that genuinely do not survive.
    """
    losses = list(export.losses)
    assert dataset.schema is not None
    if dataset.is_external_placeholder:
        losses.append('is_external_placeholder=True -- the import decides this, so it is not carried')
    sticky = DataPointComment.objects.filter(data_point__dataset=dataset).exclude(is_sticky=False).count()
    if sticky:
        losses.append(f'{sticky} comment(s) are sticky; the flag is not carried, only the text')
    return losses


class Command(BaseCommand):
    help = 'Export one dataset to the CSV upload format, with its sources registry'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instance', metavar='INSTANCE_ID')
        parser.add_argument('identifier', metavar='DATASET_ID')
        parser.add_argument('--out', required=True, metavar='DIR', help='Directory to write the CSV files into')
        parser.add_argument(
            '--from',
            dest='source',
            choices=['auto', 'db', 'dvc'],
            default='auto',
            help=(
                'Where to read the data. auto (default) uses the database when its row holds '
                'data points and DVC otherwise, and refuses when both hold data and they '
                'disagree -- say which you mean, or ask dataset_inventory about the drift.'
            ),
        )
        parser.add_argument(
            '--repo-from',
            choices=['auto', 'yaml', 'db'],
            default='auto',
            help="Which config's DVC pin to read, when reading from DVC (default: auto)",
        )
        parser.add_argument(
            '--format',
            choices=['auto', 'wide', 'long'],
            default='auto',
            help=(
                'auto (default) uses wide format unless a value series has provenance that '
                'varies by year, in which case long format states the variation instead of '
                'splitting the series across sparse rows.'
            ),
        )

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            ic = InstanceConfig.objects.get(identifier=options['instance'])
        except InstanceConfig.DoesNotExist:
            raise CommandError(f"No instance '{options['instance']}'.") from None

        export, dataset = self.read_source(ic, options)

        varies, why = export.needs_long_format()
        long_format = options['format'] == 'long' or (options['format'] == 'auto' and varies)
        if varies and options['format'] == 'auto':
            self.stdout.write(f'  long format chosen: {why}')
            self.stdout.write('  (--format wide is also lossless here, but splits that series across rows)')
        elif varies and not long_format:
            self.stdout.write(self.style.WARNING(f'  note: wide format splits a series here -- {why}'))

        out_dir = Path(options['out'])
        data_path = out_dir / f'{export.dataset_name}.csv'
        sources_path = out_dir / f'{export.dataset_name}_sources.csv'

        dataset_path = out_dir / f'{export.dataset_name}_dataset.csv'
        n_rows = write_data_csv(export, data_path, long_format)
        n_sources = write_sources_csv(export, sources_path)
        wrote_attributes = write_dataset_csv(export, dataset_path)

        self.report(export, dataset, ic, data_path, sources_path, dataset_path, n_rows, n_sources, wrote_attributes, long_format)

    def read_source(self, ic: InstanceConfig, options: dict[str, Any]) -> tuple[Export, Dataset | None]:
        """
        Read the dataset from wherever `--from` says, and refuse an ambiguous `auto`.

        `auto` picks the side that has data. When both do, it refuses rather than choose:
        a silent choice between two populated copies is the failure this whole exercise is
        about -- the export would look complete while being half of one dataset. Naming
        `--from db` or `--from dvc` is never refused, because it states the intent.
        """
        identifier = options['identifier']
        want = options['source']

        dataset: Dataset | None = None
        db_points = 0
        if want in ('auto', 'db'):
            dataset = resolve_dataset(ic, identifier)
            if dataset.schema is None:
                raise CommandError(f"'{identifier}' has no schema, so there is nothing to describe it with.")
            db_points = DataPoint.objects.filter(dataset=dataset).count()
            scope = dataset.scope
            if not (isinstance(scope, InstanceConfig) and scope.pk == ic.pk):
                self.stdout.write(self.style.WARNING(f'  note: this row is scoped to {scope!r}, not to {ic.identifier}'))

        if want == 'db':
            assert dataset is not None
            if not db_points:
                raise CommandError(
                    f"'{identifier}' holds no data points in the database. "
                    'It may be an external placeholder that lives only in DVC -- try --from dvc.'
                )
            return collect_from_db(dataset, ic), dataset

        if want == 'dvc':
            export, pin = collect_from_dvc(ic, identifier, options['repo_from'])
            self.stdout.write(f'  read from DVC at pin {(pin or "unset")[:7]}')
            return export, dataset

        return self.read_auto(ic, options, dataset, db_points), dataset

    def read_auto(self, ic: InstanceConfig, options: dict[str, Any], dataset: Dataset | None, db_points: int) -> Export:
        """Pick the side that has data, and refuse when both do."""
        identifier = options['identifier']
        try:
            dvc_export, pin = collect_from_dvc(ic, identifier, options['repo_from'])
        except CommandError:
            dvc_export, pin = None, None
        dvc_has = dvc_export is not None and bool(dvc_export.records)

        if db_points and dvc_has:
            assert dvc_export is not None
            raise CommandError(
                f"'{identifier}' holds data in both the database ({db_points} points) and DVC "
                f'({len(dvc_export.records)} cells at pin {(pin or "unset")[:7]}). Which one you mean '
                'changes the answer, so say --from db or --from dvc. '
                f'`python manage.py dataset_inventory {ic.identifier}` reports the drift between them.'
            )
        if db_points:
            assert dataset is not None
            return collect_from_db(dataset, ic)
        if dvc_has:
            assert dvc_export is not None
            self.stdout.write(f'  read from DVC at pin {(pin or "unset")[:7]}')
            return dvc_export
        raise CommandError(f"'{identifier}' holds no data in either the database or DVC.")

    def report(
        self,
        export: Export,
        dataset: Dataset | None,
        ic: InstanceConfig,
        data_path: Path,
        sources_path: Path,
        dataset_path: Path,
        n_rows: int,
        n_sources: int,
        wrote_attributes: bool,
        long_format: bool,
    ) -> None:
        """Say what was written, what could not be, and the two commands that close the loop."""
        shape = 'long' if long_format else 'wide'
        self.stdout.write(self.style.SUCCESS(f'Exported {export.identifier} ({shape} format)'))
        self.stdout.write(f'  {data_path}  {n_rows} rows, {len(export.records)} data points')
        if n_sources:
            self.stdout.write(f'  {sources_path}  {n_sources} sources')
        else:
            self.stdout.write('  no sources cited, so no registry written')
        if wrote_attributes:
            self.stdout.write(f'  {dataset_path}  dataset-level attributes')

        for loss in report_losses(export, dataset) if dataset is not None else export.losses:
            self.stdout.write(self.style.WARNING(f'  not carried: {loss}'))

        # A valueless cell is a data point, and the uploader drops blank cells unless told
        # not to. Without the flag the row simply is not in the parquet, and the cell comes
        # back from the next import as absent rather than as empty -- which is the very
        # distinction `create_data_points` goes out of its way to keep.
        has_empty = any(r.value is None for r in export.records)
        if has_empty:
            n_empty = sum(1 for r in export.records if r.value is None)
            self.stdout.write(
                self.style.WARNING(f'  {n_empty} valueless cell(s): the upload needs --keep-empty-cells, or they are dropped')
            )

        namespace = export.identifier.rsplit('/', maxsplit=1)[0] if '/' in export.identifier else '<namespace>'
        upload = f'    python -m tools.upload_new_dataset -i {data_path} -o {namespace} -d {export.dataset_name}'
        if has_empty:
            upload += ' --keep-empty-cells'
        if n_sources:
            upload += f' \\\n        --sources-csv {sources_path}'
        if wrote_attributes:
            upload += f' \\\n        --dataset-csv {dataset_path}'
        self.stdout.write('\n  Round trip:')
        self.stdout.write(upload)
        self.stdout.write(f'    python manage.py load_dvc_dataset {ic.identifier} {export.identifier} --force')
