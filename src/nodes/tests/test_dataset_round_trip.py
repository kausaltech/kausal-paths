"""
The claim in ``docs/dataset-round-trip.md``, made checkable.

Builds a dataset in the database, exports it with ``export_dataset``, pushes the CSV
through the transformation ``upload_new_dataset`` would push to DVC, imports the result
with ``load_dvc_dataset``, and asserts the two databases agree.

The DVC hop itself is not exercised, deliberately: it needs a repository and a network,
and what it does between ``build_dvc_frame`` and ``create_data_points`` is store the
frame and hand it back. Every transformation that can lose something is on this side of
that store, and every one of them runs here.

Data points are compared by natural key -- ``(year, metric, sorted category ids)`` -- not
by UUID, following ``DataPointKey`` in ``nodes/instance_serialization.py``. A re-import
mints new point UUIDs by design; see the plan's §3.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.models import ContentType

import polars as pl
import pytest

from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    Dataset,
    DatasetMetric,
    DatasetSchema,
    DatasetSchemaDimension,
    DatasetSourceReference,
    DataSource,
    Dimension,
    DimensionCategory,
    DimensionScope,
)

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import RESERVED_ROW_COLUMNS, SOURCE_TARGET_DATASET, YEAR_COLUMN
from nodes.datasets import DBDataset
from nodes.management.commands.export_dataset import (
    collect_from_db,
    write_data_csv,
    write_dataset_csv,
    write_sources_csv,
)
from nodes.management.commands.load_dvc_dataset import Command as LoadCommand
from nodes.tests.factories import InstanceConfigFactory
from nodes.units import unit_registry
from tools.upload_new_dataset import build_dvc_frame, build_dvc_metadata, load_dataset_attributes, load_sources_registry

if TYPE_CHECKING:
    from pathlib import Path

    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


def make_dataset(
    ic: InstanceConfig,
    *,
    identifier: str = 'test/energy',
    split_metric_provenance: bool = False,
    template_instruction: bool = False,
) -> Dataset:
    """
    Build a dataset exercising every feature the round trip has to carry.

    Two metrics, two dimensions, a null value, a data point with two comments, one with
    two cited sources, and a dataset-level source alongside the per-point ones.

    `template_instruction` adds the template case: an empty cell carrying the request and
    the source to obtain the figure from. `split_metric_provenance` makes the two metrics
    of one cell disagree about provenance, which is what forces the pivot to split them.
    """
    ct = ContentType.objects.get_for_model(ic)
    schema = DatasetSchema.objects.create(name='Energy', time_resolution=DatasetSchema.TimeResolution.YEARLY)
    value = DatasetMetric.objects.create(schema=schema, name='value', label='Value', unit='GWh/a', spec={'quantity': 'energy'})
    quality = DatasetMetric.objects.create(schema=schema, name='quality', label='Quality', unit='dimensionless')

    dims: dict[str, dict[str, DimensionCategory]] = {}
    for order, (dim_id, cats) in enumerate((('sector', ('residential', 'industry')), ('carrier', ('gas', 'electricity')))):
        dimension = Dimension.objects.create(name=dim_id)
        DimensionScope.objects.create(dimension=dimension, scope_content_type=ct, scope_id=ic.pk, identifier=dim_id)
        DatasetSchemaDimension.objects.create(schema=schema, dimension=dimension, order=order)
        dims[dim_id] = {c: DimensionCategory.objects.create(dimension=dimension, identifier=c, label=c.title()) for c in cats}

    dataset = Dataset.objects.create(schema=schema, identifier=identifier, scope_content_type=ct, scope_id=ic.pk)

    from datetime import date

    def add(metric: DatasetMetric, year: int, sector: str, carrier: str, val: float | None) -> DataPoint:
        point = DataPoint.objects.create(dataset=dataset, metric=metric, date=date(year, 1, 1), value=val)
        point.dimension_categories.add(dims['sector'][sector], dims['carrier'][carrier])
        return point

    add(value, 2020, 'residential', 'gas', 10.5)
    add(value, 2021, 'residential', 'gas', 11.25)
    # A null value is a data point that exists and has no number, which is not the same
    # as an absent row -- see the comment in load_dvc_dataset.create_data_points.
    add(value, 2020, 'industry', 'electricity', None)
    graded = add(quality, 2020, 'residential', 'gas', 3)
    # The commented point is the only metric at its (year, dimensions), so its provenance
    # does not have to be split away from a sibling metric's -- the ordinary case.
    metered = add(value, 2020, 'residential', 'electricity', 4.0)

    DataPointComment.objects.create(data_point=metered, text='metered')
    DataPointComment.objects.create(data_point=metered, text='revised after the 2021 audit')
    if split_metric_provenance:
        # Now the two metrics of one (Year, dimensions) disagree about provenance, so they
        # cannot share a row in a table that is wide by metric.
        DataPointComment.objects.create(data_point=graded, text='quality assessed separately')

    statistics = DataSource.objects.create(
        name='Statistics Office',
        authority='National Statistics',
        url='https://example.org/stats',
        description='Annual energy balance',
        edition='2024',
        scope_content_type=ct,
        scope_id=ic.pk,
    )
    utility = DataSource.objects.create(name='Utility', authority='City Utility', scope_content_type=ct, scope_id=ic.pk)
    whole = DataSource.objects.create(name='Energy Balance', authority='Ministry', scope_content_type=ct, scope_id=ic.pk)

    DatasetSourceReference.objects.create(data_point=metered, data_source=statistics)
    DatasetSourceReference.objects.create(data_point=metered, data_source=utility)
    DatasetSourceReference.objects.create(dataset=dataset, data_source=whole)

    if template_instruction:
        # The template case: an empty cell whose comment is the request, and whose source
        # says where the city can find the number.
        blank = add(value, 2022, 'residential', 'gas', None)
        DataPointComment.objects.create(data_point=blank, text='Please report the 2022 figure from the utility invoice')
        DatasetSourceReference.objects.create(data_point=blank, data_source=utility)

    return dataset


def snapshot(dataset: Dataset) -> dict[str, Any]:
    """Everything about a dataset the round trip is supposed to preserve, keyed naturally."""
    points: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for point in DataPoint.objects.filter(dataset=dataset).select_related('metric').prefetch_related('dimension_categories'):
        key = (
            point.date.year,
            point.metric.name,
            tuple(sorted(c.identifier or '' for c in point.dimension_categories.all())),
        )
        points.setdefault(key, []).append({
            'value': None if point.value is None else float(point.value),
            'comments': sorted(c.text for c in DataPointComment.objects.filter(data_point=point)),
            'sources': sorted(
                r.data_source.name for r in DatasetSourceReference.objects.filter(data_point=point).select_related('data_source')
            ),
        })
    assert dataset.schema is not None
    return {
        'points': points,
        'units': {m.name: m.unit for m in dataset.schema.metrics.all()},
        'dataset_sources': sorted(
            r.data_source.name for r in DatasetSourceReference.objects.filter(dataset=dataset).select_related('data_source')
        ),
    }


def across_the_dvc_boundary(frame: pl.DataFrame, units: dict[str, str]):
    """
    Stand in for the store-and-load that DVC performs between the two commands.

    `push_to_dvc` writes the frame with these units and this index-column set, and the
    reading path hands back a `PathsDataFrame` carrying them. Reproducing the two lines
    that decide them -- rather than the storage -- is what keeps this test about the
    transformations while still handing `create_data_points` the shape it really sees.
    """
    index_columns = [c for c in frame.columns if c not in units and c.lower() not in RESERVED_ROW_COLUMNS]
    meta = DataFrameMeta(
        units={col: unit_registry.parse_units(unit) for col, unit in units.items()},
        primary_keys=index_columns,
    )
    return to_ppdf(frame, meta)


def round_trip(ic: InstanceConfig, dataset: Dataset, tmp_path: Path, fmt: str) -> Dataset:
    """Export, transform as the uploader would, and import into a second dataset row."""
    export = collect_from_db(dataset, ic)
    data_path = tmp_path / f'{fmt}.csv'
    sources_path = tmp_path / f'{fmt}_sources.csv'
    write_data_csv(export, data_path, long_format=(fmt == 'long'))
    write_sources_csv(export, sources_path)

    # Read it back exactly as the uploader does: every cell a string, so nothing is
    # coerced differently here than it would be in a real upload.
    raw = pl.read_csv(data_path, infer_schema_length=0)
    # `process_datasets` splits on the `Dataset` column and drops it before the pipeline
    # runs; without that the column would be read as a dimension called "Dataset".
    raw = raw.filter(pl.col('Dataset') == export.dataset_name).drop('Dataset')
    # `--keep-empty-cells`, exactly as the command instructs when the export holds a
    # valueless cell: without it the uploader drops the blank and the cell comes back
    # absent rather than empty.
    keep_empty = any(r.value is None for r in export.records)
    frame, units, _metrics, _description = build_dvc_frame(
        raw, language='en', context=None, keep_empty_cells=keep_empty, verbose=False
    )
    stored = across_the_dvc_boundary(frame, units)

    registry = load_sources_registry(str(sources_path))
    sources_meta: list[dict[str, str | None]] = [{'name': name, **entry.fields} for name, entry in registry.items()]

    # A fresh schema, not the source's: `unique_dataset_per_scope_per_schema` allows one
    # dataset per (schema, scope), and a real import builds its own schema from the frame
    # anyway. Same metric names and units, so the comparison is still like for like.
    assert dataset.schema is not None
    target_schema = DatasetSchema.objects.create(
        name=f'{dataset.schema.name} ({fmt})', time_resolution=dataset.schema.time_resolution
    )
    metrics = {}
    for source_metric in dataset.schema.metrics.order_by('order'):
        assert source_metric.name is not None
        metrics[source_metric.name] = DatasetMetric.objects.create(
            schema=target_schema,
            name=source_metric.name,
            label=source_metric.label,
            unit=source_metric.unit,
            spec=source_metric.spec,
        )
    for sd in dataset.schema.dimensions.order_by('order'):
        DatasetSchemaDimension.objects.create(schema=target_schema, dimension=sd.dimension, order=sd.order)

    target = Dataset.objects.create(
        schema=target_schema,
        identifier=f'{dataset.identifier}-reimported-{fmt}',
        scope_content_type=dataset.scope_content_type,
        scope_id=dataset.scope_id,
    )
    LoadCommand().create_data_points(ic, stored, target, metrics, sources_meta=sources_meta)
    return target


def assert_originals_survive(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Every original data point comes back with its value, comments and sources intact."""
    for key, originals in before['points'].items():
        assert key in after['points'], f'lost {key}'
        for original in originals:
            assert original in after['points'][key], f'{key} came back changed: {original} not in {after["points"][key]}'
    assert after['units'] == before['units']
    assert after['dataset_sources'] == before['dataset_sources']


@pytest.mark.parametrize('fmt', ['wide', 'long'])
def test_db_to_csv_to_db_preserves_everything(tmp_path: Path, fmt: str):
    """
    The claim the plan's title makes: nothing that was in the database fails to come back.

    What the round trip *adds* is a separate question, and the two xfail tests below are
    where it is asked. This one is about loss.
    """
    ic = InstanceConfigFactory.create(name='round-trip', config_source='database')
    dataset = make_dataset(ic)
    before = snapshot(dataset)

    target = round_trip(ic, dataset, tmp_path, fmt)
    assert_originals_survive(before, snapshot(target))


@pytest.mark.parametrize('fmt', ['wide', 'long'])
def test_provenance_reaches_valueless_cells(tmp_path: Path, fmt: str):
    """
    A comment on an empty cell survives the round trip, because that is the template.

    An empty dataset is shipped to a city with the instruction -- or the source to obtain
    the figure from -- written in the `Comment` of the cell they are being asked to fill.
    So `Comment` is scoped to its row and reaches every metric of that row, a metric with
    no value included. Anything narrower could not say "we need this, here is where to get
    it", and `--keep-empty-cells` exists to get the empty row that far.
    """
    ic = InstanceConfigFactory.create(name='template', config_source='database')
    dataset = make_dataset(ic, template_instruction=True)
    before = snapshot(dataset)

    target = round_trip(ic, dataset, tmp_path, fmt)
    after = snapshot(target)

    key = (2022, 'value', ('gas', 'residential'))
    assert key in before['points'], 'fixture no longer builds the empty template cell'
    assert key in after['points'], 'the empty cell did not survive the round trip'
    # Across the records for that key, not just the first: in wide format the years of one
    # series that carry different provenance land on separate rows, so the instruction is on
    # one of them. `--format auto` picks long for exactly this shape.
    carried = [r for r in after['points'][key] if r['comments'] or r['sources']]
    assert carried, f'the instruction did not survive: {after["points"][key]}'
    assert all(r['value'] is None for r in after['points'][key])
    assert carried[0]['comments'] == ['Please report the 2022 figure from the utility invoice']
    assert carried[0]['sources'] == ['Utility']


@pytest.mark.parametrize('fmt', ['wide', 'long'])
@pytest.mark.xfail(
    strict=True,
    reason=(
        'Open question recorded in docs/dataset-round-trip.md §3.2. Splitting a row is the '
        'only way to give two metrics of one cell different provenance, and that is intended. '
        'What follows from it is not: the importer makes a data point for every metric of '
        '*both* rows, so a real value and a null end up under one natural key, and '
        '`DBDataset.deserialize_df` then keeps whichever `.group_by().first()` returns -- the '
        'null wins about half the time and the value is gone before any node sees it. It also '
        'reports every load to Sentry as duplicate rows.'
    ),
)
def test_metric_specific_provenance_does_not_duplicate_a_cell(tmp_path: Path, fmt: str):
    ic = InstanceConfigFactory.create(name='split-provenance', config_source='database')
    dataset = make_dataset(ic, split_metric_provenance=True)
    before = snapshot(dataset)

    target = round_trip(ic, dataset, tmp_path, fmt)
    after = snapshot(target)

    duplicated = {key: records for key, records in after['points'].items() if len(records) > 1}
    assert not duplicated, f'the same cell came back twice: {duplicated}'
    assert_originals_survive(before, after)


@pytest.mark.xfail(
    strict=True,
    reason='Same cause as test_metric_specific_provenance_does_not_duplicate_a_cell; see §3.2.',
)
def test_a_split_row_does_not_lose_the_value_a_node_reads(tmp_path: Path):
    """
    The consequence that matters: the duplicate is resolved against the value, not for it.

    `DBDataset.deserialize_df` collapses duplicate natural keys with
    `group_by(uniq_cols).first()` after sorting on `[Year, dimensions, metric]`. The value
    is not in that sort key, so which of the two rows survives is arbitrary -- and when the
    null one wins, the number is gone before any node reads the dataset. This asserts on
    the frame a node actually gets, not on the rows in the table.
    """
    ic = InstanceConfigFactory.create(name='split-loss', config_source='database')
    dataset = make_dataset(ic, split_metric_provenance=True)
    target = round_trip(ic, dataset, tmp_path, 'wide')

    original = DBDataset.deserialize_df(dataset).sort([YEAR_COLUMN, 'sector', 'carrier'])
    reloaded = DBDataset.deserialize_df(target).sort([YEAR_COLUMN, 'sector', 'carrier'])
    for metric in ('value', 'quality'):
        kept = [v for v in original[metric].to_list() if v is not None]
        back = [v for v in reloaded[metric].to_list() if v is not None]
        assert sorted(back) == sorted(kept), f'{metric}: {kept} went in, {back} came back'


def test_export_writes_the_registry_the_source_names_key_into(tmp_path: Path):
    """A `Source` cell is only provenance if the registry behind it survives too."""
    ic = InstanceConfigFactory.create(name='registry', config_source='database')
    dataset = make_dataset(ic)
    export = collect_from_db(dataset, ic)
    path = tmp_path / 'sources.csv'
    write_sources_csv(export, path)

    rows = {r['Name']: r for r in csv.DictReader(path.open(encoding='utf-8'))}
    assert set(rows) == {'Statistics Office', 'Utility', 'Energy Balance'}

    stats = rows['Statistics Office']
    assert stats['Authority'] == 'National Statistics'
    assert stats['URL'] == 'https://example.org/stats'
    assert stats['Description'] == 'Annual energy balance'
    assert stats['Edition'] == '2024'
    assert stats['Target'] == 'data_point'
    # Only a dataset-level source may name datasets; the uploader refuses it on the other.
    assert stats['Datasets'] == ''

    whole = rows['Energy Balance']
    assert whole['Target'] == SOURCE_TARGET_DATASET
    assert whole['Datasets'] == 'energy'


def test_export_uses_scope_identifiers_for_dimension_columns(tmp_path: Path):
    """
    Dimension headings must be identifiers, not translated display names.

    `sync_datasets --action csv` writes `name_i18n`, which on a non-English instance puts a
    translated heading on a column the uploader matches against an identifier. The failure
    reads as a missing dimension, well downstream of the export that caused it.
    """
    ic = InstanceConfigFactory.create(name='dim-columns', config_source='database')
    dataset = make_dataset(ic)
    Dimension.objects.filter(name='sector').update(name='Sektor', i18n={'name_de': 'Sektor'})

    export = collect_from_db(dataset, ic)
    path = tmp_path / 'wide.csv'
    write_data_csv(export, path, long_format=False)

    header = next(csv.reader(path.open(encoding='utf-8')))
    assert 'sector' in header
    assert 'Sektor' not in header


def test_null_value_survives_as_a_data_point(tmp_path: Path):
    """
    A valueless cell is a data point with no number, not an absent row.

    `load_dvc_dataset` is explicit about this and BISKO's template datasets depend on it:
    a pre-filled zero is indistinguishable from a confirmed zero. An export that dropped
    the row would turn one into the other on the next import.
    """
    ic = InstanceConfigFactory.create(name='nulls', config_source='database')
    dataset = make_dataset(ic)
    target = round_trip(ic, dataset, tmp_path, 'long')

    key = (2020, 'value', ('electricity', 'industry'))
    points = snapshot(target)['points']
    assert key in points
    assert [r['value'] for r in points[key]] == [None]


def test_dataset_level_attributes_survive_the_sidecar(tmp_path: Path):
    """
    The forecast boundary and the schema's flags travel beside the data and come back.

    Three hops, each a separate mechanism: `export_dataset` writes `<name>_dataset.csv`,
    `upload_new_dataset --dataset-csv` folds it into `metadata['dataset']`, and
    `load_dvc_dataset` reads it onto the row. They are exercised together because a
    mismatch between any two of them is invisible in the one that is right.
    """
    ic = InstanceConfigFactory.create(name='attributes', config_source='database')
    dataset = make_dataset(ic)
    dataset.spec = {'forecast_from': 2023}
    dataset.save(update_fields=['spec'])
    assert dataset.schema is not None
    dataset.schema.time_resolution = DatasetSchema.TimeResolution.MONTHLY
    dataset.schema.is_editable = False
    dataset.schema.save(update_fields=['time_resolution', 'is_editable'])

    export = collect_from_db(dataset, ic)
    path = tmp_path / 'energy_dataset.csv'
    assert write_dataset_csv(export, path)

    attributes = load_dataset_attributes(str(path), export.dataset_name)
    assert attributes == {
        'name': 'Energy',
        'forecast_from': 2023,
        'time_resolution': 'monthly',
        'is_editable': False,
    }

    # What `push_to_dvc` would write, and what the import reads back out of it.
    metadata = build_dvc_metadata('test/energy', 'energy', 'en', None, None, None, attributes)
    assert metadata['dataset'] == attributes

    target_schema = DatasetSchema.objects.create(name='Fresh', time_resolution=DatasetSchema.TimeResolution.YEARLY)
    target = Dataset.objects.create(
        schema=target_schema,
        identifier='test/energy-attrs',
        scope_content_type=dataset.scope_content_type,
        scope_id=dataset.scope_id,
    )
    LoadCommand().apply_dataset_attributes(target, target_schema, metadata)

    target.refresh_from_db()
    target_schema.refresh_from_db()
    assert target.spec['forecast_from'] == 2023
    assert target_schema.time_resolution == 'monthly'
    assert target_schema.is_editable is False


def test_dvc_file_without_the_dataset_key_changes_nothing(tmp_path: Path):
    """A `.dvc` file written before the sidecar existed must not be read as a set of blanks."""
    ic = InstanceConfigFactory.create(name='no-attributes', config_source='database')
    dataset = make_dataset(ic)
    assert dataset.schema is not None
    before = (dataset.spec, dataset.schema.time_resolution, dataset.schema.is_editable)

    LoadCommand().apply_dataset_attributes(dataset, dataset.schema, {'name': {'en': 'Energy'}})

    dataset.refresh_from_db()
    dataset.schema.refresh_from_db()
    assert (dataset.spec, dataset.schema.time_resolution, dataset.schema.is_editable) == before
