"""Tests for DataSource/DataPointComment creation in the load_dvc_dataset command."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from django.contrib.contenttypes.models import ContentType
from django.db import IntegrityError, transaction

import polars as pl
import pytest

from kausal_common.datasets.models import DataPoint, DataPointComment, Dataset, DatasetMetric, DatasetSourceReference, DataSource
from kausal_common.datasets.tests.factories import DatasetFactory, DatasetSchemaFactory

from common.polars import DataFrameMeta, to_ppdf
from nodes.management.commands.load_dvc_dataset import Command, build_dataset_plan, dataset_level_source_names
from nodes.models import DatasetMaterialization, InstanceConfig
from nodes.tests.factories import InstanceConfigFactory
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def test_create_data_points_links_source_and_comment():
    instance_config = InstanceConfigFactory.create(name='prov-cmd', config_source='database')
    schema = DatasetSchemaFactory.create()
    dataset = DatasetFactory.create(schema=schema)
    metric = DatasetMetric.objects.create(schema=schema, name='value', label='Value', unit='kt')

    df = to_ppdf(
        pl.DataFrame({
            'Year': [2020, 2021],
            'value': [1.0, 2.0],
            'source': ['NPF', 'NPF'],
            'comment': ['from appendix 2', None],
        }),
        DataFrameMeta(units={'value': unit_registry.parse_units('kt')}, primary_keys=[]),
    )

    sources_meta: list[dict[str, str | None]] = [
        {'name': 'NPF', 'authority': 'Dept of Housing', 'url': 'https://example.com/npf', 'description': 'desc'}
    ]

    cmd = Command()
    cmd.create_data_points(instance_config, df, dataset, {'value': metric}, sources_meta=sources_meta)

    assert DataPoint.objects.filter(dataset=dataset).count() == 2
    assert DatasetSourceReference.objects.filter(data_point__dataset=dataset).count() == 2

    source = DataSource.objects.get(name='NPF')
    assert source.authority == 'Dept of Housing'
    assert source.url == 'https://example.com/npf'

    comments = DataPointComment.objects.filter(data_point__dataset=dataset)
    assert comments.count() == 1
    assert comments.get().text == 'from appendix 2'

    # Re-running (e.g. a re-sync) must reuse the same DataSource row, not duplicate it.
    cmd.create_data_points(instance_config, df, dataset, {'value': metric}, sources_meta=sources_meta)
    assert DataSource.objects.filter(name='NPF').count() == 1


DATASET_LEVEL_META: list[dict[str, str | None]] = [
    {'name': 'Energiebilanz', 'authority': 'StaLa', 'url': None, 'description': None, 'edition': '2024', 'target': 'dataset'},
    {'name': 'Verkehrsmodell', 'authority': 'City', 'url': None, 'description': None, 'edition': None, 'target': 'dataset'},
]


def _plain_df() -> Any:
    return to_ppdf(
        pl.DataFrame({'Year': [2020, 2021], 'value': [1.0, 2.0]}),
        DataFrameMeta(units={'value': unit_registry.parse_units('kt')}, primary_keys=[]),
    )


def test_dataset_level_sources_attach_to_the_dataset_not_its_points():
    """A dataset with uniform provenance carries its sources once, and may carry several."""
    instance_config = InstanceConfigFactory.create(name='prov-dataset-level', config_source='database')
    schema = DatasetSchemaFactory.create()
    dataset = DatasetFactory.create(schema=schema)
    metric = DatasetMetric.objects.create(schema=schema, name='value', label='Value', unit='kt')

    cmd = Command()
    cmd.create_data_points(instance_config, _plain_df(), dataset, {'value': metric}, sources_meta=DATASET_LEVEL_META)

    refs = DatasetSourceReference.objects.filter(dataset=dataset)
    assert sorted(r.data_source.name for r in refs) == ['Energiebilanz', 'Verkehrsmodell']
    assert DatasetSourceReference.objects.filter(data_point__dataset=dataset).count() == 0
    assert DataSource.objects.get(name='Energiebilanz').edition == '2024'


def test_reimport_replaces_dataset_level_sources_without_duplicating_them():
    """
    A dataset-level reference outlives the data points, so the import has to replace the set.

    Per-point references CASCADE away with the points that ``refresh_dataset_in_place``
    deletes; these hang off the Dataset row and would otherwise gain a duplicate on every
    ``--force``, and keep a source the registry had dropped.
    """
    instance_config = InstanceConfigFactory.create(name='prov-dataset-reimport', config_source='database')
    schema = DatasetSchemaFactory.create()
    dataset = DatasetFactory.create(schema=schema)
    metric = DatasetMetric.objects.create(schema=schema, name='value', label='Value', unit='kt')

    cmd = Command()
    cmd.create_data_points(instance_config, _plain_df(), dataset, {'value': metric}, sources_meta=DATASET_LEVEL_META)
    cmd.create_data_points(instance_config, _plain_df(), dataset, {'value': metric}, sources_meta=DATASET_LEVEL_META)

    assert DatasetSourceReference.objects.filter(dataset=dataset).count() == 2

    # The registry drops one source and revises the other; both changes must land.
    revised: list[dict[str, str | None]] = [
        {
            'name': 'Energiebilanz',
            'authority': 'StaLa RLP',
            'url': None,
            'description': None,
            'edition': '2025',
            'target': 'dataset',
        },
    ]
    cmd.create_data_points(instance_config, _plain_df(), dataset, {'value': metric}, sources_meta=revised)

    refs = DatasetSourceReference.objects.filter(dataset=dataset)
    assert [r.data_source.name for r in refs] == ['Energiebilanz']
    source = DataSource.objects.get(name='Energiebilanz')
    assert (source.edition, source.authority) == ('2025', 'StaLa RLP')
    # The row is reused rather than replaced, so references held elsewhere stay valid.
    assert DataSource.objects.filter(name='Energiebilanz').count() == 1


def test_a_dataset_level_source_is_not_linked_to_rows_that_cite_it():
    """Hand-edited metadata only: the uploader refuses this combination before it reaches DVC."""
    instance_config = InstanceConfigFactory.create(name='prov-mixed-target', config_source='database')
    schema = DatasetSchemaFactory.create()
    dataset = DatasetFactory.create(schema=schema)
    metric = DatasetMetric.objects.create(schema=schema, name='value', label='Value', unit='kt')
    df = to_ppdf(
        pl.DataFrame({'Year': [2020], 'value': [1.0], 'source': ['Energiebilanz']}),
        DataFrameMeta(units={'value': unit_registry.parse_units('kt')}, primary_keys=[]),
    )

    cmd = Command()
    cmd.create_data_points(instance_config, df, dataset, {'value': metric}, sources_meta=DATASET_LEVEL_META[:1])

    assert DatasetSourceReference.objects.filter(dataset=dataset).count() == 1
    assert DatasetSourceReference.objects.filter(data_point__dataset=dataset).count() == 0


def test_sources_without_a_target_key_attach_to_data_points():
    """What every .dvc file written before dataset-level sources existed means."""
    instance_config = InstanceConfigFactory.create(name='prov-legacy-meta', config_source='database')
    schema = DatasetSchemaFactory.create()
    dataset = DatasetFactory.create(schema=schema)
    metric = DatasetMetric.objects.create(schema=schema, name='value', label='Value', unit='kt')
    df = to_ppdf(
        pl.DataFrame({'Year': [2020], 'value': [1.0], 'source': ['NPF']}),
        DataFrameMeta(units={'value': unit_registry.parse_units('kt')}, primary_keys=[]),
    )

    cmd = Command()
    cmd.create_data_points(instance_config, df, dataset, {'value': metric}, sources_meta=[{'name': 'NPF'}])

    assert DatasetSourceReference.objects.filter(data_point__dataset=dataset).count() == 1
    assert DatasetSourceReference.objects.filter(dataset=dataset).count() == 0


def test_plan_reports_the_dataset_level_sources_it_would_add_and_drop():
    instance_config = InstanceConfigFactory.create(name='prov-plan', config_source='database')
    schema = DatasetSchemaFactory.create()
    dataset = DatasetFactory.create(schema=schema)
    metric = DatasetMetric.objects.create(schema=schema, name='value', label='Value', unit='kt')

    cmd = Command()
    cmd.create_data_points(instance_config, _plain_df(), dataset, {'value': metric}, sources_meta=DATASET_LEVEL_META)

    plan = build_dataset_plan(
        ds_id='test/plan',
        dataset=dataset,
        incoming_metric_cols=['value'],
        incoming_data_points=2,
        incoming_commit='abc123',
        incoming_dataset_sources=dataset_level_source_names([
            *DATASET_LEVEL_META[:1],
            {'name': 'Handbuch', 'target': 'data_point'},
            {'name': 'Neu', 'target': 'dataset'},
        ]),
    )

    assert plan.current_dataset_sources == ['Energiebilanz', 'Verkehrsmodell']
    assert plan.incoming_dataset_sources == ['Energiebilanz', 'Neu']


def test_source_reference_must_target_exactly_one_thing():
    with pytest.raises(IntegrityError), transaction.atomic():
        DatasetSourceReference.objects.create(
            data_source=DataSource.objects.create(
                scope_content_type=ContentType.objects.get_for_model(InstanceConfig),
                scope_id=InstanceConfigFactory.create(name='prov-constraint', config_source='database').pk,
                name='Unattached',
            )
        )


def test_get_or_create_data_sources_empty():
    instance_config = InstanceConfigFactory.create(name='prov-cmd-2', config_source='database')
    cmd = Command()
    assert cmd.get_or_create_data_sources(instance_config, None) == {}


def test_sync_dataset_creates_final_materialization():
    instance_config = InstanceConfigFactory.create(name='materialized-import', config_source='database')
    dvc_dataset = SimpleNamespace(
        df=pl.DataFrame({'Year': [2020, 2021], 'value': [1.0, 2.0]}),
        units={'value': 'kt'},
        index_columns=['Year'],
        metadata={'name': {'en': 'Imported dataset'}, 'metrics': [{'column_id': 'value'}]},
    )
    context = SimpleNamespace(
        dataset_repo_spec=None,
        dimensions={},
        instance=SimpleNamespace(default_language='en'),
        load_dvc_dataset=lambda _dataset_id: dvc_dataset,
    )

    Command().sync_dataset(instance_config, cast('Any', context), 'test/import')

    dataset = Dataset.objects.get(identifier='test/import')
    materialization = DatasetMaterialization.objects.get(dataset=dataset)
    assert materialization.generation == 1
    assert materialization.content['data'] is not None
    assert len(materialization.content['data']['data']) == 2
