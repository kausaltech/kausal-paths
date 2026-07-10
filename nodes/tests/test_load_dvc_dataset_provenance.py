"""Tests for DataSource/DataPointComment creation in the load_dvc_dataset command."""

from __future__ import annotations

import polars as pl
import pytest

from kausal_common.datasets.models import DataPoint, DataPointComment, DatasetMetric, DatasetSourceReference, DataSource
from kausal_common.datasets.tests.factories import DatasetFactory, DatasetSchemaFactory

from common.polars import DataFrameMeta, to_ppdf
from nodes.management.commands.load_dvc_dataset import Command
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


def test_get_or_create_data_sources_empty():
    instance_config = InstanceConfigFactory.create(name='prov-cmd-2', config_source='database')
    cmd = Command()
    assert cmd.get_or_create_data_sources(instance_config, None) == {}
