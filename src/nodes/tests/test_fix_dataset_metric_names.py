from types import SimpleNamespace
from typing import Any, cast

import pytest

from kausal_common.datasets.models import DatasetMetric
from kausal_common.datasets.tests.factories import DatasetFactory, DatasetSchemaFactory

from datasets.management.commands.fix_dataset_metric_names import fix_metrics_for_dataset
from datasets.materialization import materialize_dataset

pytestmark = pytest.mark.django_db


def _instance():
    from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

    return InstanceConfigFactory.create(instance=InstanceFactory.create())


def test_fix_metric_name_materializes_every_dataset_using_schema(monkeypatch):
    schema = DatasetSchemaFactory.create()
    first_dataset = DatasetFactory.create(schema=schema, identifier='first', scope=_instance())
    second_dataset = DatasetFactory.create(schema=schema, identifier='second', scope=_instance())
    metric = DatasetMetric.objects.create(schema=schema, name='1.3 Car pooling', label='Car pooling', unit='kt')
    first_materialization = materialize_dataset(first_dataset)
    second_materialization = materialize_dataset(second_dataset)
    dataframe = SimpleNamespace(get_meta=lambda: SimpleNamespace(metric_cols=['13_car_pooling']))
    context = SimpleNamespace(load_dvc_dataset=lambda _dataset_id: object())
    monkeypatch.setattr(
        'datasets.management.commands.fix_dataset_metric_names.ppl.from_dvc_dataset',
        lambda _dataset: dataframe,
    )

    fixed, unfixable, names = fix_metrics_for_dataset(first_dataset, cast('Any', context))

    assert (fixed, unfixable, names) == (1, 0, [])
    metric.refresh_from_db()
    assert metric.name == '13_car_pooling'
    first_materialization.refresh_from_db()
    second_materialization.refresh_from_db()
    assert first_materialization.generation == 2
    assert second_materialization.generation == 2
