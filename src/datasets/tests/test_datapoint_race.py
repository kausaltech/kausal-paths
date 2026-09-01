from __future__ import annotations

from contextlib import suppress
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, cast

from django.db import close_old_connections
from rest_framework.test import APIClient

import pytest

from kausal_common.datasets.api import DataPointSerializer
from kausal_common.datasets.models import DataPoint
from kausal_common.datasets.tests.factories import (
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)
from kausal_common.deployment import env_bool

from users.tests.factories import UserFactory

IS_CI = env_bool('CI', default=False)


# Transactions cause reuse-db to not work due to db flushing, so we only run this test in CI.
@pytest.mark.skipif(not IS_CI, reason='Requires transactions, so only run in CI')
@pytest.mark.django_db(transaction=True)
def test_datapoint_concurrent_create_revalidates_after_dataset_lock(monkeypatch):
    user = UserFactory.create(is_superuser=True)
    schema = DatasetSchemaFactory.create()
    dimension = DimensionFactory.create()
    DatasetSchemaDimensionFactory.create(schema=schema, dimension=dimension)
    dataset = DatasetFactory.create(schema=schema)
    metric = DatasetMetricFactory.create(schema=schema)
    dimension_category = DimensionCategoryFactory.create(dimension=dimension)
    url = f'/v1/datasets/{dataset.uuid}/data_points/'
    create_data = {
        'date': '2024-01-01',
        'value': 150.0,
        'metric': str(metric.uuid),
        'dimension_categories': [str(dimension_category.uuid)],
    }
    started = Event()
    release = Event()
    parked: Queue[str] = Queue()
    results: Queue[int | BaseException] = Queue()
    real_validate = DataPointSerializer.validate

    def racing_validate(self: DataPointSerializer, data: dict[str, Any]) -> dict[str, Any]:
        data = real_validate(self, data)
        if self.context['view'].kwargs.get('dataset_uuid') == str(dataset.uuid):
            parked.put('validated')
            assert release.wait(timeout=5)
        return data

    def post_data_point() -> None:
        close_old_connections()
        try:
            client = APIClient()
            client.force_authenticate(user=user)
            started.wait(timeout=5)
            response = client.post(url, create_data, format='json')
            results.put(response.status_code)
        except BaseException as exc:
            results.put(exc)
        finally:
            close_old_connections()

    monkeypatch.setattr(DataPointSerializer, 'validate', racing_validate)
    threads = [Thread(target=post_data_point) for _ in range(2)]
    for thread in threads:
        thread.start()
    started.set()
    parked.get(timeout=5)
    with suppress(Empty):
        parked.get(timeout=1)
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    statuses = [results.get(timeout=5) for _ in threads]
    assert not [status for status in statuses if isinstance(status, BaseException)]
    statuses = cast('list[int]', statuses)
    assert sorted(statuses) == [201, 400]
    assert (
        DataPoint.objects.filter(
            dataset=dataset,
            metric=metric,
            date='2024-01-01',
            dimension_categories=dimension_category,
        ).count()
        == 1
    )
