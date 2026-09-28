from datetime import date

import pytest

from kausal_common.datasets.tests.factories import DataPointFactory, DatasetFactory, DatasetMetricFactory, DatasetSchemaFactory

from datasets.year_slots import ensure_empty_year
from nodes.models import DatasetMaterialization
from nodes.tests.factories import InstanceConfigFactory

pytestmark = pytest.mark.django_db


def test_new_year_keeps_values_and_skips_projected_metrics() -> None:
    schema = DatasetSchemaFactory.create()
    amount = DatasetMetricFactory.create(schema=schema, name='amount')
    grade = DatasetMetricFactory.create(schema=schema, name='quality', spec={'quality_of': str(amount.uuid)})
    template = DatasetFactory.create(scope=InstanceConfigFactory.create(name='Template', config_source='database'), schema=schema)
    local = DatasetFactory.create(scope=InstanceConfigFactory.create(name='Local', config_source='database'), schema=schema)
    DataPointFactory.create(dataset=template, metric=amount, date=date(2023, 1, 1), value=7)
    DataPointFactory.create(dataset=template, metric=grade, date=date(2023, 1, 1), value=None)
    previous = DataPointFactory.create(dataset=local, metric=amount, date=date(2023, 1, 1), value=9)

    assert ensure_empty_year(local, 2024, prototype=template) == 1
    assert local.data_points.get(date=date(2024, 1, 1), metric=amount).value is None
    assert not local.data_points.filter(date=date(2024, 1, 1), metric=grade).exists()
    previous.refresh_from_db()
    assert previous.value == 9
    assert ensure_empty_year(local, 2024, prototype=template) == 0
    assert DatasetMaterialization.objects.filter(dataset=local).exists()
