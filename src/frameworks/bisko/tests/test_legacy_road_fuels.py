from datetime import date
from decimal import Decimal

import pytest

from kausal_common.datasets.models import DataPoint, DataPointDimensionCategory, Dataset, DimensionCategory
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from frameworks.bisko.legacy_road_fuels import ROAD_TRANSPORT_DATASET, fold_road_biofuels
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.models import InstanceChangeOperation, InstanceModelLogEntry

pytestmark = pytest.mark.django_db

CARRIERS = ('petrol', 'bio_petrol', 'diesel', 'biodiesel', 'natural_gas', 'cng_fossil', 'cng_bio', 'electricity')


def _cells(dataset: Dataset) -> dict[tuple[str, ...], Decimal | None]:
    return {
        tuple(sorted(c.identifier or '' for c in point.dimension_categories.all())): point.value
        for point in DataPoint.objects.filter(dataset=dataset).prefetch_related('dimension_categories')
    }


def test_split_fuels_fold_into_the_carriers_municipalities_enter() -> None:
    framework = FrameworkFactory.create(identifier='bisko')
    instance = FrameworkConfigFactory.create(
        framework=framework, instance_config__name='Road city', instance_config__config_source='database'
    ).instance_config
    schema = DatasetSchemaFactory.create()
    metric = DatasetMetricFactory.create(schema=schema, name='Value', unit='MWh/a')
    categories: dict[str, DimensionCategory] = {}
    for values in (('passenger_cars', 'trucks'), CARRIERS):
        dimension = DimensionFactory.create()
        DatasetSchemaDimensionFactory.create(schema=schema, dimension=dimension)
        for identifier in values:
            categories[identifier] = DimensionCategoryFactory.create(dimension=dimension, identifier=identifier)
    dataset = DatasetFactory.create(scope=instance, schema=schema, identifier=ROAD_TRANSPORT_DATASET)
    entered = {
        ('passenger_cars', 'petrol'): Decimal(100),
        ('passenger_cars', 'bio_petrol'): Decimal(5),
        ('passenger_cars', 'biodiesel'): None,
        ('trucks', 'diesel'): Decimal(200),
        ('trucks', 'biodiesel'): Decimal(10),
        ('trucks', 'cng_fossil'): Decimal(30),
        ('trucks', 'cng_bio'): Decimal(3),
        ('trucks', 'electricity'): Decimal(1),
    }
    for (vehicle, carrier), value in entered.items():
        point = DataPointFactory.create(dataset=dataset, metric=metric, date=date(2023, 1, 1), value=value)
        for identifier in (vehicle, carrier):
            DataPointDimensionCategory.objects.create(data_point=point, dimension_category=categories[identifier])

    report = fold_road_biofuels()

    assert _cells(dataset) == {
        ('passenger_cars', 'petrol'): Decimal(105),
        # An empty cell with nothing to fold into keeps its slot under the target carrier.
        ('diesel', 'passenger_cars'): None,
        ('diesel', 'trucks'): Decimal(210),
        ('natural_gas', 'trucks'): Decimal(33),
        ('electricity', 'trucks'): Decimal(1),
    }
    assert len(report) == 5
    operation = InstanceChangeOperation.objects.get(instance_config=instance)
    # Two relabelled cells, and an update and a delete for each of the three added.
    assert InstanceModelLogEntry.objects.filter(operation=operation).count() == 8
    # A second run finds nothing to fold and records nothing.
    assert fold_road_biofuels() == []
    assert InstanceChangeOperation.objects.filter(instance_config=instance).count() == 1
