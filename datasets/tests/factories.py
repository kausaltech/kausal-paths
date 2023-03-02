from factory import Factory, Sequence, SubFactory, post_generation
from factory.faker import Faker
from factory.django import DjangoModelFactory

from nodes.tests.factories import InstanceConfigFactory

from datasets.models import (
    Dataset, DatasetDimension, DatasetMetric, DatasetDimensionSelectedCategory,
    Dimension, DimensionCategory
)


class DatasetFactory(DjangoModelFactory):
    class Meta:
        model = Dataset

    instance = SubFactory(InstanceConfigFactory)
    identifier = Sequence(lambda i: f'dataset{i}')
    years = [2020, 2021, 2022]
    name = Sequence(lambda i: f'Dataset{i}')
    table = {'schema': None, 'data': None}


class DatasetMetricFactory(DjangoModelFactory):
    class Meta:
        model = DatasetMetric

    identifier = Sequence(lambda i: f'metric{i}')
    label = Sequence(lambda i: f'Metric{i}')
    uuid = Faker('uuid4')
    unit = 'MWh'
    dataset = SubFactory(DatasetFactory)
