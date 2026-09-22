from __future__ import annotations

from datetime import date
from decimal import Decimal
from io import StringIO
from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.models import ContentType
from django.core.management import call_command
from django.db import transaction

import polars as pl
import pytest

from kausal_common.datasets.models import DataPoint
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from paths.tests.graphql import PathsTestClient

from frameworks.evidence import QUALITY_OF_SPEC_KEY
from frameworks.models import DataEvidenceKind, DataPointEvidence, DataQualityLevel, DataQualityScheme
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.datasets import DBDataset
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.instance_serialization import DatasetSnapshot, _import_dataset
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from django.test import Client

    from kausal_common.datasets.models import Dataset, DatasetMetric, DimensionCategory

    from frameworks.models import Framework
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


def make_scheme(framework: Framework) -> dict[str, DataQualityLevel]:
    scheme = DataQualityScheme.objects.create(framework=framework, identifier='bisko', version='1', name='BISKO')
    return {
        identifier: DataQualityLevel.objects.create(
            scheme=scheme, identifier=identifier, name=identifier, order=order, score=score
        )
        for order, (identifier, score) in enumerate([
            ('A', Decimal(1)),
            ('B', Decimal('0.5')),
            ('C', Decimal('0.25')),
            ('D', Decimal(0)),
        ])
    }


def make_instance() -> InstanceConfig:
    instance = InstanceFactory.create()
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        spec=InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030)),
    )


def make_member(framework: Framework) -> InstanceConfig:
    ic = make_instance()
    FrameworkConfigFactory.create(framework=framework, instance_config=ic)
    return ic


class Setup:
    def __init__(self) -> None:
        self.framework = FrameworkFactory.create()
        self.levels = make_scheme(self.framework)
        self.ic = make_member(self.framework)
        self.schema = DatasetSchemaFactory.create(name='Final energy')
        dimension = DimensionFactory.create(name='Sector')
        DatasetSchemaDimensionFactory.create(schema=self.schema, dimension=dimension)
        self.households: DimensionCategory = DimensionCategoryFactory.create(dimension=dimension, identifier='phh', label='PHH')
        self.industry: DimensionCategory = DimensionCategoryFactory.create(dimension=dimension, identifier='ind', label='IND')
        self.value: DatasetMetric = DatasetMetricFactory.create(schema=self.schema, name='Value', label='Value', unit='MWh/a')
        self.quality: DatasetMetric = DatasetMetricFactory.create(schema=self.schema, name='quality', label='quality', unit='')
        self.dataset: Dataset = DatasetFactory.create(schema=self.schema, scope=self.ic)

    def point(self, metric: DatasetMetric, category: DimensionCategory, value: float | None) -> DataPoint:
        dp = DataPointFactory.create(
            dataset=self.dataset, metric=metric, date=date(2022, 1, 1), value=None if value is None else Decimal(str(value))
        )
        dp.dimension_categories.set([category])
        return dp


@pytest.fixture
def setup() -> Setup:
    return Setup()


@pytest.fixture
def gql(client: Client, setup: Setup) -> PathsTestClient:
    client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(setup.ic)
    return tc


UPDATE = """
mutation($instanceId: ID!, $datasetId: ID!, $input: [UpdateDataPointItemInput!]!) {
  instanceEditor(instanceId: $instanceId) { datasetEditor(datasetId: $datasetId) {
    updateDataPoints(input: $input) {
      __typename
      ... on DataPointsMutationResult { dataPoints { id value evidence { kind qualityLevel { identifier score } } } }
      ... on OperationInfo { messages { kind message } }
    }
  } }
}
"""

CREATE = """
mutation($instanceId: ID!, $datasetId: ID!, $input: [CreateDataPointInput!]!) {
  instanceEditor(instanceId: $instanceId) { datasetEditor(datasetId: $datasetId) {
    createDataPoints(input: $input) {
      __typename
      ... on DataPointsMutationResult { dataPoints { id evidence { kind qualityLevel { identifier } } } }
      ... on OperationInfo { messages { kind message } }
    }
  } }
}
"""


def update(gql: PathsTestClient, setup: Setup, dp: DataPoint, **input: Any) -> dict[str, Any]:
    data = gql.query_data(
        UPDATE,
        variables={
            'instanceId': setup.ic.identifier,
            'datasetId': str(setup.dataset.uuid),
            'input': [{'dataPointId': str(dp.uuid), 'input': input}],
        },
    )
    return data['instanceEditor']['datasetEditor']['updateDataPoints']


def test_dataset_exposes_its_frameworks_quality_scheme(gql: PathsTestClient, setup: Setup) -> None:
    data = gql.query_data(
        '{ instance { editor { datasets { identifier qualitySchemes { identifier version levels { identifier score } } } } } }'
    )
    (ds,) = data['instance']['editor']['datasets']
    assert ds['qualitySchemes'] == [
        {
            'identifier': 'bisko',
            'version': '1',
            'levels': [
                {'identifier': 'A', 'score': 1.0},
                {'identifier': 'B', 'score': 0.5},
                {'identifier': 'C', 'score': 0.25},
                {'identifier': 'D', 'score': 0.0},
            ],
        }
    ]


def test_create_and_update_carry_evidence(gql: PathsTestClient, setup: Setup) -> None:
    data = gql.query_data(
        CREATE,
        variables={
            'instanceId': setup.ic.identifier,
            'datasetId': str(setup.dataset.uuid),
            'input': [
                {
                    'date': '2022-01-01',
                    'value': 10.0,
                    'metricId': str(setup.value.uuid),
                    'dimensionCategoryIds': [str(setup.households.uuid)],
                    'evidenceKind': 'OBSERVED',
                    'qualityLevelId': str(setup.levels['B'].uuid),
                }
            ],
        },
    )
    (created,) = data['instanceEditor']['datasetEditor']['createDataPoints']['dataPoints']
    assert created['evidence'] == {'kind': 'OBSERVED', 'qualityLevel': {'identifier': 'B'}}

    dp = DataPoint.objects.get(uuid=created['id'])
    result = update(gql, setup, dp, qualityLevelId=str(setup.levels['A'].uuid))
    assert result['dataPoints'][0]['evidence'] == {'kind': 'OBSERVED', 'qualityLevel': {'identifier': 'A', 'score': 1.0}}

    result = update(gql, setup, dp, qualityLevelId=None)
    assert result['dataPoints'][0]['evidence'] == {'kind': 'OBSERVED', 'qualityLevel': None}

    result = update(gql, setup, dp, evidenceKind=None)
    assert result['dataPoints'][0]['evidence'] is None
    assert not DataPointEvidence.objects.filter(data_point=dp).exists()


def test_value_edit_without_evidence_leaves_evidence_alone(gql: PathsTestClient, setup: Setup) -> None:
    dp = setup.point(setup.value, setup.households, 10)
    DataPointEvidence.objects.create(data_point=dp, kind=DataEvidenceKind.PROVIDER_DEFAULT, quality_level=setup.levels['D'])
    result = update(gql, setup, dp, value=12.0)
    assert result['dataPoints'][0]['evidence'] == {'kind': 'PROVIDER_DEFAULT', 'qualityLevel': {'identifier': 'D', 'score': 0.0}}


def test_level_of_another_framework_is_rejected(gql: PathsTestClient, setup: Setup) -> None:
    foreign = make_scheme(FrameworkFactory.create())
    dp = setup.point(setup.value, setup.households, 10)
    result = update(gql, setup, dp, qualityLevelId=str(foreign['A'].uuid))
    assert result['__typename'] == 'OperationInfo'
    assert 'not available for this dataset' in result['messages'][0]['message']
    assert not DataPointEvidence.objects.filter(data_point=dp).exists()


def test_confirmed_zero_is_bound_to_a_zero_value(gql: PathsTestClient, setup: Setup) -> None:
    nonzero = setup.point(setup.value, setup.households, 10)
    result = update(gql, setup, nonzero, evidenceKind='EXPLICIT_ZERO')
    assert result['__typename'] == 'OperationInfo'

    zero = setup.point(setup.value, setup.industry, 0)
    result = update(gql, setup, zero, evidenceKind='EXPLICIT_ZERO')
    assert result['dataPoints'][0]['evidence']['kind'] == 'EXPLICIT_ZERO'

    # Changing the value alone would leave a stale confirmation behind.
    result = update(gql, setup, zero, value=5.0)
    assert result['__typename'] == 'OperationInfo'
    zero.refresh_from_db()
    assert zero.value == 0

    result = update(gql, setup, zero, value=5.0, evidenceKind='OBSERVED')
    assert result['dataPoints'][0] == {
        'id': str(zero.uuid),
        'value': 5.0,
        'evidence': {'kind': 'OBSERVED', 'qualityLevel': None},
    }


def test_projected_metric_is_derived_from_evidence(gql: PathsTestClient, setup: Setup) -> None:
    graded = setup.point(setup.value, setup.households, 10)
    setup.point(setup.value, setup.industry, 20)  # ungraded
    stale = setup.point(setup.quality, setup.households, 0.25)  # legacy value, ignored once projected
    DataPointEvidence.objects.create(data_point=graded, quality_level=setup.levels['B'])
    setup.quality.spec = {QUALITY_OF_SPEC_KEY: str(setup.value.uuid)}
    setup.quality.save()

    dim = df_dim(setup)
    df = pl.DataFrame(DBDataset.deserialize_df(setup.dataset))
    rows = {row[dim]: (row['Value'], row['quality']) for row in df.iter_rows(named=True)}
    assert rows == {'phh': (10.0, 0.5), 'ind': (20.0, None)}

    result = update(gql, setup, stale, value=1.0)
    assert result['__typename'] == 'OperationInfo'
    assert 'derived from data-point evidence' in result['messages'][0]['message']

    metrics = gql.query_data('{ instance { editor { datasets { metrics { name qualityOf } } } } }')
    assert metrics['instance']['editor']['datasets'][0]['metrics'] == [
        {'name': 'Value', 'qualityOf': None},
        {'name': 'quality', 'qualityOf': str(setup.value.uuid)},
    ]


def df_dim(setup: Setup) -> str:
    df = DBDataset.deserialize_df(setup.dataset)
    (dim,) = [c for c in df.columns if c not in ('Year', 'Value', 'quality')]
    return dim


def test_evidence_survives_snapshot_round_trip(setup: Setup) -> None:
    author = UserFactory.create()
    # Dimensionless, so the snapshot's natural keys need no instance-scoped dimension identifiers.
    schema = DatasetSchemaFactory.create()
    metric = DatasetMetricFactory.create(schema=schema, name='Value', label='Value')
    dataset = DatasetFactory.create(schema=schema, scope=setup.ic)
    graded = DataPointFactory.create(dataset=dataset, metric=metric, date=date(2021, 1, 1), value=Decimal(10))
    zero = DataPointFactory.create(dataset=dataset, metric=metric, date=date(2022, 1, 1), value=Decimal(0))
    DataPointEvidence.objects.create(
        data_point=graded, kind=DataEvidenceKind.OBSERVED, quality_level=setup.levels['A'], created_by=author
    )
    DataPointEvidence.objects.create(data_point=zero, kind=DataEvidenceKind.EXPLICIT_ZERO)

    snap = DatasetSnapshot.from_model_for_instance(dataset, setup.ic)
    assert {(ev.kind, ev.quality_level.level if ev.quality_level else None) for ev in snap.evidence} == {
        ('observed', 'A'),
        ('explicit_zero', None),
    }

    # Another member of the same framework resolves the grade.
    member = make_member(setup.framework)
    new_ds = _import_dataset(member, snap, ContentType.objects.get_for_model(member), {})
    imported = {
        (ev.data_point.value, ev.kind, ev.quality_level.identifier if ev.quality_level else None, ev.created_by_id)
        for ev in DataPointEvidence.objects.filter(data_point__dataset=new_ds).select_related('data_point', 'quality_level')
    }
    assert imported == {(Decimal(10), 'observed', 'A', author.pk), (Decimal(0), 'explicit_zero', None, None)}

    # An instance outside the framework keeps the kind and drops the grade.
    outsider = make_instance()
    new_ds = _import_dataset(outsider, snap, ContentType.objects.get_for_model(outsider), {})
    kinds = set(DataPointEvidence.objects.filter(data_point__dataset=new_ds).values_list('kind', 'quality_level'))
    assert kinds == {('observed', None), ('explicit_zero', None)}


def run_import(setup: Setup, *args: str) -> str:
    out = StringIO()
    with transaction.atomic():
        call_command('import_quality_evidence', setup.ic.identifier, *args, stdout=out)
    return out.getvalue()


def test_import_converts_exact_grades_and_projects(setup: Setup) -> None:
    graded = setup.point(setup.value, setup.households, 10)
    setup.point(setup.quality, setup.households, 0.5)
    output = run_import(setup, '--delete-legacy')
    assert 'graded 1' in output
    assert DataPointEvidence.objects.get(data_point=graded).quality_level == setup.levels['B']
    setup.quality.refresh_from_db()
    assert setup.quality.spec[QUALITY_OF_SPEC_KEY] == str(setup.value.uuid)
    assert not DataPoint.objects.filter(metric=setup.quality).exists()


def test_import_does_not_project_values_between_grades(setup: Setup) -> None:
    setup.point(setup.value, setup.households, 10)
    setup.point(setup.value, setup.industry, 20)
    setup.point(setup.quality, setup.households, 0.5)
    setup.point(setup.quality, setup.industry, 0.2)

    output = run_import(setup, '--delete-legacy')
    assert 'match no single grade' in output
    setup.quality.refresh_from_db()
    assert QUALITY_OF_SPEC_KEY not in setup.quality.spec
    assert DataPoint.objects.filter(metric=setup.quality).count() == 2

    output = run_import(setup, '--snap-down')
    assert '0.2 -> D' in output
    setup.quality.refresh_from_db()
    assert QUALITY_OF_SPEC_KEY in setup.quality.spec
    assert set(DataPointEvidence.objects.values_list('quality_level__identifier', flat=True)) == {'B', 'D'}
