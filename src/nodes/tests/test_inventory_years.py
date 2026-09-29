"""Adding and removing historical inventory years, including skipped years."""

import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

import pytest

from kausal_common.datasets.models import DataPoint, DataPointComment, Dataset, DatasetMetric
from kausal_common.datasets.tests.factories import DataPointFactory, DatasetFactory, DatasetMetricFactory

from paths.tests.graphql import PathsTestClient

from frameworks import submissions
from frameworks.models import DataEvidenceKind, DataPointEvidence, Submission
from frameworks.tests.factories import FrameworkConfigFactory
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db

ADD = """
mutation Add($instanceId: ID!, $year: Int!) {
  instanceEditor(instanceId: $instanceId) {
    addInventoryYear(year: $year) {
      __typename
      ... on AddInventoryYearResult { year minHistorical maxHistorical historical createdCells }
      ... on OperationInfo { messages { message } }
    }
  }
}
"""

REMOVE = """
mutation Remove($instanceId: ID!, $year: Int!, $force: Boolean!) {
  instanceEditor(instanceId: $instanceId) {
    removeInventoryYear(year: $year, force: $force) {
      __typename
      ... on RemoveInventoryYearResult { year minHistorical maxHistorical historical deletedCells }
      ... on InventoryYearNotEmpty { year values evidence comments sourceReferences }
      ... on OperationInfo { messages { message } }
    }
  }
}
"""


@pytest.fixture
def db_instance_config() -> InstanceConfig:
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030))
    return InstanceConfigFactory.create(
        identifier=instance.id, instance=instance, config_source='database', owner='Test Owner', spec=spec
    )


@pytest.fixture
def gql_client(client, db_instance_config: InstanceConfig) -> PathsTestClient:
    client.force_login(UserFactory.create(is_superuser=True))
    result = PathsTestClient(client)
    result.set_instance(db_instance_config)
    return result


@pytest.fixture
def dataset(db_instance_config: InstanceConfig) -> tuple[Dataset, DatasetMetric]:
    dataset = DatasetFactory.create(scope=db_instance_config)
    assert dataset.schema is not None
    return dataset, DatasetMetricFactory.create(schema=dataset.schema)


def _add(gql: PathsTestClient, ic: InstanceConfig, year: int) -> dict[str, Any]:
    data = gql.query_data(ADD, variables={'instanceId': str(ic.pk), 'year': year})
    return data['instanceEditor']['addInventoryYear']


def _remove(gql: PathsTestClient, ic: InstanceConfig, year: int, *, force: bool = False) -> dict[str, Any]:
    data = gql.query_data(REMOVE, variables={'instanceId': str(ic.pk), 'year': year, 'force': force})
    return data['instanceEditor']['removeInventoryYear']


def _years(ic: InstanceConfig) -> tuple[int | None, int | None, list[int] | None]:
    ic.refresh_from_db()
    years = ic.ensure_spec().years
    return years.min_historical, years.max_historical, years.skipped


class TestYearsSpec:
    def test_historical_is_undeclared_until_skipped_is_set(self) -> None:
        assert YearsSpec(min_historical=2010, max_historical=2012).historical is None
        assert YearsSpec(min_historical=2010, max_historical=2012, skipped=[]).historical == [2010, 2011, 2012]

    def test_a_sparse_early_series(self) -> None:
        years = YearsSpec(min_historical=1990, max_historical=2024, skipped=list(range(1991, 2000)))
        assert years.historical == [1990, *range(2000, 2025)]

    @pytest.mark.parametrize(
        ('skipped', 'message'),
        [
            ([2012, 2011], 'sorted and unique'),
            ([2011, 2011], 'sorted and unique'),
            ([2010], 'strictly between'),
            ([2015], 'strictly between'),
        ],
    )
    def test_skipped_years_lie_strictly_inside_the_span(self, skipped: list[int], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            YearsSpec(min_historical=2010, max_historical=2015, skipped=skipped)


class TestInventoryYears:
    @pytest.mark.parametrize('with_framework', [False, True])
    def test_add_seeds_blank_cells_and_opens_no_submission(
        self,
        gql_client: PathsTestClient,
        db_instance_config: InstanceConfig,
        dataset: tuple[Dataset, DatasetMetric],
        with_framework: bool,
    ) -> None:
        ds, metric = dataset
        if with_framework:
            FrameworkConfigFactory.create(instance_config=db_instance_config)

        added = _add(gql_client, db_instance_config, 2023)
        assert added['__typename'] == 'AddInventoryYearResult', added
        assert added['createdCells'] == 1
        assert added['historical'] == list(range(2010, 2024))
        assert ds.data_points.get(date__year=2023, metric=metric).value is None
        # The first write declares the span, which until now was undeclared.
        assert _years(db_instance_config) == (2010, 2023, [])
        assert not Submission.objects.filter(instance_config=db_instance_config).exists()

    @pytest.mark.usefixtures('dataset')
    def test_skipping_ahead_and_back(self, gql_client: PathsTestClient, db_instance_config: InstanceConfig) -> None:
        added = _add(gql_client, db_instance_config, 2025)
        assert (added['maxHistorical'], added['historical'][-2:]) == (2025, [2022, 2025])
        assert _years(db_instance_config) == (2010, 2025, [2023, 2024])

        removed = _remove(gql_client, db_instance_config, 2025)
        assert removed['__typename'] == 'RemoveInventoryYearResult', removed
        assert (removed['maxHistorical'], removed['deletedCells']) == (2022, 1)
        assert _years(db_instance_config) == (2010, 2022, [])

    @pytest.mark.usefixtures('dataset')
    def test_a_year_before_the_first_widens_the_span_downwards(
        self, gql_client: PathsTestClient, db_instance_config: InstanceConfig
    ) -> None:
        assert _add(gql_client, db_instance_config, 2005)['minHistorical'] == 2005
        assert _years(db_instance_config) == (2005, 2022, [2006, 2007, 2008, 2009])

        assert _remove(gql_client, db_instance_config, 2005)['minHistorical'] == 2010
        assert _years(db_instance_config) == (2010, 2022, [])

    @pytest.mark.usefixtures('dataset')
    def test_adding_a_skipped_year_fills_the_gap(self, gql_client: PathsTestClient, db_instance_config: InstanceConfig) -> None:
        _add(gql_client, db_instance_config, 2025)
        _add(gql_client, db_instance_config, 2024)
        assert _years(db_instance_config) == (2010, 2025, [2023])

        assert _remove(gql_client, db_instance_config, 2025)['maxHistorical'] == 2024
        assert _remove(gql_client, db_instance_config, 2024)['maxHistorical'] == 2022

    def test_removing_an_inner_year_skips_it(
        self, gql_client: PathsTestClient, db_instance_config: InstanceConfig, dataset: tuple[Dataset, DatasetMetric]
    ) -> None:
        ds, metric = dataset
        DataPointFactory.create(dataset=ds, metric=metric, date=datetime.date(2015, 1, 1), value=Decimal(1))

        removed = _remove(gql_client, db_instance_config, 2015, force=True)
        assert (removed['minHistorical'], removed['maxHistorical'], removed['deletedCells']) == (2010, 2022, 1)
        assert 2015 not in removed['historical']
        assert _years(db_instance_config) == (2010, 2022, [2015])

        added = _add(gql_client, db_instance_config, 2015)
        assert 2015 in added['historical']
        assert _years(db_instance_config) == (2010, 2022, [])

    def test_a_year_with_work_needs_force(
        self, gql_client: PathsTestClient, db_instance_config: InstanceConfig, dataset: tuple[Dataset, DatasetMetric]
    ) -> None:
        ds, _metric = dataset
        _add(gql_client, db_instance_config, 2023)
        point = ds.data_points.get(date__year=2023)
        point.value = Decimal(5)
        point.save()
        DataPointComment.objects.create(data_point=point, text='From the utility report')

        refused = _remove(gql_client, db_instance_config, 2023)
        assert refused == {
            '__typename': 'InventoryYearNotEmpty',
            'year': 2023,
            'values': 1,
            'evidence': 0,
            'comments': 1,
            'sourceReferences': 0,
        }
        assert DataPoint.objects.filter(pk=point.pk).exists()
        assert _years(db_instance_config)[1] == 2023

        assert _remove(gql_client, db_instance_config, 2023, force=True)['__typename'] == 'RemoveInventoryYearResult'
        assert not DataPoint.objects.filter(pk=point.pk).exists()
        assert _years(db_instance_config)[1] == 2022

    @pytest.mark.parametrize(
        ('year', 'message'),
        [
            (2020, 'reference year'),
            (2030, 'not an inventory year'),
        ],
    )
    def test_removal_refusals(
        self,
        gql_client: PathsTestClient,
        db_instance_config: InstanceConfig,
        dataset: tuple[Dataset, DatasetMetric],
        year: int,
        message: str,
    ) -> None:
        ds, metric = dataset
        DataPointFactory.create(dataset=ds, metric=metric, date=datetime.date(year, 1, 1), value=Decimal(1))
        # Refused before the year's contents are reported, so a confirmation dialog never precedes a refusal.
        for force in (False, True):
            result = _remove(gql_client, db_instance_config, year, force=force)
            assert result['__typename'] == 'OperationInfo'
            assert message in result['messages'][0]['message']
        assert _years(db_instance_config) == (2010, 2022, None)

    @pytest.mark.usefixtures('dataset')
    def test_the_only_inventory_year_stays(self, gql_client: PathsTestClient, db_instance_config: InstanceConfig) -> None:
        db_instance_config.update_years(min_historical=2022)
        result = _remove(gql_client, db_instance_config, 2022, force=True)
        assert result['__typename'] == 'OperationInfo'
        assert 'only inventory year' in result['messages'][0]['message']

    @pytest.mark.usefixtures('dataset')
    def test_a_year_with_a_submission_cannot_be_removed(
        self, gql_client: PathsTestClient, db_instance_config: InstanceConfig
    ) -> None:
        FrameworkConfigFactory.create(instance_config=db_instance_config)
        submissions.create_submission(db_instance_config, period_start=2021, user=None)

        result = _remove(gql_client, db_instance_config, 2021, force=True)
        assert result['__typename'] == 'OperationInfo'
        assert 'has a submission (draft)' in result['messages'][0]['message']
        assert _years(db_instance_config) == (2010, 2022, None)

    @pytest.mark.usefixtures('dataset')
    def test_an_inventory_year_cannot_be_added_twice(
        self, gql_client: PathsTestClient, db_instance_config: InstanceConfig
    ) -> None:
        result = _add(gql_client, db_instance_config, 2015)
        assert result['__typename'] == 'OperationInfo'
        assert 'already an inventory year' in result['messages'][0]['message']


def test_provider_defaults_neither_block_nor_leave_with_a_year(
    gql_client: PathsTestClient, db_instance_config: InstanceConfig, dataset: tuple[Dataset, DatasetMetric]
) -> None:
    ds, _metric = dataset
    assert ds.schema is not None
    _add(gql_client, db_instance_config, 2023)
    factor = DatasetMetricFactory.create(schema=ds.schema, name='factor', label='Factor', unit='')
    default = DataPointFactory.create(dataset=ds, metric=factor, date=datetime.date(2023, 1, 1), value=Decimal('1.1'))
    DataPointEvidence.objects.create(data_point=default, kind=DataEvidenceKind.PROVIDER_DEFAULT)

    removed = _remove(gql_client, db_instance_config, 2023)
    assert removed['__typename'] == 'RemoveInventoryYearResult', removed
    assert removed['deletedCells'] == 1
    assert DataPoint.objects.filter(pk=default.pk).exists()
