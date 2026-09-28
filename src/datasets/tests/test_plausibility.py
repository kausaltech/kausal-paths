"""Advisory dataset ranges share the validation cell locator without blocking edits."""

import datetime
from decimal import Decimal
from io import StringIO
from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.test import override_settings

import pytest

from kausal_common.datasets.models import DatasetMetricValidationRule, DimensionScope
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

from datasets.models import DatasetMetricPlausibilityRange, PlausibilitySource
from datasets.plausibility import evaluate_dataset_plausibility
from frameworks.models import OrganizationPopulation
from frameworks.tests.factories import FrameworkConfigFactory
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from kausal_common.datasets.models import DatasetMetric, DimensionCategory

    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db

Range = DatasetMetricPlausibilityRange


@pytest.fixture
def db_instance_config() -> InstanceConfig:
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2010, max_historical=2023, target=2030))
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        owner='Test Owner',
        spec=spec,
    )


@pytest.fixture
def gql_client(client, db_instance_config: InstanceConfig) -> PathsTestClient:
    user = UserFactory.create(is_superuser=True)
    client.force_login(user)
    result = PathsTestClient(client)
    result.set_instance(db_instance_config)
    return result


@pytest.fixture
def source() -> PlausibilitySource:
    return PlausibilitySource.objects.create(
        identifier='example-reference',
        name='Example reference',
        url='https://example.org/reference',
        revision='example-v1',
        method='Test reference band.',
    )


@pytest.fixture
def rig(db_instance_config: InstanceConfig):
    schema = DatasetSchemaFactory.create(name='Plausibility schema')
    metric = DatasetMetricFactory.create(schema=schema, name='amount', label='Amount', unit='MWh')
    dimension = DimensionFactory.create(name='Region')
    DatasetSchemaDimensionFactory.create(schema=schema, dimension=dimension, column_name='region')
    category_a = DimensionCategoryFactory.create(dimension=dimension, identifier='a', label='A')
    category_b = DimensionCategoryFactory.create(dimension=dimension, identifier='b', label='B')
    DimensionScope.objects.create(
        dimension=dimension,
        scope_content_type=ContentType.objects.get_for_model(db_instance_config),
        scope_id=db_instance_config.pk,
        identifier='region',
    )
    dataset = DatasetFactory.create(schema=schema, identifier='plausibility-ds', scope=db_instance_config)
    return dataset, metric, category_a, category_b


def _selection(*categories: DimensionCategory) -> dict[str, list[str]]:
    selection: dict[str, list[str]] = {}
    for category in categories:
        selection.setdefault(str(category.dimension.uuid), []).append(str(category.uuid))
    return selection


def _range(
    metric: DatasetMetric,
    source: PlausibilitySource,
    *,
    framework=None,
    instance_config=None,
    identifier: str = 'example',
    **fields: Any,
) -> DatasetMetricPlausibilityRange:
    values: dict[str, Any] = {'lower': 1.0, 'upper': 5.0} | fields
    rule = DatasetMetricPlausibilityRange(
        source=source,
        metric=metric,
        framework=framework,
        instance_config=instance_config,
        identifier=identifier,
        **values,
    )
    rule.full_clean()
    rule.save()
    return rule


def _point(dataset, metric, category, value: float | None, year: int = 2023) -> None:
    DataPointFactory.create(
        dataset=dataset,
        metric=metric,
        date=datetime.date(year, 1, 1),
        value=None if value is None else Decimal(value),
        dimension_categories=[category],
    )


def _population(db_instance_config: InstanceConfig, value: int = 100, year: int = 2023):
    config = FrameworkConfigFactory.create(instance_config=db_instance_config)
    OrganizationPopulation.objects.create(
        framework=config.framework,
        organization=db_instance_config.organization,
        year=year,
        value=value,
        source_dataset='test/population',
        source_revision='test-v1',
    )
    return config.framework


def test_population_band_uses_observed_same_year_and_skips_invalid_cells(
    rig, source: PlausibilitySource, db_instance_config: InstanceConfig
):
    dataset, metric, category_a, category_b = rig
    framework = _population(db_instance_config)
    rule = _range(metric, source, framework=framework, denominator=Range.Denominator.POPULATION)
    DatasetMetricValidationRule.objects.create(
        metric=metric, rule={'kind': 'value_range', 'enforcement': 'block_edit', 'min': 0}, order=0
    )
    _point(dataset, metric, category_a, 1000)
    _point(dataset, metric, category_b, -1000)
    _point(dataset, metric, category_a, 1000, year=2024)  # no observed denominator

    findings = evaluate_dataset_plausibility(dataset)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_uuid == rule.uuid
    assert finding.aggregation == 'cell'
    assert finding.years == [2023]
    assert finding.categories == {'region': 'a'}
    assert finding.observed == 1000
    assert finding.normalized == 10
    assert finding.denominator_value == 100
    assert unit_registry.parse_units(finding.unit) == unit_registry.parse_units('MWh/cap')
    assert finding.source_identifier == 'example-reference'


def test_bound_unit_follows_metric_denominator_and_reference(rig, source: PlausibilitySource, db_instance_config):
    _, metric, _, _ = rig
    absolute = _range(metric, source, instance_config=db_instance_config)
    per_capita = _range(
        metric, source, instance_config=db_instance_config, identifier='per-capita', denominator=Range.Denominator.POPULATION
    )
    ratio = _range(
        metric,
        source,
        instance_config=db_instance_config,
        identifier='ratio',
        reference=Range.Reference.PREVIOUS_YEAR,
        max_gap_years=1,
        lower=0.5,
        upper=2.0,
    )

    assert absolute.bound_unit == unit_registry.parse_units('MWh')
    assert per_capita.bound_unit == unit_registry.parse_units('MWh/cap')
    assert ratio.bound_unit == unit_registry.parse_units('dimensionless')


def test_reference_bounds_reject_non_finite_values(rig, source: PlausibilitySource, db_instance_config: InstanceConfig):
    _, metric, _, _ = rig
    rule = _range(metric, source, instance_config=db_instance_config)
    rule.upper = float('inf')

    with pytest.raises(ValidationError, match='finite'):
        rule.full_clean()


def test_published_source_requires_url():
    source = PlausibilitySource(identifier='no-url', name='No URL', revision='v1', method='-')
    with pytest.raises(ValidationError, match='URL'):
        source.full_clean()
    source.is_example = True
    source.full_clean()


@pytest.mark.parametrize(
    ('fields', 'message'),
    [
        ({'reference': Range.Reference.PREVIOUS_YEAR}, 'maximum gap'),
        ({'reference': Range.Reference.PREVIOUS_YEAR, 'max_gap_years': 1, 'lower': 0.0}, 'positive lower bound'),
        (
            {'reference': Range.Reference.PREVIOUS_YEAR, 'max_gap_years': 1, 'denominator': Range.Denominator.POPULATION},
            'not normalized',
        ),
        ({'max_gap_years': 1}, 'Only a previous-year'),
    ],
)
def test_reference_shape_is_validated(rig, source: PlausibilitySource, db_instance_config, fields, message):
    _, metric, _, _ = rig
    rule = Range(source=source, metric=metric, instance_config=db_instance_config, identifier='bad', lower=0.5, upper=2)
    for name, value in fields.items():
        setattr(rule, name, value)
    with pytest.raises(ValidationError, match=message):
        rule.full_clean()


def test_selection_categories_must_belong_to_their_dimension(rig, source: PlausibilitySource, db_instance_config):
    _, metric, category_a, _ = rig
    stranger = DimensionCategoryFactory.create(dimension=DimensionFactory.create(name='Other'), identifier='x')
    rule = Range(source=source, metric=metric, instance_config=db_instance_config, identifier='bad', lower=1, upper=5)

    rule.selection = {str(category_a.dimension.uuid): [str(stranger.uuid)]}
    with pytest.raises(ValidationError, match='do not belong'):
        rule.full_clean()
    rule.selection = {str(stranger.dimension.uuid): [str(stranger.uuid)]}
    with pytest.raises(ValidationError, match='not in the metric schema'):
        rule.full_clean()
    rule.selection = {str(category_a.dimension.uuid): []}
    with pytest.raises(ValidationError, match='non-empty lists'):
        rule.full_clean()


def test_sum_must_list_every_schema_dimension(rig, source: PlausibilitySource, db_instance_config):
    dataset, metric, category_a, category_b = rig
    carrier = DimensionFactory.create(name='Carrier')
    DatasetSchemaDimensionFactory.create(schema=dataset.schema, dimension=carrier, column_name='carrier')
    gas = DimensionCategoryFactory.create(dimension=carrier, identifier='gas')
    rule = Range(
        source=source,
        metric=metric,
        instance_config=db_instance_config,
        identifier='sum',
        aggregation=Range.Aggregation.SUM,
        selection=_selection(category_a, category_b),
        lower=1,
        upper=5,
    )
    with pytest.raises(ValidationError, match='every dimension'):
        rule.full_clean()
    rule.selection = _selection(category_a, category_b, gas)
    rule.full_clean()


def test_sum_is_checked_once_per_year_over_its_selection(rig, source: PlausibilitySource, db_instance_config):
    dataset, metric, category_a, category_b = rig
    rule = _range(
        metric,
        source,
        instance_config=db_instance_config,
        aggregation=Range.Aggregation.SUM,
        selection=_selection(category_a, category_b),
    )
    _point(dataset, metric, category_a, 3)
    _point(dataset, metric, category_b, 4)
    _point(dataset, metric, category_a, 2, year=2024)
    _point(dataset, metric, category_b, 2, year=2024)

    findings = evaluate_dataset_plausibility(dataset)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_uuid == rule.uuid
    assert finding.aggregation == 'sum'
    assert finding.years == [2023]
    assert finding.observed == 7
    assert finding.component_count == 2
    assert finding.complete
    assert finding.categories == {}
    assert {coordinate.category for coordinate in finding.selection} == {'a', 'b'}
    assert finding.message.startswith('The sum of 2 cells')


def test_incomplete_sum_is_only_checked_against_upper_bound(rig, source: PlausibilitySource, db_instance_config):
    dataset, metric, category_a, category_b = rig
    _range(
        metric,
        source,
        instance_config=db_instance_config,
        aggregation=Range.Aggregation.SUM,
        selection=_selection(category_a, category_b),
    )
    _point(dataset, metric, category_a, 0.5)  # far too low, but b is still empty
    _point(dataset, metric, category_b, None)
    _point(dataset, metric, category_a, 10, year=2024)  # already too high
    _point(dataset, metric, category_b, None, year=2024)

    findings = evaluate_dataset_plausibility(dataset)

    assert [(finding.years, finding.complete, finding.observed) for finding in findings] == [([2024], False, 10)]
    assert 'some still empty' in findings[0].message


def test_previous_year_ratio_respects_the_maximum_gap(rig, source: PlausibilitySource, db_instance_config):
    dataset, metric, category_a, category_b = rig
    rule = _range(
        metric,
        source,
        instance_config=db_instance_config,
        selection=_selection(category_a),
        reference=Range.Reference.PREVIOUS_YEAR,
        max_gap_years=1,
        lower=0.5,
        upper=1.5,
    )
    _point(dataset, metric, category_a, 100, year=2021)
    _point(dataset, metric, category_a, 300, year=2023)  # 2022 missing
    _point(dataset, metric, category_b, 1, year=2022)
    _point(dataset, metric, category_b, 1000, year=2023)  # outside the selection

    assert evaluate_dataset_plausibility(dataset) == []

    rule.max_gap_years = 2
    rule.save(update_fields=['max_gap_years'])
    [finding] = evaluate_dataset_plausibility(dataset)
    assert finding.reference == 'previous_year'
    assert (finding.reference_year, finding.reference_value, finding.normalized) == (2021, 100, 3)
    assert finding.denominator_value is None


def test_previous_year_sum_needs_both_years_complete(rig, source: PlausibilitySource, db_instance_config):
    dataset, metric, category_a, category_b = rig
    _range(
        metric,
        source,
        instance_config=db_instance_config,
        aggregation=Range.Aggregation.SUM,
        selection=_selection(category_a, category_b),
        reference=Range.Reference.PREVIOUS_YEAR,
        max_gap_years=1,
        lower=0.8,
        upper=1.25,
    )
    # A category swap between years keeps the sum, so the sum stays quiet...
    _point(dataset, metric, category_a, 10, year=2021)
    _point(dataset, metric, category_b, 90, year=2021)
    _point(dataset, metric, category_a, 90, year=2022)
    _point(dataset, metric, category_b, 10, year=2022)
    # ...while a doubled total in a complete year is reported, and an incomplete year is not.
    _point(dataset, metric, category_a, 150, year=2023)
    _point(dataset, metric, category_b, 50, year=2023)
    _point(dataset, metric, category_a, 400, year=2024)
    _point(dataset, metric, category_b, None, year=2024)

    [finding] = evaluate_dataset_plausibility(dataset)
    assert (finding.years, finding.reference_year, finding.normalized) == ([2023], 2022, 2)


def test_reference_selector_survives_identifier_renames(rig, source: PlausibilitySource, db_instance_config):
    dataset, metric, category_a, _ = rig
    rule = _range(metric, source, instance_config=db_instance_config, selection=_selection(category_a))
    _point(dataset, metric, category_a, 10)
    category_a.identifier = 'renamed'
    category_a.save(update_fields=['identifier'])
    schema_dimension = dataset.schema.dimensions.get(dimension=category_a.dimension)
    schema_dimension.column_name = 'renamed_region'
    schema_dimension.save(update_fields=['column_name'])

    finding = evaluate_dataset_plausibility(dataset)[0]

    assert rule.selection == {str(category_a.dimension.uuid): [str(category_a.uuid)]}
    assert finding.coordinates[0].dimension_uuid == category_a.dimension.uuid
    assert finding.coordinates[0].category_uuid == category_a.uuid
    assert finding.categories == {'renamed_region': 'renamed'}


def test_range_and_finding_are_queryable_with_shared_locator(
    rig, source: PlausibilitySource, db_instance_config: InstanceConfig, gql_client: PathsTestClient
):
    dataset, metric, category_a, category_b = rig
    _range(metric, source, instance_config=db_instance_config, selection=_selection(category_a))
    _range(
        metric,
        source,
        instance_config=db_instance_config,
        identifier='total',
        aggregation=Range.Aggregation.SUM,
        selection=_selection(category_a, category_b),
        upper=50,
    )
    _point(dataset, metric, category_a, 10)
    _point(dataset, metric, category_b, None)

    payload = gql_client.query_data(
        """
        query Plausibility($datasetId: ID!) {
            instance { editor { dataset(id: $datasetId) {
                plausibilityRanges {
                    identifier aggregation reference lower upper
                    unit { standard dimensionality { dimension value } }
                    selection { dimension category dimensionUuid categoryUuid }
                    source { identifier url revision isExample }
                }
                plausibilityFindings {
                    __typename code severity metric years normalized aggregation complete componentCount
                    unit { standard dimensionality { dimension value } }
                    coordinates { dimension category dimensionUuid categoryUuid }
                    selection { category }
                    sourceIdentifier
                }
            } } }
        }
    """,
        variables={'datasetId': str(dataset.uuid)},
    )

    result = payload['instance']['editor']['dataset']
    coordinate_a = {
        'dimension': 'region',
        'category': 'a',
        'dimensionUuid': str(category_a.dimension.uuid),
        'categoryUuid': str(category_a.uuid),
    }
    coordinate_b = coordinate_a | {'category': 'b', 'categoryUuid': str(category_b.uuid)}
    unit_info = {
        'standard': 'MWh',
        'dimensionality': [
            {'dimension': '[mass]', 'value': 1.0},
            {'dimension': '[length]', 'value': 2.0},
            {'dimension': '[time]', 'value': -2.0},
        ],
    }
    source_info = {
        'identifier': 'example-reference',
        'url': 'https://example.org/reference',
        'revision': 'example-v1',
        'isExample': False,
    }
    assert result['plausibilityRanges'] == [
        {
            'identifier': 'example',
            'aggregation': 'CELL',
            'reference': 'ABSOLUTE',
            'lower': 1.0,
            'upper': 5.0,
            'unit': unit_info,
            'selection': [coordinate_a],
            'source': source_info,
        },
        {
            'identifier': 'total',
            'aggregation': 'SUM',
            'reference': 'ABSOLUTE',
            'lower': 1.0,
            'upper': 50.0,
            'unit': unit_info,
            'selection': [coordinate_a, coordinate_b],
            'source': source_info,
        },
    ]
    assert result['plausibilityFindings'] == [
        {
            '__typename': 'DatasetPlausibilityFinding',
            'code': 'outside_reference_range',
            'severity': 'WARNING',
            'metric': 'amount',
            'years': [2023],
            'normalized': 10.0,
            'aggregation': 'CELL',
            'complete': True,
            'componentCount': 1,
            'unit': unit_info,
            'coordinates': [coordinate_a],
            'selection': [],
            'sourceIdentifier': 'example-reference',
        }
    ]


def test_data_point_ranges_resolve_cell_ranges_in_metric_units(
    rig, source: PlausibilitySource, db_instance_config: InstanceConfig, gql_client: PathsTestClient
):
    dataset, metric, category_a, category_b = rig
    metric.unit = 'MWh/a'
    metric.save(update_fields=['unit'])
    framework = _population(db_instance_config)
    national = _range(
        metric, source, framework=framework, selection=_selection(category_a), denominator=Range.Denominator.POPULATION
    )
    local = _range(metric, source, instance_config=db_instance_config, identifier='local', lower=50, upper=500)
    trend = _range(
        metric,
        source,
        instance_config=db_instance_config,
        identifier='trend',
        selection=_selection(category_a),
        reference=Range.Reference.PREVIOUS_YEAR,
        max_gap_years=1,
        lower=0.8,
        upper=1.25,
    )
    _range(  # sums have no per-point bounds
        metric,
        source,
        instance_config=db_instance_config,
        identifier='total',
        aggregation=Range.Aggregation.SUM,
        selection=_selection(category_a, category_b),
    )
    _point(dataset, metric, category_a, 800, year=2022)
    _point(dataset, metric, category_a, 1000)
    _point(dataset, metric, category_b, None, year=2024)

    payload = gql_client.query_data(
        """
        query Ranges($datasetId: ID!) {
            instance { editor { dataset(id: $datasetId) {
                dataPoints { date value plausibilityRanges {
                    lower upper reference { id identifier denominator lower upper }
                } }
            } } }
        }
        """,
        variables={'datasetId': str(dataset.uuid)},
    )
    points = payload['instance']['editor']['dataset']['dataPoints']
    by_year = {(point['date'][:4], point['value']): point['plausibilityRanges'] for point in points}

    def entry(rule, lower, upper):
        return {
            'lower': lower,
            'upper': upper,
            'reference': {
                'id': str(rule.uuid),
                'identifier': rule.identifier,
                'denominator': rule.denominator.upper(),
                'lower': rule.lower,
                'upper': rule.upper,
            },
        }

    assert by_year['2023', 1000.0] == [entry(national, 100.0, 500.0), entry(local, 50.0, 500.0), entry(trend, 640.0, 1000.0)]
    # No population for 2022, and no earlier value for the trend.
    assert by_year['2022', 800.0] == [entry(national, None, None), entry(local, 50.0, 500.0), entry(trend, None, None)]
    assert by_year['2024', None] == [entry(local, 50.0, 500.0)]


def test_missing_population_keeps_reference_but_has_no_resolved_bounds(
    rig, source: PlausibilitySource, db_instance_config: InstanceConfig, gql_client: PathsTestClient
):
    dataset, metric, category_a, _ = rig
    config = FrameworkConfigFactory.create(instance_config=db_instance_config)
    rule = _range(metric, source, framework=config.framework, denominator=Range.Denominator.POPULATION)
    _point(dataset, metric, category_a, None)
    payload = gql_client.query_data(
        """
        query Ranges($datasetId: ID!) {
            instance { editor { dataset(id: $datasetId) {
                dataPoints { plausibilityRanges { lower upper reference { id } } }
            } } }
        }
        """,
        variables={'datasetId': str(dataset.uuid)},
    )
    assert payload['instance']['editor']['dataset']['dataPoints'][0]['plausibilityRanges'] == [
        {'lower': None, 'upper': None, 'reference': {'id': str(rule.uuid)}}
    ]


@override_settings(DEBUG=True)
def test_local_example_seed_is_scoped_and_labelled(rig, db_instance_config: InstanceConfig):
    dataset, metric, category_a, _ = rig
    _point(dataset, metric, category_a, 10)

    call_command(
        'seed_plausibility_example',
        db_instance_config.identifier,
        dataset.identifier,
        '--metric',
        metric.name,
        '--year',
        '2023',
        stdout=StringIO(),
    )

    rule = DatasetMetricPlausibilityRange.objects.get(identifier='ui-example')
    assert rule.instance_config_id == db_instance_config.pk
    assert rule.framework_id is None
    assert rule.source.is_example
    assert evaluate_dataset_plausibility(dataset)[0].is_example
