"""Figures stored on submissions: what is computed, under which settings, and when."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from django.core.management import call_command

import pytest

from kausal_common.datasets.tests.factories import DataPointFactory, DatasetFactory, DatasetMetricFactory

from paths.tests.graphql import PathsTestClient

from datasets.materialization import refresh_dataset_materialization
from frameworks import submissions as ops
from frameworks.indicators import IndicatorComputationError, compute_indicators, indicators_are_stale
from frameworks.models import Submission
from nodes.change_ops import change_operation
from nodes.defs.indicator_defs import SubmissionIndicatorsSpec
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.instance_loader import InstanceLoader
from nodes.instance_parser import parse_instance_snapshot
from nodes.models import InstanceChangeOperation, InstanceConfig, NodeInputPortBinding
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory
from nodes.units import unit_registry
from nodes.value_validation import ValueContract
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from django.test import Client

    from kausal_common.datasets.models import DataPoint

pytestmark = pytest.mark.django_db

SECTORS = ('homes', 'industry', 'transport')


def model_config(**overrides: Any) -> dict[str, Any]:
    """
    Describe a model with energy by sector, a quality score, and a required sector nobody supplies.

    Energy is 10 + 30 in 2020 and 20 + 40 in 2021. Transport energy is required and absent.
    """
    config: dict[str, Any] = {
        'id': 'indicator-model',
        'name': 'Indicator model',
        'owner': 'Test',
        'default_language': 'en',
        'target_year': 2030,
        'reference_year': 2020,
        'minimum_historical_year': 2020,
        'maximum_historical_year': 2021,
        'params': [
            {'id': 'weather_correction', 'label': 'Weather correction', 'value': True, 'is_customizable': True},
            {
                'id': 'facilities_in_commerce',
                'type': 'bool',
                'label': 'Facilities in commerce',
                'value': False,
                'is_customizable': False,
            },
        ],
        'dimensions': [
            {
                'id': 'sector',
                'label': 'Sector',
                'categories': [
                    {'id': 'homes', 'label': 'Homes'},
                    {'id': 'industry', 'label': 'Industry'},
                    {'id': 'transport', 'label': 'Transport'},
                ],
            }
        ],
        'nodes': [
            {
                'id': 'homes_energy',
                'name': 'Homes',
                'type': 'simple.AdditiveNode',
                'quantity': 'energy',
                'unit': 'MWh/a',
                'historical_values': [[2020, 10.0], [2021, 20.0]],
                'output_nodes': [{'id': 'energy', 'to_dimensions': [{'id': 'sector', 'categories': ['homes']}]}],
            },
            {
                'id': 'industry_energy',
                'name': 'Industry',
                'type': 'simple.AdditiveNode',
                'quantity': 'energy',
                'unit': 'MWh/a',
                'historical_values': [[2020, 30.0], [2021, 40.0]],
                'output_nodes': [{'id': 'energy', 'to_dimensions': [{'id': 'sector', 'categories': ['industry']}]}],
            },
            {
                'id': 'energy',
                'name': 'Energy',
                'type': 'simple.AdditiveNode',
                'quantity': 'energy',
                'unit': 'MWh/a',
                'input_dimensions': ['sector'],
                'output_dimensions': ['sector'],
                'output_nodes': [{'id': 'balance'}],
            },
            {
                'id': 'balance',
                'name': 'Balance',
                'type': 'simple.AdditiveNode',
                'quantity': 'energy',
                'unit': 'MWh/a',
                'input_dimensions': ['sector'],
                'output_dimensions': ['sector'],
                'input_validation': {'energy': {'shape': 'sectors'}},
            },
            {
                'id': 'quality',
                'name': 'Quality',
                'type': 'simple.AdditiveNode',
                'quantity': 'fraction',
                'unit': '%',
                'historical_values': [[2020, 50.0]],
                'output_nodes': [{'id': 'checked'}],
            },
            {
                'id': 'checked',
                'name': 'Checked',
                'type': 'simple.AdditiveNode',
                'quantity': 'fraction',
                'unit': '%',
                'input_validation': {'quality': {'required': True}},
            },
        ],
        'shapes': [
            {
                'id': 'sectors',
                'dimensions': ['sector'],
                'combinations': [{'id': sector, 'categories': {'sector': sector}} for sector in SECTORS],
                'required': [{'id': sector, 'combinations': [sector]} for sector in SECTORS],
            }
        ],
        'submission_indicators': {
            'settings': [{'parameter': 'weather_correction', 'value': False}],
            'declarations': ['facilities_in_commerce'],
            'indicators': [
                {'id': 'final_energy', 'node': 'energy', 'dimensions': ['sector']},
                {'id': 'total_energy', 'node': 'energy'},
                {'id': 'data_quality', 'node': 'quality', 'composition': 'weighted_mean', 'weight': 'total_energy'},
            ],
        },
    }
    config.update(overrides)
    return config


def load(config: dict[str, Any]) -> tuple[InstanceLoader, SubmissionIndicatorsSpec]:
    snapshot = parse_instance_snapshot(config, instance_uuid=uuid4())
    # Declare the inventory calendar: without one, requirements cannot be counted year by year.
    snapshot.spec.years.skipped = []
    loader = InstanceLoader(snapshot=snapshot)
    spec = loader.instance.source_snapshot.spec.submission_indicators if loader.instance.source_snapshot else None
    if spec is None:
        spec = SubmissionIndicatorsSpec.model_validate(config['submission_indicators'])
    return loader, spec


def test_figures_are_read_for_each_year_with_their_dimensions() -> None:
    loader, spec = load(model_config())

    computed = compute_indicators(loader.instance, spec, [2019, 2020, 2021])

    first = computed[2020].values
    assert first['final_energy'].unit == 'MWh/a'
    assert first['final_energy'].dimensions == ['sector']
    assert first['final_energy'].total == 40
    assert {cell.categories['sector']: cell.value for cell in first['final_energy'].cells} == {'homes': 10, 'industry': 30}
    assert (first['total_energy'].total, first['total_energy'].cells) == (40, [])
    assert (first['data_quality'].total, first['data_quality'].composition, first['data_quality'].weight) == (
        50,
        'weighted_mean',
        'total_energy',
    )
    assert computed[2021].values['final_energy'].total == 60
    assert computed[2019].values['final_energy'].total is None


def test_completeness_counts_requirements_not_values() -> None:
    loader, spec = load(model_config())

    computed = compute_indicators(loader.instance, spec, [2020, 2021])

    # Three required sectors and one required quality score; transport is the one not met.
    assert (computed[2020].requirements_met, computed[2020].requirements_applicable) == (3, 4)
    assert (computed[2021].requirements_met, computed[2021].requirements_applicable) == (3, 4)


def test_prescribed_settings_and_declarations_are_applied_and_recorded() -> None:
    loader, spec = load(model_config())
    context = loader.context
    assert context.get_parameter('weather_correction').get() is True

    computed = compute_indicators(loader.instance, spec, [2020])

    assert context.get_parameter('weather_correction').get() is False
    assert context.active_normalization is None
    assert computed[2020].declarations == {'facilities_in_commerce': False}


@pytest.mark.parametrize(
    ('indicator', 'message'),
    [
        ({'id': 'missing', 'node': 'nowhere'}, 'node nowhere does not exist'),
        ({'id': 'wrong', 'node': 'energy', 'dimensions': ['carrier']}, 'has no dimension carrier'),
        ({'id': 'mean', 'node': 'energy', 'composition': 'none'}, 'only a summed indicator can reduce'),
    ],
)
def test_a_declaration_the_model_cannot_serve_fails_the_whole_year(indicator: dict[str, Any], message: str) -> None:
    loader, _ = load(model_config())
    spec = SubmissionIndicatorsSpec.model_validate({'indicators': [indicator]})

    with pytest.raises(IndicatorComputationError, match=message):
        compute_indicators(loader.instance, spec, [2020])


def test_a_weight_must_be_a_summed_indicator() -> None:
    with pytest.raises(ValueError, match='weight must be a summed indicator'):
        SubmissionIndicatorsSpec.model_validate({
            'indicators': [
                {'id': 'score', 'node': 'quality', 'composition': 'weighted_mean', 'weight': 'other'},
                {'id': 'other', 'node': 'quality', 'composition': 'none'},
            ]
        })


# --- stored on submissions ---


def stored_model(*, indicator_node: str = 'consumer') -> tuple[InstanceConfig, DataPoint]:
    """Build a database-backed instance whose one node delivers a dataset value of 7 t/a for 2020."""
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(
        years=YearsSpec(reference=2020, min_historical=2020, max_historical=2021, target=2030, skipped=[]),
        submission_indicators=SubmissionIndicatorsSpec.model_validate({
            'indicators': [{'id': 'emissions', 'node': indicator_node}]
        }),
    )
    config = InstanceConfigFactory.create(
        identifier=instance.id, instance=instance, config_source='database', owner='Test', spec=spec
    )
    assert config.spec is not None
    config.spec.features.use_datasets_from_db = True
    config.save(update_fields=['spec'])
    dataset = DatasetFactory.create(identifier='activity', scope=config)
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='value', label='Value', unit='t/a')
    point = DataPointFactory.create(dataset=dataset, metric=metric, date=date(2020, 1, 1), value=Decimal(7))
    DataPointFactory.create(dataset=dataset, metric=metric, date=date(2021, 1, 1), value=Decimal(8))
    port_id = uuid4()
    unit = unit_registry.parse_units('t/a')
    node = NodeConfigFactory.create(
        instance=config,
        identifier='consumer',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
            input_ports=[InputPortDef(id=port_id, unit=unit, quantity='emissions', validation=ValueContract(required=True))],
            output_ports=[OutputPortDef(id=uuid4(), unit=unit, quantity='emissions', dimensions=[])],
        ),
    )
    NodeInputPortBinding.objects.create(instance=config, node=node, port_id=port_id, dataset=dataset, metric=metric)
    return config, point


def edit(ic: InstanceConfig, point: DataPoint | None = None, value: int | None = None) -> None:
    """Run a change operation, writing a value the way the dataset editor does when one is given."""
    with change_operation(ic, user=None, action='test.edit'):
        if point is not None:
            point.value = Decimal(value) if value is not None else None
            point.save()
            refresh_dataset_materialization(point.dataset)


def test_an_edit_rewrites_the_figures_of_open_submissions() -> None:
    ic, point = stored_model()
    submission = ops.create_submission(ic, period_start=2020, user=None)
    # A new draft shows the current figures at once.
    assert submission.indicators is not None
    assert submission.indicators.values['emissions'].total == 7

    edit(ic)

    submission.refresh_from_db()
    indicators = submission.indicators
    assert indicators is not None
    assert indicators.status == 'computed', indicators.reason
    assert (indicators.values['emissions'].total, indicators.values['emissions'].unit) == (7, 't/a')
    assert indicators.completeness is not None
    assert (indicators.completeness.met, indicators.completeness.applicable) == (1, 1)
    assert indicators.provenance.draft_head_token == ic.draft_head_token
    assert not indicators_are_stale(submission, ic)

    edit(ic, point, 9)
    submission.refresh_from_db()
    assert submission.indicators is not None
    assert submission.indicators.values['emissions'].total == 9


def test_a_write_outside_a_change_operation_leaves_the_figures_stale() -> None:
    ic, _ = stored_model()
    submission = ops.create_submission(ic, period_start=2020, user=None)
    edit(ic)
    submission.refresh_from_db()

    # An import or a sync moves the draft head without announcing the change.
    InstanceChangeOperation.objects.create(instance_config=ic, action='import', source='cli')

    assert indicators_are_stale(submission, ic)


def test_a_draft_that_cannot_be_computed_is_recorded_and_the_edit_stands() -> None:
    ic, _ = stored_model(indicator_node='nowhere')
    submission = ops.create_submission(ic, period_start=2020, user=None)

    edit(ic)

    submission.refresh_from_db()
    assert submission.indicators is not None
    assert submission.indicators.status == 'not_computable'
    assert submission.indicators.reason is not None
    assert 'node nowhere does not exist' in submission.indicators.reason
    assert submission.indicators.values == {}
    assert InstanceChangeOperation.objects.filter(instance_config=ic, action='test.edit').exists()


def test_finalising_freezes_figures_from_the_pinned_revision_and_locks_the_year() -> None:
    ic, point = stored_model()
    submission = ops.create_submission(ic, period_start=2020, user=None)
    ops.request_review(submission, user=None)

    final = ops.finalise(submission, user=None)

    assert final.indicators is not None
    assert final.indicators.values['emissions'].total == 7
    assert final.indicators.provenance.instance_revision == final.instance_revision_id
    assert final.indicators.provenance.draft_head_token is None
    assert ops.locked_years(ic) == {2020}
    with pytest.raises(ops.YearLockedError, match='2020 has a final submission'):
        ops.require_years_unlocked(ic, [2020, 2021])
    ops.require_years_unlocked(ic, [2021])

    # A later edit of the draft moves no frozen figure, and a final submission is never stale.
    edit(ic, point, 9)
    final.refresh_from_db()
    assert final.indicators is not None
    assert final.indicators.values['emissions'].total == 7
    assert not indicators_are_stale(final, ic)

    # A correction unlocks the year and gets figures of its own from the current draft.
    correction = ops.create_submission(ic, period_start=2020, user=None)
    assert ops.locked_years(ic) == frozenset()
    edit(ic)
    correction.refresh_from_db()
    assert correction.indicators is not None
    assert correction.indicators.values['emissions'].total == 9
    ops.discard(correction)
    assert ops.locked_years(ic) == {2020}


def test_a_final_submission_must_have_its_figures() -> None:
    ic, _ = stored_model(indicator_node='nowhere')
    submission = ops.create_submission(ic, period_start=2020, user=None)
    ops.request_review(submission, user=None)

    with pytest.raises(ops.SubmissionError, match='figures of the submission could not be computed'):
        ops.finalise(submission, user=None)

    submission.refresh_from_db()
    assert submission.status == 'in_review'
    ic.refresh_from_db()
    assert ic.live_revision_id is None


def test_backfill_computes_a_final_submission_from_the_revision_it_pins() -> None:
    ic, point = stored_model()
    submission = ops.create_submission(ic, period_start=2020, user=None)
    ops.request_review(submission, user=None)
    final = ops.finalise(submission, user=None)
    # The draft moves on, and the revision is superseded as the live one by a later publication.
    edit(ic, point, 9)
    ic.refresh_from_db()
    ic.publish_instance()
    Submission.objects.filter(pk=final.pk).update(indicators=None)

    call_command('materialise_submission_indicators', ic.identifier, '--apply')

    final.refresh_from_db()
    assert final.indicators is not None
    assert final.instance_revision_id != InstanceConfig.objects.get(pk=ic.pk).live_revision_id
    # What was delivered, not what the draft says today.
    assert final.indicators.values['emissions'].total == 7


def test_backfill_is_a_dry_run_unless_applied() -> None:
    ic, _ = stored_model()
    submission = ops.create_submission(ic, period_start=2020, user=None)
    Submission.objects.filter(pk=submission.pk).update(indicators=None)

    call_command('materialise_submission_indicators', ic.identifier)

    submission.refresh_from_db()
    assert submission.indicators is None


def test_the_dataset_editor_refuses_values_of_a_locked_year(client: Client) -> None:
    ic, point = stored_model()
    submission = ops.create_submission(ic, period_start=2020, user=None)
    ops.request_review(submission, user=None)
    ops.finalise(submission, user=None)
    client.force_login(UserFactory.create(is_superuser=True))
    gql = PathsTestClient(client)
    gql.set_instance(ic)
    update = """
        mutation ($instanceId: ID!, $datasetId: ID!, $pointId: ID!) {
          instanceEditor(instanceId: $instanceId) {
            datasetEditor(datasetId: $datasetId) {
              updateDataPoints(input: [{dataPointId: $pointId, input: {value: 11}}]) {
                __typename
                ... on OperationInfo { messages { message } }
              }
            }
          }
        }
    """
    variables = {'instanceId': str(ic.pk), 'datasetId': str(point.dataset.uuid), 'pointId': str(point.uuid)}

    refused = gql.query_data(update, variables=variables)['instanceEditor']['datasetEditor']['updateDataPoints']
    assert refused['__typename'] == 'OperationInfo', refused
    assert '2020 has a final submission and is read-only' in refused['messages'][0]['message']
    point.refresh_from_db()
    assert point.value == 7

    locked = gql.query_data('query { instance { editor { dataEntry { lockedYears } } } }')
    assert locked['instance']['editor']['dataEntry']['lockedYears'] == [2020]

    # A correction unlocks the year.
    ops.create_submission(ic, period_start=2020, user=None)
    accepted = gql.query_data(update, variables=variables)['instanceEditor']['datasetEditor']['updateDataPoints']
    assert accepted['__typename'] != 'OperationInfo', accepted
    point.refresh_from_db()
    assert point.value == 11
