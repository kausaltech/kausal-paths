"""
Creating an action from its effects (`createAction`): through a hook or an edge, absolute or relative.

See docs/architecture/action-hooks.md.
"""

from typing import Any

from django.test import Client

import pytest

from kausal_common.datasets.models import Dataset

from paths.tests.graphql import PathsTestClient

from nodes.defs.node_defs import ActionHookMode
from nodes.defs.transform_def import BackfillOp, ExtendOp, InterpolateOp
from nodes.models import InstanceConfig, NodeConfig, NodeInputPortBinding
from nodes.roles import instance_editor_role
from nodes.tests.factories import _port_id
from nodes.tests.test_action_hooks_graphql import _heat, _hooks, make_instance, make_municipality
from nodes.tests.test_action_wizard import _by_year, _category, _cells, _demand, _dimension, _rebuild
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db

gql = str

CREATE = gql("""
mutation Create($instanceId: ID!, $input: CreateActionInput!) {
    instanceEditor(instanceId: $instanceId) {
        createAction(input: $input) {
            __typename
            ... on CreateActionResult {
                action { ... on NodeInterface { id identifier } }
                datasets { id }
            }
            ... on ConstraintViolations { conflicts { code message } }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
""")

DELETE = gql("""
mutation Delete($instanceId: ID!, $nodeId: ID!) {
    instanceEditor(instanceId: $instanceId) { nodeEditor(nodeId: $nodeId) { delete { messages { message } } } }
}
""")


@pytest.fixture
def ic() -> InstanceConfig:
    return make_instance()


@pytest.fixture
def client(client, ic: InstanceConfig) -> PathsTestClient:
    client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(ic)
    return tc


def _create(client: PathsTestClient, ic: InstanceConfig, **input: Any) -> dict[str, Any]:
    data = client.query_data(CREATE, variables={'instanceId': ic.identifier, 'input': {'name': 'Heat pumps', **input}})
    result = data['instanceEditor']['createAction']
    assert result['__typename'] == 'CreateActionResult', result
    return result


def _refused(client: PathsTestClient, ic: InstanceConfig, **input: Any) -> str:
    variables = {'instanceId': ic.identifier, 'input': {'name': 'Heat pumps', **input}}
    return ' '.join(error.get('message', '') for error in client.query_errors(CREATE, variables=variables))


def _on_output(node: NodeConfig, **effect: Any) -> dict[str, Any]:
    return {'target': {'outputPort': {'nodeId': str(node.uuid)}}, **effect}


def _carrier(carrier: str) -> dict[str, str]:
    return {'dimensionId': str(_dimension('carrier').uuid), 'categoryId': str(_category('carrier', carrier).uuid)}


def _keep(*carriers: str) -> dict[str, Any]:
    keep = [str(_category('carrier', carrier).uuid) for carrier in carriers]
    return {'newDataset': {'categories': [{'dimensionId': str(_dimension('carrier').uuid), 'categoryIds': keep}]}}


def _action(ic: InstanceConfig) -> NodeConfig:
    return NodeConfig.objects.with_spec().get(instance=ic, identifier='heat_pumps')


def _dataset_of(result: dict[str, Any]) -> Dataset:
    return Dataset.objects.get(uuid=result['datasets'][0]['id'])


def test_a_target_value_on_an_output_port_becomes_a_hooked_action_with_its_own_dataset(
    client: PathsTestClient, ic: InstanceConfig
) -> None:
    demand = _demand(ic)

    result = _create(
        client,
        ic,
        description='Replace oil boilers',
        effects=[_on_output(demand, source=_keep('oil'), values=[{'year': 2030, 'categories': [_carrier('oil')], 'value': -40}])],
    )

    action = _action(ic)
    assert [(hook.node, hook.port, hook.mode) for hook in _hooks(action)] == [
        (demand.uuid, _port_id('demand'), ActionHookMode.ABSOLUTE)
    ]
    # The dataset belongs to the action, holds what was entered, and says where its forecast begins.
    dataset = _dataset_of(result)
    assert dataset.scope == action
    assert dataset.identifier is None
    assert _cells(dataset) == {(2022, 'Demand', ('oil',)): 0.0, (2030, 'Demand', ('oil',)): -40.0}
    assert dataset.spec['forecast_from'] == 2023
    # A hook never touches the history, so the binding fills forward only. The target takes no input.
    binding = NodeInputPortBinding.objects.get(node=action)
    kinds = {type(op) for op in binding.transformations}
    assert {InterpolateOp, ExtendOp} <= kinds
    assert BackfillOp not in kinds
    assert not NodeInputPortBinding.objects.filter(node=demand, source_node=action).exists()

    context = _rebuild(ic)
    oil = _by_year(context.get_node('demand').get_output_pl(), 'carrier', 'oil')
    assert (oil[2018], oil[2022], oil[2026], oil[2030], oil[2035]) == (100, 100, 80, 60, 60)
    assert set(_by_year(context.get_node('demand').get_output_pl(), 'carrier', 'gas').values()) == {100}


def test_a_relative_series_is_a_share_of_the_value_before_any_action(client: PathsTestClient, ic: InstanceConfig) -> None:
    heat = _heat(ic)

    values = [{'year': 2025, 'value': -0.1}, {'year': 2030, 'value': -0.3}]
    result = _create(client, ic, effects=[_on_output(heat, mode='RELATIVE', values=values)])

    action = _action(ic)
    assert [hook.mode for hook in _hooks(action)] == [ActionHookMode.RELATIVE]
    assert action.spec is not None
    port = action.spec.output_ports[0]
    assert (port.quantity, port.unit is not None and port.unit.dimensionless) == ('fraction', True)
    dataset = _dataset_of(result)
    assert _cells(dataset) == {(2022, 'Heat', ()): 0.0, (2025, 'Heat', ()): -0.1, (2030, 'Heat', ()): -0.3}
    values_by_year = _by_year(_rebuild(ic).get_node('heat').get_output_pl())
    assert (values_by_year[2022], values_by_year[2025], values_by_year[2030], values_by_year[2035]) == pytest.approx((
        100,
        90,
        70,
        70,
    ))


def test_an_effect_into_an_additive_input_is_an_edge_and_cannot_be_relative(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    into = {'target': {'additiveInputOf': str(demand.uuid)}}

    assert 'acts on an output port' in _refused(client, ic, effects=[{**into, 'mode': 'RELATIVE'}])

    _create(client, ic, effects=[{**into, 'values': [{'year': 2030, 'categories': [_carrier('oil')], 'value': -40}]}])
    action = _action(ic)
    assert _hooks(action) == []
    assert NodeInputPortBinding.objects.filter(node=demand, source_node=action).exists()
    # An edge acts on every year, so the action's zero has to reach back over the history.
    kinds = {type(op) for op in NodeInputPortBinding.objects.get(node=action).transformations}
    assert BackfillOp in kinds
    oil = _by_year(_rebuild(ic).get_node('demand').get_output_pl(), 'carrier', 'oil')
    assert (oil[2018], oil[2030]) == (100, 60)


def test_values_and_targets_that_cannot_hold_are_refused_and_nothing_is_created(
    client: PathsTestClient, ic: InstanceConfig
) -> None:
    demand, heat = _demand(ic), _heat(ic)

    def value(year: int, *carriers: str) -> dict[str, Any]:
        return {'year': year, 'categories': [_carrier(carrier) for carrier in carriers], 'value': -1}

    refusals = [
        ('one category of each of its dimensions', [_on_output(demand, values=[{'year': 2030, 'value': -1}])]),
        ('a category the effect does not have', [_on_output(demand, source=_keep('oil'), values=[value(2030, 'gas')])]),
        ('two values for one cell', [_on_output(demand, values=[value(2030, 'oil'), value(2030, 'oil')])]),
        ('after the last year of the model', [_on_output(demand, values=[value(2050, 'oil')])]),
        ('each output only once', [_on_output(heat), _on_output(heat, mode='RELATIVE')]),
        ('needs at least one effect', []),
    ]
    for message, effects in refusals:
        assert message in _refused(client, ic, effects=effects)
    assert not NodeConfig.objects.filter(instance=ic, identifier='heat_pumps').exists()
    assert not Dataset.objects.filter(identifier__isnull=True).exists()


def test_deleting_the_action_removes_its_hook_and_its_dataset(client: PathsTestClient, ic: InstanceConfig) -> None:
    heat = _heat(ic)
    result = _create(client, ic, effects=[_on_output(heat, values=[{'year': 2030, 'value': -10}])])
    dataset = _dataset_of(result)
    assert _by_year(_rebuild(ic).get_node('heat').get_output_pl())[2030] == 90

    client.query_data(DELETE, variables={'instanceId': ic.identifier, 'nodeId': 'heat_pumps'})

    assert not NodeConfig.objects.filter(instance=ic, identifier='heat_pumps').exists()
    assert not Dataset.objects.filter(pk=dataset.pk).exists()
    assert _by_year(_rebuild(ic).get_node('heat').get_output_pl())[2030] == 100


def test_a_municipal_editor_creates_an_action_on_a_node_its_framework_opened(ic: InstanceConfig) -> None:
    # The framework holds every input at the last historical year; the action's own data must still count.
    spec = ic.ensure_spec()
    spec.features.hold_inputs_after_maximum_historical_year = True
    ic.spec = spec
    ic.save(update_fields=['spec'])
    municipality = make_municipality(ic)
    heat, demand = ic.nodes.get(identifier='heat'), ic.nodes.get(identifier='demand')
    editor = UserFactory.create()
    instance_editor_role.assign_user(municipality, editor)
    django_client = Client()
    django_client.force_login(editor)
    tc = PathsTestClient(django_client)
    tc.set_instance(municipality)

    assert 'does not let actions act on it' in _refused(tc, municipality, effects=[_on_output(demand)])

    result = _create(tc, municipality, effects=[_on_output(heat, values=[{'year': 2030, 'value': -10}])])

    assert _dataset_of(result).scope == _action(municipality)
    values = _by_year(_rebuild(municipality).get_node('heat').get_output_pl())
    assert (values[2022], values[2026], values[2030], values[2035]) == (100, 95, 90, 90)
    # The framework's node is the same for everyone else.
    assert _by_year(_rebuild(ic).get_node('heat').get_output_pl())[2030] == 100

    # The editor can also remove it again: the action's own input selection and dataset go with it.
    tc.query_data(DELETE, variables={'instanceId': municipality.identifier, 'nodeId': 'heat_pumps'})
    assert not NodeConfig.objects.filter(instance=municipality).exists()
    assert not Dataset.objects.filter(identifier__isnull=True).exists()
    assert _by_year(_rebuild(municipality).get_node('heat').get_output_pl())[2030] == 100

    django_client.force_login(UserFactory.create())
    tc = PathsTestClient(django_client)
    tc.set_instance(municipality)
    assert _refused(tc, municipality, effects=[_on_output(heat)])
    assert not NodeConfig.objects.filter(instance=municipality).exists()
