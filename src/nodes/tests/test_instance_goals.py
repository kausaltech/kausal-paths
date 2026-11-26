"""An instance's goal for an outcome node: amounts or changes relative to an inventory year."""

from typing import Any

from django.test import Client

import pytest

from paths.tests.graphql import PathsTestClient

from frameworks.bisko.demo import targets
from nodes.instance_loader import InstanceLoader
from nodes.instance_serialization import build_instance_snapshot
from nodes.models import InstanceConfig, InstanceModelLogEntry, NodeConfig
from nodes.roles import instance_editor_role
from nodes.tests.test_action_hooks_graphql import _heat, make_instance, make_municipality
from nodes.tests.test_create_action import CREATE as CREATE_ACTION
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db

gql = str

GOAL_FIELDS = """
    goals {
        id label default referenceYear
        outcomeNode { identifier }
        enteredValues { year value }
        values { year goal actual isForecast }
    }
"""

SET = gql(
    """
mutation Set($instanceId: ID!, $nodeId: ID!, $input: InstanceGoalInput!) {
    instanceEditor(instanceId: $instanceId) { setInstanceGoal(nodeId: $nodeId, input: $input) { ... on InstanceType { %s } } }
}
"""
    % GOAL_FIELDS
)

CLEAR = gql(
    """
mutation Clear($instanceId: ID!, $nodeId: ID!) {
    instanceEditor(instanceId: $instanceId) { clearInstanceGoal(nodeId: $nodeId) { ... on InstanceType { %s } } }
}
"""
    % GOAL_FIELDS
)


@pytest.fixture
def ic() -> InstanceConfig:
    return make_instance()


@pytest.fixture
def client(client, ic: InstanceConfig) -> PathsTestClient:
    client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(ic)
    return tc


def _set(client: PathsTestClient, ic: InstanceConfig, node: str, **input: Any) -> list[dict[str, Any]]:
    data = client.query_data(SET, variables={'instanceId': ic.identifier, 'nodeId': node, 'input': input})
    return data['instanceEditor']['setInstanceGoal']['goals']


def _refused(client: PathsTestClient, ic: InstanceConfig, node: str, **input: Any) -> str:
    variables = {'instanceId': ic.identifier, 'nodeId': node, 'input': input}
    return ' '.join(error.get('message', '') for error in client.query_errors(SET, variables=variables))


def _path(goal: dict[str, Any], *years: int) -> dict[int, tuple[float | None, float | None]]:
    by_year = {row['year']: (row['goal'], row['actual']) for row in goal['values']}
    return {year: by_year[year] for year in years}


def test_an_absolute_goal_runs_from_the_last_inventory_year_to_the_amount_given(
    client: PathsTestClient, ic: InstanceConfig
) -> None:
    _heat(ic, is_outcome=True)

    (goal,) = _set(client, ic, 'heat', values=[{'year': 2030, 'value': 60}], label='Climate neutral heat')

    assert (goal['label'], goal['default'], goal['referenceYear']) == ('Climate neutral heat', True, None)
    assert goal['outcomeNode']['identifier'] == 'heat'
    assert goal['enteredValues'] == [{'year': 2030, 'value': 60}]
    # The path and the goal are given for the same years, so the gap can be read in each of them.
    assert _path(goal, 2021, 2022, 2026, 2030, 2035) == {
        2021: (None, 100),
        2022: (100, 100),
        2026: (80, 100),
        2030: (60, 100),
        2035: (None, 100),
    }
    assert InstanceModelLogEntry.objects.filter(operation__instance_config=ic, action='node.goal.set').count() == 1


def test_a_relative_goal_is_a_change_from_the_value_of_an_inventory_year(client: PathsTestClient, ic: InstanceConfig) -> None:
    _heat(ic, is_outcome=True)

    (goal,) = _set(client, ic, 'heat', referenceYear=2019, values=[{'year': 2028, 'value': -0.4}, {'year': 2034, 'value': -1}])

    # What was entered is kept as entered; the amounts are worked out when the goal is read.
    assert goal['referenceYear'] == 2019
    assert goal['enteredValues'] == [{'year': 2028, 'value': -0.4}, {'year': 2034, 'value': -1}]
    assert _path(goal, 2022, 2028, 2031, 2034) == {2022: (100, 100), 2028: (60, 100), 2031: (30, 100), 2034: (0, 100)}


def test_a_new_goal_replaces_the_old_one_and_clearing_removes_it(client: PathsTestClient, ic: InstanceConfig) -> None:
    _heat(ic, is_outcome=True)
    _set(client, ic, 'heat', values=[{'year': 2030, 'value': 60}])

    (goal,) = _set(client, ic, 'heat', values=[{'year': 2035, 'value': 0}])
    assert goal['enteredValues'] == [{'year': 2035, 'value': 0}]

    data = client.query_data(CLEAR, variables={'instanceId': ic.identifier, 'nodeId': 'heat'})
    assert data['instanceEditor']['clearInstanceGoal']['goals'] == []


def test_goals_that_cannot_hold_are_refused(client: PathsTestClient, ic: InstanceConfig) -> None:
    heat = _heat(ic, is_outcome=True)
    other = NodeConfig.objects.with_spec().get(pk=heat.pk)
    assert other.spec is not None

    def point(year: int, value: float) -> dict[str, float]:
        return {'year': year, 'value': value}

    refusals: list[tuple[str, dict[str, Any]]] = [
        ('at least one year', {'values': []}),
        ('one value for each year', {'values': [point(2030, 1), point(2030, 2)]}),
        ('not after the last inventory year', {'values': [point(2022, 50)]}),
        ('after the last year of the model', {'values': [point(2040, 50)]}),
        ('cannot be negative', {'values': [point(2030, -5)]}),
        ('not an inventory year', {'referenceYear': 2025, 'values': [point(2030, -0.5)]}),
        ('cannot exceed 100 %', {'referenceYear': 2019, 'values': [point(2030, -1.5)]}),
    ]
    for message, input in refusals:
        assert message in _refused(client, ic, 'heat', **input)
    assert 'not found' in _refused(client, ic, 'nowhere', values=[point(2030, 1)])

    other.spec.is_outcome = False
    NodeConfig.objects.filter(pk=heat.pk).update(spec=other.spec)
    ic.invalidate_cache()
    assert 'not an outcome' in _refused(client, ic, 'heat', values=[point(2030, 1)])
    assert InstanceModelLogEntry.objects.filter(operation__instance_config=ic, action='node.goal.set').count() == 0


def test_a_municipal_editor_sets_a_goal_on_an_outcome_its_framework_owns(ic: InstanceConfig) -> None:
    municipality = make_municipality(ic)
    heat = ic.nodes.get(identifier='heat')
    editor = UserFactory.create()
    instance_editor_role.assign_user(municipality, editor)
    django_client = Client()
    django_client.force_login(editor)
    client = PathsTestClient(django_client)
    client.set_instance(municipality)

    (goal,) = _set(client, municipality, 'heat', referenceYear=2020, values=[{'year': 2030, 'value': -0.5}])

    assert _path(goal, 2022, 2030) == {2022: (100, 100), 2030: (50, 100)}
    # The goal is a local setting for the inherited node: the template has none, and neither do its other members.
    municipality.refresh_from_db()
    assert [settings.node_uuid for settings in municipality.node_settings if settings.goals is not None] == [heat.uuid]
    template_heat = NodeConfig.objects.with_spec().get(pk=heat.pk)
    assert template_heat.spec is not None
    assert template_heat.spec.goals.root == []
    # It is part of the municipality's snapshot, so it survives publishing and restoring.
    context = InstanceLoader(snapshot=build_instance_snapshot(municipality)).context
    assert [entry.reference_year for entry in context.get_node('heat').goals.root] == [2020]

    # A measure moves the path, not the goal: the gap is the difference between the two.
    effect = {'target': {'outputPort': {'nodeId': str(heat.uuid)}}, 'values': [{'year': 2030, 'value': -30}]}
    variables = {'instanceId': municipality.identifier, 'input': {'name': 'Heat pumps', 'effects': [effect]}}
    client.query_data(CREATE_ACTION, variables=variables)
    (goal,) = _set(client, municipality, 'heat', referenceYear=2020, values=[{'year': 2030, 'value': -0.5}])
    assert _path(goal, 2030) == {2030: (50, 70)}

    InstanceConfig.objects.filter(pk=municipality.pk).update(is_locked=True)
    assert _refused(client, municipality, 'heat', values=[{'year': 2030, 'value': 10}])


def test_the_demo_target_is_set_once_and_never_over_a_target_someone_entered(
    ic: InstanceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    municipality = make_municipality(ic)
    monkeypatch.setattr(targets, 'TARGET_NODE', 'heat')

    assert targets.seed_demo_target(municipality) == f'{municipality.identifier}: demo target on heat against 2018'

    context = InstanceLoader(snapshot=build_instance_snapshot(municipality)).context
    (goal,) = context.get_node('heat').goals.root
    # The model of the test ends in 2035, so the later points of the demo path are left out.
    assert (goal.reference_year, [(value.year, value.value) for value in goal.values]) == (2018, [(2030, -0.42)])
    municipality.refresh_from_db()
    assert targets.seed_demo_target(municipality) is None
