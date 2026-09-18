"""
The custom scenario is a diff, and these tests pin down what it is a diff *against*.

Until 18 Sep 2026 the answer was always the default scenario, whatever the user was
looking at. The two GraphQL tests here are the two user-visible consequences of that,
written from the reproductions taken against `mainz-bisko`:

* overrides made in one scenario survived a visit to another and reappeared on top of
  the default, so enabling one action in the baseline produced "every action except the
  one turned off earlier";
* touching any action while a non-default scenario was active silently re-based
  everything else onto the default's values, which on Mainz made turning a measure
  *off* lower emissions.

Both now branch from the scenario that was active when the edit was made.
"""

from typing import TYPE_CHECKING, Any

import pytest

from nodes.scenario import CustomScenario, ScenarioKind
from nodes.tests.factories import ScenarioFactory
from params.storage import SettingStorage
from params.tests.factories import NumberParameterFactory

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.scenario import Scenario
    from params import Parameter

pytestmark = pytest.mark.django_db


class FakeStorage(SettingStorage):
    """The parts of the session a custom scenario touches, without a request."""

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.options: dict[str, Any] = {}
        self.active: str | None = None
        self.custom_base: str | None = None

    def reset(self):
        self.params = {}
        self.options = {}
        self.custom_base = None

    def set_param(self, id: str, val: Any):
        self.params[id] = val

    def reset_param(self, id: str):
        self.params.pop(id, None)

    def set_option(self, id: str, val: Any):
        self.options[id] = val

    def reset_option(self, id: str):
        self.options.pop(id, None)

    def clear_params(self):
        self.params = {}

    def get_customized_param_values(self) -> dict[str, Any]:
        return dict(self.params)

    def set_custom_base(self, id: str | None):
        self.custom_base = id

    def get_custom_base(self) -> str | None:
        return self.custom_base

    def set_active_scenario(self, id: str | None):
        self.active = id

    def get_active_scenario(self) -> str | None:
        return self.active


@pytest.fixture
def custom_with_storage(context: Context) -> tuple[CustomScenario, FakeStorage]:
    other = ScenarioFactory.create(id='other', kind=ScenarioKind.BASELINE)
    context.add_scenario(other)
    custom = CustomScenario(
        id='custom',
        name='Custom',
        base_scenario=context.get_default_scenario(),
    )
    context.set_custom_scenario(custom)
    storage = FakeStorage()
    custom.set_storage(storage)
    return custom, storage


def test_no_branch_yet_falls_back_to_the_default(custom_with_storage):
    custom, _ = custom_with_storage
    assert custom.resolve_base() is custom.context.get_default_scenario()


def test_a_stored_base_is_the_base(custom_with_storage):
    custom, storage = custom_with_storage
    storage.set_custom_base('other')
    assert custom.resolve_base() is custom.context.get_scenario('other')


def test_a_base_that_no_longer_exists_falls_back_and_is_forgotten(custom_with_storage):
    """A config can lose a scenario while a session still names it."""
    custom, storage = custom_with_storage
    storage.set_custom_base('a_scenario_that_was_removed')
    assert custom.resolve_base() is custom.context.get_default_scenario()
    # Cleared, so the fallback happens once rather than on every activation.
    assert storage.get_custom_base() is None


def test_the_custom_scenarios_own_id_is_not_a_valid_base(custom_with_storage):
    """Guards against recursion if a session ever stores the custom id as its own base."""
    custom, storage = custom_with_storage
    storage.set_custom_base('custom')
    assert custom.resolve_base() is custom.context.get_default_scenario()


def test_activate_applies_the_stored_base_not_the_default(custom_with_storage):
    """The base's own parameter values have to survive into the custom scenario."""
    custom, storage = custom_with_storage
    context = custom.context
    param: Parameter[Any, Any] = NumberParameterFactory.create(context=context, value=2.0)
    context.add_global_parameter(param)

    default: Scenario = context.get_default_scenario()
    other: Scenario = context.get_scenario('other')
    default.add_parameter(param, 2.0)
    other.add_parameter(param, 99.0)

    storage.set_custom_base('other')
    custom.activate()
    assert param.value == 99.0, 'the base scenario was not applied; the default won instead'


SET_PARAMETER = """
    mutation($param: ID!, $value: Boolean!) {
      setParameter(id: $param, boolValue: $value) {
        ok
      }
    }
"""

ACTIVATE_SCENARIO = """
    mutation($scenario: ID!) {
      activateScenario(id: $scenario) {
        ok
        activeScenario { id }
      }
    }
"""


@pytest.mark.usefixtures('baseline_scenario', 'custom_scenario')
def test_editing_after_switching_scenario_does_not_carry_the_earlier_diff(
    graphql_client_query_data,
    context: Context,
    action_node,
    additive_action,
):
    """
    The first reproduction: turn one action off, switch scenario, turn another on.

    The result must be the second scenario plus the one action just enabled -- not the
    default scenario plus both edits, which is what an append-only diff produced.
    """
    first = action_node.enabled_param
    second = additive_action.enabled_param
    for scenario in context.scenarios.values():
        for act in (action_node, additive_action):
            act.on_scenario_created(scenario)

    # Editing in the default scenario branches from it.
    graphql_client_query_data(SET_PARAMETER, variables={'param': first.global_id, 'value': False})
    storage = context.setting_storage
    assert storage is not None
    assert storage.get_custom_base() == context.get_default_scenario().id
    assert set(storage.get_customized_param_values()) == {first.global_id}

    # Moving to another scenario leaves the diff in place but does not extend it.
    graphql_client_query_data(ACTIVATE_SCENARIO, variables={'scenario': 'baseline'})

    # Editing there is a *new* branch, so the earlier override is gone.
    graphql_client_query_data(SET_PARAMETER, variables={'param': second.global_id, 'value': False})
    storage = context.setting_storage
    assert storage is not None
    assert storage.get_custom_base() == 'baseline'
    assert set(storage.get_customized_param_values()) == {second.global_id}, (
        'the override made in the previous scenario is still in the diff'
    )


def test_editing_in_a_named_scenario_keeps_that_scenarios_other_values(
    graphql_client_query_data,
    context: Context,
    action_node,
    baseline_scenario,
    custom_scenario,
):
    """
    The second reproduction, and the one that changed a published figure.

    A parameter the user never touched must keep the value of the scenario they were
    looking at, not the default's. On Mainz this was `selected_number`: toggling one
    action re-based all seventeen of them from Szenario 2 to Szenario 1, so turning a
    measure off lowered emissions by 957 t.
    """
    untouched: Parameter[Any, Any] = NumberParameterFactory.create(context=context, value=2.0)
    context.add_global_parameter(untouched)
    context.get_default_scenario().add_parameter(untouched, 2.0)
    baseline_scenario.add_parameter(untouched, 99.0)

    enabled = action_node.enabled_param
    for scenario in context.scenarios.values():
        action_node.on_scenario_created(scenario)

    graphql_client_query_data(ACTIVATE_SCENARIO, variables={'scenario': 'baseline'})
    assert untouched.value == 99.0

    graphql_client_query_data(SET_PARAMETER, variables={'param': enabled.global_id, 'value': False})

    assert context.active_scenario is custom_scenario
    assert untouched.value == 99.0, (
        'an untouched parameter fell back to the default scenario when the custom scenario was entered'
    )


SCENARIOS_QUERY = """
    query {
      scenarios {
        id
        kind
        baseScenario { id }
        customizedParameters
      }
    }
"""


@pytest.mark.usefixtures('baseline_scenario')
def test_graphql_reports_the_base_and_the_diff(
    graphql_client_query_data,
    context: Context,
    action_node,
    custom_scenario,
):
    """
    Report the base and the diff, which is what a "how does my scenario differ" view reads.

    Both are null and empty on every scenario but the custom one.
    """
    enabled = action_node.enabled_param
    for scenario in context.scenarios.values():
        action_node.on_scenario_created(scenario)

    graphql_client_query_data(ACTIVATE_SCENARIO, variables={'scenario': 'baseline'})
    graphql_client_query_data(SET_PARAMETER, variables={'param': enabled.global_id, 'value': False})

    data = graphql_client_query_data(SCENARIOS_QUERY)
    by_id = {s['id']: s for s in data['scenarios']}

    assert by_id['custom']['baseScenario'] == {'id': 'baseline'}
    assert by_id['custom']['customizedParameters'] == [enabled.global_id]

    for scenario_id in ('baseline', context.get_default_scenario().id):
        assert by_id[scenario_id]['baseScenario'] is None
        assert by_id[scenario_id]['customizedParameters'] == []


def test_base_is_reported_without_a_session(custom_with_storage):
    """A custom scenario with no session attached reports its fallback base, not an error."""
    custom, _ = custom_with_storage
    del custom.__pydantic_private__['_storage']
    assert not custom.has_storage()
    assert custom.resolve_base() is custom.context.get_default_scenario()
    assert custom.get_customized_param_ids() == []
