from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.node import Node
    from nodes.scenario import Scenario
    from params import Parameter

pytestmark = pytest.mark.django_db


def test_context_get_parameter_global(context: Context, number_parameter: Parameter[Any]):
    context.add_global_parameter(number_parameter)
    assert number_parameter.global_id == number_parameter.local_id
    assert context.get_parameter(number_parameter.global_id) == number_parameter


def test_context_get_parameter_local(context: Context, node: Node, number_parameter: Parameter[Any]):
    node.add_parameter(number_parameter)
    assert number_parameter.global_id != number_parameter.local_id
    assert context.get_parameter(number_parameter.global_id) == number_parameter


def test_context_activate_scenario_sets_active_scenario(context: Context, scenario: Scenario):
    assert context.active_scenario != scenario
    context.activate_scenario(scenario)
    assert context.active_scenario == scenario
