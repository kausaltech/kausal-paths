import pytest

from params.tests.factories import ParameterFactory
from nodes.tests.factories import NodeFactory, ScenarioFactory

pytestmark = pytest.mark.django_db


def test_parameter_add_scenario_setting():
    value = 'foo'
    param = ParameterFactory()
    scenario = ScenarioFactory()
    param.add_scenario_setting(scenario.id, value)
    assert param.get_scenario_setting(scenario.id) == value


def test_parameter_add_scenario_setting_twice():
    param = ParameterFactory()
    scenario = ScenarioFactory()
    param.add_scenario_setting(scenario.id, 'foo')
    with pytest.raises(Exception):
        param.add_scenario_setting(scenario.id, 'bar')


def test_parameter_global_id_global_param():
    param = ParameterFactory()
    assert param.global_id == param.local_id


def test_parameter_global_id_node_param():
    node = NodeFactory()
    param = ParameterFactory()
    param.set_node(node)
    assert param.global_id == f'{node.id}.{param.local_id}'
