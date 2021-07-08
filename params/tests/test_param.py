import pytest


def test_parameter_add_scenario_setting(parameter, scenario):
    value = 'foo'
    parameter.add_scenario_setting(scenario, value)
    assert parameter.get_scenario_setting(scenario) == value


def test_parameter_add_scenario_setting_twice(parameter, scenario):
    parameter.add_scenario_setting(scenario, 'foo')
    with pytest.raises(Exception):
        parameter.add_scenario_setting(scenario, 'bar')


def test_parameter_global_id_global_param(parameter):
    assert parameter.global_id == parameter.local_id


def test_parameter_global_id_node_param(node, parameter):
    parameter.set_node(node)
    assert parameter.global_id == f'{node.id}.{parameter.local_id}'


@pytest.mark.parametrize('setting_exists', [True, False])
def test_bool_parameter_reset_to_scenario_setting(bool_parameter, scenario, setting_exists):
    if setting_exists:
        bool_parameter.add_scenario_setting(scenario, True)
        expected = True
    else:
        expected = None
    assert bool_parameter.get() is None
    bool_parameter.reset_to_scenario_setting(scenario)
    assert bool_parameter.get() is expected
