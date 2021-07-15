import pytest

from params.param import ValidationError


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


@pytest.mark.parametrize('value', [3, 3.5, '3', '3.5'])
def test_number_parameter_clean_to_float(number_parameter, value):
    assert number_parameter.clean(value) == float(value)


@pytest.mark.parametrize('value', [None, True, [], 'foo'])
def test_number_parameter_clean_not_convertible(number_parameter, value):
    with pytest.raises(ValidationError):
        number_parameter.clean(value)


def test_number_parameter_clean_too_small(number_parameter):
    value = number_parameter.min_value - 0.1
    with pytest.raises(ValidationError):
        number_parameter.clean(value)


def test_number_parameter_clean_too_large(number_parameter):
    value = number_parameter.max_value + 0.1
    with pytest.raises(ValidationError):
        number_parameter.clean(value)


def test_bool_parameter_clean(bool_parameter):
    assert bool_parameter.clean(True) is True


@pytest.mark.parametrize('value', [None, 'true', 'True', [], 1])
def test_bool_parameter_clean_fails(bool_parameter, value):
    with pytest.raises(ValidationError):
        assert bool_parameter.clean(value)


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


def test_string_parameter_clean(string_parameter):
    assert string_parameter.clean('foo') == 'foo'


@pytest.mark.parametrize('value', [None, True, [], 1, 1.5])
def test_string_parameter_clean_fails(string_parameter, value):
    with pytest.raises(ValidationError):
        assert string_parameter.clean(value)
