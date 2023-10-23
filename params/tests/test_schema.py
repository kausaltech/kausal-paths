import pytest
from nodes.context import Context
from nodes.scenario import Scenario

from params.tests.factories import BoolParameterFactory, NumberParameterFactory, StringParameterFactory
from nodes.tests.factories import NodeFactory

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize('is_global', [True, False])
def test_parameter_interface(graphql_client_query_data, context: Context, is_global):
    param = NumberParameterFactory.create(context=context)
    if is_global:
        context.add_global_parameter(param)
    else:
        node = NodeFactory.create(context=context)
        node.add_parameter(param)
    data = graphql_client_query_data(
        '''
        query($param: ID!) {
          parameter(id: $param) {
            __typename
            id
            label
            description
            nodeRelativeId
            node {
              __typename
            }
            isCustomized
            isCustomizable
          }
        }
        ''',
        variables={'param': param.global_id}
    )
    if is_global:
        expected_node = None
    else:
        expected_node = {'__typename': 'Node'}

    expected = {
        'parameter': {
            '__typename': 'NumberParameterType',
            'id': param.global_id,
            'label': str(param.label),
            'description': str(param.description),
            'nodeRelativeId': param.local_id,
            'node': expected_node,
            'isCustomized': param.is_customized,
            'isCustomizable': param.is_customizable,
        }
    }
    assert data == expected


@pytest.mark.parametrize('default_value', [True, False])
def test_bool_parameter_type(graphql_client_query_data, context: Context, default_value, default_scenario: Scenario):
    param = BoolParameterFactory.create(context=context)
    context.add_global_parameter(param)
    default_scenario.add_parameter(param, default_value)
    data = graphql_client_query_data(
        '''
        query($param: ID!) {
          parameter(id: $param) {
            __typename
            id
            ... on BoolParameterType {
              value
              defaultValue
            }
          }
        }
        ''',
        variables={'param': param.global_id}
    )
    expected = {
        'parameter': {
            '__typename': 'BoolParameterType',
            'id': param.global_id,
            'value': param.value,
            'defaultValue': default_value,
        }
    }
    assert data == expected


def test_number_parameter_type(graphql_client_query_data, context, default_scenario):
    default_value = 42.42
    param = NumberParameterFactory(context=context)
    context.add_global_parameter(param)
    default_scenario.add_parameter(param, default_value)
    data = graphql_client_query_data(
        '''
        query($param: ID!) {
          parameter(id: $param) {
            __typename
            id
            ... on NumberParameterType {
              value
              defaultValue
              minValue
              maxValue
              step
              unit {
                __typename
              }
            }
          }
        }
        ''',
        variables={'param': param.global_id}
    )
    expected = {
        'parameter': {
            '__typename': 'NumberParameterType',
            'id': param.global_id,
            'value': param.value,
            'defaultValue': default_value,
            'minValue': param.min_value,
            'maxValue': param.max_value,
            'step': param.step,
            'unit': {
                '__typename': 'UnitType',
            },
        }
    }
    assert data == expected


def test_string_parameter_type(graphql_client_query_data, context, default_scenario):
    default_value = 'foobar'
    param = StringParameterFactory()
    context.add_global_parameter(param)
    default_scenario.add_parameter(param, default_value)
    data = graphql_client_query_data(
        '''
        query($param: ID!) {
          parameter(id: $param) {
            __typename
            id
            ... on StringParameterType {
              value
              defaultValue
            }
          }
        }
        ''',
        variables={'param': param.global_id}
    )
    expected = {
        'parameter': {
            '__typename': 'StringParameterType',
            'id': param.global_id,
            'value': param.value,
            'defaultValue': default_value,
        }
    }
    assert data == expected


def test_set_parameter_bool(graphql_client_query_data, bool_parameter, context, custom_scenario):
    context.add_global_parameter(bool_parameter)
    param_id = bool_parameter.global_id
    value = True
    data = graphql_client_query_data(
        '''
        mutation($param: ID!, $value: Boolean!) {
          setParameter(id: $param, boolValue: $value) {
            ok
            parameter {
              id
              ... on BoolParameterType {
                value
              }
            }
          }
        }
        ''',
        variables={'param': param_id, 'value': value}
    )
    expected = {
        'setParameter': {
            'ok': True,
            'parameter': {
                'id': param_id,
                'value': value,
            }
        }
    }
    assert data == expected


def test_set_parameter_number(graphql_client_query_data, number_parameter, context, custom_scenario):
    context.add_global_parameter(number_parameter)
    param_id = number_parameter.global_id
    value = 1234.5
    data = graphql_client_query_data(
        '''
        mutation($param: ID!, $value: Float!) {
          setParameter(id: $param, numberValue: $value) {
            ok
            parameter {
              id
              ... on NumberParameterType {
                value
              }
            }
          }
        }
        ''',
        variables={'param': param_id, 'value': value}
    )
    expected = {
        'setParameter': {
            'ok': True,
            'parameter': {
                'id': param_id,
                'value': value,
            }
        }
    }
    assert data == expected


def test_set_parameter_string(graphql_client_query_data, string_parameter, context, custom_scenario):
    context.add_global_parameter(string_parameter)
    param_id = string_parameter.global_id
    value = 'bar'
    data = graphql_client_query_data(
        '''
        mutation($param: ID!, $value: String!) {
          setParameter(id: $param, stringValue: $value) {
            ok
            parameter {
              id
              ... on StringParameterType {
                value
              }
            }
          }
        }
        ''',
        variables={'param': param_id, 'value': value}
    )
    expected = {
        'setParameter': {
            'ok': True,
            'parameter': {
                'id': param_id,
                'value': value,
            }
        }
    }
    assert data == expected


def test_set_parameter_activates_custom_scenario(
    graphql_client_query_data, bool_parameter, context, custom_scenario
):
    context.add_global_parameter(bool_parameter)
    assert context.active_scenario != custom_scenario
    param_id = bool_parameter.global_id
    graphql_client_query_data(
        '''
        mutation($param: ID!, $value: Boolean!) {
          setParameter(id: $param, boolValue: $value) {
            ok
          }
        }
        ''',
        variables={'param': param_id, 'value': True}
    )
    assert context.active_scenario == custom_scenario


# TODO: Check if this test make sense
def test_reset_parameter(graphql_client_query_data, string_parameter, context, default_scenario, custom_scenario):
    default_scenario_setting = 'qux'
    context.add_global_parameter(string_parameter)
    default_scenario.add_parameter(string_parameter, default_scenario_setting)
    param_id = string_parameter.global_id
    assert string_parameter.get() is None
    graphql_client_query_data(
        '''
        mutation($param: ID!) {
          resetParameter(id: $param) {
            ok
          }
        }
        ''',
        variables={'param': param_id}
    )
    context.activate_scenario(custom_scenario)
    assert string_parameter.get() == default_scenario_setting


# TODO: Check if this test make sense
def test_reset_parameter_all(graphql_client_query_data, string_parameter, context, default_scenario, custom_scenario):
    default_scenario_setting = 'qux'
    context.add_global_parameter(string_parameter)
    default_scenario.add_parameter(string_parameter, default_scenario_setting)
    assert string_parameter.get() is None
    graphql_client_query_data(
        '''
        mutation {
          resetParameter {
            ok
          }
        }
        '''
    )
    context.activate_scenario(custom_scenario)
    assert string_parameter.get() == default_scenario_setting


def test_activate_scenario(graphql_client_query_data, context, custom_scenario):
    assert context.active_scenario is not custom_scenario
    data = graphql_client_query_data(
        '''
        mutation($scenario: ID!) {
          activateScenario(id: $scenario) {
            ok
            activeScenario {
              id
            }
          }
        }
        ''',
        variables={'scenario': custom_scenario.id}
    )
    expected = {
        'activateScenario': {
            'ok': True,
            'activeScenario': {
                'id': custom_scenario.id,
            }
        }
    }
    assert data == expected
    assert context.active_scenario is custom_scenario
