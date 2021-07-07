import pytest

from params.tests.factories import BoolParameterFactory, NumberParameterFactory, StringParameterFactory

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize('is_global', [True, False])
def test_parameter_interface(graphql_client_query_data, context, action_node, is_global):
    param = NumberParameterFactory()
    if is_global:
        context.add_global_parameter(param)
    else:
        action_node.add_parameter(param)
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
        expected_node = {'__typename': 'NodeType'}
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
def test_bool_parameter_type(graphql_client_query_data, context, default_value, default_scenario):
    param = BoolParameterFactory()
    param.add_scenario_setting(default_scenario.id, default_value)
    context.add_global_parameter(param)
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
    param = NumberParameterFactory()
    param.add_scenario_setting(default_scenario.id, default_value)
    context.add_global_parameter(param)
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
    param.add_scenario_setting(default_scenario.id, default_value)
    context.add_global_parameter(param)
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


def test_action_enabled(graphql_client_query_data, action_node):
    param_id = action_node.enabled_param.global_id
    data = graphql_client_query_data(
        '''
        query($param: ID!) {
          parameter(id: $param) {
            __typename
            id
            ... on BoolParameterType {
              value
            }
          }
        }
        ''',
        variables={'param': param_id}
    )
    expected = {
        'parameter': {
            '__typename': 'BoolParameterType',
            'id': param_id,
            'value': True,
        }
    }
    assert data == expected


@pytest.mark.parametrize('enabled', [True, False])
def test_set_parameter_action_enabled(graphql_client_query_data, action_node, custom_scenario, enabled):
    param_id = action_node.enabled_param.global_id
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
        variables={'param': param_id, 'value': enabled}
    )
    expected = {
        'setParameter': {
            'ok': True,
            'parameter': {
                'id': param_id,
                'value': enabled,
            }
        }
    }
    assert data == expected
    assert action_node.is_enabled() == enabled
