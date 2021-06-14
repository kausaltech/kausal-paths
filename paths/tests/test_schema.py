import pytest

from nodes.tests.factories import ActionNodeFactory, ScenarioFactory

pytestmark = pytest.mark.django_db


def test_action_enabled(graphql_client_query_data, context, action_node):
    param_id = f'{action_node.id}.enabled'
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


def test_set_parameter_disable_action(graphql_client_query_data, context, action_node, custom_scenario):
    param_id = f'{action_node.id}.enabled'
    data = graphql_client_query_data(
        '''
        mutation($param: ID!) {
          setParameter(id: $param, boolValue: false) {
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
        variables={'param': param_id}
    )
    expected = {
        'setParameter': {
            'ok': True,
            'parameter': {
                'id': param_id,
                'value': False,
            }
        }
    }
    # TODO: Don't trust the response but check that the action is really disabled
    assert data == expected
