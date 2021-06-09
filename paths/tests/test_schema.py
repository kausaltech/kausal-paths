import pytest

pytestmark = pytest.mark.django_db


def test_action_enabled(graphql_client_query_data):
    data = graphql_client_query_data(
        '''
        query {
          parameter(id: "action.enabled") {
            __typename
            id
            ... on BoolParameterType {
              value
            }
          }
        }
        ''',
    )
    expected = {
        'parameter': {
            '__typename': 'BoolParameterType',
            'id': 'action.enabled',
            'value': True,
        }
    }
    assert data == expected


def test_set_parameter_disable_action(graphql_client_query_data):
    data = graphql_client_query_data(
        '''
        mutation {
          setParameter(id: "action.enabled", boolValue: false) {
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
    )
    expected = {
        'setParameter': {
            'ok': True,
            'parameter': {
                'id': 'action.enabled',
                'value': False,
            }
        }
    }
    # TODO: Don't trust the response but check that the action is really enabled
    assert data == expected
