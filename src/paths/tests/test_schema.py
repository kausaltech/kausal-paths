import pytest

pytestmark = pytest.mark.django_db


def test_action_enabled(graphql_client_query_data, default_scenario, action_node):
    action_node.on_scenario_created(default_scenario)
    param_id = action_node.enabled_param.global_id
    data = graphql_client_query_data(
        """
        query($param: ID!) {
          parameter(id: $param) {
            __typename
            id
            ... on BoolParameterType {
              value
            }
          }
        }
        """,
        variables={'param': param_id},
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
        """
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
        """,
        variables={'param': param_id, 'value': enabled},
    )
    expected = {
        'setParameter': {
            'ok': True,
            'parameter': {
                'id': param_id,
                'value': enabled,
            },
        }
    }
    assert data == expected
    assert action_node.is_enabled() == enabled
