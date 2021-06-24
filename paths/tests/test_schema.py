import pytest

from params.tests.factories import NumberParameterFactory

pytestmark = pytest.mark.django_db


def test_parameter_interface(graphql_client_query_data, context, action_node):
    param = NumberParameterFactory()
    action_node.register_param(param)
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
        variables={'param': param.id}
    )
    expected = {
        'parameter': {
            '__typename': 'NumberParameterType',
            'id': param.id,
            'label': str(param.label),
            'description': str(param.description),
            'nodeRelativeId': param.node_relative_id,
            'node': {
                '__typename': 'NodeType'
            },
            'isCustomized': param.is_customized,
            'isCustomizable': param.is_customizable,
        }
    }
    assert data == expected


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
