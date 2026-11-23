import pytest

from paths.tests.graphql import PathsTestClient

from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

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


INSTANCE_ORGANIZATIONS_QUERY = """
    query($instance: ID) {
      instanceOrganizations(instance: $instance) { name }
    }
"""


def test_instance_organizations_uses_the_request_instance(client):
    instance = InstanceFactory.create()
    ic = InstanceConfigFactory.create(identifier=instance.id, instance=instance)
    gql_client = PathsTestClient(client)
    gql_client.set_instance(ic)

    data = gql_client.query_data(INSTANCE_ORGANIZATIONS_QUERY)

    assert data['instanceOrganizations'] == [{'name': ic.organization.name}]


def test_instance_organizations_without_an_instance_is_an_error(client):
    gql_client = PathsTestClient(client)

    gql_client.query_errors(INSTANCE_ORGANIZATIONS_QUERY, assert_error_message='No instance given')


def test_instance_organizations_with_an_unknown_instance_is_an_error(client):
    gql_client = PathsTestClient(client)

    gql_client.query_errors(
        INSTANCE_ORGANIZATIONS_QUERY, variables={'instance': 'no-such-instance'}, assert_error_message='Instance not found'
    )
