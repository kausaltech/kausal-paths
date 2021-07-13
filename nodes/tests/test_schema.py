import pytest

pytestmark = pytest.mark.django_db

from nodes.tests.factories import NodeFactory


def test_instance_type(graphql_client_query_data, instance, instance_content):
    data = graphql_client_query_data(
        '''
        query {
          instance {
            __typename
            id
            name
            targetYear
            referenceYear
            minimumHistoricalYear
            maximumHistoricalYear
            leadTitle
            leadParagraph
          }
        }
        '''
    )
    expected = {
        'instance': {
            '__typename': 'InstanceType',
            'id': instance.id,
            'name': instance.name,
            'targetYear': instance.context.target_year,
            'referenceYear': instance.reference_year,
            'minimumHistoricalYear': instance.minimum_historical_year,
            'maximumHistoricalYear': instance.maximum_historical_year,
            'leadTitle': instance_content.lead_title,
            'leadParagraph': instance_content.lead_paragraph,
        }
    }
    assert data == expected


def test_node_type(graphql_client_query_data, additive_action, context):
    input_node = NodeFactory()
    additive_action.add_input_node(input_node)
    output_node = NodeFactory()
    additive_action.add_output_node(output_node)
    data = graphql_client_query_data(
        '''
        query($id: ID!) {
          node(id: $id) {
            __typename
            id
            name
            color
            unit {
              __typename
            }
            quantity
            targetYearGoal
            isAction
            decisionLevel
            inputNodes {
              __typename
              id
            }
            outputNodes {
              __typename
              id
            }
            descendantNodes {
              __typename
              id
            }
            upstreamActions {
              __typename
              id
            }
            metric {
              __typename
              id
            }
            outputMetrics {
              __typename
              id
            }
            impactMetric {
              __typename
              id
            }
            description
            parameters {
              __typename
              id
            }
            shortDescription
            body
          }
        }
        ''',
        variables={'id': additive_action.id}
    )
    expected = {
        'node': {
            '__typename': 'NodeType',
            'id': additive_action.id,
            'name': str(additive_action.name),
            'color': additive_action.color,
            'unit': {
                '__typename': 'UnitType'
            },
            'quantity': additive_action.quantity,
            'targetYearGoal': additive_action.target_year_goal,
            'isAction': True,
            'decisionLevel': additive_action.decision_level.name,
            'inputNodes': [{
                '__typename': 'NodeType',
                'id': input_node.id,
            }],
            'outputNodes': [{
                '__typename': 'NodeType',
                'id': output_node.id,
            }],
            'descendantNodes': [{
                '__typename': 'NodeType',
                'id': additive_action.id,
            }, {
                '__typename': 'NodeType',
                'id': output_node.id,
            }],
            'upstreamActions': [{
                '__typename': 'NodeType',
                'id': additive_action.id,
            }],
            'metric': {
                '__typename': 'ForecastMetricType',
                'id': additive_action.id,
            },
            'outputMetrics': None,  # TODO: Not implemented in schema yet
            'impactMetric': {
                '__typename': 'ForecastMetricType',
                'id': f'{additive_action.id}-{additive_action.id}-impact',
            },
            'description': f'<p>{additive_action.description}</p>',  # TODO: Does it make sense to add the p tag?
            'parameters': [{
                '__typename': 'BoolParameterType',
                'id': additive_action.enabled_param.global_id,
            }],
            'shortDescription': None,  # TODO: Not implemented in schema yet
            'body': None,  # TODO: Not set anywhere yet
        }
    }
    assert data == expected


def test_scenario_type(graphql_client_query_data, context, scenario):
    is_active = scenario == context.active_scenario
    data = graphql_client_query_data(
        '''
        query($id: ID!) {
          scenario(id: $id) {
            __typename
            id
            name
            isActive
            isDefault
          }
        }
        ''',
        variables={'id': scenario.id}
    )
    expected = {
        'scenario': {
            '__typename': 'ScenarioType',
            'id': scenario.id,
            'name': str(scenario.name),
            'isActive': is_active,
            'isDefault': scenario.default,
        }
    }
    assert data == expected
