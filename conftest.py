import json
import pytest
from django.conf import settings
from graphene_django.utils.testing import graphql_query
from pytest_factoryboy import register

from nodes.tests.factories import (
    AdditiveActionFactory, ActionNodeFactory, ContextFactory, CustomScenarioFactory, InstanceConfigFactory,
    InstanceFactory, NodeFactory, ScenarioFactory, SimpleNodeFactory
)
from pages.tests.factories import InstanceContentFactory
from params.tests.factories import (
    BoolParameterFactory, ParameterFactory, NumberParameterFactory, StringParameterFactory
)

register(BoolParameterFactory)
register(ContextFactory)
register(InstanceConfigFactory)
register(NumberParameterFactory)
register(ParameterFactory)
register(StringParameterFactory)


@pytest.fixture
def node(context):
    node = NodeFactory()
    context.add_node(node)
    return node


@pytest.fixture
def action_node(context):
    node = ActionNodeFactory()
    context.add_node(node)
    return node


@pytest.fixture
def additive_action(context):
    node = AdditiveActionFactory()
    context.add_node(node)
    return node


@pytest.fixture
def scenario(context):
    """Does not notify any nodes of the scenario's creation."""
    scenario = ScenarioFactory()
    context.add_scenario(scenario)
    return scenario


@pytest.fixture
def simple_node(context):
    node = SimpleNodeFactory()
    context.add_node(node)
    return node


@pytest.fixture(autouse=True)  # autouse=True since InstanceMiddleware requires a default scenario
def default_scenario(context):
    """Adds default scenario but doesn't notify any nodes of its creation."""
    scenario = ScenarioFactory(id='default', default=True, all_actions_enabled=True)
    context.add_scenario(scenario)
    context.activate_scenario(scenario)
    return scenario


@pytest.fixture
def baseline_scenario(context):
    """Adds baseline scenario but doesn't notify any nodes of its creation."""
    scenario = ScenarioFactory(id='baseline', all_actions_enabled=True)
    context.add_scenario(scenario)
    return scenario


@pytest.fixture
def custom_scenario(context, default_scenario):
    """Adds custom scenario but doesn't notify any nodes of its creation."""
    custom_scenario = CustomScenarioFactory(
        id='custom',
        name='Custom',
        base_scenario=default_scenario,
    )
    context.set_custom_scenario(custom_scenario)
    return custom_scenario


@pytest.fixture(autouse=True)
def instance(context):
    instance = InstanceFactory(context=context)
    from pages import global_instance
    global_instance.instance = instance
    import nodes
    nodes.models.instance_cache = {instance.id: instance}
    return instance


@pytest.fixture
def instance_content(db, instance):
    return InstanceContentFactory(identifier=instance.id)


@pytest.fixture
def graphql_client_query(client, instance_config):
    def func(*args, **kwargs):
        headers = {
            settings.INSTANCE_IDENTIFIER_HEADER: instance_config.identifier,
        }
        return graphql_query(*args, **kwargs, client=client, graphql_url='/v1/graphql/', headers=headers)
    return func


@pytest.fixture
def graphql_client_query_data(graphql_client_query):
    """Make a GraphQL request, make sure the `error` field is not present and return the `data` field."""
    def func(*args, **kwargs):
        response = graphql_client_query(*args, **kwargs)
        content = json.loads(response.content)
        assert 'errors' not in content
        return content['data']
    return func
