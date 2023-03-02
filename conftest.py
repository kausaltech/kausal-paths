import json
import pytest
from datetime import datetime
from django.utils.timezone import make_aware, utc
from graphene_django.utils.testing import graphql_query
from pytest_factoryboy import register

from nodes.tests.factories import (
    AdditiveActionFactory, ActionNodeFactory, ContextFactory, CustomScenarioFactory, InstanceConfigFactory,
    InstanceFactory, NodeFactory, ScenarioFactory, SimpleNodeFactory
)
from params.tests.factories import (
    BoolParameterFactory, ParameterFactory, NumberParameterFactory, StringParameterFactory
)
from datasets.tests.factories import (
    DatasetFactory
)
from users.tests.factories import (
    UserFactory
)

register(BoolParameterFactory)
register(ContextFactory)
register(InstanceConfigFactory)
register(NumberParameterFactory)
register(ParameterFactory)
register(StringParameterFactory)
register(DatasetFactory)
register(UserFactory)


@pytest.fixture
def node(context):
    node = NodeFactory(context=context)
    context.add_node(node)
    return node


@pytest.fixture
def action_node(context):
    node = ActionNodeFactory(context=context)
    context.add_node(node)
    return node


@pytest.fixture
def additive_action(context):
    node = AdditiveActionFactory(context=context)
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
    node = SimpleNodeFactory(context=context)
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
def instance(context, default_scenario):
    instance = InstanceFactory(context=context)
    # Replicate some code from InstanceConfig.get_instance
    # FIXME: This is likely to break
    # Make instance newer than anything we'll likely encounter so we always use the instance_cache forced below
    instance.modified_at = make_aware(datetime(3000, 1, 1, 0, 0), utc)
    # instance.context.generate_baseline_values()  # TODO
    from pages import global_instance
    global_instance.instance = instance
    from nodes import models
    setattr(models.instance_cache, instance.id, instance)
    return instance


@pytest.fixture
def graphql_client_query(client, instance_config, settings):
    def func(*args, **kwargs):
        # In tests, only headers that start with `HTTP_` are used, but in production the header names are taken verbatim
        assert not settings.INSTANCE_IDENTIFIER_HEADER.startswith('HTTP_')
        headers = {
            'HTTP_' + settings.INSTANCE_IDENTIFIER_HEADER: instance_config.identifier,
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


@pytest.fixture
def user():
    return UserFactory()


@pytest.fixture
def admin_user():
    return UserFactory(is_staff=True, is_superuser=True)
