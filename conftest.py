from __future__ import annotations

import json
from typing import TYPE_CHECKING

from graphene_django.utils.testing import graphql_query

import pytest
from pytest_factoryboy import LazyFixture, register

from nodes.scenario import ScenarioKind
from nodes.tests.factories import (
    ActionNodeFactory,
    AdditiveActionFactory,
    ContextFactory,
    CustomScenarioFactory,
    InstanceConfigFactory,
    InstanceFactory,
    NodeFactory,
    ScenarioFactory,
    SimpleNodeFactory,
)
from orgs.tests.factories import OrganizationFactory
from params.tests.factories import BoolParameterFactory, NumberParameterFactory, ParameterFactory, StringParameterFactory
from people.tests.factories import PersonFactory
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.instance import Instance


@pytest.fixture(autouse=True)
def instance():
    instance = InstanceFactory()
    return instance


@pytest.fixture(autouse=True)
def context(instance):
    return instance.context


register(BoolParameterFactory)
register(ContextFactory)
register(InstanceConfigFactory)
register(NumberParameterFactory)
register(ParameterFactory)
register(StringParameterFactory)
register(UserFactory)
register(InstanceFactory)
register(PersonFactory, user=LazyFixture(lambda user: user))
register(OrganizationFactory)


@pytest.fixture
def node(context):
    node = NodeFactory(context=context)
    return node


@pytest.fixture
def action_node(context):
    assert context.instance is not None
    node = ActionNodeFactory(context=context)
    return node


@pytest.fixture
def additive_action(context: Context):
    assert context.instance is not None
    node = AdditiveActionFactory.create(context=context)
    return node


@pytest.fixture
def scenario(context: Context):
    """
    Create new scenario and add it to the context.

    Does not notify any nodes of the scenario's creation.
    """
    scenario = ScenarioFactory.create(context=context)
    context.add_scenario(scenario)
    return scenario


@pytest.fixture
def simple_node(context):
    node = SimpleNodeFactory.create(context=context)
    return node


@pytest.fixture(autouse=True)  # autouse=True since InstanceMiddleware requires a default scenario
def default_scenario(instance: Instance, context):
    """Add default scenario but doesn't notify any nodes of its creation."""
    assert context == instance.context
    return context.get_default_scenario()


@pytest.fixture
def baseline_scenario(instance: Instance):
    """Add baseline scenario but doesn't notify any nodes of its creation."""
    context = instance.context
    scenario = ScenarioFactory.create(id='baseline', all_actions_enabled=True, kind=ScenarioKind.BASELINE)
    context.add_scenario(scenario)
    return scenario


@pytest.fixture
def custom_scenario(instance: Instance):
    """Add custom scenario but doesn't notify any nodes of its creation."""
    context = instance.context
    custom_scenario = CustomScenarioFactory.create(
        id='custom',
        name='Custom',
        base_scenario=context.get_default_scenario(),
    )
    context.set_custom_scenario(custom_scenario)
    return custom_scenario


@pytest.fixture(autouse=True)
def instance_config(instance: Instance):
    return InstanceConfigFactory(identifier=instance.id, instance=instance)


@pytest.fixture
def graphql_client_query(client, instance_config, settings):
    def func(*args, **kwargs):
        # In tests, only headers that start with `HTTP_` are used, but in production the header names are taken verbatim
        assert not settings.INSTANCE_IDENTIFIER_HEADER.startswith('HTTP_')
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


@pytest.fixture
def graphql_test_client(client, instance_config):
    """Strawberry-based GraphQL test client with the autouse instance pre-configured."""
    from paths.tests.graphql import PathsTestClient

    gql_client = PathsTestClient(client)
    gql_client.set_instance(instance_config)
    return gql_client


@pytest.fixture
def admin_user():
    return UserFactory.create(is_staff=True, is_superuser=True)
