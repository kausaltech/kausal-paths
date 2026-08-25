from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphene_django.utils.testing import graphql_query

import pytest
from pytest_factoryboy import LazyFixture, register

from kausal_common.i18n.pydantic import set_i18n_context

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
from params.tests.factories import BoolParameterFactory, NumberParameterFactory, StringParameterFactory
from people.tests.factories import PersonFactory
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.instance import Instance


# We use a fallback context for test ergonomics
_pytest_default_language_ctx = set_i18n_context('en', [])
_pytest_default_language_ctx.__enter__()


_KAUSAL_COMMON_DIR = Path(__file__).parent / 'kausal_common'


def _is_shared_test(request: pytest.FixtureRequest) -> bool:
    """
    Say whether this is a test of the shared submodule rather than of Paths.

    The autouse fixtures below build a Paths `Instance` and an `InstanceConfig` row, which is the
    right default for a test in this repo and wrong for one in `kausal_common/`: that code is
    shared with Watch, knows nothing about instances, and its tests should not need a database
    because of a conftest belonging to one of its two consumers.

    It was not a theoretical problem. `kausal_common/tests/test_storage.py` is a pure unit test of
    the S3 media backend with no `django_db` mark, so the autouse `instance_config` fixture failed
    it with *"Database access not allowed"* -- a failure whose cause is in this file and whose
    symptom is in the submodule. The other 90 shared tests only escape it because they happen to
    ask for a database anyway.

    Only `instance_config` needs the guard: it is the one that writes a row. `instance` and
    `context` build in memory, and `instance` is in any case shadowed by the `register(...)`
    call below, which defines a fixture of the same name.

    No test under `kausal_common/` requests `instance_config`, so skipping it there costs nothing.
    """
    return _KAUSAL_COMMON_DIR in Path(request.path).parents


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
def instance_config(request, instance: Instance):
    # The one autouse fixture that writes to the database, and therefore the one that has to stay
    # out of the shared submodule's way. See `_is_shared_test`.
    if _is_shared_test(request):
        return None
    return InstanceConfigFactory(identifier=instance.id, instance=instance)


@pytest.fixture
def graphql_client_query(client, instance_config, settings):
    def func(*args, **kwargs) -> Any:
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

    def func(*args, **kwargs) -> Any:
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
