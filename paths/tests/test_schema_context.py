from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, cast

from django.core.cache import cache

import pytest

from paths.context import PathsObjectCache
from paths.schema_context import InstanceRequestResources

from frameworks.models import Framework, FrameworkConfig
from nodes.models import InstanceConfig
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

if TYPE_CHECKING:
    from collections.abc import Generator

    from paths.tests.graphql import PathsTestClient


pytestmark = pytest.mark.django_db


@pytest.fixture
def instance_gql_client(client) -> tuple[PathsTestClient, InstanceConfig]:
    from paths.tests.graphql import PathsTestClient

    instance = InstanceFactory.create()
    config = InstanceConfigFactory.create(identifier=instance.id, instance=instance)
    gql_client = PathsTestClient(client)
    gql_client.set_instance(config)
    return gql_client, config


def test_instance_scoped_query_does_not_eagerly_create_runtime(
    instance_gql_client: tuple[PathsTestClient, InstanceConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nodes.models import InstanceConfig

    gql_client, _config = instance_gql_client

    def fail_enter_instance_context(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('config-only instance scope must not create a runtime Instance')

    monkeypatch.setattr(InstanceConfig, 'enter_instance_context', fail_enter_instance_context)

    data = gql_client.query_data('{ unit(value: "kg") { short } }')

    assert data['unit']['short']


def test_instance_metadata_query_uses_neither_graph_nor_runtime(
    instance_gql_client: tuple[PathsTestClient, InstanceConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nodes.models import InstanceConfig

    gql_client, config = instance_gql_client

    def fail_enter_instance_context(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('instance metadata must not create a runtime Instance')

    monkeypatch.setattr(InstanceConfig, 'enter_instance_context', fail_enter_instance_context)

    def fail_require_instance_graph(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('instance metadata must not build an InstanceGraph')

    monkeypatch.setattr('paths.schema_context.PathsGraphQLContext.require_instance_graph', fail_require_instance_graph)

    data = gql_client.query_data('{ instance { id uuid targetYear } }')

    assert data['instance']['id'] == config.identifier


@pytest.mark.parametrize(
    ('framework_name', 'instance_name', 'is_root_instance', 'expected_title'),
    [
        (None, 'Standalone', False, 'Standalone'),
        ('CADS', 'Framework landing', True, 'Framework landing'),
        ('CADS', 'Riga', False, 'CADS: Riga'),
        ('CADS', 'CADS', False, 'CADS: CADS'),
    ],
)
def test_instance_site_title(
    instance_gql_client: tuple[PathsTestClient, InstanceConfig],
    framework_name: str | None,
    instance_name: str,
    is_root_instance: bool,
    expected_title: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gql_client, config = instance_gql_client

    def fail_enter_instance_context(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('siteTitle must not create a runtime Instance')

    monkeypatch.setattr(InstanceConfig, 'enter_instance_context', fail_enter_instance_context)
    config.name = instance_name
    config.save(update_fields=['name'])
    if framework_name is not None:
        framework = Framework.objects.create(identifier='test-framework', name=framework_name)
        FrameworkConfig.objects.create(framework=framework, instance_config=config, baseline_year=2020)
        if is_root_instance:
            framework.root_instance = config
        else:
            root_config = InstanceConfigFactory.create(name='Framework landing')
            framework.root_instance = root_config
            FrameworkConfig.objects.create(framework=framework, instance_config=root_config, baseline_year=2020)
        framework.save(update_fields=['root_instance'])

    data = gql_client.query_data('{ instance { frameworkConfig { id } siteTitle } }')

    assert data['instance']['siteTitle'] == expected_title


def test_blank_yaml_metadata_fields_fall_back_to_snapshot(
    instance_gql_client: tuple[PathsTestClient, InstanceConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nodes.defs import InstanceMetadata, InstanceModelSpec
    from nodes.instance_serialization import InstanceSnapshot

    gql_client, config = instance_gql_client
    config.config_source = 'yaml'
    config.owner = ''
    config.lead_title = ''
    config.lead_paragraph = ''
    config.save(update_fields=['config_source', 'owner', 'lead_title', 'lead_paragraph'])
    snapshot = InstanceSnapshot(
        metadata=InstanceMetadata(
            uuid=config.uuid,
            identifier=config.identifier,
            owner='YAML owner',
            lead_title='YAML lead',
            lead_paragraph='YAML paragraph',
        ),
        spec=InstanceModelSpec(),
    )
    monkeypatch.setattr(
        'paths.schema_context.PathsGraphQLContext.require_instance_snapshot',
        lambda *_args, **_kwargs: snapshot,
    )

    data = gql_client.query_data('{ instance { owner leadTitle leadParagraph } }')

    assert data['instance'] == {
        'owner': 'YAML owner',
        'leadTitle': 'YAML lead',
        'leadParagraph': 'YAML paragraph',
    }


def test_persisted_yaml_metadata_fields_do_not_load_snapshot(
    instance_gql_client: tuple[PathsTestClient, InstanceConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gql_client, config = instance_gql_client
    config.config_source = 'yaml'
    config.owner = 'Database owner'
    config.lead_title = 'Database lead'
    config.lead_paragraph = 'Database paragraph'
    config.save(update_fields=['config_source', 'owner', 'lead_title', 'lead_paragraph'])

    def fail_require_snapshot(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('persisted YAML metadata must take precedence over the YAML snapshot')

    monkeypatch.setattr(
        'paths.schema_context.PathsGraphQLContext.require_instance_snapshot',
        fail_require_snapshot,
    )

    data = gql_client.query_data('{ instance { owner leadTitle leadParagraph } }')

    assert data['instance'] == {
        'owner': 'Database owner',
        'leadTitle': 'Database lead',
        'leadParagraph': 'Database paragraph',
    }


def test_blank_database_metadata_fields_do_not_build_snapshot(
    instance_gql_client: tuple[PathsTestClient, InstanceConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gql_client, config = instance_gql_client
    config.config_source = 'database'
    config.owner = ''
    config.lead_title = ''
    config.lead_paragraph = ''
    config.save(update_fields=['config_source', 'owner', 'lead_title', 'lead_paragraph'])

    def fail_require_snapshot(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('blank DB draft metadata must not build a snapshot')

    monkeypatch.setattr(
        'paths.schema_context.PathsGraphQLContext.require_instance_snapshot',
        fail_require_snapshot,
    )

    data = gql_client.query_data('{ instance { owner leadTitle leadParagraph } }')

    assert data['instance'] == {'owner': None, 'leadTitle': '', 'leadParagraph': None}


def test_runtime_instance_is_created_once_per_request(
    instance_gql_client: tuple[PathsTestClient, InstanceConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nodes.models import InstanceConfig

    gql_client, _config = instance_gql_client
    original = InstanceConfig.enter_instance_context
    enter_count = 0

    @contextmanager
    def counted_enter_instance_context(self: InstanceConfig, *args: Any, **kwargs: Any) -> Generator[Any]:
        nonlocal enter_count
        enter_count += 1
        with original(self, *args, **kwargs) as instance:
            yield instance

    monkeypatch.setattr(InstanceConfig, 'enter_instance_context', counted_enter_instance_context)

    data = gql_client.query_data('{ instance { id } nodes { id } }')

    assert data['instance']['id']
    assert data['nodes'] == []
    assert enter_count == 1


def test_published_snapshot_is_reused_when_graph_is_built(
    instance_gql_client: tuple[PathsTestClient, InstanceConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nodes import instance_graph_cache
    from nodes.models import PreferredInstanceSource

    _gql_client, config = instance_gql_client
    config.config_source = 'database'
    config.save(update_fields=['config_source'])
    config.publish_instance()
    config.refresh_from_db()
    source = instance_graph_cache.resolve_instance_source(config, PreferredInstanceSource.PUBLISHED)
    cache.delete(source.cache_key)

    original_loader = instance_graph_cache.load_instance_snapshot
    load_count = 0

    def counted_loader(*args: Any, **kwargs: Any):
        nonlocal load_count
        load_count += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(instance_graph_cache, 'load_instance_snapshot', counted_loader)
    resources = InstanceRequestResources(
        default_config=config,
        default_source=PreferredInstanceSource.PUBLISHED,
        default_tolerate_node_failures=False,
        stack=ExitStack(),
        extension=cast('Any', None),
        object_cache=PathsObjectCache(),
    )

    snapshot = resources.snapshot_for_instance_type()
    graph = resources.require_graph()

    assert snapshot is not None
    assert graph.metadata == snapshot.metadata
    assert load_count == 1
