from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction

import pytest

from kausal_common.datasets.models import DatasetSchemaScope
from kausal_common.datasets.tests.factories import DatasetFactory, DatasetSchemaFactory

from frameworks.models import FrameworkConfig
from frameworks.tests.factories import FrameworkFactory
from nodes.defs.port_def import InputPortDef
from nodes.instance_serialization import InputBindingSnapshot, InstanceSnapshot, NodePortSource, build_instance_snapshot
from nodes.template_graph import publish_template_instance, replace_input_port_bindings
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from nodes.units import unit_registry
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from django.test import Client
    from wagtail.models import Revision

    from frameworks.models import Framework
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


@dataclass
class TemplateEdition:
    framework: Framework
    revision: Revision

    @property
    def snapshot(self) -> InstanceSnapshot:
        from nodes.instance_serialization import InstanceSnapshot

        return InstanceSnapshot.from_serialized_data(self.revision.content['model_snapshot']['structured'])


def publish_edition(framework: Framework) -> TemplateEdition:
    assert framework.template_instance is not None
    return TemplateEdition(framework, publish_template_instance(framework.template_instance))


@pytest.fixture
def release() -> TemplateEdition:
    from nodes.defs.instance_defs import YearsSpec

    template = InstanceConfigFactory.create(name='Method', config_source='database', owner='Test owner')
    assert template.spec is not None
    template.spec.years = YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030)
    template.save(update_fields=['spec'])
    node = NodeConfigFactory.create(instance=template, identifier='shared')
    node.refresh_from_db()
    assert node.spec is not None
    node.spec.input_ports = [
        InputPortDef(id=uuid4(), identifier='local', unit=unit_registry.parse_units('kt/a'), binding_owner='instance'),
        InputPortDef(id=uuid4(), identifier='fixed', unit=unit_registry.parse_units('kt/a')),
    ]
    node.save(update_fields=['spec'])
    framework = FrameworkFactory.create(template_instance=template)
    return publish_edition(framework)


@pytest.fixture
def dependent_instance(release: TemplateEdition) -> InstanceConfig:
    instance = InstanceConfigFactory.create(name='Dependent instance', config_source='database', owner='Test owner')
    FrameworkConfig.objects.create(framework=release.framework, instance_config=instance)
    assert instance.spec is not None
    instance.spec.years = release.snapshot.spec.years
    instance.template_revision = release.revision
    instance.save(update_fields=['template_revision', 'spec'])
    instance.refresh_from_db()
    return instance


def test_release_is_immutable_and_nodes_are_copied_per_instance(
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
) -> None:
    snapshot = build_instance_snapshot(dependent_instance)
    shared = snapshot.nodes[0]
    assert shared.template_revision_id == release.revision.pk
    assert shared.spec is not None
    shared.spec.input_ports.clear()
    fresh_spec = build_instance_snapshot(dependent_instance).nodes[0].spec
    assert fresh_spec is not None
    assert len(fresh_spec.input_ports) == 2
    template = release.framework.template_instance
    assert template is not None
    template.nodes.update(name='Changed after release')
    assert str(build_instance_snapshot(dependent_instance).nodes[0].name) == str(release.snapshot.nodes[0].name)


def test_local_edge_can_feed_shared_input_and_restore_default(
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
) -> None:
    local = NodeConfigFactory.create(instance=dependent_instance, identifier='local')
    shared = release.snapshot.nodes[0]
    assert shared.spec is not None
    local.refresh_from_db()
    assert local.spec is not None
    port = shared.spec.input_ports[0]
    binding = InputBindingSnapshot(
        uuid=uuid4(),
        node_id=shared.uuid,
        port_id=port.id,
        position=0,
        source=NodePortSource(node_id=local.uuid, port_id=local.spec.output_ports[0].id),
    )
    replace_input_port_bindings(dependent_instance, shared.uuid, port.id, [binding])
    assert build_instance_snapshot(dependent_instance).bindings == [binding]
    replace_input_port_bindings(dependent_instance, shared.uuid, port.id, None)
    assert not build_instance_snapshot(dependent_instance).bindings


def test_fixed_port_and_cycles_are_rejected_atomically(
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
) -> None:
    shared = release.snapshot.nodes[0]
    assert shared.spec is not None
    binding = InputBindingSnapshot(
        uuid=uuid4(),
        node_id=shared.uuid,
        port_id=shared.spec.input_ports[1].id,
        position=0,
        source=NodePortSource(node_id=shared.uuid, port_id=shared.spec.output_ports[0].id),
    )
    with pytest.raises(ValidationError, match='Template controls'):
        replace_input_port_bindings(dependent_instance, shared.uuid, binding.port_id, [binding])
    binding.port_id = shared.spec.input_ports[0].id
    with pytest.raises(ValidationError, match='cycle'):
        replace_input_port_bindings(dependent_instance, shared.uuid, binding.port_id, [binding])
    assert not dependent_instance.binding_overrides.exists()


@pytest.mark.parametrize('schema_is_editable', [False, True])
def test_shared_schema_allows_one_private_dataset_per_scope(
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
    schema_is_editable: bool,
) -> None:
    from nodes.roles import instance_admin_role

    schema = DatasetSchemaFactory.create(is_editable=schema_is_editable)
    DatasetSchemaScope.objects.create(schema=schema, scope=release.framework)
    other = InstanceConfigFactory.create(name='Other dependent_instance', config_source='database')
    FrameworkConfig.objects.create(framework=release.framework, instance_config=other)
    ct = ContentType.objects.get_for_model(dependent_instance)
    own = DatasetFactory.create(schema=schema, scope_content_type=ct, scope_id=dependent_instance.pk)
    with pytest.raises(IntegrityError, match='unique_dataset_per_scope_per_schema'), transaction.atomic():
        DatasetFactory.create(schema=schema, scope_content_type=ct, scope_id=dependent_instance.pk)
    private = DatasetFactory.create(schema=schema, scope_content_type=ct, scope_id=other.pk)
    user = UserFactory.create()
    instance_admin_role.assign_user(dependent_instance, user)
    policy = own.permission_policy()
    assert policy.user_has_perm(user, 'change', own)
    assert not policy.user_has_perm(user, 'view', private)
    assert schema.permission_policy().user_has_perm(user, 'view', schema)
    assert not schema.permission_policy().user_has_perm(user, 'change', schema)


def test_schema_adoption_retains_local_schema_when_shared_schema_is_occupied(
    release: TemplateEdition, dependent_instance: InstanceConfig
) -> None:
    from frameworks.conversion import _adopt_schemas

    template = release.framework.template_instance
    assert template is not None
    shared_schema = DatasetSchemaFactory.create()
    DatasetFactory.create(schema=shared_schema, scope=template, identifier='shared')
    DatasetFactory.create(schema=shared_schema, scope=dependent_instance, identifier='existing')
    local_schema = DatasetSchemaFactory.create()
    alternative = DatasetFactory.create(schema=local_schema, scope=dependent_instance, identifier='alternative')

    _adopt_schemas(dependent_instance, release.framework, {})

    alternative.refresh_from_db()
    assert alternative.schema_id == local_schema.pk


def test_publication_retains_framework_data_revision(
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
) -> None:
    from datetime import date

    from kausal_common.datasets.tests.factories import DataPointFactory, DatasetMetricFactory

    from nodes.dataset_materialization import dataset_change
    from nodes.models import InstanceRevisionDatasetPin, NodeInputPortBinding

    template = release.framework.template_instance
    assert template is not None
    node = template.nodes.get()
    assert node.spec is not None
    dataset = DatasetFactory.create(
        identifier='reference',
        scope_content_type=ContentType.objects.get_for_model(template),
        scope_id=template.pk,
    )
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='Value', unit='kt/a')
    point = DataPointFactory.create(dataset=dataset, metric=metric, date=date(2020, 1, 1), value=42)
    NodeInputPortBinding.objects.create(
        instance=template,
        node=node,
        port_id=node.spec.input_ports[1].id,
        position=0,
        dataset=dataset,
        metric=metric,
    )
    template.invalidate_cache()
    edition = publish_edition(release.framework)
    dependent_instance.refresh_from_db()
    pin = edition.snapshot.dataset_revisions[0]
    with dataset_change(dataset):
        point.value = 99
        point.save()
    assert build_instance_snapshot(dependent_instance).dataset_revisions == [pin]
    dependent_instance.publish_instance()
    published_pin = InstanceRevisionDatasetPin.objects.get(instance_revision_id=dependent_instance.live_revision_id)
    assert published_pin.dataset_revision_id == pin.revision_id
    assert published_pin.dataset_revision.content['data']['data'][0]['Value'] == 42


def test_shared_output_can_feed_local_node_and_retains_it(
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
) -> None:
    from django.db.models import ProtectedError

    local = NodeConfigFactory.create(instance=dependent_instance, identifier='local')
    local.refresh_from_db()
    assert local.spec is not None
    port = InputPortDef(id=uuid4(), unit=unit_registry.parse_units('kt/a'))
    local.spec.input_ports = [port]
    local.save(update_fields=['spec'])
    shared = release.snapshot.nodes[0]
    assert shared.spec is not None
    binding = InputBindingSnapshot(
        uuid=uuid4(),
        node_id=local.uuid,
        port_id=port.id,
        position=0,
        source=NodePortSource(node_id=shared.uuid, port_id=shared.spec.output_ports[0].id),
    )
    replace_input_port_bindings(dependent_instance, local.uuid, port.id, [binding])
    with pytest.raises(ProtectedError):
        local.delete()
    replace_input_port_bindings(dependent_instance, local.uuid, port.id, None)
    local.delete()


def test_incompatible_edge_rolls_back(release: TemplateEdition, dependent_instance: InstanceConfig) -> None:
    from nodes.constraints.validation import InstanceConstraintError

    local = NodeConfigFactory.create(instance=dependent_instance, identifier='energy')
    local.refresh_from_db()
    assert local.spec is not None
    local.spec.output_ports[0].unit = unit_registry.parse_units('kWh/a')
    local.spec.output_ports[0].quantity = 'energy'
    local.save(update_fields=['spec'])
    shared = release.snapshot.nodes[0]
    assert shared.spec is not None
    port = shared.spec.input_ports[0]
    binding = InputBindingSnapshot(
        uuid=uuid4(),
        node_id=shared.uuid,
        port_id=port.id,
        position=0,
        source=NodePortSource(node_id=local.uuid, port_id=local.spec.output_ports[0].id),
    )
    with pytest.raises(InstanceConstraintError):
        replace_input_port_bindings(dependent_instance, shared.uuid, port.id, [binding])
    assert not dependent_instance.binding_overrides.exists()


def test_graphql_reports_binding_conflicts_without_writing(
    client: Client,
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
) -> None:
    from paths.tests.graphql import PathsTestClient

    client.force_login(UserFactory.create(is_superuser=True))
    gql_client = PathsTestClient(client)
    gql_client.set_instance(dependent_instance)
    local = NodeConfigFactory.create(instance=dependent_instance, identifier='energy')
    local.refresh_from_db()
    assert local.spec is not None
    local.spec.output_ports[0].unit = unit_registry.parse_units('kWh/a')
    local.save(update_fields=['spec'])
    shared = release.snapshot.nodes[0]
    assert shared.spec is not None
    result = gql_client.query_data(
        """
        mutation Bind($instance: ID!, $node: UUID!, $port: UUID!, $bindings: [InputPortBindingInput!]) {
            instanceEditor(instanceId: $instance) {
                setInputPortBindings(nodeId: $node, portId: $port, bindings: $bindings) {
                    __typename
                    ... on ConstraintViolations { conflicts { message } }
                }
            }
        }
    """,
        variables={
            'instance': str(dependent_instance.pk),
            'node': str(shared.uuid),
            'port': str(shared.spec.input_ports[0].id),
            'bindings': [{'sourceNodeId': str(local.uuid), 'sourcePortId': str(local.spec.output_ports[0].id)}],
        },
    )
    payload = result['instanceEditor']['setInputPortBindings']
    assert payload['__typename'] == 'ConstraintViolations'
    assert payload['conflicts']
    assert not dependent_instance.binding_overrides.exists()


def test_template_edits_wait_for_publication(release: TemplateEdition, dependent_instance: InstanceConfig) -> None:
    template = release.framework.template_instance
    assert template is not None
    original = build_instance_snapshot(dependent_instance).nodes[0]
    template.nodes.update(name='Updated method')
    template.invalidate_cache()
    assert str(build_instance_snapshot(dependent_instance).nodes[0].name) == str(original.name)
    template.publish_instance()
    dependent_instance.refresh_from_db()
    assert dependent_instance.template_revision_id == template.live_revision_id
    assert dependent_instance.template_revision_id != release.revision.pk
    assert str(build_instance_snapshot(dependent_instance).nodes[0].name) == 'Updated method'


def test_incompatible_template_publication_rolls_back_all_drafts(
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
) -> None:
    other = InstanceConfigFactory.create(
        name='Second dependent_instance', config_source='database', template_revision=release.revision
    )
    FrameworkConfig.objects.create(framework=release.framework, instance_config=other)
    shared = release.snapshot.nodes[0]
    assert shared.spec is not None
    local = NodeConfigFactory.create(instance=other)
    local.refresh_from_db()
    assert local.spec is not None
    port = shared.spec.input_ports[0]
    binding = InputBindingSnapshot(
        uuid=uuid4(),
        node_id=shared.uuid,
        port_id=port.id,
        position=0,
        source=NodePortSource(node_id=local.uuid, port_id=local.spec.output_ports[0].id),
    )
    replace_input_port_bindings(other, shared.uuid, port.id, [binding])
    template = release.framework.template_instance
    assert template is not None
    node = template.nodes.get()
    assert node.spec is not None
    node.spec.input_ports = []
    node.save(update_fields=['spec'])
    template.invalidate_cache()
    with pytest.raises(ValueError, match='missing port'):
        template.publish_instance()
    template.refresh_from_db()
    dependent_instance.refresh_from_db()
    other.refresh_from_db()
    assert template.live_revision_id == release.revision.pk
    assert dependent_instance.template_revision_id == release.revision.pk
    assert other.template_revision_id == release.revision.pk


def test_editor_computes_definition_and_binding_permissions(
    client: Client,
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
) -> None:
    from paths.tests.graphql import PathsTestClient

    client.force_login(UserFactory.create(is_superuser=True))
    gql_client = PathsTestClient(client)
    template = release.framework.template_instance
    assert template is not None
    template.nodes.update(is_editable=False)
    template_node = template.nodes.get()
    assert template_node.spec is not None
    for port in template_node.spec.input_ports:
        port.is_editable = False
    template_node.save(update_fields=['spec'])
    template.invalidate_cache()
    query = """
        query Ports($id: ID!) {
            modelInstance(instanceId: $id) {
                nodes {
                    identifier isEditable isInherited
                    editor { spec { inputPorts { identifier isEditable bindingsEditable } } }
                }
            }
        }
    """
    for instance, inherited in [(template, False), (dependent_instance, True)]:
        gql_client.set_instance(instance)
        data = gql_client.query_data(query, variables={'id': str(instance.pk)})
        node = data['modelInstance']['nodes'][0]
        assert node['isInherited'] == inherited
        assert node['isEditable'] == (not inherited)
        ports = {port['identifier']: port for port in node['editor']['spec']['inputPorts']}
        assert ports['local']['isEditable'] == (not inherited)
        assert ports['local']['bindingsEditable'] is True
        assert ports['fixed']['isEditable'] == (not inherited)
        assert ports['fixed']['bindingsEditable'] == (not inherited)


def test_template_publication_preserves_local_publication(
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
) -> None:
    dependent_instance.publish_instance()
    dependent_instance.refresh_from_db()
    published = dependent_instance.live_revision
    assert published is not None
    content = published.content
    template = release.framework.template_instance
    assert template is not None
    template.nodes.update(name='New template edition')
    template.invalidate_cache()
    template.publish_instance()
    dependent_instance.refresh_from_db()
    published.refresh_from_db()
    assert dependent_instance.live_revision_id == published.pk
    assert published.content == content
    assert dependent_instance.template_revision_id == template.live_revision_id
    assert str(build_instance_snapshot(dependent_instance).nodes[0].name) == 'New template edition'


def test_override_retains_metric(release: TemplateEdition, dependent_instance: InstanceConfig) -> None:
    from django.db.models import ProtectedError

    from kausal_common.datasets.tests.factories import DatasetMetricFactory

    from nodes.instance_serialization import DatasetMetricSource

    dataset = DatasetFactory.create(
        identifier='local-factor',
        scope_content_type=ContentType.objects.get_for_model(dependent_instance),
        scope_id=dependent_instance.pk,
    )
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='Value', unit='kt/a')
    node = release.snapshot.nodes[0]
    assert node.spec is not None
    port = node.spec.input_ports[0]
    replace_input_port_bindings(
        dependent_instance,
        node.uuid,
        port.id,
        [
            InputBindingSnapshot(
                uuid=uuid4(),
                node_id=node.uuid,
                port_id=port.id,
                position=0,
                source=DatasetMetricSource(
                    dataset='local-factor', dataset_uuid=dataset.uuid, metric='Value', metric_uuid=metric.uuid
                ),
            )
        ],
    )
    with pytest.raises(ProtectedError):
        metric.delete()
    replace_input_port_bindings(dependent_instance, node.uuid, port.id, None)
    metric.delete()


@pytest.mark.parametrize('edit_template', [False, True])
def test_local_binding_api_in_template_graph(
    client: Client,
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
    edit_template: bool,
) -> None:
    from kausal_common.datasets.tests.factories import DatasetMetricFactory

    from paths.tests.graphql import PathsTestClient

    from nodes.tests.test_dataset_bindings_graphql import BIND_DATASET, DELETE_BINDING, UPDATE_BINDING

    instance = release.framework.template_instance if edit_template else dependent_instance
    assert instance is not None
    client.force_login(UserFactory.create(is_superuser=True))
    gql_client = PathsTestClient(client)
    gql_client.set_instance(instance)
    local = NodeConfigFactory.create(instance=instance, identifier='local_calculation')
    assert local.spec is not None
    port = InputPortDef(id=uuid4(), unit=unit_registry.parse_units('kt/a'), multi=False)
    local.spec.input_ports = [port]
    local.save(update_fields=['spec'])
    if not edit_template:
        shared = release.snapshot.nodes[0]
        assert shared.spec is not None
        replace_input_port_bindings(
            instance,
            local.uuid,
            port.id,
            [
                InputBindingSnapshot(
                    uuid=uuid4(),
                    node_id=local.uuid,
                    port_id=port.id,
                    position=0,
                    source=NodePortSource(node_id=shared.uuid, port_id=shared.spec.output_ports[0].id),
                )
            ],
        )
    dataset = DatasetFactory.create(
        identifier='local-data',
        scope_content_type=ContentType.objects.get_for_model(instance),
        scope_id=instance.pk,
    )
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='Value', unit='kt/a')
    instance.invalidate_cache()
    result = gql_client.query_data(
        BIND_DATASET,
        variables={
            'instanceId': str(instance.pk),
            'nodeId': str(local.uuid),
            'input': {'portId': str(port.id), 'datasetId': str(dataset.uuid), 'metricId': str(metric.uuid), 'replace': True},
        },
    )['instanceEditor']['nodeEditor']['bindDataset']
    assert result['__typename'] == 'DatasetPortType'
    binding_id = result['id']
    result = gql_client.query_data(
        UPDATE_BINDING,
        variables={
            'instanceId': str(instance.pk),
            'bindingId': binding_id,
            'input': {'tags': ['local']},
        },
    )['instanceEditor']['bindingEditor']['updateDatasetBinding']
    assert result['__typename'] == 'DatasetPortType'
    assert result['tags'] == ['local']
    gql_client.query_data(DELETE_BINDING, variables={'instanceId': str(instance.pk), 'bindingId': binding_id})
    assert not [b for b in build_instance_snapshot(instance).bindings if b.node_id == local.uuid]


def test_dataset_editability_query_count_does_not_grow(
    client: Client,
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
) -> None:
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    from paths.tests.graphql import PathsTestClient

    client.force_login(UserFactory.create(is_superuser=True))
    gql_client = PathsTestClient(client)
    gql_client.set_instance(dependent_instance)
    ct = ContentType.objects.get_for_model(dependent_instance)
    query = """query Datasets($id: ID!) { modelInstance(instanceId: $id) {
        editor { datasets { id isEditable schemaIsEditable } }
    } }"""

    def count_queries() -> tuple[int, list]:
        with CaptureQueriesContext(connection) as queries:
            result = gql_client.query_data(query, variables={'id': str(dependent_instance.pk)})
        return len(queries), result['modelInstance']['editor']['datasets']

    first = DatasetFactory.create(scope_content_type=ct, scope_id=dependent_instance.pk)
    count_queries()  # Warm ContentType and other process-wide caches.
    small_count, _ = count_queries()
    for _ in range(12):
        DatasetFactory.create(scope_content_type=ct, scope_id=dependent_instance.pk)
    for _ in range(3):
        shared_schema = DatasetSchemaFactory.create(is_editable=False)
        DatasetSchemaScope.objects.create(schema=shared_schema, scope=release.framework)
        DatasetFactory.create(schema=shared_schema, scope_content_type=ct, scope_id=dependent_instance.pk)
    large_count, datasets = count_queries()
    assert large_count <= small_count
    assert len(datasets) == 16
    by_id = {item['id']: item for item in datasets}
    assert by_id[str(first.uuid)]['schemaIsEditable']
    assert all(item['isEditable'] for item in datasets)
    assert sum(not item['schemaIsEditable'] for item in datasets) == 3


@pytest.mark.parametrize('inherited_source', [False, True])
def test_local_edge_api_in_inherited_graph(
    client: Client, dependent_instance: InstanceConfig, release: TemplateEdition, inherited_source: bool
) -> None:
    from paths.tests.graphql import PathsTestClient

    from nodes.tests.test_model_editor import CREATE_EDGE, DELETE_EDGE, _edge_input

    client.force_login(UserFactory.create(is_superuser=True))
    gql_client = PathsTestClient(client)
    gql_client.set_instance(dependent_instance)
    source = NodeConfigFactory.create(instance=dependent_instance, identifier='source')
    target = NodeConfigFactory.create(instance=dependent_instance, identifier='target')
    dependent_instance.invalidate_cache()
    result = gql_client.query_data(
        CREATE_EDGE,
        variables={
            'instanceId': str(dependent_instance.pk),
            'input': _edge_input(dependent_instance, source.identifier, target.identifier)
            | ({'fromRef': {'nodeUuid': str(release.snapshot.nodes[0].uuid)}} if inherited_source else {}),
        },
    )['instanceEditor']['createEdge']
    assert result['__typename'] == 'NodeEdgeType'
    binding = next(b for b in build_instance_snapshot(dependent_instance).bindings if b.node_id == target.uuid)
    query = """mutation Edit($id: ID!, $binding: ID!) { instanceEditor(instanceId: $id) {
        bindingEditor(bindingId: $binding) { updateEdgeBinding(input: {tags: ["changed"]}) {
            ... on NodeEdgeType { tags }
        } }
    } }"""
    result = gql_client.query_data(query, variables={'id': str(dependent_instance.pk), 'binding': str(binding.uuid)})
    assert result['instanceEditor']['bindingEditor']['updateEdgeBinding']['tags'] == ['changed']
    gql_client.query_data(DELETE_EDGE, variables={'instanceId': str(dependent_instance.pk), 'edgeId': str(binding.uuid)})
    assert not [b for b in build_instance_snapshot(dependent_instance).bindings if b.node_id == target.uuid]


@pytest.mark.parametrize('local_inputs', [False, True])
def test_inherited_binding_query_count_does_not_grow(
    client: Client,
    release: TemplateEdition,
    dependent_instance: InstanceConfig,
    local_inputs: bool,
) -> None:
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    from kausal_common.datasets.tests.factories import DatasetMetricFactory

    from paths.tests.graphql import PathsTestClient

    from nodes.instance_serialization import DatasetMetricSource
    from nodes.models import InputPortBindingSet, NodeInputPortBinding

    template = release.framework.template_instance
    assert template is not None
    client.force_login(UserFactory.create(is_superuser=True))
    gql_client = PathsTestClient(client)
    gql_client.set_instance(dependent_instance)
    query = """query Nodes($id: ID!) { modelInstance(instanceId: $id) {
        nodes { isEditable editor { spec { inputPorts {
            isEditable bindingsEditable bindings { ... on DatasetPortType {
                dataset { id isEditable schemaIsEditable schema { metrics { id } dimensions { id } } }
            } }
        } } } }
    } }"""

    def add_nodes(count: int) -> None:
        overrides = []
        for _ in range(count):
            node = NodeConfigFactory.create(instance=template)
            assert node.spec is not None
            port = InputPortDef(id=uuid4(), unit=unit_registry.parse_units('kt/a'), binding_owner='instance')
            node.spec.input_ports = [port]
            node.save(update_fields=['spec'])
            dataset = DatasetFactory.create(
                identifier=f'input_{node.uuid.hex}',
                scope_content_type=ContentType.objects.get_for_model(template),
                scope_id=dependent_instance.pk if local_inputs else template.pk,
            )
            metric = DatasetMetricFactory.create(schema=dataset.schema, name='Value', unit='kt/a')
            assert dataset.identifier is not None
            assert metric.name is not None
            if local_inputs:
                overrides.append(
                    InputPortBindingSet(
                        instance=dependent_instance,
                        node_uuid=node.uuid,
                        port_uuid=port.id,
                        bindings=[
                            InputBindingSnapshot(
                                uuid=uuid4(),
                                node_id=node.uuid,
                                port_id=port.id,
                                position=0,
                                source=DatasetMetricSource(
                                    dataset=dataset.identifier,
                                    dataset_uuid=dataset.uuid,
                                    metric=metric.name,
                                    metric_uuid=metric.uuid,
                                ),
                            ),
                        ],
                    )
                )
            else:
                NodeInputPortBinding.objects.create(instance=template, node=node, port_id=port.id, dataset=dataset, metric=metric)
        template.invalidate_cache()
        template.publish_instance()
        dependent_instance.refresh_from_db()
        for override in overrides:
            override.save()
        dependent_instance.invalidate_cache()

    def count_queries() -> int:
        with CaptureQueriesContext(connection) as queries:
            gql_client.query_data(query, variables={'id': str(dependent_instance.pk)})
        return len(queries)

    add_nodes(1)
    count_queries()
    small_count = count_queries()
    add_nodes(12)
    count_queries()
    assert count_queries() <= small_count
