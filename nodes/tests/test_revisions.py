"""Tests for the draft/publish/revisions machinery."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from django.db import connection
from django.test.utils import CaptureQueriesContext

import pytest

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import DatasetPortSpec, InputDatasetDef, NodeSpec
from nodes.instance_serialization import (
    SNAPSHOT_SCHEMA_VERSION,
    DatasetPortSnapshot,
    EdgeSnapshot,
    InstanceSnapshot,
    NodeSnapshot,
    build_instance_snapshot,
)
from nodes.models import NodeEdge
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


pytestmark = pytest.mark.django_db


@pytest.fixture
def empty_db_instance() -> InstanceConfig:
    """Bare DB-sourced InstanceConfig, no nodes/edges."""
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
    )
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        owner='Test Owner',
        spec=spec,
    )


# ---------------------------------------------------------------------------
# Pydantic round-trip
# ---------------------------------------------------------------------------


def test_instance_snapshot_json_round_trip():
    """A minimal snapshot dumps to JSON and reloads to an equal object."""
    from kausal_common.i18n.pydantic import TranslatedString

    spec = InstanceModelSpec(
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
    )
    node_1_uuid = uuid.uuid4()
    node_2_uuid = uuid.uuid4()
    snap = InstanceSnapshot(
        spec=spec,
        nodes=[
            NodeSnapshot(
                uuid=node_1_uuid,
                identifier='n1',
                name=TranslatedString('Node 1', default_language='en'),
                color='#abc',
                is_visible=True,
            )
        ],
        edges=[
            EdgeSnapshot(
                from_node=node_1_uuid,
                to_node=node_2_uuid,
                from_port=uuid.UUID('33191571-e9c8-45ac-b624-cc0a04341d37'),
                to_port=uuid.UUID('796076a8-426b-4068-ac57-e3e333d0ef0a'),
            )
        ],
        dataset_ports=[
            DatasetPortSnapshot(
                node=node_1_uuid,
                dataset='ds',
                port_id=uuid.UUID('6c8b0551-7ccf-472b-94db-26f513d706dc'),
                metric='m',
                spec=DatasetPortSpec.from_input_dataset(InputDatasetDef(id='ds', forecast_from=2025)),
            )
        ],
    )
    dumped = snap.model_dump(mode='json')
    assert dumped['schema_version'] == SNAPSHOT_SCHEMA_VERSION

    reloaded = InstanceSnapshot.model_validate(dumped)
    assert reloaded.nodes[0].identifier == 'n1'
    assert reloaded.edges[0].from_node == node_1_uuid
    assert reloaded.dataset_ports[0].spec.forecast_from == 2025
    assert reloaded.schema_version == SNAPSHOT_SCHEMA_VERSION


def test_i18n_node_metadata_stays_dict_serializable():
    """Node display metadata remains losslessly JSON serializable."""
    from kausal_common.i18n.pydantic import TranslatedString

    snap = NodeSnapshot(
        uuid=uuid.uuid4(),
        identifier='n1',
        name=TranslatedString(en='Renamed'),
        short_name=TranslatedString(en='Short', fi='Lyhyt'),
        color='#abc',
        is_visible=True,
        spec=NodeSpec(),
    )

    dumped = snap.model_dump(mode='json')
    assert dumped['name'] == {'en': 'Renamed'}
    assert dumped['short_name'] == {'en': 'Short', 'fi': 'Lyhyt'}
    assert 'name' not in dumped['spec']


def test_instance_snapshot_schema_version_default():
    """New snapshots carry the current schema version."""
    spec = InstanceModelSpec(years=YearsSpec(target=2030))
    snap = InstanceSnapshot(spec=spec)
    assert snap.schema_version == SNAPSHOT_SCHEMA_VERSION


def test_instance_snapshot_upgrades_legacy_identifier_references():
    node_uuid = uuid.uuid4()
    legacy = {
        'schema_version': 2,
        'spec': InstanceModelSpec(years=YearsSpec(target=2030)).model_dump(mode='json'),
        'nodes': [
            {
                'identifier': 'n1',
                'spec': {
                    **NodeSpec().model_dump(mode='json'),
                    'uuid': str(node_uuid),
                    'identifier': 'n1',
                },
            }
        ],
        'edges': [
            {
                'from_node': 'n1',
                'to_node': 'n1',
                'from_port': str(uuid.uuid4()),
                'to_port': str(uuid.uuid4()),
            }
        ],
        'dataset_ports': [
            {
                'node': 'n1',
                'dataset': 'ds',
                'port_id': str(uuid.uuid4()),
                'metric': 'value',
            }
        ],
    }

    snapshot = InstanceSnapshot.from_serialized_data(legacy)

    assert snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION
    assert snapshot.nodes[0].uuid == node_uuid
    assert snapshot.edges[0].from_node == node_uuid
    assert snapshot.dataset_ports[0].node == node_uuid


def test_instance_snapshot_upgrades_v3_node_metadata():
    node_uuid = uuid.uuid4()
    legacy = {
        'schema_version': 3,
        'spec': InstanceModelSpec(years=YearsSpec(target=2030)).model_dump(mode='json'),
        'nodes': [
            {
                'uuid': str(node_uuid),
                'identifier': 'n1',
                'name': {'en': 'Outer name'},
                'description': {'en': 'Long CMS description'},
                'spec': {
                    **NodeSpec().model_dump(mode='json'),
                    'uuid': str(node_uuid),
                    'identifier': 'stale-id',
                    'name': {'en': 'Stale name'},
                    'short_name': {'en': 'Short'},
                    'description': {'en': 'Runtime description'},
                    'color': '#def',
                    'order': 5,
                    'is_visible': False,
                    'kind': 'simple',
                },
            }
        ],
    }

    snapshot = InstanceSnapshot.from_serialized_data(legacy)
    node = snapshot.nodes[0]

    assert node.identifier == 'n1'
    assert str(node.name) == 'Outer name'
    assert str(node.short_name) == 'Short'
    assert str(node.short_description) == 'Runtime description'
    assert str(node.description) == 'Long CMS description'
    assert node.spec is not None
    assert not ({'uuid', 'identifier', 'name', 'short_name', 'description', 'kind'} & node.spec.model_fields_set)


# ---------------------------------------------------------------------------
# build_instance_snapshot from an empty DB-sourced instance
# ---------------------------------------------------------------------------


def test_build_instance_snapshot_empty_instance(empty_db_instance: InstanceConfig):
    snapshot = build_instance_snapshot(empty_db_instance)
    assert snapshot.spec is empty_db_instance.spec
    assert snapshot.nodes == []
    assert snapshot.edges == []
    assert snapshot.dataset_ports == []
    assert snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION


def test_build_instance_snapshot_round_trip_through_json(empty_db_instance: InstanceConfig):
    """Build → dump → load → dump → load → dump is stable (second-dump idempotent)."""
    snap = build_instance_snapshot(empty_db_instance)
    # First pass may re-normalize (i18n strings are a known case),
    # so we take two passes and compare the stabilized forms.
    dump_1 = InstanceSnapshot.model_validate(snap.model_dump(mode='json')).model_dump(mode='json')
    dump_2 = InstanceSnapshot.model_validate(dump_1).model_dump(mode='json')
    assert dump_1 == dump_2


def test_node_layout_round_trips_through_instance_export(empty_db_instance: InstanceConfig):
    from nodes.instance_serialization import export_instance, import_instance
    from nodes.models import NodeLayout, NodeLayoutSource

    source_node = NodeConfigFactory.create(instance=empty_db_instance, identifier='positioned')
    NodeLayout.objects.create(
        node=source_node,
        x=12.5,
        y=-8.25,
        source=NodeLayoutSource.USER,
    )
    export = export_instance(empty_db_instance)
    target_instance = InstanceFactory.create()
    target = InstanceConfigFactory.create(
        identifier=target_instance.id,
        instance=target_instance,
        config_source='database',
    )

    import_instance(target, export)

    copied = NodeLayout.objects.select_related('node').get(node__instance=target)
    assert copied.node.identifier == 'positioned'
    assert (copied.x, copied.y, copied.source) == (12.5, -8.25, NodeLayoutSource.USER)


def test_revisioned_content_round_trips_through_instance_export(empty_db_instance: InstanceConfig):
    from nodes.instance_serialization import export_instance, import_instance

    empty_db_instance.lead_title = 'Source lead'
    empty_db_instance.lead_paragraph = '<p>Source paragraph</p>'
    empty_db_instance.save(update_fields=['lead_title', 'lead_paragraph'])
    source_node = NodeConfigFactory.create(instance=empty_db_instance, identifier='content-node')
    source_node.body = [{'type': 'paragraph', 'value': '<p>Source body</p>'}]
    source_node.save(update_fields=['body'])

    export = export_instance(empty_db_instance)
    target_instance = InstanceFactory.create()
    target = InstanceConfigFactory.create(
        identifier=target_instance.id,
        instance=target_instance,
        config_source='database',
    )

    import_instance(target, export)

    target.refresh_from_db()
    assert target.lead_title == 'Source lead'
    assert target.lead_paragraph == '<p>Source paragraph</p>'
    copied_node = target.nodes.get(identifier='content-node')
    assert 'Source body' in str(copied_node.body)


def test_build_instance_snapshot_does_not_hydrate_related_specs(empty_db_instance: InstanceConfig):
    source = NodeConfigFactory.create(instance=empty_db_instance, identifier='source')
    target = NodeConfigFactory.create(instance=empty_db_instance, identifier='target')
    NodeEdge.objects.create(
        instance=empty_db_instance,
        from_node=source,
        to_node=target,
        from_port=uuid.uuid4(),
        to_port=uuid.uuid4(),
    )

    with CaptureQueriesContext(connection) as queries:
        snapshot = build_instance_snapshot(empty_db_instance)

    assert [(edge.from_node, edge.to_node) for edge in snapshot.edges] == [(source.uuid, target.uuid)]
    node_sql = next(query['sql'] for query in queries if 'FROM "nodes_nodeconfig"' in query['sql'])
    edge_sql = next(query['sql'] for query in queries if 'FROM "nodes_nodeedge"' in query['sql'])
    port_sql = next(query['sql'] for query in queries if 'FROM "nodes_datasetport"' in query['sql'])
    assert 'JOIN "nodes_instanceconfig"' not in node_sql
    assert '"nodes_nodeconfig"."spec"' not in edge_sql
    assert '"nodes_nodeconfig"."spec"' not in port_sql


# ---------------------------------------------------------------------------
# serializable_data includes the snapshot
# ---------------------------------------------------------------------------


def test_serializable_data_includes_structured_snapshot(empty_db_instance: InstanceConfig):
    data = empty_db_instance.serializable_data()
    assert 'model_snapshot' in data
    ms = data['model_snapshot']
    assert ms['schema_version'] == SNAPSHOT_SCHEMA_VERSION
    assert 'structured' in ms
    # The serialized config dict is no longer written; restore goes through
    # the structured snapshot (hydrate_dict remains readable in old revisions).
    assert 'hydrate_dict' not in ms
    # Structured should validate as InstanceSnapshot
    reloaded = InstanceSnapshot.model_validate(ms['structured'])
    assert reloaded.schema_version == SNAPSHOT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# _create_from_config(source='published') fallback when no live revision
# ---------------------------------------------------------------------------


def test_create_from_config_published_falls_back_to_draft(empty_db_instance: InstanceConfig):
    """With no live_revision, source='published' falls back to draft (tables)."""
    # No revisions saved yet — falls back silently
    instance = empty_db_instance._create_from_config(source='published')
    assert instance is not None
    assert instance.id == empty_db_instance.identifier


# ---------------------------------------------------------------------------
# Phase 2: change_operation + record_change
# ---------------------------------------------------------------------------


def test_change_operation_creates_operation_row(empty_db_instance: InstanceConfig):
    from nodes.change_ops import change_operation
    from nodes.models import InstanceChangeOperation, InstanceChangeSource

    with change_operation(
        empty_db_instance,
        user=None,
        action='node.create',
        source=InstanceChangeSource.GRAPHQL,
    ) as op:
        assert op.instance_config_id == empty_db_instance.pk  # type: ignore[attr-defined]
        assert op.action == 'node.create'
        assert op.source == InstanceChangeSource.GRAPHQL.value
        assert op.user_id is None  # type: ignore[attr-defined]

    assert InstanceChangeOperation.objects.filter(pk=op.pk).exists()


def test_get_current_operation_raises_outside_block():
    from nodes.change_ops import NoActiveChangeOperation, get_current_operation

    with pytest.raises(NoActiveChangeOperation):
        get_current_operation()


def test_record_change_produces_log_entry(empty_db_instance: InstanceConfig):
    from nodes.change_ops import change_operation, record_change
    from nodes.models import NodeConfig

    with change_operation(empty_db_instance, user=None, action='node.create'):
        nc = NodeConfig.objects.create(
            instance=empty_db_instance,
            identifier='n1',
            name='Node 1',
        )
        entry = record_change(nc, action='node.create', before=None, after=nc.serializable_data())

    assert entry.operation_id  # type: ignore[attr-defined]
    assert entry.action == 'node.create'
    assert entry.data['before'] is None
    assert entry.data['after']['identifier'] == 'n1'
    assert entry.data['target_uuid'] == str(nc.uuid)
    # GFK fields
    assert entry.object_id == str(nc.pk)


def test_record_change_cascade_shares_operation(empty_db_instance: InstanceConfig):
    """Multiple record_change calls inside one block share one operation."""
    from nodes.change_ops import change_operation, record_change
    from nodes.models import InstanceModelLogEntry, NodeConfig

    with change_operation(empty_db_instance, user=None, action='node.delete'):
        n1 = NodeConfig.objects.create(instance=empty_db_instance, identifier='n1', name='N1')
        n2 = NodeConfig.objects.create(instance=empty_db_instance, identifier='n2', name='N2')
        e1 = record_change(n1, action='node.create', before=None, after=n1.serializable_data())
        e2 = record_change(n2, action='node.create', before=None, after=n2.serializable_data())

    assert e1.operation_id == e2.operation_id  # type: ignore[attr-defined]
    entries = InstanceModelLogEntry.objects.filter(operation_id=e1.operation_id)  # type: ignore[attr-defined]
    assert entries.count() == 2


def test_record_change_outside_operation_raises(empty_db_instance: InstanceConfig):
    from nodes.change_ops import NoActiveChangeOperation, record_change
    from nodes.models import NodeConfig

    nc = NodeConfig.objects.create(instance=empty_db_instance, identifier='orphan', name='Orphan')
    with pytest.raises(NoActiveChangeOperation):
        record_change(nc, action='node.create', before=None, after=nc.serializable_data())


def test_change_operation_nested_reuses_outer(empty_db_instance: InstanceConfig):
    """Nested change_operation on the same instance reuses the outer operation."""
    from nodes.change_ops import change_operation
    from nodes.models import InstanceChangeOperation

    with change_operation(empty_db_instance, user=None, action='node.create') as outer:  # noqa: SIM117
        with change_operation(empty_db_instance, user=None, action='inner') as inner:
            assert inner.pk == outer.pk

    # Only one operation was created
    assert InstanceChangeOperation.objects.filter(pk=outer.pk).count() == 1


def test_change_operation_nested_different_instance_raises(empty_db_instance: InstanceConfig):
    """Nested change_operation on a different instance raises."""
    from nodes.change_ops import change_operation

    other = InstanceConfigFactory.create(
        identifier='other',
        instance=InstanceFactory.create(),
        config_source='database',
    )
    with change_operation(empty_db_instance, user=None, action='node.create'):  # noqa: SIM117
        with pytest.raises(RuntimeError, match='different InstanceConfig'):
            with change_operation(other, user=None, action='node.create'):
                pass


# ---------------------------------------------------------------------------
# snapshot_data methods
# ---------------------------------------------------------------------------


def test_node_config_serializable_data(empty_db_instance: InstanceConfig):
    from nodes.models import NodeConfig

    nc = NodeConfig.objects.create(
        instance=empty_db_instance,
        identifier='snap_test',
        name='Snap',
        color='#123456',
        order=7,
        is_visible=False,
    )
    data = nc.serializable_data()
    assert data['identifier'] == 'snap_test'
    # Name is a TranslatedString dict keyed by language
    assert data['name'] == {'en': 'Snap'}
    assert data['color'] == '#123456'
    assert data['order'] == 7
    assert data['is_visible'] is False
    # Round-trips as NodeSnapshot
    reloaded = NodeSnapshot.model_validate(data)
    assert reloaded.identifier == 'snap_test'


# ---------------------------------------------------------------------------
# Phase 2.5 PoC: node.create / node.update / node.delete via GraphQL
# ---------------------------------------------------------------------------


@pytest.fixture
def gql_client(client, empty_db_instance: InstanceConfig):
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    user = UserFactory.create(is_superuser=True)
    client.force_login(user)
    tc = PathsTestClient(client)
    tc.set_instance(empty_db_instance)
    return tc


CREATE_NODE_PoC = """
mutation CreateNode($instanceId: ID!, $input: CreateNodeInput!) {
    instanceEditor(instanceId: $instanceId) {
        createNode(input: $input) {
            ... on NodeInterface { identifier }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""

UPDATE_NODE_PoC = """
mutation UpdateNode($instanceId: ID!, $nodeId: ID!, $input: UpdateNodeInput!) {
    instanceEditor(instanceId: $instanceId) {
        updateNode(nodeId: $nodeId, input: $input) {
            ... on NodeInterface { identifier name }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""

DELETE_NODE_PoC = """
mutation DeleteNode($instanceId: ID!, $nodeId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        deleteNode(nodeId: $nodeId) {
            messages { kind message }
        }
    }
}
"""


def _make_formula_node_input(identifier: str) -> dict[str, Any]:
    return {
        'identifier': identifier,
        'name': f'Node {identifier}',
        'config': {'formula': {'formula': 'a + b'}},
        'color': '#ff0000',
        'outputPorts': [{'unit': 'kt/a', 'quantity': 'emissions'}],
    }


def test_poc_create_node_emits_change_operation(gql_client, empty_db_instance: InstanceConfig):
    from nodes.models import InstanceChangeOperation, InstanceModelLogEntry

    pre_ops = InstanceChangeOperation.objects.filter(instance_config=empty_db_instance).count()

    gql_client.query_data(
        CREATE_NODE_PoC,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'input': _make_formula_node_input('poc_create'),
        },
    )

    ops = InstanceChangeOperation.objects.filter(instance_config=empty_db_instance).order_by('-created_at')
    assert ops.count() == pre_ops + 1
    op = ops.first()
    assert op is not None
    assert op.action == 'node.create'
    assert op.source == 'graphql'
    assert op.user is not None

    entries = InstanceModelLogEntry.objects.filter(operation=op)
    assert entries.count() == 1
    entry = entries.first()
    assert entry is not None
    assert entry.action == 'node.create'
    assert entry.data['before'] is None
    assert entry.data['after']['identifier'] == 'poc_create'


def test_poc_update_node_emits_before_and_after(gql_client, empty_db_instance: InstanceConfig):
    from nodes.models import InstanceChangeOperation, InstanceModelLogEntry

    # Create first (via GQL so we don't bypass the editor's permission path)
    gql_client.query_data(
        CREATE_NODE_PoC,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'input': _make_formula_node_input('poc_update'),
        },
    )
    nc = empty_db_instance.nodes.get(identifier='poc_update')

    # Update
    gql_client.query_data(
        UPDATE_NODE_PoC,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'nodeId': str(nc.uuid),
            'input': {'name': 'Renamed', 'color': '#00ff00'},
        },
    )

    update_op = (
        InstanceChangeOperation.objects
        .filter(instance_config=empty_db_instance, action='node.update')
        .order_by('-created_at')
        .first()
    )
    assert update_op is not None

    entries = list(InstanceModelLogEntry.objects.filter(operation=update_op))
    assert len(entries) == 1
    entry = entries[0]
    assert entry.action == 'node.update'
    # Name is now serialized as a TranslatedString dict (lang → value)
    assert entry.data['before']['name'] == {'en': 'Node poc_update'}
    assert entry.data['before']['color'] == '#ff0000'
    assert entry.data['after']['name'] == {'en': 'Renamed'}
    assert entry.data['after']['color'] == '#00ff00'
    assert entry.data['target_uuid'] == str(nc.uuid)


def test_poc_delete_node_cascades_under_single_operation(
    gql_client,
    empty_db_instance: InstanceConfig,
):
    """node.delete produces one operation bundling the node entry + cascade entries."""
    # Create two nodes + an edge between them (via ORM — simpler for the PoC)
    from nodes.defs.node_defs import NodeSpec as NodeSpecDef
    from nodes.defs.port_def import OutputPortDef
    from nodes.models import InstanceChangeOperation, InstanceModelLogEntry, NodeConfig
    from nodes.units import unit_registry

    unit = unit_registry.parse_units('kt/a')
    from nodes.tests.test_model_editor import _port_uuid as _pu

    nc_a = NodeConfig.objects.create(
        instance=empty_db_instance,
        identifier='src',
        name='Src',
        spec=NodeSpecDef(
            output_ports=[OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions')],
        ),
    )
    nc_b = NodeConfig.objects.create(
        instance=empty_db_instance,
        identifier='dst',
        name='Dst',
        spec=NodeSpecDef(
            output_ports=[OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions')],
        ),
    )
    from nodes.models import NodeEdge

    edge = NodeEdge.objects.create(
        instance=empty_db_instance,
        from_node=nc_a,
        to_node=nc_b,
        from_port=_pu('default'),
        to_port=_pu('input_1'),
    )
    src_pk = nc_a.pk

    gql_client.query_data(
        DELETE_NODE_PoC,
        variables={'instanceId': str(empty_db_instance.pk), 'nodeId': str(nc_a.uuid)},
    )

    # One operation, multiple entries under it (node.delete + cascaded edge).
    ops = list(
        InstanceChangeOperation.objects.filter(instance_config=empty_db_instance, action='node.delete'),
    )
    assert len(ops) == 1
    op = ops[0]

    entries = list(InstanceModelLogEntry.objects.filter(operation=op).order_by('id'))
    actions = [e.action for e in entries]
    # edge entries first (recorded before node), then the node itself
    assert 'node.edges.delete' in actions
    assert 'node.delete' in actions
    # All entries share the operation
    assert all(e.operation_id == op.pk for e in entries)  # type: ignore[attr-defined]

    # Edge entry preserves target_uuid for undo
    edge_entry = next(e for e in entries if e.action == 'node.edges.delete')
    assert edge_entry.data['target_uuid'] == str(edge.uuid)
    assert edge_entry.data['after'] is None
    assert edge_entry.data['before']['from_node'] == str(nc_a.uuid)

    # Node is actually gone
    assert not NodeConfig.objects.filter(pk=src_pk).exists()


# ---------------------------------------------------------------------------
# Dataset RevisionMixin + paths bridge
# ---------------------------------------------------------------------------


def test_dataset_serializable_data_bridges_to_paths():
    from kausal_common.datasets.tests.factories import DatasetFactory, DatasetMetricFactory

    from nodes.instance_serialization import DatasetSnapshot

    ds = DatasetFactory.create()
    DatasetMetricFactory.create(schema=ds.schema, name='m1', label='Metric 1', unit='kt/a')

    data = ds.serializable_data()
    # Round-trips as DatasetSnapshot
    snap = DatasetSnapshot.model_validate(data)
    assert snap.schema_version == 1
    assert any(m.identifier == 'm1' for m in snap.metrics)
    # data field is included (though None when there are no datapoints yet)
    assert 'data' in data


def test_dataset_serializable_data_includes_forecast_from():
    from kausal_common.datasets.tests.factories import DatasetFactory

    from nodes.instance_serialization import DatasetSnapshot, _import_dataset
    from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

    source = InstanceConfigFactory.create(instance=InstanceFactory.create(), config_source='database')
    target = InstanceConfigFactory.create(instance=InstanceFactory.create(), config_source='database')
    ds = DatasetFactory.create(scope=source, identifier='forecasted', spec={'forecast_from': 2025})

    snap = DatasetSnapshot.from_model_for_instance(ds, source)
    assert snap.forecast_from == 2025

    from django.contrib.contenttypes.models import ContentType

    copied = _import_dataset(target, snap, ContentType.objects.get_for_model(target), {})
    assert copied.spec == {'forecast_from': 2025}


def test_dataset_save_revision_updates_latest_revision():
    from kausal_common.datasets.tests.factories import DatasetFactory

    ds = DatasetFactory.create()
    assert ds.latest_revision_id is None
    ds.save_revision()
    ds.refresh_from_db()
    assert ds.latest_revision_id is not None


def test_dataset_port_snapshot_pins_dataset_revision(empty_db_instance: InstanceConfig):
    from kausal_common.datasets.tests.factories import DatasetFactory, DatasetMetricFactory

    from nodes.models import DatasetPort

    ds = DatasetFactory.create()
    metric = DatasetMetricFactory.create(schema=ds.schema, name='m1', label='M', unit='kt/a')
    ds.save_revision()
    ds.refresh_from_db()
    pinned_rev = ds.latest_revision_id

    nc = NodeConfigFactory.create(instance=empty_db_instance, identifier='owner', name='Owner')
    import uuid as _uuid

    port = DatasetPort.objects.create(
        instance=empty_db_instance,
        node=nc,
        port_id=_uuid.uuid4(),
        dataset=ds,
        metric=metric,
    )

    # serializable_data pins the dataset's current revision
    data = port.serializable_data()
    assert data['dataset_revision'] == pinned_rev


def _make_materialized_dataset(instance_config: InstanceConfig, identifier: str, value: str):
    from datetime import date
    from decimal import Decimal

    from kausal_common.datasets.tests.factories import DataPointFactory, DatasetFactory, DatasetMetricFactory

    from nodes.dataset_materialization import materialize_dataset

    dataset = DatasetFactory.create(identifier=identifier, scope=instance_config)
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='value', label='Value', unit='t/a')
    point = DataPointFactory.create(dataset=dataset, metric=metric, date=date(2020, 1, 1), value=Decimal(value))
    materialization = materialize_dataset(dataset)
    return dataset, metric, point, materialization


def _materialized_df_value(content: dict[str, Any]) -> float:
    from nodes.datasets import JSONDataset

    df = JSONDataset.deserialize_df(content['data'])
    return float(df['value'][0])


def test_publish_pins_current_dataset_materialization(empty_db_instance: InstanceConfig):
    from nodes.models import DatasetPort, InstanceRevisionDatasetPin

    dataset, metric, _point, materialization = _make_materialized_dataset(empty_db_instance, 'pinned', '10')
    node = NodeConfigFactory.create(instance=empty_db_instance, identifier='owner', name='Owner')
    DatasetPort.objects.create(
        instance=empty_db_instance,
        node=node,
        port_id=uuid.uuid4(),
        dataset=dataset,
        metric=metric,
    )

    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()

    pin = InstanceRevisionDatasetPin.objects.select_related('dataset_revision').get(
        instance_revision_id=empty_db_instance.live_revision_id,
        dataset=dataset,
    )
    dataset.refresh_from_db()
    assert pin.instance_config == empty_db_instance
    assert pin.dataset_uuid == dataset.uuid
    assert pin.dataset_revision_id == dataset.latest_revision_id
    assert pin.dataset_revision.content == materialization.content
    assert _materialized_df_value(pin.dataset_revision.content) == 10

    assert empty_db_instance.live_revision is not None
    structured = empty_db_instance.live_revision.content['model_snapshot']['structured']
    manifest = structured['dataset_revisions']
    assert manifest == [
        {
            'dataset_uuid': str(dataset.uuid),
            'identifier': 'pinned',
            'revision_id': pin.dataset_revision_id,
            'content_hash': materialization.content_hash,
            'generation': materialization.generation,
            'forecast_from': None,
        }
    ]


def test_published_runtime_rejects_missing_relational_dataset_pin(empty_db_instance: InstanceConfig):
    from nodes.models import DatasetPort, InstanceRevisionDatasetPin, PreferredInstanceSource

    dataset, metric, _point, _materialization = _make_materialized_dataset(empty_db_instance, 'missing-pin', '10')
    node = NodeConfigFactory.create(instance=empty_db_instance, identifier='owner', name='Owner')
    DatasetPort.objects.create(
        instance=empty_db_instance,
        node=node,
        port_id=uuid.uuid4(),
        dataset=dataset,
        metric=metric,
    )
    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()
    InstanceRevisionDatasetPin.objects.filter(instance_revision_id=empty_db_instance.live_revision_id).delete()

    with pytest.raises(RuntimeError, match='dataset manifest mismatch'):
        empty_db_instance._create_from_config(source=PreferredInstanceSource.PUBLISHED)


def test_published_dataset_payload_is_isolated_from_later_draft_edit(empty_db_instance: InstanceConfig):
    from decimal import Decimal

    from django.db import transaction

    from nodes.dataset_materialization import refresh_dataset_materialization
    from nodes.models import DatasetPort, InstanceRevisionDatasetPin

    dataset, metric, point, _materialization = _make_materialized_dataset(empty_db_instance, 'isolated', '10')
    node = NodeConfigFactory.create(instance=empty_db_instance, identifier='owner', name='Owner')
    DatasetPort.objects.create(
        instance=empty_db_instance,
        node=node,
        port_id=uuid.uuid4(),
        dataset=dataset,
        metric=metric,
    )
    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()
    published_revision = InstanceRevisionDatasetPin.objects.get(
        instance_revision_id=empty_db_instance.live_revision_id,
        dataset=dataset,
    ).dataset_revision

    with transaction.atomic():
        point.value = Decimal(20)
        point.save(update_fields=['value'])
        draft_materialization = refresh_dataset_materialization(dataset)

    published_revision.refresh_from_db()
    assert _materialized_df_value(published_revision.content) == 10
    assert _materialized_df_value(draft_materialization.content) == 20


def test_draft_and_published_runtime_share_serialized_dataset_path(empty_db_instance: InstanceConfig):
    from decimal import Decimal
    from typing import cast

    from django.db import transaction

    from nodes.dataset_materialization import refresh_dataset_materialization
    from nodes.datasets import DatasetWithFilters, SerializedDBDataset
    from nodes.defs.node_defs import SimpleConfig
    from nodes.defs.port_def import InputPortDef, OutputPortDef
    from nodes.models import DatasetPort, NodeConfig, PreferredInstanceSource
    from nodes.tests.test_model_editor import SIMPLE_NODE_CLASS, _port_uuid
    from nodes.units import unit_registry

    assert empty_db_instance.spec is not None
    empty_db_instance.spec.features.use_datasets_from_db = True
    empty_db_instance.save(update_fields=['spec'])

    dataset, metric, point, _materialization = _make_materialized_dataset(empty_db_instance, 'runtime', '10')
    port_id = _port_uuid('runtime-value')
    node = NodeConfig.objects.create(
        instance=empty_db_instance,
        identifier='runtime_node',
        name='Runtime node',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class=SIMPLE_NODE_CLASS),
            input_ports=[
                InputPortDef(
                    id=port_id,
                    unit=unit_registry.parse_units('t/a'),
                    quantity='emissions',
                )
            ],
            output_ports=[
                OutputPortDef(
                    id=_port_uuid('runtime-output'),
                    unit=unit_registry.parse_units('t/a'),
                    quantity='emissions',
                )
            ],
        ),
    )
    DatasetPort.objects.create(
        instance=empty_db_instance,
        node=node,
        port_id=port_id,
        dataset=dataset,
        metric=metric,
    )
    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()

    with transaction.atomic():
        point.value = Decimal(20)
        point.save(update_fields=['value'])
        refresh_dataset_materialization(dataset)

    published = empty_db_instance._create_from_config(source=PreferredInstanceSource.PUBLISHED)
    draft = empty_db_instance._create_from_config(source=PreferredInstanceSource.DRAFT)
    published_dataset = cast('DatasetWithFilters', published.context.nodes['runtime_node'].input_dataset_instances[0])
    draft_dataset = cast('DatasetWithFilters', draft.context.nodes['runtime_node'].input_dataset_instances[0])
    assert isinstance(published_dataset, SerializedDBDataset)
    assert isinstance(draft_dataset, SerializedDBDataset)

    with CaptureQueriesContext(connection) as published_queries:
        published_df = published_dataset.get_copy()
    with CaptureQueriesContext(connection) as draft_queries:
        draft_df = draft_dataset.get_copy()

    assert float(published_df['value'][0]) == 10
    assert float(draft_df['value'][0]) == 20
    assert sum('wagtailcore_revision' in query['sql'].lower() for query in published_queries) == 1
    assert sum('nodes_datasetmaterialization' in query['sql'].lower() for query in draft_queries) == 1
    assert not any('datasets_datapoint' in query['sql'].lower() for query in published_queries)

    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()
    republished = empty_db_instance._create_from_config(source=PreferredInstanceSource.PUBLISHED)
    republished_dataset = cast(
        'DatasetWithFilters',
        republished.context.nodes['runtime_node'].input_dataset_instances[0],
    )
    assert float(republished_dataset.get_copy()['value'][0]) == 20


def test_current_dataset_payload_store_bulk_loads_once(empty_db_instance: InstanceConfig):
    from nodes.datasets import CurrentDatasetPayloadStore, DatasetPayloadRef

    refs = []
    for identifier, value in [('one', '1'), ('two', '2')]:
        dataset, _metric, _point, materialization = _make_materialized_dataset(
            empty_db_instance,
            identifier,
            value,
        )
        refs.append(
            DatasetPayloadRef(
                payload_id=materialization.pk,
                dataset_pk=dataset.pk,
                dataset_uuid=str(dataset.uuid),
                identifier=identifier,
                content_hash=materialization.content_hash,
                generation=materialization.generation,
                forecast_from=materialization.forecast_from,
            )
        )

    store = CurrentDatasetPayloadStore(refs)
    with CaptureQueriesContext(connection) as queries:
        assert float(store.get_dataframe(refs[0])['value'][0]) == 1
        assert float(store.get_dataframe(refs[1])['value'][0]) == 2
        assert float(store.get_dataframe(refs[0])['value'][0]) == 1

    payload_queries = [query for query in queries if 'nodes_datasetmaterialization' in query['sql'].lower()]
    assert len(payload_queries) == 1


def test_revision_dataset_payload_store_bulk_loads_once(empty_db_instance: InstanceConfig):
    from nodes.datasets import DatasetPayloadRef, RevisionDatasetPayloadStore
    from nodes.models import DatasetPort

    node = NodeConfigFactory.create(instance=empty_db_instance, identifier='owner', name='Owner')
    for identifier, value in [('published-one', '1'), ('published-two', '2')]:
        dataset, metric, _point, _materialization = _make_materialized_dataset(
            empty_db_instance,
            identifier,
            value,
        )
        DatasetPort.objects.create(
            instance=empty_db_instance,
            node=node,
            port_id=uuid.uuid4(),
            dataset=dataset,
            metric=metric,
        )
    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()
    assert empty_db_instance.live_revision is not None
    structured = empty_db_instance.live_revision.content['model_snapshot']['structured']
    snapshot = InstanceSnapshot.model_validate(structured)
    refs = [
        DatasetPayloadRef(
            payload_id=pin.revision_id,
            dataset_pk=0,
            dataset_uuid=str(pin.dataset_uuid),
            identifier=pin.identifier or str(pin.dataset_uuid),
            content_hash=pin.content_hash,
            generation=None,
            forecast_from=pin.forecast_from,
        )
        for pin in snapshot.dataset_revisions
    ]

    store = RevisionDatasetPayloadStore(refs)
    with CaptureQueriesContext(connection) as queries:
        assert sorted(float(store.get_dataframe(ref)['value'][0]) for ref in refs) == [1, 2]
        assert float(store.get_dataframe(refs[0])['value'][0]) in {1, 2}

    revision_queries = [query for query in queries if 'wagtailcore_revision' in query['sql'].lower()]
    assert len(revision_queries) == 1


def test_pinned_dataset_revision_is_protected_until_instance_is_deleted(empty_db_instance: InstanceConfig):
    from django.db.models.deletion import ProtectedError

    from nodes.models import DatasetPort, InstanceRevisionDatasetPin

    dataset, metric, _point, _materialization = _make_materialized_dataset(empty_db_instance, 'retained', '10')
    node = NodeConfigFactory.create(instance=empty_db_instance, identifier='owner', name='Owner')
    DatasetPort.objects.create(
        instance=empty_db_instance,
        node=node,
        port_id=uuid.uuid4(),
        dataset=dataset,
        metric=metric,
    )
    empty_db_instance.publish_instance()

    with pytest.raises(ProtectedError):
        dataset.delete()

    instance_pk = empty_db_instance.pk
    empty_db_instance.delete()
    assert not InstanceRevisionDatasetPin.objects.filter(instance_config_id=instance_pk).exists()
    dataset.delete()


def test_export_instance_includes_schema_scoped_placeholder(empty_db_instance: InstanceConfig):
    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import DatasetSchemaScope
    from kausal_common.datasets.tests.factories import DatasetFactory, DatasetSchemaFactory

    from nodes.instance_serialization import export_instance

    schema = DatasetSchemaFactory.create(name='Placeholder schema')
    DatasetSchemaScope.objects.create(
        schema=schema,
        scope_content_type=ContentType.objects.get_for_model(empty_db_instance),
        scope_id=empty_db_instance.pk,
    )
    DatasetFactory.create(
        schema=schema,
        identifier='external/source',
        is_external_placeholder=True,
    )

    export = export_instance(empty_db_instance)

    assert [(ds.identifier, ds.is_external_placeholder) for ds in export.datasets] == [('external/source', True)]


def test_import_instance_datasets_rewires_ports_and_removes_placeholder(empty_db_instance: InstanceConfig):
    import datetime
    from decimal import Decimal

    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import Dataset, DatasetSchemaScope
    from kausal_common.datasets.tests.factories import (
        DataPointFactory,
        DatasetFactory,
        DatasetMetricFactory,
        DatasetSchemaFactory,
    )

    from nodes.instance_serialization import export_instance, import_instance_datasets
    from nodes.models import DatasetMaterialization, DatasetPort, NodeConfig

    source = empty_db_instance
    target_instance = InstanceFactory.create()
    target = InstanceConfigFactory.create(
        identifier=target_instance.id,
        instance=target_instance,
        config_source='database',
        owner='Target',
        spec=InstanceModelSpec(
            years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
        ),
    )
    ic_ct = ContentType.objects.get_for_model(source)

    source_dataset = DatasetFactory.create(identifier='real/source', scope=source)
    source_metric = DatasetMetricFactory.create(schema=source_dataset.schema, name='value', label='Value', unit='kt/a')
    DataPointFactory.create(
        dataset=source_dataset,
        metric=source_metric,
        date=datetime.date(2020, 1, 1),
        value=Decimal('42.5'),
    )

    placeholder_schema = DatasetSchemaFactory.create(name='Placeholder schema')
    DatasetSchemaScope.objects.create(
        schema=placeholder_schema,
        scope_content_type=ContentType.objects.get_for_model(target),
        scope_id=target.pk,
    )
    placeholder = DatasetFactory.create(
        schema=placeholder_schema,
        identifier='real/source',
        is_external_placeholder=True,
    )
    placeholder_metric = DatasetMetricFactory.create(schema=placeholder_schema, name='value', label='Value', unit='kt/a')
    node = NodeConfig.objects.create(instance=target, identifier='receiver', name='Receiver')
    port = DatasetPort.objects.create(
        instance=target,
        node=node,
        port_id=uuid.uuid4(),
        dataset=placeholder,
        metric=placeholder_metric,
    )

    source_export = export_instance(source)
    imported = import_instance_datasets(
        target,
        source_export.datasets,
        rewire_dataset_ports=True,
        delete_superseded_placeholders=True,
    )

    assert len(imported) == 1
    copied_dataset = Dataset.objects.get(
        scope_content_type=ic_ct,
        scope_id=target.pk,
        identifier='real/source',
    )
    assert copied_dataset.data_points.count() == 1
    assert not Dataset.objects.filter(pk=placeholder.pk).exists()
    assert DatasetMaterialization.objects.filter(dataset=copied_dataset, generation=1).exists()

    port.refresh_from_db()
    assert port.dataset == copied_dataset
    assert port.metric.schema == copied_dataset.schema


def test_import_instance_datasets_preserves_dimension_column_name(empty_db_instance: InstanceConfig):
    import datetime
    from decimal import Decimal

    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import Dataset, DatasetSchemaDimension, DimensionScope
    from kausal_common.datasets.tests.factories import (
        DataPointFactory,
        DatasetFactory,
        DatasetMetricFactory,
        DatasetSchemaDimensionFactory,
        DimensionCategoryFactory,
        DimensionFactory,
    )

    from nodes.datasets import DBDataset
    from nodes.instance_serialization import export_instance, import_instance_datasets

    source = empty_db_instance
    target_instance = InstanceFactory.create()
    target = InstanceConfigFactory.create(
        identifier=target_instance.id,
        instance=target_instance,
        config_source='database',
        owner='Target',
        spec=InstanceModelSpec(
            years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
        ),
    )
    source_ct = ContentType.objects.get_for_model(source)

    dimension = DimensionFactory.create(name='Green Mobility Action')
    DimensionScope.objects.create(
        dimension=dimension,
        scope_content_type=source_ct,
        scope_id=source.pk,
        identifier='green_mobility_action',
    )
    category = DimensionCategoryFactory.create(
        dimension=dimension,
        identifier='school_roads',
        label='School Roads',
    )
    dataset = DatasetFactory.create(identifier='actions/source', scope=source)
    DatasetSchemaDimensionFactory.create(schema=dataset.schema, dimension=dimension, column_name='action')
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='fraction', label='Fraction', unit='%')
    DataPointFactory.create(
        dataset=dataset,
        metric=metric,
        date=datetime.date(2020, 1, 1),
        value=Decimal('12.5'),
        dimension_categories=[category],
    )

    snapshot = next(ds for ds in export_instance(source).datasets if ds.identifier == 'actions/source')
    assert snapshot.dimensions == ['green_mobility_action']
    assert snapshot.dimension_columns == {'green_mobility_action': 'action'}
    assert snapshot.data is not None
    assert snapshot.data['schema']['primaryKey'] == ['Year', 'action']

    imported = import_instance_datasets(target, [snapshot], create_missing_dimensions=True)
    assert len(imported) == 1

    copied_dataset = Dataset.objects.get(scope_content_type=source_ct, scope_id=target.pk, identifier='actions/source')
    copied_schema_dim = DatasetSchemaDimension.objects.select_related('dimension').get(schema=copied_dataset.schema)
    copied_scope = DimensionScope.objects.get(
        dimension=copied_schema_dim.dimension,
        scope_content_type=source_ct,
        scope_id=target.pk,
    )
    assert copied_scope.identifier == 'green_mobility_action'
    assert copied_schema_dim.column_name == 'action'
    assert copied_dataset.data_points.count() == 1

    copied_df = DBDataset.deserialize_df(copied_dataset)
    assert copied_df.primary_keys == ['Year', 'action']
    assert 'green_mobility_action' not in copied_df.columns


def test_import_instance_preserves_dataset_only_dimension(empty_db_instance: InstanceConfig):
    import datetime
    from decimal import Decimal

    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import Dataset, DatasetSchemaDimension, DimensionScope
    from kausal_common.datasets.tests.factories import (
        DataPointFactory,
        DatasetFactory,
        DatasetMetricFactory,
        DatasetSchemaDimensionFactory,
        DimensionCategoryFactory,
        DimensionFactory,
    )

    from nodes.datasets import DBDataset
    from nodes.instance_serialization import export_instance, import_instance

    source = empty_db_instance
    target_instance = InstanceFactory.create()
    target = InstanceConfigFactory.create(
        identifier=target_instance.id,
        instance=target_instance,
        config_source='database',
        owner='Target',
    )
    source_ct = ContentType.objects.get_for_model(source)

    dimension = DimensionFactory.create(name='Green Mobility Action')
    DimensionScope.objects.create(
        dimension=dimension,
        scope_content_type=source_ct,
        scope_id=source.pk,
        identifier='green_mobility_action',
    )
    category = DimensionCategoryFactory.create(
        dimension=dimension,
        identifier='school_roads',
        label='School Roads',
    )
    dataset = DatasetFactory.create(identifier='actions/source', scope=source)
    DatasetSchemaDimensionFactory.create(schema=dataset.schema, dimension=dimension, column_name='action')
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='fraction', label='Fraction', unit='%')
    DataPointFactory.create(
        dataset=dataset,
        metric=metric,
        date=datetime.date(2020, 1, 1),
        value=Decimal('12.5'),
        dimension_categories=[category],
    )

    export = export_instance(source)
    assert 'green_mobility_action' not in {dim['id'] for dim in export.instance.spec.dimensions}
    snapshot = next(ds for ds in export.datasets if ds.identifier == 'actions/source')
    assert snapshot.dimensions == ['green_mobility_action']
    assert snapshot.dimension_columns == {'green_mobility_action': 'action'}

    import_instance(target, export)

    copied_dataset = Dataset.objects.get(scope_content_type=source_ct, scope_id=target.pk, identifier='actions/source')
    copied_schema_dim = DatasetSchemaDimension.objects.select_related('dimension').get(schema=copied_dataset.schema)
    copied_scope = DimensionScope.objects.get(
        dimension=copied_schema_dim.dimension,
        scope_content_type=source_ct,
        scope_id=target.pk,
    )
    assert copied_scope.identifier == 'green_mobility_action'
    assert copied_schema_dim.column_name == 'action'
    assert copied_dataset.data_points.count() == 1

    copied_df = DBDataset.deserialize_df(copied_dataset)
    assert copied_df.primary_keys == ['Year', 'action']


# ---------------------------------------------------------------------------
# Demo-flow mutations (edges, dimension categories, datapoints)
#
# Exercise the mutations involved in the Tuesday demo's "copy an action"
# walkthrough to verify change tracking fires for each step.
# ---------------------------------------------------------------------------


CREATE_EDGE = """
mutation CreateEdge($instanceId: ID!, $input: CreateEdgeInput!) {
    instanceEditor(instanceId: $instanceId) {
        createEdge(input: $input) {
            __typename
            ... on NodeEdgeType { fromRef { nodeId portId } portRef { nodeId portId } }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""

DELETE_EDGE = """
mutation DeleteEdge($instanceId: ID!, $edgeId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        deleteEdge(edgeId: $edgeId) { messages { kind message } }
    }
}
"""

CREATE_DIMENSION_CATEGORIES = """
mutation CreateCats($instanceId: ID!, $input: [CreateDimensionCategoryInput!]!) {
    instanceEditor(instanceId: $instanceId) {
        createDimensionCategories(input: $input) {
            ... on InstanceDimension { id categories { id identifier label } }
        }
    }
}
"""


def _build_edge_endpoints(db_instance: InstanceConfig):
    """Create two formula nodes with compatible single-port outputs; return (src, dst)."""
    from nodes.defs.node_defs import NodeSpec as NodeSpecDef
    from nodes.defs.port_def import InputPortDef, OutputPortDef
    from nodes.models import NodeConfig
    from nodes.tests.test_model_editor import _port_uuid as _pu
    from nodes.units import unit_registry

    unit = unit_registry.parse_units('kt/a')
    src = NodeConfig.objects.create(
        instance=db_instance,
        identifier='edge_src',
        name='Src',
        spec=NodeSpecDef(
            output_ports=[
                OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions'),
            ],
        ),
    )
    dst = NodeConfig.objects.create(
        instance=db_instance,
        identifier='edge_dst',
        name='Dst',
        spec=NodeSpecDef(
            input_ports=[InputPortDef(id=_pu('input'), unit=unit, quantity='emissions')],
            output_ports=[OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions')],
        ),
    )
    return src, dst


def test_poc_create_edge_emits_change_operation(gql_client, empty_db_instance: InstanceConfig):
    from nodes.models import InstanceChangeOperation, InstanceModelLogEntry, NodeEdge

    src, dst = _build_edge_endpoints(empty_db_instance)

    data = gql_client.query_data(
        CREATE_EDGE,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'input': {
                'instanceId': str(empty_db_instance.pk),
                'fromNodeId': src.identifier,
                'toNodeId': dst.identifier,
            },
        },
    )
    assert data['instanceEditor']['createEdge']['__typename'] == 'NodeEdgeType'

    op = InstanceChangeOperation.objects.filter(instance_config=empty_db_instance, action='edge.create').first()
    assert op is not None
    entries = list(InstanceModelLogEntry.objects.filter(operation=op))
    assert len(entries) == 1
    assert entries[0].action == 'edge.create'
    assert entries[0].data['before'] is None
    assert entries[0].data['after']['from_node'] == str(src.uuid)
    assert NodeEdge.objects.filter(instance=empty_db_instance).count() == 1


def test_poc_delete_edge_emits_change_operation(gql_client, empty_db_instance: InstanceConfig):
    from nodes.models import InstanceChangeOperation, InstanceModelLogEntry, NodeEdge
    from nodes.tests.test_model_editor import _port_uuid as _pu

    src, dst = _build_edge_endpoints(empty_db_instance)
    edge = NodeEdge.objects.create(
        instance=empty_db_instance,
        from_node=src,
        to_node=dst,
        from_port=_pu('default'),
        to_port=_pu('input'),
    )

    gql_client.query_data(
        DELETE_EDGE,
        variables={'instanceId': str(empty_db_instance.pk), 'edgeId': str(edge.uuid)},
    )

    op = InstanceChangeOperation.objects.filter(instance_config=empty_db_instance, action='edge.delete').first()
    assert op is not None
    entry = InstanceModelLogEntry.objects.filter(operation=op).first()
    assert entry is not None
    assert entry.action == 'edge.delete'
    assert entry.data['before']['from_node'] == str(src.uuid)
    assert entry.data['after'] is None


def test_create_edge_auto_creates_matching_target_port(
    gql_client,
    empty_db_instance: InstanceConfig,
):
    """When toPort is null on a multi-port target, a new input port is created."""
    from nodes.defs.node_defs import NodeSpec as NodeSpecDef
    from nodes.defs.port_def import InputPortDef, OutputPortDef
    from nodes.models import (
        InstanceChangeOperation,
        InstanceModelLogEntry,
        NodeConfig,
        NodeEdge,
    )
    from nodes.tests.test_model_editor import _port_uuid as _pu
    from nodes.units import unit_registry

    unit = unit_registry.parse_units('kt/a')

    def _make_node(ident: str, spec: NodeSpecDef) -> NodeConfig:
        # Direct NodeConfig.objects.create loses spec via ClusterableModel.save;
        # use queryset.update after, as _import_nodes does.
        nc = NodeConfig.objects.create(instance=empty_db_instance, identifier=ident, name=ident.title())
        NodeConfig.objects.filter(pk=nc.pk).update(spec=spec)
        nc.refresh_from_db()
        return nc

    src = _make_node(
        'auto_src',
        NodeSpecDef(
            output_ports=[
                OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions'),
            ],
        ),
    )
    # Target already has two *bound* input ports — new edge must add a third.
    existing_port_a = InputPortDef(id=_pu('existing_a'), unit=unit, quantity='emissions')
    existing_port_b = InputPortDef(id=_pu('existing_b'), unit=unit, quantity='emissions')
    dst = _make_node(
        'auto_dst',
        NodeSpecDef(
            input_ports=[existing_port_a, existing_port_b],
            output_ports=[OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions')],
        ),
    )
    # Pre-bind both existing ports so the resolver can't reuse them.
    other = _make_node(
        'auto_other',
        NodeSpecDef(
            output_ports=[
                OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions'),
            ],
        ),
    )
    NodeEdge.objects.create(
        instance=empty_db_instance,
        from_node=other,
        to_node=dst,
        from_port=_pu('default'),
        to_port=_pu('existing_a'),
    )
    NodeEdge.objects.create(
        instance=empty_db_instance,
        from_node=other,
        to_node=dst,
        from_port=_pu('default'),
        to_port=_pu('existing_b'),
    )

    gql_client.query_data(
        CREATE_EDGE,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'input': {
                'instanceId': str(empty_db_instance.pk),
                'fromNodeId': src.identifier,
                'toNodeId': dst.identifier,
                # toPort omitted — auto-create is expected
            },
        },
    )

    # Target now has 3 input ports (two existing + one fresh).
    # Default manager defers `spec` — refetch explicitly via with_spec().
    dst_with_spec = NodeConfig.objects.with_spec().get(pk=dst.pk)
    assert dst_with_spec.spec is not None
    assert len(dst_with_spec.spec.input_ports) == 3

    # The new edge wires to the freshly-added port.
    new_edge = NodeEdge.objects.filter(instance=empty_db_instance, from_node=src, to_node=dst).first()
    assert new_edge is not None
    assert new_edge.to_port == dst_with_spec.spec.input_ports[-1].id

    # One change operation with two entries: node.update + edge.create.
    op = InstanceChangeOperation.objects.filter(instance_config=empty_db_instance, action='edge.create').first()
    assert op is not None
    entries = list(InstanceModelLogEntry.objects.filter(operation=op).order_by('id'))
    actions = [e.action for e in entries]
    assert actions == ['node.update', 'edge.create']
    # The node.update entry pins the before/after input-port counts.
    node_update = entries[0]
    assert len(node_update.data['before']['spec']['input_ports']) == 2
    assert len(node_update.data['after']['spec']['input_ports']) == 3


def test_poc_create_dimension_category_emits_change_operation(
    gql_client,
    empty_db_instance: InstanceConfig,
):
    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import DimensionScope
    from kausal_common.datasets.tests.factories import DimensionFactory

    from nodes.models import InstanceChangeOperation, InstanceModelLogEntry

    dim = DimensionFactory.create(name='Sector')
    DimensionScope.objects.create(
        dimension=dim,
        scope_content_type=ContentType.objects.get_for_model(empty_db_instance),
        scope_id=empty_db_instance.pk,
        identifier='sector',
    )

    gql_client.query_data(
        CREATE_DIMENSION_CATEGORIES,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'input': [{'dimensionId': str(dim.uuid), 'identifier': 'energy', 'label': 'Energy'}],
        },
    )

    op = InstanceChangeOperation.objects.filter(instance_config=empty_db_instance, action='dimension.categories.create').first()
    assert op is not None
    entries = list(InstanceModelLogEntry.objects.filter(operation=op))
    assert len(entries) == 1
    assert entries[0].action == 'dimension.categories.create'
    assert entries[0].data['before'] is None
    assert entries[0].data['after']['identifier'] == 'energy'
    assert entries[0].data['after']['dimension_uuid'] == str(dim.uuid)


# ---------------------------------------------------------------------------
# Phase 3: optimistic locking via draftHeadToken + @instance(version)
# ---------------------------------------------------------------------------


CREATE_NODE_WITH_VERSION = """
mutation CreateNode($instanceId: ID!, $version: UUID, $input: CreateNodeInput!) {
    instanceEditor(instanceId: $instanceId, version: $version) {
        createNode(input: $input) {
            ... on NodeInterface { identifier }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""


def test_draft_head_token_advances_after_mutation(gql_client, empty_db_instance: InstanceConfig):
    """Each successful mutation produces a new InstanceChangeOperation, moving the head."""
    assert empty_db_instance.draft_head_token is None

    gql_client.query_data(
        CREATE_NODE_WITH_VERSION,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'version': None,
            'input': _make_formula_node_input('token_advance_1'),
        },
    )
    empty_db_instance.refresh_from_db()
    first_token = empty_db_instance.draft_head_token
    assert first_token is not None

    gql_client.query_data(
        CREATE_NODE_WITH_VERSION,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'version': str(first_token),
            'input': _make_formula_node_input('token_advance_2'),
        },
    )
    empty_db_instance.refresh_from_db()
    second_token = empty_db_instance.draft_head_token
    assert second_token is not None
    assert second_token != first_token


def test_stale_version_rejected_with_extensions(gql_client, empty_db_instance: InstanceConfig):
    """A mutation carrying an out-of-date version is rejected with stale_version code."""
    gql_client.query_data(
        CREATE_NODE_WITH_VERSION,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'version': None,
            'input': _make_formula_node_input('stale_baseline'),
        },
    )
    empty_db_instance.refresh_from_db()
    observed_token = empty_db_instance.draft_head_token
    assert observed_token is not None

    # Second client advances the head without re-reading
    gql_client.query_data(
        CREATE_NODE_WITH_VERSION,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'version': str(observed_token),
            'input': _make_formula_node_input('stale_advance'),
        },
    )
    empty_db_instance.refresh_from_db()
    current_token = empty_db_instance.draft_head_token
    assert current_token is not None
    assert current_token != observed_token

    # Third mutation uses the *old* token → rejected
    errors = gql_client.query_errors(
        CREATE_NODE_WITH_VERSION,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'version': str(observed_token),
            'input': _make_formula_node_input('stale_rejected'),
        },
    )
    assert len(errors) == 1
    ext = errors[0].get('extensions') or {}
    assert ext.get('code') == 'stale_version'
    assert ext.get('expectedHeadToken') == str(observed_token)
    assert ext.get('currentHeadToken') == str(current_token)
    # latestOperations should carry at least one entry (the intervening advance).
    latest = ext.get('latestOperations') or []
    assert len(latest) >= 1
    assert all('uuid' in op and 'action' in op for op in latest)

    # Side-effect check: the rejected mutation did not create the node.
    assert not empty_db_instance.nodes.filter(identifier='stale_rejected').exists()


def test_null_version_skips_check(gql_client, empty_db_instance: InstanceConfig):
    """During rollout, omitting the version token leaves the mutation unchecked."""
    gql_client.query_data(
        CREATE_NODE_WITH_VERSION,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'version': None,
            'input': _make_formula_node_input('null_version_ok'),
        },
    )
    # Now advance the head, then run another mutation with null version — succeeds.
    empty_db_instance.refresh_from_db()
    assert empty_db_instance.draft_head_token is not None

    gql_client.query_data(
        CREATE_NODE_WITH_VERSION,
        variables={
            'instanceId': str(empty_db_instance.pk),
            'version': None,
            'input': _make_formula_node_input('null_version_still_ok'),
        },
    )
    assert empty_db_instance.nodes.filter(identifier='null_version_still_ok').exists()


# ---------------------------------------------------------------------------
# Phase 4 (#1): resolver split — PreferredInstanceSource plumbing
# ---------------------------------------------------------------------------


def test_preferred_instance_source_enum_values():
    """Enum members serialize to the exact literals _create_from_config expects."""
    from nodes.models import PreferredInstanceSource

    assert PreferredInstanceSource.DRAFT.value == 'draft'
    assert PreferredInstanceSource.PUBLISHED.value == 'published'
    # StrEnum equality with raw strings — callers can pass either form.
    assert PreferredInstanceSource.DRAFT == 'draft'
    assert PreferredInstanceSource.PUBLISHED == 'published'


def test_published_source_falls_back_when_no_revision(empty_db_instance: InstanceConfig):
    """
    PUBLISHED on an instance that's never been published falls through to draft.

    Prevents 500s on freshly-created instances where the editor UI might
    pre-emptively request the published view.
    """
    from nodes.models import PreferredInstanceSource

    instance = empty_db_instance._create_from_config(source=PreferredInstanceSource.PUBLISHED)
    assert instance is not None
    assert instance.id == empty_db_instance.identifier


def test_published_source_uses_snapshot_after_publish(empty_db_instance: InstanceConfig):
    """
    After publish, draft edits are invisible to PUBLISHED readers.

    This is the observable shape of the draft/publish split: the snapshot
    captures state at publish time; subsequent draft writes don't leak to
    the published surface.
    """
    from nodes.defs.node_defs import NodeSpec as NodeSpecDef
    from nodes.defs.port_def import OutputPortDef
    from nodes.models import NodeConfig, PreferredInstanceSource
    from nodes.tests.test_model_editor import _port_uuid as _pu
    from nodes.units import unit_registry

    unit = unit_registry.parse_units('kt/a')
    NodeConfig.objects.create(
        instance=empty_db_instance,
        identifier='pub_baseline',
        name='Pub baseline',
        spec=NodeSpecDef(
            output_ports=[OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions')],
        ),
    )
    revision = empty_db_instance.save_revision(clean=False)
    empty_db_instance.publish(revision)
    empty_db_instance.refresh_from_db()

    NodeConfig.objects.create(
        instance=empty_db_instance,
        identifier='draft_only',
        name='Draft only',
        spec=NodeSpecDef(
            output_ports=[OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions')],
        ),
    )

    published = empty_db_instance._create_from_config(source=PreferredInstanceSource.PUBLISHED)
    draft = empty_db_instance._create_from_config(source=PreferredInstanceSource.DRAFT)

    assert 'pub_baseline' in published.context.nodes
    assert 'draft_only' not in published.context.nodes
    assert 'pub_baseline' in draft.context.nodes
    assert 'draft_only' in draft.context.nodes


def test_default_source_serves_draft_tables(empty_db_instance: InstanceConfig):
    """Backwards compat: no-arg _create_from_config keeps today's draft behavior."""
    from nodes.defs.node_defs import NodeSpec as NodeSpecDef
    from nodes.defs.port_def import OutputPortDef
    from nodes.models import NodeConfig
    from nodes.tests.test_model_editor import _port_uuid as _pu
    from nodes.units import unit_registry

    unit = unit_registry.parse_units('kt/a')
    NodeConfig.objects.create(
        instance=empty_db_instance,
        identifier='default_node',
        name='Default',
        spec=NodeSpecDef(
            output_ports=[OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions')],
        ),
    )

    instance = empty_db_instance._create_from_config()  # default source=DRAFT
    assert 'default_node' in instance.context.nodes


def test_directive_draft_preview_anon_rejected(client, empty_db_instance: InstanceConfig):
    """`@instance(preview: DRAFT)` from an anon caller fails with permission_denied."""
    from paths.tests.graphql import PathsTestClient

    tc = PathsTestClient(client)
    # No login — anonymous

    query = f"""
    query Q @instance(identifier: "{empty_db_instance.identifier}", preview: DRAFT) {{
        instance {{ id }}
    }}
    """
    errors = tc.query_errors(query)
    assert len(errors) >= 1
    codes = {(e.get('extensions') or {}).get('code') for e in errors}
    assert 'permission_denied' in codes


def test_resolve_preview_default_picks_published_when_revision_exists(empty_db_instance: InstanceConfig):
    """Default (no directive arg) serves PUBLISHED if the DB instance has been published."""
    from paths.schema_context import ActivateInstanceContextExtension

    # Fresh instance, no revision → default should fall back to DRAFT.
    ext = ActivateInstanceContextExtension.__new__(ActivateInstanceContextExtension)
    ctx = _make_fake_ctx(preview_mode=None, user=None)
    from nodes.models import PreferredInstanceSource

    assert ext._resolve_preview_source(empty_db_instance, ctx) == PreferredInstanceSource.DRAFT

    # Stamp a live revision (empty payload is fine; we're only testing the
    # source-selection branch, not hydration).
    revision = empty_db_instance.save_revision(clean=False)
    empty_db_instance.publish(revision)
    empty_db_instance.refresh_from_db()

    assert empty_db_instance.live_revision_id is not None
    assert ext._resolve_preview_source(empty_db_instance, ctx) == PreferredInstanceSource.PUBLISHED


def test_resolve_preview_yaml_source_always_draft(empty_db_instance: InstanceConfig):
    """Non-DB instances ignore the directive and always serve DRAFT without perm check."""
    from paths.schema import PreviewMode
    from paths.schema_context import ActivateInstanceContextExtension

    from nodes.models import PreferredInstanceSource

    empty_db_instance.config_source = 'yaml'
    empty_db_instance.save()

    ext = ActivateInstanceContextExtension.__new__(ActivateInstanceContextExtension)

    # All three directive values collapse to DRAFT for YAML sources, including
    # explicit DRAFT from an anonymous caller — no perm check fires.
    for mode in (None, PreviewMode.DRAFT, PreviewMode.PUBLISHED):
        ctx = _make_fake_ctx(preview_mode=mode, user=None)
        assert ext._resolve_preview_source(empty_db_instance, ctx) == PreferredInstanceSource.DRAFT


def _make_fake_ctx(*, preview_mode, user):
    """Minimal stand-in for PathsGraphQLContext that _resolve_preview_source uses."""

    class _FakeCtx:
        def __init__(self):
            self.preview_mode = preview_mode
            self._user = user

        def get_user(self):
            if self._user is not None:
                return self._user
            from django.contrib.auth.models import AnonymousUser

            return AnonymousUser()

    return _FakeCtx()


# ---------------------------------------------------------------------------
# Phase 0: published serving reads snapshot metadata, not live draft rows
# ---------------------------------------------------------------------------


def _make_publishable_node(ic: InstanceConfig, identifier: str, name: str, **kwargs):
    from nodes.defs.node_defs import NodeSpec as NodeSpecDef
    from nodes.defs.port_def import OutputPortDef
    from nodes.models import NodeConfig
    from nodes.tests.test_model_editor import _port_uuid as _pu
    from nodes.units import unit_registry

    return NodeConfig.objects.create(
        instance=ic,
        identifier=identifier,
        name=name,
        spec=NodeSpecDef(
            output_ports=[OutputPortDef(id=_pu('default'), unit=unit_registry.parse_units('kt/a'), quantity='emissions')],
        ),
        **kwargs,
    )


def test_published_metadata_ignores_draft_edits(empty_db_instance: InstanceConfig):
    """
    Draft metadata edits are invisible to PUBLISHED readers.

    Public resolvers read the selected NodeSnapshot. The live NodeConfig is
    attached only to the draft/editor runtime.
    """
    from nodes.models import NodeConfig, PreferredInstanceSource

    nc = _make_publishable_node(empty_db_instance, 'meta_node', 'Published name')
    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()

    NodeConfig.objects.filter(pk=nc.pk).update(name='Draft rename', is_visible=False, order=7)

    published = empty_db_instance._create_from_config(source=PreferredInstanceSource.PUBLISHED)
    node = published.context.nodes['meta_node']
    assert node.db_obj is None
    assert node.source_snapshot is not None
    assert str(node.source_snapshot.name) == 'Published name'
    assert node.source_snapshot.is_visible is True
    assert node.source_snapshot.order is None
    assert node.source_snapshot.uuid == nc.uuid
    # The node's spec comes from the revision snapshot, not the draft row.
    assert node.has_spec

    draft = empty_db_instance._create_from_config(source=PreferredInstanceSource.DRAFT)
    draft_node = draft.context.nodes['meta_node']
    assert draft_node.db_obj is not None
    assert draft_node.db_obj.pk == nc.pk
    assert draft_node.db_obj.name_i18n == 'Draft rename'
    assert draft_node.source_snapshot is not None
    assert str(draft_node.source_snapshot.name) == 'Draft rename'


def test_instance_graphql_content_comes_from_selected_snapshot(gql_client, empty_db_instance: InstanceConfig):
    """Instance content follows the selected snapshot; operational state remains live."""
    from nodes.models import _pytest_instances

    empty_db_instance.name = 'Published instance'
    empty_db_instance.owner = 'Published owner'
    empty_db_instance.lead_title = 'Published lead'
    empty_db_instance.lead_paragraph = '<p>Published paragraph</p>'
    assert empty_db_instance.spec is not None
    empty_db_instance.spec = empty_db_instance.spec.model_copy(update={'theme_identifier': 'published-theme'})
    empty_db_instance.save(
        update_fields=['name', 'owner', 'lead_title', 'lead_paragraph', 'spec'],
    )
    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()

    empty_db_instance.name = 'Draft instance'
    empty_db_instance.owner = 'Draft owner'
    empty_db_instance.lead_title = 'Draft lead'
    empty_db_instance.lead_paragraph = '<p>Draft paragraph</p>'
    assert empty_db_instance.spec is not None
    empty_db_instance.spec = empty_db_instance.spec.model_copy(update={'theme_identifier': 'draft-theme'})
    empty_db_instance.save(
        update_fields=['name', 'owner', 'lead_title', 'lead_paragraph', 'spec'],
    )
    _pytest_instances.pop(empty_db_instance.identifier, None)

    query = f"""
    query Q @instance(identifier: "{empty_db_instance.identifier}", preview: PUBLISHED) {{
        instance {{
            name
            owner
            leadTitle
            leadParagraph
            themeIdentifier
        }}
    }}
    """
    published = gql_client.query_data(query)['instance']
    assert published == {
        'name': 'Published instance',
        'owner': 'Published owner',
        'leadTitle': 'Published lead',
        'leadParagraph': '<p>Published paragraph</p>',
        'themeIdentifier': 'published-theme',
    }

    draft = gql_client.query_data(query.replace('preview: PUBLISHED', 'preview: DRAFT'))['instance']
    assert draft == {
        'name': 'Draft instance',
        'owner': 'Draft owner',
        'leadTitle': 'Draft lead',
        'leadParagraph': '<p>Draft paragraph</p>',
        'themeIdentifier': 'draft-theme',
    }


def test_published_body_survives_draft_edits(empty_db_instance: InstanceConfig):
    """StreamField body rides in the snapshot; draft body edits don't leak."""
    from nodes.models import PreferredInstanceSource

    nc = _make_publishable_node(empty_db_instance, 'body_node', 'Body node')
    nc.body = [{'type': 'paragraph', 'value': '<p>Published body</p>'}]
    nc.save(update_fields=['body'])
    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()

    nc.body = [{'type': 'paragraph', 'value': '<p>Draft body</p>'}]
    nc.save(update_fields=['body'])

    published = empty_db_instance._create_from_config(source=PreferredInstanceSource.PUBLISHED)
    node = published.context.nodes['body_node']
    assert node.db_obj is None
    assert node.source_snapshot is not None
    assert 'Published body' in str(node.source_snapshot.body)
    assert 'Draft body' not in str(node.source_snapshot.body)


def test_pre_v6_revision_content_is_completed_at_load_boundary(empty_db_instance: InstanceConfig):
    """Fields absent from old snapshots get their explicit compatibility fallback once."""
    from copy import deepcopy

    from wagtail.models import Revision

    from nodes.models import PreferredInstanceSource

    nc = _make_publishable_node(empty_db_instance, 'legacy_content', 'Legacy content')
    empty_db_instance.lead_title = 'Initially published lead'
    empty_db_instance.lead_paragraph = '<p>Initially published paragraph</p>'
    empty_db_instance.save(update_fields=['lead_title', 'lead_paragraph'])
    nc.body = [{'type': 'paragraph', 'value': '<p>Initially published body</p>'}]
    nc.save(update_fields=['body'])
    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()

    assert empty_db_instance.live_revision_id is not None
    live_revision = empty_db_instance.live_revision
    assert live_revision is not None
    content = deepcopy(live_revision.content)
    structured = content['model_snapshot']['structured']
    structured['schema_version'] = 5
    structured['metadata'].pop('lead_title')
    structured['metadata'].pop('lead_paragraph')
    structured['nodes'][0].pop('body')
    Revision.objects.filter(pk=empty_db_instance.live_revision_id).update(content=content)

    empty_db_instance.lead_title = 'Legacy fallback lead'
    empty_db_instance.lead_paragraph = '<p>Legacy fallback paragraph</p>'
    empty_db_instance.save(update_fields=['lead_title', 'lead_paragraph'])
    nc.body = [{'type': 'paragraph', 'value': '<p>Legacy fallback body</p>'}]
    nc.save(update_fields=['body'])
    empty_db_instance.refresh_from_db()

    published = empty_db_instance._create_from_config(source=PreferredInstanceSource.PUBLISHED)
    assert published.source_snapshot is not None
    assert str(published.source_snapshot.metadata.lead_title) == 'Legacy fallback lead'
    assert str(published.source_snapshot.metadata.lead_paragraph) == '<p>Legacy fallback paragraph</p>'
    node_snapshot = published.context.nodes['legacy_content'].source_snapshot
    assert node_snapshot is not None
    assert 'Legacy fallback body' in str(node_snapshot.body)


def test_published_indicator_node_uses_snapshot_reference(empty_db_instance: InstanceConfig):
    """Published indicator references remain UUID-pinned snapshot state."""
    from nodes.models import PreferredInstanceSource

    target = _make_publishable_node(empty_db_instance, 'indicator_target', 'Target')
    _make_publishable_node(empty_db_instance, 'pointing', 'Pointing', indicator_node=target)
    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()

    published = empty_db_instance._create_from_config(source=PreferredInstanceSource.PUBLISHED)
    pointing = published.context.nodes['pointing']
    indicator = published.context.nodes['indicator_target']
    assert pointing.db_obj is None
    assert pointing.source_snapshot is not None
    assert indicator.source_snapshot is not None
    assert pointing.source_snapshot.indicator_node == indicator.source_snapshot.uuid
    assert published.source_nodes_by_uuid[indicator.source_snapshot.uuid] is indicator


def test_published_editor_and_history_guarded(gql_client, empty_db_instance: InstanceConfig):
    """Editor fields and change history are draft-row governance: absent on PUBLISHED."""
    from nodes.models import _pytest_instances

    _make_publishable_node(empty_db_instance, 'guarded', 'Guarded')
    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()

    # The factory registers a pre-built empty Instance that short-circuits
    # request-path hydration; drop it so the query hydrates from the DB.
    _pytest_instances.pop(empty_db_instance.identifier, None)

    query = f"""
    query Q @instance(identifier: "{empty_db_instance.identifier}", preview: PUBLISHED) {{
        nodes {{
            identifier
            name
            editor {{ nodeType }}
            changeHistory {{ uuid }}
        }}
    }}
    """
    data = gql_client.query_data(query)
    (node_data,) = [n for n in data['nodes'] if n['identifier'] == 'guarded']
    assert node_data['name'] == 'Guarded'
    assert node_data['editor'] is None
    assert node_data['changeHistory'] == []

    # Same query on DRAFT: the superuser gets editor fields from live rows.
    draft_query = query.replace('preview: PUBLISHED', 'preview: DRAFT')
    data = gql_client.query_data(draft_query)
    (node_data,) = [n for n in data['nodes'] if n['identifier'] == 'guarded']
    assert node_data['editor'] is not None


def test_publish_instance_bumps_cache_invalidated_at(empty_db_instance: InstanceConfig):
    """Publishing must invalidate cached anonymous GraphQL results."""
    before = empty_db_instance.cache_invalidated_at
    empty_db_instance.publish_instance()
    empty_db_instance.refresh_from_db()
    assert empty_db_instance.cache_invalidated_at > before
