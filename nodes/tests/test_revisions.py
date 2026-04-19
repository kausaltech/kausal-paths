"""Tests for the draft/publish/revisions machinery."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest

from nodes.defs.instance_defs import InstanceSpec, YearsSpec
from nodes.instance_serialization import (
    SNAPSHOT_SCHEMA_VERSION,
    DatasetPortSnapshot,
    EdgeSnapshot,
    InstanceSnapshot,
    NodeSnapshot,
    build_instance_snapshot,
)
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


pytestmark = pytest.mark.django_db


@pytest.fixture
def empty_db_instance() -> InstanceConfig:
    """Bare DB-sourced InstanceConfig, no nodes/edges."""
    instance = InstanceFactory.create()
    spec = InstanceSpec(
        primary_language='en',
        owner='Test Owner',
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
    )
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        spec=spec,
    )


# ---------------------------------------------------------------------------
# Pydantic round-trip
# ---------------------------------------------------------------------------


def test_instance_snapshot_json_round_trip():
    """A minimal snapshot dumps to JSON and reloads to an equal object."""
    from kausal_common.i18n.pydantic import TranslatedString

    spec = InstanceSpec(
        primary_language='en',
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
    )
    snap = InstanceSnapshot(
        spec=spec,
        nodes=[
            NodeSnapshot(
                identifier='n1',
                name=TranslatedString('Node 1', default_language='en'),
                color='#abc',
                is_visible=True,
            )
        ],
        edges=[
            EdgeSnapshot(
                from_node='n1',
                to_node='n2',
                from_port=uuid.UUID('33191571-e9c8-45ac-b624-cc0a04341d37'),
                to_port=uuid.UUID('796076a8-426b-4068-ac57-e3e333d0ef0a'),
            )
        ],
        dataset_ports=[
            DatasetPortSnapshot(
                node='n1', dataset='ds', port_id=uuid.UUID('6c8b0551-7ccf-472b-94db-26f513d706dc'), metric='m', forecast_from=2025
            )
        ],
    )
    dumped = snap.model_dump(mode='json')
    assert dumped['schema_version'] == SNAPSHOT_SCHEMA_VERSION

    reloaded = InstanceSnapshot.model_validate(dumped)
    assert reloaded.nodes[0].identifier == 'n1'
    assert reloaded.edges[0].from_node == 'n1'
    assert reloaded.dataset_ports[0].forecast_from == 2025
    assert reloaded.schema_version == SNAPSHOT_SCHEMA_VERSION


def test_i18n_spec_assignment_stays_dict_serializable():
    """Post-validation i18n field assignment must not reintroduce compact strings."""
    from kausal_common.i18n.pydantic import TranslatedString, set_i18n_context

    from nodes.defs.node_defs import NodeSpec

    with set_i18n_context('en', []):
        spec = NodeSpec(name='Original')
        spec.name = 'Renamed'
        copied = spec.model_copy(update={'name': 'Copied'})

    snap = NodeSnapshot(
        identifier='n1',
        name=TranslatedString(en='Renamed'),
        color='#abc',
        is_visible=True,
        spec=spec,
    )

    dumped = snap.model_dump(mode='json')
    assert dumped['spec']['name'] == {'en': 'Renamed'}
    assert copied.model_dump(mode='json')['name'] == {'en': 'Copied'}


def test_instance_snapshot_schema_version_default():
    """New snapshots carry the current schema version."""
    spec = InstanceSpec(primary_language='en', years=YearsSpec(target=2030))
    snap = InstanceSnapshot(spec=spec)
    assert snap.schema_version == SNAPSHOT_SCHEMA_VERSION


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


# ---------------------------------------------------------------------------
# serializable_data includes the snapshot
# ---------------------------------------------------------------------------


def test_serializable_data_includes_structured_snapshot(empty_db_instance: InstanceConfig):
    data = empty_db_instance.serializable_data()
    assert 'model_snapshot' in data
    ms = data['model_snapshot']
    assert ms['schema_version'] == SNAPSHOT_SCHEMA_VERSION
    assert 'structured' in ms
    assert 'hydrate_dict' in ms
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
    from nodes.defs.node_defs import NodeKind, NodeSpec as NodeSpecDef
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
            kind=NodeKind.FORMULA,
            output_ports=[OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions')],
        ),
    )
    nc_b = NodeConfig.objects.create(
        instance=empty_db_instance,
        identifier='dst',
        name='Dst',
        spec=NodeSpecDef(
            kind=NodeKind.FORMULA,
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
    assert edge_entry.data['before']['from_node'] == 'src'

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


def test_dataset_save_revision_updates_latest_revision():
    from kausal_common.datasets.tests.factories import DatasetFactory

    ds = DatasetFactory.create()
    assert ds.latest_revision_id is None
    ds.save_revision()
    ds.refresh_from_db()
    assert ds.latest_revision_id is not None


def test_dataset_port_snapshot_pins_dataset_revision(empty_db_instance: InstanceConfig):
    from kausal_common.datasets.tests.factories import DatasetFactory, DatasetMetricFactory

    from nodes.models import DatasetPort, NodeConfig

    ds = DatasetFactory.create()
    metric = DatasetMetricFactory.create(schema=ds.schema, name='m1', label='M', unit='kt/a')
    ds.save_revision()
    ds.refresh_from_db()
    pinned_rev = ds.latest_revision_id

    nc = NodeConfig.objects.create(instance=empty_db_instance, identifier='owner', name='Owner')
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
            ... on NodeEdgeType { fromRef { nodeId portId } toRef { nodeId portId } }
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
    from nodes.defs.node_defs import NodeKind, NodeSpec as NodeSpecDef
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
            kind=NodeKind.FORMULA,
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
            kind=NodeKind.FORMULA,
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
    assert entries[0].data['after']['from_node'] == src.identifier
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
    assert entry.data['before']['from_node'] == src.identifier
    assert entry.data['after'] is None


def test_create_edge_auto_creates_matching_target_port(
    gql_client,
    empty_db_instance: InstanceConfig,
):
    """When toPort is null on a multi-port target, a new input port is created."""
    from nodes.defs.node_defs import NodeKind, NodeSpec as NodeSpecDef
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
            kind=NodeKind.FORMULA,
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
            kind=NodeKind.FORMULA,
            input_ports=[existing_port_a, existing_port_b],
            output_ports=[OutputPortDef(id=_pu('default'), unit=unit, quantity='emissions')],
        ),
    )
    # Pre-bind both existing ports so the resolver can't reuse them.
    other = _make_node(
        'auto_other',
        NodeSpecDef(
            kind=NodeKind.FORMULA,
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
