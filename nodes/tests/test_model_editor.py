"""Tests for the model editor GraphQL mutations (create/update/delete node, edge)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from nodes.defs.instance_defs import InstanceSpec, YearsSpec
from nodes.defs.node_defs import NodeSpec, OutputMetricDef, SimpleConfig
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory

if TYPE_CHECKING:
    from paths.tests.graphql import PathsTestClient

    from nodes.context import Context
    from nodes.models import InstanceConfig


pytestmark = pytest.mark.django_db

# Node class that InstanceLoader can import for roundtrip tests
SIMPLE_NODE_CLASS = 'nodes.simple.SimpleNode'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node_spec(**overrides: Any) -> NodeSpec:
    """Create a NodeSpec with a real node_class so InstanceLoader can hydrate it."""
    defaults: dict[str, Any] = {
        'node_class': SIMPLE_NODE_CLASS,
        'type_config': SimpleConfig(),
        'output_metrics': [OutputMetricDef(id='default', unit='kt/a', quantity='emissions')],
    }
    defaults.update(overrides)
    return NodeSpec(**defaults)


def _rebuild_from_db(ic: InstanceConfig) -> Context:
    """Refresh IC from DB, bypass test cache, and rebuild the runtime Instance via InstanceLoader."""
    from nodes.models import _pytest_instances

    ic.refresh_from_db()
    # Temporarily remove from test cache so _create_from_config() goes through the DB path
    cached = _pytest_instances.pop(ic.identifier, None)
    try:
        instance = ic._create_from_config()
    finally:
        # Restore cache to avoid breaking other fixtures
        if cached is not None:
            _pytest_instances[ic.identifier] = cached
    return instance.context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_instance_config() -> InstanceConfig:
    """Create an InstanceConfig with config_source='database' and valid years."""
    instance = InstanceFactory.create()
    spec = InstanceSpec(
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
    )
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        spec=spec,
    )


@pytest.fixture
def gql(client, db_instance_config: InstanceConfig) -> PathsTestClient:
    """Return a PathsTestClient wired to the db_instance_config."""
    from paths.tests.graphql import PathsTestClient

    tc = PathsTestClient(client)
    tc.set_instance(db_instance_config)
    return tc


# ---------------------------------------------------------------------------
# create_node
# ---------------------------------------------------------------------------

CREATE_NODE = """
mutation CreateNode($input: CreateNodeInput!) {
    createNode(input: $input) {
        ok
        node {
            identifier
            name
            color
            isVisible
            spec {
                isOutcome
                kind
            }
        }
    }
}
"""


def test_create_node_formula(gql: PathsTestClient, db_instance_config: InstanceConfig):
    data = gql.query_data(
        CREATE_NODE,
        variables={
            'input': {
                'instanceId': str(db_instance_config.pk),
                'identifier': 'new_node',
                'name': 'New Node',
                'nodeType': 'formula',
                'color': '#ff0000',
                'isOutcome': True,
            }
        },
    )
    node = data['createNode']['node']
    assert node['identifier'] == 'new_node'
    assert node['name'] == 'New Node'
    assert node['color'] == '#ff0000'
    assert node['isVisible'] is True
    assert node['spec']['isOutcome'] is True
    assert node['spec']['kind'] == 'formula'


def test_create_node_simple(gql: PathsTestClient, db_instance_config: InstanceConfig):
    data = gql.query_data(
        CREATE_NODE,
        variables={
            'input': {
                'instanceId': str(db_instance_config.pk),
                'identifier': 'simple_node',
                'nodeType': 'simple',
            }
        },
    )
    node = data['createNode']['node']
    assert node['identifier'] == 'simple_node'
    assert node['spec']['kind'] == 'simple'
    assert node['spec']['isOutcome'] is False


def test_create_node_rejects_yaml_instance(gql: PathsTestClient, db_instance_config: InstanceConfig):
    """YAML-sourced instances cannot be edited via mutations."""
    # Switch to yaml to test rejection
    db_instance_config.config_source = 'yaml'
    db_instance_config.save(update_fields=['config_source'])

    errors = gql.query_errors(
        CREATE_NODE,
        variables={
            'input': {
                'instanceId': str(db_instance_config.pk),
                'identifier': 'nope',
            }
        },
    )
    error = errors[0]
    assert 'message' in error
    assert 'YAML' in error['message']


# ---------------------------------------------------------------------------
# update_node
# ---------------------------------------------------------------------------

UPDATE_NODE = """
mutation UpdateNode($input: UpdateNodeInput!) {
    updateNode(input: $input) {
        ok
        node {
            identifier
            name
            color
            isVisible
            spec {
                isOutcome
                kind
                formula
            }
        }
    }
}
"""


def test_update_node_direct_fields(gql: PathsTestClient, db_instance_config: InstanceConfig):
    nc = NodeConfigFactory.create(instance=db_instance_config, identifier='editable', name='Old', color='#000')

    data = gql.query_data(
        UPDATE_NODE,
        variables={
            'input': {
                'nodeId': str(nc.pk),
                'name': 'Updated',
                'color': '#00ff00',
                'isVisible': False,
            }
        },
    )
    node = data['updateNode']['node']
    assert node['name'] == 'Updated'
    assert node['color'] == '#00ff00'
    assert node['isVisible'] is False


def test_update_node_spec_fields(gql: PathsTestClient, db_instance_config: InstanceConfig):
    nc = NodeConfigFactory.create(instance=db_instance_config, identifier='spec_edit')

    data = gql.query_data(
        UPDATE_NODE,
        variables={
            'input': {
                'nodeId': str(nc.pk),
                'isOutcome': True,
                'formula': 'a + b',
            }
        },
    )
    node = data['updateNode']['node']
    assert node['spec']['isOutcome'] is True
    assert node['spec']['kind'] == 'formula'
    assert node['spec']['formula'] == 'a + b'


def test_update_node_not_found(gql: PathsTestClient, db_instance_config: InstanceConfig):  # pyright: ignore[reportUnusedParameter]
    errors = gql.query_errors(
        UPDATE_NODE,
        variables={'input': {'nodeId': '999999'}},
    )
    error = errors[0]
    assert 'message' in error
    assert 'not found' in error['message'].lower()


# ---------------------------------------------------------------------------
# delete_node
# ---------------------------------------------------------------------------

DELETE_NODE = """
mutation DeleteNode($nodeId: ID!) {
    deleteNode(nodeId: $nodeId) {
        ok
    }
}
"""


def test_delete_node(gql: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeConfig

    nc = NodeConfigFactory.create(instance=db_instance_config, identifier='doomed')
    node_pk = nc.pk

    data = gql.query_data(DELETE_NODE, variables={'nodeId': str(node_pk)})
    assert data['deleteNode']['ok'] is True
    assert not NodeConfig.objects.filter(pk=node_pk).exists()


# ---------------------------------------------------------------------------
# create_edge / delete_edge
# ---------------------------------------------------------------------------

CREATE_EDGE = """
mutation CreateEdge($input: CreateEdgeInput!) {
    createEdge(input: $input) {
        ok
        edge {
            fromNodeId
            toNodeId
            fromPort
            toPort
        }
    }
}
"""

DELETE_EDGE = """
mutation DeleteEdge($edgeId: ID!) {
    deleteEdge(edgeId: $edgeId) {
        ok
    }
}
"""


def test_create_and_delete_edge(gql: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeEdge

    nc_a = NodeConfigFactory.create(instance=db_instance_config, identifier='node_a')
    nc_b = NodeConfigFactory.create(instance=db_instance_config, identifier='node_b')

    # Create
    data = gql.query_data(
        CREATE_EDGE,
        variables={
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromNodeId': 'node_a',
                'toNodeId': 'node_b',
            }
        },
    )
    edge = data['createEdge']['edge']
    assert edge['fromNodeId'] == 'node_a'
    assert edge['toNodeId'] == 'node_b'
    assert edge['toPort'] == 'from_node_a'

    # Delete
    edge_obj = NodeEdge.objects.get(instance=db_instance_config, from_node=nc_a, to_node=nc_b)
    data = gql.query_data(DELETE_EDGE, variables={'edgeId': str(edge_obj.pk)})
    assert data['deleteEdge']['ok'] is True
    assert not NodeEdge.objects.filter(pk=edge_obj.pk).exists()


# ---------------------------------------------------------------------------
# model_instance query (smoke test)
# ---------------------------------------------------------------------------

MODEL_INSTANCE_QUERY = """
query ModelInstance($id: ID!) {
    modelInstance(instanceId: $id) {
        identifier
        configSource
        spec {
            years {
                reference
                target
            }
        }
        nodes {
            identifier
            spec {
                kind
            }
        }
        edges {
            fromNodeId
            toNodeId
        }
    }
}
"""


def test_model_instance_query(gql: PathsTestClient, db_instance_config: InstanceConfig):
    NodeConfigFactory.create(instance=db_instance_config, identifier='queried_node')

    data = gql.query_data(
        MODEL_INSTANCE_QUERY,
        variables={'id': str(db_instance_config.pk)},
    )
    mi = data['modelInstance']
    assert mi['identifier'] == db_instance_config.identifier
    assert mi['configSource'] == 'database'
    node_ids = [n['identifier'] for n in mi['nodes']]
    assert 'queried_node' in node_ids


# ---------------------------------------------------------------------------
# Roundtrip: mutation → DB → runtime Context
# ---------------------------------------------------------------------------


def test_update_node_roundtrip(gql: PathsTestClient, db_instance_config: InstanceConfig):
    """After updating a node via mutation, rebuilding Instance from DB reflects the changes."""
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='rt_node',
        name='Before',
        color='#000000',
        spec=_make_node_spec(),
    )

    gql.query_data(
        UPDATE_NODE,
        variables={
            'input': {
                'nodeId': str(nc.pk),
                'name': 'After',
                'color': '#abcdef',
                'isVisible': False,
                'isOutcome': True,
            }
        },
    )

    ctx = _rebuild_from_db(db_instance_config)
    rt_node = ctx.nodes['rt_node']
    assert str(rt_node.name) == 'After'
    assert rt_node.color == '#abcdef'
    assert rt_node.is_outcome is True


def test_create_edge_roundtrip(gql: PathsTestClient, db_instance_config: InstanceConfig):
    """After creating an edge via mutation, the runtime graph reflects the connection."""
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='src',
        name='Source',
        spec=_make_node_spec(),
    )
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='dst',
        name='Destination',
        spec=_make_node_spec(),
    )

    gql.query_data(
        CREATE_EDGE,
        variables={
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromNodeId': 'src',
                'toNodeId': 'dst',
            }
        },
    )

    ctx = _rebuild_from_db(db_instance_config)
    src = ctx.nodes['src']
    dst = ctx.nodes['dst']
    assert dst in src.output_nodes
    assert src in dst.input_nodes


def test_delete_node_roundtrip(gql: PathsTestClient, db_instance_config: InstanceConfig):
    """After deleting a node, the rebuilt runtime graph no longer contains it."""
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='ephemeral',
        name='Gone Soon',
        spec=_make_node_spec(),
    )

    gql.query_data(DELETE_NODE, variables={'nodeId': str(nc.pk)})

    ctx = _rebuild_from_db(db_instance_config)
    assert 'ephemeral' not in ctx.nodes
