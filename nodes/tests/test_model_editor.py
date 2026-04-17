"""Tests for the model editor GraphQL mutations (create/update/delete node, edge)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from django.db import connection
from django.test.utils import CaptureQueriesContext

import pytest

from kausal_common.datasets.tests.factories import DatasetFactory, DatasetMetricFactory

from nodes.actions.parent import ParentActionNode
from nodes.constants import DecisionLevel
from nodes.defs.action_def import ImpactGraphType, ImpactOverviewSpec
from nodes.defs.instance_defs import ActionGroup, InstanceSpec, NormalizationSpec, YearsSpec
from nodes.defs.node_defs import ActionConfig, NodeKind, NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory, _port_id
from nodes.units import unit_registry

if TYPE_CHECKING:
    from uuid import UUID

    from paths.tests.graphql import PathsTestClient

    from nodes.actions.action import ActionNode
    from nodes.context import Context
    from nodes.models import InstanceConfig


# This way GraphQL LSPs recognize the query strings as GraphQL
gql = str


pytestmark = pytest.mark.django_db

# Node class that InstanceLoader can import for roundtrip tests
SIMPLE_NODE_CLASS = 'nodes.simple.SimpleNode'

ACTION_NODE_CLASS = 'nodes.actions.simple.AdditiveAction'
PARENT_ACTION_NODE_CLASS = 'nodes.tests.test_model_editor.ModelEditorParentActionNode'


class ModelEditorParentActionNode(ParentActionNode):
    pass


def _port_uuid(name: str) -> UUID:
    """Generate a deterministic port UUID matching the NodeConfigFactory convention."""
    return _port_id(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node_spec(**overrides: Any) -> NodeSpec:
    """Create a NodeSpec with a real node_class so InstanceLoader can hydrate it."""
    unit = unit_registry.parse_units('kt/a')
    defaults: dict[str, Any] = {
        'kind': NodeKind.SIMPLE,
        'type_config': SimpleConfig(node_class=SIMPLE_NODE_CLASS),
        'output_ports': [OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')],
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


@pytest.fixture
def gql_client(client, db_instance_config: InstanceConfig) -> PathsTestClient:
    """Return a PathsTestClient wired to the db_instance_config, authenticated as superuser."""
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    user = UserFactory.create(is_superuser=True)
    client.force_login(user)
    tc = PathsTestClient(client)
    tc.set_instance(db_instance_config)
    return tc


# ---------------------------------------------------------------------------
# create_node
# ---------------------------------------------------------------------------

CREATE_NODE = """
mutation CreateNode($instanceId: ID!, $input: CreateNodeInput!) {
    instanceEditor(instanceId: $instanceId) {
        createNode(input: $input) {
            ... on NodeInterface {
                identifier
                name
                color
                kind
                isVisible
                editor {
                    nodeGroup
                    spec {
                        outputPorts {
                            id
                            quantity
                            dimensions
                            unit {
                                standard
                            }
                        }
                        typeConfig {
                            __typename
                            ... on ActionConfigType {
                                nodeClass
                                group
                                noEffectValue
                            }
                            ... on SimpleConfigType {
                                nodeClass
                            }
                            ... on FormulaConfigType {
                                formula
                            }
                        }
                    }
                }
            }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""


def test_create_node_formula(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    data = gql_client.query_data(
        CREATE_NODE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'identifier': 'new_node',
                'name': 'New Node',
                'config': {'formula': {'formula': 'a + b'}},
                'color': '#ff0000',
                'isOutcome': True,
                'outputPorts': [{'unit': 'kt/a', 'quantity': 'emissions'}],
            },
        },
    )
    node = data['instanceEditor']['createNode']
    assert node['identifier'] == 'new_node'
    assert node['name'] == 'New Node'
    assert node['color'] == '#ff0000'
    assert node['isVisible'] is True
    assert node['kind'] == 'FORMULA'
    assert node['editor']['spec']['outputPorts'][0]['quantity'] == 'emissions'


def test_create_node_simple(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    data = gql_client.query_data(
        CREATE_NODE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'identifier': 'simple_node',
                'name': 'Simple Node',
                'kind': 'SIMPLE',
                'config': {'simple': {'nodeClass': SIMPLE_NODE_CLASS}},
                'color': '#000000',
                'outputPorts': [{'unit': 'kt/a', 'quantity': 'emissions'}],
            },
        },
    )
    node = data['instanceEditor']['createNode']
    assert node['identifier'] == 'simple_node'
    assert node['kind'] == 'SIMPLE'
    assert node['editor']['spec']['typeConfig']['nodeClass'] == SIMPLE_NODE_CLASS


def test_create_node_with_node_group_and_allow_nulls(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    data = gql_client.query_data(
        CREATE_NODE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'identifier': 'grouped_node',
                'name': 'Grouped Node',
                'kind': 'SIMPLE',
                'config': {'simple': {'nodeClass': SIMPLE_NODE_CLASS}},
                'color': '#000000',
                'nodeGroup': 'transport',
                'allowNulls': True,
                'outputPorts': [{'unit': 'kt/a', 'quantity': 'emissions'}],
            },
        },
    )
    node = data['instanceEditor']['createNode']
    assert node['editor']['nodeGroup'] == 'transport'


def test_create_node_action_with_aarhus_style_fields(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    assert db_instance_config.spec is not None
    db_instance_config.spec.action_groups = [ActionGroup(id='energy', name='Energy')]
    db_instance_config.spec.dimensions = [
        {'id': dim_id, 'label': dim_id.replace('_', ' ').title(), 'categories': []}
        for dim_id in ['energy_carrier', 'energy_usage', 'cost_type', 'sector', 'ghg']
    ]
    db_instance_config.save(update_fields=['spec'])

    data = gql_client.query_data(
        CREATE_NODE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'identifier': 'carbon_capture_and_storage',
                'name': 'Carbon Capture and Storage',
                'kind': 'ACTION',
                'config': {
                    'action': {
                        'nodeClass': ACTION_NODE_CLASS,
                        'group': 'energy',
                        'noEffectValue': 0.0,
                    },
                },
                'inputDimensions': ['energy_carrier', 'energy_usage', 'cost_type', 'sector', 'ghg'],
                'outputDimensions': ['energy_carrier', 'energy_usage', 'cost_type', 'sector', 'ghg'],
                'inputDatasets': [
                    {
                        'id': 'aarhus/energy_actions',
                        'forecast_from': 2024,
                        'filters': [{'column': 'action', 'value': 'carbon_capture_and_storage'}],
                    }
                ],
                'params': {'allow_null_categories': True},
                'outputMetrics': [
                    {'id': 'emissions', 'unit': 't/a', 'quantity': 'emissions'},
                    {'id': 'energy', 'unit': 'TJ/a', 'quantity': 'energy'},
                    {'id': 'currency', 'unit': 'DKK/a', 'quantity': 'currency'},
                ],
            },
        },
    )

    node = data['instanceEditor']['createNode']
    assert node['identifier'] == 'carbon_capture_and_storage'
    assert node['kind'] == 'ACTION'
    assert node['editor']['spec']['typeConfig']['nodeClass'] == ACTION_NODE_CLASS
    assert node['editor']['spec']['typeConfig']['group'] == 'energy'
    assert [port['quantity'] for port in node['editor']['spec']['outputPorts']] == ['emissions', 'energy', 'currency']

    nc = db_instance_config.nodes.get(identifier='carbon_capture_and_storage')
    assert nc.spec is not None
    assert nc.spec.kind == NodeKind.ACTION
    assert isinstance(nc.spec.type_config, ActionConfig)
    assert nc.spec.type_config.node_class == ACTION_NODE_CLASS
    assert nc.spec.type_config.group == 'energy'
    assert nc.spec.input_dimensions == ['energy_carrier', 'energy_usage', 'cost_type', 'sector', 'ghg']
    assert nc.spec.output_dimensions == ['energy_carrier', 'energy_usage', 'cost_type', 'sector', 'ghg']
    assert nc.spec.input_datasets[0].id == 'aarhus/energy_actions'
    assert nc.spec.input_datasets[0].forecast_from == 2024
    assert [port.column_id for port in nc.spec.output_ports] == ['emissions', 'energy', 'currency']
    allow_null_categories = next(param for param in nc.spec.params if param.local_id == 'allow_null_categories')
    assert allow_null_categories.value is True


def test_create_node_rejects_yaml_instance(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """YAML-sourced instances cannot be edited via mutations."""
    # Switch to yaml to test rejection
    db_instance_config.config_source = 'yaml'
    db_instance_config.save(update_fields=['config_source'])

    errors = gql_client.query_errors(
        CREATE_NODE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'identifier': 'nope',
                'name': 'Nope',
                'config': {'simple': {'nodeClass': SIMPLE_NODE_CLASS}},
            },
        },
    )
    error = errors[0]
    assert 'message' in error


# ---------------------------------------------------------------------------
# update_node
# ---------------------------------------------------------------------------

UPDATE_NODE = gql("""
mutation UpdateNode($instanceId: ID!, $nodeId: ID!, $input: UpdateNodeInput!) {
    instanceEditor(instanceId: $instanceId) {
        updateNode(nodeId: $nodeId, input: $input) {
            ... on Node {
                identifier
                name
                color
                kind
                isVisible
                editor {
                    nodeGroup
                }
            }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
""")


def test_update_node_direct_fields(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    nc = NodeConfigFactory.create(instance=db_instance_config, identifier='editable', name='Old', color='#000')

    data = gql_client.query_data(
        UPDATE_NODE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': str(nc.pk),
            'input': {
                'name': 'Updated',
                'color': '#00ff00',
                'isVisible': False,
            },
        },
    )
    node = data['instanceEditor']['updateNode']
    assert node['name'] == 'Updated'
    assert node['color'] == '#00ff00'
    assert node['isVisible'] is False


def test_runtime_rebuild_preserves_node_group_and_allow_nulls(db_instance_config: InstanceConfig):
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='runtime_simple',
        name='Runtime Simple',
        spec=_make_node_spec(node_group='transport', allow_nulls=True),
    )

    ctx = _rebuild_from_db(db_instance_config)
    node = ctx.nodes['runtime_simple']
    assert node.node_group == 'transport'
    assert node.allow_nulls is True


def test_instance_spec_syncs_identity_fields(db_instance_config: InstanceConfig):
    db_instance_config.refresh_from_db()
    assert db_instance_config.spec is not None
    assert db_instance_config.spec.uuid == db_instance_config.uuid
    assert db_instance_config.spec.identifier == db_instance_config.identifier
    assert str(db_instance_config.spec.name) == db_instance_config.name


def test_node_spec_syncs_identity_fields_on_save(db_instance_config: InstanceConfig):
    nc = NodeConfigFactory.create(instance=db_instance_config, identifier='spec_identity', name='Spec Identity')
    nc.refresh_from_db()
    assert nc.spec is not None
    assert nc.spec.uuid == nc.uuid
    assert nc.spec.identifier == nc.identifier
    assert str(nc.spec.name) == nc.name


def test_node_spec_syncs_display_fields_on_save(db_instance_config: InstanceConfig):
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='spec_display',
        name='Spec Display',
        color='#123456',
        order=7,
        is_visible=False,
    )
    nc.refresh_from_db()
    assert nc.spec is not None
    assert nc.spec.color == '#123456'
    assert nc.spec.order == 7
    assert nc.spec.is_visible is False


def test_runtime_rebuild_preserves_action_group_and_zero_no_effect_value(db_instance_config: InstanceConfig):
    assert db_instance_config.spec is not None
    db_instance_config.spec.action_groups = [ActionGroup(id='grp', name='Group')]
    db_instance_config.save(update_fields=['spec'])

    unit = unit_registry.parse_units('kt/a')
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='runtime_action',
        name='Runtime Action',
        spec=NodeSpec(
            kind=NodeKind.ACTION,
            type_config=ActionConfig(
                node_class=ACTION_NODE_CLASS,
                decision_level=DecisionLevel.MUNICIPALITY,
                group='grp',
                no_effect_value=0.0,
            ),
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')],
        ),
    )

    ctx = _rebuild_from_db(db_instance_config)
    action = cast('ActionNode', ctx.nodes['runtime_action'])
    assert action.group is not None
    assert action.group.id == 'grp'
    assert action.no_effect_value == 0.0


def test_runtime_rebuild_preserves_action_parent_link(db_instance_config: InstanceConfig):
    unit = unit_registry.parse_units('kt/a')
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='parent_action',
        name='Parent Action',
        spec=NodeSpec(
            kind=NodeKind.ACTION,
            type_config=ActionConfig(node_class=PARENT_ACTION_NODE_CLASS, decision_level=DecisionLevel.MUNICIPALITY),
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')],
        ),
    )
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='child_action',
        name='Child Action',
        spec=NodeSpec(
            kind=NodeKind.ACTION,
            type_config=ActionConfig(
                node_class=ACTION_NODE_CLASS, decision_level=DecisionLevel.MUNICIPALITY, parent='parent_action'
            ),
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')],
        ),
    )

    ctx = _rebuild_from_db(db_instance_config)
    parent = cast('ModelEditorParentActionNode', ctx.nodes['parent_action'])
    child = cast('ActionNode', ctx.nodes['child_action'])
    assert child.parent_action is parent
    assert child in parent.subactions


def test_runtime_rebuild_preserves_impact_overviews(db_instance_config: InstanceConfig):
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='impact_effect',
        name='Impact Effect',
        spec=_make_node_spec(),
    )
    assert db_instance_config.spec is not None
    db_instance_config.spec.impact_overviews = [
        ImpactOverviewSpec.model_validate({
            'graph_type': ImpactGraphType.SIMPLE_EFFECT,
            'effect_node_id': 'impact_effect',
            'indicator_unit': 'kt/a',
        })
    ]
    db_instance_config.save(update_fields=['spec'])

    ctx = _rebuild_from_db(db_instance_config)
    assert len(ctx.impact_overviews) == 1
    overview = ctx.impact_overviews[0]
    assert overview.spec.graph_type == ImpactGraphType.SIMPLE_EFFECT
    assert overview.effect_node.id == 'impact_effect'


def test_runtime_rebuild_preserves_normalizations(db_instance_config: InstanceConfig):
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='population_normalizer',
        name='Population',
        spec=_make_node_spec(),
    )
    assert db_instance_config.spec is not None
    db_instance_config.spec.normalizations = [
        NormalizationSpec.model_validate({
            'normalizer_node_id': 'population_normalizer',
            'quantities': [{'id': 'energy', 'unit': 'kWh/cap/a'}],
            'default': True,
        })
    ]
    db_instance_config.save(update_fields=['spec'])

    ctx = _rebuild_from_db(db_instance_config)
    normalization = ctx.normalizations['population_normalizer']
    assert normalization.normalizer_node.id == 'population_normalizer'
    assert normalization.spec.default is True
    assert ctx.default_normalization is normalization


def test_update_node_not_found(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    errors = gql_client.query_errors(
        UPDATE_NODE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': '999999',
            'input': {'name': 'ghost'},
        },
    )
    error = errors[0]
    assert 'message' in error


# ---------------------------------------------------------------------------
# delete_node
# ---------------------------------------------------------------------------

DELETE_NODE = gql("""
mutation DeleteNode($instanceId: ID!, $nodeId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        deleteNode(nodeId: $nodeId) {
            messages { kind message }
        }
    }
}
""")


def test_delete_node(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeConfig

    nc = NodeConfigFactory.create(instance=db_instance_config, identifier='doomed')
    node_pk = nc.pk

    gql_client.query_data(
        DELETE_NODE,
        variables={'instanceId': str(db_instance_config.pk), 'nodeId': str(node_pk)},
    )
    # query_data asserts no errors; just verify the node is gone
    assert not NodeConfig.objects.filter(pk=node_pk).exists()


# ---------------------------------------------------------------------------
# create_edge / delete_edge
# ---------------------------------------------------------------------------

CREATE_EDGE = gql("""
mutation CreateEdge($instanceId: ID!, $input: CreateEdgeInput!) {
    instanceEditor(instanceId: $instanceId) {
        createEdge(input: $input) {
            __typename
            ... on OperationInfo { messages { kind message } }
            ... on NodeEdgeType {
                fromRef {
                    nodeId
                    portId
                }
                toRef {
                    nodeId
                    portId
                }
                transformations {
                    __typename
                    ... on SelectCategoriesTransformationType {
                        dimension categories flatten exclude
                    }
                    ... on AssignCategoryTransformationType {
                        dimension category
                    }
                    ... on FlattenTransformationType {
                        dimension
                    }
                }
                tags
            }
        }
    }
}
""")

DELETE_EDGE = gql("""
mutation DeleteEdge($instanceId: ID!, $edgeId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        deleteEdge(edgeId: $edgeId) {
            messages { kind message }
        }
    }
}
""")


def test_create_and_delete_edge(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeEdge

    unit = unit_registry.parse_units('kt/a')
    nc_a = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='node_a',
        spec=_make_node_spec(output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')]),
    )
    nc_b = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='node_b',
        spec=_make_node_spec(
            input_ports=[InputPortDef(id=_port_uuid('input'), unit=unit, quantity='emissions')],
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')],
        ),
    )

    # Create
    data = gql_client.query_data(
        CREATE_EDGE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromNodeId': 'node_a',
                'toNodeId': 'node_b',
            },
        },
    )
    editor = data['instanceEditor']
    edge = editor['createEdge']
    assert edge['__typename'] == 'NodeEdgeType'
    assert edge['fromRef']['nodeId'] == 'node_a'
    assert edge['toRef']['nodeId'] == 'node_b'
    assert edge['toRef']['portId'] == str(_port_uuid('input'))

    # Delete
    edge_obj = NodeEdge.objects.get(instance=db_instance_config, from_node=nc_a, to_node=nc_b)
    data = gql_client.query_data(
        DELETE_EDGE,
        variables={'instanceId': str(db_instance_config.pk), 'edgeId': str(edge_obj.pk)},
    )
    assert data['instanceEditor']['deleteEdge'] is None
    assert not NodeEdge.objects.filter(pk=edge_obj.pk).exists()


def test_delete_edge_cannot_cross_instance_boundary(client, db_instance_config: InstanceConfig):
    from paths.tests.graphql import PathsTestClient

    from nodes.models import NodeEdge
    from nodes.roles import instance_admin_role
    from nodes.tests.factories import InstanceFactory
    from users.tests.factories import UserFactory

    allowed_instance = db_instance_config
    target_instance = InstanceConfigFactory.create(instance=InstanceFactory.create(), config_source='database')

    unit = unit_registry.parse_units('kt/a')
    source_node = NodeConfigFactory.create(
        instance=target_instance,
        identifier='source_node',
        spec=_make_node_spec(output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')]),
    )
    target_node = NodeConfigFactory.create(
        instance=target_instance,
        identifier='target_node',
        spec=_make_node_spec(
            input_ports=[InputPortDef(id=_port_uuid('input'), unit=unit, quantity='emissions')],
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')],
        ),
    )
    edge = NodeEdge.objects.create(
        instance=target_instance,
        from_node=source_node,
        from_port=_port_uuid('default'),
        to_node=target_node,
        to_port=_port_uuid('input'),
    )

    user = UserFactory.create()
    instance_admin_role.assign_user(allowed_instance, user)
    client.force_login(user)

    gql_client = PathsTestClient(client)
    gql_client.set_instance(allowed_instance)
    gql_client.query_errors(
        DELETE_EDGE,
        variables={'instanceId': str(allowed_instance.pk), 'edgeId': str(edge.pk)},
        assert_error_message='Edge not found',
    )

    assert NodeEdge.objects.filter(pk=edge.pk).exists()


# ---------------------------------------------------------------------------
# model_instance query (smoke test)
# ---------------------------------------------------------------------------

MODEL_INSTANCE_QUERY = gql("""
query ModelInstanceTest($id: ID!) {
    modelInstance(instanceId: $id) {
        identifier
        editor {
            configSource
            graphLayout {
                coreNodeIds
                ghostableContextSourceIds
                hubIds
                actionIds
                outcomeIds
                mainGraphNodeIds
                thresholds {
                    hubDegree
                    ghostableOutDegree
                    ghostableTotalDegree
                    ghostableAvgOutgoingSpan
                }
            }
            spec {
                years {
                    reference
                    target
                }
            }
            edges {
                fromRef {
                    nodeId
                }
                toRef {
                    nodeId
                }
            }
            datasetPorts {
                nodeRef {
                    nodeId
                    portId
                }
                dataset {
                    id
                    identifier
                    isExternalPlaceholder
                    externalRef {
                        repoUrl
                        commit
                        datasetId
                    }
                }
                metric {
                    id
                    name
                    label
                }
                externalDatasetId
                externalMetricId
            }
        }
        nodes {
            identifier
            color
            kind
            uuid
            inputNodes { identifier }
            outputNodes { identifier }
            isVisible
            editor {
                nodeGroup
                nodeType
                inputDimensions
                outputDimensions
                tags
                layoutMeta {
                    primaryClass
                    ghostable
                    ghostTargets
                    topologicalLayer
                    totalDegree
                }
                spec {
                    inputPorts {
                        id
                        quantity
                        multi
                        requiredDimensions
                        supportedDimensions
                        bindings {
                            __typename
                            ... on NodeEdgeType {
                                fromRef {
                                    nodeId
                                    portId
                                }
                                toRef {
                                    nodeId
                                    portId
                                }
                            }
                            ... on DatasetPortType {
                                nodeRef {
                                    nodeId
                                    portId
                                }
                                dataset {
                                    id
                                    identifier
                                    isExternalPlaceholder
                                    externalRef {
                                        repoUrl
                                        commit
                                        datasetId
                                    }
                                }
                                metric {
                                    id
                                    name
                                    label
                                }
                                externalDatasetId
                                externalMetricId
                            }
                        }
                    }
                    outputPorts {
                        id
                        quantity
                        dimensions
                        edges {
                            fromRef {
                                nodeId
                                portId
                            }
                            toRef {
                                nodeId
                                portId
                            }
                        }
                    }
                }
            }
        }
    }
}
""")


def _make_input_port(id: str = 'input', unit: str = 'kt/a', quantity: str = 'emissions', multi: bool = True) -> InputPortDef:
    return InputPortDef(id=_port_uuid(id), unit=unit_registry.parse_units(unit), quantity=quantity, multi=multi)


def _make_output_port(id: str = 'default', unit: str = 'kt/a', quantity: str = 'emissions') -> OutputPortDef:
    return OutputPortDef(id=_port_uuid(id), unit=unit_registry.parse_units(unit), quantity=quantity)


def test_model_instance_query(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeEdge

    source = NodeConfigFactory.create(instance=db_instance_config, identifier='source_node')
    target = NodeConfigFactory.create(instance=db_instance_config, identifier='queried_node')
    NodeEdge.objects.create(
        instance=db_instance_config,
        from_node=source,
        to_node=target,
        from_port=_port_uuid('default'),
        to_port=_port_uuid('input'),
        transformations=[],
        tags=[],
    )

    data = gql_client.query_data(
        MODEL_INSTANCE_QUERY,
        variables={'id': str(db_instance_config.pk)},
    )
    mi = data['modelInstance']
    assert mi['identifier'] == db_instance_config.identifier
    assert mi['editor']['configSource'] == 'database'
    node_ids = [n['identifier'] for n in mi['nodes']]
    assert 'queried_node' in node_ids
    assert mi['editor']['graphLayout']['thresholds']['hubDegree'] == 7
    assert 'source_node' in mi['editor']['graphLayout']['ghostableContextSourceIds']
    assert 'source_node' not in mi['editor']['graphLayout']['mainGraphNodeIds']

    node_by_id = {node['identifier']: node for node in mi['nodes']}
    assert node_by_id['source_node']['editor']['layoutMeta']['primaryClass'] == 'GHOSTABLE_CONTEXT_SOURCE'
    assert node_by_id['source_node']['editor']['layoutMeta']['ghostTargets'] == ['queried_node']
    assert node_by_id['queried_node']['editor']['layoutMeta']['primaryClass'] == 'CONTEXT_SOURCE'


def test_model_instance_query_avoids_n_plus_one_for_port_bindings(
    gql_client: PathsTestClient, db_instance_config: InstanceConfig
):
    from nodes.models import DatasetPort, NodeEdge

    dataset = DatasetFactory.create(identifier='test_dataset')
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='test_metric')

    node_count = 12
    for idx in range(node_count):
        NodeConfigFactory.create(
            instance=db_instance_config,
            identifier=f'node_{idx}',
            spec=_make_node_spec(
                input_ports=[_make_input_port()],
                output_ports=[_make_output_port()],
            ),
        )

    nodes = {nc.identifier: nc for nc in db_instance_config.nodes.all()}

    for idx in range(1, node_count):
        NodeEdge.objects.create(
            instance=db_instance_config,
            from_node=nodes[f'node_{idx - 1}'],
            from_port=_port_uuid('default'),
            to_node=nodes[f'node_{idx}'],
            to_port=_port_uuid('input'),
            transformations=[],
            tags=[],
        )
        if idx >= 2:
            NodeEdge.objects.create(
                instance=db_instance_config,
                from_node=nodes[f'node_{idx - 2}'],
                from_port=_port_uuid('default'),
                to_node=nodes[f'node_{idx}'],
                to_port=_port_uuid('input'),
                transformations=[],
                tags=[],
            )

    for idx in range(0, node_count, 2):
        DatasetPort.objects.create(
            instance=db_instance_config,
            node=nodes[f'node_{idx}'],
            port_id=_port_uuid('input'),
            dataset=dataset,
            metric=metric,
        )

    with CaptureQueriesContext(connection) as query_ctx:
        data = gql_client.query_data(
            MODEL_INSTANCE_QUERY,
            variables={'id': str(db_instance_config.pk)},
        )

    assert len(data['modelInstance']['nodes']) == node_count
    assert data['modelInstance']['editor']['datasetPorts'][0]['dataset']['identifier'] == 'test_dataset'
    assert data['modelInstance']['editor']['datasetPorts'][0]['metric']['name'] == 'test_metric'
    assert data['modelInstance']['editor']['datasetPorts'][0]['externalDatasetId'] == 'test_dataset'
    assert data['modelInstance']['editor']['datasetPorts'][0]['externalMetricId'] == 'test_metric'
    assert len(query_ctx) <= 20


def test_public_instance_nodes_hide_hidden_nodes_from_non_editors(client, db_instance_config: InstanceConfig):
    from paths.tests.graphql import PathsTestClient

    from nodes.models import _pytest_instances
    from nodes.roles import instance_admin_role
    from users.tests.factories import UserFactory

    NodeConfigFactory.create(instance=db_instance_config, identifier='visible_node', is_visible=True)
    NodeConfigFactory.create(instance=db_instance_config, identifier='hidden_node', is_visible=False)

    query = """
    query {
        instance {
            nodes {
                identifier
            }
        }
    }
    """

    public_client = PathsTestClient(client)
    public_client.set_instance(db_instance_config)
    cached = _pytest_instances.pop(db_instance_config.identifier, None)
    try:
        public_data = public_client.query_data(query)
    finally:
        if cached is not None:
            _pytest_instances[db_instance_config.identifier] = cached
    public_ids = {node['identifier'] for node in public_data['instance']['nodes']}
    assert 'visible_node' in public_ids
    assert 'hidden_node' not in public_ids

    user = UserFactory.create()
    instance_admin_role.assign_user(db_instance_config, user)
    client.force_login(user)

    editor_client = PathsTestClient(client)
    editor_client.set_instance(db_instance_config)
    cached = _pytest_instances.pop(db_instance_config.identifier, None)
    try:
        editor_data = editor_client.query_data(query)
    finally:
        if cached is not None:
            _pytest_instances[db_instance_config.identifier] = cached
    editor_ids = {node['identifier'] for node in editor_data['instance']['nodes']}
    assert 'visible_node' in editor_ids
    assert 'hidden_node' in editor_ids


# ---------------------------------------------------------------------------
# Roundtrip: mutation → DB → runtime Context
# ---------------------------------------------------------------------------


def test_update_node_roundtrip(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """After updating a node via mutation, rebuilding Instance from DB reflects the changes."""
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='rt_node',
        name='Before',
        color='#000000',
        spec=_make_node_spec(),
    )

    gql_client.query_data(
        UPDATE_NODE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': str(nc.pk),
            'input': {
                'name': 'After',
                'color': '#abcdef',
                'isVisible': False,
                'isOutcome': True,
            },
        },
    )

    ctx = _rebuild_from_db(db_instance_config)
    rt_node = ctx.nodes['rt_node']
    assert str(rt_node.name) == 'After'
    assert rt_node.color == '#abcdef'
    assert rt_node.is_outcome is True


def test_create_edge_roundtrip(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """After creating an edge via mutation, the runtime graph reflects the connection."""
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='src',
        name='Source',
        spec=_make_node_spec(output_ports=[_make_output_port()]),
    )
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='dst',
        name='Destination',
        spec=_make_node_spec(
            input_ports=[_make_input_port()],
            output_ports=[_make_output_port()],
        ),
    )

    gql_client.query_data(
        CREATE_EDGE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromNodeId': 'src',
                'toNodeId': 'dst',
            },
        },
    )

    ctx = _rebuild_from_db(db_instance_config)
    src = ctx.nodes['src']
    dst = ctx.nodes['dst']
    assert dst in src.output_nodes
    assert src in dst.input_nodes


def test_create_edge_rejects_quantity_mismatch(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='src',
        spec=_make_node_spec(output_ports=[_make_output_port()]),
    )
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='dst',
        spec=_make_node_spec(
            input_ports=[_make_input_port(unit='t/a', quantity='energy')],
            output_ports=[_make_output_port(unit='t/a', quantity='energy')],
        ),
    )

    gql_client.query_errors(
        CREATE_EDGE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromNodeId': 'src',
                'toNodeId': 'dst',
            },
        },
        assert_error_message='Quantity mismatch',
    )


def test_create_edge_rejects_second_binding_for_non_multi_port(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeEdge

    src_a = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='src_a',
        spec=_make_node_spec(output_ports=[_make_output_port()]),
    )
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='src_b',
        spec=_make_node_spec(output_ports=[_make_output_port()]),
    )
    dst = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='dst',
        spec=_make_node_spec(
            input_ports=[_make_input_port(unit='t/a', quantity='emissions', multi=False)],
            output_ports=[_make_output_port(unit='t/a', quantity='emissions')],
        ),
    )
    NodeEdge.objects.create(
        instance=db_instance_config,
        from_node=src_a,
        from_port=_port_uuid('default'),
        to_node=dst,
        to_port=_port_uuid('input'),
        transformations=[],
        tags=[],
    )

    gql_client.query_errors(
        CREATE_EDGE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromNodeId': 'src_b',
                'toNodeId': 'dst',
                'fromPort': str(_port_uuid('default')),
                'toPort': str(_port_uuid('input')),
            },
        },
        assert_error_message='already has a binding',
    )


def test_create_edge_allows_second_binding_for_multi_port(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeEdge

    src_a = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='src_a',
        spec=_make_node_spec(output_ports=[_make_output_port()]),
    )
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='src_b',
        spec=_make_node_spec(output_ports=[_make_output_port()]),
    )
    dst = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='dst',
        spec=_make_node_spec(
            input_ports=[_make_input_port(unit='t/a', quantity='emissions', multi=True)],
            output_ports=[_make_output_port(unit='t/a', quantity='emissions')],
        ),
    )
    NodeEdge.objects.create(
        instance=db_instance_config,
        from_node=src_a,
        from_port=_port_uuid('default'),
        to_node=dst,
        to_port=_port_uuid('input'),
        transformations=[],
        tags=[],
    )

    data = gql_client.query_data(
        CREATE_EDGE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromNodeId': 'src_b',
                'toNodeId': 'dst',
                'fromPort': str(_port_uuid('default')),
                'toPort': str(_port_uuid('input')),
            },
        },
    )
    edge = data['instanceEditor']['createEdge']
    assert edge['__typename'] == 'NodeEdgeType'
    assert edge['fromRef']['nodeId'] == 'src_b'
    assert edge['toRef']['portId'] == str(_port_uuid('input'))


def test_delete_node_roundtrip(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """After deleting a node, the rebuilt runtime graph no longer contains it."""
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='ephemeral',
        name='Gone Soon',
        spec=_make_node_spec(),
    )

    gql_client.query_data(
        DELETE_NODE,
        variables={'instanceId': str(db_instance_config.pk), 'nodeId': str(nc.pk)},
    )

    ctx = _rebuild_from_db(db_instance_config)
    assert 'ephemeral' not in ctx.nodes
