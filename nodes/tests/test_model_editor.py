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
from nodes.defs.instance_defs import ActionGroup, InstanceModelSpec, NormalizationSpec, YearsSpec
from nodes.defs.node_defs import ActionConfig, InputDatasetDef, NodeKind, NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.defs.transform_def import FilterColumnOp, forecast_from_transformations
from nodes.input_bindings import sync_input_bindings
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory, _port_id, register_dimensions
from nodes.units import unit_registry

if TYPE_CHECKING:
    from uuid import UUID

    from paths.tests.graphql import PathsTestClient

    from nodes.actions.action import ActionNode
    from nodes.context import Context
    from nodes.datasets import DatasetWithFilters
    from nodes.models import InstanceConfig
    from nodes.node import Node


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


_register_dimensions = register_dimensions


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


UPDATE_NODE_LAYOUTS = """
mutation UpdateNodeLayouts($instanceId: ID!, $input: [UpdateNodeLayoutInput!]!) {
    instanceEditor(instanceId: $instanceId) {
        updateNodeLayouts(input: $input) {
            ... on UpdateNodeLayoutsResult {
                layouts {
                    nodeId
                    x
                    y
                    source
                    createdBy { id }
                    lastModifiedBy { id }
                }
            }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""

CLEAR_NODE_LAYOUTS = """
mutation ClearNodeLayouts($instanceId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        clearNodeLayouts { messages { kind message } }
    }
}
"""


def test_update_node_layouts_is_shared_editor_metadata_without_change_operation(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    from nodes.models import NodeLayout, NodeLayoutSource

    first = NodeConfigFactory.create(instance=db_instance_config, identifier='first', spec=_make_node_spec())
    second = NodeConfigFactory.create(instance=db_instance_config, identifier='second', spec=_make_node_spec())
    previous_revision_id = db_instance_config.latest_revision_id
    previous_head = db_instance_config.draft_head_token

    data = gql_client.query_data(
        UPDATE_NODE_LAYOUTS,
        variables={
            'instanceId': db_instance_config.identifier,
            'input': [
                {'nodeId': str(first.uuid), 'x': 12.5, 'y': -3.25, 'source': 'USER'},
                {'nodeId': second.identifier, 'x': 100.0, 'y': 200.0, 'source': 'AUTO'},
            ],
        },
    )

    result = data['instanceEditor']['updateNodeLayouts']
    assert [(layout['nodeId'], layout['x'], layout['y'], layout['source']) for layout in result['layouts']] == [
        ('first', 12.5, -3.25, 'USER'),
        ('second', 100.0, 200.0, 'AUTO'),
    ]
    first_layout = NodeLayout.objects.get(node=first)
    assert first_layout.source == NodeLayoutSource.USER
    assert first_layout.created_by_id is not None
    assert first_layout.last_modified_by_id == first_layout.created_by_id
    db_instance_config.refresh_from_db()
    assert db_instance_config.latest_revision_id == previous_revision_id
    assert db_instance_config.draft_head_token == previous_head


def test_instance_admin_can_update_layout_of_protected_node(client, db_instance_config: InstanceConfig) -> None:
    from paths.tests.graphql import PathsTestClient

    from nodes.models import NodeLayout
    from nodes.roles import instance_admin_role
    from users.tests.factories import UserFactory

    node = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='protected_layout',
        is_editable=False,
        spec=_make_node_spec(),
    )
    user = UserFactory.create()
    instance_admin_role.assign_user(db_instance_config, user)
    client.force_login(user)
    admin_client = PathsTestClient(client)
    admin_client.set_instance(db_instance_config)

    admin_client.query_data(
        UPDATE_NODE_LAYOUTS,
        variables={
            'instanceId': db_instance_config.identifier,
            'input': [{'nodeId': str(node.uuid), 'x': 1.0, 'y': 2.0, 'source': 'USER'}],
        },
    )
    assert NodeLayout.objects.filter(node=node, x=1.0, y=2.0).exists()


def test_node_layout_is_readable_per_node_and_in_bulk(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    from nodes.models import NodeLayout, NodeLayoutSource

    node = NodeConfigFactory.create(instance=db_instance_config, identifier='positioned', spec=_make_node_spec())
    NodeLayout.objects.create(node=node, x=1.5, y=2.5, source=NodeLayoutSource.USER)
    from nodes.models import _pytest_instances

    _pytest_instances.pop(db_instance_config.identifier, None)

    with CaptureQueriesContext(connection) as queries:
        data = gql_client.query_data(
            """
            query NodeLayouts {
                instance {
                    editor {
                        nodeLayouts { nodeId x y source }
                    }
                    nodes(id: ["positioned"]) {
                        id
                        editor { layout { nodeId x y source } }
                    }
                }
            }
            """,
        )

    expected = {'nodeId': 'positioned', 'x': 1.5, 'y': 2.5, 'source': 'USER'}
    assert data['instance']['editor']['nodeLayouts'] == [expected]
    assert data['instance']['nodes'][0]['editor']['layout'] == expected
    per_node_layout_queries = [
        query['sql']
        for query in queries.captured_queries
        if 'FROM "nodes_nodelayout"' in query['sql'] and 'WHERE "nodes_nodelayout"."node_id" =' in query['sql']
    ]
    assert per_node_layout_queries == []


def test_clear_node_layouts_bypasses_model_change_tracking(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    from nodes.models import NodeLayout

    node = NodeConfigFactory.create(instance=db_instance_config, identifier='positioned', spec=_make_node_spec())
    NodeLayout.objects.create(node=node, x=1.0, y=2.0)
    previous_head = db_instance_config.draft_head_token

    gql_client.query_data(
        CLEAR_NODE_LAYOUTS,
        variables={'instanceId': db_instance_config.identifier},
    )

    assert not NodeLayout.objects.filter(node__instance=db_instance_config).exists()
    assert db_instance_config.draft_head_token == previous_head


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

SET_INSTANCE_LOCKED = """
mutation SetInstanceLocked($instanceId: ID!, $isLocked: Boolean!) {
    setInstanceLocked(instanceId: $instanceId, isLocked: $isLocked) {
        ... on SetInstanceLockedResult {
            instanceId
            isLocked
        }
        ... on OperationInfo { messages { kind message } }
    }
}
"""


INSTANCE_QUANTITY_KINDS = """
query InstanceQuantityKinds {
    instance {
        editor {
            quantityKinds {
                kind {
                    id
                    label
                }
                usedUnits {
                    count
                    unit {
                        standard
                    }
                }
            }
        }
    }
}
"""


INSTANCE_ADMIN_EDITOR_FIELDS = """
query InstanceAdminEditorFields($instanceId: ID!) {
    modelInstance(instanceId: $instanceId) {
        editor {
            configSource
        }
        nodes {
            identifier
            editor {
                spec {
                    outputPorts {
                        id
                    }
                }
            }
        }
    }
    instance {
        editor {
            configSource
        }
    }
}
"""


def test_instance_admin_can_read_model_editor_fields(client, db_instance_config: InstanceConfig):
    from paths.tests.graphql import PathsTestClient

    from nodes.roles import instance_admin_role
    from users.tests.factories import UserFactory

    NodeConfigFactory.create(instance=db_instance_config, identifier='editable_node', spec=_make_node_spec())

    user = UserFactory.create()
    instance_admin_role.assign_user(db_instance_config, user)
    client.force_login(user)

    gql_client = PathsTestClient(client)
    gql_client.set_instance(db_instance_config)
    data = gql_client.query_data(
        INSTANCE_ADMIN_EDITOR_FIELDS,
        variables={'instanceId': str(db_instance_config.pk)},
    )

    assert data['modelInstance']['editor']['configSource'] == 'database'
    assert data['instance']['editor']['configSource'] == 'database'
    node = data['modelInstance']['nodes'][0]
    assert node['identifier'] == 'editable_node'
    assert node['editor']['spec']['outputPorts'][0]['id'] == str(_port_uuid('default'))


def test_model_instance_metadata_does_not_create_runtime(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_require_instance(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('modelInstance metadata must not create a runtime Instance')

    monkeypatch.setattr('paths.schema_context.PathsGraphQLContext.require_instance', fail_require_instance)

    data = gql_client.query_data(
        """
        query ModelInstanceMetadata($instanceId: ID!) {
            modelInstance(instanceId: $instanceId) {
                identifier
                editor { configSource }
            }
        }
        """,
        variables={'instanceId': str(db_instance_config.pk)},
    )

    assert data['modelInstance'] == {
        'identifier': db_instance_config.identifier,
        'editor': {'configSource': 'database'},
    }


def test_model_instance_runtime_uses_draft_when_request_defaults_to_published(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    db_instance_config.publish_instance()
    NodeConfigFactory.create(instance=db_instance_config, identifier='draft_only', spec=_make_node_spec())

    data = gql_client.query_data(
        """
        query DraftModelInstance($instanceId: ID!) {
            modelInstance(instanceId: $instanceId) {
                nodes { identifier }
            }
        }
        """,
        variables={'instanceId': str(db_instance_config.pk)},
    )

    assert data['modelInstance']['nodes'] == [{'identifier': 'draft_only'}]


NODE_STATUS_FIELDS = gql("""
query NodeStatusFields($instanceId: ID!) {
    modelInstance(instanceId: $instanceId) {
        nodes {
            identifier
            editor {
                status
                errors {
                    phase
                    message
                }
            }
        }
    }
}
""")


def test_node_editor_exposes_status_and_errors(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    NodeConfigFactory.create(instance=db_instance_config, identifier='status_node', spec=_make_node_spec())
    data = gql_client.query_data(NODE_STATUS_FIELDS, variables={'instanceId': str(db_instance_config.pk)})
    nodes = {n['identifier']: n for n in data['modelInstance']['nodes']}
    editor = nodes['status_node']['editor']
    # Freshly loaded and not computed: status is null and no errors are recorded yet.
    assert editor['status'] is None
    assert editor['errors'] == []


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


def test_query_instance_quantity_kinds_with_used_units(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    node_specs = [
        ('emissions_kt_1', 'kt/a', 'emissions'),
        ('emissions_kt_2', 'kt/a', 'emissions'),
        ('emissions_t', 't/a', 'emissions'),
        ('energy_mwh', 'MWh/a', 'energy'),
    ]
    for identifier, unit, quantity in node_specs:
        gql_client.query_data(
            CREATE_NODE,
            variables={
                'instanceId': str(db_instance_config.pk),
                'input': {
                    'identifier': identifier,
                    'name': identifier.replace('_', ' ').title(),
                    'kind': 'SIMPLE',
                    'config': {'simple': {'nodeClass': SIMPLE_NODE_CLASS}},
                    'outputPorts': [{'unit': unit, 'quantity': quantity}],
                },
            },
        )

    from nodes.models import _pytest_instances

    _pytest_instances.pop(db_instance_config.identifier, None)
    data = gql_client.query_data(INSTANCE_QUANTITY_KINDS)
    quantity_kinds = {entry['kind']['id']: entry for entry in data['instance']['editor']['quantityKinds']}

    assert 'emissions' in quantity_kinds
    assert 'energy' in quantity_kinds
    assert quantity_kinds['emissions']['usedUnits'] == [
        {'count': 2, 'unit': {'standard': 'kt/a'}},
        {'count': 1, 'unit': {'standard': 't/a'}},
    ]
    assert quantity_kinds['energy']['usedUnits'] == [{'count': 1, 'unit': {'standard': 'MWh/a'}}]


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
    db_instance_config.save(update_fields=['spec'])
    _register_dimensions(db_instance_config, ['energy_carrier', 'energy_usage', 'cost_type', 'sector', 'ghg'])

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


def test_create_node_rejects_locked_instance(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    db_instance_config.is_locked = True
    db_instance_config.save(update_fields=['is_locked'])

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
        assert_error_message='Instance is locked',
    )

    assert (errors[0].get('extensions') or {}).get('code') == 'instance_locked'


def test_set_instance_locked_can_unlock_locked_instance(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    data = gql_client.query_data(
        SET_INSTANCE_LOCKED,
        variables={'instanceId': str(db_instance_config.pk), 'isLocked': True},
    )
    assert data['setInstanceLocked']['isLocked'] is True
    db_instance_config.refresh_from_db()
    assert db_instance_config.is_locked is True

    data = gql_client.query_data(
        SET_INSTANCE_LOCKED,
        variables={'instanceId': str(db_instance_config.pk), 'isLocked': False},
    )
    assert data['setInstanceLocked']['isLocked'] is False
    db_instance_config.refresh_from_db()
    assert db_instance_config.is_locked is False


# ---------------------------------------------------------------------------
# update_node
# ---------------------------------------------------------------------------

UPDATE_NODE = gql("""
mutation UpdateNode($instanceId: ID!, $nodeId: ID!, $input: UpdateNodeInput!) {
    instanceEditor(instanceId: $instanceId) {
        updateNode(nodeId: $nodeId, input: $input) {
            ... on NodeInterface {
                identifier
                name
                shortName
                description
                color
                kind
                isVisible
                editor {
                    nodeGroup
                    spec {
                        inputPorts {
                            id
                            quantity
                            unit {
                                standard
                            }
                        }
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
            ... on Node {
                isOutcome
            }
            ... on ActionNode {
                group {
                    id
                    name
                }
            }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
""")


UPDATE_NODE_VIA_NODE_EDITOR = gql("""
mutation UpdateNodeViaNodeEditor($instanceId: ID!, $nodeId: ID!, $input: UpdateNodeInput!) {
    instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
            update(input: $input) {
                ... on NodeInterface {
                    identifier
                    name
                    color
                    isVisible
                }
                ... on OperationInfo { messages { kind message } }
            }
        }
    }
}
""")


ADD_NODE_INPUT_PORT_VIA_NODE_EDITOR = gql("""
mutation AddNodeInputPortViaNodeEditor($instanceId: ID!, $nodeId: ID!, $input: InputPortInput!) {
    instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
            addInputPort(input: $input) {
                __typename
                ... on InputPortType {
                    id
                    quantity
                    multi
                    unit { standard }
                }
                ... on OperationInfo { messages { kind message } }
            }
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
            'nodeId': str(nc.uuid),
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


def test_node_editor_update_alias(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    nc = NodeConfigFactory.create(instance=db_instance_config, identifier='editable_nested', name='Old', color='#000')

    data = gql_client.query_data(
        UPDATE_NODE_VIA_NODE_EDITOR,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': str(nc.uuid),
            'input': {
                'name': 'Nested Updated',
                'color': '#0088ff',
                'isVisible': False,
            },
        },
    )
    node = data['instanceEditor']['nodeEditor']['update']
    assert node['name'] == 'Nested Updated'
    assert node['color'] == '#0088ff'
    assert node['isVisible'] is False

    nc.refresh_from_db()
    assert nc.name == 'Nested Updated'
    assert nc.color == '#0088ff'
    assert nc.is_visible is False


def test_protected_node_exposes_effective_permissions_and_rejects_admin_update(
    client,
    db_instance_config: InstanceConfig,
) -> None:
    from paths.tests.graphql import PathsTestClient

    from nodes.roles import instance_admin_role
    from users.tests.factories import UserFactory

    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='protected',
        name='Certified',
        is_editable=False,
        spec=_make_node_spec(),
    )
    user = UserFactory.create()
    instance_admin_role.assign_user(db_instance_config, user)
    client.force_login(user)
    admin_client = PathsTestClient(client)
    admin_client.set_instance(db_instance_config)

    data = admin_client.query_data(
        """
        query ProtectedNode($instanceId: ID!) {
            modelInstance(instanceId: $instanceId) {
                nodes {
                    identifier
                    isEditable
                    userPermissions { view change delete }
                    editor { nodeGroup }
                }
            }
        }
        """,
        variables={'instanceId': str(db_instance_config.pk)},
    )
    node = next(node for node in data['modelInstance']['nodes'] if node['identifier'] == nc.identifier)
    assert node == {
        'identifier': 'protected',
        'isEditable': False,
        'userPermissions': {'view': True, 'change': False, 'delete': False},
        'editor': None,
    }

    admin_client.query_errors(
        UPDATE_NODE_VIA_NODE_EDITOR,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': str(nc.uuid),
            'input': {'name': 'Tampered'},
        },
        assert_error_message='Permission denied',
    )
    nc.refresh_from_db()
    assert nc.name == 'Certified'


def test_superuser_can_update_protected_node(gql_client: PathsTestClient, db_instance_config: InstanceConfig) -> None:
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='protected_for_superuser',
        name='Before',
        is_editable=False,
        spec=_make_node_spec(),
    )

    gql_client.query_data(
        UPDATE_NODE_VIA_NODE_EDITOR,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': str(nc.uuid),
            'input': {'name': 'After'},
        },
    )
    nc.refresh_from_db()
    assert nc.name == 'After'


def test_node_editor_add_input_port(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeConfig

    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='editable_ports',
        spec=_make_node_spec(),
    )

    data = gql_client.query_data(
        ADD_NODE_INPUT_PORT_VIA_NODE_EDITOR,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': str(nc.uuid),
            'input': {
                'unit': 'kt/a',
                'quantity': 'emissions',
                'multi': True,
            },
        },
    )
    port = data['instanceEditor']['nodeEditor']['addInputPort']
    assert port['quantity'] == 'emissions'
    assert port['multi'] is True
    assert port['unit']['standard'] == 'kt/a'

    nc = NodeConfig.objects.get(pk=nc.pk)
    assert nc.spec is not None
    assert len(nc.spec.input_ports) == 1
    assert str(nc.spec.input_ports[0].id) == port['id']


def test_update_node_modeling_fields(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    assert db_instance_config.spec is not None
    db_instance_config.spec.action_groups = [ActionGroup(id='energy', name='Energy')]
    db_instance_config.save(update_fields=['spec'])
    _register_dimensions(db_instance_config, ['energy_carrier', 'energy_usage', 'cost_type', 'sector', 'ghg'])

    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='editable_modeling',
        name='Editable Modeling',
        spec=_make_node_spec(),
    )

    data = gql_client.query_data(
        UPDATE_NODE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': str(nc.uuid),
            'input': {
                'kind': 'ACTION',
                'shortName': 'CCS',
                'description': 'Carbon capture update',
                'nodeGroup': 'transport',
                'allowNulls': True,
                'minimumYear': 2024,
                'config': {
                    'action': {
                        'nodeClass': ACTION_NODE_CLASS,
                        'group': 'energy',
                        'noEffectValue': 0.0,
                    },
                },
                'inputPorts': [{'unit': 't/a', 'quantity': 'emissions'}],
                'inputDimensions': ['energy_carrier', 'energy_usage', 'cost_type', 'sector', 'ghg'],
                'outputDimensions': ['energy_carrier', 'energy_usage', 'cost_type', 'sector', 'ghg'],
                'params': {'allow_null_categories': True},
                'outputMetrics': [
                    {'id': 'emissions', 'unit': 't/a', 'quantity': 'emissions'},
                    {'id': 'energy', 'unit': 'TJ/a', 'quantity': 'energy'},
                    {'id': 'currency', 'unit': 'DKK/a', 'quantity': 'currency'},
                ],
                'tags': ['trial'],
            },
        },
    )

    node = data['instanceEditor']['updateNode']
    assert node['kind'] == 'ACTION'
    assert node['shortName'] == 'CCS'
    assert node['description'] == 'Carbon capture update'
    assert node['editor']['nodeGroup'] == 'transport'
    assert node['editor']['spec']['typeConfig']['nodeClass'] == ACTION_NODE_CLASS
    assert node['editor']['spec']['typeConfig']['group'] == 'energy'
    assert [port['quantity'] for port in node['editor']['spec']['outputPorts']] == ['emissions', 'energy', 'currency']

    from nodes.models import NodeConfig

    nc = NodeConfig.objects.get(pk=nc.pk)
    assert nc.spec is not None
    assert nc.description == 'Carbon capture update'
    assert nc.short_description == 'Carbon capture update'
    assert nc.short_name == 'CCS'
    assert nc.spec.kind == NodeKind.ACTION
    assert isinstance(nc.spec.type_config, ActionConfig)
    assert nc.spec.type_config.group == 'energy'
    assert nc.spec.node_group == 'transport'
    assert nc.spec.allow_nulls is True
    assert nc.spec.minimum_year == 2024
    assert nc.spec.input_dimensions == ['energy_carrier', 'energy_usage', 'cost_type', 'sector', 'ghg']
    assert nc.spec.output_dimensions == ['energy_carrier', 'energy_usage', 'cost_type', 'sector', 'ghg']
    assert nc.spec.input_ports[0].quantity == 'emissions'
    assert [port.column_id for port in nc.spec.output_ports] == ['emissions', 'energy', 'currency']
    assert nc.spec.extra.tags == ['trial']
    allow_null_categories = next(param for param in nc.spec.params if param.local_id == 'allow_null_categories')
    assert allow_null_categories.value is True


def test_update_node_requires_config_when_changing_kind(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='needs_config',
        name='Needs Config',
        spec=_make_node_spec(),
    )

    errors = gql_client.query_errors(
        UPDATE_NODE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': str(nc.uuid),
            'input': {'kind': 'ACTION'},
        },
    )

    assert 'config must be provided when changing node kind' in errors[0]['message']


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


def test_instance_metadata_projects_from_columns(db_instance_config: InstanceConfig):
    from nodes.defs.instance_defs import InstanceMetadata

    db_instance_config.refresh_from_db()
    meta = InstanceMetadata.from_model(db_instance_config)
    assert meta.uuid == db_instance_config.uuid
    assert meta.identifier == db_instance_config.identifier
    assert str(meta.name) == db_instance_config.name


def test_node_metadata_does_not_mutate_spec_on_save(db_instance_config: InstanceConfig):
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='spec_display',
        name='Spec Display',
        short_name='Short display',
        color='#123456',
        order=7,
        is_visible=False,
    )
    assert nc.spec is not None
    original_spec = nc.spec.model_dump(mode='json')
    nc.name = 'Changed display'
    nc.save()
    nc.refresh_from_db()
    assert nc.spec is not None
    assert nc.spec.model_dump(mode='json') == original_spec


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
            type_config=ActionConfig(node_class=PARENT_ACTION_NODE_CLASS, decision_level=DecisionLevel.MUNICIPALITY),
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')],
        ),
    )
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='child_action',
        name='Child Action',
        spec=NodeSpec(
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


DELETE_NODE_VIA_NODE_EDITOR = gql("""
mutation DeleteNodeViaNodeEditor($instanceId: ID!, $nodeId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
            delete {
                messages { kind message }
            }
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
        variables={'instanceId': str(db_instance_config.pk), 'nodeId': str(nc.uuid)},
    )
    # query_data asserts no errors; just verify the node is gone
    assert not NodeConfig.objects.filter(pk=node_pk).exists()


def test_node_editor_delete(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeConfig

    nc = NodeConfigFactory.create(instance=db_instance_config, identifier='doomed_nested')
    node_pk = nc.pk

    data = gql_client.query_data(
        DELETE_NODE_VIA_NODE_EDITOR,
        variables={'instanceId': str(db_instance_config.pk), 'nodeId': str(nc.uuid)},
    )
    assert data['instanceEditor']['nodeEditor']['delete'] is None
    assert not NodeConfig.objects.filter(pk=node_pk).exists()


def test_instance_admin_can_delete_node(client, db_instance_config: InstanceConfig):
    from paths.tests.graphql import PathsTestClient

    from nodes.models import NodeConfig
    from nodes.roles import instance_admin_role
    from users.tests.factories import UserFactory

    nc = NodeConfigFactory.create(instance=db_instance_config, identifier='doomed_by_admin')
    node_pk = nc.pk

    user = UserFactory.create()
    instance_admin_role.assign_user(db_instance_config, user)
    client.force_login(user)

    gql_client = PathsTestClient(client)
    gql_client.set_instance(db_instance_config)
    gql_client.query_data(
        DELETE_NODE,
        variables={'instanceId': str(db_instance_config.pk), 'nodeId': str(nc.uuid)},
    )

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
            ... on ConstraintViolations {
                conflicts {
                    code
                    message
                    value { kind nodeUuid portId direction bindingId }
                    origins { kind nodeUuid portId bindingId }
                }
            }
            ... on NodeEdgeType {
                fromRef {
                    nodeUuid
                    nodeId
                    portId
                }
                portRef {
                    nodeUuid
                    nodeId
                    portId
                }
                toRef {
                    nodeUuid
                    nodeId
                    portId
                }
                transformations {
                    __typename
                    ... on FilterDimensionType {
                        dimension categories flatten exclude
                    }
                    ... on AssignDimensionType {
                        dimension category
                    }
                    ... on FlattenType {
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

    # The edge's filter references the dimension, and the solver-backed
    # validator rejects references to dimensions the instance does not have.
    _register_dimensions(db_instance_config, ['sector'], {'sector': ['buildings']})

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

    # Create; the deprecated legacy input vocabulary is still accepted, but it
    # is stored and read back in the current one.
    data = gql_client.query_data(
        CREATE_EDGE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromNodeId': 'node_a',
                'toNodeId': 'node_b',
                'transformations': [
                    {'selectCategories': {'dimension': 'sector', 'categories': ['buildings'], 'flatten': True}},
                ],
            },
        },
    )
    editor = data['instanceEditor']
    edge = editor['createEdge']
    assert edge['__typename'] == 'NodeEdgeType'
    assert edge['fromRef']['nodeId'] == 'node_a'
    assert edge['fromRef']['nodeUuid'] == str(nc_a.uuid)
    assert edge['portRef']['nodeId'] == 'node_b'
    assert edge['portRef']['nodeUuid'] == str(nc_b.uuid)
    assert edge['portRef']['portId'] == str(_port_uuid('input'))
    assert edge['toRef'] == edge['portRef']
    assert edge['transformations'] == [
        {
            '__typename': 'FilterDimensionType',
            'dimension': 'sector',
            'categories': ['buildings'],
            'flatten': True,
            'exclude': False,
        },
    ]

    # Delete
    edge_obj = NodeEdge.objects.get(instance=db_instance_config, from_node=nc_a, to_node=nc_b)
    data = gql_client.query_data(
        DELETE_EDGE,
        variables={'instanceId': str(db_instance_config.pk), 'edgeId': str(edge_obj.uuid)},
    )
    assert data['instanceEditor']['deleteEdge'] is None
    assert not NodeEdge.objects.filter(pk=edge_obj.pk).exists()


def test_instance_admin_cannot_change_incoming_edge_of_protected_node(
    client,
    db_instance_config: InstanceConfig,
) -> None:
    from paths.tests.graphql import PathsTestClient

    from nodes.models import NodeEdge
    from nodes.roles import instance_admin_role
    from users.tests.factories import UserFactory

    unit = unit_registry.parse_units('kt/a')
    source = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='protected_edge_source',
        spec=_make_node_spec(output_ports=[OutputPortDef(id=_port_uuid('source'), unit=unit, quantity='emissions')]),
    )
    target = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='protected_edge_target',
        is_editable=False,
        spec=_make_node_spec(
            input_ports=[InputPortDef(id=_port_uuid('target'), unit=unit, quantity='emissions')],
            output_ports=[OutputPortDef(id=_port_uuid('target-output'), unit=unit, quantity='emissions')],
        ),
    )
    user = UserFactory.create()
    instance_admin_role.assign_user(db_instance_config, user)
    client.force_login(user)
    admin_client = PathsTestClient(client)
    admin_client.set_instance(db_instance_config)

    admin_client.query_errors(
        CREATE_EDGE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromRef': {'nodeUuid': str(source.uuid), 'portId': str(_port_uuid('source'))},
                'portRef': {'nodeUuid': str(target.uuid), 'portId': str(_port_uuid('target'))},
            },
        },
        assert_error_message='Permission denied',
    )
    assert not NodeEdge.objects.filter(from_node=source, to_node=target).exists()


def test_create_edge_accepts_uuid_only_references(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeEdge

    unit = unit_registry.parse_units('kt/a')
    source_port_id = _port_uuid('canonical-output')
    target_port_id = _port_uuid('canonical-input')
    source = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='canonical_source',
        spec=_make_node_spec(output_ports=[OutputPortDef(id=source_port_id, unit=unit, quantity='emissions')]),
    )
    target = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='canonical_target',
        spec=_make_node_spec(
            input_ports=[InputPortDef(id=target_port_id, unit=unit, quantity='emissions')],
            output_ports=[OutputPortDef(id=_port_uuid('canonical-target-output'), unit=unit, quantity='emissions')],
        ),
    )

    edge = gql_client.query_data(
        CREATE_EDGE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromRef': {'nodeUuid': str(source.uuid), 'portId': str(source_port_id)},
                'portRef': {'nodeUuid': str(target.uuid), 'portId': str(target_port_id)},
            },
        },
    )['instanceEditor']['createEdge']

    assert edge['fromRef']['nodeUuid'] == str(source.uuid)
    assert edge['portRef']['nodeUuid'] == str(target.uuid)
    assert NodeEdge.objects.filter(
        from_node=source,
        from_port=source_port_id,
        to_node=target,
        to_port=target_port_id,
    ).exists()


def _two_nodes_with_bindable_port(ic: InstanceConfig) -> None:
    unit = unit_registry.parse_units('kt/a')
    NodeConfigFactory.create(
        instance=ic,
        identifier='node_a',
        spec=_make_node_spec(output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')]),
    )
    NodeConfigFactory.create(
        instance=ic,
        identifier='node_b',
        spec=_make_node_spec(
            input_ports=[InputPortDef(id=_port_uuid('input'), unit=unit, quantity='emissions', multi=False)],
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')],
        ),
    )


def test_create_edge_replace_requires_an_explicit_to_port(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    _two_nodes_with_bindable_port(db_instance_config)

    gql_client.query_errors(
        CREATE_EDGE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromNodeId': 'node_a',
                'toNodeId': 'node_b',
                'replace': True,
            },
        },
        assert_error_message='requires an explicit `toPort`',
    )


def test_create_edge_replace_displaces_the_existing_edge(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeEdge

    unit = unit_registry.parse_units('kt/a')
    _two_nodes_with_bindable_port(db_instance_config)
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='node_c',
        spec=_make_node_spec(output_ports=[OutputPortDef(id=_port_uuid('other'), unit=unit, quantity='emissions')]),
    )
    nodes = {nc.identifier: nc for nc in db_instance_config.nodes.all()}
    old_edge = NodeEdge.objects.create(
        instance=db_instance_config,
        from_node=nodes['node_a'],
        from_port=_port_uuid('default'),
        to_node=nodes['node_b'],
        to_port=_port_uuid('input'),
    )

    edge = gql_client.query_data(
        CREATE_EDGE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromNodeId': 'node_c',
                'toNodeId': 'node_b',
                'toPort': str(_port_uuid('input')),
                'replace': True,
            },
        },
    )['instanceEditor']['createEdge']

    assert edge['fromRef']['nodeId'] == 'node_c'
    assert not NodeEdge.objects.filter(pk=old_edge.pk).exists()
    assert NodeEdge.objects.filter(to_node=nodes['node_b'], to_port=_port_uuid('input')).count() == 1


def test_create_edge_replace_displaces_a_dataset_binding(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import DatasetPort, NodeEdge

    _two_nodes_with_bindable_port(db_instance_config)
    nodes = {nc.identifier: nc for nc in db_instance_config.nodes.all()}
    dataset = DatasetFactory.create(identifier='occupant')
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='Energy')
    DatasetPort.objects.create(
        instance=db_instance_config,
        node=nodes['node_b'],
        port_id=_port_uuid('input'),
        dataset=dataset,
        metric=metric,
    )

    gql_client.query_data(
        CREATE_EDGE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'instanceId': str(db_instance_config.pk),
                'fromNodeId': 'node_a',
                'toNodeId': 'node_b',
                'toPort': str(_port_uuid('input')),
                'replace': True,
            },
        },
    )

    assert not DatasetPort.objects.filter(node=nodes['node_b']).exists()
    assert NodeEdge.objects.filter(to_node=nodes['node_b'], to_port=_port_uuid('input')).count() == 1


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
        variables={'instanceId': str(allowed_instance.pk), 'edgeId': str(edge.uuid)},
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
                portRef {
                    nodeId
                }
            }
            datasetPorts {
                portRef {
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
                                portRef {
                                    nodeId
                                    portId
                                }
                                transformations {
                                    __typename
                                    ... on FilterDimensionType { dimension categories flatten }
                                }
                            }
                            ... on DatasetPortType {
                                portRef {
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
                            portRef {
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
    from nodes.defs.transform_def import SelectCategoriesTransformation
    from nodes.models import NodeEdge

    _register_dimensions(db_instance_config, ['sector'], categories={'sector': ['buildings']})
    source = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='source_node',
        spec=_make_node_spec(output_ports=[_make_output_port()]),
    )
    target = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='queried_node',
        spec=_make_node_spec(input_ports=[_make_input_port()], output_ports=[_make_output_port()]),
    )
    NodeEdge.objects.create(
        instance=db_instance_config,
        from_node=source,
        to_node=target,
        from_port=_port_uuid('default'),
        to_port=_port_uuid('input'),
        transformations=[SelectCategoriesTransformation(dimension='sector', categories=['buildings'], flatten=True)],
        tags=[],
    )
    # Direct ORM writes bypass the write boundaries that keep the unified
    # input-binding mirror fresh; the port-binding view reads the mirror.
    sync_input_bindings(db_instance_config)

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

    # Edge transformations are visible through the port-binding view, in the
    # current vocabulary regardless of what the row stores.
    (port,) = node_by_id['queried_node']['editor']['spec']['inputPorts']
    (binding,) = port['bindings']
    assert binding['__typename'] == 'NodeEdgeType'
    assert binding['transformations'] == [
        {'__typename': 'FilterDimensionType', 'dimension': 'sector', 'categories': ['buildings'], 'flatten': True},
    ]


def test_model_instance_query_avoids_n_plus_one_for_port_bindings(
    gql_client: PathsTestClient, db_instance_config: InstanceConfig
):
    from nodes.models import DatasetPort, NodeEdge

    dataset = DatasetFactory.create(identifier='test_dataset')
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='test_metric')

    node_count = 15
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
    per_binding_dataset_queries = [
        query
        for query in query_ctx.captured_queries
        if 'FROM "datasets_dataset"' in query['sql'] and 'WHERE "datasets_dataset"."uuid" =' in query['sql']
    ]
    assert per_binding_dataset_queries == []
    assert len(query_ctx) <= 20


def test_dataset_ports_rebuild_multimetric_action_dataset(db_instance_config: InstanceConfig):
    from nodes.defs.node_defs import ColumnDatasetFilterDef, DatasetPortSpec
    from nodes.models import DatasetPort

    assert db_instance_config.spec is not None
    db_instance_config.spec.features.use_datasets_from_db = True
    db_instance_config.save(update_fields=['spec'])

    dataset = DatasetFactory.create(identifier='multi_metric_actions', scope=db_instance_config)
    emissions_metric = DatasetMetricFactory.create(schema=dataset.schema, name='emissions', label='Emissions', unit='t/a')
    energy_metric = DatasetMetricFactory.create(schema=dataset.schema, name='energy', label='Energy', unit='TJ/a')
    binding_spec = DatasetPortSpec.from_input_dataset(
        InputDatasetDef(
            id='multi_metric_actions',
            forecast_from=2024,
            filters=[ColumnDatasetFilterDef(column='action', value='multi_metric_action')],
        )
    )

    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='multi_metric_action',
        name='Multi metric action',
        spec=NodeSpec(
            type_config=ActionConfig(
                node_class=ACTION_NODE_CLASS,
                decision_level=DecisionLevel.MUNICIPALITY,
            ),
            input_ports=[
                _make_input_port(id='emissions', unit='t/a', quantity='emissions'),
                _make_input_port(id='energy', unit='TJ/a', quantity='energy'),
            ],
            output_ports=[
                OutputPortDef(
                    id=_port_uuid('emissions'),
                    unit=unit_registry.parse_units('t/a'),
                    quantity='emissions',
                    column_id='emissions',
                ),
                OutputPortDef(
                    id=_port_uuid('energy'),
                    unit=unit_registry.parse_units('TJ/a'),
                    quantity='energy',
                    column_id='energy',
                ),
            ],
        ),
    )
    DatasetPort.objects.create(
        instance=db_instance_config,
        node=nc,
        port_id=_port_uuid('emissions'),
        dataset=dataset,
        metric=emissions_metric,
        spec=binding_spec,
    )
    DatasetPort.objects.create(
        instance=db_instance_config,
        node=nc,
        port_id=_port_uuid('energy'),
        dataset=dataset,
        metric=energy_metric,
        spec=binding_spec,
    )

    ctx = _rebuild_from_db(db_instance_config)
    action = ctx.nodes['multi_metric_action']
    assert len(action.input_dataset_instances) == 1
    ds = cast('DatasetWithFilters', action.input_dataset_instances[0])
    assert ds.id == 'multi_metric_actions'
    assert ds.column is None
    assert forecast_from_transformations(ds.transformations) == 2024
    filter_op = next(op for op in ds.transformations if isinstance(op, FilterColumnOp))
    assert filter_op.column == 'action'
    assert filter_op.value == 'multi_metric_action'


def test_dataset_ports_rebuild_uses_dataset_forecast_default(db_instance_config: InstanceConfig):
    from nodes.models import DatasetPort

    assert db_instance_config.spec is not None
    db_instance_config.spec.features.use_datasets_from_db = True
    db_instance_config.save(update_fields=['spec'])

    dataset = DatasetFactory.create(identifier='forecast_default_dataset', scope=db_instance_config, spec={'forecast_from': 2026})
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='emissions', label='Emissions', unit='t/a')
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='uses_forecast_default',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class=SIMPLE_NODE_CLASS),
            input_ports=[_make_input_port(id='emissions', unit='t/a', quantity='emissions')],
            output_ports=[
                OutputPortDef(
                    id=_port_uuid('emissions'),
                    unit=unit_registry.parse_units('t/a'),
                    quantity='emissions',
                    column_id='emissions',
                ),
            ],
        ),
    )
    DatasetPort.objects.create(
        instance=db_instance_config,
        node=nc,
        port_id=_port_uuid('emissions'),
        dataset=dataset,
        metric=metric,
    )

    ctx = _rebuild_from_db(db_instance_config)
    node = ctx.nodes['uses_forecast_default']
    assert len(node.input_dataset_instances) == 1
    ds = cast('DatasetWithFilters', node.input_dataset_instances[0])
    assert ds.forecast_from == 2026


def test_dataset_port_sync_uses_one_port_per_dataset_metric(db_instance_config: InstanceConfig):
    from types import SimpleNamespace

    from nodes.datasets import DBDataset
    from nodes.defs.node_defs import ColumnDatasetFilterDef
    from nodes.models import DatasetPort
    from nodes.node import NodeMetric
    from nodes.spec_export import _export_input_ports, _update_dataset_ports

    dataset = DatasetFactory.create(identifier='sync_multi_metric_actions', scope=db_instance_config)
    DatasetMetricFactory.create(schema=dataset.schema, name='emissions', label='Emissions', unit='t/a')
    DatasetMetricFactory.create(schema=dataset.schema, name='energy', label='Energy', unit='TJ/a')

    context = cast('Context', SimpleNamespace(instance=SimpleNamespace(config=db_instance_config)))
    ds_instance = DBDataset(
        id='sync_multi_metric_actions',
        context=context,
        db_dataset_obj=dataset,
        transformations=InputDatasetDef(
            id='sync_multi_metric_actions',
            forecast_from=2024,
            filters=[ColumnDatasetFilterDef(column='action', value='multi_metric_action')],
        ).to_transformations(),
        forecast_from=2024,
    )
    node = cast(
        'Node',
        SimpleNamespace(
            id='multi_metric_action',
            context=context,
            input_dataset_instances=[ds_instance],
            output_metrics={
                'emissions': NodeMetric(unit='t/a', quantity='emissions', id='emissions', column_id='emissions'),
                'energy': NodeMetric(unit='TJ/a', quantity='energy', id='energy', column_id='energy'),
            },
            edges=[],
            input_dimensions={},
        ),
    )
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='multi_metric_action',
        spec=NodeSpec(
            type_config=ActionConfig(
                node_class=ACTION_NODE_CLASS,
                decision_level=DecisionLevel.MUNICIPALITY,
            ),
        ),
    )

    input_ports = _export_input_ports(node)
    assert [port.quantity for port in input_ports] == ['emissions', 'energy']

    ctx = cast('Context', SimpleNamespace(nodes={'multi_metric_action': node}))
    assert _update_dataset_ports(db_instance_config, ctx, {'multi_metric_action': nc}) == 2
    bindings = list(DatasetPort.objects.filter(node=nc).select_related('metric').order_by('metric__name'))
    assert [binding.metric.name for binding in bindings] == ['emissions', 'energy']
    assert {binding.port_id for binding in bindings} == {port.id for port in input_ports}
    assert all(binding.spec.forecast_from == 2024 for binding in bindings)
    for binding in bindings:
        filter_op = next(op for op in binding.spec.transformations if isinstance(op, FilterColumnOp))
        assert filter_op.column == 'action'


def _column_less_sync_fixture(
    db_instance_config: InstanceConfig,
    *,
    metric_names: list[str],
    node_columns: list[str],
):
    """Build a runtime node + DB dataset pair for exercising the metric-to-port pairing."""
    from types import SimpleNamespace

    from nodes.datasets import DBDataset
    from nodes.node import NodeMetric

    dataset = DatasetFactory.create(identifier='pairing_dataset', scope=db_instance_config)
    for name in metric_names:
        DatasetMetricFactory.create(schema=dataset.schema, name=name, label=name.title(), unit='t/a')

    context = cast('Context', SimpleNamespace(instance=SimpleNamespace(config=db_instance_config)))
    ds_instance = DBDataset(
        id='pairing_dataset',
        context=context,
        db_dataset_obj=dataset,
        transformations=InputDatasetDef(id='pairing_dataset').to_transformations(),
    )
    node = cast(
        'Node',
        SimpleNamespace(
            id='pairing_node',
            context=context,
            input_dataset_instances=[ds_instance],
            output_metrics={
                column: NodeMetric(unit='t/a', quantity='emissions', id=column.lower(), column_id=column)
                for column in node_columns
            },
            edges=[],
            input_dimensions={},
        ),
    )
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='pairing_node',
        spec=NodeSpec(type_config=SimpleConfig(node_class=SIMPLE_NODE_CLASS)),
    )
    return node, nc


def test_dataset_port_sync_pairs_a_renamed_metric_to_the_node_column(db_instance_config: InstanceConfig):
    """
    A schema metric named differently from the node column must still bind to a real port.

    The port UUID comes from the node-side column (where the metric is
    delivered); the metric FK names the source. Keying the port by the schema
    metric name produced bindings pointing at ports absent from the node spec.
    """
    from types import SimpleNamespace

    from nodes.models import DatasetPort
    from nodes.spec_export import _export_input_ports, _update_dataset_ports

    node, nc = _column_less_sync_fixture(db_instance_config, metric_names=['share'], node_columns=['Value'])

    input_ports = _export_input_ports(node)
    ctx = cast('Context', SimpleNamespace(nodes={'pairing_node': node}))
    assert _update_dataset_ports(db_instance_config, ctx, {'pairing_node': nc}) == 1

    binding = DatasetPort.objects.get(node=nc)
    assert binding.metric.name == 'share'
    assert binding.port_id in {port.id for port in input_ports}


def test_dataset_port_sync_drops_unmatched_extra_metrics(db_instance_config: InstanceConfig):
    """A metric with no defensible port gets no binding, as long as the dataset keeps at least one row."""
    from types import SimpleNamespace

    from nodes.models import DatasetPort
    from nodes.spec_export import _update_dataset_ports

    node, nc = _column_less_sync_fixture(
        db_instance_config, metric_names=['emissions', 'foo', 'bar'], node_columns=['emissions', 'energy']
    )

    ctx = cast('Context', SimpleNamespace(nodes={'pairing_node': node}))
    assert _update_dataset_ports(db_instance_config, ctx, {'pairing_node': nc}) == 1
    assert DatasetPort.objects.get(node=nc).metric.name == 'emissions'


def test_dataset_port_sync_keeps_an_unpairable_binding_alive(db_instance_config: InstanceConfig):
    """
    When nothing pairs, the rows stay (dangling) rather than disappear.

    ``_serialize_dataset_ports`` rebuilds ``input_datasets`` from these rows,
    so zero rows would silently remove the dataset from DB-sourced models —
    worse than an editor binding whose port id is unresolved.
    """
    from types import SimpleNamespace

    from nodes.models import DatasetPort
    from nodes.spec_export import _update_dataset_ports

    node, nc = _column_less_sync_fixture(db_instance_config, metric_names=['foo', 'bar'], node_columns=['emissions', 'energy'])

    ctx = cast('Context', SimpleNamespace(nodes={'pairing_node': node}))
    assert _update_dataset_ports(db_instance_config, ctx, {'pairing_node': nc}) == 2
    assert {dp.metric.name for dp in DatasetPort.objects.filter(node=nc)} == {'foo', 'bar'}


def test_dataset_port_forecast_from_promotes_to_dataset_default(db_instance_config: InstanceConfig):
    from nodes.dataset_materialization import materialize_dataset
    from nodes.defs.node_defs import DatasetPortSpec
    from nodes.models import DatasetMaterialization, DatasetPort
    from nodes.spec_export import _promote_dataset_forecast_defaults

    promoted_dataset = DatasetFactory.create(identifier='promoted', scope=db_instance_config)
    promoted_metric = DatasetMetricFactory.create(schema=promoted_dataset.schema, name='value', label='Value', unit='kt/a')
    original_materialization = materialize_dataset(promoted_dataset)
    conflict_dataset = DatasetFactory.create(identifier='conflict', scope=db_instance_config)
    conflict_metric = DatasetMetricFactory.create(schema=conflict_dataset.schema, name='value', label='Value', unit='kt/a')

    node_a = NodeConfigFactory.create(instance=db_instance_config, identifier='node_a', spec=_make_node_spec())
    node_b = NodeConfigFactory.create(instance=db_instance_config, identifier='node_b', spec=_make_node_spec())
    node_c = NodeConfigFactory.create(instance=db_instance_config, identifier='node_c', spec=_make_node_spec())
    node_d = NodeConfigFactory.create(instance=db_instance_config, identifier='node_d', spec=_make_node_spec())

    DatasetPort.objects.create(
        instance=db_instance_config,
        node=node_a,
        port_id=_port_uuid('input_a'),
        dataset=promoted_dataset,
        metric=promoted_metric,
        spec=DatasetPortSpec.from_input_dataset(InputDatasetDef(id='ds', forecast_from=2025)),
    )
    DatasetPort.objects.create(
        instance=db_instance_config,
        node=node_b,
        port_id=_port_uuid('input_b'),
        dataset=promoted_dataset,
        metric=promoted_metric,
    )
    DatasetPort.objects.create(
        instance=db_instance_config,
        node=node_c,
        port_id=_port_uuid('input_c'),
        dataset=conflict_dataset,
        metric=conflict_metric,
        spec=DatasetPortSpec.from_input_dataset(InputDatasetDef(id='ds', forecast_from=2024)),
    )
    DatasetPort.objects.create(
        instance=db_instance_config,
        node=node_d,
        port_id=_port_uuid('input_d'),
        dataset=conflict_dataset,
        metric=conflict_metric,
        spec=DatasetPortSpec.from_input_dataset(InputDatasetDef(id='ds', forecast_from=2025)),
    )

    assert _promote_dataset_forecast_defaults(db_instance_config) == 1

    promoted_dataset.refresh_from_db()
    conflict_dataset.refresh_from_db()
    assert promoted_dataset.spec == {'forecast_from': 2025}
    assert conflict_dataset.spec == {}

    promoted_ports = DatasetPort.objects.filter(dataset=promoted_dataset)
    assert all(port.spec.forecast_from is None for port in promoted_ports)
    assert sorted((port.spec.forecast_from or 0) for port in DatasetPort.objects.filter(dataset=conflict_dataset)) == [2024, 2025]

    materialization = DatasetMaterialization.objects.get(dataset=promoted_dataset)
    assert materialization.generation == original_materialization.generation + 1
    assert materialization.forecast_from == 2025
    assert materialization.content['forecast_from'] == 2025

    # Repair materializations left inconsistent by syncs that predate the atomic refresh.
    materialization.forecast_from = None
    materialization.content['forecast_from'] = None
    materialization.save(update_fields=['forecast_from', 'content'])

    assert _promote_dataset_forecast_defaults(db_instance_config) == 0
    materialization.refresh_from_db()
    assert materialization.generation == original_materialization.generation + 2
    assert materialization.forecast_from == 2025
    assert materialization.content['forecast_from'] == 2025


def test_dataset_port_forecast_from_not_promoted_for_external_placeholder(db_instance_config: InstanceConfig):
    """
    Promotion must not clear the binding-level forecast_from for external placeholders.

    External placeholders load via plain DVCDataset at runtime, which has no fallback to
    Dataset.spec.forecast_from (only DBDataset.from_def does). Promoting for these would clear
    the binding-level value with nothing left to read it back.
    """
    from nodes.defs.node_defs import DatasetPortSpec
    from nodes.models import DatasetPort
    from nodes.spec_export import _promote_dataset_forecast_defaults

    placeholder_dataset = DatasetFactory.create(identifier='placeholder', scope=db_instance_config, is_external_placeholder=True)
    placeholder_metric = DatasetMetricFactory.create(schema=placeholder_dataset.schema, name='value', label='Value', unit='kt/a')
    node_a = NodeConfigFactory.create(instance=db_instance_config, identifier='node_a', spec=_make_node_spec())

    DatasetPort.objects.create(
        instance=db_instance_config,
        node=node_a,
        port_id=_port_uuid('input_a'),
        dataset=placeholder_dataset,
        metric=placeholder_metric,
        spec=DatasetPortSpec.from_input_dataset(InputDatasetDef(id='ds', forecast_from=2025)),
    )

    assert _promote_dataset_forecast_defaults(db_instance_config) == 0

    placeholder_dataset.refresh_from_db()
    assert placeholder_dataset.spec == {}
    port = DatasetPort.objects.get(dataset=placeholder_dataset)
    assert port.spec.forecast_from == 2025


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
            model {
                nodes {
                    identifier
                }
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
    public_model_ids = {node['identifier'] for node in public_data['instance']['model']['nodes']}
    assert 'visible_node' in public_ids
    assert 'hidden_node' not in public_ids
    assert public_model_ids == public_ids

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
    editor_model_ids = {node['identifier'] for node in editor_data['instance']['model']['nodes']}
    assert 'visible_node' in editor_ids
    assert 'hidden_node' in editor_ids
    assert editor_model_ids == editor_ids


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
            'nodeId': str(nc.uuid),
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

    data = gql_client.query_data(
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
    result = data['instanceEditor']['createEdge']
    assert result['__typename'] == 'ConstraintViolations'
    codes = {conflict['code'] for conflict in result['conflicts']}
    assert 'quantity_mismatch' in codes
    from nodes.models import NodeEdge

    assert not NodeEdge.objects.filter(instance=db_instance_config).exists()


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
    assert edge['portRef']['portId'] == str(_port_uuid('input'))


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
        variables={'instanceId': str(db_instance_config.pk), 'nodeId': str(nc.uuid)},
    )

    ctx = _rebuild_from_db(db_instance_config)
    assert 'ephemeral' not in ctx.nodes


# ---------------------------------------------------------------------------
# Step 8: solver-backed validation, derived fields, and the role catalog
# ---------------------------------------------------------------------------

MULTIPLICATIVE_NODE_CLASS = 'nodes.simple.MultiplicativeNode'

CONSTRAINT_CONFLICTS = gql("""
query ConstraintConflicts {
    instance {
        editor {
            constraintConflicts {
                code
                message
                value { kind nodeUuid portId direction bindingId }
                origins { kind nodeUuid portId bindingId }
            }
        }
    }
}
""")

NODE_CONSTRAINT_FIELDS = gql("""
query NodeConstraintFields($instanceId: ID!) {
    modelInstance(instanceId: $instanceId) {
        nodes {
            identifier
            editor {
                spec {
                    constraintConflicts { code }
                    inputPorts {
                        identifier
                        role
                        effectiveShape {
                            dimensionUuids
                            unit { standard }
                            quantity
                        }
                    }
                    outputPorts {
                        identifier
                        effectiveShape { dimensionUuids }
                    }
                    inputPortDeclarations {
                        role
                        multi
                        repeatable
                        minCount
                        defaultCount
                        instantiatedPortIds
                    }
                    supportsAuthoredPorts
                }
            }
        }
    }
}
""")


def _quantity_mismatch_pair(ic: InstanceConfig) -> None:
    """Wire an emissions source straight into an energy-expecting port, bypassing validation."""
    from nodes.models import NodeEdge

    unit = unit_registry.parse_units('kt/a')
    src = NodeConfigFactory.create(
        instance=ic,
        identifier='mismatch_src',
        spec=_make_node_spec(output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')]),
    )
    dst = NodeConfigFactory.create(
        instance=ic,
        identifier='mismatch_dst',
        spec=_make_node_spec(
            input_ports=[InputPortDef(id=_port_uuid('input'), identifier='input', unit=unit, quantity='energy')],
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='energy')],
        ),
    )
    NodeEdge.objects.create(
        instance=ic,
        from_node=src,
        from_port=_port_uuid('default'),
        to_node=dst,
        to_port=_port_uuid('input'),
    )


def test_draft_constraint_conflicts_are_inspectable(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    _quantity_mismatch_pair(db_instance_config)

    data = gql_client.query_data(CONSTRAINT_CONFLICTS)
    conflicts = data['instance']['editor']['constraintConflicts']
    codes = {conflict['code'] for conflict in conflicts}
    assert 'quantity_mismatch' in codes
    # Origins carry UUID provenance for the editor to highlight.
    mismatch = next(conflict for conflict in conflicts if conflict['code'] == 'quantity_mismatch')
    assert any(origin['nodeUuid'] or origin['bindingId'] for origin in mismatch['origins'])


def test_node_level_conflicts_and_effective_shapes(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    _register_dimensions(db_instance_config, ['sector'], {'sector': ['industry']})
    _quantity_mismatch_pair(db_instance_config)

    data = gql_client.query_data(NODE_CONSTRAINT_FIELDS, variables={'instanceId': str(db_instance_config.pk)})
    node = next(entry for entry in data['modelInstance']['nodes'] if entry['identifier'] == 'mismatch_dst')
    spec = node['editor']['spec']
    assert 'quantity_mismatch' in {conflict['code'] for conflict in spec['constraintConflicts']}
    (port,) = spec['inputPorts']
    shape = port['effectiveShape']
    assert shape is not None
    assert shape['unit']['standard'] == 'kt/a'
    assert spec['supportsAuthoredPorts'] is False


def test_input_port_declaration_catalog(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    unit = unit_registry.parse_units('kt/a')
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='product',
        spec=_make_node_spec(
            type_config=SimpleConfig(node_class=MULTIPLICATIVE_NODE_CLASS),
            input_ports=[InputPortDef(id=_port_uuid('factor1'), identifier='factor', role='factors')],
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit, quantity='emissions')],
        ),
    )

    data = gql_client.query_data(NODE_CONSTRAINT_FIELDS, variables={'instanceId': str(db_instance_config.pk)})
    node = next(entry for entry in data['modelInstance']['nodes'] if entry['identifier'] == 'product')
    declarations = {entry['role']: entry for entry in node['editor']['spec']['inputPortDeclarations']}
    assert set(declarations) == {'factors', 'additive', 'impute'}
    factors = declarations['factors']
    assert factors['repeatable'] is True
    assert factors['minCount'] == 1
    assert factors['defaultCount'] == 2
    assert factors['instantiatedPortIds'] == [str(_port_uuid('factor1'))]
    assert declarations['additive']['multi'] is True


def test_create_node_instantiates_default_declared_ports(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeConfig

    gql_client.query_data(
        CREATE_NODE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'identifier': 'product_node',
                'name': 'Product Node',
                'kind': 'SIMPLE',
                'config': {'simple': {'nodeClass': MULTIPLICATIVE_NODE_CLASS}},
                'outputPorts': [{'unit': 'kt/a', 'quantity': 'emissions'}],
            },
        },
    )
    nc = NodeConfig.objects.get(instance=db_instance_config, identifier='product_node')
    assert nc.spec is not None
    ports = [(str(port.identifier), str(port.role), port.multi) for port in nc.spec.input_ports]
    assert ports == [
        ('factors', 'factors', False),
        ('factors2', 'factors', False),
        ('additive', 'additive', True),
    ]


def test_connect_instantiates_a_declared_factor_port(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeConfig, NodeEdge

    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='ef',
        spec=_make_node_spec(
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit_registry.parse_units('kg/vkm'))],
        ),
    )
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='mileage',
        spec=_make_node_spec(
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit_registry.parse_units('vkm/a'))],
        ),
    )
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='product',
        spec=_make_node_spec(
            type_config=SimpleConfig(node_class=MULTIPLICATIVE_NODE_CLASS),
            input_ports=[
                InputPortDef(
                    id=_port_uuid('factor1'),
                    identifier='factor',
                    role='factors',
                    unit=unit_registry.parse_units('kg/vkm'),
                ),
            ],
            output_ports=[OutputPortDef(id=_port_uuid('default'), unit=unit_registry.parse_units('kg/a'))],
        ),
    )

    def connect(from_node: str) -> dict[str, Any]:
        return gql_client.query_data(
            CREATE_EDGE,
            variables={
                'instanceId': str(db_instance_config.pk),
                'input': {
                    'instanceId': str(db_instance_config.pk),
                    'fromNodeId': from_node,
                    'toNodeId': 'product',
                },
            },
        )['instanceEditor']['createEdge']

    # First connection lands on the existing sole factor port.
    first = connect('ef')
    assert first['__typename'] == 'NodeEdgeType'
    assert first['portRef']['portId'] == str(_port_uuid('factor1'))

    # The sole port is now occupied: the second connection instantiates a
    # fresh port of the repeatable `factors` role instead of being rejected.
    second = connect('mileage')
    assert second['__typename'] == 'NodeEdgeType'
    assert second['portRef']['portId'] != str(_port_uuid('factor1'))

    nc = NodeConfig.objects.get(instance=db_instance_config, identifier='product')
    assert nc.spec is not None
    new_port = nc.spec.input_ports[-1]
    assert str(new_port.role) == 'factors'
    assert str(new_port.identifier) == 'factors'
    assert NodeEdge.objects.filter(to_node=nc, to_port=new_port.id).exists()


PUBLISH_INSTANCE = gql("""
mutation PublishInstance($instanceId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        publishModelInstance(instanceId: $instanceId) {
            __typename
            ... on InstanceType { identifier }
            ... on ConstraintViolations { conflicts { code } }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
""")


def test_publication_is_blocked_by_constraint_conflicts(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    _quantity_mismatch_pair(db_instance_config)

    result = gql_client.query_data(
        PUBLISH_INSTANCE,
        variables={'instanceId': str(db_instance_config.pk)},
    )['instanceEditor']['publishModelInstance']
    assert result['__typename'] == 'ConstraintViolations'
    assert 'quantity_mismatch' in {conflict['code'] for conflict in result['conflicts']}
    db_instance_config.refresh_from_db()
    assert db_instance_config.live_revision_id is None


def test_publication_succeeds_once_the_conflict_is_gone(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeEdge

    _quantity_mismatch_pair(db_instance_config)
    NodeEdge.objects.filter(instance=db_instance_config).delete()

    result = gql_client.query_data(
        PUBLISH_INSTANCE,
        variables={'instanceId': str(db_instance_config.pk)},
    )['instanceEditor']['publishModelInstance']
    assert result['__typename'] == 'InstanceType'
    db_instance_config.refresh_from_db()
    assert db_instance_config.live_revision_id is not None
