"""Editing an action's hooks through the model editor (`addActionHook`, `deleteActionHook`)."""

from typing import TYPE_CHECKING, Any

import pytest

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import ActionConfig, ActionHookDef, NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory, _port_id
from nodes.units import unit_registry

if TYPE_CHECKING:
    from paths.tests.graphql import PathsTestClient

    from nodes.models import InstanceConfig, NodeConfig

pytestmark = pytest.mark.django_db

gql = str

ADD_HOOK = gql("""
mutation AddHook($instanceId: ID!, $nodeId: ID!, $input: ActionHookInput!) {
    instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
            addActionHook(input: $input) {
                __typename
                ... on NodeInterface {
                    editor { spec { typeConfig { ... on ActionConfigType { hooks { node port fromPort } } } } }
                }
                ... on OperationInfo { messages { kind message } }
            }
        }
    }
}
""")

DELETE_HOOK = gql("""
mutation DeleteHook($instanceId: ID!, $nodeId: ID!, $input: ActionHookInput!) {
    instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
            deleteActionHook(input: $input) {
                __typename
                ... on OperationInfo { messages { kind message } }
            }
        }
    }
}
""")

UPDATE_NODE = gql("""
mutation UpdateNode($instanceId: ID!, $nodeId: ID!, $input: UpdateNodeInput!) {
    instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
            update(input: $input) { __typename ... on OperationInfo { messages { kind message } } }
        }
    }
}
""")


def _output(name: str, unit: str = 'kt/a', quantity: str = 'emissions') -> OutputPortDef:
    return OutputPortDef(
        id=_port_id(name), identifier=name, column_id=name, unit=unit_registry.parse_units(unit), quantity=quantity
    )


@pytest.fixture
def ic() -> InstanceConfig:
    instance = InstanceFactory.create()
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        owner='Test Owner',
        spec=InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030)),
    )


@pytest.fixture
def client(client, ic: InstanceConfig) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(ic)
    return tc


def _node(ic: InstanceConfig, identifier: str, *outputs: OutputPortDef) -> NodeConfig:
    spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.SimpleNode'), output_ports=list(outputs or [_output('out')])
    )
    return NodeConfigFactory.create(instance=ic, identifier=identifier, spec=spec)


def _action(
    ic: InstanceConfig, identifier: str = 'saving', *outputs: OutputPortDef, hooks: list[ActionHookDef] | None = None
) -> NodeConfig:
    spec = NodeSpec(
        type_config=ActionConfig(node_class='nodes.actions.simple.AdditiveAction', hooks=hooks or []),
        output_ports=list(outputs or [_output(f'{identifier}_out')]),
    )
    return NodeConfigFactory.create(instance=ic, identifier=identifier, spec=spec)


def _hooks(nc: NodeConfig) -> list[ActionHookDef]:
    from nodes.models import NodeConfig

    # The default manager defers `spec`, so `refresh_from_db()` would keep the stale value.
    spec = NodeConfig.objects.with_spec().get(pk=nc.pk).spec
    assert spec is not None
    assert isinstance(spec.type_config, ActionConfig)
    return spec.type_config.hooks


def _add(client: PathsTestClient, ic: InstanceConfig, action: NodeConfig, **input: Any) -> dict[str, Any]:
    variables = {'instanceId': ic.identifier, 'nodeId': str(action.uuid), 'input': input}
    return client.query_data(ADD_HOOK, variables=variables)['instanceEditor']['nodeEditor']['addActionHook']


def _add_errors(client: PathsTestClient, ic: InstanceConfig, action: NodeConfig, **input: Any) -> str:
    variables = {'instanceId': ic.identifier, 'nodeId': str(action.uuid), 'input': input}
    return ' '.join(error.get('message', '') for error in client.query_errors(ADD_HOOK, variables=variables))


def test_add_hook_resolves_both_ports_and_exposes_it(client: PathsTestClient, ic: InstanceConfig) -> None:
    target = _node(ic, 'demand')
    action = _action(ic)

    result = _add(client, ic, action, targetNodeId=str(target.uuid))

    hook = ActionHookDef(node='demand', port=_port_id('out'), from_port=_port_id('saving_out'))
    assert _hooks(action) == [hook]
    assert result['editor']['spec']['typeConfig']['hooks'] == [
        {'node': 'demand', 'port': str(hook.port), 'fromPort': str(hook.from_port)}
    ]


def test_add_hook_refuses_an_incompatible_unit(client: PathsTestClient, ic: InstanceConfig) -> None:
    target = _node(ic, 'demand', _output('out', 'MWh/a', 'energy'))
    action = _action(ic)

    assert 'cannot be converted' in _add_errors(client, ic, action, targetNodeId=str(target.uuid))
    assert _hooks(action) == []


def test_add_hook_refuses_a_loop_through_an_edge(client: PathsTestClient, ic: InstanceConfig) -> None:
    from nodes.models import NodeConfig, NodeInputPortBinding

    target = _node(ic, 'demand')
    action = _action(ic)
    spec = action.spec
    assert spec is not None
    spec.input_ports = [InputPortDef(id=_port_id('in'), identifier='in', unit=unit_registry.parse_units('kt/a'))]
    NodeConfig.objects.filter(pk=action.pk).update(spec=spec)
    NodeInputPortBinding.objects.create(
        instance=ic,
        node=action,
        port_id=_port_id('in'),
        position=0,
        source_node=target,
        source_port_id=_port_id('out'),
    )

    assert 'loop' in _add_errors(client, ic, action, targetNodeId=str(target.uuid))


def test_add_hook_refuses_a_loop_through_another_hook(client: PathsTestClient, ic: InstanceConfig) -> None:
    first = _action(ic, 'first')
    second = _action(ic, 'second', hooks=[ActionHookDef(node='first')])

    assert 'loop' in _add_errors(client, ic, first, targetNodeId=str(second.uuid))


def test_add_hook_needs_the_port_when_the_target_has_several(client: PathsTestClient, ic: InstanceConfig) -> None:
    target = _node(ic, 'demand', _output('a'), _output('b'))
    action = _action(ic)

    assert 'name the target port' in _add_errors(client, ic, action, targetNodeId=str(target.uuid))
    _add(client, ic, action, targetNodeId=str(target.uuid), targetPortId=str(_port_id('b')))
    assert [hook.port for hook in _hooks(action)] == [_port_id('b')]


def test_add_hook_refuses_the_same_hook_twice(client: PathsTestClient, ic: InstanceConfig) -> None:
    target = _node(ic, 'demand')
    action = _action(ic)
    _add(client, ic, action, targetNodeId=str(target.uuid))

    assert 'already acts' in _add_errors(client, ic, action, targetNodeId=str(target.uuid))


def test_only_actions_have_hooks(client: PathsTestClient, ic: InstanceConfig) -> None:
    target = _node(ic, 'demand')
    other = _node(ic, 'other')

    assert 'Only actions' in _add_errors(client, ic, other, targetNodeId=str(target.uuid))


def test_delete_hook(client: PathsTestClient, ic: InstanceConfig) -> None:
    target = _node(ic, 'demand')
    action = _action(ic, hooks=[ActionHookDef(node='demand')])

    variables = {'instanceId': ic.identifier, 'nodeId': str(action.uuid), 'input': {'targetNodeId': str(target.uuid)}}
    client.query_data(DELETE_HOOK, variables=variables)

    assert _hooks(action) == []


def test_updating_the_action_config_keeps_its_hooks(client: PathsTestClient, ic: InstanceConfig) -> None:
    """`ActionConfigInput` has no hooks, so replacing the config must not drop them."""
    _node(ic, 'demand')
    action = _action(ic, hooks=[ActionHookDef(node='demand')])

    config = {'action': {'nodeClass': 'nodes.actions.simple.AdditiveAction', 'noEffectValue': 0.0}}
    variables = {'instanceId': ic.identifier, 'nodeId': str(action.uuid), 'input': {'config': config}}
    client.query_data(UPDATE_NODE, variables=variables)

    assert _hooks(action) == [ActionHookDef(node='demand')]


def test_the_draft_graph_orders_an_action_before_the_node_it_acts_on(ic: InstanceConfig) -> None:
    from nodes.instance_graph_cache import get_instance_graph
    from nodes.models import PreferredInstanceSource

    target = _node(ic, 'demand')
    action = _action(ic, hooks=[ActionHookDef(node='demand')])

    graph = get_instance_graph(ic, PreferredInstanceSource.DRAFT, refresh=True)
    assert (action.uuid, target.uuid) in graph.hook_edges
    assert graph.nx_graph.has_edge(action.uuid, target.uuid)
