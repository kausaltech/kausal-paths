"""
Hooks in the instance editor: where an action may act, and adding and removing a hook.

See docs/architecture/action-hooks.md.
"""

from typing import Any

import pytest

from paths.tests.graphql import PathsTestClient

from frameworks.models import FrameworkConfig
from frameworks.tests.factories import FrameworkFactory
from nodes.action_effects import ValueShape, hook_source_problem
from nodes.constants import VALUE_COLUMN
from nodes.defs.instance_defs import InstanceModelSpec
from nodes.defs.node_defs import ActionConfig, ActionHookDef, ActionHookMode, NodeSpec, NodeSpecExtra, SimpleConfig
from nodes.defs.port_def import InputPortDef
from nodes.graphql.action_wizard import ADDITIVE_ACTION_CLASS
from nodes.instance_serialization import build_instance_snapshot
from nodes.models import InstanceConfig, NodeConfig
from nodes.template_graph import publish_template_instance, upgrade_template_instance
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory, _port_id
from nodes.tests.test_action_wizard import (
    CARRIERS,
    SECTORS,
    YEARS,
    _bind,
    _by_year,
    _category,
    _dataset,
    _demand,
    _dimension,
    _history,
    _port,
    _rebuild,
    register_dimensions,
)
from nodes.units import unit_registry
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db

gql = str

EDITOR = gql("""
query Editor($instanceId: ID!) {
    modelInstance(instanceId: $instanceId) {
        editor {
            hooks { actionId fromPortId targetNodeId targetPortId mode }
            actionHookTargets {
                nodeId portId quantity unit { short }
                dimensions { dimensionId categoryIds }
            }
            compositionErrors
        }
    }
}
""")

HOOK = """
mutation Hook($instanceId: ID!, $nodeId: ID!, $input: ActionHookInput!) {
    instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
            %s(input: $input) {
                ... on NodeInterface { identifier }
                ... on OperationInfo { messages { kind message } }
            }
        }
    }
}
"""
ADD = gql(HOOK % 'addActionHook')
DELETE = gql(HOOK % 'deleteActionHook')

UPDATE = gql("""
mutation Update($instanceId: ID!, $nodeId: ID!, $input: UpdateNodeInput!) {
    instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
            update(input: $input) { ... on NodeInterface { identifier } }
        }
    }
}
""")


def make_instance() -> InstanceConfig:
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(years=YEARS)
    spec.features.use_datasets_from_db = True
    ic = InstanceConfigFactory.create(
        identifier=instance.id, instance=instance, config_source='database', owner='Test Owner', spec=spec
    )
    register_dimensions(ic, ['carrier', 'sector'], {'carrier': CARRIERS, 'sector': SECTORS})
    return ic


def make_municipality(template: InstanceConfig) -> InstanceConfig:
    """Make `template` a framework template with one node open to actions and one closed; return a municipality inheriting it."""
    _heat(template, action_hooks=True, is_outcome=True)
    _demand(template)
    framework = FrameworkFactory.create(template_instance=template)
    revision = publish_template_instance(template)
    spec = InstanceModelSpec(years=YEARS)
    municipality = InstanceConfigFactory.create(
        name='Municipality', owner='Test Owner', config_source='database', template_revision=revision, spec=spec
    )
    FrameworkConfig.objects.create(framework=framework, instance_config=municipality)
    return municipality


@pytest.fixture
def ic() -> InstanceConfig:
    return make_instance()


@pytest.fixture
def client(client, ic: InstanceConfig) -> PathsTestClient:
    client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(ic)
    return tc


def _heat(ic: InstanceConfig, *, action_hooks: bool = False, is_outcome: bool = False) -> NodeConfig:
    """Create an additive node without dimensions: 100 MWh/a in every year."""
    dataset = _dataset(ic, 'test/heat', 'MWh/a', _history([()], 100.0), ())
    spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
        input_ports=[
            InputPortDef(id=_port_id('heat_in'), identifier='in', unit=unit_registry.parse_units('MWh/a'), quantity='energy')
        ],
        output_ports=[_port('heat', 'MWh/a', 'energy', [])],
        action_hooks=action_hooks,
        is_outcome=is_outcome,
    )
    nc = NodeConfigFactory.create(instance=ic, identifier='heat', name='Heat', spec=spec)
    _bind(ic, nc, 'heat_in', dataset)
    return nc


def _saving(ic: InstanceConfig, *, unit: str = 'MWh/a', quantity: str = 'energy', value: float = -10) -> NodeConfig:
    """Create an action without dimensions that gives `value` from 2030 on, and half of it in the last historical year."""
    spec = NodeSpec(
        type_config=ActionConfig(node_class=ADDITIVE_ACTION_CLASS),
        output_ports=[_port('saving', unit, quantity, []).model_copy(update={'column_id': VALUE_COLUMN})],
        # The historical value is the realised part of the measure; it must never enter the balance.
        extra=NodeSpecExtra(historical_values=[(2018, 0), (2022, value / 2)], forecast_values=[(2030, value)]),
    )
    return NodeConfigFactory.create(instance=ic, identifier='saving', name='Saving', spec=spec)


def _editor(client: PathsTestClient, ic: InstanceConfig) -> dict[str, Any]:
    return client.query_data(EDITOR, variables={'instanceId': ic.identifier})['modelInstance']['editor']


def _hook(target: NodeConfig, **extra: Any) -> dict[str, Any]:
    return {'targetNodeId': str(target.uuid), **extra}


def _add(client: PathsTestClient, ic: InstanceConfig, action: NodeConfig, input: dict[str, Any]) -> None:
    client.query_data(ADD, variables={'instanceId': ic.identifier, 'nodeId': str(action.uuid), 'input': input})


def _refused(client: PathsTestClient, ic: InstanceConfig, action: NodeConfig, input: dict[str, Any], query: str = ADD) -> str:
    variables = {'instanceId': ic.identifier, 'nodeId': str(action.uuid), 'input': input}
    return ' '.join(error.get('message', '') for error in client.query_errors(query, variables=variables))


def _hooks(action: NodeConfig) -> list[ActionHookDef]:
    spec = NodeConfig.objects.with_spec().get(pk=action.pk).spec
    assert spec is not None
    assert isinstance(spec.type_config, ActionConfig)
    return spec.type_config.hooks


def test_targets_are_output_ports_with_the_shape_an_effect_on_them_has(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand, heat, saving = _demand(ic), _heat(ic), _saving(ic)

    targets = {target['nodeId']: target for target in _editor(client, ic)['actionHookTargets']}

    # An instance's own nodes need no permission; an action is not a target.
    assert set(targets) == {str(demand.uuid), str(heat.uuid)}
    assert str(saving.uuid) not in targets
    target = targets[str(demand.uuid)]
    assert (target['portId'], target['quantity'], target['unit']['short']) == (str(_port_id('demand')), 'energy', 'MWh/yr')
    assert target['dimensions'] == [
        {
            'dimensionId': str(_dimension('carrier').uuid),
            'categoryIds': [str(_category('carrier', carrier).uuid) for carrier in CARRIERS],
        }
    ]
    assert targets[str(heat.uuid)]['dimensions'] == []


def test_a_hook_is_added_listed_computed_and_removed(client: PathsTestClient, ic: InstanceConfig) -> None:
    heat, saving = _heat(ic), _saving(ic)

    _add(client, ic, saving, _hook(heat))

    assert _hooks(saving) == [ActionHookDef(node=heat.uuid, port=_port_id('heat'), from_port=_port_id('saving'))]
    assert _editor(client, ic)['hooks'] == [
        {
            'actionId': str(saving.uuid),
            'fromPortId': str(_port_id('saving')),
            'targetNodeId': str(heat.uuid),
            'targetPortId': str(_port_id('heat')),
            'mode': 'ABSOLUTE',
        }
    ]
    values = _by_year(_rebuild(ic).get_node('heat').get_output_pl())
    assert (values[2022], values[2030]) == (100, 90)

    assert 'already acts on this port' in _refused(client, ic, saving, _hook(heat))

    client.query_data(DELETE, variables={'instanceId': ic.identifier, 'nodeId': str(saving.uuid), 'input': _hook(heat)})
    assert _hooks(saving) == []
    assert _by_year(_rebuild(ic).get_node('heat').get_output_pl())[2030] == 100
    assert 'exactly one hook' in _refused(client, ic, saving, _hook(heat), DELETE)


def test_a_relative_hook_takes_a_share(client: PathsTestClient, ic: InstanceConfig) -> None:
    heat = _heat(ic)
    saving = _saving(ic, unit='dimensionless', quantity='fraction', value=-0.25)

    _add(client, ic, saving, _hook(heat, mode='RELATIVE'))

    assert [hook.mode for hook in _hooks(saving)] == [ActionHookMode.RELATIVE]
    assert _editor(client, ic)['hooks'][0]['mode'] == 'RELATIVE'
    values = _by_year(_rebuild(ic).get_node('heat').get_output_pl())
    assert (values[2022], values[2030]) == (100, 75)


def test_hooks_that_cannot_hold_are_refused(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand, heat, saving = _demand(ic), _heat(ic), _saving(ic)

    assert 'cannot act on itself' in _refused(client, ic, saving, _hook(saving))
    assert 'Only actions' in _refused(client, ic, heat, _hook(demand))
    assert 'dimensions differ' in _refused(client, ic, saving, _hook(demand))
    assert 'share without a unit' in _refused(client, ic, saving, _hook(heat, mode='RELATIVE'))
    assert _hooks(saving) == []


def test_a_share_has_no_unit_and_an_amount_has_the_targets() -> None:
    energy, share, money = (unit_registry.parse_units(unit) for unit in ('MWh/a', 'dimensionless', 'EUR/a'))
    target = ValueShape(dimensions=frozenset(), unit=energy)
    assert hook_source_problem(ValueShape(frozenset(), energy), target, ActionHookMode.ABSOLUTE) is None
    assert hook_source_problem(ValueShape(frozenset(), share), target, ActionHookMode.RELATIVE) is None
    assert 'cannot be converted' in (hook_source_problem(ValueShape(frozenset(), money), target, ActionHookMode.ABSOLUTE) or '')
    assert 'cannot be converted' in (hook_source_problem(ValueShape(frozenset(), share), target, ActionHookMode.ABSOLUTE) or '')
    assert 'share without a unit' in (hook_source_problem(ValueShape(frozenset(), energy), target, ActionHookMode.RELATIVE) or '')


def test_updating_an_action_config_keeps_its_hooks(client: PathsTestClient, ic: InstanceConfig) -> None:
    heat, saving = _heat(ic), _saving(ic)
    _add(client, ic, saving, _hook(heat))

    config = {'action': {'nodeClass': ADDITIVE_ACTION_CLASS, 'noEffectValue': 0}}
    client.query_data(UPDATE, variables={'instanceId': ic.identifier, 'nodeId': str(saving.uuid), 'input': {'config': config}})

    assert [hook.node for hook in _hooks(saving)] == [heat.uuid]


@pytest.fixture
def municipal(ic: InstanceConfig) -> tuple[InstanceConfig, InstanceConfig]:
    return ic, make_municipality(ic)


def test_a_municipality_acts_only_on_the_nodes_its_framework_opened(
    client: PathsTestClient, municipal: tuple[InstanceConfig, InstanceConfig]
) -> None:
    template, municipality = municipal
    client.set_instance(municipality)
    heat, demand = template.nodes.get(identifier='heat'), template.nodes.get(identifier='demand')
    saving = _saving(municipality)

    assert [target['nodeId'] for target in _editor(client, municipality)['actionHookTargets']] == [str(heat.uuid)]
    assert 'does not let actions act on it' in _refused(client, municipality, saving, _hook(demand))

    _add(client, municipality, saving, _hook(heat))
    values = _by_year(_rebuild(municipality).get_node('heat').get_output_pl())
    assert (values[2022], values[2030]) == (100, 90)
    # The framework's own node is untouched: the hook belongs to the municipality's action.
    assert _by_year(_rebuild(template).get_node('heat').get_output_pl())[2030] == 100


def test_a_template_release_that_closes_a_target_leaves_the_hook_as_a_composition_error(
    client: PathsTestClient, municipal: tuple[InstanceConfig, InstanceConfig]
) -> None:
    template, municipality = municipal
    client.set_instance(municipality)
    heat = template.nodes.get(identifier='heat')
    saving = _saving(municipality)
    _add(client, municipality, saving, _hook(heat))

    spec = NodeConfig.objects.with_spec().get(pk=heat.pk).spec
    assert spec is not None
    spec.action_hooks = False
    NodeConfig.objects.filter(pk=heat.pk).update(spec=spec)
    template.invalidate_cache()
    upgrade_template_instance(municipality, publish_template_instance(template))
    municipality.refresh_from_db()

    snapshot = build_instance_snapshot(municipality)
    assert snapshot.composition_errors == ['Action saving: the framework does not let actions act on heat']
    # The draft still loads and computes, without that effect, and the authoring row keeps the hook.
    assert _by_year(_rebuild(municipality).get_node('heat').get_output_pl())[2030] == 100
    assert [hook.node for hook in _hooks(saving)] == [heat.uuid]
    client.query_data(DELETE, variables={'instanceId': municipality.identifier, 'nodeId': str(saving.uuid), 'input': _hook(heat)})
    assert build_instance_snapshot(municipality).composition_errors == []
