"""Tests for the model editor's ``setFormula`` mutation."""

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from common.polars import DataFrameMeta, PathsDataFrame, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import FormulaConfig, NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.models import InstanceModelLogEntry, NodeConfig, NodeInputPortBinding, PreferredInstanceSource
from nodes.node import Node
from nodes.scenario import Scenario, ScenarioKind
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory, _port_id
from nodes.units import unit_registry
from params.param import BoolParameter, StringParameter

if TYPE_CHECKING:
    from paths.tests.graphql import PathsTestClient

    from nodes.models import InstanceConfig


pytestmark = pytest.mark.django_db

UNIT = 'kt/a'
SOURCE_NODE_CLASS = 'nodes.tests.test_formula_editor.FormulaEditorSourceNode'
FORMULA_NODE_CLASS = 'nodes.formula.FormulaNode'

SET_FORMULA = """
mutation SetFormula($instanceId: ID!, $nodeId: ID!, $formula: String!) {
    instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
            setFormula(formula: $formula) {
                ... on NodeInterface { identifier }
                ... on OperationInfo { messages { kind message } }
            }
        }
    }
}
"""


class FormulaEditorSourceNode(Node):
    """Leaf node with a fixed output, so a formula downstream has something to compute from. Test-only."""

    def compute(self) -> PathsDataFrame:
        df = pl.DataFrame({
            YEAR_COLUMN: [2020, 2021],
            VALUE_COLUMN: [1.0, 2.0],
            FORECAST_COLUMN: [False, True],
        })
        meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(UNIT)}, primary_keys=[YEAR_COLUMN])
        return to_ppdf(df, meta)


def _output_port() -> OutputPortDef:
    return OutputPortDef(id=_port_id('default'), unit=unit_registry.parse_units(UNIT), quantity='emissions')


@pytest.fixture
def db_instance_config() -> InstanceConfig:
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2020, max_historical=2020, target=2021))
    return InstanceConfigFactory.create(
        identifier=instance.id, instance=instance, config_source='database', owner='Test Owner', spec=spec
    )


@pytest.fixture
def gql_client(client, db_instance_config: InstanceConfig) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(db_instance_config)
    return tc


def _formula_node(ic: InstanceConfig, *, type_config: SimpleConfig | FormulaConfig, params: list[Any]) -> NodeConfig:
    """Create ``source -> target``, where ``target`` is a formula node reading ``source``."""
    source = NodeConfigFactory.create(
        instance=ic,
        identifier='source',
        spec=NodeSpec(type_config=SimpleConfig(node_class=SOURCE_NODE_CLASS), output_ports=[_output_port()]),
    )
    target = NodeConfigFactory.create(
        instance=ic,
        identifier='target',
        spec=NodeSpec(
            type_config=type_config,
            params=params,
            input_ports=[InputPortDef(id=_port_id('input'), unit=unit_registry.parse_units(UNIT), quantity='emissions')],
            output_ports=[_output_port()],
        ),
    )
    NodeInputPortBinding.objects.create(
        instance=ic,
        source_node=source,
        node=target,
        source_port_id=_port_id('default'),
        port_id=_port_id('input'),
        position=0,
    )
    return target


def _param_node(ic: InstanceConfig, formula: str = 'source') -> NodeConfig:
    return _formula_node(
        ic,
        type_config=SimpleConfig(node_class=FORMULA_NODE_CLASS),
        params=[
            StringParameter(local_id='formula', value=formula),
            BoolParameter(local_id='extend_last_historical_value', value=True),
        ],
    )


def _set_formula(gql_client: PathsTestClient, ic: InstanceConfig, nc: NodeConfig, formula: str) -> dict[str, Any]:
    data = gql_client.query_data(
        SET_FORMULA,
        variables={'instanceId': str(ic.pk), 'nodeId': str(nc.uuid), 'formula': formula},
    )
    return data['instanceEditor']['nodeEditor']['setFormula']


def _refused(gql_client: PathsTestClient, ic: InstanceConfig, nc: NodeConfig, formula: str) -> str:
    errors = gql_client.query_errors(
        SET_FORMULA,
        variables={'instanceId': str(ic.pk), 'nodeId': str(nc.uuid), 'formula': formula},
    )
    return errors[0].get('message', '')


def _stored_spec(nc: NodeConfig) -> NodeSpec:
    # The default manager defers ``spec``, so ask for it explicitly rather than refreshing.
    spec = NodeConfig.objects.with_spec().get(pk=nc.pk).spec
    assert spec is not None
    return spec


def _params(nc: NodeConfig) -> dict[str, Any]:
    return {p.local_id: p.value for p in _stored_spec(nc).params}


def test_set_formula_saves_a_computable_formula_and_keeps_other_params(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    nc = _param_node(db_instance_config)

    result = _set_formula(gql_client, db_instance_config, nc, '  source * 2  ')

    assert result['identifier'] == 'target'
    assert _params(nc) == {'formula': 'source * 2', 'extend_last_historical_value': True}
    assert InstanceModelLogEntry.objects.filter(action='node.formula.update', object_id=str(nc.pk)).exists()


@pytest.mark.parametrize(
    ('formula', 'expected'),
    [
        ('sourse * 2', 'Unknown name "sourse" in formula. Names available here: extend_last_historical_value, source'),
        ('source *', 'Syntax error at the end of the formula'),
        ('source source', 'Syntax error at column 8'),
        ('-source', 'Unary operators such as "-source" are not supported'),
        ('source // 2', 'Operator source // 2 is not supported'),
        ('frobnicate(source)', 'Unknown function "frobnicate"'),
        ('', 'The formula is empty'),
    ],
)
def test_set_formula_refuses_a_formula_that_does_not_compute(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
    formula: str,
    expected: str,
) -> None:
    nc = _param_node(db_instance_config)

    message = _refused(gql_client, db_instance_config, nc, formula)

    assert expected in message
    assert _params(nc)['formula'] == 'source'
    assert not InstanceModelLogEntry.objects.filter(action='node.formula.update').exists()


def test_set_formula_refuses_a_unit_mismatch_with_the_underlying_reason(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    nc = _param_node(db_instance_config)

    message = _refused(gql_client, db_instance_config, nc, 'source * source')

    # The wrapper message alone ("Error computing node") gives the author nothing to act on;
    # the reason from deeper in the exception chain must come through.
    assert message.startswith('Node target: ')
    assert "Unit 'kt²/a²' for column Value is not compatible with 'kt/a'" in message
    assert _params(nc)['formula'] == 'source'


def test_set_formula_refuses_a_node_that_is_not_a_formula_node(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    _param_node(db_instance_config)
    source = db_instance_config.nodes.get(identifier='source')

    message = _refused(gql_client, db_instance_config, source, 'source * 2')

    assert 'is not a formula node' in message


def test_set_formula_writes_both_copies_on_a_formula_config_node(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    # A node created by the editor with kind FORMULA holds its formula only in the type
    # config, which the runtime does not read; setFormula adds the parameter it does read.
    nc = _formula_node(db_instance_config, type_config=FormulaConfig(formula='source'), params=[])

    _set_formula(gql_client, db_instance_config, nc, 'source * 2')

    assert _params(nc) == {'formula': 'source * 2'}
    type_config = _stored_spec(nc).type_config
    assert isinstance(type_config, FormulaConfig)
    assert type_config.formula == 'source * 2'


def test_set_formula_takes_effect_when_a_scenario_holds_a_copy_of_the_formula(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    # Older loaders copied every node parameter, formulas included, into the default
    # scenario. Activating that scenario resets the parameter to the copy, so a formula
    # saved only on the node would be silently overridden at runtime.
    nc = _param_node(db_instance_config)
    spec = db_instance_config.spec
    assert spec is not None
    spec.scenarios = [
        Scenario(id='default', name='Default', kind=ScenarioKind.DEFAULT, param_values={'target.formula': 'source'}),
        Scenario(id='other', name='Other', param_values={'target.formula': 'source'}),
    ]
    type(db_instance_config).objects.filter(pk=db_instance_config.pk).update(spec=spec)

    _set_formula(gql_client, db_instance_config, nc, 'source * 2')

    # What the model computes with: the draft runtime's formula, after the default
    # scenario has been activated. (Root queries in tests see the factory's in-memory
    # instance, so build the draft from the database the way the editor does.)
    db_instance_config.refresh_from_db()
    draft = db_instance_config._initialize_instance(source=PreferredInstanceSource.DRAFT)
    assert draft.context.get_node('target').get_parameter_value('formula') == 'source * 2'
    stored = db_instance_config.spec
    assert stored is not None
    assert all('target.formula' not in scenario.param_values for scenario in stored.scenarios)
    assert InstanceModelLogEntry.objects.filter(action='instance.scenario.parameter.remove').exists()
