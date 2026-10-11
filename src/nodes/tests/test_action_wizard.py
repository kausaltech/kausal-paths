"""
Creating an action from the output ports it acts on (`createActionFromPorts`).

See docs/plans/action-from-output-port.md.
"""

from datetime import date
from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.models import ContentType

import polars as pl
import pytest

from kausal_common.datasets.models import (
    DataPoint,
    DataPointDimensionCategory,
    Dataset,
    DatasetMetric,
    DatasetSchema,
    DatasetSchemaDimension,
    DatasetSchemaScope,
    Dimension,
    DimensionCategory,
)

from datasets.materialization import refresh_dataset_materialization
from nodes.constants import YEAR_COLUMN
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import ActionConfig, ActionHookDef, InputDatasetDef, NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.defs.transform_def import ExtendOp, InterpolateOp
from nodes.models import NodeConfig, NodeInputPortBinding
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory, _port_id, register_dimensions
from nodes.units import unit_registry

if TYPE_CHECKING:
    from paths.tests.graphql import PathsTestClient

    from nodes.context import Context
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db

gql = str

CREATE = gql("""
mutation Create($instanceId: ID!, $input: CreateActionFromPortsInput!) {
    instanceEditor(instanceId: $instanceId) {
        createActionFromPorts(input: $input) {
            __typename
            ... on CreateActionFromPortsResult {
                action { ... on NodeInterface { id identifier } }
                datasets { id }
            }
            ... on ConstraintViolations { conflicts { code message } }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
""")

CANDIDATES = gql("""
query Candidates($instanceId: ID!, $target: OutputPortRefInput!) {
    modelInstance(instanceId: $instanceId) {
        editor {
            effectSourceCandidates(target: $target) {
                datasets { datasetId metricId }
                nodes { nodeId portId }
            }
            hooks { actionId fromPortId targetNodeId targetPortId }
        }
    }
}
""")

YEARS = YearsSpec(reference=2018, min_historical=2018, max_historical=2022, target=2030, model_end=2035)
CARRIERS = ['oil', 'electricity', 'gas']


@pytest.fixture
def ic() -> InstanceConfig:
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(years=YEARS)
    spec.features.use_datasets_from_db = True
    ic = InstanceConfigFactory.create(
        identifier=instance.id, instance=instance, config_source='database', owner='Test Owner', spec=spec
    )
    register_dimensions(ic, ['carrier'], {'carrier': CARRIERS})
    return ic


@pytest.fixture
def client(client, ic: InstanceConfig) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(ic)
    return tc


def _carrier() -> Dimension:
    return Dimension.objects.get(scopes__identifier='carrier')


def _category(identifier: str) -> DimensionCategory:
    return DimensionCategory.objects.get(dimension=_carrier(), identifier=identifier)


def _dataset(ic: InstanceConfig, identifier: str, unit: str, values: dict[int, dict[str, float]], dims: bool = True) -> Dataset:
    """Create an instance dataset with one metric, by carrier unless `dims` is false."""
    ct = ContentType.objects.get_for_model(ic)
    schema = DatasetSchema.objects.create(name=identifier)
    DatasetSchemaScope.objects.create(schema=schema, scope_content_type=ct, scope_id=ic.pk)
    if dims:
        DatasetSchemaDimension.objects.create(schema=schema, dimension=_carrier(), order=0)
    metric = DatasetMetric.objects.create(schema=schema, name='value', label='Value', unit=unit)
    dataset = Dataset.objects.create(schema=schema, identifier=identifier, scope_content_type=ct, scope_id=ic.pk)
    for year, by_carrier in values.items():
        for carrier, value in by_carrier.items():
            point = DataPoint.objects.create(dataset=dataset, metric=metric, date=date(year, 1, 1), value=value)
            if dims:
                DataPointDimensionCategory.objects.create(data_point=point, dimension_category=_category(carrier))
    return dataset


def _energy_port(name: str, unit: str = 'MWh/a', quantity: str = 'energy', dims: list[str] | None = None) -> OutputPortDef:
    return OutputPortDef(
        id=_port_id(name),
        identifier=name,
        unit=unit_registry.parse_units(unit),
        quantity=quantity,
        dimensions=['carrier'] if dims is None else dims,
    )


def _demand(ic: InstanceConfig) -> NodeConfig:
    """Create a node whose output is a dataset of energy use by carrier, 100 MWh/a each, 2018-2022."""
    dataset = _dataset(ic, 'test/demand', 'MWh/a', {year: dict.fromkeys(CARRIERS, 100.0) for year in range(2018, 2023)})
    spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
        input_ports=[
            InputPortDef(id=_port_id('in'), identifier='in', unit=unit_registry.parse_units('MWh/a'), quantity='energy')
        ],
        output_ports=[_energy_port('demand')],
        input_dimensions=['carrier'],
        output_dimensions=['carrier'],
    )
    nc = NodeConfigFactory.create(instance=ic, identifier='demand', name='Demand', spec=spec)
    assert dataset.schema is not None
    NodeInputPortBinding.objects.create(
        instance=ic,
        node=nc,
        port_id=_port_id('in'),
        dataset=dataset,
        metric=dataset.schema.metrics.get(),
        transformations=InputDatasetDef(id='test/demand', column='value', forecast_from=2023).to_transformations(),
    )
    return nc


def _costs(ic: InstanceConfig, dims: list[str] | None = None) -> NodeConfig:
    spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.SimpleNode'),
        output_ports=[_energy_port('costs', 'MEUR/a', 'currency', dims=dims)],
        output_dimensions=['carrier'] if dims is None else dims,
    )
    return NodeConfigFactory.create(instance=ic, identifier='costs', name='Heating costs', spec=spec)


def _create(client: PathsTestClient, ic: InstanceConfig, **input: Any) -> dict[str, Any]:
    data = client.query_data(CREATE, variables={'instanceId': ic.identifier, 'input': {'name': 'Heat pumps', **input}})
    return data['instanceEditor']['createActionFromPorts']


def _create_errors(client: PathsTestClient, ic: InstanceConfig, **input: Any) -> str:
    variables = {'instanceId': ic.identifier, 'input': {'name': 'Heat pumps', **input}}
    return ' '.join(error.get('message', '') for error in client.query_errors(CREATE, variables=variables))


def _action(ic: InstanceConfig) -> NodeConfig:
    return NodeConfig.objects.with_spec().get(instance=ic, identifier='heat_pumps')


def _cells(dataset: Dataset) -> dict[tuple[int, str, str | None], float | None]:
    return {
        (point.date.year, point.metric.label, next((c.identifier for c in point.dimension_categories.all()), None)): (
            float(point.value) if point.value is not None else None
        )
        for point in dataset.data_points.select_related('metric').prefetch_related('dimension_categories')
    }


def _rebuild(ic: InstanceConfig) -> Context:
    from nodes.models import test_instance_registry

    ic.refresh_from_db()
    cached = test_instance_registry.pop(ic.identifier, None)
    try:
        return ic._create_from_config().context
    finally:
        if cached is not None:
            test_instance_registry[ic.identifier] = cached


def _by_year(df, carrier: str) -> dict[int, float]:
    df = df.filter(pl.col('carrier') == carrier)
    return dict(zip(df[YEAR_COLUMN].to_list(), df[df.metric_cols[0]].to_list(), strict=True))


def test_a_new_dataset_effect_creates_the_action_its_hook_dataset_and_binding(
    client: PathsTestClient, ic: InstanceConfig
) -> None:
    demand = _demand(ic)
    keep = [str(_category('oil').uuid), str(_category('electricity').uuid)]

    result = _create(
        client,
        ic,
        effects=[
            {
                'target': {'nodeId': str(demand.uuid)},
                'source': {'newDataset': {'categories': [{'dimensionId': str(_carrier().uuid), 'categoryIds': keep}]}},
            }
        ],
    )

    assert result['__typename'] == 'CreateActionFromPortsResult'
    action = _action(ic)
    assert action.spec is not None
    assert isinstance(action.spec.type_config, ActionConfig)
    (output,) = action.spec.output_ports
    assert output.dimensions == ['carrier']
    assert output.unit == unit_registry.parse_units('MWh/a')
    assert action.spec.type_config.hooks == [ActionHookDef(node='demand', port=_port_id('demand'), from_port=output.id)]

    (dataset,) = Dataset.objects.qs.for_node(action)
    assert dataset.identifier is None
    assert [d['id'] for d in result['datasets']] == [str(dataset.uuid)]
    # Zeros in the last historical year, for the kept categories only.
    assert _cells(dataset) == {(2022, 'Demand', 'oil'): 0.0, (2022, 'Demand', 'electricity'): 0.0}

    (binding,) = NodeInputPortBinding.objects.filter(node=action)
    assert binding.dataset == dataset
    kinds = [type(op) for op in binding.transformations]
    assert InterpolateOp in kinds
    assert ExtendOp in kinds


def test_the_effect_interpolates_holds_after_the_last_year_and_leaves_history_alone(
    client: PathsTestClient, ic: InstanceConfig
) -> None:
    demand = _demand(ic)
    keep = [str(_category('oil').uuid), str(_category('electricity').uuid)]
    _create(
        client,
        ic,
        effects=[
            {
                'target': {'nodeId': str(demand.uuid)},
                'source': {'newDataset': {'categories': [{'dimensionId': str(_carrier().uuid), 'categoryIds': keep}]}},
            }
        ],
    )
    (dataset,) = Dataset.objects.qs.for_node(_action(ic))
    assert dataset.schema is not None
    metric = dataset.schema.metrics.get()
    for carrier, value in (('oil', -40.0), ('electricity', 12.0)):
        point = DataPoint.objects.create(dataset=dataset, metric=metric, date=date(2030, 1, 1), value=value)
        DataPointDimensionCategory.objects.create(data_point=point, dimension_category=_category(carrier))
    # The data grid does this after every edit; the runtime reads the materialized copy.
    refresh_dataset_materialization(dataset)

    ctx = _rebuild(ic)
    with ctx.run():
        action = ctx.get_action('heat_pumps')
        action.enabled_param.set(True)
        output = ctx.get_node('demand').get_output_pl()
        oil, gas = _by_year(output, 'oil'), _by_year(output, 'gas')
        assert [oil[year] for year in (2018, 2022)] == [100, 100]
        assert oil[2026] == pytest.approx(80)
        assert oil[2030] == pytest.approx(60)
        assert oil[2035] == pytest.approx(60)
        assert _by_year(output, 'electricity')[2035] == pytest.approx(112)
        # A category the action left out is not touched.
        assert gas[2035] == pytest.approx(100)
        action.enabled_param.set(False)
        assert _by_year(ctx.get_node('demand').get_output_pl(), 'oil')[2035] == pytest.approx(100)


def test_effects_from_an_existing_dataset_and_from_a_node(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    costs = _costs(ic)
    savings = _dataset(ic, 'test/savings', 'GWh/a', {2030: {'oil': -0.04}})
    source_spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.SimpleNode'),
        output_ports=[_energy_port('saving', 'kEUR/a', 'currency')],
        output_dimensions=['carrier'],
    )
    cost_saving = NodeConfigFactory.create(instance=ic, identifier='cost_saving', spec=source_spec)

    result = _create(
        client,
        ic,
        effects=[
            {'target': {'nodeId': str(demand.uuid)}, 'source': {'dataset': {'datasetId': str(savings.uuid)}}},
            {'target': {'nodeId': str(costs.uuid)}, 'source': {'node': {'nodeId': str(cost_saving.uuid)}}},
        ],
    )

    assert result['datasets'] == []
    action = _action(ic)
    assert not Dataset.objects.qs.for_node(action).exists()
    bindings = NodeInputPortBinding.objects.filter(node=action)
    assert {binding.dataset_id for binding in bindings if binding.dataset_id} == {savings.pk}
    assert {binding.source_node_id for binding in bindings if binding.source_node_id} == {cost_saving.pk}
    assert action.spec is not None
    assert isinstance(action.spec.type_config, ActionConfig)
    assert [hook.node for hook in action.spec.type_config.hooks] == ['demand', 'costs']
    # An existing dataset gets no zeros written into it.
    assert savings.data_points.count() == 1


def test_effects_with_different_dimensions_are_refused(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    costs = _costs(ic, dims=[])

    errors = _create_errors(
        client, ic, effects=[{'target': {'nodeId': str(demand.uuid)}}, {'target': {'nodeId': str(costs.uuid)}}]
    )

    assert 'same dimensions' in errors


def test_a_dataset_without_the_target_dimensions_is_refused(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    total = _dataset(ic, 'test/total', 'MWh/a', {2030: {'oil': -1}}, dims=False)

    errors = _create_errors(
        client, ic, effects=[{'target': {'nodeId': str(demand.uuid)}, 'source': {'dataset': {'datasetId': str(total.uuid)}}}]
    )

    assert 'dimensions differ' in errors
    assert not NodeConfig.objects.filter(instance=ic, identifier='heat_pumps').exists()


def test_another_nodes_own_dataset_is_refused(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    _create(client, ic, effects=[{'target': {'nodeId': str(demand.uuid)}}])
    (owned,) = Dataset.objects.qs.for_node(_action(ic))

    errors = _create_errors(
        client,
        ic,
        name='Other',
        effects=[{'target': {'nodeId': str(demand.uuid)}, 'source': {'dataset': {'datasetId': str(owned.uuid)}}}],
    )

    assert 'instance datasets' in errors


def test_a_node_source_downstream_of_the_target_is_refused(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    downstream_spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
        input_ports=[InputPortDef(id=_port_id('in'), identifier='in', unit=unit_registry.parse_units('MWh/a'))],
        output_ports=[_energy_port('total')],
        input_dimensions=['carrier'],
        output_dimensions=['carrier'],
    )
    downstream = NodeConfigFactory.create(instance=ic, identifier='total', spec=downstream_spec)
    NodeInputPortBinding.objects.create(
        instance=ic, node=downstream, port_id=_port_id('in'), source_node=demand, source_port_id=_port_id('demand')
    )

    errors = _create_errors(
        client, ic, effects=[{'target': {'nodeId': str(demand.uuid)}, 'source': {'node': {'nodeId': str(downstream.uuid)}}}]
    )

    assert 'loop' in errors
    assert not NodeConfig.objects.filter(instance=ic, identifier='heat_pumps').exists()


def test_candidates_are_the_sources_that_fit_and_hooks_are_listed(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    fits = _dataset(ic, 'test/fits', 'GWh/a', {2030: {'oil': -1}})
    _dataset(ic, 'test/no_carrier', 'MWh/a', {2030: {'oil': -1}}, dims=False)
    _dataset(ic, 'test/wrong_unit', 'MEUR/a', {2030: {'oil': -1}})
    _costs(ic, dims=[])

    variables = {'instanceId': ic.identifier, 'target': {'nodeId': str(demand.uuid)}}
    editor = client.query_data(CANDIDATES, variables=variables)['modelInstance']['editor']
    candidates = editor['effectSourceCandidates']

    datasets = {entry['datasetId'] for entry in candidates['datasets']}
    assert str(fits.uuid) in datasets
    assert len(datasets) == 2  # `fits`, and the target's own input data
    assert candidates['nodes'] == []  # the target itself is downstream of itself; costs has other units
    assert editor['hooks'] == []

    _create(client, ic, effects=[{'target': {'nodeId': str(demand.uuid)}}])
    editor = client.query_data(CANDIDATES, variables=variables)['modelInstance']['editor']
    action = _action(ic)
    assert action.spec is not None
    assert editor['hooks'] == [
        {
            'actionId': str(action.uuid),
            'fromPortId': str(action.spec.output_ports[0].id),
            'targetNodeId': str(demand.uuid),
            'targetPortId': str(_port_id('demand')),
        }
    ]


CREATE_PORT_DATASET = gql("""
mutation CreatePortDataset($instanceId: ID!, $nodeId: ID!, $portIds: [UUID!]!) {
    instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
            createPortDataset(portIds: $portIds) {
                __typename
                ... on Dataset { id }
                ... on OperationInfo { messages { kind message } }
            }
        }
    }
}
""")


def test_create_port_dataset_gives_the_ports_their_own_bound_metrics(client: PathsTestClient, ic: InstanceConfig) -> None:
    spec = NodeSpec(
        type_config=ActionConfig(node_class='nodes.actions.simple.AdditiveAction'),
        input_ports=[
            InputPortDef(
                id=_port_id('in'),
                identifier='in',
                unit=unit_registry.parse_units('MWh/a'),
                quantity='energy',
                paired_output_port_id=_port_id('out'),
            )
        ],
        # A paired input port is labelled by its output port.
        output_ports=[_energy_port('out').model_copy(update={'label': 'Saved energy'})],
        input_dimensions=['carrier'],
        output_dimensions=['carrier'],
    )
    action = NodeConfigFactory.create(instance=ic, identifier='saving', spec=spec)

    variables = {'instanceId': ic.identifier, 'nodeId': str(action.uuid), 'portIds': [str(_port_id('in'))]}
    result = client.query_data(CREATE_PORT_DATASET, variables=variables)['instanceEditor']['nodeEditor']['createPortDataset']

    (dataset,) = Dataset.objects.qs.for_node(action)
    assert result == {'__typename': 'Dataset', 'id': str(dataset.uuid)}
    assert dataset.schema is not None
    assert [sd.dimension.uuid for sd in dataset.schema.dimensions.all()] == [_carrier().uuid]
    (metric,) = dataset.schema.metrics.all()
    assert (metric.label, metric.unit) == ('Saved energy', 'MWh/a')
    (binding,) = NodeInputPortBinding.objects.filter(node=action)
    assert (binding.port_id, binding.metric) == (_port_id('in'), metric)


EDITOR_DATASET = gql("""
query EditorDataset($id: ID!) {
    instance { editor { dataset(id: $id) { id identifier } } }
}
""")


def test_the_editor_opens_the_actions_own_dataset_by_id(client: PathsTestClient, ic: InstanceConfig) -> None:
    """Owned datasets stay out of the instance's lists, but the dataset editor must find them."""
    demand = _demand(ic)
    _create(client, ic, effects=[{'target': {'nodeId': str(demand.uuid)}}])
    (dataset,) = Dataset.objects.qs.for_node(_action(ic))

    data = client.query_data(EDITOR_DATASET, variables={'id': str(dataset.uuid)})

    assert data['instance']['editor']['dataset'] == {'id': str(dataset.uuid), 'identifier': None}


DATASET_REQUIREMENT_GROUPS = gql("""
query DatasetRequirementGroups($id: ID!) {
    instance { editor { dataset(id: $id) { validationViolations { requirementGroup } } } }
}
""")


def test_dataset_violations_still_answer_the_retired_requirement_group(client: PathsTestClient, ic: InstanceConfig) -> None:
    """The UI on main still selects `requirementGroup`; without the field every dataset page fails."""
    dataset = _dataset(ic, 'test/any', 'MWh/a', {2030: {'oil': 1}})

    data = client.query_data(DATASET_REQUIREMENT_GROUPS, variables={'id': str(dataset.uuid)})

    assert data['instance']['editor']['dataset']['validationViolations'] == []


def test_a_single_output_port_finds_its_metric_without_a_column_id(ic: InstanceConfig) -> None:
    """The output preview of a node whose only port names no column, as in a node synced from YAML."""
    from nodes.metric import DimensionalMetric

    demand = _demand(ic)
    assert demand.spec is not None
    port = demand.spec.output_ports[0]
    assert port.column_id is None

    ctx = _rebuild(ic)
    with ctx.run():
        metric = DimensionalMetric.from_output_port(ctx.get_node('demand'), port)

    assert metric.values
