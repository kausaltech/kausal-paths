"""
Creating an action from the nodes it acts on (`createActionOnNodes`).

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
from nodes.defs.node_defs import FormulaConfig, InputDatasetDef, NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.defs.transform_def import BackfillOp, ExtendOp, InterpolateOp
from nodes.models import InstanceConfig, NodeConfig, NodeInputPortBinding
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory, _port_id, register_dimensions
from nodes.units import unit_registry

if TYPE_CHECKING:
    from paths.tests.graphql import PathsTestClient

    from nodes.context import Context

pytestmark = pytest.mark.django_db

gql = str

CREATE = gql("""
mutation Create($instanceId: ID!, $input: CreateActionOnNodesInput!) {
    instanceEditor(instanceId: $instanceId) {
        createActionOnNodes(input: $input) {
            __typename
            ... on CreateActionOnNodesResult {
                action { ... on NodeInterface { id identifier } }
                datasets { id }
            }
            ... on ConstraintViolations { conflicts { code message } }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
""")

EDITOR = gql("""
query Editor($instanceId: ID!, $target: UUID!) {
    modelInstance(instanceId: $instanceId) {
        editor {
            actionTargets { nodeId available reason }
            effectSourceCandidates(targetNodeId: $target) {
                datasets { datasetId metricId }
                nodes { nodeId portId }
            }
        }
    }
}
""")

YEARS = YearsSpec(reference=2018, min_historical=2018, max_historical=2022, target=2030, model_end=2035)
CARRIERS = ['oil', 'electricity', 'gas']
SECTORS = ['homes', 'offices']


@pytest.fixture
def ic() -> InstanceConfig:
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(years=YEARS)
    spec.features.use_datasets_from_db = True
    ic = InstanceConfigFactory.create(
        identifier=instance.id, instance=instance, config_source='database', owner='Test Owner', spec=spec
    )
    register_dimensions(ic, ['carrier', 'sector'], {'carrier': CARRIERS, 'sector': SECTORS})
    return ic


@pytest.fixture
def client(client, ic: InstanceConfig) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(ic)
    return tc


def _dimension(identifier: str) -> Dimension:
    return Dimension.objects.get(scopes__identifier=identifier)


def _category(dimension: str, identifier: str) -> DimensionCategory:
    return DimensionCategory.objects.get(dimension=_dimension(dimension), identifier=identifier)


def _dataset(
    ic: InstanceConfig, identifier: str, unit: str, values: dict[int, dict[tuple[str, ...], float]], dims: tuple[str, ...]
) -> Dataset:
    """Create an instance dataset with one metric over `dims`; `values` maps year -> category tuple -> value."""
    ct = ContentType.objects.get_for_model(ic)
    schema = DatasetSchema.objects.create(name=identifier)
    DatasetSchemaScope.objects.create(schema=schema, scope_content_type=ct, scope_id=ic.pk)
    for order, dim in enumerate(dims):
        DatasetSchemaDimension.objects.create(schema=schema, dimension=_dimension(dim), order=order)
    metric = DatasetMetric.objects.create(schema=schema, name='value', label='Value', unit=unit)
    dataset = Dataset.objects.create(schema=schema, identifier=identifier, scope_content_type=ct, scope_id=ic.pk)
    for year, cells in values.items():
        for categories, value in cells.items():
            point = DataPoint.objects.create(dataset=dataset, metric=metric, date=date(year, 1, 1), value=value)
            for dim, category in zip(dims, categories, strict=True):
                DataPointDimensionCategory.objects.create(data_point=point, dimension_category=_category(dim, category))
    refresh_dataset_materialization(dataset)
    return dataset


def _port(name: str, unit: str, quantity: str, dims: list[str]) -> OutputPortDef:
    return OutputPortDef(
        id=_port_id(name), identifier=name, unit=unit_registry.parse_units(unit), quantity=quantity, dimensions=dims
    )


def _bind(ic: InstanceConfig, nc: NodeConfig, port: str, dataset: Dataset) -> None:
    assert dataset.schema is not None
    NodeInputPortBinding.objects.create(
        instance=ic,
        node=nc,
        port_id=_port_id(port),
        dataset=dataset,
        metric=dataset.schema.metrics.get(),
        transformations=InputDatasetDef(id=dataset.identifier or '', column='value', forecast_from=2023).to_transformations(),
    )


def _history(categories: list[tuple[str, ...]], value: float) -> dict[int, dict[tuple[str, ...], float]]:
    return {year: dict.fromkeys(categories, value) for year in range(2018, 2036)}


def _demand(ic: InstanceConfig) -> NodeConfig:
    """Create an additive node: energy use by carrier, 100 MWh/a each."""
    dataset = _dataset(ic, 'test/demand', 'MWh/a', _history([(c,) for c in CARRIERS], 100.0), ('carrier',))
    spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
        input_ports=[
            InputPortDef(id=_port_id('in'), identifier='in', unit=unit_registry.parse_units('MWh/a'), quantity='energy')
        ],
        output_ports=[_port('demand', 'MWh/a', 'energy', ['carrier'])],
        input_dimensions=['carrier'],
    )
    nc = NodeConfigFactory.create(instance=ic, identifier='demand', name='Demand', spec=spec)
    _bind(ic, nc, 'in', dataset)
    return nc


def _costs(ic: InstanceConfig) -> NodeConfig:
    """Create an additive node without the carrier dimension, by sector: 10 MEUR/a each."""
    dataset = _dataset(ic, 'test/costs', 'MEUR/a', _history([(s,) for s in SECTORS], 10.0), ('sector',))
    spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
        input_ports=[
            InputPortDef(id=_port_id('cost_in'), identifier='in', unit=unit_registry.parse_units('MEUR/a'), quantity='currency')
        ],
        output_ports=[_port('costs', 'MEUR/a', 'currency', ['sector'])],
        input_dimensions=['sector'],
    )
    nc = NodeConfigFactory.create(instance=ic, identifier='costs', name='Heating costs', spec=spec)
    _bind(ic, nc, 'cost_in', dataset)
    return nc


def _create(client: PathsTestClient, ic: InstanceConfig, **input: Any) -> dict[str, Any]:
    data = client.query_data(CREATE, variables={'instanceId': ic.identifier, 'input': {'name': 'Heat pumps', **input}})
    return data['instanceEditor']['createActionOnNodes']


def _create_errors(client: PathsTestClient, ic: InstanceConfig, **input: Any) -> str:
    variables = {'instanceId': ic.identifier, 'input': {'name': 'Heat pumps', **input}}
    return ' '.join(error.get('message', '') for error in client.query_errors(CREATE, variables=variables))


def _action(ic: InstanceConfig) -> NodeConfig:
    return NodeConfig.objects.with_spec().get(instance=ic, identifier='heat_pumps')


def _cells(dataset: Dataset) -> dict[tuple[int, str, tuple[str, ...]], float | None]:
    return {
        (point.date.year, point.metric.label, tuple(sorted(str(c.identifier) for c in point.dimension_categories.all()))): (
            float(point.value) if point.value is not None else None
        )
        for point in dataset.data_points.select_related('metric').prefetch_related('dimension_categories')
    }


def _write(dataset: Dataset, label: str, year: int, values: dict[tuple[tuple[str, str], ...], float]) -> None:
    assert dataset.schema is not None
    metric = dataset.schema.metrics.get(label=label)
    for categories, value in values.items():
        wanted = {_category(dim, category).pk for dim, category in categories}
        # Fill the cell if the dataset already has it (the wizard leaves the target year empty), as the grid does.
        existing = [
            point
            for point in DataPoint.objects.filter(dataset=dataset, metric=metric, date=date(year, 1, 1))
            if {c.pk for c in point.dimension_categories.all()} == wanted
        ]
        if existing:
            DataPoint.objects.filter(pk=existing[0].pk).update(value=value)
            continue
        point = DataPoint.objects.create(dataset=dataset, metric=metric, date=date(year, 1, 1), value=value)
        for dim, category in categories:
            DataPointDimensionCategory.objects.create(data_point=point, dimension_category=_category(dim, category))
    # The data grid does this after every edit; the runtime reads the materialized copy.
    refresh_dataset_materialization(dataset)


def _rebuild(ic: InstanceConfig) -> Context:
    from nodes.models import test_instance_registry

    ic.refresh_from_db()
    cached = test_instance_registry.pop(ic.identifier, None)
    try:
        return ic._create_from_config().context
    finally:
        if cached is not None:
            test_instance_registry[ic.identifier] = cached


def _by_year(df: Any, dim: str | None = None, category: str | None = None) -> dict[int, float]:
    if dim is not None:
        df = df.filter(pl.col(dim) == category)
    return dict(zip(df[YEAR_COLUMN].to_list(), df[df.metric_cols[0]].to_list(), strict=True))


def _carrier_narrowing(*carriers: str) -> dict[str, Any]:
    keep = [str(_category('carrier', carrier).uuid) for carrier in carriers]
    return {'newDataset': {'categories': [{'dimensionId': str(_dimension('carrier').uuid), 'categoryIds': keep}]}}


def test_a_new_dataset_effect_creates_the_action_its_dataset_and_an_edge_into_the_additive_input(
    client: PathsTestClient, ic: InstanceConfig
) -> None:
    demand = _demand(ic)

    result = _create(client, ic, effects=[{'targetNodeId': str(demand.uuid), 'source': _carrier_narrowing('oil', 'electricity')}])

    assert result['__typename'] == 'CreateActionOnNodesResult'
    action = _action(ic)
    assert action.spec is not None
    (output,) = action.spec.output_ports
    assert output.dimensions == ['carrier']
    assert output.unit == unit_registry.parse_units('MWh/a')

    # The dataset belongs to the action, not to the instance.
    (dataset,) = Dataset.objects.qs.for_node(action)
    assert dataset.identifier is None
    assert [d['id'] for d in result['datasets']] == [str(dataset.uuid)]
    # Zeros in the last historical year and empty cells in the target year, for the kept categories only.
    assert _cells(dataset) == {
        (2022, 'Demand', ('oil',)): 0.0,
        (2022, 'Demand', ('electricity',)): 0.0,
        (2030, 'Demand', ('oil',)): None,
        (2030, 'Demand', ('electricity',)): None,
    }

    (binding,) = NodeInputPortBinding.objects.filter(node=action)
    assert binding.dataset == dataset
    kinds = {type(op) for op in binding.transformations}
    assert {InterpolateOp, BackfillOp, ExtendOp} <= kinds

    # The action feeds the target's additive input, added to the node's own input.
    demand = NodeConfig.objects.with_spec().get(pk=demand.pk)
    assert demand.spec is not None
    (edge,) = NodeInputPortBinding.objects.filter(node=demand, source_node=action)
    port = next(port for port in demand.spec.input_ports if port.id == edge.port_id)
    assert port.role == 'additive'
    assert port.multi


def test_the_effect_is_added_last_interpolates_and_holds_after_the_last_year(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    _create(client, ic, effects=[{'targetNodeId': str(demand.uuid), 'source': _carrier_narrowing('oil', 'electricity')}])
    (dataset,) = Dataset.objects.qs.for_node(_action(ic))
    _write(dataset, 'Demand', 2030, {(('carrier', 'oil'),): -40.0, (('carrier', 'electricity'),): 12.0})

    ctx = _rebuild(ic)
    with ctx.run():
        action = ctx.get_action('heat_pumps')
        action.enabled_param.set(True)
        output = ctx.get_node('demand').get_output_pl()
        oil = _by_year(output, 'carrier', 'oil')
        assert [oil[year] for year in (2018, 2022)] == [100, 100]
        assert oil[2026] == pytest.approx(80)
        assert oil[2030] == pytest.approx(60)
        assert oil[2035] == pytest.approx(60)
        assert _by_year(output, 'carrier', 'electricity')[2035] == pytest.approx(112)
        # A category the action left out is not touched.
        assert _by_year(output, 'carrier', 'gas')[2035] == pytest.approx(100)
        action.enabled_param.set(False)
        assert _by_year(ctx.get_node('demand').get_output_pl(), 'carrier', 'oil')[2035] == pytest.approx(100)


def test_targets_with_different_dimensions_share_one_dataset_with_empty_cells(
    client: PathsTestClient, ic: InstanceConfig
) -> None:
    demand = _demand(ic)
    costs = _costs(ic)

    result = _create(
        client,
        ic,
        effects=[
            {'targetNodeId': str(demand.uuid), 'source': _carrier_narrowing('oil')},
            {'targetNodeId': str(costs.uuid)},
        ],
    )

    assert result['__typename'] == 'CreateActionOnNodesResult'
    action = _action(ic)
    assert action.spec is not None
    assert [port.dimensions for port in action.spec.output_ports] == [['carrier'], ['sector']]
    (dataset,) = Dataset.objects.qs.for_node(action)
    assert dataset.schema is not None
    assert [sd.dimension.uuid for sd in dataset.schema.dimensions.order_by('order')] == [
        _dimension('carrier').uuid,
        _dimension('sector').uuid,
    ]
    # Each metric has cells only in its own dimensions; the other dimension stays empty.
    assert _cells(dataset) == {
        (2022, 'Demand', ('oil',)): 0.0,
        (2022, 'Heating costs', ('homes',)): 0.0,
        (2022, 'Heating costs', ('offices',)): 0.0,
        (2030, 'Demand', ('oil',)): None,
        (2030, 'Heating costs', ('homes',)): None,
        (2030, 'Heating costs', ('offices',)): None,
    }
    assert NodeInputPortBinding.objects.filter(node=action, dataset=dataset).count() == 2

    _write(dataset, 'Demand', 2030, {(('carrier', 'oil'),): -40.0})
    _write(dataset, 'Heating costs', 2030, {(('sector', 'homes'),): -2.0})

    ctx = _rebuild(ic)
    with ctx.run():
        ctx.get_action('heat_pumps').enabled_param.set(True)
        demand_out = ctx.get_node('demand').get_output_pl()
        costs_out = ctx.get_node('costs').get_output_pl()
        assert set(demand_out.dim_ids) == {'carrier'}
        assert set(costs_out.dim_ids) == {'sector'}
        assert _by_year(demand_out, 'carrier', 'oil')[2030] == pytest.approx(60)
        assert _by_year(demand_out, 'carrier', 'oil')[2022] == pytest.approx(100)
        assert _by_year(costs_out, 'sector', 'homes')[2035] == pytest.approx(8)
        assert _by_year(costs_out, 'sector', 'offices')[2035] == pytest.approx(10)


def test_a_multiplicative_node_gets_the_effect_on_its_additive_port(client: PathsTestClient, ic: InstanceConfig) -> None:
    activity = _dataset(ic, 'test/activity', 'kWh/a', _history([()], 1000.0), ())
    factor = _dataset(ic, 'test/factor', 'kg/kWh', _history([()], 0.5), ())
    spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.MultiplicativeNode'),
        input_ports=[
            InputPortDef(id=_port_id('f1'), identifier='factor', role='factors', unit=unit_registry.parse_units('kWh/a')),
            InputPortDef(id=_port_id('f2'), identifier='factor2', role='factors', unit=unit_registry.parse_units('kg/kWh')),
        ],
        output_ports=[_port('emissions', 'kg/a', 'emissions', [])],
    )
    emissions = NodeConfigFactory.create(instance=ic, identifier='emissions', name='Emissions', spec=spec)
    _bind(ic, emissions, 'f1', activity)
    _bind(ic, emissions, 'f2', factor)

    _create(client, ic, effects=[{'targetNodeId': str(emissions.uuid)}])
    emissions = NodeConfig.objects.with_spec().get(pk=emissions.pk)
    assert emissions.spec is not None
    (edge,) = NodeInputPortBinding.objects.filter(node=emissions, source_node=_action(ic))
    assert next(port for port in emissions.spec.input_ports if port.id == edge.port_id).role == 'additive'

    (dataset,) = Dataset.objects.qs.for_node(_action(ic))
    _write(dataset, 'Emissions', 2030, {(): -100.0})
    ctx = _rebuild(ic)
    with ctx.run():
        ctx.get_action('heat_pumps').enabled_param.set(True)
        output = _by_year(ctx.get_node('emissions').get_output_pl())
        assert output[2020] == pytest.approx(500)
        assert output[2030] == pytest.approx(400)


def test_a_formula_node_gets_the_effect_on_a_new_port_its_formula_does_not_name(
    client: PathsTestClient, ic: InstanceConfig
) -> None:
    demand = _demand(ic)
    spec = NodeSpec(
        type_config=FormulaConfig(formula='demand * 2'),
        input_ports=[InputPortDef(id=_port_id('f_in'), identifier='demand', unit=unit_registry.parse_units('MWh/a'))],
        output_ports=[_port('doubled', 'MWh/a', 'energy', ['carrier'])],
        input_dimensions=['carrier'],
    )
    doubled = NodeConfigFactory.create(instance=ic, identifier='doubled', name='Doubled', spec=spec)
    NodeInputPortBinding.objects.create(
        instance=ic, node=doubled, port_id=_port_id('f_in'), source_node=demand, source_port_id=_port_id('demand')
    )

    _create(client, ic, effects=[{'targetNodeId': str(doubled.uuid), 'source': _carrier_narrowing('gas')}])
    doubled = NodeConfig.objects.with_spec().get(pk=doubled.pk)
    assert doubled.spec is not None
    assert len(doubled.spec.input_ports) == 2

    (dataset,) = Dataset.objects.qs.for_node(_action(ic))
    _write(dataset, 'Doubled', 2030, {(('carrier', 'gas'),): -50.0})
    ctx = _rebuild(ic)
    with ctx.run():
        ctx.get_action('heat_pumps').enabled_param.set(True)
        output = ctx.get_node('doubled').get_output_pl()
        assert _by_year(output, 'carrier', 'gas')[2020] == pytest.approx(200)
        assert _by_year(output, 'carrier', 'gas')[2030] == pytest.approx(150)
        assert _by_year(output, 'carrier', 'oil')[2030] == pytest.approx(200)


def test_a_node_with_several_output_ports_is_not_a_target(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
        output_ports=[_port('a', 'MWh/a', 'energy', []), _port('b', 'MWh/a', 'energy', [])],
    )
    split = NodeConfigFactory.create(instance=ic, identifier='split', name='Split', spec=spec)

    editor = client.query_data(EDITOR, variables={'instanceId': ic.identifier, 'target': str(demand.uuid)})
    targets = {entry['nodeId']: entry for entry in editor['modelInstance']['editor']['actionTargets']}
    assert targets[str(demand.uuid)]['available'] is True
    assert targets[str(split.uuid)]['available'] is False
    assert 'output ports' in targets[str(split.uuid)]['reason']

    errors = _create_errors(client, ic, effects=[{'targetNodeId': str(split.uuid)}])
    assert 'output ports' in errors
    assert not NodeConfig.objects.filter(instance=ic, identifier='heat_pumps').exists()


def test_effects_from_an_existing_dataset_and_from_a_node(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    costs = _costs(ic)
    savings = _dataset(ic, 'test/savings', 'GWh/a', {2030: {('oil',): -0.04}}, ('carrier',))
    source_spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
        output_ports=[_port('saving', 'kEUR/a', 'currency', ['sector'])],
    )
    cost_saving = NodeConfigFactory.create(instance=ic, identifier='cost_saving', spec=source_spec)

    result = _create(
        client,
        ic,
        effects=[
            {'targetNodeId': str(demand.uuid), 'source': {'dataset': {'datasetId': str(savings.uuid)}}},
            {'targetNodeId': str(costs.uuid), 'source': {'node': {'nodeId': str(cost_saving.uuid)}}},
        ],
    )

    assert result['datasets'] == []
    action = _action(ic)
    assert not Dataset.objects.qs.for_node(action).exists()
    bindings = NodeInputPortBinding.objects.filter(node=action)
    assert {binding.dataset_id for binding in bindings if binding.dataset_id} == {savings.pk}
    assert {binding.source_node_id for binding in bindings if binding.source_node_id} == {cost_saving.pk}
    assert set(NodeInputPortBinding.objects.filter(source_node=action).values_list('node__identifier', flat=True)) == {
        'demand',
        'costs',
    }
    # An existing dataset gets no zeros written into it.
    assert savings.data_points.count() == 1


def test_a_dataset_without_the_target_dimensions_is_refused(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    total = _dataset(ic, 'test/total', 'MWh/a', {2030: {(): -1}}, ())

    errors = _create_errors(
        client, ic, effects=[{'targetNodeId': str(demand.uuid), 'source': {'dataset': {'datasetId': str(total.uuid)}}}]
    )

    assert 'dimensions differ' in errors
    assert not NodeConfig.objects.filter(instance=ic, identifier='heat_pumps').exists()


def test_a_node_source_downstream_of_the_target_is_refused(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    downstream_spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
        input_ports=[InputPortDef(id=_port_id('t_in'), identifier='in', unit=unit_registry.parse_units('MWh/a'))],
        output_ports=[_port('total', 'MWh/a', 'energy', ['carrier'])],
        input_dimensions=['carrier'],
    )
    downstream = NodeConfigFactory.create(instance=ic, identifier='total', spec=downstream_spec)
    NodeInputPortBinding.objects.create(
        instance=ic, node=downstream, port_id=_port_id('t_in'), source_node=demand, source_port_id=_port_id('demand')
    )

    errors = _create_errors(
        client, ic, effects=[{'targetNodeId': str(demand.uuid), 'source': {'node': {'nodeId': str(downstream.uuid)}}}]
    )

    assert 'loop' in errors
    assert not NodeConfig.objects.filter(instance=ic, identifier='heat_pumps').exists()


def test_candidates_are_the_sources_that_fit(client: PathsTestClient, ic: InstanceConfig) -> None:
    demand = _demand(ic)
    fits = _dataset(ic, 'test/fits', 'GWh/a', {2030: {('oil',): -1}}, ('carrier',))
    _dataset(ic, 'test/no_carrier', 'MWh/a', {2030: {(): -1}}, ())
    _dataset(ic, 'test/wrong_unit', 'MEUR/a', {2030: {('oil',): -1}}, ('carrier',))

    editor = client.query_data(EDITOR, variables={'instanceId': ic.identifier, 'target': str(demand.uuid)})
    candidates = editor['modelInstance']['editor']['effectSourceCandidates']

    datasets = {entry['datasetId'] for entry in candidates['datasets']}
    assert str(fits.uuid) in datasets
    assert len(datasets) == 2  # `fits`, and the target's own input data
    assert candidates['nodes'] == []  # the target is downstream of itself


EDITOR_DATASET = gql("""
query EditorDataset($id: ID!) {
    instance { editor { dataset(id: $id) { id identifier } } }
}
""")


def test_the_editor_opens_the_actions_own_dataset_by_id(client: PathsTestClient, ic: InstanceConfig) -> None:
    """Owned datasets stay out of the instance's lists, but the dataset editor must find them."""
    demand = _demand(ic)
    _create(client, ic, effects=[{'targetNodeId': str(demand.uuid)}])
    (dataset,) = Dataset.objects.qs.for_node(_action(ic))

    data = client.query_data(EDITOR_DATASET, variables={'id': str(dataset.uuid)})

    assert data['instance']['editor']['dataset'] == {'id': str(dataset.uuid), 'identifier': None}


def test_a_yaml_sourced_instance_lists_its_nodes_as_unavailable(client: PathsTestClient, ic: InstanceConfig) -> None:
    """The node graph asks for `actionTargets` on every instance, so it must not fail on one the editor cannot change."""
    demand = _demand(ic)
    InstanceConfig.objects.filter(pk=ic.pk).update(config_source='yaml')

    query = gql("""
    query T($instanceId: ID!) {
        modelInstance(instanceId: $instanceId) { editor { actionTargets { nodeId available reason } } }
    }
    """)
    editor = client.query_data(query, variables={'instanceId': ic.identifier})

    (target,) = editor['modelInstance']['editor']['actionTargets']
    assert target['nodeId'] == str(demand.uuid)
    assert target['available'] is False
    assert 'YAML' in target['reason']


CATEGORIES = gql("""
query Categories($instanceId: ID!, $target: UUID!) {
    modelInstance(instanceId: $instanceId) {
        editor { actionTargetCategories(targetNodeId: $target) { dimensionId categoryIds } }
    }
}
""")


def test_the_effect_offers_only_the_categories_the_target_output_has(client: PathsTestClient, ic: InstanceConfig) -> None:
    """The carrier dimension has three categories, but the target's data has two; only those are offered."""
    dataset = _dataset(ic, 'test/two', 'MWh/a', _history([('oil',), ('electricity',)], 100.0), ('carrier',))
    spec = NodeSpec(
        type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
        input_ports=[
            InputPortDef(id=_port_id('in'), identifier='in', unit=unit_registry.parse_units('MWh/a'), quantity='energy')
        ],
        output_ports=[_port('demand', 'MWh/a', 'energy', ['carrier'])],
        input_dimensions=['carrier'],
    )
    demand = NodeConfigFactory.create(instance=ic, identifier='demand', name='Demand', spec=spec)
    _bind(ic, demand, 'in', dataset)

    data = client.query_data(CATEGORIES, variables={'instanceId': ic.identifier, 'target': str(demand.uuid)})
    (entry,) = data['modelInstance']['editor']['actionTargetCategories']
    assert entry['dimensionId'] == str(_dimension('carrier').uuid)
    assert entry['categoryIds'] == [str(_category('carrier', c).uuid) for c in ('oil', 'electricity')]

    _create(client, ic, effects=[{'targetNodeId': str(demand.uuid)}])
    (owned,) = Dataset.objects.qs.for_node(_action(ic))
    assert {cell: value for cell, value in _cells(owned).items() if cell[0] == 2022} == {
        (2022, 'Demand', ('oil',)): 0.0,
        (2022, 'Demand', ('electricity',)): 0.0,
    }


def test_empty_target_year_cells_leave_the_target_unchanged(client: PathsTestClient, ic: InstanceConfig) -> None:
    """Before the modeller fills in the target year, the action adds nothing."""
    demand = _demand(ic)
    _create(client, ic, effects=[{'targetNodeId': str(demand.uuid)}])

    ctx = _rebuild(ic)
    with ctx.run():
        ctx.get_action('heat_pumps').enabled_param.set(True)
        output = ctx.get_node('demand').get_output_pl()
        for carrier in CARRIERS:
            values = list(_by_year(output, 'carrier', carrier).values())
            assert values == pytest.approx([100.0] * len(values))
