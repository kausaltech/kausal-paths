"""
GraphQL tests for input-port bindings: datasets bound to ports and edge bindings.

The dataset workflow these cover, which is the point of the surface: add an
input port, bind a dataset metric to it, then change the transformations to
keep one category of a dimension and drop the rest along with the dimension.
The edge tests cover the same ``bindingEditor`` resolving ``NodeEdge`` rows,
with its own kind-typed update mutation and the legacy vocabulary presented
and stored in the current one.
"""

from typing import TYPE_CHECKING, Any

import pytest

from kausal_common.datasets.tests.factories import DatasetFactory, DatasetMetricFactory

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import NodeKind, NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory, _port_id
from nodes.units import unit_registry

if TYPE_CHECKING:
    from paths.tests.graphql import PathsTestClient

    from nodes.models import InstanceConfig

gql = str

pytestmark = pytest.mark.django_db

SIMPLE_NODE_CLASS = 'nodes.simple.SimpleNode'


@pytest.fixture
def gql_client(client, db_instance_config: InstanceConfig) -> PathsTestClient:
    """Return a client wired to the instance and authenticated as a superuser."""
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(db_instance_config)
    return tc


@pytest.fixture
def db_instance_config() -> InstanceConfig:
    instance = InstanceFactory.create()
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        spec=InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030)),
    )


def _node_spec(**overrides: Any) -> NodeSpec:
    unit = unit_registry.parse_units('kt/a')
    defaults: dict[str, Any] = {
        'kind': NodeKind.SIMPLE,
        'type_config': SimpleConfig(node_class=SIMPLE_NODE_CLASS),
        'input_ports': [],
        'output_ports': [OutputPortDef(id=_port_id('default'), unit=unit, quantity='emissions')],
    }
    defaults.update(overrides)
    return NodeSpec(**defaults)


BIND_DATASET = gql("""
    mutation BindDataset($instanceId: ID!, $nodeId: ID!, $input: BindDatasetInput!) {
      instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
          bindDataset(input: $input) {
            ... on DatasetPortType {
              id
              portRef { nodeId portId }
              nodeRef { nodeId portId }
              metric { name }
              transformations {
                __typename
                ... on PortTransformation { kind isSystemManaged }
                ... on FilterDimensionType { dimension categories flatten }
              }
            }
            ... on OperationInfo { messages { kind message } }
          }
        }
      }
    }
""")

UPDATE_BINDING = gql("""
    mutation UpdateBinding($instanceId: ID!, $bindingId: ID!, $input: UpdateDatasetBindingInput!) {
      instanceEditor(instanceId: $instanceId) {
        bindingEditor(bindingId: $bindingId) {
          updateDatasetBinding(input: $input) {
            ... on DatasetPortType {
              id
              tags
              metric { name }
              transformations {
                __typename
                ... on PortTransformation { kind isSystemManaged }
                ... on FilterDimensionType { dimension categories flatten }
              }
            }
            ... on OperationInfo { messages { kind message } }
          }
        }
      }
    }
""")

DELETE_BINDING = gql("""
    mutation DeleteBinding($instanceId: ID!, $bindingId: ID!) {
      instanceEditor(instanceId: $instanceId) {
        bindingEditor(bindingId: $bindingId) {
          deleteBinding { messages { kind message } }
        }
      }
    }
""")

ADD_INPUT_PORT = gql("""
    mutation AddPort($instanceId: ID!, $nodeId: ID!, $input: InputPortInput!) {
      instanceEditor(instanceId: $instanceId) {
        nodeEditor(nodeId: $nodeId) {
          addInputPort(input: $input) {
            ... on InputPortType { id identifier }
            ... on OperationInfo { messages { kind message } }
          }
        }
      }
    }
""")


def _dataset_with_metric(ic: InstanceConfig, identifier: str = 'heating', metric: str = 'Energy', unit: str = 'kt/a'):
    dataset = DatasetFactory.create(identifier=identifier, scope=ic)
    dataset_metric = DatasetMetricFactory.create(schema=dataset.schema, name=metric, label=metric, unit=unit)
    return dataset, dataset_metric


def test_add_port_then_bind_a_dataset_metric_to_it(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """The whole point of the surface: a port, then a dataset on it, in two calls."""
    NodeConfigFactory.create(instance=db_instance_config, identifier='consumer', spec=_node_spec())
    dataset, metric = _dataset_with_metric(db_instance_config)

    port = gql_client.query_data(
        ADD_INPUT_PORT,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'identifier': 'heating', 'unit': 'kt/a', 'quantity': 'emissions'},
        },
    )['instanceEditor']['nodeEditor']['addInputPort']
    assert port['identifier'] == 'heating'

    binding = gql_client.query_data(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': port['id'], 'datasetId': str(dataset.uuid), 'metricId': str(metric.uuid)},
        },
    )['instanceEditor']['nodeEditor']['bindDataset']

    assert binding['portRef']['portId'] == port['id']
    assert binding['nodeRef'] == binding['portRef']
    assert binding['metric']['name'] == 'Energy'
    # A default list is generated, so a client needs no knowledge of the
    # generated markers just to create a working binding — and the interface
    # tells it uniformly which entries to render read-only.
    assert [(t['kind'], t['isSystemManaged']) for t in binding['transformations']] == [
        ('select_metric', True),
        ('index_temporal', True),
        ('remap_legacy_years', True),
    ]


def test_a_port_can_be_addressed_by_its_identifier(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    node = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'))]
        ),
    )
    _dataset_with_metric(db_instance_config)

    binding = gql_client.query_data(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'heating', 'datasetId': 'heating', 'metricId': 'Energy'},
        },
    )['instanceEditor']['nodeEditor']['bindDataset']

    assert binding['portRef']['portId'] == str(_port_id('input'))
    assert node.identifier == binding['portRef']['nodeId']


def test_transformations_are_replaced_as_a_whole_list(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """
    Keep one category of a dimension and drop the rest along with the dimension.

    That is a single `filterDimension` with `flatten`: selecting one category and
    summing over the dimension leaves the value unchanged and the column gone.
    """
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'))]
        ),
    )
    _dataset_with_metric(db_instance_config)
    binding_id = gql_client.query_data(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'heating', 'datasetId': 'heating', 'metricId': 'Energy'},
        },
    )['instanceEditor']['nodeEditor']['bindDataset']['id']

    updated = gql_client.query_data(
        UPDATE_BINDING,
        variables={
            'instanceId': str(db_instance_config.pk),
            'bindingId': binding_id,
            'input': {
                'transformations': [
                    {'selectMetric': True},
                    {'indexTemporal': True},
                    {'remapLegacyYears': True},
                    {'filterDimension': {'dimension': 'building_heat_source', 'categories': ['electricity'], 'flatten': True}},
                ],
                'tags': ['inventory_only'],
            },
        },
    )['instanceEditor']['bindingEditor']['updateDatasetBinding']

    kinds = [t['__typename'] for t in updated['transformations']]
    assert kinds == ['SelectMetricType', 'IndexTemporalType', 'RemapLegacyYearsType', 'FilterDimensionType']
    last = updated['transformations'][-1]
    assert last['dimension'] == 'building_heat_source'
    assert last['categories'] == ['electricity']
    assert last['flatten'] is True
    assert updated['tags'] == ['inventory_only']


def test_dropping_the_generated_metric_selection_is_rejected(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """
    A binding that names a metric column must keep selecting it.

    Whole-list replacement means a client can leave out a transformation it
    didn't understand; that has to fail loudly rather than silently change what
    the binding reads.
    """
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'))]
        ),
    )
    _dataset_with_metric(db_instance_config)
    binding_id = gql_client.query_data(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'heating', 'datasetId': 'heating', 'metricId': 'Energy'},
        },
    )['instanceEditor']['nodeEditor']['bindDataset']['id']

    gql_client.query_errors(
        UPDATE_BINDING,
        variables={
            'instanceId': str(db_instance_config.pk),
            'bindingId': binding_id,
            'input': {'transformations': [{'dropNulls': True}]},
        },
        assert_error_message='must include `selectMetric`',
    )


def test_edge_vocabulary_is_rejected_on_a_dataset_binding(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """Applicability is the input type's field list: the edge vocabulary cannot even be expressed."""
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'))]
        ),
    )
    _dataset_with_metric(db_instance_config)
    binding_id = gql_client.query_data(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'heating', 'datasetId': 'heating', 'metricId': 'Energy'},
        },
    )['instanceEditor']['nodeEditor']['bindDataset']['id']

    gql_client.query_errors(
        UPDATE_BINDING,
        variables={
            'instanceId': str(db_instance_config.pk),
            'bindingId': binding_id,
            'input': {'transformations': [{'selectCategories': {'dimension': 'sector', 'categories': ['a']}}]},
        },
        assert_error_message='DatasetTransformationInput',
    )


def test_binding_to_an_occupied_non_multi_port_is_rejected(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[
                InputPortDef(
                    id=_port_id('input'),
                    identifier='heating',
                    unit=unit_registry.parse_units('kt/a'),
                    multi=False,
                )
            ]
        ),
    )
    _dataset_with_metric(db_instance_config)
    variables = {
        'instanceId': str(db_instance_config.pk),
        'nodeId': 'consumer',
        'input': {'portId': 'heating', 'datasetId': 'heating', 'metricId': 'Energy'},
    }
    gql_client.query_data(BIND_DATASET, variables=variables)

    gql_client.query_errors(BIND_DATASET, variables=variables, assert_error_message='already has a dataset bound')


def test_binding_a_metric_whose_unit_does_not_fit_the_port_is_rejected(
    gql_client: PathsTestClient, db_instance_config: InstanceConfig
):
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'))]
        ),
    )
    _dataset_with_metric(db_instance_config, metric='Area', unit='m**2')

    gql_client.query_errors(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'heating', 'datasetId': 'heating', 'metricId': 'Area'},
        },
        assert_error_message='not compatible with port unit',
    )


def test_binding_to_a_port_that_does_not_exist_is_rejected(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """Ports are created explicitly; binding never conjures one."""
    NodeConfigFactory.create(instance=db_instance_config, identifier='consumer', spec=_node_spec())
    _dataset_with_metric(db_instance_config)

    gql_client.query_errors(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'nonexistent', 'datasetId': 'heating', 'metricId': 'Energy'},
        },
        assert_error_message='not found on node',
    )


def test_deleting_a_binding_leaves_the_port(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import DatasetPort

    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'))]
        ),
    )
    _dataset_with_metric(db_instance_config)
    binding_id = gql_client.query_data(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'heating', 'datasetId': 'heating', 'metricId': 'Energy'},
        },
    )['instanceEditor']['nodeEditor']['bindDataset']['id']

    gql_client.query_data(
        DELETE_BINDING,
        variables={'instanceId': str(db_instance_config.pk), 'bindingId': binding_id},
    )

    assert not DatasetPort.objects.filter(node=nc).exists()
    nc.refresh_from_db()
    assert nc.spec is not None
    assert [port.identifier for port in nc.spec.input_ports] == ['heating']


# ---------------------------------------------------------------------------
# Metric validation on bind and update
# ---------------------------------------------------------------------------


def test_omitted_metric_id_still_validates_the_unit(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """The sole-metric convenience path must not skip the unit-fit check."""
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'))]
        ),
    )
    _dataset_with_metric(db_instance_config, metric='Area', unit='m**2')

    gql_client.query_errors(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'heating', 'datasetId': 'heating'},
        },
        assert_error_message='not compatible with port unit',
    )


def test_nulls_on_bind_mean_the_defaults(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """
    On bindDataset, null and omitted both mean "use the default": it is create-style.

    Apollo-style clients null-fill unspecified optional variables, so these
    fields are plain optionals rather than Maybe — there is no existing value
    for an absent field to leave untouched.
    """
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'))]
        ),
    )
    _dataset_with_metric(db_instance_config)

    binding = gql_client.query_data(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'heating', 'datasetId': 'heating', 'metricId': None, 'transformations': None},
        },
    )['instanceEditor']['nodeEditor']['bindDataset']

    # Null metricId falls back to the dataset's sole metric; null
    # transformations get the generated default list — column-less here, so
    # the whole frame is consumed and no selectMetric marker is needed.
    assert binding['metric']['name'] == 'Energy'
    assert [t['kind'] for t in binding['transformations']] == ['index_temporal', 'remap_legacy_years']


def _bound_binding_id(gql_client: PathsTestClient, ic: InstanceConfig) -> str:
    return gql_client.query_data(
        BIND_DATASET,
        variables={
            'instanceId': str(ic.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'heating', 'datasetId': 'heating', 'metricId': 'Energy'},
        },
    )['instanceEditor']['nodeEditor']['bindDataset']['id']


def test_changing_the_metric_updates_the_column_and_checks_the_unit(
    gql_client: PathsTestClient, db_instance_config: InstanceConfig
):
    from nodes.models import DatasetPort

    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'))]
        ),
    )
    dataset, _ = _dataset_with_metric(db_instance_config)
    DatasetMetricFactory.create(schema=dataset.schema, name='Emissions', label='Emissions', unit='t/a')
    DatasetMetricFactory.create(schema=dataset.schema, name='Area', label='Area', unit='m**2')
    binding_id = _bound_binding_id(gql_client, db_instance_config)

    # A metric whose unit cannot supply the port is rejected on update too,
    # not only at bind time.
    gql_client.query_errors(
        UPDATE_BINDING,
        variables={
            'instanceId': str(db_instance_config.pk),
            'bindingId': binding_id,
            'input': {'metricId': 'Area'},
        },
        assert_error_message='not compatible with port unit',
    )

    updated = gql_client.query_data(
        UPDATE_BINDING,
        variables={
            'instanceId': str(db_instance_config.pk),
            'bindingId': binding_id,
            'input': {'metricId': 'Emissions'},
        },
    )['instanceEditor']['bindingEditor']['updateDatasetBinding']

    assert updated['metric']['name'] == 'Emissions'
    row = DatasetPort.objects.get(uuid=binding_id)
    assert row.metric.name == 'Emissions'
    # The column follows the new metric.
    assert row.spec.column == 'Emissions'


def test_metric_of_a_fanned_out_binding_cannot_be_changed(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """A column-less binding has one row per metric; stamping one metric on all rows would corrupt it."""
    from nodes.models import DatasetPort

    unit = unit_registry.parse_units('kt/a')
    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[
                InputPortDef(id=_port_id('input'), identifier='heating', unit=unit),
                InputPortDef(id=_port_id('input2'), identifier='cooling', unit=unit),
            ]
        ),
    )
    dataset, metric_a = _dataset_with_metric(db_instance_config)
    metric_b = DatasetMetricFactory.create(schema=dataset.schema, name='Emissions', label='Emissions', unit='t/a')
    rows = [
        DatasetPort.objects.create(
            instance=db_instance_config,
            node=nc,
            port_id=port_id,
            dataset=dataset,
            metric=metric,
            dataset_index=0,
        )
        for port_id, metric in ((_port_id('input'), metric_a), (_port_id('input2'), metric_b))
    ]

    gql_client.query_errors(
        UPDATE_BINDING,
        variables={
            'instanceId': str(db_instance_config.pk),
            'bindingId': str(rows[0].uuid),
            'input': {'metricId': 'Emissions'},
        },
        assert_error_message='cannot be changed',
    )


# ---------------------------------------------------------------------------
# Replacing bindings atomically with bindDataset(replace: true)
# ---------------------------------------------------------------------------


def test_replace_swaps_the_dataset_on_an_occupied_port(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import DatasetPort

    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[
                InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'), multi=False)
            ]
        ),
    )
    _dataset_with_metric(db_instance_config, identifier='old_data')
    new_dataset, _ = _dataset_with_metric(db_instance_config, identifier='new_data')

    variables: dict[str, Any] = {
        'instanceId': str(db_instance_config.pk),
        'nodeId': 'consumer',
        'input': {'portId': 'heating', 'datasetId': 'old_data', 'metricId': 'Energy'},
    }
    gql_client.query_data(BIND_DATASET, variables=variables)

    variables['input'] = {'portId': 'heating', 'datasetId': 'new_data', 'metricId': 'Energy', 'replace': True}
    binding = gql_client.query_data(BIND_DATASET, variables=variables)['instanceEditor']['nodeEditor']['bindDataset']

    assert binding['portRef']['portId'] == str(_port_id('input'))
    rows = list(DatasetPort.objects.filter(node=nc))
    assert len(rows) == 1
    assert rows[0].dataset == new_dataset


def test_replace_displaces_an_edge_binding(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """Edge-to-dataset replacement is one atomic call; two mutations would leave the port observably unbound."""
    from nodes.models import DatasetPort, NodeEdge

    edge = _edge_between_two_nodes(db_instance_config)
    _dataset_with_metric(db_instance_config)

    gql_client.query_data(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'heating', 'datasetId': 'heating', 'metricId': 'Energy', 'replace': True},
        },
    )

    assert not NodeEdge.objects.filter(pk=edge.pk).exists()
    assert DatasetPort.objects.filter(node=edge.to_node, port_id=_port_id('input')).count() == 1


def test_replace_on_a_multi_port_is_rejected(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[
                InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'), multi=True)
            ]
        ),
    )
    _dataset_with_metric(db_instance_config)

    gql_client.query_errors(
        BIND_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'nodeId': 'consumer',
            'input': {'portId': 'heating', 'datasetId': 'heating', 'metricId': 'Energy', 'replace': True},
        },
        assert_error_message='`replace` is ambiguous',
    )


def test_a_rejected_replace_leaves_the_old_binding(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """Validation runs before anything is deleted, so a failed bind cannot strand the port."""
    from nodes.models import DatasetPort

    nc = NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='consumer',
        spec=_node_spec(
            input_ports=[
                InputPortDef(id=_port_id('input'), identifier='heating', unit=unit_registry.parse_units('kt/a'), multi=False)
            ]
        ),
    )
    old_dataset, _ = _dataset_with_metric(db_instance_config, identifier='old_data')
    _dataset_with_metric(db_instance_config, identifier='wrong_unit', metric='Area', unit='m**2')

    variables: dict[str, Any] = {
        'instanceId': str(db_instance_config.pk),
        'nodeId': 'consumer',
        'input': {'portId': 'heating', 'datasetId': 'old_data', 'metricId': 'Energy'},
    }
    gql_client.query_data(BIND_DATASET, variables=variables)

    variables['input'] = {'portId': 'heating', 'datasetId': 'wrong_unit', 'metricId': 'Area', 'replace': True}
    gql_client.query_errors(BIND_DATASET, variables=variables, assert_error_message='not compatible with port unit')

    rows = list(DatasetPort.objects.filter(node=nc))
    assert len(rows) == 1
    assert rows[0].dataset == old_dataset


# ---------------------------------------------------------------------------
# Edge bindings through the same bindingEditor
# ---------------------------------------------------------------------------

UPDATE_EDGE_BINDING = gql("""
    mutation UpdateEdgeBinding($instanceId: ID!, $bindingId: ID!, $input: UpdateEdgeBindingInput!) {
      instanceEditor(instanceId: $instanceId) {
        bindingEditor(bindingId: $bindingId) {
          updateEdgeBinding(input: $input) {
            ... on NodeEdgeType {
              id
              tags
              transformations {
                __typename
                ... on FilterDimensionType { dimension categories groups flatten exclude }
                ... on AssignDimensionType { dimension category }
                ... on FlattenType { dimension }
              }
            }
            ... on OperationInfo { messages { kind message } }
          }
        }
      }
    }
""")


def _edge_between_two_nodes(ic: InstanceConfig, transformations: list[Any] | None = None):
    from nodes.models import NodeEdge

    unit = unit_registry.parse_units('kt/a')
    producer = NodeConfigFactory.create(instance=ic, identifier='producer', spec=_node_spec())
    consumer = NodeConfigFactory.create(
        instance=ic,
        identifier='consumer',
        spec=_node_spec(input_ports=[InputPortDef(id=_port_id('input'), identifier='heating', unit=unit)]),
    )
    return NodeEdge.objects.create(
        instance=ic,
        from_node=producer,
        from_port=_port_id('default'),
        to_node=consumer,
        to_port=_port_id('input'),
        transformations=transformations or [],
    )


def test_edge_transformations_are_updated_through_the_binding_editor(
    gql_client: PathsTestClient, db_instance_config: InstanceConfig
):
    edge = _edge_between_two_nodes(db_instance_config)

    updated = gql_client.query_data(
        UPDATE_EDGE_BINDING,
        variables={
            'instanceId': str(db_instance_config.pk),
            'bindingId': str(edge.uuid),
            'input': {
                'transformations': [
                    {'filterDimension': {'dimension': 'sector', 'groups': ['industry'], 'flatten': True}},
                    {'assignDimension': {'dimension': 'scope', 'category': 'scope1'}},
                ],
                'tags': ['difference'],
            },
        },
    )['instanceEditor']['bindingEditor']['updateEdgeBinding']

    assert updated['tags'] == ['difference']
    assert [t['__typename'] for t in updated['transformations']] == ['FilterDimensionType', 'AssignDimensionType']
    assert updated['transformations'][0]['groups'] == ['industry']

    edge.refresh_from_db()
    assert [op.kind for op in edge.transformations] == ['filter_dimension', 'assign_dimension']


def test_legacy_edge_rows_are_presented_and_stored_in_the_current_vocabulary(
    gql_client: PathsTestClient, db_instance_config: InstanceConfig
):
    """A tags-only update of a legacy row converges its stored transformations too."""
    from nodes.defs.transform_def import SelectCategoriesTransformation

    edge = _edge_between_two_nodes(
        db_instance_config,
        transformations=[SelectCategoriesTransformation(dimension='sector', categories=['buildings'], flatten=True)],
    )

    updated = gql_client.query_data(
        UPDATE_EDGE_BINDING,
        variables={
            'instanceId': str(db_instance_config.pk),
            'bindingId': str(edge.uuid),
            'input': {'tags': ['non_additive']},
        },
    )['instanceEditor']['bindingEditor']['updateEdgeBinding']

    assert updated['transformations'] == [
        {
            '__typename': 'FilterDimensionType',
            'dimension': 'sector',
            'categories': ['buildings'],
            'groups': [],
            'flatten': True,
            'exclude': False,
        },
    ]
    edge.refresh_from_db()
    assert [op.kind for op in edge.transformations] == ['filter_dimension']


def test_the_kind_typed_update_mutations_reject_the_other_kind(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    edge = _edge_between_two_nodes(db_instance_config)

    gql_client.query_errors(
        UPDATE_BINDING,
        variables={
            'instanceId': str(db_instance_config.pk),
            'bindingId': str(edge.uuid),
            'input': {'tags': []},
        },
        assert_error_message='use updateEdgeBinding',
    )


def test_deleting_an_edge_binding_leaves_the_ports(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    from nodes.models import NodeEdge

    edge = _edge_between_two_nodes(db_instance_config)

    gql_client.query_data(
        DELETE_BINDING,
        variables={'instanceId': str(db_instance_config.pk), 'bindingId': str(edge.uuid)},
    )

    assert not NodeEdge.objects.filter(pk=edge.pk).exists()
    consumer = edge.to_node
    consumer.refresh_from_db()
    assert consumer.spec is not None
    assert [port.identifier for port in consumer.spec.input_ports] == ['heating']
