from datetime import date
from typing import TYPE_CHECKING
from uuid import uuid4

from django.db import connection
from django.test.utils import CaptureQueriesContext

import pytest

from kausal_common.datasets.category_domain import DatasetCategoryDomain
from kausal_common.datasets.models import DataPoint, DatasetMetricValidationRule, DimensionScope
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)
from kausal_common.i18n.pydantic import set_i18n_context

from paths.tests.graphql import PathsTestClient

from datasets.data_entry import DataEntryQuery
from datasets.materialization import materialize_dataset
from datasets.validation import InstanceDatasetValidationError
from frameworks.models import DataPointEvidence, DataQualityLevel, DataQualityScheme
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.defs.binding_def import DatasetBindingDef, NodePortRef
from nodes.defs.data_entry import (
    DataEntryDatasetSpec,
    DataEntryPlacementSpec,
    DataEntrySectionSpec,
    DataEntrySliceSpec,
    DataEntrySpec,
)
from nodes.defs.graph import DatasetMeta, DatasetMetricMeta, DimensionCategoryMeta, DimensionMeta
from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import NodeSpec
from nodes.defs.port_def import InputPortDef
from nodes.instance_graph import InstanceGraph, NodeMeta, build_instance_graph
from nodes.instance_serialization import build_instance_snapshot, export_instance, import_instance_copy
from nodes.models import InstanceConfig, NodeInputPortBinding
from nodes.spec_sync import sync_parsed_instance_to_db
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from nodes.units import unit_registry
from nodes.value_validation import QualifierRequirement, ValueContract
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from pathlib import Path

    from django.test import Client

pytestmark = pytest.mark.django_db


def make_query(*, pinned: bool = False) -> DataEntryQuery:
    config = InstanceConfigFactory.create(name='Data entry', config_source='database')
    schema = DatasetSchemaFactory.create(name='Data')
    metric = DatasetMetricFactory.create(schema=schema, name='Value', unit='kWh')
    dataset = DatasetFactory.create(scope=config, schema=schema)
    DataPointFactory.create(dataset=dataset, metric=metric, date=date(2022, 1, 1), value=4)
    point = DataPointFactory.create(dataset=dataset, metric=metric, date=date(2023, 1, 1), value=8)
    revision = dataset.save_revision() if pinned else None
    if pinned:
        point.value = 99
        point.save()
    node, port, section = uuid4(), uuid4(), uuid4()
    with set_i18n_context('en', []):
        graph = InstanceGraph(
            instance_id=config.uuid,
            metadata=InstanceMetadata(primary_language='en'),
            spec=InstanceModelSpec(
                years=YearsSpec(min_historical=2022, max_historical=2024, skipped=[]),
                data_entry=DataEntrySpec(
                    sections=[
                        DataEntrySectionSpec(
                            id=section,
                            name='Section',
                            tables=[
                                DataEntryPlacementSpec(id=uuid4(), node_id=node, port_id=port),
                            ],
                        )
                    ]
                ),
            ),
            nodes=(
                NodeMeta(
                    id=node,
                    identifier='input',
                    node_class_path='nodes.simple.AdditiveNode',
                    spec=NodeSpec(
                        input_ports=[InputPortDef(id=port, binding_owner='instance', unit=unit_registry.parse_units('kWh'))],
                    ),
                ),
            ),
            datasets=(
                DatasetMeta(
                    id=dataset.uuid,
                    schema_id=schema.uuid,
                    metrics=(DatasetMetricMeta(id=metric.uuid, identifier='Value', unit='kWh'),),
                ),
            ),
            pinned_revisions={dataset.uuid: revision.pk} if revision else {},
            bindings=(
                DatasetBindingDef(
                    id=uuid4(),
                    port_ref=NodePortRef(node_uuid=node, node_id='input', port_id=port),
                    dataset_uuid=dataset.uuid,
                    metric_uuid=metric.uuid,
                ),
            ),
        )
    return DataEntryQuery(graph, UserFactory.create(is_superuser=True))


def test_year_selection_and_observation_identity() -> None:
    query = make_query()
    selections = query.graph.data_entry.sections[0].selections
    points = query.points(selections, [2023])
    assert len(points) == 1
    assert points[0].value == 8
    assert points[0].id is not None
    assert query.points(selections, []) == []
    assert len(query.points(selections, None)) == 2
    with pytest.raises(ValueError, match='inventory calendar'):
        query.points(selections, [2021])


def test_pinned_data_does_not_read_newer_observations() -> None:
    query = make_query(pinned=True)
    points = query.points(query.graph.data_entry.sections[0].selections, [2023])
    assert points[0].value == 8
    # The revision records each cell's data point, so a pinned edition names the same cell
    # the draft does, and a comment thread can follow it from one to the other.
    assert points[0].id == DataPoint.objects.get(date__year=2023, dataset__uuid=query.graph.datasets[0].id).uuid


def test_summary_reads_no_data_point_rows() -> None:
    """Findings come from the payload; the data point table is not read for a summary."""
    query = make_query()
    for dataset in query.datasets.values():
        materialize_dataset(dataset)
    with CaptureQueriesContext(connection) as queries:
        query.findings(None, [2023])
    assert not [q for q in queries.captured_queries if 'datasets_datapoint' in q['sql']]


def test_permission_denial_removes_points_and_counts() -> None:
    original = make_query()
    query = DataEntryQuery(original.graph, UserFactory.create())
    assert not query.datasets
    assert query.points(query.graph.data_entry.sections[0].selections, [2023]) == []
    assert query.findings(None, [2023]) == []


def test_required_value_contract_counts_an_entirely_empty_inventory_year() -> None:
    query = make_query()
    query.graph.nodes[0].spec.input_ports[0].validation = ValueContract(required=True)
    findings = query.findings(None, [2024])
    assert len(findings) == 1
    assert findings[0].code == 'missing_required_value'
    assert findings[0].years == (2024,)
    assert query.findings(None, []) == []


def test_a_closed_shape_reports_entered_values_outside_it() -> None:
    original = make_query()
    data = original.graph.model_dump()
    # The table is dimensionless and the domain allows no combination, so every value is outside.
    data['datasets'][0]['category_domain'] = DatasetCategoryDomain(mode='closed')
    with set_i18n_context('en', []):
        query = DataEntryQuery(InstanceGraph.model_validate(data), original.user)
    findings = [finding for finding in query.findings(None, [2023]) if finding.code == 'outside_shape']
    assert [finding.years for finding in findings] == [(2023,)]


def test_graphql_section_query_and_counts(client: Client) -> None:
    query = make_query()
    config = InstanceConfig.objects.get(uuid=query.graph.instance_id)
    config.spec = query.graph.spec
    config.save(update_fields=['spec'])
    node = query.graph.nodes[0]
    row = NodeConfigFactory.create(instance=config, uuid=node.id, identifier='input', spec=node.spec)
    dataset = next(iter(query.datasets.values()))
    assert dataset.schema is not None
    metric = dataset.schema.metrics.get()
    NodeInputPortBinding.objects.create(
        instance=config, node=row, port_id=node.spec.input_ports[0].id, dataset=dataset, metric=metric
    )
    client.force_login(query.user)
    gql = PathsTestClient(client)
    gql.set_instance(config)
    result = gql.query_data("""{
      instance { editor { dataEntry {
        years defaultYear canManageYears
        sections { id name problemCounts(years: [2023]) { total annual yearless }
          tables { isEditable dataPoints(years: [2023]) { id year value } }
        }
      } } }
    }""")['instance']['editor']['dataEntry']
    assert result['years'] == [2022, 2023, 2024]
    assert result['defaultYear'] == 2024
    assert result['sections'][0]['tables'][0]['dataPoints'][0]['value'] == 8
    assert result['sections'][0]['problemCounts']['total'] == 0


def test_shared_findings_count_once_per_section_and_once_globally() -> None:
    original = make_query()
    dataset = next(iter(original.datasets.values()))
    dimension = DimensionFactory.create(name='Sector')
    DimensionScope.objects.create(
        dimension=dimension, scope_content_type=dataset.scope_content_type, scope_id=dataset.scope_id, identifier='sector'
    )
    DatasetSchemaDimensionFactory.create(schema=dataset.schema, dimension=dimension)
    a = DimensionCategoryFactory.create(dimension=dimension, identifier='a')
    b = DimensionCategoryFactory.create(dimension=dimension, identifier='b')
    for point in dataset.data_points.all():
        point.dimension_categories.set([a if point.date.year == 2022 else b])
    assert dataset.schema is not None
    metric = dataset.schema.metrics.get()
    DatasetMetricValidationRule.objects.create(
        metric=metric, rule={'kind': 'dimension_sum', 'enforcement': 'block_publish', 'dimension': 'sector', 'target': 1}
    )
    data = original.graph.model_dump(mode='python')
    data['dimensions'] = [
        DimensionMeta(
            id=dimension.uuid,
            identifier='sector',
            categories=(
                DimensionCategoryMeta(id=a.uuid, identifier='a'),
                DimensionCategoryMeta(id=b.uuid, identifier='b'),
            ),
        ).model_dump()
    ]
    data['datasets'][0]['declared_dimension_ids'] = [dimension.uuid]
    with set_i18n_context('en', []):
        graph = InstanceGraph.model_validate(data)
        definition = graph.spec.data_entry
        assert isinstance(definition, DataEntrySpec)
        first = definition.sections[0]
        assert isinstance(first.tables[0], DataEntryPlacementSpec)
        first.tables[0].slices = [DataEntrySliceSpec(categories={dimension.uuid: [a.uuid]})]
        second = first.model_copy(deep=True)
        second.id = uuid4()
        second.tables[0].id = uuid4()
        assert isinstance(second.tables[0], DataEntryPlacementSpec)
        second.tables[0].slices = [DataEntrySliceSpec(categories={dimension.uuid: [b.uuid]})]
        definition.sections.append(second)
    query = DataEntryQuery(graph, original.user)
    assert len(query.findings(first.id, [2023])) == 1
    with CaptureQueriesContext(connection) as repeated:
        assert len(query.findings(second.id, [2023])) == 1
        assert len(query.findings(None, [2023])) == 1
    assert len(repeated) == 0
    assert len(query._data) == 1


def test_required_grade_uses_evidence_from_the_selected_payload() -> None:
    query = make_query()
    config = InstanceConfig.objects.get(uuid=query.graph.instance_id)
    framework = FrameworkFactory.create(identifier='entry')
    FrameworkConfigFactory.create(framework=framework, instance_config=config)
    scheme = DataQualityScheme.objects.create(framework=framework, identifier='quality', version='1', name='Quality')
    level = DataQualityLevel.objects.create(scheme=scheme, identifier='A', name='Measured', score=1, order=0)
    query.graph.nodes[0].spec.input_ports[0].validation = ValueContract(
        required=True, qualifiers={'entry_quality.coverage': QualifierRequirement(min=1)}
    )
    assert [finding.code for finding in query.findings(None, [2023])] == ['required_qualifier']
    dataset = next(iter(query.datasets.values()))
    point = dataset.data_points.get(date=date(2023, 1, 1))
    DataPointEvidence.objects.create(data_point=point, quality_level=level)
    refreshed = DataEntryQuery(query.graph, query.user)
    assert refreshed.findings(None, [2023]) == []


def test_graphql_keeps_manual_tables_separate_and_in_order(client: Client) -> None:
    query = make_query()
    dataset = next(iter(query.datasets.values()))
    assert dataset.schema is not None
    first_metric = dataset.schema.metrics.get()
    second_metric = DatasetMetricFactory.create(schema=dataset.schema, name='Population', unit='cap')
    DataPointFactory.create(dataset=dataset, metric=second_metric, date=date(2023, 1, 1), value=123)
    config = InstanceConfig.objects.get(uuid=query.graph.instance_id)
    with set_i18n_context('en', []):
        first = DataEntryDatasetSpec(id=uuid4(), dataset_id=dataset.uuid, metric_ids=[second_metric.uuid])
        second = DataEntryDatasetSpec(id=uuid4(), dataset_id=dataset.uuid, metric_ids=[first_metric.uuid])
        config.spec = query.graph.spec.model_copy(deep=True)
        config.spec.data_entry = DataEntrySpec(
            sections=[DataEntrySectionSpec(id=uuid4(), name='Ordered', tables=[first, second])]
        )
    config.save()
    # No bindings or nodes: explicit tables alone must include the dataset in the graph catalog.
    client.force_login(query.user)
    gql = PathsTestClient(client)
    gql.set_instance(config)
    tables = gql.query_data("""{
      instance { editor { dataEntry { sections { tables {
        id entryId dataset { id } metricSelections { metricId }
        dataPoints(years: [2023]) { value }
      } } } } }
    }""")['instance']['editor']['dataEntry']['sections'][0]['tables']
    assert [table['entryId'] for table in tables] == [str(first.id), str(second.id)]
    assert len({table['id'] for table in tables}) == 2
    assert [table['dataset']['id'] for table in tables] == [str(dataset.uuid)] * 2
    assert [table['dataPoints'][0]['value'] for table in tables] == [123, 8]


@pytest.mark.parametrize('identifier', ['city/energy', None])
def test_copy_remaps_unbound_dataset_table(identifier: str | None) -> None:
    query = make_query()
    dataset = next(iter(query.datasets.values()))
    dataset.identifier = identifier
    dataset.save(update_fields=['identifier'])
    assert dataset.schema is not None
    metric = dataset.schema.metrics.get()
    source = InstanceConfig.objects.get(uuid=query.graph.instance_id)
    with set_i18n_context('en', []):
        table = DataEntryDatasetSpec(id=uuid4(), dataset_id=dataset.uuid, metric_ids=[metric.uuid])
        source.spec = InstanceModelSpec(
            data_entry=DataEntrySpec(sections=[DataEntrySectionSpec(id=uuid4(), name='Direct', tables=[table])])
        )
    source.save()
    exported = export_instance(source)
    target = InstanceConfigFactory.create(name='Copy', config_source='database', spec=InstanceModelSpec())
    import_instance_copy(target, exported)
    snapshot = build_instance_snapshot(target)
    assert isinstance(snapshot.spec.data_entry, DataEntrySpec)
    copied = snapshot.spec.data_entry.sections[0].tables[0]
    assert isinstance(copied, DataEntryDatasetSpec)
    assert copied.id != table.id
    assert copied.dataset_id != dataset.uuid
    assert copied.metric_ids != table.metric_ids
    graph = build_instance_graph(snapshot)
    assert graph.data_entry.sections[0].selections[0].dataset_id == copied.dataset_id
    assert copied.metric_ids is not None
    assert graph.data_entry.sections[0].selections[0].metric_id == copied.metric_ids[0]


def test_direct_dataset_respects_revision_pin() -> None:
    original = make_query(pinned=True)
    data = original.graph.model_dump(mode='python')
    data['bindings'] = []
    with set_i18n_context('en', []):
        graph = InstanceGraph.model_validate(data)
        graph.spec.data_entry = DataEntrySpec(
            sections=[
                DataEntrySectionSpec(
                    id=uuid4(),
                    name='Pinned',
                    tables=[DataEntryDatasetSpec(id=uuid4(), dataset_id=graph.datasets[0].id)],
                )
            ]
        )
    query = DataEntryQuery(graph, original.user, published=True)
    assert query.points(graph.data_entry.sections[0].selections, [2023])[0].value == 8
    assert not query.is_editable(graph.datasets[0].id)


def test_yaml_sync_resolves_direct_dataset_and_metric_to_persisted_uuids(tmp_path: Path) -> None:
    query = make_query()
    config = InstanceConfig.objects.get(uuid=query.graph.instance_id)
    dataset = next(iter(query.datasets.values()))
    dataset.identifier = 'city/energy'
    dataset.save(update_fields=['identifier'])
    assert dataset.schema is not None
    metric = dataset.schema.metrics.get()
    config_path = tmp_path / 'direct.yaml'
    config_path.write_text(f"""id: {config.identifier}
name: Direct dataset
owner: Test
default_language: en
target_year: 2030
reference_year: 2023
minimum_historical_year: 2022
nodes: []
datasets:
- id: city/energy
data_entry:
  sections:
  - id: inputs
    name: Inputs
    tables:
    - id: energy
      kind: dataset
      dataset: city/energy
      metrics: [Value]
""")
    sync_parsed_instance_to_db(config.identifier, config_path)
    config.refresh_from_db()
    spec = config.ensure_spec().data_entry
    assert isinstance(spec, DataEntrySpec)
    table = spec.sections[0].tables[0]
    assert isinstance(table, DataEntryDatasetSpec)
    assert table.dataset_id == dataset.uuid
    assert table.metric_ids == [metric.uuid]
    assert build_instance_graph(build_instance_snapshot(config)).data_entry.sections[0].selections


def test_layout_errors_are_editor_problems_not_section_findings(client: Client) -> None:
    query = make_query()
    config = InstanceConfig.objects.get(uuid=query.graph.instance_id)
    dataset = next(iter(query.datasets.values()))
    with set_i18n_context('en', []):
        bad = DataEntryPlacementSpec(id=uuid4(), node_id=uuid4(), port_id=uuid4())
        config.spec = query.graph.spec.model_copy(deep=True)
        section = DataEntrySectionSpec(
            id=uuid4(),
            name='Inputs',
            tables=[
                DataEntryDatasetSpec(id=uuid4(), dataset_id=dataset.uuid),
                bad,
            ],
        )
        config.spec.data_entry = DataEntrySpec(sections=[section])
    config.save()
    client.force_login(query.user)
    gql = PathsTestClient(client)
    gql.set_instance(config)
    result = gql.query_data("""{
      instance { editor {
        problems { __typename code enforcement ... on DataEntryDefinitionProblem { sectionId nodeId portId } }
        dataEntry {
          problemCounts(years: [2023]) { total }
          sections { problems(years: [2023]) { code } problemCounts(years: [2023]) { total } }
        }
      } }
    }""")['instance']['editor']
    problem = next(problem for problem in result['problems'] if problem['code'] == 'invalid_placement')
    assert problem['enforcement'] == 'BLOCK_PUBLISH'
    assert problem['sectionId'] == str(section.id)
    assert problem['nodeId'] == str(bad.node_id)
    assert result['dataEntry']['problemCounts']['total'] == 0
    assert result['dataEntry']['sections'][0]['problems'] == []
    published = gql.query_data(
        """mutation Publish($id: ID!) {
      instanceEditor(instanceId: $id) { publishModelInstance(instanceId: $id) {
        __typename ... on DataEntryDefinitionProblems { problems { code enforcement } }
      } }
    }""",
        variables={'id': str(config.pk)},
    )['instanceEditor']['publishModelInstance']
    assert published['__typename'] == 'DataEntryDefinitionProblems'
    assert published['problems'][0]['code'] == 'invalid_placement'
    config.refresh_from_db()
    assert config.live_revision_id is None


def test_submission_only_dataset_rule_allows_publication_but_blocks_submission() -> None:
    query = make_query()
    config = InstanceConfig.objects.get(uuid=query.graph.instance_id)
    dataset = next(iter(query.datasets.values()))
    assert dataset.schema is not None
    metric = dataset.schema.metrics.get()
    DatasetMetricValidationRule.objects.create(
        metric=metric,
        rule={'kind': 'value_range', 'max': 1, 'enforcement': 'block_submission'},
    )
    with set_i18n_context('en', []):
        config.spec = query.graph.spec.model_copy(deep=True)
        config.spec.data_entry = DataEntrySpec(
            sections=[
                DataEntrySectionSpec(
                    id=uuid4(),
                    name='Input',
                    tables=[
                        DataEntryDatasetSpec(id=uuid4(), dataset_id=dataset.uuid),
                    ],
                )
            ]
        )
    config.save()
    config.publish_instance()
    config.refresh_from_db()
    revision = config.live_revision_id
    assert revision is not None
    with pytest.raises(InstanceDatasetValidationError) as exc:
        config.publish_instance(require_submittable=True)
    assert {violation.enforcement for violation in exc.value.violations} == {'block_submission'}
    config.refresh_from_db()
    assert config.live_revision_id == revision
