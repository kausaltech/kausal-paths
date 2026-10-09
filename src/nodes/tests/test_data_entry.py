from pathlib import Path
from uuid import UUID, uuid4

import pytest

from kausal_common.i18n.pydantic import set_i18n_context

from nodes.data_entry import EntryResolver, intersect, rectangle, subtract
from nodes.data_entry_yaml import YAMLDataEntry, YAMLDataset, YAMLSection, resolve_yaml_data_entry
from nodes.defs.binding_def import DatasetBindingDef, EdgeBindingDef, NodePortRef
from nodes.defs.data_entry import (
    ComposedDataEntrySpec,
    DataEntryAnchorSpec,
    DataEntryAutodiscoverSpec,
    DataEntryDatasetSpec,
    DataEntryPlacementSpec,
    DataEntrySectionAmendmentSpec,
    DataEntrySectionSpec,
    DataEntrySliceSpec,
    DataEntrySpec,
    remap_data_entry,
)
from nodes.defs.graph import DatasetMeta, DatasetMetricMeta, DimensionCategoryMeta, DimensionMeta
from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec
from nodes.defs.node_defs import NodeSpec
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.defs.transform_def import AssignDimensionOp, FilterColumnOp, FilterDimensionOp
from nodes.instance_graph import InstanceGraph, NodeMeta
from nodes.instance_loader import InstanceYAMLConfig
from nodes.instance_parser import parse_instance_snapshot
from nodes.instance_problems import data_entry_definition_problems
from nodes.instance_serialization import export_instance, import_instance_copy
from nodes.models import InstanceConfig
from nodes.node import Node
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


@pytest.fixture
def graph() -> InstanceGraph:
    dimension, a, b = uuid4(), uuid4(), uuid4()
    node, port, dataset, metric = (uuid4() for _ in range(4))
    with set_i18n_context('en', []):
        return InstanceGraph(
            instance_id=uuid4(),
            metadata=InstanceMetadata(),
            spec=InstanceModelSpec(),
            dimensions=(
                DimensionMeta(
                    id=dimension,
                    identifier='sector',
                    categories=(
                        DimensionCategoryMeta(id=a, identifier='a'),
                        DimensionCategoryMeta(id=b, identifier='b'),
                    ),
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
                    id=dataset, schema_id=uuid4(), declared_dimension_ids=(dimension,), metrics=(DatasetMetricMeta(id=metric),)
                ),
            ),
            bindings=(
                DatasetBindingDef(
                    id=uuid4(),
                    port_ref=NodePortRef(node_uuid=node, node_id='input', port_id=port),
                    dataset_uuid=dataset,
                    metric_uuid=metric,
                ),
            ),
        )


def section(graph: InstanceGraph, categories: list[UUID] | None = None) -> DataEntrySectionSpec:
    with set_i18n_context('en', []):
        return DataEntrySectionSpec(
            id=uuid4(),
            name='Section',
            tables=[
                DataEntryPlacementSpec(
                    id=uuid4(),
                    node_id=graph.nodes[0].id,
                    port_id=graph.nodes[0].spec.input_ports[0].id,
                    slices=[] if categories is None else [DataEntrySliceSpec(categories={graph.dimensions[0].id: categories})],
                )
            ],
        )


def test_rectangle_difference_preserves_holes() -> None:
    x, y = uuid4(), uuid4()
    a, b, c, d = (uuid4() for _ in range(4))
    universe = rectangle({x: frozenset((a, b)), y: frozenset((c, d))})
    selected = rectangle({x: frozenset((a,)), y: frozenset((c,))})
    remainder = subtract(universe, selected)
    assert len(remainder) == 2
    assert all(intersect(item, selected) is None for item in remainder)
    assert intersect(*remainder) is None
    assert sum(len(dict(item)[x]) * len(dict(item)[y]) for item in remainder) == 3


def test_partition_keeps_partial_unplaced_selection(graph: InstanceGraph) -> None:
    dimension = graph.dimensions[0]
    graph.spec.data_entry = DataEntrySpec(sections=[section(graph, [dimension.categories[0].id])])
    layout = graph.data_entry
    assert len(layout.sections) == 2
    assert layout.sections[1].kind == 'unplaced'
    assert dict(layout.sections[1].selections[0].rectangles[0])[dimension.id] == frozenset((dimension.categories[1].id,))


def test_local_claim_precedes_template_display_order(graph: InstanceGraph) -> None:
    inherited = section(graph)
    local = section(graph, [graph.dimensions[0].categories[0].id])
    graph.spec.data_entry = ComposedDataEntrySpec(
        template=DataEntrySpec(sections=[inherited]), local=DataEntrySpec(sections=[local])
    )
    layout = graph.data_entry
    assert [item.id for item in layout.sections] == [inherited.id, local.id]
    assert not intersect(layout.sections[0].selections[0].rectangles[0], layout.sections[1].selections[0].rectangles[0])
    assert layout.sections[0].selections[0].placement == 'template_explicit'


def test_amendment_preserves_omitted_metadata() -> None:
    with set_i18n_context('en', []):
        amendment = DataEntrySectionAmendmentSpec(section_id=uuid4())
        restored = DataEntrySectionAmendmentSpec.model_validate_json(amendment.model_dump_json())
        assert restored.model_fields_set == {'section_id', 'tables'}
        assert 'description' not in restored.model_dump()
        assert DataEntrySectionAmendmentSpec(section_id=uuid4(), description=None).model_dump()['description'] is None
        with pytest.raises(ValueError, match='cannot be null'):
            DataEntrySectionAmendmentSpec(section_id=uuid4(), name=None)


def test_composed_layout_survives_graph_serialization(graph: InstanceGraph) -> None:
    graph.spec.data_entry = ComposedDataEntrySpec(template=DataEntrySpec(sections=[section(graph)]), local=DataEntrySpec())
    with set_i18n_context('en', []):
        restored = InstanceGraph.model_validate_json(graph.model_dump_json())
        assert restored.data_entry.sections[0].selections == graph.data_entry.sections[0].selections
        assert str(restored.data_entry.sections[0].name) == str(graph.data_entry.sections[0].name)


def test_invalid_placement_remains_inspectable(graph: InstanceGraph) -> None:
    invalid = section(graph)
    assert isinstance(invalid.tables[0], DataEntryPlacementSpec)
    invalid.tables[0].port_id = uuid4()
    graph.spec.data_entry = DataEntrySpec(sections=[invalid])
    assert graph.data_entry.sections[0].problems[0].code == 'invalid_placement'
    assert graph.data_entry.sections[-1].kind == 'unplaced'


def test_bisko_yaml_sections_have_stable_shared_ids() -> None:

    config = InstanceYAMLConfig.load_for_entrypoint(Path('configs/pruefstadt-bisko.yaml'))
    assert config.data is not None
    first = parse_instance_snapshot(config.data, instance_uuid=uuid4())
    second = parse_instance_snapshot(config.data, instance_uuid=uuid4())
    assert isinstance(first.spec.data_entry, DataEntrySpec)
    assert isinstance(second.spec.data_entry, DataEntrySpec)
    assert [section.id for section in first.spec.data_entry.sections] == [
        section.id for section in second.spec.data_entry.sections
    ]
    assert len(first.spec.data_entry.sections) == 8
    nodes = {node.uuid: node for node in first.nodes}
    for section in first.spec.data_entry.sections:
        for table in section.tables:
            assert isinstance(table, DataEntryPlacementSpec)
            assert nodes[table.node_id].identifier != 'passenger_kilometers_own'
            spec = nodes[table.node_id].spec
            assert spec is not None
            assert any(port.id == table.port_id and port.binding_owner == 'instance' for port in spec.input_ports)


def test_excluded_filter_and_assignment_translate_backwards(graph: InstanceGraph) -> None:

    dim = graph.dimensions[0]
    binding = graph.bindings[0]
    binding.transformations.extend([
        FilterDimensionOp(dimension='sector', categories=['a'], exclude=True, flatten=True),
        AssignDimensionOp(dimension='sector', category='a'),
    ])
    resolver = EntryResolver(graph)
    selected, reasons = resolver.translate(binding, rectangle({dim.id: frozenset([dim.categories[0].id])}))
    assert selected == rectangle({dim.id: frozenset([dim.categories[1].id])})
    assert not reasons
    assert resolver.translate(binding, rectangle({dim.id: frozenset([dim.categories[1].id])}))[0] is None


def test_unknown_filter_widens_and_explicit_claim_reports_problem(graph: InstanceGraph) -> None:

    graph.bindings[0].transformations.append(FilterColumnOp(column='sector', ref='selected_sector'))
    resolver = EntryResolver(graph)
    selected, reasons = resolver.translate(graph.bindings[0], rectangle({graph.dimensions[0].id: frozenset([uuid4()])}))
    assert selected == ()
    assert reasons
    graph.spec.data_entry = DataEntrySpec(sections=[section(graph)])
    assert graph.data_entry.sections[0].problems[0].code == 'inexact_placement'


def test_remap_layout_keeps_template_reference_and_maps_local_coordinates(graph: InstanceGraph) -> None:

    original = section(graph, [graph.dimensions[0].categories[0].id])
    inherited = uuid4()
    spec = DataEntrySpec(sections=[original], amendments=[DataEntrySectionAmendmentSpec(section_id=inherited)])
    new_section, new_node, new_dim, new_cat = (uuid4() for _ in range(4))
    copied = remap_data_entry(
        spec,
        {
            original.id: new_section,
            graph.nodes[0].id: new_node,
            graph.dimensions[0].id: new_dim,
            graph.dimensions[0].categories[0].id: new_cat,
        },
    )
    assert copied.sections[0].id == new_section
    assert isinstance(copied.sections[0].tables[0], DataEntryPlacementSpec)
    assert copied.sections[0].tables[0].node_id == new_node
    assert copied.sections[0].tables[0].slices[0].categories == {new_dim: [new_cat]}
    assert copied.amendments[0].section_id == inherited
    assert original.id != new_section


class PreservingEntryNode(Node):
    @classmethod
    def data_entry_dependencies(cls, meta: NodeMeta, output_id: UUID) -> tuple[UUID, ...]:  # noqa: ARG003
        return tuple(port.id for port in meta.spec.input_ports)


def anchored_graph(graph: InstanceGraph) -> tuple[InstanceGraph, UUID, UUID]:
    source = graph.nodes[0]
    source_output, target_id, target_input, target_output = (uuid4() for _ in range(4))
    data = graph.model_dump(mode='python')
    data['nodes'][0]['node_class_path'] = 'nodes.tests.test_data_entry.PreservingEntryNode'
    data['nodes'][0]['spec']['output_ports'] = [
        OutputPortDef(id=source_output, unit=unit_registry.parse_units('kWh'), dimensions=['sector'])
    ]
    data['nodes'] = [
        *data['nodes'],
        NodeMeta(
            id=target_id,
            identifier='consumer',
            node_class_path='nodes.tests.test_data_entry.PreservingEntryNode',
            spec=NodeSpec(
                input_ports=[InputPortDef(id=target_input, unit=unit_registry.parse_units('kWh'))],
                output_ports=[OutputPortDef(id=target_output, unit=unit_registry.parse_units('kWh'), dimensions=['sector'])],
            ),
        ).model_dump(),
    ]
    data['bindings'] = [
        *data['bindings'],
        EdgeBindingDef(
            id=uuid4(),
            port_ref=NodePortRef(node_uuid=target_id, node_id='consumer', port_id=target_input),
            from_ref=NodePortRef(node_uuid=source.id, node_id='input', port_id=source_output),
        ).model_dump(),
    ]
    with set_i18n_context('en', []):
        return InstanceGraph.model_validate(data), source_output, target_output


def test_sliced_anchor_stops_overlap_and_continues_remainder(graph: InstanceGraph) -> None:
    graph, source_output, target_output = anchored_graph(graph)
    dim = graph.dimensions[0]
    with set_i18n_context('en', []):
        first = DataEntrySectionSpec(
            id=uuid4(),
            name='First',
            tables=[
                DataEntryAutodiscoverSpec(
                    id=uuid4(),
                    anchors=[
                        DataEntryAnchorSpec(
                            node_id=graph.nodes[0].id,
                            output_port_id=source_output,
                            slices=[DataEntrySliceSpec(categories={dim.id: [dim.categories[0].id]})],
                        )
                    ],
                )
            ],
        )
        second = DataEntrySectionSpec(
            id=uuid4(),
            name='Second',
            tables=[
                DataEntryAutodiscoverSpec(
                    id=uuid4(),
                    anchors=[
                        DataEntryAnchorSpec(
                            node_id=graph.nodes[1].id,
                            output_port_id=target_output,
                        )
                    ],
                )
            ],
        )
    graph.spec.data_entry = DataEntrySpec(sections=[first, second])
    layout = graph.data_entry
    assert len(layout.sections) == 2
    assert dict(layout.sections[0].selections[0].rectangles[0])[dim.id] == frozenset([dim.categories[0].id])
    assert dict(layout.sections[1].selections[0].rectangles[0])[dim.id] == frozenset([dim.categories[1].id])


def test_exact_claim_beats_earlier_approximate_claim(graph: InstanceGraph) -> None:
    graph, _source_output, target_output = anchored_graph(graph)
    with set_i18n_context('en', []):
        early = DataEntrySectionSpec(
            id=uuid4(),
            name='Approximate',
            tables=[
                DataEntryAutodiscoverSpec(
                    id=uuid4(),
                    anchors=[
                        DataEntryAnchorSpec(
                            node_id=graph.nodes[1].id,
                            output_port_id=target_output,
                        )
                    ],
                )
            ],
        )
        later = DataEntrySectionSpec(
            id=uuid4(),
            name='Exact',
            tables=[
                DataEntryAutodiscoverSpec(
                    id=uuid4(),
                    anchors=[
                        DataEntryAnchorSpec(
                            node_id=graph.nodes[1].id,
                            output_port_id=target_output,
                        )
                    ],
                )
            ],
        )
    graph.spec.data_entry = DataEntrySpec(sections=[early, later])
    resolver = EntryResolver(graph)
    # Exercise partition independently: two paths reached the same input with
    # different precision. The narrower exact claim owns the overlap.
    resolver.resolve()
    resolver.claims = [(3 if section == early.id else 2, section, claim) for _, section, claim in resolver.claims]
    layout = resolver.partition()
    assert not layout.sections[0].selections
    assert layout.sections[1].selections


def test_absent_layout_preserves_legacy_snapshot_hash_input() -> None:
    with set_i18n_context('en', []):
        spec = InstanceModelSpec()
        assert 'data_entry' not in spec.model_dump(mode='json')
        spec._is_composed = True
        assert 'data_entry' not in spec.model_dump(mode='json')


def test_a_copy_gives_a_layout_new_identities_that_follow_its_node() -> None:
    source = InstanceConfigFactory.create(name='Source', config_source='database', spec=InstanceModelSpec())
    node = NodeConfigFactory.create(instance=source)
    assert node.spec is not None
    with set_i18n_context(source.primary_language, source.other_languages):
        definition = DataEntrySectionSpec(
            id=uuid4(),
            name='Input',
            tables=[
                DataEntryAutodiscoverSpec(
                    id=uuid4(),
                    anchors=[
                        DataEntryAnchorSpec(
                            node_id=node.uuid,
                            output_port_id=node.spec.output_ports[0].id,
                        )
                    ],
                )
            ],
        )
    assert source.spec is not None
    source.spec.data_entry = DataEntrySpec(sections=[definition])
    source.save(update_fields=['spec'])
    exported = export_instance(source)
    target = InstanceConfigFactory.create(name='Target', config_source='database', spec=InstanceModelSpec())
    import_instance_copy(target, exported)
    restored = InstanceConfig.objects.get(pk=target.pk).ensure_spec().data_entry
    assert isinstance(restored, DataEntrySpec)
    assert restored.sections[0].id != definition.id
    assert isinstance(restored.sections[0].tables[0], DataEntryAutodiscoverSpec)
    assert isinstance(definition.tables[0], DataEntryAutodiscoverSpec)
    assert restored.sections[0].tables[0].anchors[0].node_id == target.nodes.get().uuid
    assert restored.sections[0].tables[0].anchors[0].node_id != node.uuid
    copied_node = target.nodes.get_queryset().with_spec().get()
    assert copied_node.spec is not None
    assert restored.sections[0].tables[0].anchors[0].output_port_id == copied_node.spec.output_ports[0].id
    assert restored.sections[0].tables[0].anchors[0].output_port_id != definition.tables[0].anchors[0].output_port_id


def test_rekeying_gives_a_layout_new_identities_that_follow_its_node() -> None:
    source = InstanceConfigFactory.create(name='Source', config_source='database', spec=InstanceModelSpec())
    node = NodeConfigFactory.create(instance=source)
    assert node.spec is not None
    port_id = node.spec.output_ports[0].id
    with set_i18n_context(source.primary_language, source.other_languages):
        definition = DataEntrySectionSpec(
            id=uuid4(),
            name='Input',
            tables=[
                DataEntryAutodiscoverSpec(id=uuid4(), anchors=[DataEntryAnchorSpec(node_id=node.uuid, output_port_id=port_id)])
            ],
        )
    assert source.spec is not None
    source.spec.data_entry = DataEntrySpec(sections=[definition])
    source.save(update_fields=['spec'])

    copy, rekeying = export_instance(source).rekeyed()

    entry = copy.instance.spec.data_entry
    assert isinstance(entry, DataEntrySpec)
    (section,) = entry.sections
    (table,) = section.tables
    assert isinstance(table, DataEntryAutodiscoverSpec)
    assert section.id == rekeying.mapping[definition.id]
    assert table.id == rekeying.mapping[definition.tables[0].id]
    assert (table.anchors[0].node_id, table.anchors[0].output_port_id) == (rekeying.mapping[node.uuid], rekeying.mapping[port_id])


def test_discovery_expands_in_place_and_manual_table_reserves_cells(graph: InstanceGraph) -> None:
    graph, _, output = anchored_graph(graph)
    dim = graph.dimensions[0]
    with set_i18n_context('en', []):
        discovery = DataEntryAutodiscoverSpec(
            id=uuid4(), anchors=[DataEntryAnchorSpec(node_id=graph.nodes[1].id, output_port_id=output)]
        )
        manual = DataEntryDatasetSpec(
            id=uuid4(),
            dataset_id=graph.datasets[0].id,
            slices=[DataEntrySliceSpec(categories={dim.id: [dim.categories[0].id]})],
        )
        spec = DataEntrySectionSpec(id=uuid4(), name='Ordered', tables=[discovery, manual])
    graph.spec.data_entry = DataEntrySpec(sections=[spec])
    selections = EntryResolver(graph).resolve().sections[0].selections
    assert [item.table_id for item in selections] == [discovery.id, manual.id]
    assert dict(selections[0].rectangles[0])[dim.id] == frozenset([dim.categories[1].id])
    assert selections[1].placement == 'instance_explicit'
    spec.tables.reverse()
    reordered = EntryResolver(graph).resolve().sections[0].selections
    assert list(reversed(reordered)) == list(selections)


def test_direct_dataset_does_not_require_a_binding(graph: InstanceGraph) -> None:
    data = graph.model_dump(mode='python')
    data['bindings'] = []
    with set_i18n_context('en', []):
        unbound = InstanceGraph.model_validate(data)
        manual = DataEntryDatasetSpec(id=uuid4(), dataset_id=graph.datasets[0].id)
        unbound.spec.data_entry = DataEntrySpec(sections=[DataEntrySectionSpec(id=uuid4(), name='Manual', tables=[manual])])
    selection = unbound.data_entry.sections[0].selections[0]
    assert selection.table_id == manual.id
    assert selection.dataset_id == manual.dataset_id
    assert selection.port_id is None


def test_overlapping_explicit_entries_report_a_problem(graph: InstanceGraph) -> None:
    authored = section(graph)
    authored.tables.append(DataEntryDatasetSpec(id=uuid4(), dataset_id=graph.datasets[0].id))
    graph.spec.data_entry = DataEntrySpec(sections=[authored])
    assert any(problem.code == 'overlapping_tables' for problem in graph.data_entry.problems)


def test_amendment_replaces_table_by_uuid_and_appends_new_entries(graph: InstanceGraph) -> None:
    inherited = section(graph)
    replacement = DataEntryDatasetSpec(id=inherited.tables[0].id, dataset_id=graph.datasets[0].id)
    stale = DataEntryDatasetSpec(id=uuid4(), dataset_id=uuid4())
    graph.spec.data_entry = ComposedDataEntrySpec(
        template=DataEntrySpec(sections=[inherited]),
        local=DataEntrySpec(amendments=[DataEntrySectionAmendmentSpec(section_id=inherited.id, tables=[replacement, stale])]),
    )
    resolver = EntryResolver(graph)
    layout = resolver.resolve()
    assert [table.id for table in resolver.sections[inherited.id].spec.tables] == [replacement.id, stale.id]
    assert layout.sections[0].selections[0].placement == 'instance_explicit'
    assert layout.sections[0].selections[0].port_id is None
    assert layout.sections[0].problems[0].dataset_id == stale.dataset_id
    assert isinstance(inherited.tables[0], DataEntryPlacementSpec)


def test_dataset_and_table_identity_remapping(graph: InstanceGraph) -> None:
    original = section(graph)
    table = DataEntryDatasetSpec(id=uuid4(), dataset_id=graph.datasets[0].id, metric_ids=[graph.datasets[0].metrics[0].id])
    original.tables = [table]
    new_table, new_dataset, new_metric = uuid4(), uuid4(), uuid4()
    remapped = remap_data_entry(
        DataEntrySpec(sections=[original]),
        {
            table.id: new_table,
            table.dataset_id: new_dataset,
            graph.datasets[0].metrics[0].id: new_metric,
        },
    )
    copied = remapped.sections[0].tables[0]
    assert isinstance(copied, DataEntryDatasetSpec)
    assert (copied.id, copied.dataset_id, copied.metric_ids) == (new_table, new_dataset, [new_metric])
    assert table.dataset_id == graph.datasets[0].id


def test_yaml_dataset_identifier_resolves_to_catalog_uuid(graph: InstanceGraph) -> None:
    dataset = graph.datasets[0]
    with set_i18n_context('en', []):
        meta = DatasetMeta(
            id=dataset.id,
            identifier='city/energy',
            schema_id=dataset.schema_id,
            metrics=(DatasetMetricMeta(id=dataset.metrics[0].id, identifier='energy'),),
        )
        config = YAMLDataEntry(
            sections=[
                YAMLSection(
                    id='energy',
                    name='Energy',
                    tables=[YAMLDataset(id='manual', kind='dataset', dataset='city/energy', metrics=['energy'])],
                )
            ]
        )
    namespace = uuid4()
    first = resolve_yaml_data_entry(config, namespace, [], [], [meta])
    second = resolve_yaml_data_entry(config, namespace, [], [], [meta])
    assert first == second
    table = first.sections[0].tables[0]
    assert isinstance(table, DataEntryDatasetSpec)
    assert table.dataset_id == dataset.id
    assert table.metric_ids == [dataset.metrics[0].id]
    assert 'city/energy' not in first.model_dump_json()
    with pytest.raises(ValueError, match='Unknown data-entry dataset'):
        resolve_yaml_data_entry(config, namespace, [], [], [])


def test_duplicate_table_identity_is_rejected(graph: InstanceGraph) -> None:
    authored = section(graph)
    authored.tables.append(authored.tables[0])
    with pytest.raises(ValueError, match='table UUIDs must be unique'):
        DataEntrySpec(sections=[authored])


def test_discovery_keeps_incompatible_datasets_separate(graph: InstanceGraph) -> None:
    graph, _, output = anchored_graph(graph)
    data = graph.model_dump(mode='python')
    port = uuid4()
    other_dataset, other_metric = uuid4(), uuid4()
    data['datasets'] = list(data['datasets'])
    data['bindings'] = list(data['bindings'])
    data['nodes'][0]['spec']['input_ports'].append(
        InputPortDef(id=port, binding_owner='instance', unit=unit_registry.parse_units('cap'))
    )
    data['datasets'].append(
        DatasetMeta(
            id=other_dataset,
            schema_id=uuid4(),
            metrics=(DatasetMetricMeta(id=other_metric, unit='cap', quantity='population'),),
        )
    )
    data['bindings'].append(
        DatasetBindingDef(
            id=uuid4(),
            port_ref=NodePortRef(node_uuid=graph.nodes[0].id, node_id='input', port_id=port),
            dataset_uuid=other_dataset,
            metric_uuid=other_metric,
        )
    )
    with set_i18n_context('en', []):
        graph = InstanceGraph.model_validate(data)
        discovery = DataEntryAutodiscoverSpec(
            id=uuid4(), anchors=[DataEntryAnchorSpec(node_id=graph.nodes[1].id, output_port_id=output)]
        )
        graph.spec.data_entry = DataEntrySpec(sections=[DataEntrySectionSpec(id=uuid4(), name='Mixed', tables=[discovery])])
    selections = graph.data_entry.sections[0].selections
    assert len(selections) == 2
    assert [selection.dataset_id for selection in selections] == sorted([other_dataset, graph.datasets[0].id])
    assert next(selection for selection in selections if selection.dataset_id == other_dataset).rectangles == ((),)
    assert not graph.data_entry.problems


def test_module_replacements_apply_to_direct_table_sources() -> None:
    included = {
        'data_entry': {
            'namespace': str(uuid4()),
            'sections': [
                {
                    'id': 'inputs',
                    'name': 'Inputs',
                    'tables': [{'id': 'energy', 'kind': 'dataset', 'dataset': 'default/energy'}],
                }
            ],
        }
    }
    merged: dict[str, object] = {}
    InstanceYAMLConfig._merge_data_entry(merged, included, [{'from': 'default/energy', 'to': 'city/energy'}])
    with set_i18n_context('en', []):
        parsed = YAMLDataEntry.model_validate(merged['data_entry'])
    table = parsed.sections[0].tables[0]
    assert isinstance(table, YAMLDataset)
    assert table.dataset == 'city/energy'


def test_composed_layout_rejects_table_identity_shared_between_sections(graph: InstanceGraph) -> None:
    inherited = section(graph)
    local = section(graph)
    local.tables[0].id = inherited.tables[0].id
    graph.spec.data_entry = ComposedDataEntrySpec(
        template=DataEntrySpec(sections=[inherited]),
        local=DataEntrySpec(sections=[local]),
    )
    assert any(problem.code == 'duplicate_table' for problem in graph.data_entry.problems)
    assert not graph.data_entry.sections[1].selections


def test_unbound_discovered_inputs_are_not_definition_errors(graph: InstanceGraph) -> None:
    graph, _, output = anchored_graph(graph)
    data = graph.model_dump(mode='python')
    data['bindings'] = [binding for binding in data['bindings'] if binding['kind'] != 'dataset']
    with set_i18n_context('en', []):
        graph = InstanceGraph.model_validate(data)
        graph.spec.data_entry = DataEntrySpec(
            sections=[
                DataEntrySectionSpec(
                    id=uuid4(),
                    name='Optional',
                    tables=[
                        DataEntryAutodiscoverSpec(
                            id=uuid4(), anchors=[DataEntryAnchorSpec(node_id=graph.nodes[1].id, output_port_id=output)]
                        ),
                    ],
                )
            ]
        )
    assert any(problem.code == 'unbound_input' for problem in graph.data_entry.problems)
    assert not data_entry_definition_problems(graph)
    # An authored manual table that cannot resolve its source is a definition error.
    graph.spec.data_entry = DataEntrySpec(sections=[section(graph)])
    graph.__dict__.pop('data_entry', None)
    assert data_entry_definition_problems(graph)[0].code == 'unbound_input'


@pytest.mark.parametrize('is_template', [False, True])
@pytest.mark.parametrize('direct', [False, True])
def test_placeholder_declarations_are_allowed_only_on_templates(graph: InstanceGraph, is_template: bool, direct: bool) -> None:
    data = graph.model_dump(mode='python')
    data['metadata']['is_template'] = is_template
    data['datasets'][0]['is_external_placeholder'] = True
    with set_i18n_context('en', []):
        graph = InstanceGraph.model_validate(data)
        declared = section(graph)
        if direct:
            declared.tables = [DataEntryDatasetSpec(id=uuid4(), dataset_id=graph.datasets[0].id)]
        graph.spec.data_entry = DataEntrySpec(sections=[declared])
    assert bool(graph.data_entry.sections[0].selections) == is_template
    assert bool(data_entry_definition_problems(graph)) != is_template


def test_template_placeholder_still_validates_slice_dimensions(graph: InstanceGraph) -> None:
    data = graph.model_dump(mode='python')
    data['metadata']['is_template'] = True
    data['datasets'][0]['is_external_placeholder'] = True
    data['datasets'][0]['declared_dimension_ids'] = []
    with set_i18n_context('en', []):
        graph = InstanceGraph.model_validate(data)
    graph.spec.data_entry = DataEntrySpec(sections=[section(graph, [graph.dimensions[0].categories[0].id])])
    assert data_entry_definition_problems(graph)[0].code == 'invalid_slice'


def test_template_placeholder_still_validates_selected_metrics(graph: InstanceGraph) -> None:
    data = graph.model_dump(mode='python')
    data['metadata']['is_template'] = True
    data['datasets'][0]['is_external_placeholder'] = True
    with set_i18n_context('en', []):
        graph = InstanceGraph.model_validate(data)
        declared = section(graph)
        declared.tables = [DataEntryDatasetSpec(id=uuid4(), dataset_id=graph.datasets[0].id, metric_ids=[uuid4()])]
    graph.spec.data_entry = DataEntrySpec(sections=[declared])
    assert data_entry_definition_problems(graph)[0].code == 'invalid_metric'
