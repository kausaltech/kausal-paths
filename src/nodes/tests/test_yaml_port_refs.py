from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from nodes.models import NodeConfig, NodeInputPortBinding
from nodes.spec_sync import sync_parsed_instance_to_db
from nodes.tests.factories import InstanceConfigFactory
from nodes.yaml_port_refs import AmbiguousYamlPortReferenceError, YamlPortReferenceCatalog

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.django_db


def test_catalog_preserves_exact_structural_port_references():
    node_id = uuid4()
    source_id = uuid4()
    source_port_id = uuid4()
    output_port_id = uuid4()
    edge_port_id = uuid4()
    dataset_port_id = uuid4()
    fallback = uuid4()
    catalog = YamlPortReferenceCatalog(
        output_ports={(source_id, 'emissions'): {output_port_id}},
        input_roles={(node_id, 'additive'): {edge_port_id}},
        edge_ports={(node_id, source_id, source_port_id): {edge_port_id}},
        dataset_ports={(node_id, 'inventory', 2, 'energy'): {dataset_port_id}},
        dataset_groups={(node_id, 'inventory', 2): {dataset_port_id}},
    )

    assert catalog.output_port_id(source_id, ('emissions',), fallback) == output_port_id
    assert catalog.input_role_id(node_id, 'additive', fallback) == edge_port_id
    assert catalog.edge_port_id(node_id, source_id, source_port_id, fallback) == edge_port_id
    assert catalog.dataset_port_id(node_id, 'inventory', 2, 'energy', fallback) == dataset_port_id


def test_catalog_uses_fallback_only_for_a_genuinely_new_structure():
    catalog = YamlPortReferenceCatalog()
    fallback = uuid4()

    assert catalog.output_port_id(uuid4(), ('new_metric',), fallback) == fallback


def test_catalog_rejects_ambiguous_anonymous_dataset_ports():
    node_id = uuid4()
    port_ids = {uuid4(), uuid4()}
    catalog = YamlPortReferenceCatalog(dataset_groups={(node_id, 'inventory', 0): port_ids})

    with pytest.raises(AmbiguousYamlPortReferenceError, match='Ambiguous persisted ports'):
        catalog.dataset_port_id(
            node_id,
            'inventory',
            0,
            'Value',
            uuid4(),
            allow_group_fallback=True,
            fail_on_ambiguous=True,
        )


def test_catalog_accepts_an_existing_deterministic_id_in_an_ambiguous_group():
    node_id = uuid4()
    existing = uuid4()
    catalog = YamlPortReferenceCatalog(dataset_groups={(node_id, 'inventory', 0): {existing, uuid4()}})

    assert (
        catalog.dataset_port_id(
            node_id,
            'inventory',
            0,
            'Value',
            existing,
            allow_group_fallback=True,
        )
        == existing
    )


def test_catalog_does_not_reuse_one_existing_port_for_new_sibling_columns():
    node_id = uuid4()
    existing = uuid4()
    fallback = uuid4()
    catalog = YamlPortReferenceCatalog(dataset_groups={(node_id, 'inventory', 0): {existing}})

    assert catalog.dataset_port_id(node_id, 'inventory', 0, 'new_column', fallback) == fallback


def test_role_fanout_reuses_an_unclassified_port_only_for_the_first_role() -> None:
    key = (uuid4(), uuid4(), uuid4())
    existing, fallback = uuid4(), uuid4()
    catalog = YamlPortReferenceCatalog(edge_ports={key: {existing}}, edge_ports_by_role={(*key, None): {existing}})

    assert catalog.edge_port_id(*key, fallback, role='removing', reuse_unclassified=True) == existing
    assert catalog.edge_port_id(*key, fallback, role='inserting') == fallback


def test_role_fanout_still_rejects_duplicate_ports_with_the_same_role() -> None:
    key = (uuid4(), uuid4(), uuid4())
    catalog = YamlPortReferenceCatalog(edge_ports_by_role={(*key, 'removing'): {uuid4(), uuid4()}})

    with pytest.raises(AmbiguousYamlPortReferenceError, match='removing'):
        catalog.edge_port_id(*key, uuid4(), role='removing', reuse_unclassified=True)


@pytest.mark.parametrize('reassign_port_ids', [False, True])
def test_sync_preserves_shared_rate_ports_and_bindings_on_repeat(tmp_path: Path, reassign_port_ids: bool) -> None:
    yaml_path = tmp_path / 'shared-rates.yaml'
    yaml_path.write_text("""
id: shared-rates
name: Shared rates
owner: Test
default_language: en
target_year: 2030
reference_year: 2020
minimum_historical_year: 2010
nodes:
- id: rate
  name: Rate
  type: simple.AdditiveNode
  quantity: fraction
  unit: '%/a'
- id: stock
  name: Stock
  type: costs.DilutionNode
  quantity: emission_factor
  unit: g/vkm
  input_nodes:
  - id: rate
    tags: [removing, inserting]
""")
    instance = InstanceConfigFactory.create(identifier='shared-rates', name='Shared rates')
    sync_parsed_instance_to_db('shared-rates', yaml_path=yaml_path)
    node = NodeConfig.objects.get(instance=instance, identifier='stock')
    spec = node.spec
    assert spec is not None
    assert [port.role for port in spec.input_ports] == ['removing', 'inserting']

    if reassign_port_ids:
        # Persisted identity wins even when it differs from the parser's generated UUIDs.
        for port in spec.input_ports:
            new_id = uuid4()
            NodeInputPortBinding.objects.filter(node=node, port_id=port.id).update(port_id=new_id)
            port.id = new_id
        NodeConfig.objects.filter(pk=node.pk).update(spec=spec)

    expected_ports = {port.role: port.id for port in spec.input_ports}
    expected_bindings = set(NodeInputPortBinding.objects.filter(node=node).values_list('pk', 'uuid', 'port_id'))
    assert len(expected_bindings) == 2

    sync_parsed_instance_to_db('shared-rates', yaml_path=yaml_path)

    node.refresh_from_db()
    assert node.spec is not None
    assert {port.role: port.id for port in node.spec.input_ports} == expected_ports
    assert set(NodeInputPortBinding.objects.filter(node=node).values_list('pk', 'uuid', 'port_id')) == expected_bindings
