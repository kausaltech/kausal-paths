from uuid import uuid4

import pytest

from nodes.yaml_port_refs import AmbiguousYamlPortReferenceError, YamlPortReferenceCatalog

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
