"""Shared legacy rates retain their meaning across snapshot and stored-draft upgrades."""

from copy import deepcopy
from importlib import import_module
from uuid import uuid4

from django.apps import apps
from django.db import connection

import pytest

from nodes.defs.node_defs import NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef
from nodes.defs.transform_def import DropNullsOp
from nodes.instance_serialization import InstanceSnapshot, build_instance_snapshot
from nodes.legacy_port_roles import split_legacy_edge_roles
from nodes.models import NodeInputPortBinding
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize('authored_role', [None, 'removing'])
def test_upgrade_preserves_binding_identity_transformations_and_authored_roles(authored_role: str | None) -> None:
    instance = InstanceConfigFactory.create(name='Legacy rates', config_source='database')
    source = NodeConfigFactory.create(instance=instance, identifier='rate')
    assert source.spec is not None
    port = InputPortDef(id=uuid4(), identifier='legacy_rate', role=authored_role, unit=unit_registry.parse_units('1/a'))
    target = NodeConfigFactory.create(
        instance=instance,
        identifier='stock',
        spec=NodeSpec(type_config=SimpleConfig(node_class='nodes.costs.DilutionNode'), input_ports=[port]),
    )
    binding = NodeInputPortBinding.objects.create(
        instance=instance,
        node=target,
        port_id=port.id,
        source_node=source,
        source_port_id=source.spec.output_ports[0].id,
        tags=['removing', 'inserting'],
        transformations=[DropNullsOp()],
    )
    old = build_instance_snapshot(instance).model_dump(mode='json')
    old['schema_version'] = 18
    original = deepcopy(old)
    upgraded = InstanceSnapshot.from_serialized_data(old)
    assert old == original
    assert upgraded.snapshot_kind == original['snapshot_kind']
    node = next(node for node in upgraded.nodes if node.uuid == target.uuid)
    selected = [item for item in upgraded.bindings if item.node_id == target.uuid]
    assert node.spec is not None
    assert selected[0].uuid == binding.uuid
    assert selected[0].port_id == port.id
    if authored_role is not None:
        assert len(selected) == 1
        assert node.spec.input_ports[0].role == authored_role
    else:
        assert [item.role for item in node.spec.input_ports] == ['removing', 'inserting']
        assert len(selected) == 2
        assert selected[0].node_source == selected[1].node_source
        assert selected[0].transformations == selected[1].transformations == [DropNullsOp()]
        assert selected[0].tags == selected[1].tags == ['removing', 'inserting']
    again, bindings = split_legacy_edge_roles(node.spec, selected, ('removing', 'inserting'))
    assert again == node.spec
    assert bindings == selected

    migration = import_module('nodes.migrations.0090_split_legacy_edge_roles')
    migration.forwards(apps, connection.schema_editor())
    migrated = build_instance_snapshot(instance)
    actual = next(node for node in migrated.nodes if node.uuid == target.uuid)
    assert actual.spec == node.spec
    assert [item for item in migrated.bindings if item.node_id == target.uuid] == selected
    migration.forwards(apps, connection.schema_editor())
    assert build_instance_snapshot(instance).model_dump(mode='json') == migrated.model_dump(mode='json')
