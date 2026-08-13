"""Tests for the parse-only sync write half."""

from __future__ import annotations

import uuid

import pytest

from kausal_common.i18n.pydantic import TranslatedString

from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import NodeSpec
from nodes.instance_serialization import InstanceSnapshot, NodeSnapshot, reconcile_snapshot_node_metadata
from nodes.models import NodeConfig
from nodes.spec_sync import _apply_metadata_columns, _upsert_node_configs
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory

pytestmark = pytest.mark.django_db


@pytest.fixture
def db_instance() -> object:
    instance = InstanceFactory.create()
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        spec=InstanceModelSpec(years=YearsSpec(target=2030)),
    )


def _snapshot_with_node(ic, *, uuid, identifier: str) -> InstanceSnapshot:
    spec = NodeSpec()
    return InstanceSnapshot(
        spec=ic.spec,
        nodes=[NodeSnapshot(uuid=uuid, identifier=identifier, spec=spec)],
    )


def test_metadata_sync_preserves_db_authored_name_and_owner(db_instance):
    db_instance.name = 'Database name'
    db_instance.owner = 'Database owner'
    db_instance.i18n = {
        'name_fi': 'Tietokannan nimi',
        'owner_fi': 'Tietokannan omistaja',
    }
    snapshot = InstanceSnapshot(
        metadata=InstanceMetadata(
            name=TranslatedString(en='YAML name', fi='YAML-nimi'),
            owner=TranslatedString(en='YAML owner', fi='YAML-omistaja'),
            primary_language='en',
            other_languages=['fi'],
        ),
        spec=db_instance.spec,
    )

    _apply_metadata_columns(db_instance, snapshot)

    assert db_instance.name == 'Database name'
    assert db_instance.owner == 'Database owner'
    assert db_instance.i18n == {
        'name_fi': 'Tietokannan nimi',
        'owner_fi': 'Tietokannan omistaja',
    }
    assert db_instance.primary_language == 'en'
    assert db_instance.other_languages == ['fi']


def test_metadata_sync_seeds_blank_name_and_owner(db_instance):
    db_instance.name = ''
    db_instance.owner = ''
    db_instance.i18n = {}
    snapshot = InstanceSnapshot(
        metadata=InstanceMetadata(
            name=TranslatedString(en='YAML name', fi='YAML-nimi'),
            owner=TranslatedString(en='YAML owner', fi='YAML-omistaja'),
            primary_language='en',
            other_languages=['en', 'fi'],
        ),
        spec=db_instance.spec,
    )

    _apply_metadata_columns(db_instance, snapshot)

    assert db_instance.name == 'YAML name'
    assert db_instance.owner == 'YAML owner'
    assert db_instance.i18n == {
        'name_fi': 'YAML-nimi',
        'owner_fi': 'YAML-omistaja',
    }
    assert db_instance.primary_language == 'en'
    assert db_instance.other_languages == ['fi']


def test_uuid_matched_rename_does_not_mark_row_stale(db_instance):
    """
    A uuid-matched node under a new identifier must not stale the row it updated.

    The row keeps its old identifier (fields are only set when unset), so
    identifier-based stale detection would have flagged the very row the
    upsert touched.
    """
    row = NodeConfigFactory.create(instance=db_instance, identifier='old_name')

    node_configs = _upsert_node_configs(db_instance, _snapshot_with_node(db_instance, uuid=row.uuid, identifier='new_name'))

    assert node_configs[row.uuid].pk == row.pk
    row.refresh_from_db()
    assert row.is_stale is False


def test_untouched_rows_still_go_stale(db_instance):
    """Rows absent from the snapshot are marked stale (and auto-created ones removed)."""
    kept = NodeConfigFactory.create(instance=db_instance, identifier='kept')
    NodeConfigFactory.create(instance=db_instance, identifier='dropped')

    _upsert_node_configs(db_instance, _snapshot_with_node(db_instance, uuid=kept.uuid, identifier='kept'))

    kept.refresh_from_db()
    assert kept.is_stale is False
    dropped_qs = NodeConfig.objects.filter(instance=db_instance, identifier='dropped')
    # Auto-created rows (no pages, no creator) are deleted outright; either
    # way the row must not survive as active.
    assert not dropped_qs.filter(is_stale=False).exists()


def test_new_node_preserves_negative_display_order(db_instance):
    node_uuid = uuid.uuid4()
    snapshot = InstanceSnapshot(
        spec=db_instance.spec,
        nodes=[NodeSnapshot(uuid=node_uuid, identifier='first', order=-1, spec=NodeSpec())],
    )

    _upsert_node_configs(db_instance, snapshot)

    assert NodeConfig.objects.get(instance=db_instance, uuid=node_uuid).order == -1


def test_legacy_row_without_spec_is_bootstrapped_from_snapshot(db_instance):
    row = NodeConfigFactory.create(
        instance=db_instance,
        name=None,
        color='',
        order=None,
        is_visible=True,
        spec=None,
        i18n=None,
    )
    snapshot = InstanceSnapshot(
        metadata=InstanceMetadata(primary_language='en', other_languages=['fi']),
        spec=db_instance.spec,
        nodes=[
            NodeSnapshot(
                uuid=row.uuid,
                identifier=row.identifier,
                name=TranslatedString(en='YAML name', fi='YAML-nimi'),
                short_description=TranslatedString(en='Description', fi='Kuvaus'),
                color='#123456',
                order=-1,
                is_visible=False,
                spec=NodeSpec(),
            )
        ],
    )

    _upsert_node_configs(db_instance, snapshot)

    row = NodeConfig.objects.with_spec().get(pk=row.pk)
    assert row.name == 'YAML name'
    assert row.short_description == '<p>Description</p>\n'
    assert row.i18n == {'name_fi': 'YAML-nimi', 'short_description_fi': '<p>Kuvaus</p>\n'}
    assert row.color == '#123456'
    assert row.order == -1
    assert row.is_visible is False
    assert row.spec == NodeSpec()


def test_sync_reconciles_authoritative_node_metadata_before_upsert(db_instance):
    row = NodeConfigFactory.create(
        instance=db_instance,
        name='Database name',
        short_name='DB short name',
        color='',
        order=None,
        is_visible=False,
        spec=None,
        i18n={'name_fi': 'Tietokannan nimi'},
    )
    source = InstanceSnapshot(
        metadata=InstanceMetadata(primary_language='en', other_languages=['fi']),
        spec=db_instance.spec,
        nodes=[
            NodeSnapshot(
                uuid=row.uuid,
                identifier=row.identifier,
                name=TranslatedString(en='YAML name', fi='YAML-nimi'),
                short_name=TranslatedString(en='YAML short name', fi='YAML-lyhytnimi'),
                color='#123456',
                order=-1,
                is_visible=True,
                spec=NodeSpec(),
            )
        ],
    )

    snapshot = reconcile_snapshot_node_metadata(source, [row])
    reconciled = snapshot.nodes[0]

    assert reconciled.name is not None
    assert reconciled.name.i18n == {'en': 'Database name', 'fi': 'Tietokannan nimi'}
    assert reconciled.short_name is not None
    assert reconciled.short_name.i18n == {'en': 'DB short name', 'fi': 'YAML-lyhytnimi'}
    assert reconciled.color == '#123456'
    assert reconciled.order == -1
    assert reconciled.is_visible is False

    _upsert_node_configs(db_instance, snapshot, [row])

    row = NodeConfig.objects.with_spec().get(pk=row.pk)
    assert row.name == 'Database name'
    assert row.short_name == 'DB short name'
    assert row.i18n == {'name_fi': 'Tietokannan nimi', 'short_name_fi': 'YAML-lyhytnimi'}
    assert row.color == '#123456'
    assert row.order == -1
    assert row.is_visible is False
    assert row.spec == NodeSpec()


def test_yaml_runtime_nodes_use_reconciled_metadata_snapshots(instance_config, instance, node):
    node.short_name = TranslatedString(en='YAML short name')
    node.is_visible = True
    row = NodeConfigFactory.create(
        instance=instance_config,
        identifier=node.id,
        name='Database name',
        short_name='DB short name',
        color='',
        order=None,
        is_visible=False,
        spec=None,
    )

    instance_config.update_instance_from_configs(instance, node_refs=True)

    assert node.source_snapshot is not None
    assert node.source_snapshot.uuid == row.uuid
    assert str(node.source_snapshot.name) == 'Database name'
    assert str(node.source_snapshot.short_name) == 'DB short name'
    assert node.source_snapshot.color == 'pink'
    assert node.source_snapshot.is_visible is False
    assert instance.source_nodes_by_uuid[row.uuid] is node
