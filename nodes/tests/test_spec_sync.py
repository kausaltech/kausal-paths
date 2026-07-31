"""Tests for the parse-only sync write half."""

from __future__ import annotations

import pytest

from kausal_common.i18n.pydantic import TranslatedString

from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import NodeSpec
from nodes.instance_serialization import InstanceSnapshot, NodeSnapshot
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
    spec = NodeSpec(uuid=uuid, identifier=identifier)
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
