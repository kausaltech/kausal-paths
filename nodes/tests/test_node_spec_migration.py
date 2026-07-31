import importlib
import json
from types import SimpleNamespace

from django.apps import apps
from django.db import connection

import pytest

from nodes.tests.factories import NodeConfigFactory

pytestmark = pytest.mark.django_db


def test_node_metadata_is_backfilled_and_removed_from_raw_spec():
    migration = importlib.import_module('nodes.migrations.0055_nodeconfig_metadata_and_computation_spec')
    nc = NodeConfigFactory.create(
        name='Column name',
        short_name=None,
        short_description=None,
        color='#ffffff',
        order=None,
        is_visible=True,
        i18n={'name_fi': 'Sarakkeen nimi'},
    )
    assert nc.spec is not None
    legacy_spec = {
        **nc.spec.model_dump(mode='json'),
        'uuid': str(nc.uuid),
        'identifier': nc.identifier,
        'name': {'en': 'Legacy name', 'fi': 'Vanha nimi'},
        'short_name': {'en': 'Short label', 'fi': 'Lyhyt nimi', 'es-US': 'Nombre corto'},
        'description': {'en': 'Runtime description', 'fi': 'Kuvaus', 'es-US': 'Descripción'},
        'color': '#123456',
        'order': 7,
        'is_visible': False,
        'kind': 'simple',
    }
    table = connection.ops.quote_name(nc._meta.db_table)
    with connection.cursor() as cursor:
        cursor.execute(f'UPDATE {table} SET spec = %s WHERE id = %s', [json.dumps(legacy_spec), nc.pk])  # noqa: S608

    schema_editor = SimpleNamespace(connection=connection, quote_name=connection.ops.quote_name)
    migration.split_node_metadata_from_spec(apps, schema_editor)

    nc.refresh_from_db()
    assert nc.name == 'Legacy name'
    assert nc.short_name == 'Short label'
    assert nc.short_description == '<p>Runtime description</p>\n'
    assert nc.color == '#123456'
    assert nc.order == 7
    assert nc.is_visible is False
    assert nc.i18n == {
        'name_fi': 'Vanha nimi',
        'short_name_fi': 'Lyhyt nimi',
        'short_name_es_us': 'Nombre corto',
        'short_description_fi': '<p>Kuvaus</p>\n',
        'short_description_es_us': '<p>Descripción</p>\n',
    }
    with connection.cursor() as cursor:
        cursor.execute(f'SELECT spec FROM {table} WHERE id = %s', [nc.pk])  # noqa: S608
        raw_spec = cursor.fetchone()[0]
    if isinstance(raw_spec, str):
        raw_spec = json.loads(raw_spec)
    assert not migration._SPEC_METADATA_FIELDS & raw_spec.keys()


def test_node_metadata_keeps_authored_order_and_rich_description():
    migration = importlib.import_module('nodes.migrations.0055_nodeconfig_metadata_and_computation_spec')
    nc = NodeConfigFactory.create(order=99, short_description='<p>Authored description</p>', i18n={})
    assert nc.spec is not None
    legacy_spec = {
        **nc.spec.model_dump(mode='json'),
        'order': 7,
        'description': {'en': 'Runtime description'},
    }
    table = connection.ops.quote_name(nc._meta.db_table)
    with connection.cursor() as cursor:
        cursor.execute(f'UPDATE {table} SET spec = %s WHERE id = %s', [json.dumps(legacy_spec), nc.pk])  # noqa: S608

    schema_editor = SimpleNamespace(connection=connection, quote_name=connection.ops.quote_name)
    migration.split_node_metadata_from_spec(apps, schema_editor)

    nc.refresh_from_db()
    assert nc.order == 99
    assert nc.short_description == '<p>Authored description</p>'


def test_legacy_i18n_locale_suffixes_are_normalized():
    migration = importlib.import_module('nodes.migrations.0056_nodeconfig_signed_order_and_i18n')
    nc = NodeConfigFactory.create(i18n={})
    table = connection.ops.quote_name(nc._meta.db_table)
    with connection.cursor() as cursor:
        cursor.execute(
            f'UPDATE {table} SET i18n = %s WHERE id = %s',  # noqa: S608
            [json.dumps({'name_es-US': 'Nombre', 'name_fi': 'Nimi'}), nc.pk],
        )

    schema_editor = SimpleNamespace(connection=connection, quote_name=connection.ops.quote_name)
    migration.normalize_node_i18n_keys(apps, schema_editor)

    nc.refresh_from_db()
    assert nc.i18n == {'name_es_us': 'Nombre', 'name_fi': 'Nimi'}
