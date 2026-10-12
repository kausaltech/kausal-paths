import importlib
import json
from types import SimpleNamespace

from django.apps import apps
from django.db import connection

import pytest

from nodes.defs.instance_defs import InstanceModelSpec
from nodes.scenario import Scenario, ScenarioKind
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory

pytestmark = pytest.mark.django_db


def test_sparse_migration_upgrades_raw_formula_specs_without_dropping_later_fields() -> None:
    instance = InstanceConfigFactory.create(
        name='Legacy instance',
        config_source='database',
        spec=InstanceModelSpec(
            scenarios=[
                Scenario(id='default', name='Default', kind=ScenarioKind.DEFAULT, param_values={'formula.formula': 'old capture'})
            ]
        ),
    )
    node = NodeConfigFactory.create(instance=instance, identifier='formula')
    assert node.spec is not None
    raw = node.spec.model_dump(mode='json')
    raw['type_config'] = {'kind': 'simple', 'node_class': 'formula.FormulaNode'}
    raw['params'] = [{'local_id': 'formula', 'type': 'string', 'value': '1 + 2'}]
    raw['pipeline'] = None
    raw['extra']['other'] = {}
    with connection.cursor() as cursor:
        cursor.execute('UPDATE nodes_nodeconfig SET spec = %s WHERE id = %s', [json.dumps(raw), node.pk])

    migration = importlib.import_module('nodes.migrations.0082_sparse_template_specs')
    migration.migrate_specs(apps, SimpleNamespace(connection=connection))

    with connection.cursor() as cursor:
        cursor.execute('SELECT spec FROM nodes_nodeconfig WHERE id = %s', [node.pk])
        stored = cursor.fetchone()[0]
    migrated = json.loads(stored) if isinstance(stored, str) else stored
    assert migrated == {**raw, 'type_config': {'kind': 'formula', 'formula': '1 + 2'}, 'params': []}
    instance.refresh_from_db()
    assert instance.spec is not None
    assert instance.spec.scenarios[0].param_values == {}


@pytest.mark.parametrize('config_source', ['yaml', 'database'])
def test_sparse_migration_preserves_specs_when_a_scenario_changes_a_formula(config_source: str) -> None:
    instance = InstanceConfigFactory.create(
        name='Legacy instance',
        config_source=config_source,
        spec=InstanceModelSpec(scenarios=[Scenario(id='alternative', name='Alternative', param_values={'formula.formula': '3'})]),
    )
    node = NodeConfigFactory.create(instance=instance, identifier='formula')
    assert node.spec is not None
    raw = node.spec.model_dump(mode='json')
    raw['type_config'] = {'kind': 'simple', 'node_class': 'formula.FormulaNode'}
    raw['params'] = [{'local_id': 'formula', 'type': 'string', 'value': '1 + 2'}]
    raw['pipeline'] = None
    with connection.cursor() as cursor:
        cursor.execute('UPDATE nodes_nodeconfig SET spec = %s WHERE id = %s', [json.dumps(raw), node.pk])

    migration = importlib.import_module('nodes.migrations.0082_sparse_template_specs')
    schema_editor = SimpleNamespace(connection=connection)
    if config_source == 'database':
        with pytest.raises(ValueError, match=r'Scenario alternative changes formula formula\.formula'):
            migration.migrate_specs(apps, schema_editor)
    else:
        migration.migrate_specs(apps, schema_editor)

    with connection.cursor() as cursor:
        cursor.execute('SELECT spec FROM nodes_nodeconfig WHERE id = %s', [node.pk])
        stored = cursor.fetchone()[0]
    assert (json.loads(stored) if isinstance(stored, str) else stored) == raw
    instance.refresh_from_db()
    assert instance.spec is not None
    assert instance.spec.scenarios[0].param_values == {'formula.formula': '3'}
