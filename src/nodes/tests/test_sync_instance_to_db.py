from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import patch

from django.core.management import call_command

import pytest

from frameworks.tests.factories import FrameworkConfigFactory
from nodes.management.commands.sync_instance_to_db import Command
from nodes.tests.factories import InstanceConfigFactory

if TYPE_CHECKING:
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize('selection', ['--all', '--all-db'])
def test_bulk_sync_selects_instances(selection: str, instance_config: InstanceConfig) -> None:
    database_instance = InstanceConfigFactory.create(name='Database instance', config_source='database')
    yaml_instance = InstanceConfigFactory.create(name='YAML instance', config_source='yaml')
    framework_instance = InstanceConfigFactory.create(name='Framework instance', config_source='database')
    FrameworkConfigFactory.create(instance_config=framework_instance)

    with patch.object(Command, 'sync_one_instance') as sync:
        call_command('sync_instance_to_db', selection, '--dry-run', stdout=StringIO())

    expected = {database_instance.identifier}
    if selection == '--all':
        expected.add(yaml_instance.identifier)
        expected.add(instance_config.identifier)
    assert {call.args[0] for call in sync.call_args_list} == expected
    assert all(call.kwargs == {'dry_run': True} for call in sync.call_args_list)


def test_all_db_honors_skip() -> None:
    selected = InstanceConfigFactory.create(name='Selected instance', config_source='database')
    skipped = InstanceConfigFactory.create(name='Skipped instance', config_source='database')

    with patch.object(Command, 'sync_one_instance') as sync:
        call_command('sync_instance_to_db', '--all-db', '--skip', skipped.identifier, stdout=StringIO())

    sync.assert_called_once_with(selected.identifier, dry_run=False)
