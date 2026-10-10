"""Refresh editable defaults from each municipality's published template pin."""

from typing import TYPE_CHECKING, cast

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from frameworks.bisko.activation import _ensure_local_inputs
from frameworks.bisko.default_sources import POPULATION_DATASET
from frameworks.models import FrameworkConfig
from nodes.models import InstanceConfig
from nodes.template_graph import lock_template_for_publication, template_snapshot

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = 'Refresh provider defaults without overwriting municipal edits; dry run unless --apply is specified.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('--instance', action='append', default=[])
        parser.add_argument('--apply', action='store_true')

    def handle(self, *args: object, **options: object) -> None:
        configs = FrameworkConfig.objects.filter(framework__identifier='bisko').select_related('instance_config')
        requested = cast('list[str]', options['instance'])
        if requested:
            configs = configs.filter(instance_config__identifier__in=requested)
            found = set(configs.values_list('instance_config__identifier', flat=True))
            missing = set(requested) - found
            if missing:
                raise CommandError(f'Unknown BISKO instances: {sorted(missing)}')
        try:
            with transaction.atomic():
                for config in configs:
                    instance = config.instance_config
                    if instance.is_locked or instance.template_revision_id is None:
                        self.stdout.write(f'{instance.identifier}: skipped (locked or no published template pin)')
                        continue
                    lock_template_for_publication(instance.pk)
                    instance = InstanceConfig.objects.select_for_update().get(pk=instance.pk)
                    if POPULATION_DATASET not in {dataset.identifier for dataset in template_snapshot(instance).all_datasets()}:
                        self.stdout.write(f'{instance.identifier}: skipped (upgrade the template pin first)')
                        continue
                    _ensure_local_inputs(instance)
                    self.stdout.write(f'{instance.identifier}: refreshed defaults from its published template pin')
                if not options['apply']:
                    transaction.set_rollback(True)
        except ValueError as error:
            raise CommandError(str(error)) from error
        if not options['apply']:
            self.stdout.write('Dry run: database changes rolled back.')
