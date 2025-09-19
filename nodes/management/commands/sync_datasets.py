from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandParser
from django.db import transaction

from kausal_common.datasets.sync import DatasetDjangoAdapter, DatasetJSONAdapter

from nodes.models import InstanceConfig


class Command(BaseCommand):
    help = 'Synchronize or export datasets'

    dry_run: bool = False
    yes: bool = False

    def add_arguments(self, parser: CommandParser):
        parser.add_argument(
            'action',
            type=str,
            choices=['import', 'export'],
            help='Action to perform',
        )
        parser.add_argument(
            'instance',
            type=str,
            help='Instance identifier',
        )
        parser.add_argument('file', type=Path, help='JSON file to read from or write to')
        parser.add_argument('--dry-run', '-N', action='store_true', help='Only show the changes that would be made')
        parser.add_argument('--yes', '-y', action='store_true', help='Do not prompt for anything')

    @transaction.atomic
    def handle(self, *args, **options):
        action = options['action']
        file_path = options['file']
        self.dry_run = options['dry_run']
        self.yes = options['yes']

        instance_identifier = options['instance']
        # if action in ('export', 'export_configs', 'import_configs') and not instance_identifier:
        #    self.stderr.write(self.style.ERROR('Instance identifier is required for the subaction'))
        #    exit(1)

        if action == 'import':
            self.import_data(file_path, instance_identifier)
        elif action == 'export':
            self.export_data(file_path, instance_identifier)

    def import_data(self, file_path: Path, instance_identifier: str):
        try:
            ic = InstanceConfig.objects.get(identifier=instance_identifier)
        except InstanceConfig.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Instance with identifier '{instance_identifier}' not found"))
            return

        js = DatasetJSONAdapter(file_path)
        js.load()

        dj = DatasetDjangoAdapter(scope=ic, interactive=True)
        if self.yes:
            dj.allow_related_deletion = True
        with dj.start():
            dj.load()
            dj.sync_from(js)
            if self.dry_run:
                self.stderr.write(self.style.WARNING('Dry run requested; no changes done.'))
                dj.rollback()

    def export_data(self, file_path: Path, instance_identifier: str):
        try:
            ic = InstanceConfig.objects.get(identifier=instance_identifier)
        except InstanceConfig.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Instance with identifier '{instance_identifier}' not found"))
            return

        # ct = ContentType.objects.get_for_model(ic)
        dj = DatasetDjangoAdapter(ic)
        with dj.start():
            dj.load()

        if self.dry_run:
            self.stderr.write(self.style.WARNING('Dry run requested; no changes done.'))
            exit(0)

        dj.save_json(file_path)
        self.stdout.write(self.style.SUCCESS(f'Successfully exported datasets to {file_path}'))

    def export_configs(self, file_path: Path, instance_identifier: str):
        try:
            ic = InstanceConfig.objects.get(identifier=instance_identifier)
        except InstanceConfig.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Instance with identifier '{instance_identifier}' not found"))
            return

        dj = DatasetDjangoAdapter(ic)
        with dj.start():
            dj.load()

        if self.dry_run:
            self.stderr.write(self.style.WARNING('Dry run requested; no changes done.'))
            exit(0)

        dj.save_json(file_path)

        self.stdout.write(self.style.SUCCESS(f'Successfully exported framework data to {file_path}'))
