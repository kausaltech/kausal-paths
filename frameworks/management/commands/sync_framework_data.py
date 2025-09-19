from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandParser
from django.db import transaction

from frameworks.models import Framework
from frameworks.sync_configs import FrameworkConfigDjangoAdapter, FrameworkConfigJSONAdapter
from frameworks.sync_frameworks import FrameworkDjangoAdapter, FrameworkJSONAdapter, FrameworkModel


class Command(BaseCommand):
    help = 'Synchronize or export Framework metadata'

    dry_run: bool = False
    yes: bool = False

    def add_arguments(self, parser: CommandParser):
        parser.add_argument(
            'action',
            type=str,
            choices=['import', 'export', 'import_configs', 'export_configs'],
            help='Action to perform',
        )
        parser.add_argument('file', type=Path, help='JSON file to read from or write to')
        parser.add_argument('--dry-run', '-N', action='store_true', help='Only show the changes that would be made')
        parser.add_argument('--framework', type=str, help='Framework identifier (required for export)')
        parser.add_argument('--yes', '-y', action='store_true', help='Do not prompt for anything')

    @transaction.atomic
    def handle(self, *args, **options):
        action = options['action']
        file_path = options['file']
        self.dry_run = options['dry_run']
        self.yes = options['yes']

        framework_identifier = options['framework']
        if action in ('export', 'export_configs', 'import_configs') and not framework_identifier:
            self.stderr.write(self.style.ERROR('Framework identifier is required for the subaction'))
            exit(1)

        if action == 'import':
            self.import_data(file_path)
        elif action == 'export':
            self.export_data(file_path, framework_identifier)
        elif action == 'export_configs':
            self.export_configs(file_path, framework_identifier)
        elif action == 'import_configs':
            self.import_configs(file_path, framework_identifier)

    def import_data(self, file_path: Path):
        js = FrameworkJSONAdapter(file_path)
        js.load()

        fw_objs = js.get_all(FrameworkModel)
        assert len(fw_objs) == 1
        fw = fw_objs[0]

        dj = FrameworkDjangoAdapter(interactive=True)
        if self.yes:
            dj.allow_related_deletion = True
        with dj.start():
            dj.load(framework_id=fw.identifier)
            dj.sync_from(js)
            if self.dry_run:
                self.stderr.write(self.style.WARNING('Dry run requested; no changes done.'))
                dj.rollback()

    def export_data(self, file_path: Path, framework_identifier: str):
        try:
            framework = Framework.objects.get(identifier=framework_identifier)
        except Framework.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Framework with identifier '{framework_identifier}' not found"))
            return

        dj = FrameworkDjangoAdapter(interactive=True)
        with dj.start():
            dj.load(framework_id=framework.identifier)

        if self.dry_run:
            self.stderr.write(self.style.WARNING('Dry run requested; no changes done.'))
            exit(0)

        dj.save_json(file_path)
        self.stdout.write(self.style.SUCCESS(f'Successfully exported framework data to {file_path}'))

    def export_configs(self, file_path: Path, framework_identifier: str):
        try:
            fw = Framework.objects.get(identifier=framework_identifier)
        except Framework.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Framework with identifier '{framework_identifier}' not found"))
            return

        dj = FrameworkConfigDjangoAdapter(fw.identifier)
        with dj.start():
            dj.load()

        if self.dry_run:
            self.stderr.write(self.style.WARNING('Dry run requested; no changes done.'))
            exit(0)

        dj.save_json(file_path)

        self.stdout.write(self.style.SUCCESS(f'Successfully exported framework data to {file_path}'))

    def import_configs(self, file_path: Path, framework_identifier: str):
        try:
            fw = Framework.objects.get(identifier=framework_identifier)
        except Framework.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Framework with identifier '{framework_identifier}' not found"))
            return

        js = FrameworkConfigJSONAdapter(fw.identifier, file_path)
        js.load()
        dj = FrameworkConfigDjangoAdapter(fw.identifier)
        if self.yes:
            dj.allow_related_deletion = True
        with dj.start():
            dj.load()
            dj.sync_from(js)
            if self.dry_run:
                self.stderr.write(self.style.WARNING('Dry run requested; no changes done.'))
                dj.rollback()
