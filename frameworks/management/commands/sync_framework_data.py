from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandParser
from django.db import transaction

from frameworks.models import Framework
from frameworks.sync import FrameworkDjangoAdapter, FrameworkJSONAdapter, FrameworkModel


class Command(BaseCommand):
    help = 'Synchronize or export Framework metadata'

    dry_run: bool = False
    yes: bool = False

    def add_arguments(self, parser: CommandParser):
        parser.add_argument('action', type=str, choices=['import', 'export'], help='Action to perform')
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

        if action == 'import':
            self.import_data(file_path)
        elif action == 'export':
            framework_identifier = options['framework']
            if not framework_identifier:
                self.stderr.write(self.style.ERROR('Framework identifier is required for export'))
                return
            self.export_data(file_path, framework_identifier)

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
                self.stderr.write(self.style.WARNING("Dry run requested; no changes done."))
                dj.rollback()

    def export_data(self, file_path: Path, framework_identifier: str):
        try:
            framework = Framework.objects.get(identifier=framework_identifier)
        except Framework.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Framework with identifier '{framework_identifier}' not found"))
            return

        data = {
            'framework': framework.to_dict(),
            'sections': framework.export_sections(),
        }

        with file_path.open('w') as file:
            json.dump(data, file, indent=2)

        self.stdout.write(self.style.SUCCESS(f"Successfully exported framework data to {file_path}"))
