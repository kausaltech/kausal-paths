from __future__ import annotations

from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand
from django.db import transaction

from kausal_common.datasets.sync import DatasetDjangoAdapter

from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from django.core.management.base import CommandParser


class Command(BaseCommand):
    help = 'Copy datasets from one instance to another (in-DB copy via diffsync)'

    dry_run: bool = False
    yes: bool = False

    def add_arguments(self, parser: CommandParser):
        parser.add_argument('source', type=str, help='Source instance identifier')
        parser.add_argument('destination', type=str, help='Destination instance identifier')
        parser.add_argument('--dry-run', '-N', action='store_true', help='Only show the changes that would be made')
        parser.add_argument('--yes', '-y', action='store_true', help='Do not prompt for anything')
        parser.add_argument(
            '--datasets',
            '-d',
            type=str,
            action='append',
            metavar='DATASET_IDENTIFIER',
            help='Limit to specific dataset(s) by identifier (can be repeated)',
        )

    @transaction.atomic
    def handle(self, *_args, **options):
        self.dry_run = options['dry_run']
        self.yes = options['yes']
        _dataset_identifiers: list[str] | None = options.get('datasets')

        source_id = options['source']
        dest_id = options['destination']

        try:
            source_ic = InstanceConfig.objects.get(identifier=source_id)
        except InstanceConfig.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Source instance '{source_id}' not found"))
            return

        try:
            dest_ic = InstanceConfig.objects.get(identifier=dest_id)
        except InstanceConfig.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Destination instance '{dest_id}' not found"))
            return

        src = DatasetDjangoAdapter(scope=source_ic)
        dst = DatasetDjangoAdapter(scope=dest_ic, interactive=True)
        if self.yes:
            dst.allow_related_deletion = True

        with src.start():
            src.load()
            with dst.start():
                dst.load()
                dst.sync_from(src)
                if self.dry_run:
                    self.stderr.write(self.style.WARNING('Dry run requested; no changes done.'))
                    dst.rollback()
                    return

        self.stdout.write(self.style.SUCCESS(f"Successfully copied datasets from '{source_id}' to '{dest_id}'"))
