"""Import BKG administrative organizations from a staged Parquet snapshot."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from frameworks.models import Framework
from orgs.import_bkg import import_bkg_organizations

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = 'Reconcile BKG ARS/AGS organizations and attach state roots to a framework.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('source', type=Path, help='Staged administrative-divisions Parquet file')
        parser.add_argument('--framework', default='bisko', help='Framework identifier')
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            framework = Framework.objects.get(identifier=options['framework'])
            with transaction.atomic():
                result = import_bkg_organizations(options['source'], framework=framework)
                self.stdout.write(str(result))
                if options['dry_run']:
                    transaction.set_rollback(True)
                    self.stdout.write('Dry run: database changes rolled back.')
        except (Framework.DoesNotExist, ValueError, FileNotFoundError) as error:
            raise CommandError(str(error)) from error
