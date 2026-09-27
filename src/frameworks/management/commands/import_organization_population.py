"""Project the BISKO provider population dataset onto municipality organizations."""

from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from frameworks.models import Framework
from frameworks.population import refresh_population_from_dvc


class Command(BaseCommand):
    help = 'Load the pinned BISKO population dataset into organization-linked annual observations.'

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args: Any, **options: Any) -> None:
        framework = Framework.objects.filter(identifier='bisko').first()
        if framework is None:
            raise CommandError('BISKO framework not found.')
        try:
            with transaction.atomic():
                result = refresh_population_from_dvc(framework)
                if options['dry_run']:
                    transaction.set_rollback(True)
        except ValueError as error:
            raise CommandError(str(error)) from error
        self.stdout.write(
            f'{result.observations} municipality-year observations, {result.years[0]}-{result.years[-1]}, '
            f'source revision {result.source_revision}'
        )
        if options['dry_run']:
            self.stdout.write('Dry run: database changes rolled back.')
