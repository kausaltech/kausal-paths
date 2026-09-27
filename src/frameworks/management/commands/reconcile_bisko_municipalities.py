"""Repair already activated BISKO municipalities after template-local input migration."""

from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from kausal_common.datasets.models import Dataset

from frameworks.activation import ActivationError, activate_bisko_municipality
from frameworks.models import Framework, FrameworkConfig
from nodes.models import InputPortBindingSet

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = 'Reconcile named BISKO municipal instances with local datasets, identity, and a draft submission.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instances', nargs='+', metavar='INSTANCE', help='BISKO instance identifier; repeat for each town.')
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args: Any, **options: Any) -> None:
        framework = Framework.objects.filter(identifier='bisko').first()
        if framework is None:
            raise CommandError('BISKO framework not found.')
        names = options['instances']
        configs = {
            config.instance_config.identifier: config
            for config in FrameworkConfig.objects.filter(
                framework=framework, instance_config__identifier__in=names
            ).select_related('instance_config__organization')
        }
        missing = set(names) - configs.keys()
        if missing:
            raise CommandError(f'Not a BISKO framework instance: {", ".join(sorted(missing))}')
        results: list[str] = []
        try:
            with transaction.atomic():
                for name in names:
                    original = configs[name]
                    config, _ = activate_bisko_municipality(framework, original.instance_config.organization)
                    instance = config.instance_config
                    local_count = Dataset.objects.for_instance_config(instance).filter(identifier__startswith='kommune/').count()
                    override_count = InputPortBindingSet.objects.filter(instance=instance).count()
                    results.append(
                        f'{name}: {local_count} local datasets, {override_count} binding overrides, '
                        f'{instance.submissions.count()} submissions'
                    )
                if options['dry_run']:
                    transaction.set_rollback(True)
        except ActivationError as error:
            raise CommandError(str(error)) from error
        for line in results:
            self.stdout.write(line)
        if options['dry_run']:
            self.stdout.write('Dry run: database changes rolled back.')
