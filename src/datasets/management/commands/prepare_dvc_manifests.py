from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError

from loguru import logger

from nodes.models import InstanceConfig, PreferredInstanceSource

if TYPE_CHECKING:
    from django.core.management.base import CommandParser


class Command(BaseCommand):
    help = 'Prepare shared DVC source manifests for selected active instances without downloading datasets.'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('identifiers', nargs='*', help='Instance identifiers to prepare')
        parser.add_argument('--in-customer-use', action='store_true', help='Select instances marked as in customer use')

    def prepare_instance(self, config: InstanceConfig) -> bool:
        with config.enter_instance_context(source=PreferredInstanceSource.PUBLISHED) as instance:
            try:
                return instance.context.dvc_source_manifest is not None
            finally:
                instance.clean()

    def handle(self, *args: Any, **options: Any) -> None:
        identifiers = options['identifiers']
        if not identifiers and not options['in_customer_use']:
            raise CommandError('Provide instance identifiers or --in-customer-use.')
        instances = InstanceConfig.objects.filter(is_active=True)
        if options['in_customer_use']:
            instances = instances.filter(in_customer_use=True)
        if identifiers:
            instances = instances.filter(identifier__in=identifiers)
            missing = set(identifiers) - set(instances.values_list('identifier', flat=True))
            if missing:
                raise CommandError('No selected active instance: %s' % ', '.join(sorted(missing)))
        failed = []
        for config in instances.order_by('identifier').iterator():
            try:
                prepared = self.prepare_instance(config)
            except Exception:
                logger.exception('Unable to prepare DVC manifests for {}', config.identifier)
                failed.append(config.identifier)
            else:
                self.stdout.write(f'{config.identifier}: {"prepared" if prepared else "no pinned DVC inputs"}')
        if failed:
            raise CommandError('Failed instances: %s' % ', '.join(failed))
