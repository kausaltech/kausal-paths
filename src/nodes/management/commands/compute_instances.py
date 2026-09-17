import time
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError

from loguru import logger

from nodes.models import InstanceConfig, PreferredInstanceSource

if TYPE_CHECKING:
    from django.core.management.base import CommandParser


class Command(BaseCommand):
    help = 'Compute default and baseline outcome nodes for selected active instances, populating normal model caches.'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('identifiers', nargs='*', help='Instance identifiers to compute')
        parser.add_argument('--in-customer-use', action='store_true', help='Select instances marked as in customer use')

    def compute_instance(self, config: InstanceConfig) -> None:
        with config.enter_instance_context(source=PreferredInstanceSource.PUBLISHED) as instance:
            try:
                context = instance.context
                outcomes = context.get_outcome_nodes()
                default = context.get_default_scenario()
                scenarios = [default]
                baseline = context.scenarios.get('baseline')
                if baseline is not None and baseline is not default:
                    scenarios.append(baseline)
                with context.run():
                    for scenario in scenarios:
                        with scenario.override(set_active=True):
                            for node in outcomes:
                                node.get_output_pl()
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
        succeeded = 0
        failed = []
        started = time.monotonic()
        for config in instances.order_by('identifier').iterator():
            instance_started = time.monotonic()
            self.stdout.write(f'Computing {config.identifier}...')
            try:
                self.compute_instance(config)
            except Exception:
                logger.exception('Unable to compute instance {}', config.identifier)
                failed.append(config.identifier)
                self.stderr.write(f'Failed {config.identifier} after {time.monotonic() - instance_started:.1f}s')
            else:
                succeeded += 1
                self.stdout.write(f'Computed {config.identifier} in {time.monotonic() - instance_started:.1f}s')
        self.stdout.write(f'Computed {succeeded} instances; {len(failed)} failed; elapsed {time.monotonic() - started:.1f}s')
        if failed:
            raise CommandError('Failed instances: %s' % ', '.join(failed))
