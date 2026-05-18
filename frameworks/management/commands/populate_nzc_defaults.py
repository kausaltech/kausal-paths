from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand, CommandError

from frameworks.models import FrameworkConfig

if TYPE_CHECKING:
    from django.core.management.base import CommandParser


class Command(BaseCommand):
    help = 'Populate NZC MeasureDataPoint defaults from MeasureTemplateDefaultDataPoint records'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('--framework-config', type=int, metavar='PK', help='Target a single FrameworkConfig by primary key')
        parser.add_argument('--instance', metavar='IDENTIFIER', help='Target a single InstanceConfig by identifier')

    def handle(self, *_args, **options) -> None:
        fk_pk: int | None = options.get('framework_config')
        instance_id: str | None = options.get('instance')

        if fk_pk:
            configs = list(FrameworkConfig.objects.filter(pk=fk_pk))
            if not configs:
                raise CommandError(f'FrameworkConfig with pk={fk_pk} not found')
        elif instance_id:
            from nodes.models import InstanceConfig

            try:
                ic = InstanceConfig.objects.get(identifier=instance_id)
            except InstanceConfig.DoesNotExist:
                raise CommandError(f'InstanceConfig with identifier={instance_id!r} not found')  # noqa: B904
            fc = FrameworkConfig.objects.filter(instance_config=ic).first()
            if fc is None:
                self.stdout.write(self.style.WARNING(f'Instance {instance_id!r} has no associated FrameworkConfig'))
                return
            configs = [fc]
        else:
            configs = list(FrameworkConfig.objects.filter(framework__identifier='nzc'))

        for fc in configs:
            count = fc.populate_measure_defaults_from_default_data_points()
            self.stdout.write(self.style.SUCCESS(f'FC {fc.pk} ({fc}): updated {count} data point(s)'))
