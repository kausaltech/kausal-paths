from django.core.management.base import BaseCommand, CommandParser

from kausal_common.datasets.models import Dataset

from nodes.dataset_materialization import materialize_dataset


class Command(BaseCommand):
    help = 'Build or refresh current serialized payloads for DB-backed datasets.'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('--instance', dest='instance_identifier')
        parser.add_argument('--batch-size', type=int, default=100)

    def handle(self, *args: object, **options: object) -> None:
        instance_identifier = options['instance_identifier']
        batch_size = options['batch_size']
        assert instance_identifier is None or isinstance(instance_identifier, str)
        assert isinstance(batch_size, int)

        qs = Dataset.objects.filter(is_external_placeholder=False).order_by('pk')
        if instance_identifier is not None:
            from nodes.models import InstanceConfig

            instance = InstanceConfig.objects.get(identifier=instance_identifier)
            qs = Dataset.objects.qs.for_instance_config(instance).filter(is_external_placeholder=False).order_by('pk')

        count = 0
        for dataset in qs.iterator(chunk_size=batch_size):
            materialize_dataset(dataset)
            count += 1
            if count % batch_size == 0:
                self.stdout.write(f'Materialized {count} datasets')
        self.stdout.write(self.style.SUCCESS(f'Materialized {count} datasets'))
