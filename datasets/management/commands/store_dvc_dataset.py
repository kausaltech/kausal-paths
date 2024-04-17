from django.core.management.base import BaseCommand

from rich.console import Console
from rich.table import Table

from datasets.models import Dataset
from nodes.models import InstanceConfig
from nodes.node import Context


class Command(BaseCommand):
    help = 'Create a dataset in DB based on a DVC dataset'

    def add_arguments(self, parser):
        parser.add_argument('instance', metavar='INSTANCE_ID', type=str)
        parser.add_argument('dataset', metavar='DATASET_ID', type=str)
        parser.add_argument('dvc_path', metavar='DVC_PATH', nargs='?', type=str)
        parser.add_argument('--repo-url', metavar='URL', type=str)

    def list_datasets(self, ic: InstanceConfig):
        print("Available datasets:")
        table = Table()
        table.add_column('Identifier')
        table.add_column('Name')
        table.add_column('Updated at')
        qs = ic.datasets.all()
        for ds in qs:
            table.add_row(str(ds.identifier), ds.name, str(ds.updated_at))
        console = Console()
        console.print(table)

    def store_dataset(self, ic: InstanceConfig, ctx: Context, ds_id: str, dvc_path: str | None = None, repo_url: str | None = None):
        ds: Dataset | None = ic.datasets.filter(identifier=ds_id).first()
        if ds is None:
            print("Dataset '%s' not found" % ds_id)
            self.list_datasets(ic)
            exit(1)

        ds.store_to_dvc(dvc_path, repo_url)

    def handle(self, *args, **options):
        instance_id = options['instance']
        ic = InstanceConfig.objects.get(identifier=instance_id)
        ctx = ic.get_instance().context
        if not options['dataset']:
            pass
        self.store_dataset(ic, ctx, options['dataset'], options['dvc_path'], repo_url=options['repo_url'])
