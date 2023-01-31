from dvc_pandas import Dataset as DVCDataset, Repository
from rich.table import Table
from rich.console import Console

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from datasets.models import Dataset, DatasetDimension, Dimension, DatasetMetric
from common import polars as ppl
from nodes.datasets import JSONDataset
from nodes.models import InstanceConfig
from nodes.node import Context
from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN


class Command(BaseCommand):
    help = 'Create a dataset in DB based on a DVC dataset'

    def add_arguments(self, parser):
        parser.add_argument('instance', metavar='INSTANCE_ID', type=str)
        parser.add_argument('dataset', metavar='DATASET_ID', type=str)
        parser.add_argument('dvc_path', metavar='DVC_PATH', type=str)
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


    def store_dataset(self, ic: InstanceConfig, ctx: Context, ds_id: str, dvc_path: str, repo_url: str | None = None):
        ds: Dataset | None = ic.datasets.filter(identifier=ds_id).first()
        if ds is None:
            print("Dataset '%s' not found" % ds_id)
            self.list_datasets(ic)
            exit(1)

        assert ds.table is not None
        df = JSONDataset.deserialize_df(ds.table)
        if 'uuid' in df.columns:
            df = df.drop(columns=['uuid'])
        r = ctx.dataset_repo
        repo = Repository(repo_url=repo_url or r.repo_url, dvc_remote=r.dvc_remote)
        repo.set_target_commit(None)
        dvc_ds = DVCDataset(df, dvc_path, ds.updated_at)
        repo.push_dataset(dvc_ds)

    def handle(self, *args, **options):
        instance_id = options['instance']
        ic = InstanceConfig.objects.get(identifier=instance_id)
        ctx = ic.get_instance().context
        if not options['dataset']:
            pass
        self.store_dataset(ic, ctx, options['dataset'], options['dvc_path'], repo_url=options['repo_url'])
