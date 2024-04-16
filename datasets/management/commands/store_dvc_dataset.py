from django.core.management.base import BaseCommand

from dvc_pandas import Dataset as DVCDataset, Repository
from rich.console import Console
from rich.table import Table

from common.i18n import get_translated_string_from_modeltrans
from datasets.models import Dataset
from nodes.datasets import JSONDataset
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

        if dvc_path is None:
            if not ds.dvc_identifier:
                if ctx.dataset_repo_default_path:
                    ds.dvc_identifier = '%s/%s' % (ctx.dataset_repo_default_path, ds.identifier)
                else:
                    raise Exception("No DVC path provided but Dataset objects does not have dvc_identifier set")
            ds_dvc_id: str = ds.dvc_identifier
            dvc_path = ds_dvc_id

        assert ds.table is not None
        df = JSONDataset.deserialize_df(ds.table)
        if 'uuid' in df.columns:
            df = df.drop(columns=['uuid'])
        df = df.dropna(how='all')
        r = ctx.dataset_repo
        repo = Repository(repo_url=repo_url or r.repo_url, dvc_remote=r.dvc_remote)
        repo.set_target_commit(None)
        name = get_translated_string_from_modeltrans(ds, 'name', ctx.instance.default_language).i18n
        metrics = [dict(
            id=m.identifier,
            label=get_translated_string_from_modeltrans(m, 'label', ctx.instance.default_language).i18n,
        ) for m in ds.metrics.all()]
        metadata = dict(name=name, identifier=ds.identifier, metrics=metrics)
        dvc_ds = DVCDataset(
            df, identifier=dvc_path, modified_at=ds.updated_at, metadata=metadata
        )
        repo.push_dataset(dvc_ds)

        if ds.dvc_identifier != dvc_path:
            ds.dvc_identifier = dvc_path
            ds.save(update_fields=['dvc_identifier'])

    def handle(self, *args, **options):
        instance_id = options['instance']
        ic = InstanceConfig.objects.get(identifier=instance_id)
        ctx = ic.get_instance().context
        if not options['dataset']:
            pass
        self.store_dataset(ic, ctx, options['dataset'], options['dvc_path'], repo_url=options['repo_url'])
