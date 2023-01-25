import pint_pandas
import polars as pl
import pandas as pd

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
        parser.add_argument('instance', metavar='INSTANCE_ID', type=str, nargs=1)
        parser.add_argument('datasets', metavar='DATASET_ID', type=str, nargs='*')
        parser.add_argument('--force', type=bool)

    def sync_dataset(self, ic: InstanceConfig, ctx: Context, ds_id: str, force: bool = False):
        ds = ctx.load_dvc_dataset(ds_id)
        df = ppl.from_pandas(ds.df)
        dims = []
        metrics = []
        meta = df.get_meta()
        all_cols = list(df.columns)
        for col in meta.metric_cols:
            unit = meta.units[col]
            metrics.append(DatasetMetric(identifier=col, label=col, unit=str(unit)))
            all_cols.remove(col)
        for col in meta.dim_ids:
            dim = ctx.dimensions[col]
            ic.sync_dimension(dim)
            df = df.with_column(dim.series_to_ids_pl(df[col]))
            dims.append(ic.dimensions.get(identifier=col))
            all_cols.remove(col)

        for col in all_cols:
            if col.lower() == FORECAST_COLUMN.lower():
                if col != FORECAST_COLUMN:
                    df = df.rename({col: FORECAST_COLUMN})
            elif col.lower() == YEAR_COLUMN.lower():
                if col != YEAR_COLUMN:
                    df = df.rename({col: YEAR_COLUMN})
            else:
                raise Exception("Unknown column '%s'" % col)

        identifier = ds_id.split('/')[-1]
        obj = ic.datasets.filter(identifier=identifier).first()
        if obj is None or force:
            if obj is not None:
                print("Removing existing dataset")
                obj.delete()
            obj = Dataset(identifier=identifier, instance=ic, name=ds_id)
            obj.save()
            for m in metrics:
                m.dataset = obj
                m.save()
            for d in dims:
                obj.dimensions.add(d)
            obj.save()
        else:
            print("Dataset %s already exists" % identifier)
            return

        pdf = df.to_pandas()
        obj.table = JSONDataset.serialize_df(pdf, add_uuids=True)
        obj.save()

    def handle(self, *args, **options):
        instance_id = options['instance'][0]
        ic = InstanceConfig.objects.get(identifier=instance_id)
        ctx = ic.get_instance().context
        if not options['datasets']:
            print("Available datasets:")
            ctx.generate_baseline_values()
            for ds_id, ds in ctx.dvc_datasets.items():
                print(ds_id)
            exit()

        for ds_id in options['datasets']:
            with transaction.atomic():
                self.sync_dataset(ic, ctx, ds_id, options['force'])
