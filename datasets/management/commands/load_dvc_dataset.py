import pint_pandas
import pandas as pd

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from datasets.models import Dataset, DatasetDimension, Dimension, DatasetMetric
from nodes.datasets import JSONDataset
from nodes.models import InstanceConfig
from nodes.node import Context
from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN


class Command(BaseCommand):
    help = 'Create a dataset in DB based on a DVC dataset'

    def add_arguments(self, parser):
        parser.add_argument('instance', metavar='INSTANCE_ID', type=str, nargs=1)
        parser.add_argument('datasets', metavar='DATASET_ID', type=str, nargs='+')

    def sync_dataset(self, ic: InstanceConfig, ctx: Context, ds_id: str):
        ds = ctx.load_dvc_dataset(ds_id)
        df = ds.df
        dims = []
        metrics = []
        for name, dt in list(df.dtypes.items()):
            assert isinstance(name, str)
            if isinstance(dt, pint_pandas.PintType):
                metrics.append(DatasetMetric(identifier=name, label=name, unit=str(dt.units)))
            elif isinstance(dt, (str, pd.CategoricalDtype)):
                dim = ctx.dimensions[name]
                ic.sync_dimension(dim)
                df[name] = dim.series_to_ids(df[name])
                dims.append(ic.dimensions.get(identifier=name))
            elif name.lower() == FORECAST_COLUMN.lower():
                if name != FORECAST_COLUMN:
                    df = df.rename(columns={name: FORECAST_COLUMN})
            elif name.lower() == YEAR_COLUMN.lower():
                if name != YEAR_COLUMN:
                    df = df.rename(columns={name: YEAR_COLUMN})
            else:
                raise Exception("Unknown column '%s': %s" % (name, dt))

        identifier = ds_id.split('/')[-1]
        obj = ic.datasets.filter(identifier=identifier).first()
        if obj is None:
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

        df = df.set_index([YEAR_COLUMN, *[dim.identifier for dim in dims]])
        obj.table = JSONDataset.serialize_df(df, add_uuids=True)
        obj.save()

    def handle(self, *args, **options):
        instance_id = options['instance'][0]
        ic = InstanceConfig.objects.get(identifier=instance_id)
        ctx = ic.get_instance().context
        for ds_id in options['datasets']:
            with transaction.atomic():
                self.sync_dataset(ic, ctx, ds_id)
