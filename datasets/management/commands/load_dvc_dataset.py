import polars as pl

from rich import print

from django.core.management.base import BaseCommand
from django.db import transaction
from common.i18n import TranslatedString
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
        parser.add_argument('--force', action='store_true')

    def sync_dataset(self, ic: InstanceConfig, ctx: Context, ds_id: str, force: bool = False):
        ds = ctx.load_dvc_dataset(ds_id)
        df = ppl.from_dvc_dataset(ds)
        metadata = ds.metadata or {}
        metrics_meta = {m.get('id', None): m for m in metadata.get('metrics', [])}
        assert df is not None
        dims: list[Dimension] = []
        metrics = []
        meta = df.get_meta()
        all_cols = list(df.columns)
        for col in meta.metric_cols:
            unit = meta.units[col]
            mm = metrics_meta.get(col)
            mobj = DatasetMetric(identifier=col, unit=str(unit))
            if mm and 'label' in mm:
                label = TranslatedString(default_language=ctx.instance.default_language, **mm.get('label'))
                label.set_modeltrans_field(mobj, 'label', ctx.instance.default_language)
            metrics.append(mobj)
            all_cols.remove(col)

        for col in meta.dim_ids:
            dim = ctx.dimensions[col]
            Dimension.sync_dimension(ic, dim)
            df = df.with_columns([dim.series_to_ids_pl(df[col])])
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

            obj = Dataset(identifier=identifier, dvc_identifier=ds_id, instance=ic, table={}, years=[])
            name = metadata.get('name', None)
            if name is not None:
                if isinstance(name, dict):
                    name = TranslatedString(**name)
                    name.set_modeltrans_field(obj, 'name', default_language=ctx.instance.default_language)
                else:
                    obj.name = name
            else:
                obj.name = ds_id
            obj.save()
            print('Created dataset %s' % obj)
            print('Metrics:')
            for m in metrics:
                m.dataset = obj
                m.save()
                print('\t%s' % m)
            print('Dimensions:')
            for d in dims:
                print('\t%s' % d)
                dd = DatasetDimension.objects.create(dataset=obj, dimension=d)
                cats = []
                df_cats = set(df[d.identifier].unique())
                print('\tCategories:')
                for cat in d.categories.all():
                    if cat.identifier in df_cats:
                        cats.append(cat)
                        print('\t\t%s' % cat)
                assert len(cats) == len(df_cats)
                dd.selected_categories.set(cats)
                dd.save()
            obj.save()
        else:
            print("Dataset %s already exists" % identifier)
            return

        for col, dt in df.schema.items():
            if dt == pl.Categorical:
                df = df.with_columns(pl.col(col).cast(pl.Utf8))

        obj.table = JSONDataset.serialize_df(df, add_uuids=True)
        obj.years = obj.generate_years_from_data()
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
