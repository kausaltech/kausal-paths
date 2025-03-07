from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand
from django.db import transaction

import polars as pl
from rich import print

from kausal_common.datasets.models import (
    DataPoint,
    Dataset,
    DatasetMetric,
    DatasetSchema,
    DatasetSchemaDimension,
    DatasetSchemaMetric,
    DatasetSchemaScope,
    Dimension,
    DimensionCategory,
    DimensionScope,
)

from common import polars as ppl
from common.i18n import TranslatedString
from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN
from nodes.datasets import JSONDataset
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.dimensions import Dimension as DimensionSpec, DimensionCategory as DimensionCategorySpec
    from nodes.units import Unit


class Command(BaseCommand):
    help = 'Create a dataset in DB based on a DVC dataset'

    # Map dimension identifiers to a dict mapping dimension category identifiers to DimensionCategory instances
    dimension_categories: dict[str, dict[str, DimensionCategory]] = {}

    def add_arguments(self, parser):
        parser.add_argument('instance', metavar='INSTANCE_ID', type=str, nargs=1)
        parser.add_argument('datasets', metavar='DATASET_ID', type=str, nargs='*')
        # TODO: implement force
        # parser.add_argument('--force', action='store_true')

    def sync_dataset(self, instance_config: InstanceConfig, ctx: Context, ds_id: str):
        dvc_ds = ctx.load_dvc_dataset(ds_id)
        df = ppl.from_dvc_dataset(dvc_ds)
        self.rename_value_columns(df)
        df_metadata = df.get_meta()
        dvc_metadata = dvc_ds.metadata or {}

        identifier = ds_id.split('/')[-1]
        assert identifier == dvc_metadata['identifier']
        dataset, created = Dataset.objects.get_or_create(
            scope_content_type=ContentType.objects.get_for_model(instance_config),
            scope_id=instance_config.pk,
            identifier=identifier,
        )
        if created:
            print(f"Created dataset {dataset}")
        else:
            # TODO: We might want to update the dataset instead of aborting
            print(f"Dataset {identifier} exists already for instance {instance_config}. Aborting.")
            return

        self.create_dataset_schema(
            dataset=dataset,
            instance_config=instance_config,
            default_language=ctx.instance.default_language,
            name_i18n=dvc_metadata['name'],
        )

        metrics_meta = {m.get('id', None): m for m in dvc_metadata.get('metrics', [])}  # Don't blame me for this!
        # Map metric identifiers (column names) to Metric instances
        metrics = {
            col: self.create_metric(
                unit=df_metadata.units[col],
                schema=dataset.schema,
                default_language=ctx.instance.default_language,
                label_i18n=metrics_meta.get(col, {}).get('label'),
            )
            for col in df_metadata.metric_cols
        }

        for col in df_metadata.dim_ids:
            self.get_or_create_dimension(
                schema=dataset.schema,
                instance_config=instance_config,
                default_language=ctx.instance.default_language,
                spec=ctx.dimensions[col],
            )
            # Does the following ever do anything to the column `col` in `df` other than converting strings to
            # `pl.Categorical`?
            new_col = ctx.dimensions[col].series_to_ids_pl(df[col], allow_null=True)
            # Let's throw in an assert and find out.
            assert new_col.equals(df[col])
            df = df.with_columns(new_col)

        for col, dt in df.schema.items():
            if dt == pl.Categorical:
                df = df.with_columns(pl.col(col).cast(pl.Utf8))
        self.create_data_points(df, dataset, metrics)

    def create_dataset_schema(
        self, dataset: Dataset, instance_config: InstanceConfig, default_language: str, name_i18n: dict[str, str] | None
    ) -> DatasetSchema:
        dataset.schema = DatasetSchema(
            time_resolution=DatasetSchema.TimeResolution.YEARLY,  # TODO: allow other granularities
            # unit=?,  # What the hell is this for in DatasetSchema?
        )
        if name_i18n is not None:
            name = TranslatedString(default_language=default_language, **name_i18n)
            name.set_modeltrans_field(dataset.schema, 'name', default_language)
        dataset.schema.save()
        print(f"Created dataset schema {dataset.schema}")
        dataset.save(update_fields=['schema'])
        print(f"Setting scope of schema {dataset.schema} to {instance_config}")
        DatasetSchemaScope.objects.create(
            schema=dataset.schema,
            scope_content_type=ContentType.objects.get_for_model(instance_config),
            scope_id=instance_config.pk,
        )
        return dataset.schema

    def create_data_points(self, df: ppl.PathsDataFrame, dataset: Dataset, metrics: dict[str, DatasetMetric]):
        meta = df.get_meta()
        table = JSONDataset.serialize_df(df)
        # We might not need to serialize `df` to create the data points, but I didn't check what the manipulations
        # of `df` above and the serialization do, so I'll take the serialization like the old version of
        # this management command did.
        for row in table['data']:
            year = date(year=row['Year'], month=1, day=1)  # FIXME: other granularities?
            for metric_identifier, metric in metrics.items():
                value = row[metric_identifier]
                if value is None:
                    continue
                data_point = DataPoint.objects.create(
                    dataset=dataset,
                    date=year,
                    metric=metric,
                    value=value,
                )
                for dimension in meta.dim_ids:
                    dim_cat_identifier = row[dimension]
                    if dim_cat_identifier:
                        cat = self.dimension_categories[dimension][dim_cat_identifier]
                        data_point.dimension_categories.add(cat)

    def rename_value_columns(self, df: ppl.PathsDataFrame):
        meta = df.get_meta()
        value_columns = (c for c in df.columns if c not in meta.metric_cols and c not in meta.dim_ids)
        for col in value_columns:
            if col.lower() == FORECAST_COLUMN.lower():
                if col != FORECAST_COLUMN:
                    df = df.rename({col: FORECAST_COLUMN})
            elif col.lower() == YEAR_COLUMN.lower():
                if col != YEAR_COLUMN:
                    df = df.rename({col: YEAR_COLUMN})
            else:
                raise Exception(f"Unknown column {col}")

    def create_metric(
        self, unit: Unit, schema: DatasetSchema, default_language: str, label_i18n: dict[str, str] | None
    ) -> DatasetMetric:
        metric = DatasetMetric(unit=str(unit))
        if label_i18n is not None:
            label = TranslatedString(default_language=default_language, **label_i18n)
            label.set_modeltrans_field(metric, 'label', default_language)
        metric.save()
        print(f"Created metric {metric} and linking it to schema {schema}")
        DatasetSchemaMetric.objects.create(schema=schema, metric=metric)
        return metric

    def get_or_create_dimension(
        self, schema: DatasetSchema, instance_config: InstanceConfig, default_language: str, spec: DimensionSpec
    ) -> Dimension:
        try:
            existing_scope = DimensionScope.objects.get(
                scope_content_type=ContentType.objects.get_for_model(instance_config),
                scope_id=instance_config.pk,
                identifier=spec.id,
            )
        except DimensionScope.DoesNotExist:
            return self.create_dimension(schema, instance_config, default_language, spec)
        print(
            f"There is already a dimension with identifier {spec.id} for {instance_config}; skipping creation of "
            f"Dimension, DimensionCategory and DimensionScope instances and linking the existing dimension to the "
            f"schema {schema}"
        )
        DatasetSchemaDimension.objects.create(schema=schema, dimension=existing_scope.dimension)
        return existing_scope.dimension

    def create_dimension(
        self, schema: DatasetSchema, instance_config: InstanceConfig, default_language: str, spec: DimensionSpec
    ) -> Dimension:
        dimension = Dimension()
        spec.label.set_modeltrans_field(dimension, 'name', default_language)
        dimension.save()
        print(f"Created dimension {dimension} and linking it to schema {schema}")
        DatasetSchemaDimension.objects.create(schema=schema, dimension=dimension)
        assert spec.id not in self.dimension_categories
        self.dimension_categories[spec.id] = {}
        for cat_spec in spec.categories:
            cat = self.create_dimension_category(
                dimension=dimension,
                default_language=default_language,
                spec=cat_spec,
            )
            self.dimension_categories[spec.id][cat_spec.id] = cat
        print(f"Setting scope of dimension {dimension} to {instance_config}")
        DimensionScope.objects.create(
            dimension=dimension,
            scope_content_type=ContentType.objects.get_for_model(instance_config),
            scope_id=instance_config.pk,
            identifier=spec.id,
        )
        return dimension

    def create_dimension_category(
        self, dimension: Dimension, default_language: str, spec: DimensionCategorySpec
    ) -> DimensionCategory:
        cat = DimensionCategory(dimension=dimension)
        spec.label.set_modeltrans_field(cat, 'label', default_language)
        cat.save()
        print(f"Created dimension category {cat}")
        return cat

    def handle(self, *args, **options):
        instance_id = options['instance'][0]
        ic = InstanceConfig.objects.get(identifier=instance_id)
        ctx = ic.get_instance().context
        if not options['datasets']:
            print("Available datasets:")
            ctx.generate_baseline_values()
            for ds_id in ctx.dvc_datasets.keys():
                print(ds_id)
            exit()

        for ds_id in options['datasets']:
            with transaction.atomic():
                self.sync_dataset(ic, ctx, ds_id)
