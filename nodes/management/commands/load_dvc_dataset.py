from __future__ import annotations

from datetime import date
from functools import lru_cache
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


@lru_cache
def get_dimension(instance_config: InstanceConfig, identifier: str) -> Dimension:
    scope = DimensionScope.objects.get(
            scope_content_type=ContentType.objects.get_for_model(instance_config),
            scope_id=instance_config.pk,
            identifier=identifier,
    )
    return scope.dimension


@lru_cache
def get_dimension_category(instance_config: InstanceConfig, dimension_identifier: str, identifier: str) -> DimensionCategory:
    dimension = get_dimension(instance_config, dimension_identifier)
    print('dimension', dimension, 'identifier:', identifier)
    return DimensionCategory.objects.get(dimension=dimension, identifier=identifier)


class Command(BaseCommand):
    help = 'Create a dataset in DB based on a DVC dataset'

    # Map dimension identifiers to a dict mapping dimension category identifiers to DimensionCategory instances
    dimension_categories: dict[str, dict[str, DimensionCategory]] = {}

    def add_arguments(self, parser):
        parser.add_argument('instance', metavar='INSTANCE_ID', type=str, nargs=1)
        parser.add_argument('datasets', metavar='DATASET_ID', type=str, nargs='*')
        parser.add_argument('--all', action='store_true', help='Sync all datasets')
        parser.add_argument('--ignore-prefix', action='append',
                           help='Ignore datasets with IDs starting with the specified prefix. Can be used multiple times.')
        parser.add_argument('--force', action='store_true')

    def sync_dataset(self, instance_config: InstanceConfig, ctx: Context, ds_id: str, force: bool = False):
        dvc_ds = ctx.load_dvc_dataset(ds_id)
        df = ppl.from_dvc_dataset(dvc_ds)
        self.rename_value_columns(df)
        df_metadata = df.get_meta()
        dvc_metadata = dvc_ds.metadata or {}

        identifier = ds_id
        get_kwargs = dict(
            scope_content_type=ContentType.objects.get_for_model(instance_config),
            scope_id=instance_config.pk,
            identifier=identifier,
        )
        try:
            dataset = Dataset.objects.get(**get_kwargs)
        except Dataset.DoesNotExist:
            pass
        else:
            if force:
                schema = dataset.schema
                assert schema is not None
                if schema.datasets.count() > 1:
                    print("Dataset exists already, but schema is linked to other datasets as well. Aborting.")
                    return
                print(f"Deleting existing dataset '{dataset}'")
                dataset.delete()
                print(f"Deleting dataset schema '{schema}'")
                schema.delete()
            else:
                print(
                    f"Dataset '{dataset}' with identifier '{identifier}' exists already for instance "
                    f"'{instance_config}'. Aborting."
                )
                return

        schema = self.create_dataset_schema(
            instance_config=instance_config,
            default_language=ctx.instance.default_language,
            name_i18n=dvc_metadata['name'],
        )
        create_kwargs = dict(
            **get_kwargs,
            schema=schema,
        )
        dataset = Dataset.objects.create(**create_kwargs)
        print(f"Created dataset '{dataset}'")

        metrics_meta = {m.get('id', None): m for m in dvc_metadata.get('metrics', [])}  # Don't blame me for this!
        # Map metric identifiers (column names) to Metric instances
        metrics = {
            col: self.create_metric(
                col=col,
                unit=df_metadata.units[col],
                schema=schema,
                default_language=ctx.instance.default_language,
                label_i18n=metrics_meta.get(col, {}).get('label'),
            )
            for col in df_metadata.metric_cols
        }

        for col in df_metadata.dim_ids:
            self.get_or_create_dimension(
                schema=schema,
                instance_config=instance_config,
                default_language=ctx.instance.default_language,
                spec=ctx.dimensions[col],
            )
            # Does the following ever do anything to the column `col` in `df` other than converting strings to
            # `pl.Categorical`?
            new_col = ctx.dimensions[col].series_to_ids_pl(df[col], allow_null=True)
            # Let's throw in an assert and find out.
            # assert new_col.equals(df[col])
            df = df.with_columns(new_col)

        for col, dt in df.schema.items():
            if dt == pl.Categorical:
                df = df.with_columns(pl.col(col).cast(pl.Utf8))
        self.create_data_points(instance_config, df, dataset, metrics)

    def create_dataset_schema(
        self, instance_config: InstanceConfig, default_language: str, name_i18n: dict[str, str] | None
    ) -> DatasetSchema:
        schema = DatasetSchema(
            time_resolution=DatasetSchema.TimeResolution.YEARLY,  # TODO: allow other granularities
            # unit=?,  # What the hell is this for in DatasetSchema?
        )
        if name_i18n is not None:
            name = TranslatedString(default_language=default_language, **name_i18n)
            name.set_modeltrans_field(schema, 'name', default_language)
        schema.save()
        print(f"Created dataset schema '{schema}'")
        print(f"Setting scope of schema '{schema}' to '{instance_config}'")
        DatasetSchemaScope.objects.create(
            schema=schema,
            scope_content_type=ContentType.objects.get_for_model(instance_config),
            scope_id=instance_config.pk,
        )
        return schema

    def create_data_points(
        self,
        instance_config: InstanceConfig,
        df: ppl.PathsDataFrame,
        dataset: Dataset,
        metrics: dict[str, DatasetMetric]
    ):
        meta = df.get_meta()
        table = JSONDataset.serialize_df(df)
        # We might not need to serialize `df` to create the data points, but I didn't check what the manipulations
        # of `df` above and the serialization do, so I'll take the serialization like the old version of
        # this management command did.
        num_created = 0
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
                num_created += 1
                for dimension in meta.dim_ids:
                    dim_cat_identifier = row[dimension]
                    if dim_cat_identifier:
                        cat = get_dimension_category(instance_config, dimension, dim_cat_identifier)
                        data_point.dimension_categories.add(cat)
        print(f"Created {num_created} data points")

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
                print(df)
                raise Exception(f"Unknown column {col}")

    def create_metric(
        self, col: str, unit: Unit, schema: DatasetSchema, default_language: str, label_i18n: dict[str, str] | None
    ) -> DatasetMetric:
        metric = DatasetMetric(schema=schema, name=col, unit=str(unit))
        if label_i18n is not None:
            label = TranslatedString(default_language=default_language, **label_i18n)
            label.set_modeltrans_field(metric, 'label', default_language)
        metric.save()
        print(f"Created metric '{metric}' and linking it to schema '{schema}'")
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
            f"There is already a dimension with identifier '{spec.id}' for '{instance_config}'; skipping creation of "
            f"Dimension, DimensionCategory and DimensionScope instances and linking the existing dimension to the "
            f"schema '{schema}'"
        )
        DatasetSchemaDimension.objects.create(schema=schema, dimension=existing_scope.dimension)
        return existing_scope.dimension

    def create_dimension(
        self, schema: DatasetSchema, instance_config: InstanceConfig, default_language: str, spec: DimensionSpec
    ) -> Dimension:
        dimension = Dimension()
        label = spec.label
        assert isinstance(label, TranslatedString)
        label.set_modeltrans_field(dimension, 'name', default_language)
        dimension.save()
        print(f"Created dimension '{dimension}' and linking it to schema '{schema}'")
        DatasetSchemaDimension.objects.create(schema=schema, dimension=dimension)
        for cat_spec in spec.categories:
            self.create_dimension_category(
                dimension=dimension,
                default_language=default_language,
                spec=cat_spec,
            )
        print(f"Setting scope of dimension '{dimension}' to '{instance_config}'")
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
        cat = DimensionCategory(dimension=dimension, identifier=spec.id)
        label = spec.label
        assert isinstance(label, TranslatedString)
        label.set_modeltrans_field(cat, 'label', default_language)
        cat.save()
        print(f"Created dimension category '{cat}'")
        return cat

    def handle(self, *args, **options):
        instance_id = options['instance'][0]
        ic = InstanceConfig.objects.get(identifier=instance_id)
        ctx = ic.get_instance().context
        if not options['datasets']:
            if not options['all']:
                print("Available datasets:")
                dvc_dataset_ids = sorted(ctx.get_all_dvc_dataset_ids())
                for ds_id in dvc_dataset_ids:
                    print(ds_id)
                exit()
            else:
                dvc_dataset_ids = sorted(ctx.get_all_dvc_dataset_ids())
                ds_ids = dvc_dataset_ids
        else:
            ds_ids = options['datasets']

        ignore_prefixes = options.get('ignore_prefix') or []

        # Ensure all prefixes end with / character for filtering
        normalized_prefixes = []
        for prefix in ignore_prefixes:
            if not prefix.endswith('/'):
                normalized_prefixes.append(f"{prefix}/")
            else:
                normalized_prefixes.append(prefix)

        if normalized_prefixes:
            filtered_ds_ids = [ds_id for ds_id in ds_ids
                              if not any(ds_id.startswith(prefix) for prefix in normalized_prefixes)]

            ignored_count = len(ds_ids) - len(filtered_ds_ids)
            if ignored_count > 0:
                display_prefixes = [p[:-1] for p in normalized_prefixes]
                print(f"Ignoring {ignored_count} dataset(s) with prefix(es): {', '.join(display_prefixes)}")

            ds_ids = filtered_ds_ids

        for ds_id in ds_ids:
            with transaction.atomic():
                self.sync_dataset(ic, ctx, ds_id, force=options['force'])
