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
    DataPointComment,
    Dataset,
    DatasetMetric,
    DatasetSchema,
    DatasetSchemaDimension,
    DatasetSchemaScope,
    DatasetSourceReference,
    DataSource,
    Dimension,
    DimensionCategory,
    DimensionScope,
)
from kausal_common.i18n.pydantic import TranslatedString

from common import polars as ppl
from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN
from nodes.dataset_placeholders import make_external_dataset_ref, sync_dataset_placeholder
from nodes.datasets import JSONDataset
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.dimensions import Dimension as DimensionSpec, DimensionCategory as DimensionCategorySpec
    from nodes.units import Unit

# Must match notebooks/upload_new_dataset.py's SOURCE_NAME_SEPARATOR: a 'Source' cell may
# join multiple citation names when a value was derived from more than one.
SOURCE_NAME_SEPARATOR = '; '


def _translated_metadata(values: dict[str, str], default_language: str) -> TranslatedString:
    values = values.copy()
    if default_language not in values:
        values[default_language] = values.get('en') or next(iter(values.values()))
    return TranslatedString(default_language=default_language, **values)


def _label_from_identifier(identifier: str) -> str:
    return identifier.replace('_', ' ').replace('-', ' ').title()


def _parse_column_dimension_mappings(values: list[str]) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for value in values:
        if '=' not in value:
            raise ValueError("--create-dimension-from-column values must use the syntax 'column=dimension_id'")
        column, dimension_id = value.split('=', maxsplit=1)
        column = column.strip()
        dimension_id = dimension_id.strip()
        if not column or not dimension_id:
            raise ValueError("--create-dimension-from-column values must use the syntax 'column=dimension_id'")
        mappings[column] = dimension_id
    return mappings


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
    return DimensionCategory.objects.get(dimension=dimension, identifier=identifier)


class Command(BaseCommand):
    help = 'Create a dataset in DB based on a DVC dataset'

    # Map dimension identifiers to a dict mapping dimension category identifiers to DimensionCategory instances
    dimension_categories: dict[str, dict[str, DimensionCategory]] = {}

    def add_arguments(self, parser):
        parser.add_argument('instance', metavar='INSTANCE_ID', type=str, nargs=1)
        parser.add_argument('datasets', metavar='DATASET_ID', type=str, nargs='*')
        parser.add_argument('--all', action='store_true', help='Sync all datasets')
        parser.add_argument(
            '--metadata-only',
            action='store_true',
            help='Create placeholder dataset, schema and metric objects without importing datapoints',
        )
        parser.add_argument(
            '--ignore-prefix',
            action='append',
            help='Ignore datasets with IDs starting with the specified prefix. Can be used multiple times.',
        )
        parser.add_argument(
            '--create-dimension-from-column',
            action='append',
            default=[],
            metavar='COLUMN=DIMENSION_ID',
            help=(
                'Create or update a scoped dimension from a DVC index column, then import that column under '
                'DIMENSION_ID. Can be used multiple times.'
            ),
        )
        parser.add_argument('--force', action='store_true')

    def sync_dataset(  # noqa: C901
        self,
        instance_config: InstanceConfig,
        ctx: Context,
        ds_id: str,
        force: bool = False,
        metadata_only: bool = False,
        create_dimensions_from_columns: dict[str, str] | None = None,
    ):
        create_dimensions_from_columns = create_dimensions_from_columns or {}
        if metadata_only:
            sync_dataset_placeholder(instance_config, ctx, ds_id, force=force, reporter=print)
            return

        dvc_ds = ctx.load_dvc_dataset(ds_id)
        df = ppl.from_dvc_dataset(dvc_ds)
        self.rename_value_columns(df)
        df_metadata = df.get_meta()
        dvc_metadata = dvc_ds.metadata or {}

        identifier = ds_id
        scope_content_type = ContentType.objects.get_for_model(instance_config)
        get_kwargs = dict(
            scope_content_type=scope_content_type,
            scope_id=instance_config.pk,
            identifier=identifier,
        )
        dataset: Dataset | None = None
        schema: DatasetSchema | None = None
        datasets = list(
            Dataset.objects
            .get_queryset()
            .for_instance_config(instance_config)
            .filter(identifier=identifier)
            .select_related('schema')[:2]
        )
        if len(datasets) > 1:
            raise RuntimeError(f"Multiple datasets with identifier '{identifier}' exist for instance '{instance_config}'.")
        if datasets:
            dataset = datasets[0]
            if dataset.is_external_placeholder:
                print(f"Dataset '{dataset}' with identifier '{identifier}' is an external placeholder. Replacing.")
                dataset.is_external_placeholder = False
                dataset.scope_content_type = scope_content_type
                dataset.scope_id = instance_config.pk
                dataset.external_ref = make_external_dataset_ref(ctx, ds_id)
                dataset.save(update_fields=['is_external_placeholder', 'scope_content_type', 'scope_id', 'external_ref'])
                schema = dataset.schema
                assert schema is not None
            elif force:
                schema = dataset.schema
                assert schema is not None
                if schema.datasets.count() > 1:
                    print('Dataset exists already, but schema is linked to other datasets as well. Aborting.')
                    return
                print(f"Deleting existing dataset '{dataset}'")
                dataset.delete()
                print(f"Deleting dataset schema '{schema}'")
                schema.delete()
            else:
                print(f"Dataset '{dataset}' with identifier '{identifier}' exists for instance '{instance_config}'. Aborting.")
                return

        if schema is None:
            schema = self.create_dataset_schema(
                instance_config=instance_config,
                default_language=ctx.instance.default_language,
                name_i18n=dvc_metadata['name'],
                description_i18n=dvc_metadata.get('description'),
            )
        if dataset is None:
            dataset = Dataset.objects.create(**get_kwargs, schema=schema, external_ref=make_external_dataset_ref(ctx, ds_id))
            print(f"Created dataset '{dataset}'")

        # Match DB metric columns (DVC units keys) to meta: column_id is the physical column name; id is optional slug.
        metrics_meta = {
            (m.get('column_id') or m.get('id')): m for m in dvc_metadata.get('metrics') or [] if m.get('column_id') or m.get('id')
        }

        metrics = {m.name: m for m in schema.metrics.all() if m.name}
        # Map metric identifiers (column names) to Metric instances
        metrics.update({
            col: self.create_metric(
                col=col,
                unit=df_metadata.units[col],
                schema=schema,
                default_language=ctx.instance.default_language,
                label_i18n=metrics_meta.get(col, {}).get('label'),
            )
            for col in df_metadata.metric_cols
            if col not in metrics
        })

        df, column_dimensions = self.sync_dimensions(
            schema=schema,
            instance_config=instance_config,
            ctx=ctx,
            df=df,
            create_dimensions_from_columns=create_dimensions_from_columns,
        )

        for col, dt in df.schema.items():
            if dt == pl.Categorical:
                df = df.with_columns(pl.col(col).cast(pl.Utf8))
        self.create_data_points(
            instance_config,
            df,
            dataset,
            metrics,
            column_dimensions=column_dimensions,
            sources_meta=dvc_metadata.get('sources'),
        )

    def create_dataset_schema(
        self,
        instance_config: InstanceConfig,
        default_language: str,
        name_i18n: dict[str, str] | None,
        description_i18n: dict[str, str] | None = None,
    ) -> DatasetSchema:
        schema = DatasetSchema(
            time_resolution=DatasetSchema.TimeResolution.YEARLY,  # TODO: allow other granularities
            # unit=?,  # What the hell is this for in DatasetSchema?
        )
        if name_i18n is not None:
            name = _translated_metadata(name_i18n, default_language)
            name.set_modeltrans_field(schema, 'name', default_language)
        if description_i18n:
            schema.description = description_i18n.get(default_language) or next(iter(description_i18n.values()))
        schema.save()
        print(f"Created dataset schema '{schema}'")
        print(f"Setting scope of schema '{schema}' to '{instance_config}'")
        DatasetSchemaScope.objects.create(
            schema=schema,
            scope_content_type=ContentType.objects.get_for_model(instance_config),
            scope_id=instance_config.pk,
        )
        return schema

    def get_or_create_data_sources(
        self, instance_config: InstanceConfig, sources_meta: list[dict[str, str | None]] | None
    ) -> dict[str, DataSource]:
        """Resolve a dvc_metadata['sources'] list into DataSource rows, keyed by name."""
        if not sources_meta:
            return {}
        scope_content_type = ContentType.objects.get_for_model(instance_config)
        result: dict[str, DataSource] = {}
        for fields in sources_meta:
            name = fields['name']
            assert name is not None
            source, _ = DataSource.objects.get_or_create(
                scope_content_type=scope_content_type,
                scope_id=instance_config.pk,
                name=name,
                defaults={
                    'authority': fields.get('authority'),
                    'url': fields.get('url'),
                    'description': fields.get('description'),
                },
            )
            result[name] = source
        return result

    def link_data_point_sources(self, data_point: DataPoint, source_cell: str, data_sources: dict[str, DataSource]) -> None:
        """Link data_point to each DataSource named in source_cell (SOURCE_NAME_SEPARATOR-joined for >1 citation)."""
        for name in source_cell.split(SOURCE_NAME_SEPARATOR):
            data_source = data_sources.get(name)
            if data_source is not None:
                DatasetSourceReference.objects.create(data_point=data_point, data_source=data_source)
            else:
                print(f"Source '{name}' not found in dvc_metadata['sources']; skipping.")

    def create_data_points(
        self,
        instance_config: InstanceConfig,
        df: ppl.PathsDataFrame,
        dataset: Dataset,
        metrics: dict[str, DatasetMetric],
        *,
        column_dimensions: dict[str, str] | None = None,
        sources_meta: list[dict[str, str | None]] | None = None,
    ):
        column_dimensions = column_dimensions or {}
        data_sources = self.get_or_create_data_sources(instance_config, sources_meta)
        meta = df.get_meta()
        table = JSONDataset.serialize_df(df)
        # We might not need to serialize `df` to create the data points, but I didn't check what the manipulations
        # of `df` above and the serialization do, so I'll take the serialization like the old version of
        # this management command did.
        # 'Source'/'Comment' (or, for plain_csv_wide, 'Description') are reserved per-row columns:
        # not dimensions, read back here into DataSource/DataPointComment links.
        source_col = next((c for c in df.columns if c.lower() == 'source'), None)
        comment_col = next((c for c in df.columns if c.lower() in ('comment', 'description')), None)
        num_created = 0
        for row in table['data']:
            year_val = row['Year']
            year = date(year=year_val, month=1, day=1)
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
                for column in meta.dim_ids:
                    dimension_identifier = column_dimensions.get(column, column)
                    dim_cat_identifier = row[column]
                    if dim_cat_identifier:
                        try:
                            cat = get_dimension_category(instance_config, dimension_identifier, dim_cat_identifier)
                        except DimensionCategory.DoesNotExist:
                            print(f"Dimension category '{dim_cat_identifier}' not found. Did you run --update-instance?")
                            raise
                        data_point.dimension_categories.add(cat)
                if source_col and row.get(source_col):
                    self.link_data_point_sources(data_point, row[source_col], data_sources)
                if comment_col and row.get(comment_col):
                    DataPointComment.objects.create(data_point=data_point, text=row[comment_col])
        print(f'Created {num_created} data points')

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
                raise Exception(f'Unknown column {col}')

    def get_category_identifiers_from_column(self, df: ppl.PathsDataFrame, column: str) -> list[str]:
        return sorted(str(value) for value in df[column].drop_nulls().unique().to_list())

    def sync_dimensions(
        self,
        schema: DatasetSchema,
        instance_config: InstanceConfig,
        ctx: Context,
        df: ppl.PathsDataFrame,
        create_dimensions_from_columns: dict[str, str],
    ) -> tuple[ppl.PathsDataFrame, dict[str, str]]:
        df_metadata = df.get_meta()
        dim_ids = set(df_metadata.dim_ids)
        for column in create_dimensions_from_columns:
            if column not in dim_ids:
                raise ValueError(
                    f"Column '{column}' is not a dimension/index column in the DVC dataset. "
                    + f'Available dimension columns: {", ".join(sorted(dim_ids))}'
                )

        column_dimensions: dict[str, str] = {}
        for col in df_metadata.dim_ids:
            if dimension_identifier := create_dimensions_from_columns.get(col):
                self.remove_schema_dimensions_for_column(
                    schema=schema,
                    instance_config=instance_config,
                    column_name=col,
                    keep_dimension_identifier=dimension_identifier,
                )
                self.get_or_create_dimension_from_column(
                    schema=schema,
                    instance_config=instance_config,
                    column_name=col,
                    dimension_identifier=dimension_identifier,
                    category_identifiers=self.get_category_identifiers_from_column(df, col),
                )
                column_dimensions[col] = dimension_identifier
                continue

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

        return df, column_dimensions

    def create_metric(
        self, col: str, unit: Unit, schema: DatasetSchema, default_language: str, label_i18n: dict[str, str] | None
    ) -> DatasetMetric:
        metric = DatasetMetric(schema=schema, name=col, label=col, unit=str(unit))
        if label_i18n is not None:
            label = _translated_metadata(label_i18n, default_language)
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
            f"There is already a dimension with identifier '{spec.id}' for '{instance_config}'; "
            + 'skipping creation of Dimension, DimensionCategory and DimensionScope instances and '
            + f"linking the existing dimension to the schema '{schema}'"
        )
        dimension = existing_scope.dimension
        if schema.dimensions.filter(dimension=dimension).exists():
            print(f"Dimension '{dimension}' is already linked to schema '{schema}'")
            return dimension
        print(f"Linking dimension '{existing_scope.dimension}' to schema '{schema}'")
        DatasetSchemaDimension.objects.create(schema=schema, dimension=existing_scope.dimension)
        return existing_scope.dimension

    def remove_schema_dimensions_for_column(
        self,
        schema: DatasetSchema,
        instance_config: InstanceConfig,
        column_name: str,
        keep_dimension_identifier: str,
    ) -> None:
        scope_content_type = ContentType.objects.get_for_model(instance_config)
        for schema_dim in schema.dimensions.select_related('dimension'):
            scope = DimensionScope.objects.filter(
                dimension=schema_dim.dimension,
                scope_content_type=scope_content_type,
                scope_id=instance_config.pk,
            ).first()
            if scope is None or scope.identifier == keep_dimension_identifier:
                continue
            dimension_column = schema_dim.column_name or scope.identifier
            if dimension_column != column_name:
                continue
            print(
                f"Removing dimension '{schema_dim.dimension}' from schema '{schema}' "
                + f"because column '{column_name}' is now mapped to '{keep_dimension_identifier}'"
            )
            schema_dim.delete()

    def get_or_create_dimension_from_column(
        self,
        schema: DatasetSchema,
        instance_config: InstanceConfig,
        column_name: str,
        dimension_identifier: str,
        category_identifiers: list[str],
    ) -> Dimension:
        try:
            existing_scope = DimensionScope.objects.get(
                scope_content_type=ContentType.objects.get_for_model(instance_config),
                scope_id=instance_config.pk,
                identifier=dimension_identifier,
            )
        except DimensionScope.DoesNotExist:
            dimension = Dimension.objects.create(name=_label_from_identifier(dimension_identifier))
            DimensionScope.objects.create(
                dimension=dimension,
                scope_content_type=ContentType.objects.get_for_model(instance_config),
                scope_id=instance_config.pk,
                identifier=dimension_identifier,
            )
            print(f"Created dimension '{dimension}' from dataset column '{column_name}'")
        else:
            dimension = existing_scope.dimension
            print(f"There is already a dimension with identifier '{dimension_identifier}' for '{instance_config}'")

        existing_categories = set(dimension.categories.values_list('identifier', flat=True))
        created_count = 0
        for category_identifier in category_identifiers:
            if category_identifier in existing_categories:
                continue
            DimensionCategory.objects.create(
                dimension=dimension,
                identifier=category_identifier,
                label=_label_from_identifier(category_identifier),
            )
            created_count += 1
        if created_count:
            print(f"Created {created_count} categories for dimension '{dimension_identifier}'")
        schema_dim = schema.dimensions.filter(dimension=dimension).first()
        column_name_to_store = column_name if column_name != dimension_identifier else None
        if schema_dim is None:
            print(f"Linking dimension '{dimension}' to schema '{schema}'")
            DatasetSchemaDimension.objects.create(
                schema=schema,
                dimension=dimension,
                column_name=column_name_to_store,
            )
        elif schema_dim.column_name != column_name_to_store:
            schema_dim.column_name = column_name_to_store
            schema_dim.save(update_fields=['column_name'])
            print(f"Updated dimension '{dimension}' column mapping in schema '{schema}'")
        else:
            print(f"Dimension '{dimension}' is already linked to schema '{schema}'")
        return dimension

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
                print('Available datasets:')
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
        create_dimensions_from_columns = _parse_column_dimension_mappings(options['create_dimension_from_column'])

        # Ensure all prefixes end with / character for filtering
        normalized_prefixes = []
        for prefix in ignore_prefixes:
            if not prefix.endswith('/'):
                normalized_prefixes.append(f'{prefix}/')
            else:
                normalized_prefixes.append(prefix)

        if normalized_prefixes:
            filtered_ds_ids = [ds_id for ds_id in ds_ids if not any(ds_id.startswith(prefix) for prefix in normalized_prefixes)]

            ignored_count = len(ds_ids) - len(filtered_ds_ids)
            if ignored_count > 0:
                display_prefixes = [p[:-1] for p in normalized_prefixes]
                print(f'Ignoring {ignored_count} dataset(s) with prefix(es): {", ".join(display_prefixes)}')

            ds_ids = filtered_ds_ids

        for ds_id in ds_ids:
            with transaction.atomic():
                self.sync_dataset(
                    ic,
                    ctx,
                    ds_id,
                    force=options['force'],
                    metadata_only=options['metadata_only'],
                    create_dimensions_from_columns=create_dimensions_from_columns,
                )
