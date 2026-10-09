"""Datasets read from a dvc-pandas repository."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, override

import polars as pl

from kausal_common.logging.errors import capture_error
from kausal_common.perf.perf_context import PerfKind, estimate_size_bytes

from common import polars as ppl
from datasets.runtime.base import DatasetWithFilters, measure_dataset_call
from nodes.constants import (
    FORECAST_COLUMN,
    RESERVED_ROW_COLUMNS,
    UNCERTAINTY_COLUMN,
    VALUE_COLUMN,
    YEAR_COLUMN,
)
from nodes.defs.transform_def import (
    ExtendOp,
    InterpolateOp,
    PortTransformOp,
)
from nodes.exceptions import DatasetError
from nodes.units import Unit, unit_registry

if TYPE_CHECKING:
    import dvc_pandas
    from rich.repr import RichReprResult

    from kausal_common.perf.perf_context import PerfAttrs

    from datasets.prepared import PreparedRecipe
    from nodes.context import Context
    from nodes.defs.node_defs import InputDatasetDef


@dataclass
class DVCDataset(DatasetWithFilters):
    """Dataset that is loaded by dvc-pandas."""

    # The output can be customized further by specifying a column and filters.
    # If `input_dataset` is not specified, we default to `id` being
    # the dvc-pandas dataset identifier.
    input_dataset: str | None = None
    unit: Unit | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.unit is not None:
            assert isinstance(self.unit, Unit)

    @classmethod
    def from_def(cls, ds_def: InputDatasetDef, context: Context) -> Self:
        kwargs = super().kwargs_from_def(ds_def)
        cls.apply_forecast_defaults(kwargs, context)
        return cls(
            id=ds_def.id,
            context=context,
            **kwargs,
            input_dataset=ds_def.input_dataset,
        )

    def __rich_repr__(self) -> RichReprResult:
        yield from super().__rich_repr__()
        if self.input_dataset is not None and self.input_dataset != self.id:
            yield 'input_dataset', self.input_dataset

    def get_span_attrs(self) -> PerfAttrs:
        attrs = super().get_span_attrs()
        attrs['dataset.input.id'] = self.input_dataset or self.id
        return attrs

    @property
    def cache_key(self) -> str | None:
        # Not cached: the key follows `calculate_hash`, which keeps its value only while no
        # transformation reads a parameter.
        return self.get_cache_key()

    def cache_get(self) -> ppl.PathsDataFrame | None:
        if self.context.skip_cache:
            return None
        attrs = self.get_span_attrs()
        if self.cache_key is None:
            return None
        with self.context.perf_context.exec_named(
            kind=PerfKind.DATASET,
            id=self.id,
            op='cache_get',
            attrs=attrs,
        ) as event:
            res = self.context.cache.get(self.cache_key)
            if event is not None:
                event.set_attr('cache.hit', res.is_hit)
                event.set_attr('cache.kind', res.kind.name.lower())
                if res.obj is not None:
                    event.set_attr('dataset.in_memory.bytes', estimate_size_bytes(res.obj))

        if res.is_hit:
            if not isinstance(res.obj, ppl.PathsDataFrame):
                capture_error('Cached dataset %s (key: %s) is not a PathsDataFrame' % (self.id, self.cache_key))
                return None
            return res.obj
        return None

    def cache_set(self, df: ppl.PathsDataFrame) -> None:
        if self.cache_key is None:
            return
        attrs = self.get_span_attrs()
        with self.context.perf_context.exec_named(
            kind=PerfKind.DATASET,
            id=self.id,
            op='cache_set',
            attrs=attrs,
        ):
            self.context.cache.set(self.cache_key, df, expiry=0)

    @measure_dataset_call('dataset.dvc.convert')
    def _convert_dvc_dataset(self, dvc_ds: dvc_pandas.Dataset) -> ppl.PathsDataFrame:
        df = ppl.from_dvc_dataset(dvc_ds)
        return self._drop_reserved_columns(df)

    @staticmethod
    def _drop_reserved_columns(df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        """
        Drop per-row provenance columns, so DVC and the database yield the same frame.

        `Source` and `Comment` travel to DVC as ordinary columns, because the parquet has
        nowhere else to put them; `load_dvc_dataset` reads them back into `DataSource` and
        `DataPointComment` records. The database path therefore never surfaces them, while
        the DVC path did -- so the same dataset had two extra string columns depending on
        which source the instance was configured for, and node code that survived one
        could fail on the other.

        Only columns that are neither an index nor a metric are dropped. A dataset that
        genuinely has a dimension called `source` keeps it: `upload_new_dataset` excludes
        the reserved names from `index_columns`, so anything still in `primary_keys` under
        one of those names got there deliberately and is not provenance.
        """
        droppable = [
            col
            for col in df.columns
            if col.lower() in RESERVED_ROW_COLUMNS and col not in df.primary_keys and col not in df.metric_cols
        ]
        if not droppable:
            return df
        return df.drop(droppable)

    def dvc_source_id(self) -> str | None:
        return self.input_dataset or self.id

    def prepared_prefix_length(self) -> int:
        before_temporal, _ = self.transformation_groups_at_temporal_fill()
        return next(
            (index for index, op in enumerate(before_temporal) if not op.persistent_cache_safe),
            len(before_temporal),
        )

    def prepared_recipe(self) -> PreparedRecipe | None:
        from datasets.prepared import PREPARATION_VERSION, PreparedRecipe

        if self.context.skip_cache or not self.context.use_prepared_dataset_cache:
            return None
        source_id = self.dvc_source_id()
        if source_id is None:
            return None
        manifest = self.context.dvc_source_manifest
        source = manifest.datasets.get(source_id) if manifest is not None else None
        if source is None:
            return None
        return PreparedRecipe({
            'version': PREPARATION_VERSION,
            'source': source.model_dump(
                mode='json',
                include={'hash_algorithm', 'content_hash', 'units', 'index_columns', 'metadata'},
            ),
            'column': self.column,
            'empty_to_zero': 'empty_to_zero' in self.tags,
            'operations': [op.cache_hash_data(self.context) for op in self.transformations[: self.prepared_prefix_length()]],
            **({'qualifiers': self.qualifier_source.hash_data()} if self.qualifier_source is not None else {}),
        })

    @measure_dataset_call('dataset.prepare', capture_df_result=False)
    def _load_prepared_prefix(self) -> tuple[ppl.PathsDataFrame, int]:
        recipe = self.prepared_recipe()
        length = self.prepared_prefix_length()
        if recipe is not None:
            store = self.context.prepared_dataset_store
            if not store.prefetched:
                store.prefetch(
                    candidate
                    for node in self.context.nodes.values()
                    for dataset in node.input_dataset_instances
                    if (candidate := dataset.prepared_recipe()) is not None
                )
                store.prefetched = True
            frame = store.get(recipe)
            if frame is not None:
                return frame, length
        source = self.context.load_dvc_dataset(self.input_dataset or self.id)
        frame = self._convert_dvc_dataset(source)
        frame = self.apply_transformations(frame, self.transformations[:length], metric_column=self.column)
        if recipe is not None:
            self.context.prepared_dataset_store.put(recipe, frame)
        return frame, length

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        obj = self.cache_get()
        if obj is not None:
            return obj

        df, prefix_length = self._load_prepared_prefix()
        df = self._filter_and_process_df(df, prepared_prefix_length=prefix_length)
        df = self.after_transformations(df)
        if self.context.sample_size > 0:
            df = self.sampler.interpret(self, df)
        if self.cache_key:
            self.cache_set(df)

        return df

    def get_unit(self) -> Unit:
        if self.unit:
            return self.unit
        df = self.load_internal()
        if VALUE_COLUMN in df.columns:
            meta = df.get_meta()
            if VALUE_COLUMN not in meta.units:
                raise DatasetError(self, 'Dataset %s does not have a unit' % self.id)
            return meta.units[VALUE_COLUMN]
        raise DatasetError(self, 'Dataset %s does not have the value column' % self.id)

    def hash_data(self) -> dict[str, Any]:
        source = {
            'input_dataset': self.input_dataset,
            'dvc_id': self.input_dataset or self.id,
        }
        if self.context.dataset_repo_spec is not None:
            source['repository_url'] = self.context.dataset_repo_spec.url
            source['commit_id'] = self.context.dataset_repo_spec.commit
        return {'source': source, 'pipeline': self.pipeline_hash_data()}


@dataclass
class GenericDataset(DVCDataset):
    """Dataset that already filters for relevant columns."""

    def prepared_recipe(self) -> PreparedRecipe | None:
        # This subclass has its own preparation pipeline. Opt in only once that
        # pipeline declares the dependencies of its custom conversion stages.
        return None

    # Supported languages: Czech, Danish, English, Finnish, German, Latvian, Polish, Swedish
    characterlookup = str.maketrans(
        {
            '.': '',
            ',': '',
            ':': '',
            '-': '',
            '(': '',
            ')': '',
            ' ': '_',
            '/': '_',
            '&': 'and',
            'ä': 'a',
            'å': 'a',
            'ą': 'a',
            'á': 'a',
            'ā': 'a',
            'ć': 'c',
            'č': 'c',
            'ď': 'd',
            'ę': 'e',
            'é': 'e',
            'ě': 'e',
            'ē': 'e',
            'ģ': 'g',
            'í': 'i',
            'ī': 'i',
            'ķ': 'k',
            'ł': 'l',
            'ļ': 'l',
            'ń': 'n',
            'ň': 'n',
            'ņ': 'n',
            'ö': 'o',
            'ø': 'o',
            'ó': 'o',
            'ř': 'r',
            'ś': 's',
            'š': 's',
            'ť': 't',
            'ü': 'u',
            'ú': 'u',
            'ů': 'u',
            'ū': 'u',
            'ý': 'y',
            'ź': 'z',
            'ż': 'z',
            'ž': 'z',
            'æ': 'ae',
            'ß': 'ss',
        },
    )

    def implement_unit_col(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        """Create separate metric columns for each unique unit in the DataFrame."""
        if 'Unit' not in df.columns:
            return df

        unique_units = df['Unit'].unique()
        if len(unique_units) == 1:
            df = df.set_unit(VALUE_COLUMN, unique_units[0])
            df = df.drop('Unit')
            return df

        meta = df.get_meta()
        result = df.copy()

        new_units = meta.units.copy()
        if VALUE_COLUMN in new_units:
            del new_units[VALUE_COLUMN]

        # Create a new metric column for each unit
        for unit_str in unique_units:
            column_name = f'{VALUE_COLUMN}_{unit_str.replace("/", "_per_")}'

            # Create a filtered column with values only where Unit matches
            result = result.with_columns(
                pl.when(pl.col('Unit') == unit_str).then(pl.col(VALUE_COLUMN)).otherwise(None).alias(column_name)
            )

            # Add unit to metadata
            unit = unit_registry.parse_units(unit_str)
            new_units[column_name] = unit

        result = result.drop([VALUE_COLUMN, 'Unit'])
        new_meta = ppl.DataFrameMeta(primary_keys=meta.primary_keys, units=new_units)

        return ppl.to_ppdf(result, meta=new_meta)

    # -----------------------------------------------------------------------------------
    def convert_names_to_ids(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        context = self.context
        exset = {YEAR_COLUMN, VALUE_COLUMN, FORECAST_COLUMN, UNCERTAINTY_COLUMN, 'Unit', 'UUID'}
        exset |= {col for col in df.columns if col.startswith(f'{VALUE_COLUMN}_')}
        exset |= set(df.metric_cols)
        cols = list(set(df.columns) - exset)

        # Convert index level names from labels to IDs.
        collookup = {}
        for col in cols:
            collookup[col] = col.lower().translate(self.characterlookup)
        df = df.rename(collookup)

        for col in cols:
            if col in context.dimensions:
                df = df.with_columns(context.dimensions[col].series_to_ids_pl(df[col]))

        return df

    # -----------------------------------------------------------------------------------
    def drop_unnecessary_levels(self, df: ppl.PathsDataFrame, droplist: list[str]) -> ppl.PathsDataFrame:
        # Get all metric columns from the DataFrame's metadata
        metric_cols = list(df.get_meta().units.keys())

        # Only drop rows where all metric columns are null
        if metric_cols:
            null_condition = pl.lit(True)  # noqa: FBT003
            for col in metric_cols:
                null_condition = null_condition & pl.col(col).is_null()
            df = df.filter(~null_condition)

        # Drop filter levels and empty dimension levels.
        drops = [d for d in droplist if d in df.columns]

        for col in list(set(df.columns) - set(drops)):
            vals = df[col].unique().to_list()
            if vals in [['.'], [None]]:
                drops.append(col)

        df = df.drop(drops)
        return df

    # -----------------------------------------------------------------------------------

    @measure_dataset_call('dataset.transform')
    def _transform_data(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = self.drop_unnecessary_levels(df, droplist=['Description', 'Quantity'])
        df = self.implement_unit_col(df)
        return self.convert_names_to_ids(df)

    @measure_dataset_call('dataset.index')
    def _index_data(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        new_dims = [col for col, dtype in zip(df.columns, df.dtypes, strict=True) if dtype in [pl.Utf8(), pl.Categorical()]]
        return df.add_to_index([dim for dim in new_dims if dim not in df.dim_ids])

    def _generic_transformation_groups(self) -> tuple[list[PortTransformOp], list[PortTransformOp]]:
        """Keep temporal filling after GenericDataset has established its metric columns."""
        data_ops, temporal_ops = self.transformation_groups_at_temporal_fill()
        # GenericDataset has always interpolated and extended its inputs. That is
        # class behavior, not authored pipeline state, so it is applied here at
        # execution time instead of being written into `self.transformations`,
        # which must stay equal to the authored spec (the runtime export
        # serializes it back).
        kinds = {op.kind for op in self.transformations}
        if 'interpolate' not in kinds:
            temporal_ops = [*temporal_ops, InterpolateOp()]
        if 'extend' not in kinds:
            temporal_ops = [*temporal_ops, ExtendOp()]
        return data_ops, temporal_ops

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        # Don't call DVCDataset.load directly since it does post_process too early
        # Instead, replicate the parts we need but with different ordering

        cached_df = self.cache_get()
        if cached_df is not None:
            return cached_df

        if self.input_dataset:
            ds_id = self.input_dataset
        else:
            ds_id = self.id

        dvc_ds = self.context.load_dvc_dataset(ds_id)
        assert dvc_ds.df is not None
        df = self._convert_dvc_dataset(dvc_ds)

        data_ops, temporal_ops = self._generic_transformation_groups()
        df = self.apply_transformations(df, data_ops, metric_column=self.column)
        ppl.validate_ppdf(df)

        # Now do GenericDataset specific processing
        df = self._transform_data(df)

        # Only AFTER metric columns exist, handle interpolation
        if FORECAST_COLUMN not in df.columns:
            df = df.with_columns(pl.lit(False).alias(FORECAST_COLUMN))  # noqa: FBT003

        df = self._index_data(df)
        df = self.apply_transformations(df, temporal_ops, metric_column=self.column)

        # Finalize processing
        if self.context.sample_size > 0:
            df = self._sample(df)
        self.cache_set(df)

        return df
