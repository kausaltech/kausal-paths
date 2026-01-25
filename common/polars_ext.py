from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, cast

import numpy as np
import polars as pl
from polars import type_aliases as pl_types

import common.polars as ppl
from nodes.constants import FORECAST_COLUMN, UNCERTAINTY_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.units import unit_registry

if TYPE_CHECKING:
    import pandas as pd

    from nodes.context import Context
    from nodes.dimensions import Dimension
    from nodes.node import Node, NodeMetric
    from nodes.units import Unit


type Dimensions = dict[str, 'Dimension']
type Metrics = dict[str, 'NodeMetric']

type DF = ppl.PathsDataFrame | pl.DataFrame


@pl.api.register_dataframe_namespace('paths')
class PathsExt:
    _df: ppl.PathsDataFrame

    def __init__(self, df: DF) -> None:
        if not isinstance(df, ppl.PathsDataFrame):
            df = ppl.to_ppdf(df)
        self._df = df

        self.OPERATIONS: dict[str, Callable[..., ppl.PathsDataFrame]] = {
            'abs': self._absolute,
            'absolute': self._absolute,
            'add_missing_years': self._add_missing_years,
            'arithmetic_inverse': self._arithmetic_inverse,
            'bring_to_maximum_historical_year': self._bring_to_maximum_historical_year,
            'complement': self._complement,
            'complement_cumulative_product': self._complement_cumulative_product,
            'cumulative': self._cumulative,
            'cumulative_product': self._cumulative_product,
            'difference': self._difference,
            'drop_infs': self._drop_infs,
            'drop_nans': self._drop_nans,
            'drop_unnecessary_levels': self._drop_unnecessary_levels,
            'empty_to_zero': self._empty_to_zero,
            'exp': self._exponential,
            'expectation': self._expectation,
            'extend_both_ways': self._extend_both_ways,
            'extend_forecast_values': self._extend_forecast_values,
            'extend_to_history': self._extend_to_history,
            'extend_values': self._extend_values,
            'extrapolate': self._extrapolate,
            'forecast_only': self._forecast_only,
            'geometric_inverse': self._geometric_inverse,
            'ignore_content': self._ignore_content,
            'indifferent_history_ratio': self._indifferent_history_ratio,
            'inventory_only': self._inventory_only,
            'log': self._logarithmic,
            'make_nonnegative': self._make_nonnegative,
            'make_nonpositive': self._make_nonpositive,
            'minus': self._arithmetic_inverse,
            'ratio_to_last_historical_value': self._ratio_to_last_historical_value,
            'round_to_five': self._round_to_five_significant_digits,
            'scale_by_reference_year': self._scale_by_reference_year,
            'truncate_before_start': self._truncate_before_start,
            'truncate_beyond_end': self._truncate_beyond_end,
        }

    def to_pandas(self, meta: ppl.DataFrameMeta | None = None) -> pd.DataFrame:
        return self._df.to_pandas(meta=meta)

    def to_wide(self, meta: ppl.DataFrameMeta | None = None, only_category_names: bool = False) -> ppl.PathsDataFrame:  # noqa: C901, PLR0912, PLR0915
        """Project the DataFrame wide (dimension categories become columns) and group by year."""

        df = self._df

        if meta is None:
            meta = df.get_meta()
        dim_ids = sorted(meta.dim_ids)
        metric_cols = list(meta.units.keys())
        if not metric_cols:
            raise Exception("No metric columns in DF")

        if only_category_names and len(metric_cols) > 1:
            raise Exception("When only_category_names=True, only one metric supported")

        if only_category_names and len(dim_ids) != 1:
            raise Exception("When only_category_names=True, must have exactly one dimension")

        dim_casts = []
        for col in dim_ids + metric_cols:
            if col not in df.columns:
                raise Exception("Column %s from metadata is not present in DF" % col)
            if col in dim_ids and df.schema[col] == pl.Categorical:
                dim_casts.append(pl.col(col).cast(pl.Utf8))
        if dim_casts:
            df = df.with_columns(dim_casts)

        # Create a column '_dims' with all the categories included
        if not dim_ids:
            return df

        if YEAR_COLUMN in df.columns:
            dim_cols = [*dim_ids, YEAR_COLUMN]
        else:
            dim_cols = dim_ids
        dup = df.select(dim_cols).is_duplicated()
        if any(dup):
            print(df.filter(dup))
            raise ValueError("Dataframe has duplicate rows.")

        def format_col(dim: str) -> pl.Expr:
            if only_category_names:
                return pl.col(dim)
            return pl.format('{}:{}', pl.lit(dim), pl.col(dim))

        def format_metric(metric_col: str, col: str) -> str:
            if only_category_names:
                return '%s' % col
            return '%s@%s' % (metric_col, col)

        df = df.with_columns([
            pl.concat_list([
                format_col(dim) for dim in dim_ids
            ]).list.join('/').alias('_dims'),
        ])
        mdf = None
        units = {}
        index_cols = [YEAR_COLUMN]
        if FORECAST_COLUMN in df.columns:
            index_cols.append(FORECAST_COLUMN)
        for metric_col in metric_cols:
            tdf = df.pivot(on='_dims', index=index_cols, values=metric_col)  # noqa: PD010
            cols = [col for col in tdf.columns if col not in index_cols]
            metric_unit = meta.units.get(metric_col)
            if metric_unit is not None:
                for col in cols:
                    units[format_metric(metric_col, col)] = metric_unit
            tdf = ppl.to_ppdf(
                df=tdf.rename({col: format_metric(metric_col, col) for col in cols}),
                meta=ppl.DataFrameMeta(primary_keys=[YEAR_COLUMN], units=units),
            )
            if tdf.paths.index_has_duplicates():
                tdf = tdf.paths.sum_over_dims()

            if mdf is None:
                mdf = tdf
            else:
                if FORECAST_COLUMN in index_cols:
                    tdf = tdf.drop(FORECAST_COLUMN)
                joined = mdf.join(tdf, on=YEAR_COLUMN)
                mdf = ppl.to_ppdf(joined, meta=mdf.get_meta())
        assert mdf is not None
        mdf = mdf.sort(YEAR_COLUMN)
        meta2 = ppl.DataFrameMeta(
            units=units,
            primary_keys=[YEAR_COLUMN],
        )
        return ppl.PathsDataFrame._from_pydf(
            mdf._df,
            meta=meta2,
        )

    def to_narrow(self, assign_dimension: str | None = None, assign_metric: str | None = None) -> ppl.PathsDataFrame:  # noqa: C901, PLR0912
        df: ppl.PathsDataFrame | pl.DataFrame = self._df
        assert isinstance(df, ppl.PathsDataFrame)
        id_cols = [YEAR_COLUMN]
        if FORECAST_COLUMN in df.columns:
            id_cols.append(FORECAST_COLUMN)

        widened_cols = []
        renames = {}
        for col in df.columns:
            if col in id_cols:
                continue
            new_col = col
            if assign_dimension:
                new_col = '%s:%s' % (assign_dimension, new_col)
            if assign_metric:
                new_col = '%s@%s' % (assign_metric, new_col)
            if '@' in new_col:
                widened_cols.append(new_col)
            if col != new_col:
                renames[col] = new_col

        if not len(widened_cols):  # noqa: PLC1802
            return df

        if renames:
            df = df.copy().rename(renames)

        meta = df.get_meta()
        units: dict[str, Unit] = {}
        primary_keys = [YEAR_COLUMN]
        for col in widened_cols:
            metric, dims = col.split('@')
            unit = meta.units[col]
            if metric in units:
                if units[metric] != unit:
                    raise Exception('Unit mismatch in metric %s' % metric)
            else:
                units[metric] = unit
            for dim_parts in dims.split('/'):
                dim_id, _cat_id = dim_parts.split(':')
                if dim_id not in primary_keys:
                    primary_keys.append(dim_id)

        meta.units = units
        meta.primary_keys = primary_keys

        tdf = df.unpivot(index=id_cols).with_columns([
            pl.col('variable').str.split('@').alias('_tmp'),
        ]).with_columns([
            pl.col('_tmp').list.first().alias('Metric'),
            pl.col('_tmp').list.last().str.split('/').alias('_dims'),
        ])
        df = ppl.to_ppdf(tdf)
        first = df['_dims'][0]
        dim_ids = [x.split(':')[0] for x in first]
        dim_cols = [pl.col('_dims').list.get(idx).str.split(':').list.get(1).alias(col) for idx, col in enumerate(dim_ids)]
        df = df.with_columns(dim_cols)
        df = df.pivot(on='Metric', values='value', index=[*id_cols, *dim_ids])  # noqa: PD010
        df = df.with_columns([pl.col(dim).cast(pl.Categorical) for dim in dim_ids])
        return ppl.to_ppdf(df, meta=meta)

    def make_forecast_rows(self, end_year: int) -> ppl.PathsDataFrame:
        pdf = self._df
        if isinstance(pdf, ppl.PathsDataFrame):
            meta = pdf.get_meta()
            df = pl.DataFrame(pdf._df)
        else:
            meta = None
            df = pdf
        y = df[YEAR_COLUMN]
        if y.n_unique() != len(y):
            raise Exception("DataFrame has duplicated years")

        if FORECAST_COLUMN not in df.columns:
            last_hist_year = y.max()
        else:
            last_hist_year = df.filter(~pl.col(FORECAST_COLUMN))[YEAR_COLUMN].max()
            if last_hist_year is None:
                last_hist_year = df[YEAR_COLUMN].max()
        assert isinstance(last_hist_year, int)
        if last_hist_year >= end_year:
            return ppl.to_ppdf(df, meta=meta)
        years = pl.DataFrame(data=range(last_hist_year + 1, end_year + 1), schema=[YEAR_COLUMN])
        if len(years):
            df = df.join(years, on=YEAR_COLUMN, how='outer', coalesce=True).sort(YEAR_COLUMN)
            df = df.with_columns([
                pl.when(pl.col(YEAR_COLUMN) > last_hist_year).then(pl.lit(value=True))\
                    .otherwise(pl.col(FORECAST_COLUMN)).alias(FORECAST_COLUMN),
            ])
        return ppl.to_ppdf(df, meta=meta)

    def nafill_pad(self) -> ppl.PathsDataFrame:
        """
        Fill N/A values by propagating the last valid observation forward.

        Requires a DF in wide format (indexed by year).
        """

        df = self._df
        y = df[YEAR_COLUMN]
        if y.n_unique() != len(y):
            raise Exception("DataFrame has duplicated years")

        meta = df.get_meta()
        zdf = df.fill_null(strategy='forward')
        return ppl.to_ppdf(zdf, meta=meta)

    def sum_over_dims(self, dims: str | list[str] | None = None) -> ppl.PathsDataFrame:
        df = self._df
        meta = df.get_meta()
        if FORECAST_COLUMN in df.columns:
            fc = [pl.any(FORECAST_COLUMN)]
        else:
            fc = []

        if dims is None:
            dims = meta.dim_ids
        elif isinstance(dims, str):
            dims =  [dims]
        remaining_keys = list(meta.primary_keys)
        for dim in dims:
            remaining_keys.remove(dim)

        known_cols = set(meta.primary_keys) | set(meta.metric_cols) | {FORECAST_COLUMN, YEAR_COLUMN}
        sum_cols = list(meta.metric_cols)
        for col, dt in df.schema.items():
            if col in known_cols:
                continue
            if dt.is_numeric():
                sum_cols.append(col)

        zdf = df.group_by(remaining_keys).agg([
            *[pl.sum(col).alias(col) for col in sum_cols],
            *fc,
        ]).sort(remaining_keys)
        return ppl.to_ppdf(zdf, meta=meta)

    def get_category_mismatch(
        self,
        other: ppl.PathsDataFrame,
        output: ppl.PathsDataFrame,
    ) -> str: # dict[str, dict[str, list[str]]]:
        """
        Get categories that were dropped or added during a join operation.

        Args:
            other: The right DataFrame that was joined with self
            output: The resulting DataFrame from the join

        Returns:
            A dictionary mapping dimension IDs to a dict with keys:
            - 'units': [unit_left, unit_right] for the (first) metric column
            - 'dropped_left': categories in self but not in output
            - 'dropped_right': categories in other but not in output
            - 'added_from_right': categories in output and other but not in self
            - 'added_from_left': categories in output and self but not in other

        """
        sdf = self._df
        mismatch: dict[str, dict[str, list[str]]] = {}

        # Get all dimension columns from all three DataFrames
        all_dim_ids = set(sdf.dim_ids) | set(other.dim_ids) | set(output.dim_ids)

        for dim_id in all_dim_ids:
            # Get categories from input DataFrames
            self_cats: set[str] = set()
            other_cats: set[str] = set()

            if dim_id in sdf.columns:
                # Note: unique().to_list() is necessary - cannot do set(sdf[dim_id]) directly
                # FIXME There are Nones in data, but can we allow that?
                self_cats = set(sdf[dim_id].unique().drop_nulls().to_list())
            if dim_id in other.columns:
                other_cats = set(other[dim_id].unique().drop_nulls().to_list())

            # Get categories from output DataFrame
            output_cats: set[str] = set()
            if dim_id in output.columns:
                output_cats = set(output[dim_id].unique().drop_nulls().to_list())

            # Categories dropped from or added to left or right specifically
            dropped_left = sorted(self_cats - output_cats)
            dropped_right = sorted(other_cats - output_cats)
            added_from_right = sorted((output_cats & other_cats) - self_cats)
            added_from_left = sorted((output_cats & self_cats) - other_cats)

            # Only include dimension if there are any mismatches
            if dropped_left or dropped_right or added_from_right or added_from_left:
                unit_left = unit_right = '-'
                metrics_left = sdf.metric_cols
                if metrics_left:
                    unit_left = str(sdf.get_unit(metrics_left[0]))
                metrics_right = other.metric_cols
                if metrics_right:
                    unit_right = str(other.get_unit(metrics_right[0]))
                mismatch[dim_id] = {
                    'units': [unit_left, unit_right],
                    'dropped_left': dropped_left,
                    'dropped_right': dropped_right,
                    'added_from_right': added_from_right,
                    'added_from_left': added_from_left,
                }

        if mismatch:
            import json
            # TODO Serialize dict to string for now, since _explanation expects str
            # ensure_ascii=False preserves Unicode characters like · (middle dot)
            mismatch_str = json.dumps(mismatch, ensure_ascii=False)
        else:
            mismatch_str = ''
        return mismatch_str

    def join_over_index(  # noqa: C901, PLR0912
        self,
        other: ppl.PathsDataFrame,
        how: Literal['left', 'outer', 'inner'] = 'left',
        index_from: Literal['left', 'right', 'union'] = 'left',
    ) -> ppl.PathsDataFrame:
        sdf = self._df
        sm = sdf.get_meta()
        om = other.get_meta()
        # Join on subset of keys
        join_on = list(set(sm.primary_keys) & set(om.primary_keys))
        if not len(join_on):  # noqa: PLC1802
            if len(other) == 1:  # A single value copied to all rows
                #df = sdf.with_columns(other).paths._df
                raise Exception("invalid access")
            raise ValueError("No shared primary keys between joined DFs")

        for col in join_on:
            sdt = sdf[col].dtype
            if sdt != other[col].dtype:
                other = other.with_columns([pl.col(col).cast(sdt)])
        pl_how: pl_types.JoinStrategy = how
        if how == 'outer':
            pl_how = 'outer_coalesce'
        df = sdf.join(other, on=join_on, how=pl_how)
        fc_right = '%s_right' % FORECAST_COLUMN
        meta = sm.copy()
        if FORECAST_COLUMN in df.columns and fc_right in df.columns:
            df = df.with_columns([
                pl.col(FORECAST_COLUMN).fill_null(value=False) | pl.col(fc_right).fill_null(value=False),
            ])
            df = df.drop(fc_right)
        for col in om.metric_cols:
            col_right = '%s_right' % col
            if col_right in df.columns:
                meta.units[col_right] = om.units[col]
            elif col in df.columns:
                meta.units[col] = om.units[col]

        if index_from == 'left':
            pass
        elif index_from == 'right':
            meta.primary_keys = om.primary_keys
        elif index_from == 'union':
            meta.primary_keys = list(set(sm.primary_keys) | set(om.primary_keys))
        else:
            raise ValueError("Invalid value for 'index_from'")

        out = ppl.to_ppdf(df, meta=meta)

        cat_mismatch = self.get_category_mismatch(other, out)
        if cat_mismatch:
            out._explanation.append(cat_mismatch)
        if out.paths.index_has_duplicates():
            print(out)
            raise ValueError("Resulting DF has duplicated rows")
        return out

    def duplicated_index_rows(self) -> pl.DataFrame:
        df = self._df
        assert df._primary_keys
        ldf = df.lazy()
        dupes = (
            ldf.group_by(df._primary_keys)
            .agg(pl.count())
            .filter(pl.col('count') > 1)
            .collect()
        )
        return dupes

    def index_has_duplicates(self) -> bool:
        df = self._df
        if not df._primary_keys:
            return False
        ldf = df.lazy()
        dupes = ldf.group_by(df._primary_keys).agg(pl.count()).filter(pl.col('count') > 1).limit(1).collect()
        return len(dupes) > 0

    def cast_index_to_str(self) -> ppl.PathsDataFrame:
        df = self._df
        cast_exprs: list[pl.Expr] = []
        for col in df.dim_ids:
            if df.schema[col] != pl.Utf8:
                cast_exprs.append(pl.col(col).cast(pl.Utf8))  # noqa: PERF401
        if cast_exprs:
            df = df.with_columns(cast_exprs)
        return df

    def add_with_dims(
            self,
            odf: ppl.PathsDataFrame,
            how: Literal['left', 'inner', 'outer'] = 'outer'
        ) -> ppl.PathsDataFrame:
        """Add two PathsDataFrames with dimension awareness."""
        df = self._df
        if len(df.metric_cols) != 1 or len(odf.metric_cols) != 1:
            raise Exception("Currently adding only one metric column is supported.")
        val_col = df.metric_cols[0]

        if set(df.dim_ids) != set(odf.dim_ids):
            raise ValueError(f"Dimensions must match for addition: {df.dim_ids} vs {odf.dim_ids}.")

        # Ensure same unit for addition
        output_unit = df.get_unit(val_col)
        odf = odf.ensure_unit(val_col, output_unit)

        # For addition: how='outer', index_from='left' because we want all rows but not new dimensions
        jdf = df.paths.join_over_index(
            odf,
            how=how,
            index_from='left'
        )

        jdf = jdf.with_columns([
            (pl.col(val_col).fill_null(0.0) + pl.col(f"{val_col}_right").fill_null(0.0)).alias(val_col)
        ])

        cols = [YEAR_COLUMN, FORECAST_COLUMN, val_col] + df.dim_ids
        jdf = jdf.select([col for col in cols if col in jdf.columns])
        mismatch = df.paths.get_category_mismatch(odf, jdf)
        if mismatch:
            jdf._explanation.append(mismatch)
        return jdf

    def multiply_with_dims(
            self,
            odf: ppl.PathsDataFrame,
            how: Literal['left', 'inner', 'outer'] = 'inner'
        ) -> ppl.PathsDataFrame:
        """Multiply two PathsDataFrames, handling dimensions and units properly."""
        df = self._df
        if len(df.metric_cols) != 1 or len(odf.metric_cols) != 1:
            raise Exception("Currently multiplying only one metric column is supported.")
        val_col = df.metric_cols[0]

        left_unit = df.get_unit(val_col)
        right_unit = odf.get_unit(val_col)
        output_unit = left_unit * right_unit
        meta = df.get_meta()
        all_dims = list(set(df.dim_ids) | set(odf.dim_ids)) + [YEAR_COLUMN]

        # For multiplication: how='inner', index_from='union' to ensure both factors and include all dimensions
        jdf = df.paths.join_over_index(
            odf,
            how=how,
            index_from='union'
        )

        jdf = jdf.with_columns([
            (pl.col(val_col) * pl.col(f"{val_col}_right")).alias(val_col) # null factor must give null
        ])

        new_units = meta.units.copy()
        new_units[val_col] = output_unit

        cols = [FORECAST_COLUMN, val_col] + all_dims
        jdf = jdf.select([col for col in cols if col in jdf.columns])

        new_meta = ppl.DataFrameMeta(primary_keys=all_dims, units=new_units)
        out = ppl.to_ppdf(jdf, meta=new_meta)

        cat_mismatch = df.paths.get_category_mismatch(odf, out)
        if cat_mismatch:
            out._explanation.append(cat_mismatch)
        return out

    def divide_with_dims(
            self,
            odf: ppl.PathsDataFrame,
            how: Literal['left', 'inner', 'outer'] = 'inner'
        ) -> ppl.PathsDataFrame:
        """Divide two PathsDataFrames, handling dimensions and units properly."""
        df = self._df
        if len(df.metric_cols) != 1 or len(odf.metric_cols) != 1:
            raise Exception("Currently dividing only one metric column is supported.")
        val_col = df.metric_cols[0]

        left_unit = df.get_unit(val_col)
        right_unit = odf.get_unit(val_col)
        output_unit = cast("Unit", left_unit / right_unit)
        meta = df.get_meta()
        all_dims = list(set(df.dim_ids) | set(odf.dim_ids)) + [YEAR_COLUMN]

        jdf = df.paths.join_over_index(
            odf,
            how=how,
            index_from='union'
        )

        jdf = jdf.with_columns([
            (pl.col(val_col) / pl.col(f"{val_col}_right")).alias(val_col)
        ])

        new_units = meta.units.copy()
        new_units[val_col] = output_unit

        cols = [FORECAST_COLUMN, val_col] + all_dims
        jdf = jdf.select([col for col in cols if col in jdf.columns])

        new_meta = ppl.DataFrameMeta(primary_keys=all_dims, units=new_units)
        out = ppl.to_ppdf(jdf, meta=new_meta)

        cat_mismatch = df.paths.get_category_mismatch(odf, out)
        if cat_mismatch:
            out._explanation.append(cat_mismatch)
        return out

    def add_df(self, odf: ppl.PathsDataFrame, how: Literal['left', 'outer'] = 'left') -> ppl.PathsDataFrame:
        df = self._df
        if len(self._df.metric_cols) != 1 or len(odf.metric_cols) != 1:
            raise Exception("Currently adding only one metric column is supported")
        out_col = df.metric_cols[0]
        input_col = odf.metric_cols[0]
        odf = odf.ensure_unit(input_col, df.get_unit(out_col)).rename({input_col: '_Right'})
        df = df.paths.join_over_index(odf, how=how)
        expr = (pl.col(out_col).fill_null(0) + pl.col('_Right').fill_null(0)).alias(out_col)
        df = df.with_columns(expr).drop('_Right')
        return df

    # TODO Streamline add_with_dims, multiply_with_dims, add_df, and coalesce_df
    def coalesce_df(
        self,
        odf: ppl.PathsDataFrame,
        how: Literal['left', 'outer'] = 'outer',
        debug: bool = False,
        id: str = ''
    ) -> ppl.PathsDataFrame:
        df = self._df
        if len(self._df.metric_cols) != 1 or len(odf.metric_cols) != 1:
            raise Exception("Currently coalescing only one metric column is supported")
        out_col = df.metric_cols[0]
        input_col = odf.metric_cols[0]
        odf = odf.ensure_unit(input_col, df.get_unit(out_col)).rename({input_col: '_Right'})
        df = df.paths.join_over_index(odf, how=how)
        if debug:
            print(f"In node {id}, column '{out_col}' is prioritised over '_Right' if available.")
            print(df)
        expr = pl.coalesce([pl.col(out_col), pl.col('_Right')]).alias(out_col)
        df = df.with_columns(expr).drop('_Right')
        return df

    def compare_df( # Based on add_with_dims
            self,
            odf: ppl.PathsDataFrame,
            how: Literal['left', 'inner', 'outer'] = 'outer', # TODO Should this rather be inner?
            op: Literal['eq', 'ne', 'gt', 'ge', 'lt', 'le'] = 'eq'
        ) -> ppl.PathsDataFrame:
        """Add two PathsDataFrames with dimension awareness."""
        df = self._df
        if len(df.metric_cols) != 1 or len(odf.metric_cols) != 1:
            raise Exception("Currently adding only one metric column is supported.")
        val_col = df.metric_cols[0]

        if set(df.dim_ids) != set(odf.dim_ids):
            raise ValueError(f"Dimensions must match for comparison: {df.dim_ids} vs {odf.dim_ids}.")

        # Ensure same unit for comparison
        output_unit = df.get_unit(val_col)
        odf = odf.ensure_unit(val_col, output_unit)

        # For comparison: how='outer', index_from='left' because we want all rows but not new dimensions
        jdf = df.paths.join_over_index(
            odf,
            how=how,
            index_from='left'
        )
        right_col = pl.col(f"{val_col}_right")
        opfunc = {
            'eq': pl.col(val_col).eq(right_col),
            'ne': pl.col(val_col).ne(right_col),
            'gt': pl.col(val_col).gt(right_col),
            'ge': pl.col(val_col).ge(right_col),
            'lt': pl.col(val_col).lt(right_col),
            'le': pl.col(val_col).le(right_col),
        }
        if op not in opfunc:
            raise ValueError(f"Invalid operation: {op}")
        expr = opfunc[op]
        jdf = jdf.with_columns(expr.cast(pl.Float64).alias(val_col))

        cols = [YEAR_COLUMN, FORECAST_COLUMN, val_col] + df.dim_ids
        jdf = jdf.select([col for col in cols if col in jdf.columns])

        # Comparison results are boolean, so set unit to dimensionless
        jdf = jdf.set_unit(val_col, 'dimensionless', force=True)

        return jdf

    def concat_vertical(self, other: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = self._df
        df_cols = set(df.columns)
        other_cols = set(other.columns)
        if df_cols != other_cols:
            raise Exception("Mismatching columns: %s vs. %s" % (df_cols, other_cols))
        cast_exprs = []
        for col in df_cols:
            if df.schema[col] != other.schema[col]:
                cast_exprs.append(pl.col(col).cast(df.schema[col], strict=True))
            unit = df._units.get(col)
            if unit:
                other_unit = other._units.get(col)
                if other_unit != unit:
                    raise Exception("Unit mismatch in column '%s': %s vs. %s" % (col, unit, other_unit))

        if cast_exprs:
            other = other.with_columns(cast_exprs)

        meta = df.get_meta()
        zdf = pl.concat([df, other], how='vertical')
        df = ppl.to_ppdf(zdf, meta=meta)

        if df.paths.index_has_duplicates():
            raise Exception("Concatenation resulted in duplicated index rows")

        return df

    def add_sum_column(self, metric_col: str, output_col: str, over_dims: list[str] | None = None) -> ppl.PathsDataFrame:
        df = self._df
        if over_dims is None:
            over_dims = df.dim_ids
        sdf = df.select_metrics([metric_col]).paths.sum_over_dims(over_dims)
        sdf = sdf.rename({metric_col: output_col})
        df = df.paths.join_over_index(sdf)
        return df

    def calculate_shares(self, metric_col: str, output_col: str, over_dims: list[str] | None = None) -> ppl.PathsDataFrame:
        df = self._df
        df = df.paths.add_sum_column(metric_col, '_Sum', over_dims=over_dims)
        df = df.divide_cols([metric_col, '_Sum'], output_col)
        df = df.drop('_Sum')
        return df

    def print_year(self, year: int | list[int]):
        if not isinstance(year, list | tuple):
            year = [year]
        df = self._df.filter(pl.col(YEAR_COLUMN).is_in(year))
        print(df)

    def get_last_historical_year(self) -> int | None:
        df = self._df
        max_year = df.filter(~pl.col(FORECAST_COLUMN))[YEAR_COLUMN].max()
        if not max_year:
            return None
        assert isinstance(max_year, int)
        return max_year

# ----------------- Standard PathsDataFrame unary operations with only node parameter

    def _absolute(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.with_columns(pl.col(VALUE_COLUMN).abs().alias(VALUE_COLUMN))

    # Copied from gpc.DatasetNode
    def _add_missing_years(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        # Add forecast column if needed.
        if FORECAST_COLUMN not in df.columns:
            df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))

        # Add missing years and interpolate missing values.
        df = df.paths.to_wide()
        yearrange = range(df[YEAR_COLUMN].min(), (df[YEAR_COLUMN].max() + 1))  # type: ignore
        nullcount = df.null_count().sum_horizontal()[0]

        if (len(df[YEAR_COLUMN].unique()) < len(yearrange)) | (nullcount > 0):
            yeardf = ppl.PathsDataFrame({YEAR_COLUMN: yearrange})
            yeardf._units = {}
            yeardf._primary_keys = [YEAR_COLUMN]

            df = df.paths.join_over_index(yeardf, how='outer')
            for col in list(set(df.columns) - {YEAR_COLUMN, FORECAST_COLUMN}):
                df = df.with_columns(pl.col(col).interpolate())
                df = df.with_columns(pl.col(col).fill_null(strategy='backward'))

            df = df.with_columns(pl.col(FORECAST_COLUMN).fill_null(strategy='backward'))

        df = df.paths.to_narrow()
        return df

    def _arithmetic_inverse(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.multiply_quantity(VALUE_COLUMN, unit_registry('-1 * dimensionless'))

    def _bring_to_maximum_historical_year(self, df: ppl.PathsDataFrame, context: Context) -> ppl.PathsDataFrame:
        max_year = context.instance.maximum_historical_year
        if max_year is None:
            return df
        is_forecast = False
        df = df.with_columns(
            pl.when(pl.col(YEAR_COLUMN) <= max_year)
            .then(pl.lit(is_forecast))
            .otherwise(pl.col(FORECAST_COLUMN))
            .alias(FORECAST_COLUMN)
        )
        return df

    def _complement(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        u = df.get_unit(VALUE_COLUMN)
        if not u.is_compatible_with('dimensionless'):
            raise ValueError(
                f"The unit is {u} but it must be compatible with dimensionless for taking complement."
            )
        # if node.quantity not in ['fraction', 'probability']:
        #     logger.warning(
        #         f"The quantity for taking complement should be fraction or probability. Are you operating with node {node.id}?"
        #     )
        df = df.ensure_unit(VALUE_COLUMN, unit='dimensionless')
        return df.with_columns((pl.lit(1.0) - pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))

    def _complement_cumulative_product(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.cumprod(VALUE_COLUMN, complement=True)

    def _cumulative(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.cumulate(VALUE_COLUMN)

    def _cumulative_product(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.cumprod(VALUE_COLUMN)

    def _difference(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.diff(VALUE_COLUMN)

    def _drop_infs(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        """Drop Inf cells in long format."""
        assert isinstance(df, ppl.PathsDataFrame)
        return df.filter(pl.col(VALUE_COLUMN).is_finite())

    def _drop_nans(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        """Drop NaN cells in long format."""
        assert isinstance(df, ppl.PathsDataFrame)
        return df.filter(pl.col(VALUE_COLUMN).is_not_nan())

    def _drop_unnecessary_levels(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        """Drop empty dimensions."""
        metric_cols = list(df.get_meta().units.keys())

        # Only drop rows where all metric columns are null
        if metric_cols:
            null_condition = pl.lit(True)  # noqa: FBT003
            for col in metric_cols:
                null_condition = null_condition & pl.col(col).is_null()
            df = df.filter(~null_condition)

        null_cols = [col for col in df.columns if df[col].null_count() == len(df)]
        df = df.drop(null_cols)
        return df

    def _empty_to_zero(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.with_columns(
            pl.when(pl.col(VALUE_COLUMN).is_nan())
            .then(pl.lit(0.0))
            .otherwise(pl.col(VALUE_COLUMN))
            .alias(VALUE_COLUMN),
        )

    def _exponential(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')
        return df.with_columns(pl.col(VALUE_COLUMN).exp().alias(VALUE_COLUMN))

    def _expectation(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        if UNCERTAINTY_COLUMN in df.columns:
            meta = df.get_meta()
            cols = [col for col in df.primary_keys if col != UNCERTAINTY_COLUMN]
            dfp = df.group_by(cols, maintain_order=True).agg([
                pl.col(VALUE_COLUMN).mean().alias(VALUE_COLUMN),
                pl.col(FORECAST_COLUMN).any().alias(FORECAST_COLUMN)
            ])
            dfp = dfp.with_columns(pl.lit('expectation').alias(UNCERTAINTY_COLUMN))
            df = ppl.to_ppdf(dfp, meta)
        return df

    def _extend_both_ways(self, df: ppl.PathsDataFrame, context: Context) -> ppl.PathsDataFrame:
        out = self._extend_to_history(df, context)
        out = self._extend_forecast_values(out, context)
        return self._bring_to_maximum_historical_year(out, context)

    def _extend_forecast_values(self, df: ppl.PathsDataFrame, context: Context) -> ppl.PathsDataFrame:
        from nodes.calc import extend_last_forecast_value_pl
        end_year = context.instance.model_end_year
        return extend_last_forecast_value_pl(df, end_year)

    def _extend_to_history(self, df: ppl.PathsDataFrame, context: Context) -> ppl.PathsDataFrame:
        from nodes.calc import extend_to_history_pl
        start_year = context.instance.minimum_historical_year
        return extend_to_history_pl(df, start_year)

    def _extend_values(self, df: ppl.PathsDataFrame, context: Context) -> ppl.PathsDataFrame:
        from nodes.calc import extend_last_historical_value_pl
        end_year = context.instance.model_end_year
        return extend_last_historical_value_pl(df, end_year)

    def _extrapolate(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        """Replace NaNs and Nulls by extrapolating from existing values in wide format."""
        df = df.paths.to_wide()
        df = df.with_columns([
            pl.col(col)
            .map_elements(lambda x: None if (x is not None and np.isnan(x)) else x, return_dtype=pl.Float64)
            .interpolate(method='linear')
            .forward_fill()
            .backward_fill()
            .alias(col)
            for col in df.columns if col in df.metric_cols
        ])
        df = df.select([
            col for col in df.columns
            if df.select(pl.col(col).is_not_null().any()).item()
        ])
        df = df.paths.to_narrow()

        return df

    def _forecast_only(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.filter(pl.col(FORECAST_COLUMN))

    def _geometric_inverse(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.divide_quantity(VALUE_COLUMN, unit_registry('1 * dimensionless'))

    # FIXME Current version requires output metric of the target node. Use baskets instead.
    def _ignore_content(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        no_effect_value = getattr(self, 'no_effect_value', 0.0)
        df = df.with_columns(pl.lit(no_effect_value).alias(VALUE_COLUMN))
        m = target_node.get_default_output_metric()
        return df.set_unit(VALUE_COLUMN, m.unit, force=True)

    def _indifferent_history_ratio(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.with_columns(
            pl.when(pl.col(FORECAST_COLUMN))
            .then(pl.col(VALUE_COLUMN))
            .otherwise(pl.lit(1.0)).alias(VALUE_COLUMN)
        )

    def _inventory_only(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        df = df.with_columns(  # TODO A non-elegant way to ensure there is at least one historical row.
            pl.when(pl.col(FORECAST_COLUMN) & (pl.count() == 1))
            .then(pl.lit(value=False))
            .otherwise(pl.col(FORECAST_COLUMN))
            .alias(FORECAST_COLUMN),
        )
        return df.filter(~pl.col(FORECAST_COLUMN))

    def _logarithmic(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')
        return df.with_columns(pl.col(VALUE_COLUMN).log().alias(VALUE_COLUMN))

    def _make_nonnegative(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.with_columns(pl.max_horizontal(VALUE_COLUMN, 0.0))

    def _make_nonpositive(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        return df.with_columns(pl.min_horizontal(VALUE_COLUMN, 0.0))

    def _ratio_to_last_historical_value(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        year = cast('int', df.filter(~df[FORECAST_COLUMN])[YEAR_COLUMN].max())
        return self._scale_by_reference_year(df, year)

    def _round_to_five_significant_digits(self, df: ppl.PathsDataFrame, _context: Context) -> ppl.PathsDataFrame:
        """Round values to 5 significant digits rather than 5 decimal places. Zero and NaN have special handling."""
        n_significant = 5
        val_col = pl.col(VALUE_COLUMN)
        order = val_col.abs().log10().floor()
        power = n_significant - 1 - order
        multiplier = pl.lit(10.0) ** power

        rounded = (
            pl.when((val_col == 0.0) | val_col.is_nan())
            .then(val_col)
            .otherwise((val_col * multiplier).round() / multiplier)
        )

        return df.with_columns(rounded.alias(VALUE_COLUMN))

    # FIXME: This is a duplicate of the method in SimpleNode and is not a standard operation.
    def _scale_by_reference_year(self, df: ppl.PathsDataFrame, year: int | None = None) -> ppl.PathsDataFrame:
        if not year:
            return df
        if len(df.dim_ids) == 0:
            reference = df.filter(pl.col(YEAR_COLUMN).eq(year))[VALUE_COLUMN][0]
            df = df.with_columns((pl.col(VALUE_COLUMN) / pl.lit(reference)).alias(VALUE_COLUMN))
        else:
            meta = df.get_meta()
            reference = df.filter(pl.col(YEAR_COLUMN).eq(year))
            zdf = df.join(reference, on=df.dim_ids)
            zdf = zdf.with_columns((pl.col(VALUE_COLUMN) / pl.col(VALUE_COLUMN + '_right')).alias(VALUE_COLUMN))
            zdf = zdf.drop([VALUE_COLUMN + '_right', FORECAST_COLUMN + '_right', YEAR_COLUMN + '_right'])
            df = ppl.to_ppdf(zdf, meta=meta)

        df = df.clear_unit(VALUE_COLUMN)
        df = df.set_unit(VALUE_COLUMN, 'dimensionless')
        return df

    def _truncate_before_start(self, df: ppl.PathsDataFrame, context: Context) -> ppl.PathsDataFrame:
        baseline_year = context.instance.reference_year
        return df.filter(pl.col(YEAR_COLUMN).ge(baseline_year))

    def _truncate_beyond_end(self, df: ppl.PathsDataFrame, context: Context) -> ppl.PathsDataFrame:
        end_year = context.instance.model_end_year
        return df.filter(pl.col(YEAR_COLUMN).le(end_year))
