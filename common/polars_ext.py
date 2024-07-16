from __future__ import annotations

from typing import Literal, TypeAlias, TYPE_CHECKING, Union

import pandas as pd

import polars as pl
from polars import type_aliases as pl_types
import common.polars as ppl
from nodes.units import Unit
from nodes.constants import YEAR_COLUMN, FORECAST_COLUMN


if TYPE_CHECKING:
    from nodes.dimensions import Dimension
    from nodes.node import NodeMetric


Dimensions: TypeAlias = dict[str, 'Dimension']
Metrics: TypeAlias = dict[str, 'NodeMetric']

DF: TypeAlias = Union[ppl.PathsDataFrame, pl.DataFrame]


@pl.api.register_dataframe_namespace('paths')
class PathsExt:
    _df: ppl.PathsDataFrame

    def __init__(self, df: DF) -> None:
        if not isinstance(df, ppl.PathsDataFrame):
            df = ppl.to_ppdf(df)
        self._df = df

    def to_pandas(self, meta: ppl.DataFrameMeta | None = None) -> pd.DataFrame:
        return self._df.to_pandas(meta=meta)

    def to_wide(self, meta: ppl.DataFrameMeta | None = None, only_category_names: bool = False) -> ppl.PathsDataFrame:
        """Project the DataFrame wide (dimension categories become columns) and group by year."""

        df = self._df

        if meta is None:
            meta = df.get_meta()
        dim_ids = list(sorted(meta.dim_ids))
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
            if col in dim_ids:
                if df.schema[col] == pl.Categorical:
                    dim_casts.append(pl.col(col).cast(pl.Utf8))
        if dim_casts:
            df = df.with_columns(dim_casts)

        # Create a column '_dims' with all the categories included
        if not dim_ids:
            return df

        def format_col(dim: str):
            if only_category_names:
                return pl.col(dim)
            else:
                return pl.format('{}:{}', pl.lit(dim), pl.col(dim))

        def format_metric(metric_col: str, col: str):
            if only_category_names:
                return '%s' % col
            else:
                return '%s@%s' % (metric_col, col)

        df = df.with_columns([
            pl.concat_list([
                format_col(dim) for dim in dim_ids
            ]).list.join('/').alias('_dims')
        ])
        mdf = None
        units = {}
        index_cols = [YEAR_COLUMN]
        if FORECAST_COLUMN in df.columns:
            index_cols.append(FORECAST_COLUMN)
        for metric_col in metric_cols:
            tdf = df.pivot(index=index_cols, columns='_dims', values=metric_col)
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
                mdf = mdf.join(tdf, on=YEAR_COLUMN)
        assert mdf is not None
        mdf = mdf.sort(YEAR_COLUMN)
        return ppl.PathsDataFrame._from_pydf(
            mdf._df,
            meta=ppl.DataFrameMeta(units=units, primary_keys=[YEAR_COLUMN])
        )

    def to_narrow(self, assign_dimension: str | None = None, assign_metric: str | None = None) -> ppl.PathsDataFrame:
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

        if not len(widened_cols):
            return df  # type: ignore

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
                dim_id, cat_id = dim_parts.split(':')
                if dim_id not in primary_keys:
                    primary_keys.append(dim_id)

        meta.units = units
        meta.primary_keys = primary_keys

        tdf = df.melt(id_vars=id_cols).with_columns([
            pl.col('variable').str.split('@').alias('_tmp')
        ]).with_columns([
            pl.col('_tmp').list.first().alias('Metric'),
            pl.col('_tmp').list.last().str.split('/').alias('_dims'),
        ])
        df = ppl.to_ppdf(tdf)
        first = df['_dims'][0]
        dim_ids = [x.split(':')[0] for x in first]
        dim_cols = [pl.col('_dims').list.get(idx).str.split(':').list.get(1).alias(col) for idx, col in enumerate(dim_ids)]
        df = df.with_columns(dim_cols)
        df = df.pivot(values='value', index=[*id_cols, *dim_ids], columns='Metric')
        df = df.with_columns([pl.col(dim).cast(pl.Categorical) for dim in dim_ids])
        return ppl.to_ppdf(df, meta=meta)

    def make_forecast_rows(self, end_year: int) -> ppl.PathsDataFrame:
        df: DF = self._df
        if isinstance(df, ppl.PathsDataFrame):
            meta = df.get_meta()
        else:
            meta = None
        y = df[YEAR_COLUMN]
        if y.n_unique() != len(y):
            raise Exception("DataFrame has duplicated years")

        if FORECAST_COLUMN not in df.columns:
            last_hist_year = y.max()
        else:
            last_hist_year = df.filter(~pl.col(FORECAST_COLUMN))[YEAR_COLUMN].max()
            if last_hist_year is None:
                last_hist_year = 2021  # FIXME. This may occur with forecast_values.
        assert isinstance(last_hist_year, int)
        years = pl.DataFrame(data=range(last_hist_year + 1, end_year + 1), schema=[YEAR_COLUMN])
        if len(years):
            df = df.join(years, on=YEAR_COLUMN, how='outer_coalesce').sort(YEAR_COLUMN)
            df = df.with_columns([
                pl.when(pl.col(YEAR_COLUMN) > last_hist_year).then(pl.lit(True)).otherwise(pl.col(FORECAST_COLUMN)).alias(FORECAST_COLUMN)
            ])
        return ppl.to_ppdf(df, meta=meta)

    def nafill_pad(self) -> ppl.PathsDataFrame:
        """Fill N/A values by propagating the last valid observation forward.

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

        zdf = df.group_by(remaining_keys).agg([
            *[pl.sum(col).alias(col) for col in meta.metric_cols],
            *fc,
        ]).sort(YEAR_COLUMN)
        return ppl.to_ppdf(zdf, meta=meta)

    def join_over_index(
        self,
        other: ppl.PathsDataFrame,
        how: Literal['left', 'outer'] = 'left',  # FIXME Should also have option 'inner'
        index_from: Literal['left', 'right', 'union'] = 'left'
    ):
        sdf = self._df
        sm = sdf.get_meta()
        om = other.get_meta()
        # Join on subset of keys
        join_on = list(set(sm.primary_keys) & set(om.primary_keys))
        if not len(join_on):
            if len(other) == 1:  # A single value copied to all rows
                #df = sdf.with_columns(other).paths._df
                raise Exception("invalid access")
            else:
                raise ValueError("No shared primary keys between joined DFs")
        else:
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
                pl.col(FORECAST_COLUMN).fill_null(False) | pl.col(fc_right).fill_null(False)
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
        cast_exprs = []
        for col in df.dim_ids:
            if df.schema[col] != pl.Utf8:
                cast_exprs.append(pl.col(col).cast(pl.Utf8))
        if cast_exprs:
            df = df.with_columns(cast_exprs)
        return df

    def add_with_dims(self, odf: ppl.PathsDataFrame, how: Literal['left', 'outer'] = 'left') -> ppl.PathsDataFrame:
        df = self._df
        if len(df.metric_cols) != 1:
            raise Exception("Currently adding only one metric column is supported")
        val_col = df.metric_cols[0]

        output_unit = df.get_unit(val_col)
        meta = df.get_meta()
        cols = df.columns
        odf = odf.ensure_unit(val_col, output_unit).select([YEAR_COLUMN, *meta.dim_ids, val_col, FORECAST_COLUMN])

        for dim in meta.dim_ids:
            dt = df[dim].dtype
            if odf[dim].dtype != dt:
                odf = odf.with_columns([pl.col(dim).cast(df[dim].dtype)])

        left_fc = pl.col(FORECAST_COLUMN)
        right_fc = pl.col(FORECAST_COLUMN + '_right')
        left_val = pl.col(val_col)
        right_val = pl.col(val_col + '_right')
        pl_how: pl_types.JoinStrategy = how
        if how == 'outer':
            left_fc = left_fc.fill_null(False)
            right_fc = right_fc.fill_null(False)

            left_val = left_val.fill_null(0)
            right_val = right_val.fill_null(0)
            pl_how = 'outer_coalesce'
        elif how == 'left':
            right_fc = right_fc.fill_null(False)
            right_val = right_val.fill_null(False)

        df = ppl.to_ppdf(df.join(odf, on=[YEAR_COLUMN, *meta.dim_ids], how=pl_how), meta=meta)
        df = df.with_columns([
            left_val + right_val,
            left_fc | right_fc
        ])
        df = df.select(cols)
        return df

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
        if not isinstance(year, (list, tuple)):
            year = [year]
        df = self._df.filter(pl.col(YEAR_COLUMN).is_in(year))
        print(df)
