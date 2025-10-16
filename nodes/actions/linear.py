from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from django.utils.translation import gettext_lazy as _
from pydantic import BaseModel, Field, RootModel, validator

import pandas as pd
import polars as pl

from common import polars as ppl
from nodes.constants import (
    FLOW_ID_COLUMN,
    FLOW_ROLE_COLUMN,
    FLOW_ROLE_TARGET,
    FORECAST_COLUMN,
    NODE_COLUMN,
    VALUE_COLUMN,
    YEAR_COLUMN,
)
from nodes.exceptions import NodeError
from params import Parameter, ParameterWithUnit
from params.param import BoolParameter, NumberParameter, ValidationError

from .action import ActionNode

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nodes.node import Node
    from nodes.units import Unit


class ReduceAmount(BaseModel):
    year: int
    amount: float


class ReduceTarget(BaseModel):
    node: str | int | None = None
    # dimension_id -> category_id
    categories: dict[str, str] = Field(default_factory=dict)


class ReduceFlow(BaseModel):
    target: ReduceTarget
    amounts: list[ReduceAmount]

    @validator('amounts')
    def enough_years(cls, v):  # noqa: N805
        if len(v) < 2:
            raise ValueError("Must supply values for at least two years")
        return v

    def make_index(
        self, output_nodes: list[Node], extra_level: str | None = None, extra_level_values: Iterable | None = None,
    ) -> pd.MultiIndex:
        dims: dict[str, set[str]] = {dim: set() for dim in list(self.target.categories.keys())}
        nodes: set[str] = set()

        def get_node_id(node: str | int | None) -> str:
            if isinstance(node, str):
                return node
            if node is None:
                nr = 0
            else:
                nr = node
            return output_nodes[nr].id

        nodes.add(get_node_id(self.target.node))
        for dim, cat in self.target.categories.items():
            dims[dim].add(cat)

        level_list = list(dims.keys())
        cat_list: list[Iterable] = [dims[dim] for dim in level_list]
        level_list.insert(0, 'node')
        cat_list.insert(0, nodes)
        if extra_level:
            level_list.append(extra_level)
            assert extra_level_values is not None
            cat_list.append(extra_level_values)
        index = pd.MultiIndex(cat_list, names=level_list)
        return index


class ReduceParameterValue(RootModel):
    root: list[ReduceFlow]


@dataclass
class ReduceParameter(ParameterWithUnit, Parameter):
    value: ReduceParameterValue | None = None

    def __post_init__(self, unit_str: str | None = None):
        self._init_unit(unit_str)
        super().__post_init__()

    def serialize_value(self) -> Any:
        return super().serialize_value()

    def clean(self, value: Any) -> ReduceParameterValue:
        if not isinstance(value, list):
            raise ValidationError(self, "Input must be a list")

        return ReduceParameterValue.validate(value)


class ReduceAction(ActionNode):
    explanation = _("""Define action with parameters <i>reduce</i> and <i>multiplier</i>.""")
    allowed_parameters: ClassVar[list[Parameter]] = [
        ReduceParameter(local_id='reduce'),
        NumberParameter(local_id='multiplier'),
    ]

    def _compute_one(self, flow_id: str, param: ReduceFlow, unit: Unit) -> ppl.PathsDataFrame:
        amounts = sorted(param.amounts, key=lambda x: x.year)
        data = [[a.year, a.amount] for a in param.amounts]
        cols = [YEAR_COLUMN, 'Target']

        multiplier = self.get_parameter_value('multiplier', required=False, units=True)
        if multiplier:
            mult = multiplier.to('dimensionless').m
        else:
            mult = 1

        df = pl.DataFrame(data, schema=cols, orient='row')

        years = pl.DataFrame(range(amounts[0].year, self.get_end_year() + 1), schema=[YEAR_COLUMN])
        df = years.join(df, how='left', on=YEAR_COLUMN)
        dupes = df.filter(pl.col(YEAR_COLUMN).is_duplicated())
        if len(dupes):
            raise NodeError(self, "Duplicate rows")

        df = df.group_by(YEAR_COLUMN).agg(pl.first('Target')).sort(YEAR_COLUMN)
        df = df.with_columns(df['Target'].interpolate()).fill_null(0)
        value_cols = [col for col in df.columns if col != YEAR_COLUMN]
        if self.is_enabled():
            df = df.with_columns([(pl.cum_sum(col) * pl.lit(mult)).alias(col) for col in value_cols])
        else:
            df = df.with_columns([pl.lit(float(0)).alias(col) for col in value_cols])

        targets = [('Target', param.target)]

        all_dims = set(param.target.categories.keys())

        def get_node_id(node: str | int | None) -> str:
            if isinstance(node, str):
                return node
            if node is None:
                nr = 0
            else:
                nr = node
            return self.output_nodes[nr].id

        def make_target_df(target: ReduceTarget, valuecol: str) -> pl.DataFrame:
            target_dims = set(target.categories.keys())
            null_dims = all_dims - target_dims
            target_cats = sorted(target.categories.items(), key=lambda x: x[0])
            cat_exprs = [pl.lit(cat).alias(dim) for dim, cat in target_cats]
            if not self.is_enabled():
                value_expr = pl.lit(0.0)
            else:
                value_expr = pl.col(valuecol)
            tdf = df.select([
                pl.col(YEAR_COLUMN),
                pl.lit(get_node_id(target.node)).alias(NODE_COLUMN),
                pl.lit(FLOW_ROLE_TARGET).alias(FLOW_ROLE_COLUMN),
                *cat_exprs,
                *[pl.lit(None).cast(pl.Utf8).alias(null_dim) for null_dim in null_dims],
                value_expr.alias(VALUE_COLUMN),
            ])
            return tdf

        dfs = [make_target_df(target, col) for col, target in targets]
        df = pl.concat(dfs).sort(YEAR_COLUMN)
        #df = df.groupby([NODE_COLUMN, *all_dims, YEAR_COLUMN]).agg(pl.sum(VALUE_COLUMN)).sort(YEAR_COLUMN)
        df = df.with_columns([
            pl.lit(True).alias(FORECAST_COLUMN),  # noqa: FBT003
            pl.lit(flow_id).alias(FLOW_ID_COLUMN),
        ])
        meta = ppl.DataFrameMeta(
            units={VALUE_COLUMN: unit},
            primary_keys=[FLOW_ID_COLUMN, YEAR_COLUMN, NODE_COLUMN, *all_dims]
        )
        ret = ppl.to_ppdf(df, meta=meta)
        return ret

    def compute_effect_flow(self) -> ppl.PathsDataFrame:
        po = self.get_parameter('reduce')
        value = po.get()
        assert isinstance(value, ReduceParameterValue)

        dfs: list[ppl.PathsDataFrame] = []
        for idx, entry in enumerate(value.root):
            df = self._compute_one(str(idx), entry, po.get_unit())
            dfs.append(df)

        all_pks = set()
        for df in dfs:
            all_pks.update(df.primary_keys)

        meta = dfs[0].get_meta()
        meta.primary_keys = list(all_pks)
        sdf = pl.concat(dfs, how='diagonal')
        df = ppl.to_ppdf(sdf, meta=meta)
        return df

    def compute_effect(self) -> ppl.PathsDataFrame:
        df = self.compute_effect_flow().drop([FLOW_ID_COLUMN, FLOW_ROLE_COLUMN])
        meta = df.get_meta()
        sdf = df.group_by(df.primary_keys).agg([pl.sum(VALUE_COLUMN), pl.first(FORECAST_COLUMN)])
        sdf = sdf.sort(meta.primary_keys)
        df = ppl.to_ppdf(sdf, meta=meta)
        return df

    def has_multinode_output(self) -> bool:
        po = self.get_parameter('reduce')
        value = po.get()
        assert isinstance(value, ReduceParameterValue)
        return len(value.root) > 1

class DatasetReduceAction(ActionNode):
    explanation = _("""
    Receive goal input from a dataset or node and cause a linear effect.

    The output will be a time series with the difference to the
    last historical value of the input node.

    The goal input can also be relative (for e.g. percentage
    reductions), in which case the input will be treated as
    a multiplier.
    """)

    allowed_parameters: ClassVar[list[Parameter]] = [
        BoolParameter(local_id='relative_goal'),
    ]

    def compute_effect(self) -> ppl.PathsDataFrame:  # noqa: C901, PLR0915

        # ----- Create historical DataFrame, containing values for only the last historical year.
        n = self.get_input_node(tag='historical', required=False)
        if n is None:
            df = self.get_input_dataset_pl(tag='historical')
            if FORECAST_COLUMN not in df.columns:
                df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
            assert len(df.metric_cols) == 1
            df = df.rename({df.metric_cols[0]: VALUE_COLUMN})
        else:
            df = n.get_output_pl(target_node=self)
            df = df.filter(~pl.col(FORECAST_COLUMN))  # FIXME FOR DIFF

        max_hist_year = df[YEAR_COLUMN].max()
        df = df.filter(pl.col(YEAR_COLUMN) == max_hist_year)
        df = df.paths.cast_index_to_str()

        # ----- Create goal DataFrame.
        goal_input_df = self.get_input_dataset_pl(tag='goal', required=False)
        if goal_input_df is None:
            goal_input_node = self.get_input_node(tag='goal', required=True)
            goal_input_df = goal_input_node.get_output_pl(target_node=self)
        assert goal_input_df is not None
        gdf = goal_input_df

        assert len(gdf.metric_cols) == 1
        gdf = (
            gdf.rename({gdf.metric_cols[0]: VALUE_COLUMN})
            .with_columns(pl.lit(value=True).alias(FORECAST_COLUMN))
        )

        if not set(gdf.dim_ids).issubset(set(self.input_dimensions.keys())):
            raise NodeError(self, "Dimension mismatch to input nodes")

        gdf = gdf.paths.cast_index_to_str()

        # ----- Filter historical DataFrame to only the categories specified in the goal dataset.
        exprs = [pl.col(dim_id).is_in(gdf[dim_id].unique()) for dim_id in gdf.dim_ids]
        if exprs:
            df = df.filter(pl.all_horizontal(exprs))

        # ----- If goals are relative (i.e. a multiplier), transform into absolute values by
        #       multiplying with the last historical values.
        is_mult = self.get_parameter_value('relative_goal', required=False)
        if is_mult:
            gdf = gdf.rename({VALUE_COLUMN: 'Multiplier'})
            hdf = df.drop(YEAR_COLUMN)
            metric_cols = [m.column_id for m in self.output_metrics.values()]
            hdf = hdf.rename({m: 'Historical%s' % m for m in metric_cols})
            gdf = gdf.paths.join_over_index(hdf, how='outer', index_from='union')
            gdf = gdf.filter(~pl.all_horizontal([pl.col('Historical%s' % col).is_null() for col in metric_cols]))
            for m in self.output_metrics.values():
                col = m.column_id
                gdf = gdf.multiply_cols(['Multiplier', 'Historical%s' % col], col, out_unit=m.unit)
                gdf = gdf.with_columns(pl.col(col).fill_nan(None))
            gdf = gdf.select_metrics(metric_cols)

        # ----- Concatenate historical and goal DataFrames, ensuring null columns are dropped.
        df = df.paths.to_wide()
        gdf = gdf.paths.to_wide()

        meta = gdf.get_meta()
        gdf = gdf.filter(pl.col(YEAR_COLUMN) > max_hist_year)
        df = ppl.to_ppdf(pl.concat([df, gdf], how='diagonal'), meta=meta)

        df = df.drop([m for m in df.metric_cols if df[m].is_null().all()])
        df = df.with_columns([
            pl.when(~pl.col(FORECAST_COLUMN))
              .then(pl.col(m).fill_null(0.0))
              .otherwise(pl.col(m))
            for m in df.metric_cols
        ])
        df = df.paths.make_forecast_rows(end_year=self.get_end_year())
        df = df.with_columns([pl.col(m).interpolate() for m in df.metric_cols])

        # ----- Recalculate the timeseries to be the difference relative to the last historical year.
        exprs = [pl.col(m) - pl.first(m) for m in df.metric_cols]
        df = df.select([YEAR_COLUMN, FORECAST_COLUMN, *exprs])
        df = df.filter(pl.col(FORECAST_COLUMN))
        df = df.filter(pl.col(YEAR_COLUMN).le(self.get_end_year()))
        df = df.paths.to_narrow()
        for m in self.output_metrics.values():
            if m.column_id not in df.metric_cols:
                raise NodeError(self, "Metric column '%s' not found in output" % m.column_id)
            if not self.is_enabled():
                # Replace non-null columns with 0 when action is not enabled
                df = df.with_columns(
                    pl.when(pl.col(m.column_id).is_null()).then(None).otherwise(0.0).alias(m.column_id)
                )
            df = df.ensure_unit(m.column_id, m.unit)
        return df


class DatasetDifferenceAction(ActionNode):  # FIXME Merge with DatasetReduceAction
    explanation = _("""
    Receive goal input from a dataset or node and cause an effect.

    The output will be a time series with the difference to the
    predicted baseline value of the input node. So, there is a difference
    to the DatasetReduceAction only if the predicted baseline trend changes in time.

    The goal input can also be relative (for e.g. percentage
    reductions), in which case the input will be treated as
    a multiplier.
    """)

    allowed_parameters: ClassVar[list[Parameter]] = [
        BoolParameter(local_id='relative_goal'),
    ]

    def compute_effect(self) -> ppl.PathsDataFrame:
        n = self.get_input_node(tag='baseline', required=False)
        if n is None:
            df = self.get_input_dataset_pl(tag='baseline')
            if FORECAST_COLUMN not in df.columns:
                df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
            assert len(df.metric_cols) == 1
            df = df.rename({df.metric_cols[0]: VALUE_COLUMN})
        else:
            df = n.get_output_pl(target_node=self)
            # df = df.filter(~pl.col(FORECAST_COLUMN))  # FIXME FOR DIFF

        # max_year = df[YEAR_COLUMN].max()
        # df = df.filter(pl.col(YEAR_COLUMN) == max_year)

        gdf = self.get_input_dataset_pl(tag='goal', required=False)
        if gdf is None:
            gn = self.get_input_node(tag='goal', required=True)
            gdf = gn.get_output_pl(target_node=self)

        if not set(gdf.dim_ids).issubset(set(self.input_dimensions.keys())):
            raise NodeError(self, "Dimension mismatch to input nodes")

        # Filter historical data with only the categories that are
        # specified in the goal dataset.

        exprs = [pl.col(dim_id).is_in(gdf[dim_id].unique()) for dim_id in gdf.dim_ids]
        df = df.filter(pl.all_horizontal(exprs))

        assert len(gdf.metric_cols) == 1
        gdf = (
            gdf.rename({gdf.metric_cols[0]: VALUE_COLUMN})
            .with_columns(pl.lit(True).alias(FORECAST_COLUMN))  # noqa: FBT003
        )

        is_mult = self.get_parameter_value('relative_goal', required=False)
        if is_mult:
            # If the goal series is relative (i.e. a multiplier), transform
            # it into absolute values by multiplying with the last historical values.
            gdf = gdf.rename({VALUE_COLUMN: 'Multiplier'})
            hdf = df.drop(YEAR_COLUMN)
            metric_cols = [m.column_id for m in self.output_metrics.values()]
            hdf = hdf.rename({m: 'Historical%s' % m for m in metric_cols})
            gdf = gdf.paths.join_over_index(hdf, how='outer', index_from='union')
            assert gdf is not None
            gdf = gdf.filter(~pl.all_horizontal([pl.col('Historical%s' % col).is_null() for col in metric_cols]))
            for m in self.output_metrics.values():
                col = m.column_id
                gdf = gdf.multiply_cols(['Multiplier', 'Historical%s' % col], col, out_unit=m.unit)
                gdf = gdf.with_columns(pl.col(col).fill_nan(None))
            gdf = gdf.select_metrics(metric_cols)

        bdf = df.paths.to_wide().filter(pl.col(FORECAST_COLUMN).eq(False))  # noqa: FBT003
        gdf = gdf.paths.to_wide()

        meta = bdf.get_meta()
        gdf = ppl.to_ppdf(pl.concat([bdf, gdf], how='diagonal'), meta=meta)
        gdf = gdf.paths.make_forecast_rows(end_year=self.get_end_year())
        gdf = gdf.with_columns([pl.col(m).interpolate() for m in gdf.metric_cols])

        # Change the time series to be a difference to the baseline
        gdf = gdf.paths.to_narrow()

        df = df.paths.join_over_index(gdf)
        df = df.subtract_cols([VALUE_COLUMN + '_right', VALUE_COLUMN], VALUE_COLUMN)
        df = df.drop(VALUE_COLUMN + '_right')

        for m in self.output_metrics.values():
            if not self.is_enabled():
                # Replace non-null columns with 0 when action is not enabled
                df = df.with_columns(
                    pl.when(pl.col(m.column_id).is_null()).then(None).otherwise(0.0).alias(m.column_id)
                )
            df = df.ensure_unit(m.column_id, m.unit)
        return df
