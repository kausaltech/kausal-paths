from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from django.utils.translation import gettext_lazy as _

import polars as pl

from common import polars as ppl
from nodes.actions.params import ReduceParameter, ReduceParameterValue
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
from params.param import BoolParameter, NumberParameter

from .action import ActionNode

if TYPE_CHECKING:
    from nodes.actions.params import ReduceFlow, ReduceTarget
    from nodes.node import NodeMetric
    from nodes.units import Unit
    from params import Parameter


class ReduceAction(ActionNode):
    explanation = _("""Define action with parameters <i>reduce</i> and <i>multiplier</i>.""")
    allowed_parameters: ClassVar[list[Parameter[Any]]] = [
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
            raise NodeError(self, 'Duplicate rows')

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
        # df = df.groupby([NODE_COLUMN, *all_dims, YEAR_COLUMN]).agg(pl.sum(VALUE_COLUMN)).sort(YEAR_COLUMN)
        df = df.with_columns([
            pl.lit(True).alias(FORECAST_COLUMN),  # noqa: FBT003
            pl.lit(flow_id).alias(FLOW_ID_COLUMN),
        ])
        meta = ppl.DataFrameMeta(units={VALUE_COLUMN: unit}, primary_keys=[FLOW_ID_COLUMN, YEAR_COLUMN, NODE_COLUMN, *all_dims])
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
    """
    Multi-metric-capable variant of DatasetReduceAction.

    Processes each output metric independently through the full hist→goal→delta
    pipeline, then combines results. Single-metric nodes are a special case with one
    metric in the loop.

    Two source layouts are supported:

    * **Per-metric**: dataset/node tagged with both the data role and the metric id,
      e.g. ``tags: [historical, renovation_rate]``.  Each metric gets its own source.
    * **Shared**: dataset/node tagged with only the data role, e.g. ``tags: [historical]``.
      If the source has a column named after the metric it is extracted; otherwise the
      single column is used as-is (e.g. a shared multiplier with ``relative_goal: true``).
    """

    allowed_parameters: ClassVar[list[Parameter[Any]]] = [
        BoolParameter(local_id='relative_goal'),
    ]

    @staticmethod
    def _extract_col_as_value(df: ppl.PathsDataFrame, col: str) -> ppl.PathsDataFrame:
        """Return a narrow PathsDataFrame with only `col` renamed to VALUE_COLUMN."""
        old_meta = df.get_meta()
        key_cols = [c for c in df.columns if c not in df.metric_cols]
        return ppl.to_ppdf(
            df.select([*key_cols, col]).rename({col: VALUE_COLUMN}),
            meta=ppl.DataFrameMeta(primary_keys=old_meta.primary_keys, units={VALUE_COLUMN: old_meta.units[col]}),
        )

    def _get_metric_data(self, metric_id: str, col: str, data_role: str, is_multi_metric: bool) -> ppl.PathsDataFrame:  # noqa: C901, PLR0912, PLR0915
        """
        Load hist or goal data for one metric, returning a narrow df with VALUE_COLUMN.

        In multi-metric mode the lookup order is:
        1. Per-metric tagged source: dataset/edge with both ``data_role`` and ``metric_id`` in tags.
        2. Shared source: dataset/edge with only ``data_role`` in tags.
           If the shared source has the named ``col`` column (multi-metric node), that column is
           extracted; otherwise the single metric column is used as-is (e.g. a shared multiplier).
        """
        is_forecast = data_role == 'goal'

        if is_multi_metric:
            # 1. Per-metric tagged source
            for ds in self.input_dataset_instances:
                if data_role in ds.tags and metric_id in ds.tags:
                    df = ds.get_copy()
                    assert len(df.metric_cols) == 1
                    df = df.rename({df.metric_cols[0]: VALUE_COLUMN})
                    if not is_forecast and FORECAST_COLUMN in df.columns:
                        df = df.filter(~pl.col(FORECAST_COLUMN))
                    return df.with_columns(pl.lit(value=is_forecast).alias(FORECAST_COLUMN))
            for edge in self.edges:
                if edge.output_node is self and data_role in edge.tags and metric_id in edge.tags:
                    df = edge.input_node.get_output_pl(target_node=self)
                    if not is_forecast:
                        df = df.filter(~pl.col(FORECAST_COLUMN))
                    assert len(df.metric_cols) == 1
                    df = df.rename({df.metric_cols[0]: VALUE_COLUMN})
                    return df.with_columns(pl.lit(value=is_forecast).alias(FORECAST_COLUMN))

            # 2. Shared (role-only tagged) source
            for ds in self.input_dataset_instances:
                if data_role in ds.tags and metric_id not in ds.tags:
                    df = ds.get_copy()
                    if col in df.columns:
                        df = self._extract_col_as_value(df, col)
                    else:
                        assert len(df.metric_cols) == 1
                        df = df.rename({df.metric_cols[0]: VALUE_COLUMN})
                    if not is_forecast and FORECAST_COLUMN in df.columns:
                        df = df.filter(~pl.col(FORECAST_COLUMN))
                    return df.with_columns(pl.lit(value=is_forecast).alias(FORECAST_COLUMN))
            for edge in self.edges:
                if edge.output_node is self and data_role in edge.tags and metric_id not in edge.tags:
                    full_df = edge.input_node.get_output_pl(target_node=self)
                    if col in full_df.columns:
                        df = self._extract_col_as_value(full_df, col)
                    else:
                        assert len(full_df.metric_cols) == 1
                        df = full_df.rename({full_df.metric_cols[0]: VALUE_COLUMN})
                    if not is_forecast and FORECAST_COLUMN in df.columns:
                        df = df.filter(~pl.col(FORECAST_COLUMN))
                    return df.with_columns(pl.lit(value=is_forecast).alias(FORECAST_COLUMN))

            raise NodeError(self, "No %s data found for metric '%s'" % (data_role, metric_id))

        if data_role == 'historical':
            n = self.get_input_node(tag='historical', required=False)
            if n is not None:
                df = n.get_output_pl(target_node=self)
                return df.filter(~pl.col(FORECAST_COLUMN))
            df = self.get_input_dataset_pl(tag='historical')
            if FORECAST_COLUMN not in df.columns:
                df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
            else:
                df = df.filter(~pl.col(FORECAST_COLUMN))
            assert len(df.metric_cols) == 1
            return df.rename({df.metric_cols[0]: VALUE_COLUMN})

        goal_df = self.get_input_dataset_pl(tag='goal', required=False)
        if goal_df is None:
            goal_node = self.get_input_node(tag='goal', required=True)
            goal_df = goal_node.get_output_pl(target_node=self)
        assert goal_df is not None
        assert len(goal_df.metric_cols) == 1
        goal_df = goal_df.rename({goal_df.metric_cols[0]: VALUE_COLUMN})
        return goal_df.with_columns(pl.lit(value=True).alias(FORECAST_COLUMN))

    def compute_effect(self) -> ppl.PathsDataFrame:  # noqa: C901, PLR0915
        is_multi_metric = len(self.output_metrics) > 1
        result_df: ppl.PathsDataFrame | None = None
        max_hist_year: int | None = None

        for metric_id, metric in self.output_metrics.items():
            col = metric.column_id

            df = self._get_metric_data(metric_id, col, 'historical', is_multi_metric)
            if max_hist_year is None:
                max_hist_year_val = df[YEAR_COLUMN].max()
                assert isinstance(max_hist_year_val, int)
                max_hist_year = max_hist_year_val
            df = df.filter(pl.col(YEAR_COLUMN) == max_hist_year)
            df = df.paths.cast_index_to_str()

            gdf = self._get_metric_data(metric_id, col, 'goal', is_multi_metric)
            if not set(gdf.dim_ids).issubset(set(self.input_dimensions.keys())):
                raise NodeError(self, 'Dimension mismatch in goal data for metric %s' % metric_id)
            gdf = gdf.paths.cast_index_to_str()

            exprs = [pl.col(dim_id).is_in(gdf[dim_id].unique()) for dim_id in gdf.dim_ids]
            if exprs:
                df = df.filter(pl.all_horizontal(exprs))

            is_mult = self.get_parameter_value('relative_goal', required=False)
            if is_mult:
                # df has VALUE_COLUMN = historical for this metric (at max_hist_year).
                # gdf has VALUE_COLUMN = multiplier (shared or per-metric).
                # Produce absolute goal = historical x multiplier.
                gdf = gdf.rename({VALUE_COLUMN: 'Multiplier'})
                hdf = df.drop(YEAR_COLUMN).rename({VALUE_COLUMN: 'HistoricalValue'})
                gdf = gdf.paths.join_over_index(hdf, how='outer', index_from='union')
                gdf = gdf.filter(~pl.col('HistoricalValue').is_null())
                gdf = gdf.multiply_cols(['Multiplier', 'HistoricalValue'], VALUE_COLUMN, out_unit=metric.unit)
                gdf = gdf.with_columns(pl.col(VALUE_COLUMN).fill_nan(None))
                gdf = gdf.select_metrics([VALUE_COLUMN])

            # Harmonize both to the metric's unit before widening so concatenation is unit-consistent
            df = df.ensure_unit(VALUE_COLUMN, metric.unit)
            gdf = gdf.ensure_unit(VALUE_COLUMN, metric.unit)
            df = df.paths.to_wide()
            gdf = gdf.paths.to_wide()
            meta = gdf.get_meta()
            gdf = gdf.filter(pl.col(YEAR_COLUMN) > max_hist_year)
            df = df.with_columns(pl.col(YEAR_COLUMN).cast(pl.Int64))
            gdf = gdf.with_columns(pl.col(YEAR_COLUMN).cast(pl.Int64))
            df = ppl.to_ppdf(pl.concat([df, gdf], how='diagonal'), meta=meta)
            df = df.drop([m for m in df.metric_cols if df[m].is_null().all()])
            df = df.with_columns([
                pl.when(~pl.col(FORECAST_COLUMN)).then(pl.col(m).fill_null(0.0)).otherwise(pl.col(m)) for m in df.metric_cols
            ])
            df = df.paths.make_forecast_rows(end_year=self.get_end_year())
            df = df.with_columns([pl.col(m).interpolate() for m in df.metric_cols])
            # Forward-fill plateau past the last goal year (interpolate() leaves tail as NaN)
            df = df.with_columns([pl.col(m).fill_nan(None).forward_fill() for m in df.metric_cols])
            delta_exprs = [pl.col(m) - pl.first(m) for m in df.metric_cols]
            df = df.select([YEAR_COLUMN, FORECAST_COLUMN, *delta_exprs])
            df = df.filter(pl.col(FORECAST_COLUMN))
            df = df.filter(pl.col(YEAR_COLUMN).le(self.get_end_year()))
            df = df.paths.to_narrow()

            if is_multi_metric:
                old_meta = df.get_meta()
                df = ppl.to_ppdf(
                    df.rename({VALUE_COLUMN: col}),
                    meta=ppl.DataFrameMeta(primary_keys=old_meta.primary_keys, units={col: old_meta.units[VALUE_COLUMN]}),
                )

            result_df = df if result_df is None else result_df.paths.join_over_index(df, how='outer', index_from='union')

        assert result_df is not None
        df = result_df
        # After joining metrics with different dimensional spans via join_over_index,
        # some rows have null in dimensions that weren't used by a particular metric.
        # Drop those rows — the unused dimension has "dropped off" for that metric.
        for dim_id in df.dim_ids:
            if df[dim_id].null_count() > 0:
                df = ppl.to_ppdf(df.filter(pl.col(dim_id).is_not_null()), meta=df.get_meta())
        for m in self.output_metrics.values():
            if m.column_id not in df.metric_cols:
                raise NodeError(self, "Metric column '%s' not found in output" % m.column_id)
            if not self.is_enabled():
                df = df.with_columns(
                    pl
                    .when(pl.col(m.column_id).is_null() | pl.col(m.column_id).is_nan())
                    .then(None)
                    .otherwise(0.0)
                    .alias(m.column_id)
                )
            df = df.ensure_unit(m.column_id, m.unit)
        # ensure_unit uses numpy which converts null→NaN; convert back to null
        df = df.with_columns([pl.col(c).fill_nan(None) for c in df.metric_cols])
        return df


class DatasetReduceAction2(ActionNode):
    """
    Multi-metric action: interpolate each output metric from a historical baseline to a goal value.

    Then returns the year-by-year delta (change from baseline).

    Two source layouts are supported per metric:

    * **Per-metric**: dataset/node tagged with both the data role and the metric id,
      e.g. ``tags: [historical, renovation_rate]``.
    * **Shared**: dataset/node tagged with only the data role, e.g. ``tags: [historical]``.
      If the source has a column named after the metric it is extracted; otherwise the
      single column is used as-is (e.g. a shared multiplier with ``relative_goal: true``).
    """

    allowed_parameters: ClassVar[list[Parameter[Any]]] = [
        *ActionNode.allowed_parameters,
        BoolParameter(local_id='relative_goal'),
    ]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _extract_col_as_value(df: ppl.PathsDataFrame, col: str) -> ppl.PathsDataFrame:
        """Drop all metric columns except `col` and rename it to VALUE_COLUMN."""
        key_cols = [c for c in df.columns if c not in df.metric_cols]
        return df.select([*key_cols, col]).rename({col: VALUE_COLUMN})

    @staticmethod
    def _apply_forecast_filter(df: ppl.PathsDataFrame, *, is_forecast: bool) -> ppl.PathsDataFrame:
        """Keep only rows matching `is_forecast`, then stamp the Forecast column."""
        if FORECAST_COLUMN in df.columns:
            df = df.filter(pl.col(FORECAST_COLUMN) if is_forecast else ~pl.col(FORECAST_COLUMN))
        return df.with_columns(pl.lit(value=is_forecast).alias(FORECAST_COLUMN))

    def _ensure_columns(self, df: ppl.PathsDataFrame, of: ppl.PathsDataFrame) -> tuple[ppl.PathsDataFrame, ppl.PathsDataFrame]:
        of_dims = [dim for dim in of.dim_ids if dim not in df.dim_ids]
        df = df.with_columns([pl.lit(None, dtype=of[dim].dtype).alias(dim) for dim in of_dims])
        df = df.add_to_index(of_dims)

        df_dims = [dim for dim in df.dim_ids if dim not in of.dim_ids]
        of = of.with_columns([pl.lit(None, dtype=df[dim].dtype).alias(dim) for dim in df_dims])
        of = of.add_to_index(df_dims)

        return df, of

    # ----------------------------------------------------------- source lookup

    def _find_metric_source(self, metric_id: str, col: str, data_role: str) -> ppl.PathsDataFrame:
        """
        Return the raw source data for one metric with VALUE_COLUMN renamed but no Forecast filtering applied.

        Lookup order:
        1. Per-metric dataset  (tagged with data_role AND metric_id)
        2. Per-metric edge     (tagged with data_role AND metric_id)
        3. Shared dataset      (tagged with data_role only; col extracted if present)
        4. Shared edge         (tagged with data_role only)
        """
        for ds in self.input_dataset_instances:
            if data_role in ds.tags and metric_id in ds.tags:
                df = ds.get_copy()
                assert len(df.metric_cols) == 1
                return df.rename({df.metric_cols[0]: VALUE_COLUMN})
        for edge in self.edges:
            if edge.output_node is self and data_role in edge.tags and metric_id in edge.tags:
                df = edge.input_node.get_output_pl(target_node=self)
                assert len(df.metric_cols) == 1
                return df.rename({df.metric_cols[0]: VALUE_COLUMN})
        for ds in self.input_dataset_instances:
            if data_role in ds.tags and metric_id not in ds.tags:
                df = ds.get_copy()
                target_col = col if col in df.columns else df.metric_cols[0]
                return self._extract_col_as_value(df, target_col)
        for edge in self.edges:
            if edge.output_node is self and data_role in edge.tags and metric_id not in edge.tags:
                full_df = edge.input_node.get_output_pl(target_node=self)
                target_col = col if col in full_df.columns else full_df.metric_cols[0]
                return self._extract_col_as_value(full_df, target_col)
        raise NodeError(self, "No %s data found for metric '%s'" % (data_role, metric_id))

    # ---------------------------------------- per-metric pipeline steps

    def _load_baseline(self, metric_id: str, col: str) -> ppl.PathsDataFrame:
        """
        Return the historical baseline for one metric at its natural max historical year.

        Split is by Forecast column when present; otherwise all rows in the
        historical-tagged source are treated as historical.  The max-year row
        (per dimension category) becomes the baseline anchor.
        """
        df = self._find_metric_source(metric_id, col, 'historical')
        df = self._apply_forecast_filter(df, is_forecast=False)
        baseline_year_val = df[YEAR_COLUMN].max()
        assert isinstance(baseline_year_val, int)
        df = df.filter(pl.col(YEAR_COLUMN) == baseline_year_val).paths.cast_index_to_str()
        return df

    def _load_goal(self, metric_id: str, col: str, baseline_year: int) -> ppl.PathsDataFrame:
        """
        Return goal data for one metric, excluding years at or before baseline_year.

        Split is by Forecast column when present; otherwise all rows in the
        goal-tagged source are treated as goals.  The year guard prevents overlap
        with the baseline row when both sources share rows at the same year.
        """
        gdf = self._find_metric_source(metric_id, col, 'goal')
        gdf = self._apply_forecast_filter(gdf, is_forecast=True)
        if not set(gdf.dim_ids).issubset(set(self.input_dimensions.keys())):
            raise NodeError(self, 'Dimension mismatch in goal data for metric %s' % metric_id)
        gdf = gdf.paths.cast_index_to_str()
        return gdf.filter(pl.col(YEAR_COLUMN) > baseline_year)

    def _build_metric_delta(
        self,
        baseline: ppl.PathsDataFrame,
        goal: ppl.PathsDataFrame,
        metric: NodeMetric,
    ) -> ppl.PathsDataFrame:
        """
        Interpolate from baseline to goal and return year-by-year deltas (forecast years only).

        baseline: single-year snapshot at each metric's natural baseline year, Forecast=False
        goal:     target values at future years, Forecast=True, year > baseline_year
        """
        # Restrict baseline to dimension categories present in goal
        exprs = [pl.col(dim_id).is_in(goal[dim_id].unique()) for dim_id in goal.dim_ids]
        if exprs:
            baseline = baseline.filter(pl.all_horizontal(exprs))

        # Expand relative goal (multiplier * baseline value) to absolute values
        is_mult = self.get_parameter_value('relative_goal', required=False)
        if is_mult:
            goal = goal.rename({VALUE_COLUMN: 'Multiplier'})
            hdf = baseline.drop(YEAR_COLUMN).rename({VALUE_COLUMN: 'HistoricalValue'})
            goal = goal.paths.join_over_index(hdf, how='outer', index_from='union')
            goal = goal.filter(~pl.col('HistoricalValue').is_null())
            goal = goal.multiply_cols(['Multiplier', 'HistoricalValue'], VALUE_COLUMN, out_unit=metric.unit)
            goal = goal.with_columns(pl.col(VALUE_COLUMN).fill_nan(None))
            goal = goal.select_metrics([VALUE_COLUMN])

        # Harmonize units, widen, concatenate baseline + goal years
        baseline = baseline.ensure_unit(VALUE_COLUMN, metric.unit).paths.to_wide()
        goal = goal.ensure_unit(VALUE_COLUMN, metric.unit).paths.to_wide()
        meta = goal.get_meta()
        baseline = baseline.with_columns(pl.col(YEAR_COLUMN).cast(pl.Int64))
        goal = goal.with_columns(pl.col(YEAR_COLUMN).cast(pl.Int64))
        df = ppl.to_ppdf(pl.concat([baseline, goal], how='diagonal'), meta=meta)

        # Fill gaps in baseline rows (null → 0 for non-forecast anchor)
        df = df.drop([m for m in df.metric_cols if df[m].is_null().all()])
        df = df.with_columns([
            pl.when(~pl.col(FORECAST_COLUMN)).then(pl.col(m).fill_null(0.0)).otherwise(pl.col(m)) for m in df.metric_cols
        ])

        # Fill every year, interpolate between known points, plateau past last goal year
        df = df.paths.make_forecast_rows(end_year=self.get_end_year())
        df = df.with_columns([pl.col(m).interpolate() for m in df.metric_cols])
        df = df.with_columns([pl.col(m).fill_nan(None).forward_fill() for m in df.metric_cols])

        # delta = value - baseline_value; pl.first() is the Forecast=False anchor row
        delta_exprs = [pl.col(m) - pl.first(m) for m in df.metric_cols]
        df = df.select([YEAR_COLUMN, FORECAST_COLUMN, *delta_exprs])
        df = df.filter(pl.col(YEAR_COLUMN).le(self.get_end_year()))
        return df.paths.to_narrow()

    # ------------------------------------------------------- main entry point

    def compute_effect(self) -> ppl.PathsDataFrame:
        is_multi_metric = len(self.output_metrics) > 1
        if is_multi_metric or self.get_parameter_value('allow_null_categories', required=False):
            self.allow_null_categories = True  # type: ignore[attr-defined]
        result_df: ppl.PathsDataFrame | None = None

        for metric_id, metric in self.output_metrics.items():
            col = metric.column_id
            baseline = self._load_baseline(metric_id, col)
            baseline_year_val = baseline[YEAR_COLUMN].max()
            assert isinstance(baseline_year_val, int)
            goal = self._load_goal(metric_id, col, baseline_year_val)
            df = self._build_metric_delta(baseline, goal, metric)
            if is_multi_metric:
                df = df.rename({VALUE_COLUMN: col})

            if result_df is None:
                result_df = df
            else:
                # Expand both dfs to share all dimension columns (null for missing),
                # then outer-join treating null==null so metrics with different
                # dimensional spans land on separate rows rather than broadcasting.
                df, result_df = self._ensure_columns(df, result_df)
                result_df = result_df.paths.join_over_index(df, how='outer', index_from='union', nulls_equal=True)

        assert result_df is not None
        df = result_df
        # Rows with null in a dimension column represent a metric whose data does
        # not span that dimension.  Do NOT filter them out here — _get_output_for_target
        # drops all-null dim columns per-edge after metric extraction.

        for m in self.output_metrics.values():
            if m.column_id not in df.metric_cols:
                raise NodeError(self, "Metric column '%s' not found in output" % m.column_id)
            if not self.is_enabled():
                df = df.with_columns(
                    pl
                    .when(pl.col(m.column_id).is_null() | pl.col(m.column_id).is_nan())
                    .then(None)
                    .otherwise(0.0)
                    .alias(m.column_id)
                )
            df = df.ensure_unit(m.column_id, m.unit)

        # ensure_unit uses numpy which converts null → NaN; restore nulls
        df = df.with_columns([pl.col(c).fill_nan(None) for c in df.metric_cols])
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
            raise NodeError(self, 'Dimension mismatch to input nodes')

        # Filter historical data with only the categories that are
        # specified in the goal dataset.

        exprs = [pl.col(dim_id).is_in(gdf[dim_id].unique()) for dim_id in gdf.dim_ids]
        if exprs:
            df = df.filter(pl.all_horizontal(exprs))

        assert len(gdf.metric_cols) == 1
        gdf = (
            gdf.rename({gdf.metric_cols[0]: VALUE_COLUMN}).with_columns(pl.lit(True).alias(FORECAST_COLUMN))  # noqa: FBT003
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
        bdf = bdf.with_columns(pl.col(YEAR_COLUMN).cast(pl.Int64))
        gdf = gdf.with_columns(pl.col(YEAR_COLUMN).cast(pl.Int64))
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
                df = df.with_columns(pl.when(pl.col(m.column_id).is_null()).then(None).otherwise(0.0).alias(m.column_id))
            df = df.ensure_unit(m.column_id, m.unit)
        return df
