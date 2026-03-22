from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from common import polars as ppl
from nodes.constants import (
    FORECAST_COLUMN,
    IMPACT_COLUMN,
    IMPACT_GROUP,
    REFERENCE_SCENARIO_GROUP,
    SCENARIO_ACTION_GROUP,
    YEAR_COLUMN,
)
from nodes.exceptions import NodeError

if TYPE_CHECKING:
    import pandas as pd

    from nodes.node import Node


def nafill_all_forecast_years(df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    pldf = ppl.from_pandas(df)
    meta = pldf.get_meta()
    pldf = pldf.paths.to_wide(meta=meta).paths.make_forecast_rows(end_year).paths.to_narrow()
    df = pldf.paths.to_pandas(meta=meta)
    return df


def extend_last_historical_value_pl(df: ppl.PathsDataFrame, end_year: int) -> ppl.PathsDataFrame:
    if FORECAST_COLUMN not in df.columns:
        df = df.with_columns([pl.lit(False).alias(FORECAST_COLUMN)])  # noqa: FBT003
    if all(df[FORECAST_COLUMN]):  # Nothing to extend if there are no historical values
        return df
    df = df.paths.to_wide()
    df = df.paths.make_forecast_rows(end_year)
    last_hist_year = df.filter(pl.col(FORECAST_COLUMN).eq(False))[YEAR_COLUMN].max()  # noqa: FBT003
    df = df.paths.nafill_pad()
    if last_hist_year is not None:
        fc_cond = pl.col(YEAR_COLUMN) > last_hist_year
    else:
        fc_cond = pl.lit(True)  # noqa: FBT003
    fc = pl.when(fc_cond).then(True).otherwise(False)  # noqa: FBT003
    df = df.with_columns([fc.alias(FORECAST_COLUMN)])
    df = df.paths.to_narrow()
    return df


def extend_last_historical_value(df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    pdf = ppl.from_pandas(df)
    pdf = extend_last_historical_value_pl(pdf, end_year)
    df = pdf.paths.to_pandas()
    return df


def extend_last_forecast_value_pl(df: ppl.PathsDataFrame, end_year: int) -> ppl.PathsDataFrame:
    if FORECAST_COLUMN not in df.columns:
        raise ValueError('There is no FORECAST_COLUMN.')
    last_forecast_year = df[YEAR_COLUMN].max()
    df = df.paths.to_wide()
    df = df.paths.make_forecast_rows(end_year)
    df = df.paths.nafill_pad()
    if last_forecast_year is not None:
        df = df.with_columns(
            pl
            .when(pl.col(YEAR_COLUMN) > last_forecast_year)
            .then(pl.lit(value=True))
            .otherwise(pl.col(FORECAST_COLUMN))
            .alias(FORECAST_COLUMN)
        )
    df = df.paths.to_narrow()
    return df


def extend_to_history_pl(df: ppl.PathsDataFrame, start_year: int) -> ppl.PathsDataFrame:
    if FORECAST_COLUMN not in df.columns:
        raise ValueError('There is no FORECAST_COLUMN.')
    end_year = df[YEAR_COLUMN].max()
    assert isinstance(end_year, int)
    df_year = ppl.PathsDataFrame(pl.int_range(start_year, end_year + 1, eager=True).alias(YEAR_COLUMN)).add_to_index(YEAR_COLUMN)
    df = df.paths.to_wide()
    df = df.paths.join_over_index(df_year, how='outer')
    df = df.with_columns(pl.all().fill_null(strategy='backward'))
    df = df.paths.to_narrow()
    return df


AR5GWP100 = {
    'co2_eq': 1.0,
    'co2': 1.0,
    'ch4': 28.0,
    'n2o': 265.0,
    'hfc': 1300.0,
    'pfc': 6630.0,
    'sf6': 23500.0,
    'nf3': 16100.0,
    'co2_biogen': 0.0,
}


def convert_to_co2e(df: ppl.PathsDataFrame, dim_id: str) -> ppl.PathsDataFrame:
    if len(df.metric_cols) > 1:
        raise Exception('Only one metric column supported')
    if dim_id not in df.primary_keys:
        raise Exception("Greenhouse gas column '%s' not in primary keys (%s)" % (dim_id, df.primary_keys))
    metric_col = df.metric_cols[0]

    gwp_items = list(AR5GWP100.items())
    gwp = pl.DataFrame(gwp_items, schema=[dim_id, 'gwp_factor'], orient='row')
    gdf = ppl.to_ppdf(gwp, meta=ppl.DataFrameMeta(units={}, primary_keys=[dim_id]))
    gdf = gdf.set_unit('gwp_factor', 'dimensionless')

    df = df.paths.join_over_index(gdf, how='left', index_from='left')
    if df['gwp_factor'].null_count():
        print(df)
        raise Exception('Some greenhouse gases failed to convert')
    df = df.multiply_cols([metric_col, 'gwp_factor'], metric_col)
    df = df.drop('gwp_factor')
    df = df.paths.sum_over_dims([dim_id])
    return df


def compute_scenario_impact(  # Gets inspiration from ActionNode.compute_impact
    target_node: Node,
    reference_scenario_id: str = 'baseline',
) -> ppl.PathsDataFrame:
    """
    Compute scenario impact: current scenario vs reference scenario for target_node.

    Always "current vs reference" (no baseline/action-style variant). Returns three
    blocks in IMPACT_COLUMN: Scenario, Reference, Impact (same pattern as compute_impact).
    """
    ref_scenario = target_node.context.scenarios[reference_scenario_id]
    current_df = target_node.get_output_pl()
    with ref_scenario.override():
        reference_df = target_node.get_output_pl()

    if len(current_df) != len(reference_df):
        raise NodeError(
            target_node,
            'Scenario impact: current and reference output row counts differ (%d vs %d)' % (len(current_df), len(reference_df)),
        )

    metrics = target_node.output_metrics.values()
    mcols = [m.column_id for m in metrics]
    renames = {}
    for m in mcols:
        if m not in reference_df.metric_cols:
            raise NodeError(
                target_node,
                'Output of %s did not contain the %s metric column' % (target_node.id, m),
            )
        renames[m] = '%s:Reference' % m
    reference_df = reference_df.rename(renames)
    df = current_df.paths.join_over_index(reference_df)

    impact_cols = []
    impact_units = {}
    for m in mcols:
        wc = pl.col(m)
        wref = pl.col('%s:Reference' % m)
        tol = 1e-6
        wref = pl.when((wc - wref).abs() < (tol * wref).abs()).then(wc).otherwise(wref)
        ic_name = '%s:Impact' % m
        ic = (wc - wref).alias(ic_name)
        impact_cols.append(ic)
        impact_units[ic_name] = df.get_unit(m)

    df = df.with_columns(impact_cols)
    for col, unit in impact_units.items():
        df = df.set_unit(col, unit)
    common_cols = [YEAR_COLUMN, *df.dim_ids, FORECAST_COLUMN]
    # Long format (concat blocks) instead of pivot: IMPACT_COLUMN is a dimension, same
    # metric column names in every block, so downstream can filter e.g. IMPACT_GROUP and
    # get one table with a single schema; pivot would give wide columns (m1_Scenario, ...).
    current_block = df.select([*common_cols, pl.lit(SCENARIO_ACTION_GROUP).alias(IMPACT_COLUMN), *mcols])
    reference_block = df.select(
        [
            *common_cols,
            pl.lit(REFERENCE_SCENARIO_GROUP).alias(IMPACT_COLUMN),
            *[pl.col('%s:Reference' % m).alias(m) for m in mcols],
        ],
    )
    impact_block = df.select(
        [*common_cols, pl.lit(IMPACT_GROUP).alias(IMPACT_COLUMN), *[pl.col('%s:Impact' % m).alias(m) for m in mcols]],
    )

    meta = current_block.get_meta()
    zdf = pl.concat([current_block, reference_block, impact_block], how='vertical')
    result = ppl.to_ppdf(zdf, meta=meta)
    result = result.add_to_index(IMPACT_COLUMN)
    return result
