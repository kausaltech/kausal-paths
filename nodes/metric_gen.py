from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import sentry_sdk

from paths.const import MODEL_CALC_OP

from common import polars as ppl
from common.i18n import gettext as _

from .actions.action import ActionImpact, ActionNode, ImpactOverview
from .actions.shift import ShiftAction
from .constants import (
    FORECAST_COLUMN,
    NODE_COLUMN,
    SCENARIO_COLUMN,
    STACKABLE_QUANTITIES,
    UNCERTAINTY_COLUMN,
    VALUE_COLUMN,
    YEAR_COLUMN,
)
from .exceptions import NodeError
from .simple import AdditiveNode, RelativeNode

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from nodes.dimensions import Dimension
    from nodes.scenario import Scenario
    from nodes.visualizations import VisualizationNodeOutput

    from .goals import NodeGoalsEntry
    from .node import Node, NodeMetric


from .metric import (
    DimensionalMetric,
    DimensionKind,
    MetricCategory,
    MetricCategoryGroup,
    MetricData,
    MetricDimension,
    MetricDimensionGoal,
    MetricYearlyGoal,
    NormalizerNode,
)


def _make_id(node: Node, *args: str) -> str:
    return ':'.join([node.id, *args])


def _create_input_nodes_dimension(node: Node, input_nodes: list[Node]) -> MetricDimension:
    cats = [
        MetricCategory(
            id=_make_id(node, 'node', n.id),
            original_id=n.id,
            label=str(n.short_name or n.name),
            color=n.color,
            order=n.order,
        )
        for n in input_nodes
    ]
    mdim = MetricDimension(
        id=_make_id(node, 'node', NODE_COLUMN),
        label=_('Sectors'),
        categories=cats,
        original_id=NODE_COLUMN,
        kind=DimensionKind.NODE,
    )
    mdim.ensure_unique_colors()
    return mdim


def _create_scenario_dim(node: Node, scenarios: Sequence[Scenario]) -> MetricDimension | None:
    cats: list[MetricCategory] = []
    seen_scenarios = set[str]()
    for scenario in scenarios:
        if scenario.id in seen_scenarios:
            continue
        seen_scenarios.add(scenario.id)
        cats.append(
            MetricCategory(
                id=_make_id(node, 'scenario', scenario.id),
                original_id=scenario.id,
                label=str(scenario.name),
                color=None,
                order=None,
            )
        )
    if len(cats) <= 1:
        return None
    scenario_dim = MetricDimension(
        id=_make_id(node, 'scenario', SCENARIO_COLUMN),
        label=_('Scenarios'),
        categories=cats,
        original_id=SCENARIO_COLUMN,
        kind=DimensionKind.SCENARIO,
    )
    return scenario_dim


def _get_df(
    node: Node, scenarios: Sequence[Scenario], input_nodes: list[Node] | None = None
) -> tuple[ppl.PathsDataFrame, list[MetricDimension]]:
    new_dims = []
    scenario_dim = _create_scenario_dim(node, scenarios)
    if scenario_dim:
        new_dims.append(scenario_dim)

    if input_nodes is None:
        def get_output_without_input_nodes() -> ppl.PathsDataFrame:
            return node.get_output_pl()

        get_output_func = get_output_without_input_nodes
    else:
        new_dims.append(_create_input_nodes_dimension(node, input_nodes))

        # Get output with input node IDs as categories (in a column `NODE_COLUMN`)
        def get_output_with_input_nodes() -> ppl.PathsDataFrame:
            return node.add_nodes_pl(df=None, nodes=input_nodes, keep_nodes=True)

        get_output_func = get_output_with_input_nodes

    if not scenario_dim:
        return get_output_func(), new_dims

    def add_scenario_column(df: ppl.PathsDataFrame, scenario_id: str) -> ppl.PathsDataFrame:
        return df.with_columns(pl.lit(scenario_id).alias(SCENARIO_COLUMN)).add_to_index(SCENARIO_COLUMN)

    scenario_dfs: list[ppl.PathsDataFrame] = []
    for scenario in scenarios:
        if scenario != node.context.active_scenario:
            with scenario.override():
                sdf = get_output_func()
        else:
            sdf = get_output_func()
        sdf = add_scenario_column(sdf, scenario.id)
        scenario_dfs.append(sdf)

    df = _join_scenario_dfs(scenario_dfs)
    return df, new_dims


def _compute_values(
    node: Node,
    scenarios: Sequence[Scenario] = (),
    include_input_nodes: bool = True,
) -> tuple[ppl.PathsDataFrame, list[MetricDimension]]:
    def include_as_input(node: Node) -> bool:
        if isinstance(node, ActionNode):
            return False
        return all(not dim.is_internal for dim in node.output_dimensions.values())

    # Add a functionality to show scenario impacts for nodes rather than outputs.
    # tst = node.context.get_parameter_value('show_scenario_impacts', required=False)
    # baseline = node.context.get_scenario('baseline')

    # Use inputs nodes as categories for the dimension "Sectors" in some cases.
    input_nodes: list[Node] | None
    input_nodes = [input_node for input_node in node.input_nodes if include_as_input(input_node)]
    if (
        not include_input_nodes
        or not isinstance(node, AdditiveNode)
        or len(input_nodes) <= 1
        or node.input_dataset_instances
        or isinstance(node, RelativeNode)
        or len(input_nodes) != len(node.input_nodes)
    ):
        # Inputs nodes can't reasonably be summed up, so we just use the output
        # without the input nodes dimension.
        input_nodes = None

    df, extra_dims = _get_df(node, scenarios=scenarios, input_nodes=input_nodes)
    return df, extra_dims

    # if tst:
    #     with baseline.override():
    #         ddf = node.add_nodes_pl(None, node.input_nodes, keep_nodes=True)
    #     df = df.paths.join_over_index(ddf)
    #     df = df.with_columns((pl.col(VALUE_COLUMN) - pl.col(VALUE_COLUMN + '_right')).alias(VALUE_COLUMN))
    #     df.drop(VALUE_COLUMN + '_right')

    # if tst:
    #     with baseline.override():
    #         ddf = node.get_output_pl()
    #     df = df.paths.join_over_index(ddf)
    #     df = df.with_columns((pl.col(VALUE_COLUMN) - pl.col(VALUE_COLUMN + '_right')).alias(VALUE_COLUMN))
    #     df.drop(VALUE_COLUMN + '_right')


def _make_goal_values(goal: NodeGoalsEntry) -> list[MetricYearlyGoal]:
    return [
        MetricYearlyGoal(
            year=y.year,
            value=y.value,
            is_interpolated=y.is_interpolated,
        )
        for y in goal.get_values()
    ]


def _make_goal(node: Node, dims: list[MetricDimension], goal: NodeGoalsEntry) -> MetricDimensionGoal | None:
    def make_id(*args: str) -> str:
        return _make_id(node, *args)

    data_dims_by_id = {dim.original_id: dim for dim in dims if dim.kind == DimensionKind.COMMON}
    cat_ids: list[str] = []
    group_ids: list[str] = []

    for dim_id, goal_dim in goal.dimensions.items():
        if dim_id not in data_dims_by_id:
            return None
        dim = data_dims_by_id[dim_id]
        node_dim = node.output_dimensions[dim_id]
        for grp_id in goal_dim.groups:
            # Check if all categories for the group are present in the data dimension
            node_dim_cats = node_dim.get_cats_for_group(grp_id)
            for cat in node_dim_cats:
                if cat.id not in dim.cats_by_original_id:
                    return None

            cat_ids += [make_id(dim.original_id, 'cat', cat.id) for cat in node_dim_cats]
            group_ids.append(make_id(dim.original_id, 'group', grp_id))

        for cat_id in goal_dim.categories:
            if cat_id not in dim.cats_by_original_id:
                return None
            cat_ids.append(make_id(dim.original_id, 'cat', cat_id))

    return MetricDimensionGoal(
        categories=cat_ids,
        groups=group_ids,
        values=_make_goal_values(goal),
    )


def _get_goals(node: Node, dims: list[MetricDimension]) -> list[MetricDimensionGoal]:
    goals: list[MetricDimensionGoal] = []

    if not node.goals:
        return []

    for goal in node.goals.root:
        out = _make_goal(node, dims, goal)
        if out is None:
            continue
        goals.append(out)
    return goals


def _make_data_dimension(
    node: Node,
    dim_id: str,
    dim: Dimension,
    df: ppl.PathsDataFrame,
) -> MetricDimension:
    def make_id(*args: str) -> str:
        return _make_id(node, *args)

    ordered_groups: list[MetricCategoryGroup] = []
    group_id_map: dict[str, str] = {}
    if dim.groups:
        # If the dimension has groups, add a column with group IDs
        df = df.with_columns(dim.ids_to_groups(pl.col(dim_id).alias('_Groups')))

        # Unique group IDs in the output data
        df_groups = set(df['_Groups'].unique())

        for grp in dim.groups:
            if grp.id not in df_groups:
                continue
            grp_id = make_id(dim.id, 'group', grp.id)
            group_id_map[grp.id] = grp_id
            ordered_groups.append(
                MetricCategoryGroup(
                    id=grp_id,
                    label=str(grp.label),
                    color=grp.color,
                    order=grp.order,
                    original_id=grp.id,
                )
            )
        assert len(ordered_groups) == len(df_groups)

    # Unique category IDs in the output data
    df_cats = set(df[dim_id].unique())

    # Ordered list of categories that are present in the output data
    ordered_cats: list[MetricCategory] = []
    for cat in dim.categories:
        if cat.id not in df_cats:
            continue
        cat_id = make_id(dim.id, 'cat', cat.id)
        ordered_cats.append(
            MetricCategory(
                id=cat_id,
                label=str(cat.label),
                color=cat.color,
                order=cat.order,
                original_id=cat.id,
                group=group_id_map[cat.group] if cat.group else None,
            )
        )

    assert len(df_cats) == len(ordered_cats)

    mdim = MetricDimension(
        id=make_id('dim', dim.id),
        label=str(dim.label),
        help_text=str(dim.help_text),
        categories=ordered_cats,
        original_id=dim.id,
        groups=ordered_groups,
    )
    return mdim


def _generate_output_data(node: Node, dims: list[MetricDimension], df: ppl.PathsDataFrame,
                          null_handling: Literal['zero', 'drop', 'keep'] = 'zero') -> MetricData:
    if df.paths.index_has_duplicates():
        raise NodeError(node, 'DataFrame index has duplicates')

    forecast_from = df.filter(pl.col(FORECAST_COLUMN).eq(other=True))[YEAR_COLUMN].min()
    if forecast_from is not None:
        assert isinstance(forecast_from, int)

    years = df[YEAR_COLUMN].unique().sort().to_list()
    idx_df = DimensionalMetric.generate_index_df(dims, years)

    # idx_names = [dim.original_id for dim in dims] + [YEAR_COLUMN]
    # idx_dfs = [pl.LazyFrame(dim.get_original_cat_ids(), schema=[dim.original_id], orient='row') for dim in dims] + [
    #     pl.LazyFrame(years, schema=['Year']),
    # ]
    # idf_lazy = idx_dfs[0]
    # for d in idx_dfs[1:]:
    #     idf_lazy = idf_lazy.join(d, how='cross')
    # idx_df = idf_lazy.collect()
    # idx_df = idx_df.select(idx_names)

    idx_exprs = [pl.col(n).cast(pl.Utf8) if n != YEAR_COLUMN else pl.col(n) for n in idx_df.columns]
    df = df.select([*idx_exprs, VALUE_COLUMN, FORECAST_COLUMN]).sort(by=idx_exprs)
    jdf = idx_df.join(df, how='left', on=idx_exprs, validate='1:1')
    assert len(df.metric_cols) == 1
    metric_col = df.metric_cols[0]
    vals: list[float | None]
    if null_handling == 'drop':
        vals = jdf[metric_col].drop_nulls().to_list()
    elif null_handling == 'keep':
        vals = jdf[metric_col].to_list()
    else:
        vals = jdf[metric_col].fill_null(0).to_list()
    return MetricData(
        years=years,
        values=vals,
        forecast_from=forecast_from,
    )


def _from_node_metric(node: Node, m: NodeMetric, scenarios: Sequence[Scenario]) -> DimensionalMetric:
    def make_id(*args: str) -> str:
        return _make_id(node, *args)

    dims: list[MetricDimension] = []
    with node.context.start_span('Compute metric values', op=MODEL_CALC_OP):
        df, dims = _compute_values(node, scenarios)

    if (UNCERTAINTY_COLUMN in df.columns and
        not df.filter(pl.col(UNCERTAINTY_COLUMN) == 'median').is_empty()):
        df = df.filter(pl.col(UNCERTAINTY_COLUMN) == 'median')

    if node.context.active_normalization:
        normalizer, df = node.context.active_normalization.normalize_output(m, df)
    else:
        normalizer = None

    for dim_id, dim in node.output_dimensions.items():
        data_dim = _make_data_dimension(node, dim_id, dim, df)
        dims.append(data_dim)

    goals = _get_goals(node, dims)

    stackable = m.quantity in STACKABLE_QUANTITIES
    if isinstance(node, ActionNode) and m.quantity in ('mix',):
        stackable = False

    data = _generate_output_data(node, dims, df)

    if normalizer:
        nnode = NormalizerNode(id=normalizer.id, name=str(normalizer.name))
    else:
        nnode = None

    dm = DimensionalMetric(
        id=node.id,
        name=str(node.name),
        dimensions=dims,
        values=data.values,
        years=data.years,
        forecast_from=data.forecast_from,
        normalized_by=nnode,
        stackable=stackable,
        goals=goals,
        unit=df.get_unit(m.column_id),
    )
    return dm


def _join_scenario_dfs(scenario_dfs: list[ppl.PathsDataFrame]) -> ppl.PathsDataFrame:
    first_df = scenario_dfs[0]
    meta = first_df.get_meta()
    sdfs = [first_df]
    for sdf in scenario_dfs[1:]:
        if not sdf.get_meta().is_equal(meta, ignore_order=True):
            with sentry_sdk.new_scope() as scope:
                scope.set_context('scenario_dfs', dict(first=first_df.serialize_meta(), other=sdf.serialize_meta()))
                sentry_sdk.capture_message('Scenario dataframes have different metadata', level='error')
            raise ValueError('Scenario dataframes have different metadata')
        sdfs.append(sdf.select(first_df.columns))

    df = pl.concat(sdfs)
    return ppl.to_ppdf(df, meta=meta)


def metric_from_node(
    node: Node,
    metric: NodeMetric | None = None,
    extra_scenarios: Sequence[Scenario] = (),
) -> DimensionalMetric | None:
    from nodes.actions.linear import ReduceAction

    if metric is None:
        try:
            m = node.get_default_output_metric()
        except Exception:
            return None
    else:
        # FIXME: Get goals only for the chosen metric
        m = metric

    if isinstance(node, ReduceAction) and node.has_multinode_output():
        return None
    if isinstance(node, ShiftAction):
        return None

    with sentry_sdk.new_scope() as scope, node.context.start_span('Dimension metric for: %s' % node.id, op=MODEL_CALC_OP):
        scope.set_tag('node_id', node.id)
        scope.set_context(
            'metric_dim',
            dict(
                metric_id=m.id,
                node_id=node.id,
                instance_id=node.context.instance.id,
                active_scenario=node.context.active_scenario.id,
                extra_scenarios=[s.id for s in extra_scenarios],
            ),
        )
        return _from_node_metric(node, m, extra_scenarios)


def metric_from_visualization(node: Node, visualization: VisualizationNodeOutput) -> DimensionalMetric | None:
    dims: list[MetricDimension] = []
    scenarios: list[Scenario] = []
    if visualization.scenarios:
        for scenario_id in visualization.scenarios:
            scenarios.append(node.context.get_scenario(scenario_id))  # noqa: PERF401

        scenario_dim = _create_scenario_dim(node, scenarios)
    else:
        scenarios = [node.context.active_scenario]
        scenario_dim = None

    def add_scenario_column(df: ppl.PathsDataFrame, scenario_id: str) -> ppl.PathsDataFrame:
        return df.with_columns(pl.lit(scenario_id).alias(SCENARIO_COLUMN)).add_to_index(SCENARIO_COLUMN)

    if scenario_dim:
        scenario_dfs: list[ppl.PathsDataFrame] = []
        for scenario in scenarios:
            if scenario != node.context.active_scenario:
                with scenario.override():
                    sdf = visualization.get_output(node)
            else:
                sdf = visualization.get_output(node)
            sdf = add_scenario_column(sdf, scenario.id)
            truncate = scenario.param_values.get('measure_data_override', False)
            if truncate:
                observed_years = set(node.context.instance.measure_years or [])
                sdf = sdf.with_columns(
                    pl.when(pl.col(FORECAST_COLUMN) | ~pl.col(YEAR_COLUMN).is_in(observed_years))
                    .then(pl.lit(None))
                    .otherwise(pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN)
                )
            scenario_dfs.append(sdf)

        try:
            df = _join_scenario_dfs(scenario_dfs)
        except Exception:
            return None
        dims.append(scenario_dim)
    else:
        df = visualization.get_output(node)

    for dim_id in df.dim_ids:
        if dim_id == SCENARIO_COLUMN:
            continue
        dim = node.output_dimensions[dim_id]
        data_dim = _make_data_dimension(node, dim_id, dim, df)
        dims.append(data_dim)

    unit = df.get_unit(df.metric_cols[0])
    data = _generate_output_data(node, dims, df, null_handling='keep')
    dm = DimensionalMetric(
        id='%s:%s' % (node.id, visualization.id),
        name=str(node.name),
        dimensions=dims,
        values=data.values,
        years=data.years,
        forecast_from=data.forecast_from,
        stackable=True,
        goals=[],
        normalized_by=None,
        unit=unit,
    )
    return dm


def from_action_impact(  # noqa: C901
    action_impact: ActionImpact,
    root: ImpactOverview,
    col: str,
) -> DimensionalMetric:
    import pandas as pd

    action = action_impact.action

    def make_id(*args: str) -> str:
        return ':'.join([action.id, *args])

    df = action_impact.df
    if col == 'Cost':
        dimensions = root.cost_node.output_dimensions.items()
        if root.invert_cost:
            df = df.with_columns((pl.col(col) * pl.lit(-1.0)).alias(col))
    elif col == 'Impact':
        dimensions = root.impact_node.output_dimensions.items()
        if root.invert_impact:
            df = df.with_columns((pl.col(col) * pl.lit(-1.0)).alias(col))
    else:
        raise ValueError('Unknown column %s' % col)

    dims: list[MetricDimension] = []

    for dim_id, dim in dimensions:
        if dim_id == 'iteration': # FIXME Check that this actually makes sense.
            continue
        df_cats = set(df[dim_id].unique())
        ordered_cats = []
        for cat in dim.categories:
            if cat.id not in df_cats:
                continue
            cat_id = make_id(dim.id, 'cat', cat.id)
            ordered_cats.append(
                MetricCategory(
                    id=cat_id,
                    label=str(cat.label),
                    color=cat.color,
                    order=cat.order,
                    original_id=cat.id,
                )
            )

        assert len(df_cats) == len(ordered_cats)

        mdim = MetricDimension(
            id=make_id('dim', dim.id),
            label=str(dim.label),
            help_text=str(dim.help_text),
            categories=ordered_cats,
            original_id=dim.id,
        )
        dims.append(mdim)

    forecast_from = df.filter(pl.col(FORECAST_COLUMN).eq(other=True))[YEAR_COLUMN].min()
    if forecast_from is not None:
        assert isinstance(forecast_from, int)

    if df.paths.index_has_duplicates():
        raise Exception('DataFrame index has duplicates')

    just_cats = [dim.get_original_cat_ids() for dim in dims]
    years = list(df[YEAR_COLUMN].unique().sort())
    idx_names = [dim.original_id for dim in dims] + [YEAR_COLUMN]
    idx_vals = pd.MultiIndex.from_product(just_cats + [years]).to_list()
    idx_df = pl.DataFrame(idx_vals, orient='row', schema={col: df.schema[col] for col in idx_names})

    jdf = idx_df.join(df, how='left', on=idx_names)
    vals: list[float | None] = jdf[col].fill_null(0).to_list()
    goals: list[MetricDimensionGoal] = []

    dm = DimensionalMetric(  # Normalization or grouping is not possible at the moment.
        id=action.id + '_' + col.lower(),
        name=str(action.name),
        dimensions=dims,
        values=vals,
        years=years,
        forecast_from=forecast_from,
        stackable=True,  # Stackability checked already.
        goals=goals,
        normalized_by=None,
        unit=df.get_unit(col),
    )
    return dm
