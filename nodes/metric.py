from __future__ import annotations

import functools
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import pandas as pd
import polars as pl

from paths.const import MODEL_CALC_OP

from common import polars as ppl
from common.i18n import gettext as _

from .actions.action import ActionEfficiency, ActionEfficiencyPair, ActionNode
from .actions.shift import ShiftAction
from .constants import (
    BASELINE_VALUE_COLUMN,
    FLOW_ID_COLUMN,
    FLOW_ROLE_COLUMN,
    FLOW_ROLE_SOURCE,
    FLOW_ROLE_TARGET,
    FORECAST_COLUMN,
    NODE_COLUMN,
    SCENARIO_COLUMN,
    STACKABLE_QUANTITIES,
    VALUE_COLUMN,
    YEAR_COLUMN,
)
from .exceptions import NodeError
from .simple import AdditiveNode, RelativeNode

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pint

    from nodes.dimensions import Dimension
    from nodes.scenario import Scenario

    from .goals import NodeGoalsEntry
    from .node import Node, NodeMetric
    from .units import Unit


@dataclass
class YearlyValue:
    year: int
    value: float


class SplitValues(TypedDict):
    historical: list[YearlyValue]
    forecast: list[YearlyValue]
    baseline: list[YearlyValue]
    cumulative_forecast_value: float | None


@dataclass
class Metric:
    id: str
    name: str
    df: ppl.PathsDataFrame
    node: Node | None = None
    unit: pint.Unit | None = None

    split_values: SplitValues | None = field(init=False)

    def __post_init__(self):
        self.split_values = None

    @staticmethod
    def from_node(node: Node, goal_id: str | None = None) -> None | Metric:  # noqa: C901, PLR0912
        try:
            m = node.get_default_output_metric()
        except Exception:
            return None

        df = node.get_output_pl()
        if FORECAST_COLUMN not in df:
            node.logger.error("Output does not have the Forecast column")
            return None

        if goal_id:
            goal = node.context.instance.get_goals(goal_id)
            df = goal.filter_df(df)
        else:
            goal = None

        if node.context.active_normalization:
            _, df = node.context.active_normalization.normalize_output(m, df)

        if m.column_id != VALUE_COLUMN:
            df = df.rename({m.column_id: VALUE_COLUMN})

        meta = df.get_meta()
        if meta.dim_ids:
            if m.quantity not in STACKABLE_QUANTITIES:
                return None
            df = df.paths.sum_over_dims()

        meta = df.get_meta()
        if 'baseline' in node.context.scenarios:
            bdf = node.get_baseline_values()
            if node.context.active_normalization:
                _, bdf = node.context.active_normalization.normalize_output(m, bdf)

            if goal:
                bdf = goal.filter_df(bdf)

            bdf_meta = bdf.get_meta()
            if bdf_meta.dim_ids:
                bdf = bdf.paths.sum_over_dims()
            bdf = bdf.select([
                YEAR_COLUMN,
                pl.col(m.column_id).alias(BASELINE_VALUE_COLUMN),
            ])
            tdf = df.join(bdf, on=YEAR_COLUMN, how='left').sort(YEAR_COLUMN)
            meta.units[BASELINE_VALUE_COLUMN] = bdf_meta.units[m.column_id]
            df = ppl.to_ppdf(tdf, meta=meta)

        if len(df.filter(df[YEAR_COLUMN].is_duplicated())):
            raise NodeError(node, "Metric has duplicated years")

        nulls = (df[VALUE_COLUMN].is_nan() | df[VALUE_COLUMN].is_null()).sum()
        if nulls:
            print(df)
            raise NodeError(node, "Metric has nans or nulls")

        return Metric(id=node.id, name=str(node.name), unit=df.get_unit(VALUE_COLUMN), node=node, df=df)

    def split_df(self) -> SplitValues | None:
        if self.split_values is not None:
            return self.split_values

        if self.df is None or VALUE_COLUMN not in self.df.columns:
            return None

        df = self.df.drop_nulls()

        hist = []
        forecast = []
        baseline = []
        for row in df.to_dicts():
            is_fc = row[FORECAST_COLUMN]
            val = row[VALUE_COLUMN]
            year = row[YEAR_COLUMN]
            if np.isnan(val):
                raise Exception("Metric %s contains NaN values" % self.id)
            if not is_fc:
                hist.append(YearlyValue(year=year, value=val))
            else:
                bl_val = row.get(BASELINE_VALUE_COLUMN)
                if bl_val is not None:
                    if np.isnan(bl_val):
                        raise Exception("Metric %s baseline contains NaN values" % self.id)
                    baseline.append(YearlyValue(year=year, value=bl_val))
                forecast.append(YearlyValue(year=year, value=val))

        cum_fc = df.filter(pl.col(FORECAST_COLUMN))[VALUE_COLUMN].sum()

        out = SplitValues(
            historical=hist,
            forecast=forecast,
            cumulative_forecast_value=cum_fc,
            baseline=baseline,
        )
        self.split_values = out
        return out

    def get_historical_values(self) -> list[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return []
        return vals['historical']

    def get_forecast_values(self) -> list[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return []
        return vals['forecast']

    def get_baseline_forecast_values(self) -> list[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return []
        return vals['baseline']

    def get_cumulative_forecast_value(self) -> float | None:
        vals = self.split_df()
        if not vals:
            return None
        return vals['cumulative_forecast_value']

    @property
    def yearly_cumulative_unit(self) -> pint.Unit | None:
        if not self.unit:
            return None
        # Check if the unit as a time divisor
        dim = self.unit.dimensionality.get('[time]')
        if dim is None or dim > -1:
            return None
        year_unit = self.unit._REGISTRY('year').units
        return self.unit * year_unit


@dataclass
class MetricCategory:
    id: str
    original_id: str
    label: str
    color: str | None
    order: int | None
    group: str | None = None


@dataclass
class MetricCategoryGroup:
    id: str
    original_id: str
    label: str
    color: str | None
    order: int | None


LabColorVals = tuple[float, float, float]

class DimensionKind(Enum):
    COMMON = 'common'
    NODE = 'node'
    SCENARIO = 'scenario'


@dataclass
class MetricDimension:
    id: str
    original_id: str
    label: str
    categories: list[MetricCategory]
    groups: list[MetricCategoryGroup] = field(default_factory=list)
    help_text: str | None = None
    kind: DimensionKind = DimensionKind.COMMON

    def get_original_cat_ids(self) -> list[str]:
        return [cat.original_id for cat in self.categories]

    def ensure_unique_colors(self):
        from colormath.color_conversions import convert_color  # type: ignore
        from colormath.color_objects import LabColor, sRGBColor  # type: ignore

        color_counts = Counter(cat.color.lower() for cat in self.categories if cat.color is not None)
        color_map: dict[str, list[str]] = {}
        LAB_Kn = 18  # noqa: N806
        for color, count in color_counts.items():
            if count == 1:
                continue
            rgb = sRGBColor.new_from_rgb_hex(color)
            lab: LabColor = convert_color(rgb, LabColor)
            vals = cast(LabColorVals, lab.get_value_tuple())
            start = list(vals)
            start[0] -= LAB_Kn * 1
            end = list(vals)
            if count > 2:
                end[0] += LAB_Kn * 1

            step = 1.0 / (count - 1)
            colors_out = []
            for i in range(count):
                t = step * i
                c = cast(LabColorVals, tuple(start[j] + t * (end[j] - start[j]) for j in range(3)))
                out = LabColor(*c)
                out_rgb: sRGBColor = convert_color(out, sRGBColor)
                out_rgb.rgb_r = out_rgb.clamped_rgb_r
                out_rgb.rgb_g = out_rgb.clamped_rgb_g
                out_rgb.rgb_b = out_rgb.clamped_rgb_b
                colors_out.append(out_rgb.get_rgb_hex())
            color_map[color] = colors_out
        for cat in self.categories:
            if not cat.color:
                continue
            color = cat.color.lower()
            if color not in color_map:
                continue
            cat.color = color_map[color].pop(0)


@dataclass
class MetricYearlyGoal:
    year: int
    value: float
    is_interpolated: bool


@dataclass
class MetricDimensionGoal:
    categories: list[str]
    groups: list[str]
    values: list[MetricYearlyGoal] = field(default_factory=list)


@dataclass
class DimensionalMetric:
    id: str
    name: str
    dimensions: list[MetricDimension]
    values: list[float]
    years: list[int]
    unit: Unit
    stackable: bool
    forecast_from: int | None
    goals: list[MetricDimensionGoal]
    normalized_by: Node | None
    scenarios: tuple[str, ...] = ()

    _index_names: list[list[str]] = field(default_factory=list, init=False)

    @classmethod
    def _make_id(cls, node: Node, *args: str) -> str:
        return ':'.join([node.id, *args])

    @classmethod
    def _create_input_nodes_dimension(cls, node: Node, input_nodes: list[Node]) -> MetricDimension:
        cats = [MetricCategory(
            id=cls._make_id(node, 'node', n.id),
            original_id=n.id,
            label=str(n.short_name or n.name),
            color=n.color,
            order=n.order,
        ) for n in input_nodes]
        mdim = MetricDimension(
            id=cls._make_id(node, 'node', NODE_COLUMN), label=_('Sectors'), categories=cats,
            original_id=NODE_COLUMN,
            kind=DimensionKind.NODE,
        )
        mdim.ensure_unique_colors()
        return mdim

    @classmethod
    def _compute_values(
        cls, node: Node, extra_scenarios: Sequence[Scenario] = (),
    ) -> tuple[ppl.PathsDataFrame, list[MetricDimension]]:
        def include_as_input(node: Node) -> bool:
            if isinstance(node, ActionNode):
                return False
            return all(not dim.is_internal for dim in node.output_dimensions.values())

        extra_dims: list[MetricDimension] = []
        current_scenario_id = node.context.active_scenario.id
        scenario_dim: MetricDimension | None = None
        if extra_scenarios:
            cats: list[MetricCategory] = []
            seen_scenarios = set[str]()
            for scenario in [node.context.active_scenario, *extra_scenarios]:
                if scenario.id in seen_scenarios:
                    continue
                seen_scenarios.add(scenario.id)
                cats.append(MetricCategory(
                    id=cls._make_id(node, 'scenario', scenario.id),
                    original_id=scenario.id,
                    label=str(scenario.name),
                    color=None,
                    order=None,
                ))
            scenario_dim = MetricDimension(
                id=cls._make_id(node, 'scenario', SCENARIO_COLUMN), label=_('Scenarios'), categories=cats,
                original_id=SCENARIO_COLUMN, kind=DimensionKind.SCENARIO,
            )
            extra_dims.append(scenario_dim)

        # Add a functionality to show scenario impacts for nodes rather than outputs.
        # tst = node.context.get_parameter_value('show_scenario_impacts', required=False)
        # baseline = node.context.get_scenario('baseline')

        # Use inputs nodes as categories for the dimension "Sectors" in some cases.
        input_nodes = [input_node for input_node in node.input_nodes if include_as_input(input_node)]
        if (
            not isinstance(node, AdditiveNode)
            or len(input_nodes) <= 1
            or node.input_dataset_instances
            or isinstance(node, RelativeNode)
            or len(input_nodes) != len(node.input_nodes)
        ):
            # Inputs nodes can't reasonably be summed up, so we just use the output
            def get_output_without_input_nodes() -> ppl.PathsDataFrame:
                return node.get_output_pl()
            get_output_func = get_output_without_input_nodes
        else:
            # Get output with input node IDs as categories (in a column `NODE_COLUMN`)
            def get_output_with_input_nodes() -> ppl.PathsDataFrame:
                return node.add_nodes_pl(df=None, nodes=input_nodes, keep_nodes=True)
            get_output_func = get_output_with_input_nodes
            extra_dims.append(cls._create_input_nodes_dimension(node, input_nodes))

        # Get output in currently active scenario
        df = get_output_func()
        if scenario_dim:
            # Compute output in other scenarios
            def add_scenario_column(df: ppl.PathsDataFrame, scenario_id: str) -> ppl.PathsDataFrame:
                return df.with_columns(pl.lit(scenario_id).alias(SCENARIO_COLUMN)).add_to_index(SCENARIO_COLUMN)

            df = add_scenario_column(df, current_scenario_id)
            scenario_dfs: list[ppl.PathsDataFrame] = [df]
            meta = df.get_meta()
            for scenario_cat in scenario_dim.categories:
                scenario = node.context.get_scenario(scenario_cat.original_id)
                if scenario.id == current_scenario_id:
                    continue
                with scenario.override():
                    sdf = add_scenario_column(get_output_func(), scenario.id)
                scenario_dfs.append(sdf)
                assert sdf.get_meta() == meta
            df = ppl.to_ppdf(pl.concat(scenario_dfs), meta=meta)

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

    @classmethod
    def _make_goal_values(cls, goal: NodeGoalsEntry) -> list[MetricYearlyGoal]:
        return [MetricYearlyGoal(
            year=y.year, value=y.value, is_interpolated=y.is_interpolated,
        ) for y in goal.get_values()]

    @classmethod
    def _make_data_dimension(
        cls, node: Node, dim_id: str, dim: Dimension, df: ppl.PathsDataFrame,
    ) -> tuple[MetricDimension, list[MetricDimensionGoal]]:
        def make_id(*args: str) -> str:
            return cls._make_id(node, *args)

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
                ordered_groups.append(MetricCategoryGroup(
                    id=grp_id, label=str(grp.label), color=grp.color, order=grp.order,
                    original_id=grp.id,
                ))
            assert len(ordered_groups) == len(df_groups)

        # Unique category IDs in the output data
        df_cats = set(df[dim_id].unique())

        # Ordered list of categories that are present in the output data
        ordered_cats: list[MetricCategory] = []
        goals: list[MetricDimensionGoal] = []
        for cat in dim.categories:
            if cat.id not in df_cats:
                continue
            cat_id = make_id(dim.id, 'cat', cat.id)
            ordered_cats.append(MetricCategory(
                id=cat_id, label=str(cat.label), color=cat.color, order=cat.order,
                original_id=cat.id, group=group_id_map[cat.group] if cat.group else None,
            ))
            if node.goals:
                # Check if there are goals for this specific dimension category
                ng = node.goals.get_exact_match(dim.id, categories=[cat.id])
                if ng:
                    goals.append(
                        MetricDimensionGoal(categories=[cat_id], groups=[], values=cls._make_goal_values(ng)),
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
        return mdim, goals

    @classmethod
    def _from_node_metric(cls, node: Node, m: NodeMetric, extra_scenarios: Sequence[Scenario]) -> DimensionalMetric:
        def make_id(*args: str) -> str:
            return cls._make_id(node, *args)

        dims: list[MetricDimension] = []

        with node.context.start_span('Compute metric values', op=MODEL_CALC_OP):
            df, dims = cls._compute_values(node, extra_scenarios)

        if node.context.active_normalization:
            normalizer, df = node.context.active_normalization.normalize_output(m, df)
        else:
            normalizer = None

        goals: list[MetricDimensionGoal] = []
        if node.goals:
            for goal in node.goals.root:
                cat_ids: list[str] = []
                group_ids: list[str] = []
                for dim_id, goal_dim in goal.dimensions.items():
                    dim = node.output_dimensions[dim_id]
                    for grp_id in goal_dim.groups:
                        cat_ids += [make_id(dim.id, 'cat', cat.id) for cat in dim.get_cats_for_group(grp_id)]
                        group_ids.append(make_id(dim.id, 'group', grp_id))
                    for cat_id in goal_dim.categories:
                        cat_ids.append(make_id(dim.id, 'cat', cat_id))
                goals.append(MetricDimensionGoal(
                    categories=cat_ids, values=cls._make_goal_values(goal), groups=group_ids,
                ))

        for dim_id, dim in node.output_dimensions.items():
            data_dim, cat_goals = cls._make_data_dimension(node, dim_id, dim, df)
            dims.append(data_dim)
            goals.extend(cat_goals)

        forecast_from = df.filter(pl.col(FORECAST_COLUMN).eq(other=True))[YEAR_COLUMN].min()
        if forecast_from is not None:
            assert isinstance(forecast_from, int)

        if df.paths.index_has_duplicates():
            raise NodeError(node, "DataFrame index has duplicates")

        years = df.select(YEAR_COLUMN).unique().sort(by=YEAR_COLUMN)
        idx_names = [dim.original_id for dim in dims] + [YEAR_COLUMN]
        idx_dfs = [pl.LazyFrame(dim.get_original_cat_ids(), schema=[dim.original_id], orient='row') for dim in dims] + [
            pl.LazyFrame(years, schema=['Year']),
        ]
        idf_lazy = idx_dfs[0]
        for d in idx_dfs[1:]:
            idf_lazy = idf_lazy.join(d, how='cross')
        idx_df = idf_lazy.collect()
        idx_df = idx_df.select(idx_names)

        idx_exprs = [pl.col(n).cast(pl.Utf8) if n != YEAR_COLUMN else pl.col(n) for n in idx_names]
        df = df.select([*idx_exprs, VALUE_COLUMN, FORECAST_COLUMN]).sort(by=idx_exprs)
        jdf = idx_df.join(df, how='left', on=idx_exprs, validate='1:1')
        vals: list[float] = jdf[m.column_id].fill_null(0).to_list()
        stackable = m.quantity in STACKABLE_QUANTITIES
        if isinstance(node, ActionNode) and m.quantity in ('mix',):
            stackable = False

        dm = DimensionalMetric(
            id=node.id, name=str(node.name), dimensions=dims,
            values=vals, years=years[YEAR_COLUMN].to_list(), unit=df.get_unit(m.column_id),
            forecast_from=forecast_from, normalized_by=normalizer,
            stackable=stackable, goals=goals,
        )
        return dm


    @classmethod
    def from_node(
        cls, node: Node, metric: NodeMetric | None = None, extra_scenarios: Sequence[Scenario] = (),
    ) -> DimensionalMetric | None:

        if metric is None:
            try:
                m = node.get_default_output_metric()
            except Exception:
                return None
        else:
            # FIXME: Get goals only for the chosen metric
            m = metric

        with node.context.start_span('Dimension metric for: %s' % node.id, op=MODEL_CALC_OP):
            return cls._from_node_metric(node, m, extra_scenarios)

    @classmethod
    def from_action_efficiency(
        cls, action_efficiency: ActionEfficiency, root: ActionEfficiencyPair, col: str,
    ) -> DimensionalMetric:
        action = action_efficiency.action
        def make_id(*args: str) -> str:
            return ':'.join([action.id, *args])
        df = action_efficiency.df
        if col=='Cost':
            dimensions = root.cost_node.output_dimensions.items()
            if root.invert_cost:
                df = df.with_columns((pl.col(col) * pl.lit(-1.0)).alias(col))
        elif col=='Impact':
            dimensions = root.impact_node.output_dimensions.items()
            if root.invert_impact:
                df = df.with_columns((pl.col(col) * pl.lit(-1.0)).alias(col))
        else:
            raise ValueError("Unknown column %s" % col)

        dims: list[MetricDimension] = []

        for dim_id, dim in dimensions:
            df_cats = set(df[dim_id].unique())
            ordered_cats = []
            for cat in dim.categories:
                if cat.id not in df_cats:
                    continue
                cat_id = make_id(dim.id, 'cat', cat.id)
                ordered_cats.append(MetricCategory(
                    id=cat_id, label=str(cat.label), color=cat.color, order=cat.order,
                    original_id=cat.id,
                ))

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
            raise Exception("DataFrame index has duplicates")

        just_cats = [dim.get_original_cat_ids() for dim in dims]
        years = list(df[YEAR_COLUMN].unique().sort())
        idx_names = [dim.original_id for dim in dims] + [YEAR_COLUMN]
        idx_vals = pd.MultiIndex.from_product(just_cats + [years]).to_list()
        idx_df = pl.DataFrame(idx_vals, orient='row', schema={col: df.schema[col] for col in idx_names})

        jdf = idx_df.join(df, how='left', on=idx_names)
        vals: list[float] = jdf[col].fill_null(0).to_list()
        goals: list[MetricDimensionGoal] = []

        dm = DimensionalMetric(  # Normalization or grouping is not possible at the moment.
            id=action.id + '_' + col.lower(),
            name=str(action.name),
            dimensions=dims,
            values=vals,
            years=years,
            unit=df.get_unit(col),
            forecast_from = forecast_from,
            stackable=True,  # Stackability checked already.
            goals=goals,
            normalized_by=None,
        )
        return dm


@dataclass
class FlowNode:
    id: str
    label: str
    color: str | None = None


@dataclass
class FlowLinks:
    year: int
    is_forecast: bool
    sources: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    values: list[float | None] = field(default_factory=list)
    absolute_source_values: list[float] = field(default_factory=list)


@dataclass
class DimensionalFlow:
    id: str
    nodes: list[FlowNode]
    sources: list[str]
    unit: Unit
    links: list[FlowLinks]

    @classmethod
    def from_action_node(cls, node: ActionNode) -> None | DimensionalFlow:  # noqa: C901, PLR0915
        if not isinstance(node, ShiftAction):
            return None

        context = node.context

        df = node.compute_effect_flow()
        source_rows = (
            df.filter(pl.col(FLOW_ROLE_COLUMN) == FLOW_ROLE_SOURCE)
            .drop([YEAR_COLUMN, VALUE_COLUMN]).unique()
        )

        source_nodes: dict[str, Node] = {node_id: node.context.get_node(node_id) for node_id in source_rows[NODE_COLUMN].unique()}

        # By node id
        source_dfs: dict[str, ppl.PathsDataFrame] = {}
        # By flow node id
        source_values: dict[str, dict[int, float]] = {}

        dims = []
        dim_cats = {}
        for dim_id in df.primary_keys:
            if dim_id not in context.dimensions:
                continue
            dim = context.dimensions[dim_id]
            dims.append(dim)
            dim_cats[dim_id] = dim.cat_map

        flow_nodes: dict[str, FlowNode] = {}

        def get_flow_node(row: dict, is_source: bool) -> FlowNode:
            path_parts = []
            label_parts = []

            node = context.nodes[row[NODE_COLUMN]]
            path_parts.append(node.id)
            # If the node shares dimensions, use the dimension labels
            # instead of the node name.
            # FIXME: This might not work in some cases
            for dim in dims:
                if dim.id in node.output_dimensions:
                    break
            else:
                if len(source_nodes) > 1:
                    label_parts.append(str(node.short_name) if node.short_name else str(node.name))

            sdf_exprs = []
            for dim in dims:
                cat_id = row[dim.id]
                if not cat_id:
                    continue
                cat = dim_cats[dim.id][cat_id]
                path_parts.append(cat.id)
                label_parts.append(str(cat.label))
                sdf_exprs.append(pl.col(dim.id).eq(cat.id))

            flow_node_id = ':'.join(path_parts)
            if flow_node_id in flow_nodes:
                return flow_nodes[flow_node_id]

            flow_nodes[flow_node_id] = FlowNode(id=flow_node_id, label=' / '.join(label_parts))
            if is_source:
                sdf = source_dfs.get(node.id)
                if sdf is None:
                    sdf = source_dfs[node.id] = node.get_output_pl()
                val_col = node.get_default_output_metric().column_id
                if sdf_exprs:
                    sdf = sdf.filter(functools.reduce(lambda a, b: a & b, sdf_exprs))
                sdf = sdf.select([YEAR_COLUMN, val_col])
                assert not sdf.paths.index_has_duplicates()
                assert flow_node_id not in source_values
                source_values[flow_node_id] = {x[YEAR_COLUMN]: x[val_col] for x in sdf.to_dicts()}

            return flow_nodes[flow_node_id]

        sources = {}
        source_list = []
        for row in source_rows.to_dicts():
            flow_node = get_flow_node(row, is_source=True)
            if flow_node not in source_list:
                source_list.append(flow_node.id)
            flow_id = row[FLOW_ID_COLUMN]
            assert flow_id not in sources
            sources[flow_id] = flow_node

        links: dict[int, FlowLinks] = {}

        first_forecast_year: int = df.filter(pl.col(FORECAST_COLUMN))[YEAR_COLUMN].min() # type: ignore
        year = first_forecast_year - 1
        links[first_forecast_year - 1] = FlowLinks(
            year=year,
            is_forecast=False,
            absolute_source_values=[source_values[src_id][year] for src_id in source_list],
        )

        target_rows = df.filter(pl.col(FLOW_ROLE_COLUMN) == FLOW_ROLE_TARGET)
        for row in target_rows.to_dicts():
            flow_id = row[FLOW_ID_COLUMN]
            source = sources[flow_id]
            year = row[YEAR_COLUMN]
            link = links.get(year)
            if link is None:
                link = FlowLinks(year=year, is_forecast=True)
                links[year] = link
                link.absolute_source_values = [source_values[src_id][year] for src_id in source_list]
            target = get_flow_node(row, is_source=False)
            link.sources.append(source.id)
            link.targets.append(target.id)
            link.values.append(row[VALUE_COLUMN])

        assert node.unit is not None
        return DimensionalFlow(
            id=node.id,
            nodes=list(flow_nodes.values()),
            unit=node.unit,
            links=sorted(links.values(), key=lambda x: x.year),
            sources=source_list,
        )
