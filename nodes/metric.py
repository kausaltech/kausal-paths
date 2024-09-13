from __future__ import annotations
from collections import Counter

import functools
from dataclasses import dataclass, field
from typing import List, Optional, TypedDict

import numpy as np
import pandas as pd
import pint
import polars as pl
from colormath.color_objects import sRGBColor, LabColor  # type: ignore
from colormath.color_conversions import convert_color  # type: ignore

from common import polars as ppl
from common.i18n import gettext as _
from .node import Node
from .actions import ActionNode, ActionEfficiency, ActionEfficiencyPair
from .actions.shift import ShiftAction
from .constants import (
    BASELINE_VALUE_COLUMN, FLOW_ID_COLUMN, FLOW_ROLE_COLUMN, FLOW_ROLE_SOURCE,
    FLOW_ROLE_TARGET, FORECAST_COLUMN, NODE_COLUMN, STACKABLE_QUANTITIES,
    VALUE_COLUMN, YEAR_COLUMN,
)
from .exceptions import NodeError
from .goals import NodeGoalsEntry
from .node import NodeMetric
from .simple import AdditiveNode, RelativeNode
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
    node: Optional[Node] = None
    unit: Optional[pint.Unit] = None

    split_values: SplitValues | None = field(init=False)

    def __post_init__(self):
        self.split_values = None

    @staticmethod
    def from_node(node: Node, goal_id: str | None = None):
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
        if node.baseline_values is not None:
            bdf = node.baseline_values
            if node.context.active_normalization:
                _, bdf = node.context.active_normalization.normalize_output(m, bdf)

            if goal:
                bdf = goal.filter_df(bdf)

            bdf_meta = bdf.get_meta()
            if bdf_meta.dim_ids:
                bdf = bdf.paths.sum_over_dims()
            bdf = bdf.select([
                YEAR_COLUMN,
                pl.col(m.column_id).alias(BASELINE_VALUE_COLUMN)
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
            baseline=baseline
        )
        self.split_values = out
        return out

    def get_historical_values(self) -> List[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return []
        return vals['historical']

    def get_forecast_values(self) -> List[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return []
        return vals['forecast']

    def get_baseline_forecast_values(self) -> List[YearlyValue]:
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
    def yearly_cumulative_unit(self) -> Optional[pint.Unit]:
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


@dataclass
class MetricDimension:
    id: str
    original_id: str
    label: str
    categories: list[MetricCategory]
    groups: list[MetricCategoryGroup] = field(default_factory=list)
    help_text: str | None = None

    def get_original_cat_ids(self):
        return [cat.original_id for cat in self.categories]

    def ensure_unique_colors(self):
        color_counts = Counter(cat.color.lower() for cat in self.categories if cat.color is not None)
        color_map: dict[str, list[str]] = {}
        LAB_Kn = 18
        for color, count in color_counts.items():
            if count == 1:
                continue
            rgb = sRGBColor.new_from_rgb_hex(color)
            lab: LabColor = convert_color(rgb, LabColor)
            vals = lab.get_value_tuple()
            start = list(vals)
            start[0] -= LAB_Kn * 1
            end = list(vals)
            if count > 2:
                end[0] += LAB_Kn * 1

            step = 1.0 / (count - 1)
            colors_out = []
            for i in range(count):
                c = []
                t = step * i
                for j in range(3):
                    c.append(start[j] + t * (end[j] - start[j]))
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

    @classmethod
    def from_node(cls, node: Node, metric: NodeMetric | None = None) -> DimensionalMetric | None:
        def make_id(*args: str):
            return ':'.join([node.id, *args])

        if metric is None:
            try:
                m = node.get_default_output_metric()
            except Exception:
                return None
        else:
            # FIXME: Get goals only for the chosen metric
            m = metric

        dims: list[MetricDimension] = []

        def include_as_input(node: Node):
            if isinstance(node, ActionNode):
                return False
            for dim in node.output_dimensions.values():
                if dim.is_internal:
                    return False
            return True

        # Add a functionality to show scenario impacts for nodes rather than outputs.
        tst = node.context.get_parameter_value('show_scenario_impacts', required=False)
        baseline = node.context.get_scenario('baseline')

        # Use inputs nodes as categories for the dimension "Sectors" in some cases.
        input_nodes = [node for node in node.input_nodes if include_as_input(node)]
        if (
            isinstance(node, AdditiveNode) and len(input_nodes) > 1
            and not node.input_dataset_instances and not isinstance(node, RelativeNode)
        ):
            df = node.add_nodes_pl(None, node.input_nodes, keep_nodes=True)

            if tst:
                with baseline.override():
                    ddf = node.add_nodes_pl(None, node.input_nodes, keep_nodes=True)
                df = df.paths.join_over_index(ddf)
                df = df.with_columns((pl.col(VALUE_COLUMN) - pl.col(VALUE_COLUMN + '_right')).alias(VALUE_COLUMN))
                df.drop(VALUE_COLUMN + '_right')

            cats = [MetricCategory(
                id=make_id('node', n.id),
                original_id=n.id,
                label=str(n.short_name or n.name),
                color=n.color,
                order=n.order
            ) for n in node.input_nodes]
            mdim = MetricDimension(
                id=make_id('node', NODE_COLUMN), label=_('Sectors'), categories=cats,
                original_id=NODE_COLUMN,
            )
            mdim.ensure_unique_colors()
            dims.append(mdim)
        else:
            df = node.get_output_pl()

            if tst:
                with baseline.override():
                    ddf = node.get_output_pl()
                df = df.paths.join_over_index(ddf)
                df = df.with_columns((pl.col(VALUE_COLUMN) - pl.col(VALUE_COLUMN + '_right')).alias(VALUE_COLUMN))
                df.drop(VALUE_COLUMN + '_right')

        if node.context.active_normalization:
            normalizer, df = node.context.active_normalization.normalize_output(m, df)
        else:
            normalizer = None

        def make_goal_values(goal: NodeGoalsEntry):
            return [MetricYearlyGoal(
                year=y.year, value=y.value, is_interpolated=y.is_interpolated
            ) for y in goal.get_values()]

        goals: list[MetricDimensionGoal] = []
        if node.goals:
            for goal in node.goals.root:
                cat_ids: list[str] = []
                group_ids = []
                for dim_id, goal_dim in goal.dimensions.items():
                    dim = node.output_dimensions[dim_id]
                    for grp_id in goal_dim.groups:
                        cat_ids += [make_id(dim.id, 'cat', cat.id) for cat in dim.get_cats_for_group(grp_id)]
                        group_ids.append(make_id(dim.id, 'group', grp_id))
                    for cat_id in goal_dim.categories:
                        cat_ids.append(make_id(dim.id, 'cat', cat_id))
                goals.append(MetricDimensionGoal(
                    categories=cat_ids, values=make_goal_values(goal), groups=group_ids
                ))

        for dim_id, dim in node.output_dimensions.items():
            df_cats = set(df[dim_id].unique())

            ordered_groups = []
            group_id_map = {}
            if dim.groups:
                df = df.with_columns(dim.ids_to_groups(pl.col(dim_id).alias('_Groups')))
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

            ordered_cats = []
            for cat in dim.categories:
                if cat.id not in df_cats:
                    continue
                cat_id = make_id(dim.id, 'cat', cat.id)
                ordered_cats.append(MetricCategory(
                    id=cat_id, label=str(cat.label), color=cat.color, order=cat.order,
                    original_id=cat.id, group=group_id_map[cat.group] if cat.group else None
                ))
                if node.goals:
                    ng = node.goals.get_exact_match(dim.id, categories=[cat.id])
                    if ng:
                        goals.append(
                            MetricDimensionGoal(categories=[cat_id], groups=[], values=make_goal_values(ng))
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
            dims.append(mdim)

        forecast_from = df.filter(pl.col(FORECAST_COLUMN).eq(True))[YEAR_COLUMN].min()
        if forecast_from is not None:
            assert isinstance(forecast_from, int)

        if df.paths.index_has_duplicates():
            raise NodeError(node, "DataFrame index has duplicates")

        just_cats = [dim.get_original_cat_ids() for dim in dims]
        years = list(df[YEAR_COLUMN].unique().sort())
        idx_names = [dim.original_id for dim in dims] + [YEAR_COLUMN]
        idx_vals = pd.MultiIndex.from_product(just_cats + [years]).to_list()
        idx_df = pl.DataFrame(idx_vals, orient='row', schema={col: df.schema[col] for col in idx_names})

        jdf = idx_df.join(df, how='left', on=idx_names)
        vals: list[float] = jdf[m.column_id].fill_null(0).to_list()
        stackable = m.quantity in STACKABLE_QUANTITIES
        if isinstance(node, ActionNode) and m.quantity in ('mix',):
            stackable = False
        dm = DimensionalMetric(
            id=node.id, name=str(node.name), dimensions=dims,
            values=vals, years=years, unit=df.get_unit(m.column_id),
            forecast_from=forecast_from, normalized_by=normalizer,
            stackable=stackable, goals=goals,
        )
        return dm

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
        elif col=='Impact':
            dimensions = root.impact_node.output_dimensions.items()

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
            id=action.id,
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
