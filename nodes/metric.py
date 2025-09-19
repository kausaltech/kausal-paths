from __future__ import annotations

import functools
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from pydantic import BaseModel, Field

import numpy as np
import polars as pl
import sentry_sdk

from common import polars as ppl

from .actions.shift import ShiftAction
from .constants import (
    BASELINE_VALUE_COLUMN,
    FLOW_ID_COLUMN,
    FLOW_ROLE_COLUMN,
    FLOW_ROLE_SOURCE,
    FLOW_ROLE_TARGET,
    FORECAST_COLUMN,
    NODE_COLUMN,
    STACKABLE_QUANTITIES,
    VALUE_COLUMN,
    YEAR_COLUMN,
)
from .exceptions import NodeError
from .units import Unit  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pint

    from nodes.scenario import Scenario
    from nodes.visualizations import VisualizationNodeOutput

    from .actions.action import ActionImpact, ActionNode, ImpactOverview
    from .node import Node, NodeMetric


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
            node.logger.error('Output does not have the Forecast column')
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
            bdf = bdf.select(
                [
                    YEAR_COLUMN,
                    pl.col(m.column_id).alias(BASELINE_VALUE_COLUMN),
                ]
            )
            tdf = df.join(bdf, on=YEAR_COLUMN, how='left').sort(YEAR_COLUMN)
            meta.units[BASELINE_VALUE_COLUMN] = bdf_meta.units[m.column_id]
            df = ppl.to_ppdf(tdf, meta=meta)

        if len(df.filter(df[YEAR_COLUMN].is_duplicated())):
            raise NodeError(node, 'Metric has duplicated years')

        nulls = (df[VALUE_COLUMN].is_nan() | df[VALUE_COLUMN].is_null()).sum()
        if nulls:
            print(df)
            raise NodeError(node, 'Metric has nans or nulls')

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
                raise Exception('Metric %s contains NaN values' % self.id)
            if not is_fc:
                hist.append(YearlyValue(year=year, value=val))
            else:
                bl_val = row.get(BASELINE_VALUE_COLUMN)
                if bl_val is not None:
                    if np.isnan(bl_val):
                        raise Exception('Metric %s baseline contains NaN values' % self.id)
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


class MetricCategory(BaseModel):
    id: str
    original_id: str
    label: str
    color: str | None
    order: int | None
    group: str | None = None


class MetricCategoryGroup(BaseModel):
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


class MetricDimension(BaseModel):
    id: str
    original_id: str
    label: str
    categories: list[MetricCategory]
    groups: list[MetricCategoryGroup] = Field(default_factory=list)
    help_text: str | None = None
    kind: DimensionKind = DimensionKind.COMMON

    def get_original_cat_ids(self) -> list[str]:
        return [cat.original_id for cat in self.categories]

    @cached_property
    def cats_by_original_id(self) -> dict[str, MetricCategory]:
        return {cat.original_id: cat for cat in self.categories}

    @cached_property
    def cats_by_id(self) -> dict[str, MetricCategory]:
        return {cat.id: cat for cat in self.categories}

    def _cat_ids_by_prop(self, prop: Literal['original_id', 'id']) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for cat in self.categories:
            if prop == 'original_id':
                val = cat.original_id
            else:
                val = cat.id
            out.setdefault(val, set()).add(cat.id)
        return out

    @cached_property
    def cat_ids_by_group(self) -> dict[str, set[str]]:
        return self._cat_ids_by_prop('id')

    @cached_property
    def original_cat_ids_by_group(self) -> dict[str, set[str]]:
        return self._cat_ids_by_prop('original_id')

    def ensure_unique_colors(self):
        from colormath.color_conversions import convert_color  # type: ignore
        from colormath.color_objects import LabColor, sRGBColor  # type: ignore

        color_counts = Counter(cat.color.lower() for cat in self.categories if cat.color is not None)
        color_map: dict[str, list[str]] = {}
        LAB_Kn = 18
        for color, count in color_counts.items():
            if count == 1:
                continue
            rgb = sRGBColor.new_from_rgb_hex(color)
            lab: LabColor = convert_color(rgb, LabColor)
            vals = cast('LabColorVals', lab.get_value_tuple())
            start = list(vals)
            start[0] -= LAB_Kn * 1
            end = list(vals)
            if count > 2:
                end[0] += LAB_Kn * 1

            step = 1.0 / (count - 1)
            colors_out = []
            for i in range(count):
                t = step * i
                c = cast('LabColorVals', tuple(start[j] + t * (end[j] - start[j]) for j in range(3)))
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


class MetricYearlyGoal(BaseModel):
    year: int
    value: float
    is_interpolated: bool


class MetricDimensionGoal(BaseModel):
    categories: list[str]
    groups: list[str]
    values: list[MetricYearlyGoal] = Field(default_factory=list)


class NormalizerNode(BaseModel):
    id: str
    name: str


@dataclass
class MetricData:
    forecast_from: int | None
    years: list[int]
    values: list[float]


class DimensionalMetric(BaseModel):
    id: str
    name: str
    dimensions: list[MetricDimension]
    values: list[float]
    years: list[int]
    stackable: bool
    forecast_from: int | None
    goals: list[MetricDimensionGoal]
    normalized_by: NormalizerNode | None
    unit: Unit
    measure_datapoint_years: list[int] = Field(default_factory=list)

    def to_df(self, drop_single_cat_dims: bool = False) -> ppl.PathsDataFrame:
        idx_df = self.generate_index_df(self.dimensions, self.years)
        data = pl.DataFrame(self.values, schema=[VALUE_COLUMN])
        df = pl.concat([idx_df, data], how='horizontal')
        casts = []
        for dim in self.dimensions:
            casts.append(pl.col(dim.original_id).cast(pl.Categorical))  # noqa: PERF401
        df = df.with_columns(casts)
        dim_ids = [dim.original_id for dim in self.dimensions]
        pdf = ppl.to_ppdf(df, meta=ppl.DataFrameMeta(units={VALUE_COLUMN: self.unit}, primary_keys=[*dim_ids, YEAR_COLUMN]))
        if drop_single_cat_dims:
            drop_cols = [dim.original_id for dim in self.dimensions if len(dim.categories) <= 1]
            if drop_cols:
                pdf = pdf.drop(drop_cols)
        return pdf

    @classmethod
    def generate_index_df(cls, dims: list[MetricDimension], years: list[int]) -> pl.DataFrame:
        idx_names = [dim.original_id for dim in dims] + [YEAR_COLUMN]
        idx_dfs = [pl.LazyFrame(dim.get_original_cat_ids(), schema=[dim.original_id], orient='row') for dim in dims] + [
            pl.LazyFrame(years, schema=[YEAR_COLUMN]),
        ]
        idf_lazy = idx_dfs[0]
        for d in idx_dfs[1:]:
            idf_lazy = idf_lazy.join(d, how='cross')
        idx_df = idf_lazy.collect()
        idx_df = idx_df.select(idx_names)
        return idx_df

    @classmethod
    def from_node(
        cls,
        node: Node,
        metric: NodeMetric | None = None,
        extra_scenarios: Sequence[Scenario] = (),
    ) -> DimensionalMetric | None:
        from .metric_gen import metric_from_node

        with sentry_sdk.start_span(name='Metric from node %s' % node.id, op='model.metric'):
            return metric_from_node(node, metric, extra_scenarios)

    @classmethod
    def from_visualization(cls, node: Node, visualization: VisualizationNodeOutput) -> DimensionalMetric | None:
        from .metric_gen import metric_from_visualization

        with sentry_sdk.start_span(name='Metric from node %s visualization' % node.id, op='model.metric'):
            return metric_from_visualization(node, visualization)

    @classmethod
    def from_action_impact(cls, action_impact: ActionImpact, root: ImpactOverview, col: str) -> DimensionalMetric | None:
        from .metric_gen import from_action_impact

        return from_action_impact(action_impact, root, col)

    def get_dimension(self, dim_id: str) -> MetricDimension:
        for dim in self.dimensions:
            if dim_id in (dim.id, dim.original_id):
                return dim
        raise ValueError(f'Dimension {dim_id} not found')

    def plot(self, dim_id: str | None = None):
        import altair as alt  # type: ignore

        df = self.to_df(drop_single_cat_dims=True).with_columns(pl.col('Year').cast(pl.Utf8))
        x = alt.X(field='Year', type='temporal')
        y = alt.Y('Value:Q', title=str(self.unit))
        kwargs: dict[str, Any] = {}
        dims = list(self.dimensions)
        width = 300
        for dim in self.dimensions:
            if dim.kind == DimensionKind.SCENARIO:
                kwargs['column'] = dim.original_id
                dims.remove(dim)
                width *= len(dim.categories)
        if not dim_id and len(dims) > 1:
            dim_id = dims[0].id
        if dim_id:
            dim = self.get_dimension(dim_id)
            scale = alt.Scale(domain=dim.get_original_cat_ids(), range=[cat.color or '' for cat in dim.categories])
            color = alt.Color(field=dim.original_id, type='nominal', scale=scale)
            other_dims = [d for d in df.dim_ids if d not in (dim.original_id, YEAR_COLUMN)]
            if other_dims:
                df = df.paths.sum_over_dims(other_dims)
            kwargs['color'] = color
        else:
            color = None
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x=x, y=y, **kwargs)
            .properties(
                title=self.name,
                width=width,
            )
        )
        return chart.interactive()


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
        source_rows = df.filter(pl.col(FLOW_ROLE_COLUMN) == FLOW_ROLE_SOURCE).drop([YEAR_COLUMN, VALUE_COLUMN]).unique()

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

        first_forecast_year: int = df.filter(pl.col(FORECAST_COLUMN))[YEAR_COLUMN].min()  # type: ignore
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
