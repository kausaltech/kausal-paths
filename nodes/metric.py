from __future__ import annotations

import numpy as np
from itertools import groupby
from typing import Any, Dict, List, Optional, Union, TypedDict
from dataclasses import dataclass, field

import pandas as pd
import pint
import polars as pl

from common import polars as ppl
from nodes import Node
from nodes.actions import ActionNode
from nodes.actions.shift import ShiftAction
from nodes.constants import (
    ACTIVITY_QUANTITIES, BASELINE_VALUE_COLUMN, DEFAULT_METRIC,
    FLOW_ROLE_COLUMN, FLOW_ROLE_TARGET, FLOW_ROLE_SOURCE, FLOW_ID_COLUMN,
    FORECAST_COLUMN, NODE_COLUMN, STACKABLE_QUANTITIES, VALUE_COLUMN, YEAR_COLUMN
)
from nodes.exceptions import NodeError
from nodes.units import Unit


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
    def from_node(node: Node):
        df = node.get_output_pl()
        if VALUE_COLUMN not in df.columns:
            return None
        if len(node.output_metrics) > 1 and DEFAULT_METRIC not in node.output_metrics:
            return None

        if len(node.output_metrics) == 1:
            m = list(node.output_metrics.values())[0]
        else:
            m = node.output_metrics[DEFAULT_METRIC]

        if m.column_id != VALUE_COLUMN:
            df = df.rename({m.column_id: VALUE_COLUMN})

        meta = df.get_meta()
        if meta.dim_ids:
            if m.quantity not in ACTIVITY_QUANTITIES:
                return None
            df = df.paths.sum_over_dims()

        meta = df.get_meta()
        if node.baseline_values is not None:
            bdf = node.baseline_values
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

        return Metric(id=node.id, name=str(node.name), unit=node.unit, node=node, df=df)

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
        if self.unit.dimensionality.get('[time]') > -1:
            return None
        year_unit = self.unit._REGISTRY('year').units
        return self.unit * year_unit


@dataclass
class MetricCategory:
    id: str
    label: str


@dataclass
class MetricDimension:
    id: str
    label: str
    categories: list[MetricCategory]

    def get_cat_ids(self):
        return [cat.id for cat in self.categories]


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
    normalized_by: Node | None

    @classmethod
    def from_node(cls, node: Node) -> DimensionalMetric | None:
        df = node.get_output_pl()

        try:
            m = node.get_default_output_metric()
        except Exception:
            return None

        if node.context.active_normalization:
            normalizer, df = node.context.active_normalization.normalize_output(m, df)
        else:
            normalizer = None

        dims = []
        for dim_id, dim in node.output_dimensions.items():
            if dim.groups:
                df = df.with_columns(dim.ids_to_groups(pl.col(dim_id)))
                meta = df.get_meta()
                gdf = df.groupby(df.primary_keys, maintain_order=True).agg([pl.sum(m.column_id), pl.first(FORECAST_COLUMN)])
                df = ppl.to_ppdf(gdf, meta=meta)
                groups = set(df[dim_id].unique())
                ordered_groups = []
                for grp in dim.groups:
                    if grp.id in groups:
                        ordered_groups.append(MetricCategory(id=grp.id, label=str(grp.label)))
                assert len(groups) == len(ordered_groups)
                ordered_cats = ordered_groups
            else:
                cats = set(df[dim_id].unique())
                ordered_cats = []
                for cat in dim.categories:
                    if cat.id in cats:
                        ordered_cats.append(MetricCategory(id=cat.id, label=str(cat.label)))
                assert len(cats) == len(ordered_cats)
            mdim = MetricDimension(id=dim.id, label=str(dim.label), categories=ordered_cats)
            dims.append(mdim)

        forecast_from = df.filter(pl.col(FORECAST_COLUMN) == True)[YEAR_COLUMN].min()
        assert isinstance(forecast_from, int)

        if df.paths.index_has_duplicates():
            raise NodeError(node, "DataFrame index has duplicates")

        just_cats = [dim.get_cat_ids() for dim in dims]
        years = list(df[YEAR_COLUMN].unique().sort())
        idx_names = [dim.id for dim in dims] + [YEAR_COLUMN]
        idx_vals = pd.MultiIndex.from_product(just_cats + [years]).to_list()
        idx_df = pl.DataFrame(idx_vals, orient='row', schema={col: df.schema[col] for col in idx_names})

        jdf = idx_df.join(df, how='left', on=idx_names)
        vals: list[float] = jdf[m.column_id].to_list()
        dm = DimensionalMetric(
            id=node.id, name=str(node.name), dimensions=dims,
            values=vals, years=years, unit=df.get_unit(m.column_id),
            forecast_from=forecast_from, normalized_by=normalizer,
            stackable=m.quantity in STACKABLE_QUANTITIES
        )
        return dm


@dataclass
class FlowNode:
    id: str
    label: str


@dataclass
class FlowLinks:
    year: int
    is_forecast: bool
    sources: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    values: list[float | None] = field(default_factory=list)


@dataclass
class DimensionalFlow:
    id: str
    nodes: list[FlowNode]
    unit: Unit
    links: list[FlowLinks]

    @classmethod
    def from_action_node(cls, node: ActionNode):
        if not isinstance(node, ShiftAction):
            return None

        context = node.context

        #flow_nodes = []
        df = node.compute_effect_flow()
        dims = []
        dim_cats = {}
        for dim_id in df.primary_keys:
            if dim_id not in context.dimensions:
                continue
            dim = context.dimensions[dim_id]
            dims.append(dim)
            dim_cats[dim_id] = dim.cat_map

        nodes: dict[str, FlowNode] = {}

        def get_node(row):
            path_parts = []
            label_parts = []
            if NODE_COLUMN in row:
                node = context.nodes[row[NODE_COLUMN]]
                path_parts.append(node.id)
                label_parts.append(str(node.name))
            else:
                node = None

            cats = []
            for dim in dims:
                cat_id = row[dim.id]
                if not cat_id:
                    continue
                cat = dim_cats[dim.id][cat_id]
                path_parts.append(cat.id)
                label_parts.append(str(cat.label))

            node_id = ':'.join(path_parts)
            if node_id in nodes:
                return nodes[node_id]
            nodes[node_id] = FlowNode(id=node_id, label=' / '.join(label_parts))
            return nodes[node_id]

        source_rows = (
            df.filter(pl.col(FLOW_ROLE_COLUMN) == FLOW_ROLE_SOURCE)
            .drop([YEAR_COLUMN, VALUE_COLUMN]).unique()
        )
        sources = {}
        for row in source_rows.to_dicts():
            flow_node = get_node(row)
            flow_id = row[FLOW_ID_COLUMN]
            assert flow_id not in sources
            sources[flow_id] = flow_node

        links: dict[int, FlowLinks] = {}
        target_rows = df.filter(pl.col(FLOW_ROLE_COLUMN) == FLOW_ROLE_TARGET)
        for row in target_rows.to_dicts():
            flow_id = row[FLOW_ID_COLUMN]
            source = sources[flow_id]
            year = row[YEAR_COLUMN]
            link = links.get(year)
            if link is None:
                link = FlowLinks(year=year, is_forecast=True)
                links[year] = link
            target = get_node(row)
            link.sources.append(source.id)
            link.targets.append(target.id)
            link.values.append(row[VALUE_COLUMN])

        assert node.unit is not None
        return DimensionalFlow(
            id=node.id,
            nodes=list(nodes.values()),
            unit=node.unit,
            links=list(links.values())
        )
