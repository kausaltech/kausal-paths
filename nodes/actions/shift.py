from dataclasses import dataclass
from typing import ClassVar, Iterable, Any, List

from pydantic import BaseModel, RootModel, model_validator, Field, validator
import pandas as pd
import polars as pl

from nodes.constants import (
    FLOW_ROLE_COLUMN, FLOW_ROLE_SOURCE, FLOW_ROLE_TARGET, FLOW_ID_COLUMN, FORECAST_COLUMN, NODE_COLUMN,
    VALUE_COLUMN, YEAR_COLUMN
)
from nodes.exceptions import NodeError
from nodes.node import Node
from nodes.units import Unit
from params.param import ValidationError
from common import polars as ppl
from params import Parameter, ParameterWithUnit

from .action import ActionNode


class ShiftTarget(BaseModel):
    node: str | int | None = None
    # dimension_id -> category_id
    categories: dict[str, str] = Field(default_factory=dict)


class ShiftAmount(BaseModel):
    year: int
    source_amount: float
    dest_amounts: list[float]


class ShiftEntry(BaseModel):
    source: ShiftTarget
    dests: list[ShiftTarget]
    amounts: list[ShiftAmount]

    # @validator('amounts')
    # def enough_years(cls, v):
    #     if len(v) < 2:
    #         raise ValueError("Must supply values for at least two years")
    #     return v

    @model_validator(mode='after')
    def dimensions_must_match(self):
        existing_dims: set = set()
        def validate_target(target: ShiftTarget):
            dims = set(target.categories.keys())
            if not existing_dims:
                existing_dims.update(dims)
                return
            if dims != existing_dims:
                raise ValueError("Dimensions for yearly values for each target be equal")

        validate_target(self.source)
        for dest in self.dests:
            validate_target(dest)
        return self

    def make_index(
        self, output_nodes: list[Node], extra_level: str | None = None, extra_level_values: Iterable | None = None,
    ) -> pd.MultiIndex:
        dims: dict[str, set[str]] = {dim: set() for dim in list(self.source.categories.keys())}
        nodes: set[str] = set()

        def get_node_id(node: str | int | None):
            if isinstance(node, str):
                return node
            if node is None:
                nr = 0
            else:
                nr = node
            return output_nodes[nr].id

        nodes.add(get_node_id(self.source.node))
        for dim, cat in self.source.categories.items():
            dims[dim].add(cat)
        for dest in self.dests:
            nodes.add(get_node_id(dest.node))
            for dim, cat in dest.categories.items():
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


class ShiftParameterValue(RootModel):
    root: List[ShiftEntry]


@dataclass
class ShiftParameter(ParameterWithUnit, Parameter):
    value: ShiftParameterValue | None = None

    def __post_init__(self, unit_str: str | None = None):
        self._init_unit(unit_str)
        super().__post_init__()

    def serialize_value(self) -> Any:
        return super().serialize_value()

    def clean(self, value: Any) -> ShiftParameterValue:
        if not isinstance(value, list):
            raise ValidationError(self, "Input must be a list")

        try:
            return ShiftParameterValue.model_validate(value)
        except:
            from rich import print
            print(value)
            raise


class ShiftAction(ActionNode):
    allowed_parameters: ClassVar[list[Parameter]] = ActionNode.allowed_parameters + [
        ShiftParameter(local_id='shift')
    ]

    def _compute_one(self, flow_id: str, param: ShiftEntry, unit: Unit):
        amounts = sorted(param.amounts, key=lambda x: x.year)
        data = [[a.year, a.source_amount, *a.dest_amounts] for a in param.amounts]
        if len(data) == 1:
            row = data[0].copy()
            row[0] = self.get_end_year()
            data.append(row)
        dest_cols = ['Dest%d' % idx for idx in range(len(param.dests))]
        cols = [YEAR_COLUMN, 'Source', *dest_cols]

        zdf = pl.DataFrame(data, schema=cols, orient='row')
        df = zdf.with_columns(pl.sum_horizontal(dest_cols).alias('DestSum')).lazy()
        df = df.with_columns(pl.col(dest_cols) / pl.col('DestSum') * (pl.col('Source') * -1)).drop('DestSum')

        years = pl.LazyFrame(pl.int_range(amounts[0].year, self.get_end_year() + 1, eager=True), schema=[YEAR_COLUMN])
        df = years.join(df.lazy(), how='left', on=YEAR_COLUMN)
        #dupes = df.filter(pl.col(YEAR_COLUMN).is_duplicated())
        #if len(dupes):
        #    raise NodeError(self, "Duplicate rows")
        df = df.groupby(YEAR_COLUMN).agg([pl.first(col) for col in ('Source', *dest_cols)]).sort(YEAR_COLUMN)
        df = df.with_columns(pl.col(col).interpolate().fill_null(0.0) for col in ('Source', *dest_cols))

        value_cols = [col for col in df.columns if col != YEAR_COLUMN]
        if self.is_enabled():
            df = df.with_columns([pl.cumsum(col).alias(col) for col in value_cols])
        else:
            df = df.with_columns([pl.lit(float(0)).alias(col) for col in value_cols])

        targets = [('Source', param.source), *[('Dest%d' % idx, param.dests[idx]) for idx in range(len(param.dests))]]

        all_dims = set(list(param.source.categories.keys()))
        for dest in param.dests:
            all_dims.update(list(dest.categories.keys()))

        def get_node(node_id: str | int | None) -> Node:
            if isinstance(node_id, str):
                for node in self.output_nodes:
                    if node.id == node_id:
                        return node
                else:
                    raise NodeError(self, "Node %s not listed in output_nodes" % node_id)

            if node_id is None:
                nr = 0
            else:
                nr = node_id
            return self.output_nodes[nr]

        def make_target_df(df: pl.LazyFrame, target: ShiftTarget, valuecol: str):
            target_dims = set(target.categories.keys())
            null_dims = all_dims - target_dims
            node = get_node(target.node)
            target_cats = sorted(target.categories.items(), key=lambda x: x[0])

            for dim_id, cat_id in target_cats:
                if dim_id not in node.input_dimensions:
                    raise NodeError(self, "Dimension %s not found in node %s input dimensions" % (dim_id, node.id))
                dim = node.input_dimensions[dim_id]
                if cat_id not in dim.cat_map:
                    raise NodeError(self, "Category %s not found in node %s input dimension %s" % (cat_id, node.id, dim.id))

            cat_exprs = [pl.lit(cat).alias(dim) for dim, cat in target_cats]

            if not self.is_enabled():
                value_expr = pl.lit(0.0)
            else:
                value_expr = pl.col(valuecol)
            tdf = df.select([
                pl.col(YEAR_COLUMN),
                pl.lit(node.id).alias(NODE_COLUMN),
                pl.lit(FLOW_ROLE_SOURCE if valuecol == 'Source' else FLOW_ROLE_TARGET).alias(FLOW_ROLE_COLUMN),
                *cat_exprs,
                *[pl.lit(None).cast(pl.Utf8).alias(null_dim) for null_dim in null_dims],
                value_expr.alias(VALUE_COLUMN),
            ])
            return tdf

        df = df.collect()
        dfs = [make_target_df(df.lazy(), target, col) for col, target in targets]
        df = pl.concat(dfs).sort(YEAR_COLUMN)
        #df = df.groupby([NODE_COLUMN, *all_dims, YEAR_COLUMN]).agg(pl.sum(VALUE_COLUMN)).sort(YEAR_COLUMN)
        df = df.with_columns([
            pl.lit(True).alias(FORECAST_COLUMN),
            pl.lit(flow_id).alias(FLOW_ID_COLUMN),
        ])
        zdf = df.collect()
        meta = ppl.DataFrameMeta(
            units={VALUE_COLUMN: unit},
            primary_keys=[FLOW_ID_COLUMN, YEAR_COLUMN, NODE_COLUMN, *all_dims]
        )
        ret = ppl.to_ppdf(zdf, meta=meta)
        return ret

    def compute_effect_flow(self) -> ppl.PathsDataFrame:
        po = self.get_parameter('shift')
        value = po.get()
        assert isinstance(value, ShiftParameterValue)

        dfs = []
        for idx, entry in enumerate(value.root):
            df = self._compute_one(str(idx), entry, po.get_unit())
            dfs.append(df)

        meta = dfs[0].get_meta()
        sdf = pl.concat(dfs)
        df = ppl.to_ppdf(sdf, meta=meta)
        return df

    def compute_effect(self) -> ppl.PathsDataFrame:
        df = self.compute_effect_flow().drop(columns=[FLOW_ID_COLUMN, FLOW_ROLE_COLUMN])
        meta = df.get_meta()
        sdf = df.groupby(df.primary_keys).agg([pl.sum(VALUE_COLUMN), pl.first(FORECAST_COLUMN)])
        sdf = sdf.sort(meta.primary_keys)
        df = ppl.to_ppdf(sdf, meta=meta)
        df = self.implement_multiplier(df)
        return df
