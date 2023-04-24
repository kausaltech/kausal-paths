from dataclasses import dataclass
from typing import ClassVar, Iterable, Any, List

from pydantic import BaseModel, root_validator, Field, validator
import pandas as pd
import polars as pl

from nodes.constants import (
    FLOW_ROLE_COLUMN, FLOW_ROLE_SOURCE, FLOW_ROLE_TARGET, FLOW_ID_COLUMN, FORECAST_COLUMN, NODE_COLUMN,
    VALUE_COLUMN, YEAR_COLUMN
)
from nodes.node import Node
from nodes.units import Unit
from params.param import NumberParameter, ValidationError
from common import polars as ppl
from params import Parameter, ParameterWithUnit

from .action import ActionNode


class ReduceAmount(BaseModel):
    year: int
    amount: float


class ReduceTarget(BaseModel):
    node: str | int | None
    # dimension_id -> category_id
    categories: dict[str, str] = Field(default_factory=dict)


class ReduceFlow(BaseModel):
    target: ReduceTarget
    amounts: list[ReduceAmount]

    @validator('amounts')
    def enough_years(cls, v):
        if len(v) < 2:
            raise ValueError("Must supply values for at least two years")
        return v

    def make_index(
        self, output_nodes: list[Node], extra_level: str | None = None, extra_level_values: Iterable | None = None,
    ) -> pd.MultiIndex:
        dims: dict[str, set[str]] = {dim: set() for dim in list(self.target.categories.keys())}
        nodes: set[str] = set()

        def get_node_id(node: str | int | None):
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


class ReduceParameterValue(BaseModel):
    __root__: List[ReduceFlow]


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
    allowed_parameters: ClassVar[list[Parameter]] = [
        ReduceParameter(local_id='reduce'),
        NumberParameter(local_id='multiplier'),
    ]

    def _compute_one(self, flow_id: str, param: ReduceFlow, unit: Unit):
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
        df = pl.concat([df, years], how='diagonal').sort(YEAR_COLUMN)
        df = df.groupby(YEAR_COLUMN).agg([pl.sum(col) for col in ('Target',)]).sort(YEAR_COLUMN)
        df = df.interpolate().fill_null(0)

        value_cols = [col for col in df.columns if col != YEAR_COLUMN]
        if self.is_enabled():
            df = df.with_columns([(pl.cumsum(col) * pl.lit(mult)).alias(col) for col in value_cols])
        else:
            df = df.with_columns([pl.lit(float(0)).alias(col) for col in value_cols])

        targets = [('Target', param.target)]

        all_dims = set(list(param.target.categories.keys()))

        def get_node_id(node: str | int | None):
            if isinstance(node, str):
                return node
            if node is None:
                nr = 0
            else:
                nr = node
            return self.output_nodes[nr].id

        def make_target_df(target: ReduceTarget, valuecol: str):
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
            pl.lit(True).alias(FORECAST_COLUMN),
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
        for idx, entry in enumerate(value.__root__):
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
        df = self.compute_effect_flow().drop(columns=[FLOW_ID_COLUMN, FLOW_ROLE_COLUMN])
        meta = df.get_meta()
        sdf = df.groupby(df.primary_keys).agg([pl.sum(VALUE_COLUMN), pl.first(FORECAST_COLUMN)])
        sdf = sdf.sort(meta.primary_keys)
        df = ppl.to_ppdf(sdf, meta=meta)
        return df
