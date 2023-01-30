from dataclasses import dataclass
from typing import ClassVar, Iterable, Any, List

from pydantic import BaseModel, root_validator, Field, validator
import pandas as pd
import polars as pl

from nodes.constants import FORECAST_COLUMN, NODE_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.node import NodeError, Node
from nodes.units import Unit
from params.param import ValidationError
from common import polars as ppl
from params import Parameter, register_parameter_type, ParameterWithUnit

from .action import ActionNode


class ShiftTarget(BaseModel):
    node: str | int | None
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

    @validator('amounts')
    def enough_years(cls, v):
        if len(v) < 2:
            raise ValueError("Must supply values for at least two years")
        return v

    @root_validator
    def dimensions_must_match(cls, obj: dict):
        existing_dims: set = set()
        def validate_target(target: ShiftTarget):
            dims = set(target.categories.keys())
            if not existing_dims:
                existing_dims.update(dims)
                return
            if dims != existing_dims:
                raise ValueError("Dimensions for yearly values for each target be equal")

        if 'source' not in obj or 'dest' not in obj:
            return obj

        validate_target(obj['source'])
        for dest in obj['dests']:
            validate_target(dest)
        return obj

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


class ShiftParameterValue(BaseModel):
    __root__: List[ShiftEntry]


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

        return ShiftParameterValue.validate(value)


class ShiftAction(ActionNode):
    allowed_parameters: ClassVar[list[Parameter]] = [
        ShiftParameter(local_id='shift')
    ]

    def _compute_one(self, param: ShiftEntry, unit: Unit):
        amounts = sorted(param.amounts, key=lambda x: x.year)

        data = [[a.year, a.source_amount, *a.dest_amounts] for a in param.amounts]
        dest_cols = ['Dest%d' % idx for idx in range(len(param.dests))]
        cols = [YEAR_COLUMN, 'Source', *dest_cols]

        df = pl.DataFrame(data, columns=cols)
        dest_sum = df.select(pl.col(dest_cols)).sum(axis=1)
        df = df.with_columns(pl.col(dest_cols) / dest_sum * (pl.col('Source') * -1))

        years = pl.DataFrame(range(amounts[0].year, self.context.model_end_year + 1), columns=[YEAR_COLUMN])
        df = pl.concat([df, years], how='diagonal').sort(YEAR_COLUMN)
        df = df.groupby(YEAR_COLUMN).agg([pl.sum(col) for col in ('Source', *dest_cols)]).sort(YEAR_COLUMN)
        df = df.interpolate().fill_null(0)

        value_cols = [col for col in df.columns if col != YEAR_COLUMN]
        if self.is_enabled():
            df = df.with_columns([pl.cumsum(col).alias(col) for col in value_cols])
        else:
            df = df.with_columns([pl.lit(float(0)).alias(col) for col in value_cols])

        targets = [('Source', param.source), *[('Dest%d' % idx, param.dests[idx]) for idx in range(len(param.dests))]]
        dims = list(param.source.categories.keys())

        def get_node_id(node: str | int | None):
            if isinstance(node, str):
                return node
            if node is None:
                nr = 0
            else:
                nr = node
            return self.output_nodes[nr].id

        def make_target_df(target: ShiftTarget, valuecol: str):
            target_dims = [(dim, target.categories[dim]) for dim in dims]
            tdf = df.select([
                pl.col(YEAR_COLUMN),
                pl.lit(get_node_id(target.node)).alias(NODE_COLUMN),
                *[pl.lit(cat).alias(dim) for dim, cat in target_dims],
                pl.col(valuecol).alias(VALUE_COLUMN),
            ])
            return tdf

        dfs = [make_target_df(target, col) for col, target in targets]
        df = pl.concat(dfs).sort(YEAR_COLUMN)

        df = df.groupby([NODE_COLUMN, *dims, YEAR_COLUMN]).agg(pl.sum(VALUE_COLUMN)).sort(YEAR_COLUMN)
        df = df.with_columns([pl.lit(True).alias(FORECAST_COLUMN)])
        meta = ppl.DataFrameMeta(units={VALUE_COLUMN: unit}, primary_keys=[YEAR_COLUMN, NODE_COLUMN, *dims])
        return ppl.to_ppdf(df, meta=meta)

    def compute_effect(self) -> ppl.PathsDataFrame:
        po = self.get_parameter('shift')
        value = po.get()
        assert isinstance(value, ShiftParameterValue)

        df = None
        for entry in value.__root__:
            edf = self._compute_one(entry, po.get_unit())
            if df is None:
                df = edf
            else:
                df = df.paths.add_with_dims(edf, how='outer')
        assert df is not None
        return df
