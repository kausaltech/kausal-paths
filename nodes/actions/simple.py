from cmath import nan
from typing import ClassVar, Iterable

from pydantic import BaseModel, root_validator, Field

import pandas as pd
import pint_pandas
import polars as pl


from params.param import Parameter
from params import PercentageParameter, NumberParameter
from nodes.constants import FORECAST_COLUMN, NODE_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.node import NodeError
from .action import ActionNode



class AdditiveAction(ActionNode):
    """Simple action that produces an additive change to a value."""
    no_effect_value = 0

    def compute_effect(self):
        df = self.get_input_dataset()
        if not self.is_enabled():
            df[VALUE_COLUMN] = 0.0
            df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])
        return df


class CumulativeAdditiveAction(ActionNode):
    """Additive action where the effect is cumulative and remains in the future."""

    allowed_parameters: ClassVar[list[Parameter]] = [
        PercentageParameter('target_year_ratio', min_value=0),
    ]

    def add_cumulatively(self, df):
        target_year = self.get_target_year()
        df = df.reindex(range(df.index.min(), target_year + 1))
        df[FORECAST_COLUMN] = True

        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue

            val = df[col]
            if hasattr(val, 'pint'):
                val = val.pint.m
            val = val.fillna(0).cumsum()

            target_year_ratio = self.get_parameter_value('target_year_ratio', required=False)
            if target_year_ratio is not None:
                val *= target_year_ratio / 100

            df[col] = val
            if not self.is_enabled():
                df[col] = 0.0
            df[col] = self.ensure_output_unit(df[col])

        return df

    def compute_effect(self):
        df = self.get_input_dataset()
        return self.add_cumulatively(df)


class LinearCumulativeAdditiveAction(CumulativeAdditiveAction):
    allowed_parameters = CumulativeAdditiveAction.allowed_parameters + [
        NumberParameter('target_year_level'),
        NumberParameter(
            local_id='action_delay',
            label='Years of delay (a)',
        ),
    ]

    """Cumulative additive action where a yearly target is set and the effect is linear."""
    def compute_effect(self):
        df = self.get_input_dataset()
        start_year = df.index.min()
        delay = self.get_parameter_value('action_delay', required=False)
        if delay is not None:
            start_year = start_year + int(delay)
        end_year = df.index.max()
        df = df.reindex(range(start_year, end_year + 1))
        df[FORECAST_COLUMN] = True

        target_year_level = self.get_parameter_value('target_year_level', required=False)
        if target_year_level is not None:
            if set(df.columns) != set([VALUE_COLUMN, FORECAST_COLUMN]):
                raise NodeError(self, "target_year_level parameter can only be used with single-value nodes")
            df.loc[end_year, VALUE_COLUMN] = target_year_level
            if delay is not None:
                df.loc[range(start_year + 1, end_year), VALUE_COLUMN] = nan

        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            dt = df.dtypes[col]
            df[col] = df[col].pint.m.interpolate(method='linear').diff().fillna(0).astype(dt)

        df = self.add_cumulatively(df)
        return df


class EmissionReductionAction(ActionNode):
    """Simple emission reduction impact"""

    no_effect_value = 0

    def compute_effect(self):
        df = self.get_input_dataset()
        df[VALUE_COLUMN] = 0 - df[VALUE_COLUMN]
        return df


class ExponentialAction(ActionNode):
    allowed_parameters = [
        NumberParameter(
            local_id='current_value',
            unit_str='EUR/t',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='annual_change',
            unit_str='%',
            is_customizable=True,
        ),
    ]

    def compute_exponential(self):
        current_value = self.get_parameter('current_value', required=True)
        pt = pint_pandas.PintType(current_value.get_unit())
        base_value = self.get_parameter('annual_change', required=True)
        base_unit = base_value.get_unit()
        if self.is_enabled():
            current_value = current_value.value
            base_value = base_value.value
        else:
            current_value = current_value.scenario_settings['default']
            base_value = base_value.scenario_settings['default']
        base_value = 1 + (base_value * base_unit).to('dimensionless').m
        start_year = self.context.instance.minimum_historical_year
        target_year = self.get_target_year()
        current_year = self.context.instance.maximum_historical_year

        df = pd.DataFrame(
            {VALUE_COLUMN: range(start_year - current_year, target_year - current_year + 1)},
            index=range(start_year, target_year + 1))
        val = current_value * base_value ** df[VALUE_COLUMN]
        df[VALUE_COLUMN] = val.astype(pt)
        df[FORECAST_COLUMN] = df.index > current_year

        return df

    def compute_effect(self):
        return self.compute_exponential()



class ShiftTarget(BaseModel):
    node_id: str
    # dimension_id -> category_id
    categories: dict[str, str] = Field(default_factory=dict)


class ShiftAmount(BaseModel):
    year: int
    source_amount: float
    dest_amounts: list[float]


class ShiftParameter(BaseModel):
    source: ShiftTarget
    dests: list[ShiftTarget]
    amounts: list[ShiftAmount]

    @root_validator
    def dimensions_must_match(cls, obj: dict):
        if len(obj['amounts']) < 2:
            raise ValueError("Must supply values for at least two years")

        existing_dims: set = set()
        def validate_target(target: ShiftTarget):
            dims = set(target.categories.keys())
            if not existing_dims:
                existing_dims.update(dims)
                return
            if dims != existing_dims:
                raise ValueError("Dimensions for yearly values for each target be equal")

        validate_target(obj['source'])
        for dest in obj['dests']:
            validate_target(dest)
        return obj

    def make_index(self, extra_level: str | None = None, extra_level_values: Iterable | None = None) -> pd.MultiIndex:
        dims: dict[str, set[str]] = {dim: set() for dim in list(self.source.categories.keys())}
        nodes: set[str] = set()
        nodes.add(self.source.node_id)
        for dim, cat in self.source.categories.items():
            dims[dim].add(cat)
        for dest in self.dests:
            nodes.add(dest.node_id)
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


class ShiftAction(ActionNode):
    def compute_effect(self) -> pd.DataFrame:
        param = ShiftParameter(
            source=ShiftTarget(
                node_id='building_floor_area_existing', categories=dict(building_use='residence', building_heat_source='oil'),
            ),
            dests=[
                ShiftTarget(
                    node_id='building_floor_area_existing', categories=dict(building_use='residence', building_heat_source='geothermal'),
                ),
                ShiftTarget(
                    node_id='building_floor_area_existing', categories=dict(building_use='residence', building_heat_source='district_heat'),
                ),
            ],
            amounts=[
                ShiftAmount(year=2023, source_amount=-50000, dest_amounts=[50, 50]),
                ShiftAmount(year=2030, source_amount=-50000, dest_amounts=[25, 50]),
            ],
        )
        amounts = sorted(param.amounts, key=lambda x: x.year)

        data = [[a.year, a.source_amount, *a.dest_amounts] for a in param.amounts]
        dest_cols = ['Dest%d' % idx for idx in range(len(param.dests))]
        cols = [YEAR_COLUMN, 'Source', *dest_cols]

        pl.Config.set_tbl_rows(50)

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

        def make_target_df(target: ShiftTarget, valuecol: str):
            target_dims = [(dim, target.categories[dim]) for dim in dims]
            tdf = df.select([
                pl.col(YEAR_COLUMN),
                pl.lit(target.node_id).alias(NODE_COLUMN),
                *[pl.lit(cat).alias(dim) for dim, cat in target_dims],
                pl.col(valuecol).alias(VALUE_COLUMN),
            ])
            return tdf

        dfs = [make_target_df(target, col) for col, target in targets]
        df = pl.concat(dfs).sort(YEAR_COLUMN)

        df = df.groupby([NODE_COLUMN, *dims, YEAR_COLUMN]).agg(pl.sum(VALUE_COLUMN)).sort(YEAR_COLUMN)
        pdf = df.to_pandas().set_index([NODE_COLUMN, *dims, YEAR_COLUMN])
        pdf[VALUE_COLUMN] = pdf[VALUE_COLUMN].astype(pint_pandas.PintType(self.unit))
        pdf[FORECAST_COLUMN] = True
        return pdf
