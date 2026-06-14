from __future__ import annotations

from cmath import nan
from typing import TYPE_CHECKING, Any, ClassVar, cast

from django.utils.translation import gettext_lazy as _

import numpy as np
import polars as pl

from common import polars as ppl
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.exceptions import NodeError
from nodes.generic import GenericNode
from nodes.gpc import DatasetNode
from nodes.simple import SimpleNode
from params import BoolParameter, NumberParameter, StringParameter

from .action import ActionNode

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.generic import OperationReturn
    from params.base import Parameter


class GenericAction(GenericNode, ActionNode):
    allowed_parameters = [
        *GenericNode.allowed_parameters,
        *ActionNode.allowed_parameters,
    ]

    no_effect_value = 0.0

    def compute_effect(self) -> PathsDataFrame:
        df = super().compute()
        if not self.is_enabled():
            df = df.with_columns(pl.lit(self.no_effect_value).alias(VALUE_COLUMN))
        return df

    def compute(self) -> PathsDataFrame:
        return self.compute_effect()


class SCurveAction(GenericAction):
    """
    S-curve action using two parameters (max_impact, max_year) from a dataset.

    Computes a non-linear sigmoid effect: y = A / (1 + exp(-k * (x - x0)))

    The dataset must contain a 'parameter' dimension column with values
    'max_impact' (= A, the maximum value) and 'max_year' (the year at which
    98% of the impact has occurred).  All other dimension columns are matched
    against the single input node's primary keys to identify which combo of
    dimensions to apply each S-curve to.

    Requires exactly one input node supplying historical background data and
    exactly one input dataset supplying the S-curve parameters.
    """

    explanation = _(
        'S-curve action. Computes a non-linear effect from max_impact and max_year '
        + 'parameters in the dataset. Requires one input node for historical background '
        + 'data and one input dataset with max_impact / max_year rows.'
    )
    allowed_parameters = [*GenericAction.allowed_parameters]
    no_effect_value = 0.0

    def _newton_raphson_estimator(
        self,
        y1: float,
        y2: float,
        x1: float,
        x2: float,
        a: float,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple[float, float]:
        """Fit S-curve slope k and midpoint x0 from two (x, y) observations."""
        z1 = np.log(y1 / (a - y1))
        z2 = np.log(y2 / (a - y2))

        k: float = 0.1
        x0: float = (x1 + x2) / 2

        for _iter in range(max_iter):
            f = np.array([k * (x1 - x0) - z1, k * (x2 - x0) - z2])
            j = np.array([[x1 - x0, -k], [x2 - x0, -k]])

            det = np.linalg.det(j)
            if np.abs(det) < 1e-10:
                raise ValueError('Jacobian is singular; adjust initial guesses.')

            delta = np.linalg.solve(j, -f)
            k_new = float(k + delta[0])
            x0_new = float(x0 + delta[1])

            if np.abs(k_new - k) < tol and np.abs(x0_new - x0) < tol:
                return k_new, x0_new

            k, x0 = k_new, x0_new

        return k, x0

    def _apply_scurve_parameters(
        self,
        df: ppl.PathsDataFrame,
        params: ppl.PathsDataFrame,
    ) -> ppl.PathsDataFrame:
        """Fit S-curve (slope, x0, ymax) per dimension combination and attach to df."""
        index_columns = [col for col in params.primary_keys if col in df.primary_keys and col != YEAR_COLUMN]
        if not index_columns:
            raise NodeError(
                self,
                'SCurveAction requires at least one shared primary key'
                + '(other than Year) between the input node and the parameter dataset.',
            )

        out = df.with_columns([
            pl.lit(None).cast(pl.Float64).alias('slope'),
            pl.lit(None).cast(pl.Float64).alias('x0'),
            pl.lit(None).cast(pl.Float64).alias('ymax'),
        ])
        indices = params.select(index_columns).unique()

        for row in indices.rows():
            filter_dict = {col: row[indices.columns.index(col)] for col in index_columns}

            filtered_df = df
            filtered_param = params
            for col, value in filter_dict.items():
                filtered_df = filtered_df.filter(pl.col(col) == value)
                filtered_param = filtered_param.filter(pl.col(col) == value)

            if len(filtered_df) == 0:
                continue

            x2 = filtered_param.filter(pl.col('parameter') == 'max_year').select(VALUE_COLUMN).item()
            filtered_param = filtered_param.ensure_unit(VALUE_COLUMN, df.get_unit(VALUE_COLUMN))
            a = filtered_param.filter(pl.col('parameter') == 'max_impact').select(VALUE_COLUMN).item()

            dfnow = filtered_df.filter(~pl.col(FORECAST_COLUMN))
            x1 = float(dfnow.select(YEAR_COLUMN).max().item())
            y1 = float(filtered_df.filter(pl.col(YEAR_COLUMN) == int(x1)).select(VALUE_COLUMN).item())
            a = float(a)
            x2 = float(x2)

            if y1 < a:
                y1 = min(max(y1 / a, 0.02), 0.98) * a
                y2 = 0.98 * a
                slope, x0 = self._newton_raphson_estimator(y1, y2, x1, x2, a)
            elif y1 == a:
                slope, x0 = 100.0, 100.0
            else:
                anew = y1
                y2new = min(max(a / anew, 0.02), 0.98) * anew
                y1new = 0.98 * anew
                slope, x0 = self._newton_raphson_estimator(y1new, y2new, x1, x2, anew)
                a = anew

            mask = pl.lit(True)  # noqa: FBT003
            for col, value in filter_dict.items():
                mask = mask & (pl.col(col) == value)
            out = out.with_columns(pl.when(mask).then(slope).otherwise(pl.col('slope')).alias('slope'))
            out = out.with_columns(pl.when(mask).then(x0).otherwise(pl.col('x0')).alias('x0'))
            out = out.with_columns(pl.when(mask).then(a).otherwise(pl.col('ymax')).alias('ymax'))

        return out

    def compute_effect(self) -> ppl.PathsDataFrame:
        df = self.get_input_node().get_output_pl(target_node=self)
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')

        params = self.get_input_dataset_pl()
        # GenericDataset extends the data across all model years; collapse params to
        # a single representative year so .item() works in _apply_scurve_parameters.
        target_year = self.context.instance.target_year
        if YEAR_COLUMN in params.primary_keys:
            params_at_target = params.filter(pl.col(YEAR_COLUMN) == target_year)
            if len(params_at_target) > 0:
                params = params_at_target

        df = self._apply_scurve_parameters(df, params)

        df = df.with_columns(
            (pl.col('ymax') / (1.0 + (-pl.col('slope') * (pl.col(YEAR_COLUMN) - pl.col('x0'))).exp())).alias('out')
        )
        df = df.set_unit('out', df.get_unit(VALUE_COLUMN))
        df = df.with_columns(pl.when(pl.col(FORECAST_COLUMN)).then(pl.col('out')).otherwise(pl.col(VALUE_COLUMN)).alias('out'))
        df = df.subtract_cols(['out', VALUE_COLUMN], VALUE_COLUMN)
        assert self.unit is not None, f'Node {self.id} must have unit defined.'
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        df = df.drop(['out', 'ymax', 'slope', 'x0'])

        if not self.is_enabled():
            df = df.with_columns(pl.lit(self.no_effect_value).alias(VALUE_COLUMN))
        return df


class ValueAction(GenericAction):
    """
    Action that outputs a user-adjustable constant (e.g. a value weight).

    Combines GenericAction with ConstantNode-like behaviour: no input datasets,
    single 'constant' parameter, output is that constant over the full timeline.
    Used for moral/value nodes where the "action" is how much we weight that value.
    """

    allowed_parameters = [
        *GenericAction.allowed_parameters,
        NumberParameter(
            local_id='constant',
            label=_('Weight'),
            description=_('How much to weight this value (0 = ignore in priorities)'),
            is_customizable=True,
        ),
    ]
    no_effect_value = 0.0
    default_color = '#9B59B6'  # Purple/violet for value nodes; override in config or via group

    def compute_effect(self) -> PathsDataFrame:
        if not self.is_enabled():
            constant = self.no_effect_value
        else:
            val = self.get_parameter_value('constant', required=False, units=True)
            if val is None:
                constant = 1.0
            else:
                constant = float(val.m) if hasattr(val, 'm') else float(val)
        assert self.unit is not None
        start_year = self.context.instance.minimum_historical_year
        end_year = self.context.instance.model_end_year
        last_historical_year = getattr(self.context.instance, 'maximum_historical_year', None) or start_year
        years = list(range(start_year, end_year + 1))
        df = pl.DataFrame({
            YEAR_COLUMN: years,
            VALUE_COLUMN: [constant] * len(years),
            FORECAST_COLUMN: [y > last_historical_year for y in years],
        })
        meta = ppl.DataFrameMeta(units={VALUE_COLUMN: self.unit}, primary_keys=[YEAR_COLUMN])
        return ppl.to_ppdf(df, meta=meta)


class AdditiveAction(ActionNode):
    explanation = _("""Simple action that produces an additive change to a value.""")
    no_effect_value = 0.0

    def compute_effect(self):
        df = self.get_input_dataset_pl()

        if self.get_parameter_value('allow_null_categories', required=False):
            self.allow_null_categories = True

        for m in self.output_metrics.values():
            if not self.is_enabled():
                df = df.with_columns(
                    pl.when(pl.col(m.column_id).is_null()).then(None).otherwise(self.no_effect_value).alias(m.column_id)
                )
            df = df.ensure_unit(m.column_id, m.unit)

        return df


class AdditiveAction2(AdditiveAction, SimpleNode):  # FIXME Merge with AdditiveAction
    allowed_parameters = [*AdditiveAction.allowed_parameters, *SimpleNode.allowed_parameters]

    def compute_effect(self):
        df = super().compute_effect()
        multiplier = self.get_parameter_value('multiplier', required=False, units=True)
        if multiplier is not None:
            df = df.multiply_quantity(VALUE_COLUMN, multiplier)
        return df


# FIXME Update to deal with old-fashioned multi-metric nodes such as Tampere/private_building_energy_renovation
class CumulativeAdditiveAction(ActionNode):
    explanation = _("""Additive action where the effect is cumulative and remains in the future.""")

    allowed_parameters: ClassVar[list[Parameter[Any]]] = [
        NumberParameter(local_id='target_year_ratio', min_value=0, unit_str='%'),
    ]

    def add_cumulatively(self, df):
        end_year = self.get_end_year()
        df = df.reindex(range(df.index.min(), end_year + 1))
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
                assert isinstance(target_year_ratio, (float, int))
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
        NumberParameter(local_id='target_year_level'),
        NumberParameter(
            local_id='action_delay',
            label=_('Years of delay (a)'),
        ),
        NumberParameter(local_id='multiplier'),
    ]

    explanation = _("""Cumulative additive action where a yearly target is set and the effect is linear.
    This can be modified with these parameters:
    target_year_level is the value to be reached at the target year.
    action_delay is the year when the implementation of the action starts.
    multiplier scales the size of the impact (useful between scenarios).
    """)

    def compute_effect(self):
        df = self.get_input_dataset()
        start_year = df.index.min()
        delay = self.get_parameter_value_float('action_delay', required=False)
        if delay is not None:
            start_year = start_year + int(delay)
        target_year = self.get_target_year()
        df = df.reindex(range(start_year, target_year + 1))
        df[FORECAST_COLUMN] = True

        target_year_level = cast('float | None', self.get_parameter_value('target_year_level', required=False))
        if target_year_level is not None:
            if set(df.columns) != {VALUE_COLUMN, FORECAST_COLUMN}:
                raise NodeError(self, 'target_year_level parameter can only be used with single-value nodes')
            df.loc[target_year, VALUE_COLUMN] = target_year_level
            if delay is not None:
                df.loc[range(start_year + 1, target_year), VALUE_COLUMN] = nan

        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            dt = df.dtypes[col]
            df[col] = df[col].pint.m.interpolate(method='linear').diff().fillna(0).astype(dt)

        df = self.add_cumulatively(df)
        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            multiplier = self.get_parameter_value('multiplier', required=False, units=True)
            if multiplier is not None:
                df[col] *= multiplier
            df[col] = self.ensure_output_unit(df[col])
        return df


class EmissionReductionAction(ActionNode):
    explanation = _("""Simple emission reduction impact""")

    no_effect_value = 0

    def compute_effect(self):
        df = self.get_input_dataset()
        df[VALUE_COLUMN] = 0 - df[VALUE_COLUMN]
        return df


class TrajectoryAction(ActionNode):
    explanation = _("""
    TrajectoryAction uses select_category() to select a category from a dimension
    and then possibly do some relative or absolute conversions.
    """)
    allowed_parameters = [
        *ActionNode.allowed_parameters,
        StringParameter(local_id='dimension'),
        StringParameter(local_id='category'),
        NumberParameter(local_id='category_number'),
        NumberParameter(local_id='baseline_year'),
        NumberParameter(local_id='baseline_year_level'),
        BoolParameter(local_id='keep_dimension'),
    ]

    def compute_effect(self):
        df = self.get_input_dataset_pl()
        dim_id = self.get_parameter_value_str('dimension', required=True)
        cat_id = self.get_parameter_value_str('category', required=False)
        cat_no = self.get_parameter_value_int('category_number', required=False)
        year = self.get_parameter_value_int('baseline_year', required=False)
        level = self.get_parameter_value('baseline_year_level', units=True, required=False)
        keep = self.get_typed_parameter_value('keep_dimension', bool, required=False)
        if not self.is_enabled():
            cat_id = 'baseline'  # FIXME Generalize this
            cat_no = None

        df = df.select_category(dim_id, cat_id, cat_no, year, level, keep)
        assert self.unit is not None
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df


class GpcTrajectoryAction(TrajectoryAction, DatasetNode):
    explanation = _("""
    GpcTrajectoryAction is a trajectory action that uses the DatasetNode to fetch the dataset.
    """)
    allowed_parameters = [*TrajectoryAction.allowed_parameters, *DatasetNode.allowed_parameters]

    def compute_effect(self):
        df = DatasetNode.compute(self)
        dim_id = self.get_parameter_value_str('dimension', required=True)
        cat_id = self.get_parameter_value_str('category', required=False)
        cat_no = self.get_parameter_value_int('category_number', required=False)
        year = self.get_parameter_value_int('baseline_year', required=False)
        level = self.get_parameter_value_float('baseline_year_level', units=True, required=False)
        keep = self.get_typed_parameter_value('keep_dimension', bool, required=False)
        if not self.is_enabled():
            cat_id = 'baseline'  # FIXME Generalize this
            cat_no = None

        df = df.select_category(dim_id, cat_id, cat_no, year, level, keep)
        assert self.unit is not None
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df


class ParameterAction(ActionNode):
    allowed_parameters = [
        *ActionNode.allowed_parameters,
        NumberParameter(local_id='from_value', description=_('Starting parameter value'), is_customizable=True),
        NumberParameter(
            local_id='percent_change', description=_('Annual percent change in parameter value'), is_customizable=True
        ),
        NumberParameter(local_id='from_year', description=_('Starting year'), is_customizable=True),
        NumberParameter(local_id='default_value', description=_('Default parameter value'), is_customizable=False),
        NumberParameter(
            local_id='default_change', description=_('Default annual percent change in parameter value'), is_customizable=False
        ),
    ]

    def compute_effect(self):
        fromyear = self.get_parameter_value_int('from_year', required=False)
        if not fromyear:
            fromyear = self.context.instance.maximum_historical_year + 1  # type: ignore
        toyear = self.context.instance.target_year + 1

        if self.is_enabled():
            fromvalue = self.get_parameter_value('from_value', required=True)
            percentchange = self.get_parameter_value('percent_change', required=False)
        else:
            fromvalue = self.get_parameter_value('default_value', required=True)
            percentchange = self.get_parameter_value('default_change', required=False)

        if percentchange:
            percentchange = 1.0 + (percentchange / 100.0)  # type: ignore
        else:
            percentchange = 1.0

        df = pl.DataFrame({'Year': range(fromyear, toyear)}).with_columns(
            (pl.lit(fromvalue) * pl.lit(percentchange) ** (pl.col('Year') - fromyear)).alias('Value'),
            pl.lit(value=True).alias('Forecast'),
        )

        meta = ppl.DataFrameMeta(units={'Value': self.unit}, primary_keys=['Year'])  # type: ignore
        df = ppl.to_ppdf(df, meta=meta)

        return df


class ChpAction(GenericAction):
    """
    Produces CHP emission-factor allocation fractions as a standalone action.

    Outputs a PathsDataFrame with an energy_carrier dimension (electricity /
    district_heating) where VALUE is the allocation fraction for each carrier.
    Wire the output into a multiply node together with the average CHP fuel
    emission factor to obtain carrier-specific emission factors without routing
    df through ChpNode.

    All method parameters are customizable so they are visible and adjustable
    in the public UI, unlike the YAML-only params on ChpNode.
    """

    allowed_parameters = [
        *GenericAction.allowed_parameters,
        StringParameter(local_id='method', label=_('Emission splitting method'), is_customizable=True),
        NumberParameter(
            local_id='electricity_fraction',
            label=_('Fraction of electricity in the output energy'),
            is_customizable=True,
        ),
        NumberParameter(
            local_id='t_supply',
            label=_('Temperature (K) of district heating supply'),
            is_customizable=True,
        ),
        NumberParameter(
            local_id='t_return',
            label=_('Temperature (K) of district heating return flow'),
            is_customizable=True,
        ),
        NumberParameter(
            local_id='electricity_reference_efficiency',
            label=_('Efficiency of producing electricity separately'),
            is_customizable=True,
        ),
        NumberParameter(
            local_id='heat_reference_efficiency',
            label=_('Efficiency of producing heat separately'),
            is_customizable=True,
        ),
    ]

    DEFAULT_OPERATIONS = 'chp_fractions'

    def _compute_allocation_fractions(self) -> tuple[float, float]:
        method = self.get_parameter_value_str('method', required=True)
        f_el = self.get_parameter_value_float('electricity_fraction', required=True)

        if method == 'energy_content':
            z_el, z_heat = 1.0, 1.0
        elif method == 'work_potential':
            t_supply = self.get_parameter_value_float('t_supply', required=True)
            t_return = self.get_parameter_value_float('t_return', required=True)
            z_el, z_heat = 1.0, 1.0 - t_return / t_supply
        elif method == 'bisko':
            t_supply = self.get_parameter_value_float('t_supply', required=True)
            z_el, z_heat = 1.0, 1.0 - 283.0 / t_supply
        elif method == 'efficiency':
            n_el = self.get_parameter_value_float('electricity_reference_efficiency', required=True)
            n_heat = self.get_parameter_value_float('heat_reference_efficiency', required=True)
            z_el, z_heat = 1.0 / n_el, 1.0 / n_heat
        else:
            raise NodeError(
                self,
                f"Parameter 'method' got value {method!r}; must be one of: "
                + 'energy_content, work_potential, bisko, efficiency.',
            )

        a_el = z_el * f_el / (z_el * f_el + z_heat * (1.0 - f_el))
        return a_el, 1.0 - a_el

    def _operation_chp_fractions(self, df: PathsDataFrame | None) -> OperationReturn:
        if df is not None:
            raise NodeError(self, "Operation 'chp_fractions' must be the only operation.")

        a_el, a_heat = self._compute_allocation_fractions()

        instance = self.context.instance
        start_year = instance.reference_year
        end_year = instance.model_end_year
        last_hist = instance.maximum_historical_year or start_year
        years = list(range(start_year, end_year + 1))
        n = len(years)

        out = ppl.PathsDataFrame({
            YEAR_COLUMN: years * 2,
            'energy_carrier': ['electricity'] * n + ['district_heating'] * n,
        })
        out._units = {}
        out._primary_keys = [YEAR_COLUMN, 'energy_carrier']
        out = out.with_columns([
            pl.Series(VALUE_COLUMN, [a_el] * n + [a_heat] * n),
            (pl.col(YEAR_COLUMN) > pl.lit(last_hist)).alias(FORECAST_COLUMN),
        ]).set_unit(VALUE_COLUMN, 'dimensionless')
        return out

    def compute_effect(self) -> PathsDataFrame:
        # Fractions are always physically meaningful; bypass GenericAction's no_effect_value override.
        return GenericNode.compute(self)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.OPERATIONS['chp_fractions'] = self._operation_chp_fractions
