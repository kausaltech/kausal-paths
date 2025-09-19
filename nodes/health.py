from __future__ import annotations

from django.utils.translation import gettext_lazy as _

import polars as pl

from .calc import extend_last_historical_value_pl
from .constants import VALUE_COLUMN
from .exceptions import NodeError
from .gpc import DatasetNode

"""
In the commit https://github.com/kausaltech/kausal-paths/tree/bc011657c16a8a285e1e4f5311bab6a8c38317f7
these outdated parts were removed from the repo:
configs/healthimpact.yaml
configs/ilmastoruoka.yaml
nodes/health.py, ovariable-based parts
nodes/ovariable.py
"""


class AttributableFractionRR(DatasetNode):
    explanation = _(
        """
        Calculate attributable fraction when the ERF function is relative risk.

        AF=r/(r+1) if r >= 0; AF=r if r<0. Therefore, if the result
        is smaller than 0, we should use r instead. It can be converted from the result:
        r/(r+1)=s <=> r=s/(1-s)
        """
    )

    def compute(self):
        df = self.get_gpc_dataset()
        df = self.convert_names_to_ids(df)

        test = df['er_function'].unique()
        if len(test) != 1 or test[0] != 'relative_risk':
            raise NodeError(self, 'All of the rows must be for relative_risk as Er_function.')

        droplist = ['sector', 'quantity', 'parameter', 'er_function']
        params = {}
        for param in ['beta', 'threshold', 'rr_min']:
            dfp = df.filter(pl.col('parameter') == param)
            dfp = self.drop_unnecessary_levels(dfp, droplist)
            dfp = self.implement_unit_col(dfp)
            dfp = extend_last_historical_value_pl(dfp, end_year=self.get_end_year())
            params[param] = dfp

        if len(self.input_nodes) != 1:
            raise NodeError(self, 'The node must have exactly one input node for exposure.')
        dfn = self.input_nodes[0].get_output_pl(target_node=self)

        beta = params['beta']
        threshold = params['threshold']
        # rr_min = params['rr_min']  # FIXME add to formula

        dfn = dfn.paths.join_over_index(threshold, how='left', index_from='union')
        dfn = dfn.subtract_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN)
        dfn = dfn.drop(VALUE_COLUMN + '_right')
        dfn = dfn.paths.join_over_index(beta, how='left', index_from='union')
        dfn = dfn.multiply_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN)
        dfn = dfn.drop(VALUE_COLUMN + '_right')
        dfn = dfn.with_columns(pl.col(VALUE_COLUMN).exp().alias(VALUE_COLUMN))
        dfn = dfn.drop_nulls()

        frexposed = 1  # FIXME Make an input node

        r = dfn.with_columns([(pl.lit(frexposed) * (pl.col(VALUE_COLUMN) - pl.lit(1.0))).alias(VALUE_COLUMN)])
        r = r.with_columns([(pl.col(VALUE_COLUMN) / (pl.col(VALUE_COLUMN) + pl.lit(1.0))).alias(VALUE_COLUMN)])
        r = r.with_columns(
            pl.when(pl.col(VALUE_COLUMN).ge(pl.lit(0.0)))
            .then(pl.col(VALUE_COLUMN))
            .otherwise(pl.col(VALUE_COLUMN) / (pl.lit(1.0) - pl.col(VALUE_COLUMN)))
            .alias(VALUE_COLUMN)
        )

        return r


# --------- Previous version of PopulationAttributableFraction.

# Outdated treatment of exposure-response functions. Use as inspiration.

# if erf_type == 'unit risk':
#     slope = check_erf_units(route + '_m1')
#     threshold = check_erf_units(route + '_p1')
#     target_population = unit_registry('1 person')
#     out = (exposure - threshold) * slope * frexposed * p_illness
#     out = (out / target_population / period) / incidence

# elif erf_type == 'step function':
#     lower = check_erf_units(route + '_p1')
#     upper = check_erf_units(route + '_p1_2')
#     target_population = unit_registry('1 person')
#     out = (exposure >= lower) * 1
#     out = (out * (exposure <= upper) * -1 + 1) * frexposed * p_illness
#     out = (out / target_population / period) / incidence

# elif erf_type == 'relative risk':
#     beta = check_erf_units(route + '_m1')
#     threshold = check_erf_units(route + '_p1')
#     out = exposure - threshold
#     # out[VALUE_COLUMN] = np.where(  # FIXME
#     #     out[VALUE_COLUMN] < unit_registry('0 mg/kg/d'),
#     #     out[VALUE_COLUMN] * 0,
#     #     out[VALUE_COLUMN])
#     out = (out * beta).exp()
#     out = postprocess_relative(rr=out, frexposed=frexposed)

# elif erf_type == 'linear relative':
#     k = check_erf_units(route + '_m1')
#     threshold = check_erf_units(route + '_p1')
#     out = exposure - threshold
#     # out[VALUE_COLUMN] = np.where(  # FIXME
#     #     out[VALUE_COLUMN] < unit_registry('0 mg/kg/d'),
#     #     out[VALUE_COLUMN] * 0,
#     #     out[VALUE_COLUMN])
#     out = out * k
#     out = postprocess_relative(rr=out + 1, frexposed=frexposed)

# elif erf_type == 'relative Hill':
#     Imax = check_erf_units(route + '_p0')
#     ed50 = check_erf_units(route + '_p1')
#     out = (exposure * Imax) / (exposure + ed50) + 1
#     out = postprocess_relative(rr=out, frexposed=frexposed)

# elif erf_type == 'beta poisson approximation':
#     p1 = check_erf_units(route + '_p0')
#     p2 = check_erf_units(route + '_p1')
#     out = (exposure / p2 + 1) ** (p1 * -1) * -1 + 1
#     out = out * frexposed * p_illness

# elif erf_type == 'exact beta poisson':
#     p1 = check_erf_units(route + '_p0_2')
#     p2 = check_erf_units(route + '_p0')
#     # Remove unit: exposure is an absolute number of microbes ingested
#     s = exposure[VALUE_COLUMN].pint.to('cfu/d')
#     s = s / unit_registry('cfu/d')
#     exposure[VALUE_COLUMN] = s
#     out = (exposure * p1 / (p1 + p2) * -1).exp() * -1 + 1
#     out = out * frexposed * p_illness

# elif erf_type == 'exponential':
#     k = check_erf_units(route + '_m1')
#     out = (exposure * k * -1).exp() * -1 + 1
#     out = out * frexposed * p_illness

# elif erf_type == 'polynomial':
#     threshold = check_erf_units(route + '_p1')
#     p0 = check_erf_units(route + '_p0')
#     p1 = check_erf_units(route + '_m1')
#     p2 = check_erf_units(route + '_m2')
#     p3 = check_erf_units(route + '_m3')
#     x = exposure - threshold
#     out = x ** 3 * p3 + x ** 2 * p2 + x * p1 + p0
#     out = out * frexposed * p_illness

# else:
#     out = exposure / exposures
