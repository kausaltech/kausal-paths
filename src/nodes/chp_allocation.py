"""
Allocation of a combined heat and power plant's emissions between its heat and its electricity.

A combined heat and power (CHP) plant burns one fuel and sells two products, so its emissions are
divided between them. The methods form a hierarchy, and the code follows it:

1. **Weighted split** (`chp_heat_share`). Every method weights each unit of a product and divides
   the emissions in proportion to the weighted output. With electricity fraction ``f`` (the
   electricity share of the energy produced) and weights ``w_el`` and ``w_heat``, the heat share
   is ``w_heat (1 - f) / (w_el f + w_heat (1 - f))``. Equal weights give the energy-content
   method, and the reciprocals of reference efficiencies give the efficiency method.
2. **Exergetic split** (`exergetic_heat_share`). The weights are the work each product could do:
   electricity counts in full, and heat counts with the Carnot factor ``1 - t_ambient / t_mean``,
   where ``t_mean`` is the thermodynamic mean temperature of the heat network,
   ``(t_supply - t_return) / ln(t_supply / t_return)``.
3. **BISKO's split** (`bisko_heat_share`). BISKO criterion 6 prescribes the exergetic split with
   the ambient temperature fixed at 283 K.

Each function works row by row over inputs with the same index, so a plant or network dimension
needs no change here. They back the formula functions of the same names (`nodes.formula`).

`nodes.generic.ChpNode` and `BiskoChpNode` have their own implementation of the same methods. The
BISKO 1.3 module uses them and is frozen, so they are deliberately left as they are.
"""

from typing import TYPE_CHECKING

import polars as pl

from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.exceptions import NodeError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from common.polars import PathsDataFrame
    from nodes.node import Node

BISKO_T_AMBIENT = 283.0
"""
Ambient temperature (K) that BISKO fixes for the exergetic split.

This is T_U in the Carnot factor 1 - T_U / T_A of the Methodenpapier (Anhang, *Berechnung der
exergetischen Allokation*). It is not a property of the heat network: the network's own supply
and return temperatures are city data, and they enter through T_A, the thermodynamic mean.
"""


def _years(df: PathsDataFrame) -> str:
    return ', '.join(str(year) for year in df[YEAR_COLUMN].unique().sort())


def _join_inputs(node: Node, function: str, inputs: Mapping[str, tuple[PathsDataFrame, str]]) -> PathsDataFrame:
    """
    Join the inputs of an allocation function into one frame, with a column per input.

    ``inputs`` maps each column name to its frame and the unit it is converted to. A row is kept
    only where every input has a value: nothing is interpolated, because whether a year has a
    split decides which route the emissions take downstream. A row is forecast if any of its
    inputs is.
    """
    names = list(inputs)
    keys = inputs[names[0]][0].primary_keys
    parts: list[PathsDataFrame] = []
    for name, (frame, unit) in inputs.items():
        if VALUE_COLUMN not in frame.columns or len(frame.metric_cols) != 1:
            raise NodeError(node, '%s(): %s must have exactly one value column.' % (function, name))
        if set(frame.primary_keys) != set(keys):
            raise NodeError(
                node,
                '%s(): every input must have the same index; %s has %s, %s has %s.'
                % (function, name, ', '.join(frame.primary_keys), names[0], ', '.join(keys)),
            )
        converted = frame.ensure_unit(VALUE_COLUMN, unit)
        forecast = pl.col(FORECAST_COLUMN) if FORECAST_COLUMN in converted.columns else pl.lit(value=False)
        part = converted.select([*keys, pl.col(VALUE_COLUMN).alias(name), forecast.alias(f'{name}__forecast')])
        parts.append(part.filter(pl.col(name).is_not_null()))

    df = parts[0]
    for part in parts[1:]:
        df = df.paths.join_over_index(part, how='inner')
    forecast_cols = [f'{name}__forecast' for name in names]
    df = df.with_columns(pl.any_horizontal(forecast_cols).alias(FORECAST_COLUMN)).drop(forecast_cols)

    invalid = df.filter(pl.any_horizontal([~pl.col(name).is_finite() for name in names]))
    if invalid.height:
        raise NodeError(node, 'CHP inputs must be finite; invalid values in year(s) %s.' % _years(invalid))
    return df


def _split(node: Node, df: PathsDataFrame, electricity_weight: pl.Expr, heat_weight: pl.Expr) -> PathsDataFrame:
    """
    Return the heat share of the weighted split, from column ``electricity_fraction`` and two weights.

    The result keeps the index and the Forecast column of ``df`` and has a dimensionless Value.
    """
    bad = df.filter((pl.col('electricity_fraction') < 0.0) | (pl.col('electricity_fraction') > 1.0))
    if bad.height:
        raise NodeError(node, 'The electricity fraction must be between 0 and 1. It is not in year(s) %s.' % _years(bad))
    weights = df.with_columns([electricity_weight.alias('_w_el'), heat_weight.alias('_w_heat')])
    bad = weights.filter((pl.col('_w_el') <= 0.0) | (pl.col('_w_heat') <= 0.0))
    if bad.height:
        raise NodeError(node, 'The allocation weights must be positive. They are not in year(s) %s.' % _years(bad))

    # Both weights are positive, so the denominator is too, whatever the electricity fraction.
    f = pl.col('electricity_fraction')
    heat = pl.col('_w_heat') * (pl.lit(1.0) - f)
    out = weights.with_columns((heat / (pl.col('_w_el') * f + heat)).alias(VALUE_COLUMN))
    out = out.select([*df.primary_keys, VALUE_COLUMN, FORECAST_COLUMN])
    return out.set_unit(VALUE_COLUMN, 'dimensionless', force=True)


def chp_heat_share(
    node: Node,
    electricity_fraction: PathsDataFrame,
    electricity_weight: PathsDataFrame,
    heat_weight: PathsDataFrame,
) -> PathsDataFrame:
    """
    Return the share of a CHP plant's emissions allocated to its heat, by weighting the two products.

    ``electricity_fraction`` is the electricity share of the energy the plant produces, in any unit
    compatible with dimensionless, and the weights are what a unit of each product counts for;
    they must be positive and are dimensionless. The frames must have the same index, and a row
    has a share only where all of them have a value. The result is dimensionless.
    """
    df = _join_inputs(
        node,
        'chp_heat_share',
        {
            'electricity_fraction': (electricity_fraction, 'dimensionless'),
            'electricity_weight': (electricity_weight, 'dimensionless'),
            'heat_weight': (heat_weight, 'dimensionless'),
        },
    )
    return _split(node, df, pl.col('electricity_weight'), pl.col('heat_weight'))


def _exergetic_split(node: Node, df: PathsDataFrame, t_ambient: pl.Expr) -> PathsDataFrame:
    """Weight heat by its Carnot factor, from columns ``t_supply`` and ``t_return`` in K, and electricity by 1."""
    t_supply, t_return = pl.col('t_supply'), pl.col('t_return')
    bad = df.filter(t_supply < t_return)
    if bad.height:
        raise NodeError(
            node,
            'The district heating supply temperature must not be below the return temperature. It is in year(s) %s.'
            % _years(bad),
        )
    bad = df.filter(t_return <= t_ambient)
    if bad.height:
        raise NodeError(
            node,
            'The district heating return temperature must exceed the ambient temperature, otherwise the '
            'exergetic split has no heat to allocate. It does not in year(s) %s.' % _years(bad),
        )
    # The entropic mean of a flow cooled from supply to return. Equal temperatures (a steam
    # network condensing at one temperature) are the limit of the same formula.
    t_mean = pl.when(t_supply == t_return).then(t_supply).otherwise((t_supply - t_return) / (t_supply / t_return).log())
    return _split(node, df, pl.lit(1.0), pl.lit(1.0) - t_ambient / t_mean)


def exergetic_heat_share(
    node: Node,
    electricity_fraction: PathsDataFrame,
    t_supply: PathsDataFrame,
    t_return: PathsDataFrame,
    t_ambient: PathsDataFrame,
) -> PathsDataFrame:
    """
    Return the share of a CHP plant's emissions allocated to its heat by the exergetic (Carnot) method.

    ``t_supply`` and ``t_return`` are the temperatures of the plant's heat network and
    ``t_ambient`` the ambient temperature, in any unit compatible with K (°C works). See
    `chp_heat_share` for the electricity fraction, the index and the result.
    """
    df = _join_inputs(
        node,
        'exergetic_heat_share',
        {
            'electricity_fraction': (electricity_fraction, 'dimensionless'),
            't_supply': (t_supply, 'K'),
            't_return': (t_return, 'K'),
            't_ambient': (t_ambient, 'K'),
        },
    )
    return _exergetic_split(node, df, pl.col('t_ambient'))


def bisko_heat_share(
    node: Node,
    electricity_fraction: PathsDataFrame,
    t_supply: PathsDataFrame,
    t_return: PathsDataFrame,
) -> PathsDataFrame:
    """
    Return the share of a CHP plant's emissions that BISKO allocates to its heat.

    This is `exergetic_heat_share` with the ambient temperature fixed at `BISKO_T_AMBIENT`.
    """
    df = _join_inputs(
        node,
        'bisko_heat_share',
        {
            'electricity_fraction': (electricity_fraction, 'dimensionless'),
            't_supply': (t_supply, 'K'),
            't_return': (t_return, 'K'),
        },
    )
    return _exergetic_split(node, df, pl.lit(BISKO_T_AMBIENT))
