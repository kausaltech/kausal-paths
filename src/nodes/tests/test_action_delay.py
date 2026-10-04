"""
The ``action_delay`` parameter: postponing an action's effect in time.

The delay is applied once, in ``ActionNode.compute_output()``, to every action class.
It shifts the *effect* -- the difference between the enabled and the disabled output --
so these tests use an action whose disabled output is not zero, which is what an action
emitting a level or a factor looks like.
"""

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.actions.action import ACTION_DELAY_PARAM_ID, ActionNode
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.dimensions import Dimension, DimensionCategory
from nodes.exceptions import NodeError
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry
from params.param import NumberParameter

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.context import Context

pytestmark = pytest.mark.django_db

LAST_HISTORICAL_YEAR = 2018


class _FixedAction(ActionNode):
    """An action that returns one caller-supplied frame when enabled and another when disabled."""

    def __init__(self, *args: Any, enabled_df: PathsDataFrame, disabled_df: PathsDataFrame, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._enabled_df = enabled_df
        self._disabled_df = disabled_df

    def compute_effect(self) -> PathsDataFrame:
        return self._enabled_df if self.is_enabled() else self._disabled_df


def _make_context(identifier: str) -> Context:
    instance = InstanceFactory.create(id=identifier, name=identifier)
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name=identifier)
    assert instance.maximum_historical_year == LAST_HISTORICAL_YEAR
    ctx = instance.context
    ctx.dimensions['sector'] = Dimension(
        id='sector',
        label=TranslatedString('Sector', default_language='en'),
        categories=[
            DimensionCategory(id='x', label=TranslatedString('X', default_language='en')),
            DimensionCategory(id='y', label=TranslatedString('Y', default_language='en')),
        ],
    )
    return ctx


def _ppdf(values: dict[int, float] | dict[tuple[int, str], float]) -> PathsDataFrame:
    """Build a frame from ``{year: value}``, or ``{(year, sector): value}`` for a dimensioned one."""
    keys = list(values)
    if isinstance(keys[0], tuple):
        df = pl.DataFrame(
            {
                YEAR_COLUMN: [k[0] for k in keys],  # type: ignore[index]
                'sector': [k[1] for k in keys],  # type: ignore[index]
                VALUE_COLUMN: list(values.values()),
            },
            schema={YEAR_COLUMN: pl.Int64, 'sector': pl.String, VALUE_COLUMN: pl.Float64},
        )
        pks = [YEAR_COLUMN, 'sector']
    else:
        df = pl.DataFrame(
            {YEAR_COLUMN: keys, VALUE_COLUMN: list(values.values())},
            schema={YEAR_COLUMN: pl.Int64, VALUE_COLUMN: pl.Float64},
        )
        pks = [YEAR_COLUMN]
    df = df.with_columns((pl.col(YEAR_COLUMN) > LAST_HISTORICAL_YEAR).alias(FORECAST_COLUMN))
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kt/a')}, primary_keys=pks)
    return to_ppdf(df, meta)


def _action(
    context: Context,
    enabled: dict[int, float] | dict[tuple[int, str], float],
    disabled: dict[int, float] | dict[tuple[int, str], float],
    *,
    delay: float | None = None,
    is_enabled: bool = True,
) -> _FixedAction:
    edf = _ppdf(enabled)
    dims = edf.dim_ids or None
    action = _FixedAction(
        id='action',
        context=context,
        name=TranslatedString('action', default_language='en'),
        unit=unit_registry.parse_units('kt/a'),
        quantity='emissions',
        enabled_df=edf,
        disabled_df=_ppdf(disabled),
        output_dimension_ids=dims,
        input_dimension_ids=dims,
    )
    if delay is not None:
        param = NumberParameter(local_id=ACTION_DELAY_PARAM_ID, is_customizable=True)
        param.set(delay)
        action.add_parameter(param)
    context.add_node(action)
    action.finalize_init()
    action.enabled_param.set(is_enabled)
    return action


def _add_global_delay(context: Context, delay: float) -> NumberParameter:
    param = NumberParameter(local_id=ACTION_DELAY_PARAM_ID, is_customizable=True)
    param.set(delay)
    context.add_global_parameter(param)
    return param


def _values(df: PathsDataFrame) -> dict[Any, float]:
    if df.dim_ids:
        return {(r[YEAR_COLUMN], r['sector']): r[VALUE_COLUMN] for r in df.sort([YEAR_COLUMN, 'sector']).to_dicts()}
    return {r[YEAR_COLUMN]: r[VALUE_COLUMN] for r in df.sort(YEAR_COLUMN).to_dicts()}


# Disabled: a flat level of 10. Enabled: the action lowers it by 1 a year from 2019,
# and had already lowered it by 1 in the last historical year.
DISABLED: dict[int, float] = {2017: 10.0, 2018: 10.0, 2019: 10.0, 2020: 10.0, 2021: 10.0, 2022: 10.0}
ENABLED: dict[int, float] = {2017: 10.0, 2018: 9.0, 2019: 8.0, 2020: 7.0, 2021: 6.0, 2022: 5.0}


def test_no_delay_leaves_the_output_alone():
    ctx = _make_context('delay-none')
    action = _action(ctx, ENABLED, DISABLED)
    assert action.get_delay_years() == 0
    assert _values(action.get_output_pl()) == ENABLED


def test_delay_shifts_the_effect_and_freezes_progress_during_the_delay():
    ctx = _make_context('delay-shift')
    action = _action(ctx, ENABLED, DISABLED, delay=2)
    # History is untouched. For two years the effect stays at its 2018 level (-1),
    # then the 2019 effect (-2) arrives in 2021 and the 2020 effect (-3) in 2022.
    assert _values(action.get_output_pl()) == {2017: 10.0, 2018: 9.0, 2019: 9.0, 2020: 9.0, 2021: 8.0, 2022: 7.0}


def test_delay_past_the_model_end_leaves_only_historical_progress():
    ctx = _make_context('delay-long')
    action = _action(ctx, ENABLED, DISABLED, delay=10)
    assert _values(action.get_output_pl()) == {2017: 10.0, 2018: 9.0, 2019: 9.0, 2020: 9.0, 2021: 9.0, 2022: 9.0}


def test_global_and_own_delay_add_up():
    ctx = _make_context('delay-sum')
    _add_global_delay(ctx, 1)
    action = _action(ctx, ENABLED, DISABLED, delay=1)
    assert action.get_delay_years() == 2
    assert _values(action.get_output_pl()) == {2017: 10.0, 2018: 9.0, 2019: 9.0, 2020: 9.0, 2021: 8.0, 2022: 7.0}


def test_changing_the_global_delay_recomputes_the_action():
    ctx = _make_context('delay-global-change')
    param = _add_global_delay(ctx, 0)
    action = _action(ctx, ENABLED, DISABLED)
    assert _values(action.get_output_pl()) == ENABLED
    param.set(2.0)
    assert _values(action.get_output_pl())[2022] == 7.0


def test_disabled_action_ignores_the_delay():
    ctx = _make_context('delay-disabled')
    action = _action(ctx, ENABLED, DISABLED, delay=2, is_enabled=False)
    assert _values(action.get_output_pl()) == DISABLED


def test_delay_is_applied_per_dimension_category():
    ctx = _make_context('delay-dims')
    disabled = {(y, s): 10.0 for y in (2018, 2019, 2020) for s in ('x', 'y')}
    enabled = {
        (2018, 'x'): 10.0, (2019, 'x'): 8.0, (2020, 'x'): 6.0,
        (2018, 'y'): 10.0, (2019, 'y'): 9.0, (2020, 'y'): 8.0,
    }  # fmt: skip
    action = _action(ctx, enabled, disabled, delay=1)
    assert _values(action.get_output_pl()) == {
        (2018, 'x'): 10.0, (2018, 'y'): 10.0,
        (2019, 'x'): 10.0, (2019, 'y'): 10.0,
        (2020, 'x'): 8.0, (2020, 'y'): 9.0,
    }  # fmt: skip


@pytest.mark.parametrize('delay', [-1, 1.5])
def test_delay_must_be_a_whole_number_of_years(delay: float):
    ctx = _make_context(f'delay-invalid-{delay}')
    action = _action(ctx, ENABLED, DISABLED, delay=delay)
    with pytest.raises(NodeError, match='whole number of years'):
        action.get_delay_years()


def test_disabled_output_of_a_different_shape_is_refused():
    ctx = _make_context('delay-shape')
    action = _action(ctx, ENABLED, {2017: 10.0, 2018: 10.0}, delay=1)
    with pytest.raises(NodeError, match='different shape'):
        action.get_output_pl()


def test_generic_action_sees_both_node_and_action_global_parameters():
    from nodes.actions.simple import GenericAction
    from nodes.generic import GenericNode

    for param_id in [*GenericNode.global_parameters, *ActionNode.global_parameters]:
        assert param_id in GenericAction.global_parameters


def test_without_maximum_historical_year_the_last_non_forecast_year_is_the_boundary():
    ctx = _make_context('delay-no-max-hist')
    ctx.instance.maximum_historical_year = None
    action = _action(ctx, ENABLED, DISABLED, delay=2)
    assert _values(action.get_output_pl()) == {2017: 10.0, 2018: 9.0, 2019: 9.0, 2020: 9.0, 2021: 8.0, 2022: 7.0}


def test_an_all_forecast_output_without_historical_year_shifts_whole():
    ctx = _make_context('delay-all-forecast')
    ctx.instance.maximum_historical_year = None
    enabled = {2019: 8.0, 2020: 7.0, 2021: 6.0, 2022: 5.0}
    disabled = dict.fromkeys(enabled, 10.0)
    action = _action(ctx, enabled, disabled, delay=2)
    assert _values(action.get_output_pl()) == {2019: 10.0, 2020: 10.0, 2021: 8.0, 2022: 7.0}
