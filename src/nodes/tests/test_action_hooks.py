"""Actions acting on other nodes' outputs (`nodes.hooks`)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import polars as pl
import pytest

from nodes.constants import YEAR_COLUMN
from nodes.exceptions import NodeError
from nodes.instance_loader import InstanceLoader
from nodes.instance_parser import InstanceParseError, parse_instance_snapshot
from nodes.instance_serialization import InstanceSnapshot

pytestmark = pytest.mark.django_db


def _config(*, actions: list[dict[str, Any]], nodes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        'id': 'hook_test',
        'default_language': 'en',
        'supported_languages': [],
        'name': 'Hook test',
        'owner': 'Owner',
        'target_year': 2030,
        'model_end_year': 2030,
        'minimum_historical_year': 2018,
        'maximum_historical_year': 2020,
        'reference_year': 2018,
        'nodes': nodes
        if nodes is not None
        else [
            _energy_node('demand', historical_values=[[2018, 100], [2019, 100], [2020, 100]], forecast_values=[[2025, 100]]),
            _energy_node('total', input_nodes=['demand']),
        ],
        'actions': actions,
    }


def _energy_node(node_id: str, **extra: Any) -> dict[str, Any]:
    return {'id': node_id, 'type': 'simple.AdditiveNode', 'name': node_id, 'unit': 'kWh/a', 'quantity': 'energy', **extra}


def _saving(**extra: Any) -> dict[str, Any]:
    return {
        'id': 'saving',
        'type': 'simple.AdditiveAction',
        'name': 'Saving',
        'unit': 'kWh/a',
        'quantity': 'energy',
        # Historical values are the realised part of the measure; they must never enter the balance.
        'historical_values': [[2018, -5], [2020, -5]],
        'forecast_values': [[2025, -10]],
        'output_nodes': [{'id': 'demand', 'hook': True}],
        **extra,
    }


def _context(config: dict[str, Any]):
    return InstanceLoader(snapshot=parse_instance_snapshot(config, instance_uuid=uuid4())).context


def _values(df, years: tuple[int, ...] = (2018, 2019, 2020, 2025)) -> dict[int, float]:
    values = dict(zip(df[YEAR_COLUMN].to_list(), df[df.metric_cols[0]].to_list(), strict=True))
    return {year: values[year] for year in years if year in values}


def test_hook_adds_the_action_to_the_target_output_in_forecast_years_only() -> None:
    ctx = _context(_config(actions=[_saving()]))
    with ctx.run():
        saving = ctx.get_action('saving')
        saving.enabled_param.set(True)
        assert _values(ctx.get_node('total').get_output_pl()) == {2018: 100, 2019: 100, 2020: 100, 2025: 90}
        saving.enabled_param.set(False)
        assert _values(ctx.get_node('total').get_output_pl()) == {2018: 100, 2019: 100, 2020: 100, 2025: 100}


def test_hook_is_not_an_input_of_the_target_but_is_upstream_of_it() -> None:
    ctx = _context(_config(actions=[_saving()]))
    demand, total, saving = ctx.get_node('demand'), ctx.get_node('total'), ctx.get_action('saving')
    assert saving not in demand.input_nodes
    assert demand not in saving.output_nodes
    assert [hook.action for hook in demand.hooks] == [saving]
    assert saving.is_connected_to(total)
    assert saving in total.get_upstream_nodes()


def test_action_impact_through_a_hook() -> None:
    ctx = _context(_config(actions=[_saving()]))
    with ctx.run():
        saving = ctx.get_action('saving')
        saving.enabled_param.set(True)
        impact = saving.compute_impact(ctx.get_node('total'))
        values = impact.filter(pl.col('Impact') == 'Impact')
        assert _values(values) == {2018: 0, 2019: 0, 2020: 0, 2025: -10}


def test_hooks_on_one_node_add_up() -> None:
    other = _saving(id='other', forecast_values=[[2025, -1]])
    ctx = _context(_config(actions=[_saving(), other]))
    with ctx.run():
        for action_id in ('saving', 'other'):
            ctx.get_action(action_id).enabled_param.set(True)
        assert _values(ctx.get_node('demand').get_output_pl())[2025] == 89


def test_hook_survives_the_snapshot_round_trip() -> None:
    """The DB stores the spec as JSON; the hook must come back from it."""
    snapshot = parse_instance_snapshot(_config(actions=[_saving()]), instance_uuid=uuid4())
    restored = InstanceSnapshot.model_validate_json(snapshot.model_dump_json())
    ctx = InstanceLoader(snapshot=restored).context
    with ctx.run():
        ctx.get_action('saving').enabled_param.set(True)
        assert _values(ctx.get_node('demand').get_output_pl())[2025] == 90


def test_hook_from_a_downstream_node_is_a_cycle() -> None:
    config = _config(actions=[_saving(input_nodes=['total'], type='simple.AdditiveAction')])
    with pytest.raises(Exception, match='loops'):
        _context(config)


def test_only_actions_can_hook() -> None:
    nodes = [
        _energy_node('demand', historical_values=[[2020, 1]]),
        _energy_node('other', historical_values=[[2020, 1]], output_nodes=[{'id': 'demand', 'hook': True}]),
    ]
    with pytest.raises(InstanceParseError, match='only actions'):
        parse_instance_snapshot(_config(actions=[], nodes=nodes), instance_uuid=uuid4())


def test_hook_with_mismatched_dimensions_fails_at_compute() -> None:
    config = _config(actions=[_saving()])
    config['dimensions'] = [{'id': 'sector', 'label': 'Sector', 'categories': [{'id': 'a', 'label': 'A'}]}]
    config['nodes'][0]['output_dimensions'] = ['sector']
    config['nodes'][0]['input_dimensions'] = ['sector']
    config['nodes'][0].pop('historical_values')
    config['nodes'][0].pop('forecast_values')
    config['nodes'][1]['input_nodes'] = [{'id': 'demand', 'from_dimensions': [{'id': 'sector', 'flatten': True}]}]
    config['nodes'].insert(
        0,
        {
            **_energy_node('source', historical_values=[[2020, 1]], forecast_values=[[2025, 1]]),
            'output_nodes': [{'id': 'demand', 'to_dimensions': [{'id': 'sector', 'categories': ['a']}]}],
        },
    )
    ctx = _context(config)
    with ctx.run():
        ctx.get_action('saving').enabled_param.set(True)
        with pytest.raises(NodeError, match='do not match'):
            ctx.get_node('demand').get_output_pl()


def _factor_node(value: float = 0.8) -> dict[str, Any]:
    return {
        'id': 'factor',
        'type': 'simple.AdditiveNode',
        'name': 'Factor',
        'unit': 'dimensionless',
        'quantity': 'fraction',
        'historical_values': [[2020, 0.5]],
        'forecast_values': [[2025, value]],
    }


def _relative(action_id: str = 'relative', factor: str = 'factor') -> dict[str, Any]:
    """Build an action that makes a relative effect explicit: it states the amount the hook adds."""
    return {
        'id': action_id,
        'type': 'formula.FormulaAction',
        'name': 'Relative',
        'unit': 'kWh/a',
        'quantity': 'energy',
        'input_nodes': [factor],
        'params': [{'id': 'formula', 'value': f'demand * ({factor} - 1)'}],
        'output_nodes': [{'id': 'demand', 'hook': True}],
    }


def _relative_config(*actions: dict[str, Any], factors: tuple[float, ...] = (0.8,)) -> dict[str, Any]:
    config = _config(actions=list(actions))
    for idx, value in enumerate(factors):
        node = _factor_node(value)
        if idx:
            node['id'] = f'factor{idx}'
        config['nodes'].append(node)
    return config


def test_formula_action_reads_the_un_hooked_value_of_the_node_it_acts_on() -> None:
    ctx = _context(_relative_config(_relative()))
    action = ctx.get_action('relative')
    assert [hook.reads_base for hook in action.hook_targets] == [True]
    with ctx.run():
        action.enabled_param.set(True)
        assert _values(ctx.get_node('total').get_output_pl()) == {2018: 100, 2019: 100, 2020: 100, 2025: 80}
        action.enabled_param.set(False)
        assert _values(ctx.get_node('total').get_output_pl())[2025] == 100


def test_relative_actions_share_the_un_hooked_value_and_do_not_compound() -> None:
    """Two -20 % changes remove 40 % of the un-hooked value, not 36 %."""
    ctx = _context(_relative_config(_relative(), _relative('again', 'factor1'), factors=(0.8, 0.8)))
    with ctx.run():
        for action_id in ('relative', 'again'):
            ctx.get_action(action_id).enabled_param.set(True)
        assert _values(ctx.get_node('demand').get_output_pl())[2025] == pytest.approx(60)


def test_changing_the_target_invalidates_the_action_reading_it() -> None:
    from nodes.node_cache import HashingState

    ctx = _context(_relative_config(_relative()))
    demand, action = ctx.get_node('demand'), ctx.get_action('relative')
    demand.hasher.calculate_hash(HashingState())
    assert action.hasher._get_cached_hash() is not None
    demand.hasher.mark_modified()
    assert action.hasher._get_cached_hash() is None
    assert demand.hasher._get_cached_hash() is None
