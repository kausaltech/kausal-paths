"""
A dashboard card shows its own `year` when it has one, and the target year otherwise.

The year exists for quantities whose year of interest is not the target year, such as a
budget that a scenario uses up by the end of the model: at the target year such a card would
report a scenario as within its budget although it overdraws it a year later.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from pages.blocks import DashboardCardBlock

pytestmark = pytest.mark.django_db


def _node(target_year: int) -> Any:
    scenarios = {'a': SimpleNamespace(id='a'), 'b': SimpleNamespace(id='b')}
    return SimpleNamespace(get_target_year=lambda: target_year, context=SimpleNamespace(scenarios=scenarios))


@pytest.mark.parametrize(('values', 'expected'), [({'year': 2050}, 2050), ({'year': None}, 2035), ({}, 2035)])
def test_card_year_falls_back_to_the_target_year(values: dict, expected: int) -> None:
    assert DashboardCardBlock()._card_year(_node(2035), values) == expected


def test_scenario_values_are_read_in_the_card_year(monkeypatch: pytest.MonkeyPatch) -> None:
    block = DashboardCardBlock()
    node = _node(2035)
    asked: list[tuple[int, str]] = []

    def value_for_year(_node: Any, year: int, scenario: Any = None) -> float:
        asked.append((year, scenario.id))
        return 1.0

    monkeypatch.setattr(block, 'node', lambda _info, _values: node)
    monkeypatch.setattr(block, '_value_for_year', value_for_year)

    result = list(block.scenario_values(info=None, values={'year': 2050}))  # type: ignore[arg-type]

    assert asked == [(2050, 'a'), (2050, 'b')]
    assert {value.year for value in result} == {2050}
