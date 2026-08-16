from pathlib import Path

import pytest

from nodes.management.commands.test_graphql import _diff_responses

pytestmark = pytest.mark.django_db


def _response(impacts: list[dict]) -> dict:
    return {'data': {'page': {'impacts': impacts}}}


def test_action_impacts_are_compared_by_action_id() -> None:
    first = {'action': {'id': 'first'}, 'value': 1.0}
    second = {'action': {'id': 'second'}, 'value': 2.0}
    recorded = {'response': _response([first, second])}
    current = _response([second, first])

    assert _diff_responses(Path('query.json'), recorded, current) is False


def test_action_impact_values_are_still_compared() -> None:
    recorded = {'response': _response([{'action': {'id': 'first'}, 'value': 1.0}])}
    current = _response([{'action': {'id': 'first'}, 'value': 2.0}])

    assert _diff_responses(Path('query.json'), recorded, current) is True
