from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.instance import Instance
    from nodes.node import Node
    from nodes.scenario import Scenario
    from params import Parameter

pytestmark = pytest.mark.django_db


def test_context_get_parameter_global(context: Context, number_parameter: Parameter[Any]):
    context.add_global_parameter(number_parameter)
    assert number_parameter.global_id == number_parameter.local_id
    assert context.get_parameter(number_parameter.global_id) == number_parameter


def test_context_get_parameter_local(context: Context, node: Node, number_parameter: Parameter[Any]):
    node.add_parameter(number_parameter)
    assert number_parameter.global_id != number_parameter.local_id
    assert context.get_parameter(number_parameter.global_id) == number_parameter


def test_context_activate_scenario_sets_active_scenario(context: Context, scenario: Scenario):
    assert context.active_scenario != scenario
    context.activate_scenario(scenario)
    assert context.active_scenario == scenario


def _set_cache_identity(instance: Instance, *, invalidated_at: datetime) -> None:
    instance.__dict__['config'] = SimpleNamespace(
        uuid=UUID('3a45b35f-bb52-4145-bb8f-f6d506689246'),
        cache_invalidated_at=invalidated_at,
    )


def test_instance_hash_changes_when_instance_cache_is_invalidated():
    from nodes.tests.factories import InstanceFactory

    first = InstanceFactory.create(id='same-instance')
    second = InstanceFactory.create(id='same-instance')
    _set_cache_identity(first, invalidated_at=datetime(2026, 1, 1, tzinfo=UTC))
    _set_cache_identity(second, invalidated_at=datetime(2026, 1, 2, tzinfo=UTC))

    assert first.context.instance_hash != second.context.instance_hash


def test_node_hash_includes_instance_timeline():
    from nodes.node_cache import HashingState
    from nodes.tests.factories import InstanceFactory, SimpleNodeFactory

    first = InstanceFactory.create(id='same-instance', maximum_historical_year=2022)
    second = InstanceFactory.create(id='same-instance', maximum_historical_year=2023)
    invalidated_at = datetime(2026, 1, 1, tzinfo=UTC)
    _set_cache_identity(first, invalidated_at=invalidated_at)
    _set_cache_identity(second, invalidated_at=invalidated_at)
    first_node = SimpleNodeFactory.create(id='same_node', context=first.context)
    second_node = SimpleNodeFactory.create(id='same_node', context=second.context)

    first_hash = first_node.hasher.calculate_hash(HashingState())
    second_hash = second_node.hasher.calculate_hash(HashingState())

    assert first.context.instance_hash != second.context.instance_hash
    assert first_hash != second_hash
    assert any(part_type == 'instance' for part_type, _part, _value in first_node.hasher.last_hash_parts)
