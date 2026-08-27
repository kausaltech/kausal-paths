from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import loguru
import pytest

from nodes.management.commands.test_instance import CheckState, Command, InstanceDetail, NodeDetail

if TYPE_CHECKING:
    from nodes.context import Context


@pytest.fixture(autouse=True)
def default_scenario() -> None:
    """Override the project-wide fixture; these command tests use lightweight contexts."""


@pytest.fixture(autouse=True)
def instance_config() -> None:
    """Override the project-wide database fixture; these tests do not use Django models."""


def make_context(instance_id: str, nodes: list[Any]) -> Context:
    instance = SimpleNamespace(id=instance_id)
    ctx = SimpleNamespace(
        instance=instance,
        active_scenario=SimpleNamespace(id='default'),
        get_outcome_nodes=lambda: nodes,
    )
    for node in nodes:
        node.context = ctx
    return cast('Context', ctx)


def make_command(*, reference_failed: bool) -> Command:
    command = Command()
    command.all_nodes = False
    command.compare = True
    command.check_perf = False
    command.maxfail = 1
    command.nr_fails = 0
    command.state = CheckState(
        failed_instances={'test-instance'} if reference_failed else set(),
        instance_details=[
            InstanceDetail(
                instance_id='test-instance',
                failure_at='nodes' if reference_failed else None,
                nodes=[NodeDetail(node_id='outcome')],
            )
        ],
    )
    return command


@pytest.mark.parametrize(('reference_failed', 'expected'), [(True, True), (False, False)])
def test_run_nodes_uses_instance_failure_as_comparison_fallback(
    monkeypatch: pytest.MonkeyPatch, *, reference_failed: bool, expected: bool
) -> None:
    node = SimpleNamespace(id='outcome')
    ctx = make_context('test-instance', [node])
    command = make_command(reference_failed=reference_failed)
    monkeypatch.setattr(command, 'check_node', lambda _node: 'output')

    assert command.run_nodes(loguru.logger, ctx) is expected
    assert command.nr_fails == (0 if reference_failed else 1)


@pytest.mark.parametrize(('reference_failed', 'expected'), [(True, True), (False, False)])
def test_run_action_impacts_uses_instance_failure_as_comparison_fallback(
    monkeypatch: pytest.MonkeyPatch, *, reference_failed: bool, expected: bool
) -> None:
    node = SimpleNamespace(id='outcome')
    ctx = make_context('test-instance', [node])
    action = SimpleNamespace(
        id='action',
        is_enabled=lambda: True,
        get_downstream_nodes=lambda **_kwargs: [node],
    )
    ctx_with_actions = cast('Any', ctx)
    ctx_with_actions.get_actions = lambda: [action]
    command = make_command(reference_failed=reference_failed)
    monkeypatch.setattr(command, 'handle_action_impact_output', lambda _logger, _action, _node: 'output')

    assert command.run_action_impacts(loguru.logger, ctx) is expected
    assert command.nr_fails == (0 if reference_failed else 1)


def test_failed_instance_detected_from_instance_details() -> None:
    state = CheckState(instance_details=[InstanceDetail(instance_id='test-instance', failure_at='nodes')])

    assert state.has_failed_instance('test-instance')


def test_successful_instances_excludes_both_failure_representations() -> None:
    state = CheckState(
        checked_instances={'passing', 'failed-set', 'failed-detail'},
        failed_instances={'failed-set'},
        instance_details=[InstanceDetail(instance_id='failed-detail', failure_at='init')],
    )

    assert state.successful_instances() == {'passing'}
