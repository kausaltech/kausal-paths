from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import loguru
import pytest

from datasets.validation import RuleViolation
from nodes.constraints.values import BindingValue, ConstraintConflict, ConstraintOrigin, DatasetSourceValue
from nodes.instance_graph_cache import ResolvedInstanceSource
from nodes.management.commands import test_instance
from nodes.management.commands.test_instance import (
    CheckState,
    Command,
    InstanceDetail,
    NodeDetail,
    ProblemDetail,
    resolve_all_nodes,
)

if TYPE_CHECKING:
    from pathlib import Path

    from nodes.context import Context
    from nodes.instance import Instance
    from nodes.instance_graph import InstanceGraph
    from nodes.models import InstanceConfig


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


@pytest.mark.parametrize(
    ('store', 'all_nodes', 'outcomes_only', 'expected'),
    [
        (True, False, False, True),
        (True, False, True, False),
        (False, False, False, False),
        (False, True, False, True),
    ],
)
def test_resolve_all_nodes(*, store: bool, all_nodes: bool, outcomes_only: bool, expected: bool) -> None:
    assert resolve_all_nodes(store=store, all_nodes=all_nodes, outcomes_only=outcomes_only) is expected


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


def make_problem(message: str = 'Filter keeps no observed category') -> ProblemDetail:
    return ProblemDetail.from_conflict(
        ConstraintConflict(
            code='disjoint_category_filter',
            message=message,
            value=BindingValue(UUID(int=1)),
            origins=(ConstraintOrigin(kind='binding', binding_id=UUID(int=1)),),
        )
    )


def test_problem_state_round_trip(tmp_path: Path) -> None:
    conflict = make_problem()
    violation = ProblemDetail.from_violation(
        RuleViolation(
            rule_uuid=UUID(int=2),
            kind='value_range',
            enforcement='block_publish',
            metric_uuid=UUID(int=3),
            metric='energy',
            dataset_uuid=UUID(int=4),
            years=[2020],
            categories={'sector': 'transport'},
            message='Outside allowed range',
        )
    )
    command = make_command(reference_failed=False)
    command.compare = False
    state_file = tmp_path / 'state.json'
    command.state.set_output_file(state_file)

    assert command.check_problems(loguru.logger, 'test-instance', [conflict, violation])
    command.state.save()
    restored = CheckState.model_validate_json(state_file.read_text())
    assert restored.instance_details[0].problems == [conflict, violation]
    assert violation.severity == 'warning'
    assert conflict.details['value_kind'] == 'BindingValue'


def test_problem_output_includes_location_and_message() -> None:
    command = make_command(reference_failed=False)
    command.compare = False
    output = StringIO()
    sink = loguru.logger.add(output, format='{message}')
    try:
        command.check_problems(loguru.logger, 'test-instance', [make_problem()])
    finally:
        loguru.logger.remove(sink)

    printed = output.getvalue()
    assert 'Validation problems for test-instance: 1' in printed
    assert '[disjoint_category_filter] Filter keeps no observed category' in printed
    assert '00000000-0000-0000-0000-000000000001' in printed
    assert 'origins' in printed


def test_collect_problems_uses_loaded_graph_and_observed_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = cast('InstanceGraph', object())
    config = cast('InstanceConfig', object())
    instance = cast('Instance', SimpleNamespace(context=SimpleNamespace(instance_graph=graph)))
    source = ResolvedInstanceSource(str(UUID(int=1)), 'database-draft', 'test-version')
    events: list[str] = []
    violation = RuleViolation(
        rule_uuid=UUID(int=2),
        kind='value_range',
        enforcement='block_edit',
        metric_uuid=UUID(int=3),
        metric='energy',
        message='Outside allowed range',
    )
    conflict = ConstraintConflict('disjoint_category_filter', 'Empty filter', DatasetSourceValue(UUID(int=1)), ())

    def collect_violations(actual_config: InstanceConfig) -> list[RuleViolation]:
        assert actual_config is config
        events.append('materializations')
        return [violation]

    def solve_constraints(
        actual_config: InstanceConfig, actual_graph: InstanceGraph, actual_source: ResolvedInstanceSource
    ) -> SimpleNamespace:
        assert actual_config is config
        assert actual_graph is graph
        assert actual_source is source
        events.append('solve')
        return SimpleNamespace(conflicts=(conflict,))

    monkeypatch.setattr(test_instance, 'resolve_instance_source', lambda *_args: source)
    monkeypatch.setattr(test_instance, 'collect_instance_dataset_violations', collect_violations)
    monkeypatch.setattr(test_instance, 'solve_instance_constraints', solve_constraints)
    monkeypatch.setattr(test_instance, 'collect_instance_value_violations', lambda _instance: [])

    problems = Command().collect_problems(config, instance)

    assert events == ['materializations', 'solve']
    assert {problem.kind for problem in problems} == {'constraint_conflict', 'dataset_validation_violation'}
    assert all(problem.severity == 'error' for problem in problems)


@pytest.mark.parametrize('ignore_fixed', [False, True])
@pytest.mark.parametrize(
    ('reference', 'current', 'expected', 'expected_ignoring_fixed'),
    [
        ([], [], True, True),
        (['a'], ['a'], True, True),
        (['a', 'b'], ['b', 'a'], True, True),
        ([], ['a'], False, False),
        (['a'], [], False, True),
        (['a'], ['b'], False, False),
        (['a'], ['a', 'a'], False, False),
        (['a', 'a'], ['a'], False, True),
    ],
)
def test_compare_problems(
    reference: list[str], current: list[str], *, expected: bool, expected_ignoring_fixed: bool, ignore_fixed: bool
) -> None:
    # Problem regressions must also fail when the reference instance failed computation.
    command = make_command(reference_failed=True)
    command.ignore_fixed_problems = ignore_fixed
    reference_problems = [make_problem(message) for message in reference]
    command.state.instance_details[0].problems = reference_problems

    result = command.check_problems(loguru.logger, 'test-instance', [make_problem(message) for message in current])

    assert result is (expected_ignoring_fixed if ignore_fixed else expected)
    assert command.state.instance_details[0].problems == reference_problems


def test_problem_comparison_retains_value_kind_and_ignores_origin_order() -> None:
    first = ConstraintOrigin(kind='binding', binding_id=UUID(int=1))
    second = ConstraintOrigin(kind='dataset_profile', binding_id=UUID(int=1))
    conflict = ConstraintConflict('disjoint_category_filter', 'Empty filter', BindingValue(UUID(int=1)), (first, second))
    reordered = ConstraintConflict(conflict.code, conflict.message, conflict.value, (second, first))
    dataset_source = ConstraintConflict(conflict.code, conflict.message, DatasetSourceValue(UUID(int=1)), conflict.origins)

    assert ProblemDetail.from_conflict(conflict) == ProblemDetail.from_conflict(reordered)
    assert ProblemDetail.from_conflict(conflict) != ProblemDetail.from_conflict(dataset_source)


def test_old_reference_defaults_to_no_problems() -> None:
    state = CheckState.model_validate_json('{"instance_details": [{"instance_id": "old"}]}')

    assert state.instance_details[0].problems == []


@pytest.mark.parametrize('spec_only', [False, True])
def test_check_instance_fails_on_problem_changes_after_testing(monkeypatch: pytest.MonkeyPatch, *, spec_only: bool) -> None:
    command = make_command(reference_failed=False)
    command.logger = loguru.logger
    command.spec_only = spec_only
    command.store = False
    command.include_impacts = False
    command.trace_rss = command.trace_tracemalloc = command.trace_new_objects = False
    command.dry_run = True
    events: list[str] = []
    ctx = SimpleNamespace(
        cache=SimpleNamespace(clear=lambda: None),
        scenarios={},
        run=nullcontext,
    )
    instance = cast('Instance', SimpleNamespace(id='test-instance', context=ctx, clean=lambda: events.append('clean')))
    config = cast('InstanceConfig', SimpleNamespace(identifier='test-instance', get_instance=lambda: instance))
    monkeypatch.setattr(command, 'collect_problems', lambda _ic, _instance: [make_problem()])

    def run_nodes(_logger: loguru.Logger, _ctx: Context) -> bool:
        events.append('nodes')
        return True

    monkeypatch.setattr(command, 'run_nodes', run_nodes)
    for method in [
        'dump_instance_graph',
        'dump_scenario_manifest',
        'maybe_log_new_objects',
        'maybe_log_tracemalloc',
        'maybe_log_rss',
    ]:
        monkeypatch.setattr(command, method, lambda *_args: None)
    original_check = command.check_problems

    def check_problems(logger: loguru.Logger, instance_id: str, problems: list[ProblemDetail]) -> bool:
        events.append('problems')
        return original_check(logger, instance_id, problems)

    monkeypatch.setattr(command, 'check_problems', check_problems)

    assert not command.check_instance(config)
    assert events == (['problems', 'clean'] if spec_only else ['nodes', 'problems', 'clean'])
    assert command.state.instance_details[0].failure_at == 'problems'
