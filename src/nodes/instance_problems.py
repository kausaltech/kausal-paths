"""Shared model-editor checks, independent of GraphQL and municipal finding counts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from common.validation import ValidationOperation, blocks_operation
from datasets.materialization import collect_instance_dataset_violations
from nodes.constraints.validation import solve_instance_constraints
from nodes.instance_graph import build_instance_graph
from nodes.instance_graph_cache import resolve_instance_source
from nodes.instance_loader import InstanceLoader
from nodes.instance_serialization import build_instance_snapshot
from nodes.models import PreferredInstanceSource
from nodes.value_validation import collect_instance_value_violations

if TYPE_CHECKING:
    from datasets.validation import RuleViolation
    from nodes.constraints.solver import ConstraintSolveResult
    from nodes.constraints.values import ConstraintConflict
    from nodes.data_entry import EntryProblem
    from nodes.instance_graph import InstanceGraph
    from nodes.models import InstanceConfig
    from nodes.value_validation import ValueValidationViolation


def data_entry_definition_problems(graph: InstanceGraph) -> list[EntryProblem]:
    """Only authored layout failures; global optional input states are not layout defects."""
    if graph.spec.data_entry is None:
        return []
    return [problem for problem in graph.data_entry.problems if problem.blocks_publication]


class DataEntryDefinitionError(ValueError):
    def __init__(self, problems: list[EntryProblem]) -> None:
        self.problems = problems
        super().__init__('Invalid data-entry definition: ' + '; '.join(problem.message for problem in problems))


def require_valid_data_entry_definition(graph: InstanceGraph) -> None:
    problems = data_entry_definition_problems(graph)
    if problems:
        raise DataEntryDefinitionError(problems)


@dataclass
class InstanceProblems:
    constraints: tuple[ConstraintConflict, ...]
    datasets: list[RuleViolation]
    values: list[ValueValidationViolation]
    data_entry: list[EntryProblem]

    def blocking(self, operation: ValidationOperation) -> InstanceProblems:
        return InstanceProblems(
            constraints=self.constraints,
            datasets=[problem for problem in self.datasets if blocks_operation(problem.enforcement, operation)],
            values=[problem for problem in self.values if blocks_operation(problem.enforcement, operation)],
            data_entry=self.data_entry if operation != 'edit' else [],
        )

    @property
    def messages(self) -> list[str]:
        return (
            [problem.message for problem in self.constraints]
            + [problem.message for problem in self.datasets]
            + [problem.message for problem in self.values]
            + [problem.message for problem in self.data_entry]
        )


def collect_instance_problems(
    config: InstanceConfig,
    *,
    graph: InstanceGraph | None = None,
    constraints: ConstraintSolveResult | None = None,
    value_violations: list[ValueValidationViolation] | None = None,
) -> InstanceProblems:
    """Use request-provided results, or inspect a fresh draft for provisioning checks."""
    if graph is None:
        graph = build_instance_graph(build_instance_snapshot(config))
    if constraints is None:
        constraints = solve_instance_constraints(config, graph, resolve_instance_source(config, PreferredInstanceSource.DRAFT))
    if value_violations is None:
        value_violations = []
        # Broken structure is already reported; it may not support computation yet.
        if not constraints.conflicts and any(
            port.validation is not None for node in graph.nodes for port in node.spec.input_ports
        ):
            instance = InstanceLoader.from_snapshot(build_instance_snapshot(config), instance_config=config).instance
            try:
                value_violations = collect_instance_value_violations(instance)
            finally:
                instance.clean()
    return InstanceProblems(
        constraints.conflicts,
        collect_instance_dataset_violations(config),
        value_violations,
        data_entry_definition_problems(graph),
    )
