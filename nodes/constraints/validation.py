"""
Solver-backed structural validation for binding edits and publication.

A candidate edit is expressed as a :class:`BindingChange` and validated by
comparing two solves: the current graph (baseline) and the graph with the
change applied. Only conflicts absent from the baseline reject the change,
so pre-existing model debt never blocks an unrelated edit, while anything
the candidate introduces — directly or by making distant facts contradictory
— is reported with full origin provenance.

Binding additions and removals ride the solver's :class:`GraphOverlay`.
Additions the overlay deliberately cannot express (a new authored input
port, a dataset absent from the graph's bound-dataset catalog) are applied
by round-tripping the graph through its serialized form — the same path the
L2 cache uses — so the hypothetical graph is a fully rebound, independent
value and the cached graph is never touched.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nodes.constraints.solver import ConstraintSolveResult, GraphOverlay
from nodes.dataset_shape import bound_dataset_metric_pairs, load_dataset_shape_profiles
from nodes.defs.binding_def import DatasetBindingDef
from nodes.instance_graph import InstanceGraph

if TYPE_CHECKING:
    from uuid import UUID

    from nodes.constraints.values import ConstraintConflict
    from nodes.dataset_shape import DatasetMetricPair, DatasetShapeProfile
    from nodes.defs.binding_def import AnyPortBindingDef
    from nodes.defs.graph import DatasetMeta
    from nodes.defs.port_def import InputPortDef
    from nodes.instance_graph_cache import ResolvedInstanceSource
    from nodes.models import InstanceConfig


@dataclass(frozen=True, slots=True)
class BindingChange:
    """A candidate edit to the instance's input-port bindings."""

    add_bindings: tuple[AnyPortBindingDef, ...] = ()
    """New bindings, as unbound definitions built the same way the graph builder builds them."""
    remove_binding_ids: frozenset[UUID] = frozenset()
    """Bindings displaced by the change (replace/rebind flows)."""
    add_input_ports: tuple[tuple[UUID, InputPortDef], ...] = ()
    """Ports created as part of the change, keyed by target node UUID."""
    add_datasets: tuple[DatasetMeta, ...] = ()
    """Catalog entries for bound datasets the graph has not seen before."""

    @property
    def needs_hypothetical_graph(self) -> bool:
        return bool(self.add_input_ports or self.add_datasets)


@dataclass(frozen=True, slots=True)
class BindingValidation:
    """Outcome of validating one :class:`BindingChange`."""

    new_conflicts: tuple[ConstraintConflict, ...]
    """Conflicts introduced by the change; empty means the change is acceptable."""
    result: ConstraintSolveResult
    """The full solve of the graph with the change applied."""

    @property
    def ok(self) -> bool:
        return not self.new_conflicts


def graph_with_additions(graph: InstanceGraph, change: BindingChange) -> InstanceGraph:
    """
    Build an independent copy of ``graph`` carrying the change's port and dataset additions.

    Graph-owned children refuse rebinding, so the copy goes through the
    serialized representation (as the L2 cache does); deserialization
    re-runs child binding and derived-state construction from scratch.
    """
    data = graph.model_dump(mode='json')
    nodes_by_id = {node['id']: node for node in data['nodes']}
    for node_uuid, port in change.add_input_ports:
        node = nodes_by_id.get(str(node_uuid))
        if node is None:
            raise ValueError(f'Cannot add port to unknown node {node_uuid}')
        node['spec']['input_ports'].append(port.model_dump(mode='json'))
    known_datasets = {dataset['id'] for dataset in data['datasets']}
    for dataset in change.add_datasets:
        if str(dataset.id) in known_datasets:
            continue
        data['datasets'].append(dataset.model_dump(mode='json'))
    return InstanceGraph.model_validate(data)


def _profiles_for(
    config: InstanceConfig,
    graph: InstanceGraph,
    source: ResolvedInstanceSource,
    add_bindings: tuple[AnyPortBindingDef, ...] = (),
) -> dict[DatasetMetricPair, DatasetShapeProfile]:
    pairs = set(bound_dataset_metric_pairs(graph))
    for binding in add_bindings:
        if isinstance(binding, DatasetBindingDef) and binding.dataset_uuid is not None and binding.metric_uuid is not None:
            pairs.add((binding.dataset_uuid, binding.metric_uuid))
    return load_dataset_shape_profiles(config, graph, source, pairs=pairs)


def validate_binding_change(
    config: InstanceConfig,
    graph: InstanceGraph,
    source: ResolvedInstanceSource,
    change: BindingChange,
) -> BindingValidation:
    """
    Solve the graph with and without the candidate change and report what the change introduces.

    Pre-existing conflicts are the baseline and never reject a change; a
    conflict is attributed to the change when it is absent from the baseline
    solve, which also covers conflicts the candidate causes at a distance.
    """
    target = graph_with_additions(graph, change) if change.needs_hypothetical_graph else graph
    profiles = _profiles_for(config, target, source, change.add_bindings)
    baseline = graph.solve_constraints(profiles=profiles)
    overlay = GraphOverlay(add_bindings=change.add_bindings, remove_binding_ids=change.remove_binding_ids)
    result = target.solve_constraints(profiles=profiles, overlay=overlay)
    known = set(baseline.conflicts)
    new_conflicts = tuple(conflict for conflict in result.conflicts if conflict not in known)
    return BindingValidation(new_conflicts=new_conflicts, result=result)


def solve_instance_constraints(
    config: InstanceConfig,
    graph: InstanceGraph,
    source: ResolvedInstanceSource,
) -> ConstraintSolveResult:
    """Solve the complete graph with observed dataset shape profiles."""
    return graph.solve_constraints(profiles=_profiles_for(config, graph, source))


class InstanceConstraintError(Exception):
    """Structural constraint conflicts that block publication or strict computation."""

    def __init__(self, conflicts: tuple[ConstraintConflict, ...]) -> None:
        self.conflicts = conflicts
        summary = '; '.join(conflict.message for conflict in conflicts[:5])
        if len(conflicts) > 5:
            summary += f' (and {len(conflicts) - 5} more)'
        super().__init__(f'Instance has {len(conflicts)} structural constraint conflict(s): {summary}')


def require_valid_instance_constraints(
    config: InstanceConfig,
    graph: InstanceGraph,
    source: ResolvedInstanceSource,
) -> ConstraintSolveResult:
    """Strict whole-graph validation: any conflict raises :class:`InstanceConstraintError`."""
    result = solve_instance_constraints(config, graph, source)
    if result.conflicts:
        raise InstanceConstraintError(result.conflicts)
    return result
