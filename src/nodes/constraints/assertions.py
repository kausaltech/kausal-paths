"""Find the output-dimension assertions that shape inference makes redundant."""

from dataclasses import dataclass
from uuid import UUID

from nodes.constraints.values import PortValue
from nodes.defs.binding_def import EdgeBindingDef
from nodes.instance_graph import InstanceGraph


@dataclass(frozen=True, slots=True)
class OutputAssertion:
    node_id: UUID
    port_id: UUID
    node_class: str
    asserted: tuple[str, ...]
    inferred: tuple[str, ...] | None
    """What the solver infers with every redundant assertion withdrawn; ``None`` when it cannot."""
    is_outcome: bool

    @property
    def redundant(self) -> bool:
        """Whether inference reproduces the assertion, and nothing else needs it to be stated."""
        return not self.is_outcome and self.inferred is not None and set(self.inferred) == set(self.asserted)


def classify_output_assertions(graph: InstanceGraph) -> list[OutputAssertion]:
    """
    Find which output assertions inference can stand in for, and what it infers for the others.

    An assertion on a node without shape rules is often what lets the nodes downstream of it
    infer anything, so the assertions cannot be judged one by one or all withdrawn at once.
    Start with every assertion withdrawn. Of the ports inference then fails to reproduce, put
    back only those with no failing node upstream — a failure further down may be just the
    consequence — and solve again, until inference reproduces every withdrawn assertion.
    Putting one back only adds facts, so a port found redundant stays redundant, and each
    round puts at least one back, so the loop ends. Upstream-first is a choice: the solver
    also infers backwards, and the opposite order would keep consumers' assertions instead.

    An outcome node's assertion always stays: it is the requested result shape, which
    inference checks rather than replaces. The solve uses no dataset profiles, like the
    runtime that resolves the same ports.
    """
    asserted = {
        (node.id, port.id): (node, port) for node in graph.nodes for port in node.spec.output_ports if port.dimensions is not None
    }
    ancestors = _ancestors(graph)
    kept = {key for key, (node, _port) in asserted.items() if node.spec.is_outcome}
    # For each assertion, what the last solve that withdrew it inferred in its place.
    inferred_in_place: dict[tuple[UUID, UUID], tuple[str, ...] | None] = {}
    while True:
        withdrawn = set(asserted) - kept
        inferred = _infer_without(graph, withdrawn)
        inferred_in_place.update({key: inferred.get(key) for key in withdrawn})
        failing = {
            key
            for key in withdrawn
            if (found := inferred.get(key)) is None or set(found) != set(asserted[key][1].dimensions or ())
        }
        if not failing:
            break
        failing_nodes = {node_id for node_id, _port_id in failing}
        kept |= {key for key in failing if not ancestors[key[0]] & failing_nodes}

    return [
        OutputAssertion(
            node_id=node.id,
            port_id=port.id,
            node_class=node.node_class_path,
            asserted=tuple(port.dimensions or ()),
            inferred=inferred_in_place.get(key),
            is_outcome=node.spec.is_outcome,
        )
        for key, (node, port) in asserted.items()
    ]


def _ancestors(graph: InstanceGraph) -> dict[UUID, frozenset[UUID]]:
    """Every node's transitive upstream nodes, through edge bindings."""
    upstream: dict[UUID, set[UUID]] = {node.id: set() for node in graph.nodes}
    for binding in graph.bindings:
        if (
            isinstance(binding, EdgeBindingDef)
            and binding.port_ref.node_uuid is not None
            and binding.from_ref.node_uuid is not None
        ):
            upstream.setdefault(binding.port_ref.node_uuid, set()).add(binding.from_ref.node_uuid)

    result: dict[UUID, frozenset[UUID]] = {}

    def visit(node_id: UUID) -> frozenset[UUID]:
        if node_id not in result:
            result[node_id] = frozenset()  # a cycle is a graph diagnostic elsewhere; do not recurse forever
            found: set[UUID] = set()
            for parent in upstream.get(node_id, ()):
                found.add(parent)
                found |= visit(parent)
            result[node_id] = frozenset(found)
        return result[node_id]

    for node_id in upstream:
        visit(node_id)
    return result


def _infer_without(graph: InstanceGraph, withdrawn: set[tuple[UUID, UUID]]) -> dict[tuple[UUID, UUID], tuple[str, ...] | None]:
    """Solve the graph with the given output assertions withdrawn, and report every output port's dimensions."""
    data = graph.model_dump(mode='json')
    for node in data['nodes']:
        for port in node['spec']['output_ports']:
            if (UUID(node['id']), UUID(port['id'])) in withdrawn:
                port['dimensions'] = None
    result = InstanceGraph.model_validate(data).solve_constraints()
    inferred: dict[tuple[UUID, UUID], tuple[str, ...] | None] = {}
    for node in graph.nodes:
        for port in node.spec.output_ports:
            shape = result.shapes.get(PortValue(node.id, port.id, 'output'))
            if shape is None or shape.dimensions is None:
                inferred[(node.id, port.id)] = None
            else:
                inferred[(node.id, port.id)] = tuple(sorted(graph.dimension_by_id[dim].identifier for dim in shape.dimensions))
    return inferred
