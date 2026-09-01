"""
Compile node-class shape rules against an ``InstanceGraph``.

Compilation resolves each node class's declared rules and validates them
against that node's ports and the instance dimension registry. Two failure
modes are deliberately kept apart:

* a structurally invalid rule (wrong port direction, unknown value,
  cycle, unknown dimension) is a bug in a node class or a rule compiler
  and raises ``ShapeRuleError`` naming the class;
* a node whose legacy spec simply lacks ports for a required role compiles
  to no rules and a ``missing_role_port`` diagnostic — incompleteness never
  blocks anything.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nodes.constraints.rules import (
    AnyShapeRule,
    MissingPortRoleError,
    ShapeRuleError,
    rule_dimension_ids,
    rule_input_ids,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from nodes.instance_graph import InstanceGraph, InstanceGraphDiagnostic, NodeMeta


@dataclass(frozen=True)
class ShapeRuleCompilation:
    """Derived, non-serialized result of compiling every node's shape rules."""

    rules_by_node: Mapping[UUID, tuple[AnyShapeRule, ...]]
    diagnostics: tuple[InstanceGraphDiagnostic, ...]
    untrusted_node_ids: frozenset[UUID] = frozenset()
    """
    Nodes whose class computation is overridden below the ``shape_rules``
    declaration. Their inherited rules are skipped, and the solver must not
    apply structural claims that ride on the class contract either — most
    importantly multi-port aggregate homogeneity, which legacy specs of such
    classes routinely violate (heterogeneous inputs grouped onto one port).
    """


_SEMANTIC_COMPUTE_METHODS = ('compute', '_compute', 'perform_operation', 'operate_pairwise')


def _computation_override_below_rules(node_class: type) -> type | None:
    """
    Find a subclass that changes the computation below the ``shape_rules`` declaration.

    A rule describes the algebra of the class that declared it. A subclass
    that overrides the computation without re-declaring its rules (e.g. a
    legacy city-specific ``AdditiveNode`` subclass with a custom mix-weighted
    ``compute()``) inherits rules that may *lie* about its shapes; those
    compile to nothing plus a diagnostic. Re-declaring ``shape_rules`` in the
    subclass — even as a plain re-assignment — is the explicit opt-in.
    """
    from nodes.node import Node

    rules_owner = next(klass for klass in node_class.__mro__ if 'shape_rules' in klass.__dict__)
    if rules_owner is Node:
        return None  # the base declaration is empty; there is nothing to lie about
    for klass in node_class.__mro__:
        if klass is rules_owner:
            return None
        if any(method in klass.__dict__ for method in _SEMANTIC_COMPUTE_METHODS):
            return klass
    return None


def _validate_node_rules(graph: InstanceGraph, meta: NodeMeta, rules: Sequence[AnyShapeRule]) -> None:  # noqa: C901
    input_port_ids = {port.id for port in meta.spec.input_ports}
    output_port_ids = {port.id for port in meta.spec.output_ports}

    def fail(message: str) -> None:
        raise ShapeRuleError(f'{meta.node_class_path} (node {meta.identifier or meta.id}): {message}')

    # Several rules constraining one *output port* is legitimate — that is how
    # "the additive aggregate conforms to the product result" is expressed.
    # An intermediate, by contrast, is defined by exactly one rule.
    produced = [rule.output for rule in rules]
    produced_set = set(produced)
    intermediate_outputs = [value for value in produced if value not in output_port_ids]
    if len(intermediate_outputs) != len(set(intermediate_outputs)):
        fail('multiple rules produce the same intermediate value')

    for rule in rules:
        for input_id in rule_input_ids(rule):
            if input_id == rule.output:
                fail(f'rule {rule.kind!r} consumes its own output {input_id}')
            if input_id in input_port_ids or input_id in produced_set:
                continue
            if input_id in output_port_ids:
                fail(f'rule {rule.kind!r} uses output port {input_id} as an input')
            fail(f'rule {rule.kind!r} references unknown input value {input_id}')
        if rule.output in input_port_ids:
            fail(f'rule {rule.kind!r} writes to input port {rule.output}')
        for dimension_id in rule_dimension_ids(rule):
            if dimension_id not in graph.dimension_by_id:
                fail(f'rule {rule.kind!r} references unknown dimension {dimension_id}')

    # Intermediate values must resolve without cycles: repeatedly admit rules
    # whose inputs are all known, starting from the node's input ports.
    resolved: set[UUID] = set(input_port_ids)
    pending = list(rules)
    while pending:
        remaining = [rule for rule in pending if not all(value in resolved for value in rule_input_ids(rule))]
        if len(remaining) == len(pending):
            fail('rules form a cycle through intermediate values')
        resolved.update(rule.output for rule in pending if rule not in remaining)
        pending = remaining


def compile_shape_rules(graph: InstanceGraph) -> ShapeRuleCompilation:
    from nodes.instance_graph import InstanceGraphDiagnostic

    rules_by_node: dict[UUID, tuple[AnyShapeRule, ...]] = {}
    diagnostics: list[InstanceGraphDiagnostic] = []
    untrusted_node_ids: set[UUID] = set()
    for meta in graph.nodes:
        try:
            node_class = meta.node_class
        except ImportError as exc:
            diagnostics.append(
                InstanceGraphDiagnostic(
                    code='unresolved_node_class',
                    message=f'Node class {meta.node_class_path!r} cannot be imported: {exc}',
                    node_id=meta.id,
                )
            )
            rules_by_node[meta.id] = ()
            continue
        overriding_class = _computation_override_below_rules(node_class)
        if overriding_class is not None:
            rules_owner = next(klass for klass in node_class.__mro__ if 'shape_rules' in klass.__dict__)
            diagnostics.append(
                InstanceGraphDiagnostic(
                    code='inherited_shape_rules_skipped',
                    message=(
                        f'{overriding_class.__module__}.{overriding_class.__qualname__} overrides the computation of '
                        f'{rules_owner.__qualname__} without re-declaring shape_rules; the inherited rules are not trusted'
                    ),
                    node_id=meta.id,
                )
            )
            rules_by_node[meta.id] = ()
            untrusted_node_ids.add(meta.id)
            continue
        try:
            rules = tuple(node_class.shape_rules(meta))
        except MissingPortRoleError as exc:
            diagnostics.append(
                InstanceGraphDiagnostic(
                    code='missing_role_port',
                    message=f'{meta.node_class_path}: {exc}; node compiles to no shape rules',
                    node_id=meta.id,
                )
            )
            rules_by_node[meta.id] = ()
            continue
        _validate_node_rules(graph, meta, rules)
        rules_by_node[meta.id] = rules
    return ShapeRuleCompilation(
        rules_by_node=rules_by_node,
        diagnostics=tuple(diagnostics),
        untrusted_node_ids=frozenset(untrusted_node_ids),
    )
