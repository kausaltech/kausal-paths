"""
Declarative port shape rules.

A rule relates node-local value identities (UUIDs) to each other: which
inputs must share a shape, which multiply together, and which transform
dimensions between one input and one output. Node classes declare rules
through ``Node.shape_rules(meta)``; pipeline operations compile to the same
union. The solver (step 7 of the instance-graph plan) consumes compiled
rules; nothing here reads dataframes or runtime state.

Rules are immutable values containing UUIDs only. Role selectors like
``'factors'`` are resolved against a specific node *before* a rule is
constructed — see ``NodeMeta.require_input_port()`` and friends.

A rule's ``inputs``/``input`` and ``output`` normally reference the node's
own port UUIDs. Compiled pipelines may additionally chain rules through
intermediate value UUIDs: an intermediate is valid as a rule input exactly
when it is produced as another rule's output on the same node. Compilation
validates that closure (see ``nodes/constraints/compile.py``).
"""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShapeRuleError(Exception):
    """A structurally invalid shape rule: a bug in a node class or compiler."""


class MissingPortRoleError(ValueError):
    """
    A role required by a shape rule has no port on this particular node.

    Distinct from ``ShapeRuleError``: this is a property of one node's data
    (e.g. a legacy spec whose ports could not be classified), not a class
    bug. Compilation records it as a graph diagnostic and skips the node.
    """

    def __init__(self, node_id: UUID, direction: Literal['input', 'output'], role: str) -> None:
        self.node_id = node_id
        self.direction = direction
        self.role = role
        super().__init__(f'Node {node_id} has no {direction} port for role {role!r}')


class ShapeRuleBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')


class SameShapeRule(ShapeRuleBase):
    """
    Every input, and the output, carries the same shape.

    Dimensions are equal, units are convertible, and quantities are equal.
    For a multi port this constrains every delivered binding value, the port
    aggregate, and the output alike.
    """

    kind: Literal['same'] = 'same'
    inputs: tuple[UUID, ...] = Field(min_length=1)
    output: UUID


class ProductShapeRule(ShapeRuleBase):
    """
    The output is the product of the inputs over the product of the inverse inputs.

    Output dimensions are the union over *all* operands — division does not
    remove a dimension — while the output unit is the product of the
    ``inputs`` units divided by the product of the ``inverse_inputs`` units.
    Products happen across distinct ports only — bindings on one (multi) port
    are always a homogeneous ``same``-shaped aggregate.
    """

    kind: Literal['product'] = 'product'
    inputs: tuple[UUID, ...] = ()
    inverse_inputs: tuple[UUID, ...] = ()
    output: UUID

    @model_validator(mode='after')
    def _validate_operands(self) -> ProductShapeRule:
        if not self.inputs and not self.inverse_inputs:
            raise ValueError('ProductShapeRule: at least one operand is required')
        return self


class DimensionTransformRule(ShapeRuleBase):
    """
    One input transformed into one output with declared dimension effects.

    ``requires`` must be present on the input; ``consumes`` (a subset of
    ``requires``) is removed from the output; ``produces`` is added to it.
    A ``transparent`` rule passes all other dimensions through; a
    non-transparent rule additionally caps the output at exactly
    ``(input ∩ requires) - consumes + produces``.
    """

    kind: Literal['dimension_transform'] = 'dimension_transform'
    input: UUID
    output: UUID
    requires: frozenset[UUID] = frozenset()
    consumes: frozenset[UUID] = frozenset()
    produces: frozenset[UUID] = frozenset()
    transparent: bool = True

    @model_validator(mode='after')
    def _validate_dimension_sets(self) -> DimensionTransformRule:
        if not self.consumes.issubset(self.requires):
            raise ValueError('DimensionTransformRule: consumes must be a subset of requires')
        if self.requires & self.produces:
            raise ValueError('DimensionTransformRule: requires and produces must be disjoint')
        return self


type AnyShapeRule = SameShapeRule | ProductShapeRule | DimensionTransformRule
"""The plain rule union, for annotations and isinstance dispatch."""

type PortShapeRule = Annotated[
    SameShapeRule | ProductShapeRule | DimensionTransformRule,
    Field(discriminator='kind'),
]
"""The discriminated union, for Pydantic fields."""


def rule_input_ids(rule: AnyShapeRule) -> tuple[UUID, ...]:
    if isinstance(rule, DimensionTransformRule):
        return (rule.input,)
    if isinstance(rule, ProductShapeRule):
        return (*rule.inputs, *rule.inverse_inputs)
    return rule.inputs


def rule_dimension_ids(rule: AnyShapeRule) -> frozenset[UUID]:
    if isinstance(rule, DimensionTransformRule):
        return rule.requires | rule.consumes | rule.produces
    return frozenset()
