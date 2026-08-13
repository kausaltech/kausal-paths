"""
Compile canonical pipeline operations to port shape rules.

A pipeline is the node's own computation, so its shape semantics compile to
the same rule union that node classes declare: ``add``/``subtract`` are
``same``-shape, ``multiply``/``divide`` are ``product``, ``identity``/``clip``
pass one shape through. Rules chain through intermediate value UUIDs derived
deterministically from the node UUID and the operation's ``result_id``; the
operation that produces the pipeline output writes directly to the output
port UUID.

Scalar and parameter operands are shape-neutral in v1: a scalar does not pin
the other operand's dimensions, so it simply drops out of the compiled rule
(its unit contribution is a solver concern, revisited in step 7). Dataset
references are not compilable yet and suppress the operation's rule with a
note rather than compiling something wrong.

Nothing consumes this in production yet: authored ``PipelineConfig`` is still
a stub, and lowered ``PipelineNodeIR`` exists only for compare tooling. The
compiler is exercised through tests until one of those grows real operations.
"""

from typing import TYPE_CHECKING
from uuid import UUID, uuid5

from nodes.constraints.rules import AnyShapeRule, ProductShapeRule, SameShapeRule, ShapeRuleError
from nodes.pipeline.ops.arithmetic import (
    AddOperationSpec,
    AnyOperationSpec,
    ClipOperationSpec,
    DivideOperationSpec,
    IdentityOperationSpec,
    MultiplyOperationSpec,
    SubtractOperationSpec,
)
from nodes.pipeline.ops.base import (
    DatasetInputRef,
    IntermediateInputRef,
    OperationInput,
    ParameterInputRef,
    PortInputRef,
    ScalarValue,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nodes.defs.node_defs import NodeSpec


def _intermediate_value_id(node_uuid: UUID, result_key: str) -> UUID:
    return uuid5(node_uuid, f'pipeline-intermediate:{result_key}')


class _UnsupportedInputError(Exception):
    def __init__(self, note: str) -> None:
        self.note = note
        super().__init__(note)


def compile_pipeline_operations(  # noqa: C901, PLR0912, PLR0915
    *,
    node_uuid: UUID,
    spec: NodeSpec,
    operations: Sequence[AnyOperationSpec],
    output_port_id: UUID,
    output_ref: str | None = None,
) -> tuple[tuple[AnyShapeRule, ...], tuple[str, ...]]:
    """
    Compile one pipeline's operations to shape rules.

    Returns the compiled rules and human-readable notes for operations that
    could not contribute a rule. Referencing an input port identifier the
    node does not have is a structural error and raises ``ShapeRuleError``.
    """
    known_results: dict[str, UUID] = {}
    if output_ref is not None:
        output_index = next(
            (index for index, op in enumerate(operations) if op.result_id == output_ref),
            None,
        )
        if output_index is None:
            raise ShapeRuleError(f'Pipeline output_ref {output_ref!r} matches no operation result on node {node_uuid}')
    else:
        output_index = len(operations) - 1

    def resolve_input(value: OperationInput) -> UUID | None:
        """Return the value's UUID, or None when it is shape-neutral."""
        match value:
            case PortInputRef():
                if value.port not in spec.input_port_by_id:
                    raise ShapeRuleError(f'Pipeline on node {node_uuid} references unknown input port {value.port!r}')
                return value.port
            case IntermediateInputRef():
                result = known_results.get(value.ref)
                if result is None:
                    raise ShapeRuleError(f'Pipeline on node {node_uuid} references unknown intermediate {value.ref!r}')
                return result
            case ScalarValue() | ParameterInputRef():
                return None
            case DatasetInputRef():
                raise _UnsupportedInputError(f'dataset reference {value.dataset!r} is not compilable to shape rules yet')
        raise ShapeRuleError(f'Pipeline on node {node_uuid} has unsupported input type {type(value).__name__}')

    rules: list[AnyShapeRule] = []
    notes: list[str] = []
    for index, op in enumerate(operations):
        result_key = op.result_id or f'#{index}'
        result_id = output_port_id if index == output_index else _intermediate_value_id(node_uuid, result_key)

        operands: list[OperationInput]
        inverse_operands: list[OperationInput] = []
        match op:
            case AddOperationSpec() | SubtractOperationSpec() | MultiplyOperationSpec():
                operands = [op.input, *op.values]
            case DivideOperationSpec():
                operands = [op.input]
                inverse_operands = [op.other]
            case IdentityOperationSpec() | ClipOperationSpec():
                operands = [op.input]
            case _:
                notes.append(f'operation {op.kind!r} ({result_key}) has no shape-rule compilation')
                if op.result_id is not None:
                    known_results[op.result_id] = result_id
                continue

        try:
            inputs = tuple(value for value in (resolve_input(operand) for operand in operands) if value is not None)
            inverse_inputs = tuple(
                value for value in (resolve_input(operand) for operand in inverse_operands) if value is not None
            )
        except _UnsupportedInputError as exc:
            notes.append(f'operation {op.kind!r} ({result_key}): {exc.note}')
            if op.result_id is not None:
                known_results[op.result_id] = result_id
            continue

        if not inputs and not inverse_inputs:
            notes.append(f'operation {op.kind!r} ({result_key}) has only shape-neutral operands')
        elif isinstance(op, (MultiplyOperationSpec, DivideOperationSpec)):
            rules.append(ProductShapeRule(inputs=inputs, inverse_inputs=inverse_inputs, output=result_id))
        else:
            rules.append(SameShapeRule(inputs=inputs, output=result_id))

        if op.result_id is not None:
            known_results[op.result_id] = result_id

    return tuple(rules), tuple(notes)
