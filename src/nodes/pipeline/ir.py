from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from paths.identifiers import DatasetIdentifier, Identifier, MixedCaseIdentifier, NodeIdentifier, NodePortIdentifier

from .ops.base import ANONYMOUS_STEP_PREFIX, IntermediateInputRef, OperationSpec, step_key
from .ops.union import AnyOperationSpec


class InputNodeBinding(BaseModel):
    """
    Runtime-only binding for a canonical input port.

    This lets legacy runtime nodes lower into port-based pipelines without
    already having DB/YAML input-port wiring in place.
    """

    kind: Literal['input_node'] = 'input_node'
    node: NodeIdentifier | None = None
    tag: Identifier | None = None
    index: int | None = None
    metric: MixedCaseIdentifier | None = None

    @model_validator(mode='after')
    def _validate_selector(self) -> InputNodeBinding:
        selectors = [self.node is not None, self.tag is not None, self.index is not None]
        if sum(selectors) > 1:
            raise ValueError("InputNodeBinding accepts at most one of 'node', 'tag', or 'index'")
        if self.index is not None and self.index < 0:
            raise ValueError('Input node index must be non-negative')
        return self


class InputDatasetBinding(BaseModel):
    """Runtime-only binding from a canonical port to a node input dataset."""

    kind: Literal['input_dataset'] = 'input_dataset'
    dataset: DatasetIdentifier | None = None
    tag: Identifier | None = None
    index: int | None = None

    @model_validator(mode='after')
    def _validate_selector(self) -> InputDatasetBinding:
        selectors = [self.dataset is not None, self.tag is not None, self.index is not None]
        if sum(selectors) > 1:
            raise ValueError("InputDatasetBinding accepts at most one of 'dataset', 'tag', or 'index'")
        if self.index is not None and self.index < 0:
            raise ValueError('Input dataset index must be non-negative')
        return self


PipelinePortBinding = Annotated[InputNodeBinding | InputDatasetBinding, Field(discriminator='kind')]


class PipelineSpec(BaseModel):
    """
    Canonical pipeline specification independent of legacy runtime wiring.

    The output is the step named ``output_ref``, or else the last step. A
    formula is a text view of the same specification (``nodes.pipeline.formula``).
    """

    operations: list[AnyOperationSpec] = Field(min_length=1)
    output_ref: Identifier | None = None
    description: str | None = None
    """The modeller's explanation of the whole computation."""

    @model_validator(mode='after')
    def _validate_steps(self) -> PipelineSpec:  # noqa: C901, PLR0912
        """
        Keep every pipeline expressible as a formula.

        An anonymous step is an inline sub-expression: it is used exactly once,
        by a later step, and carries no description (a comment needs a line of
        its own). Only the output may be anonymous and unused.
        """
        names: set[str] = set()
        for operation in self.operations:
            name = operation.result_id
            if name is None:
                continue
            if name.startswith(ANONYMOUS_STEP_PREFIX):
                raise ValueError(f'Step name {name!r} is reserved for anonymous steps')
            if name in names:
                raise ValueError(f'Step name {name!r} is used more than once')
            names.add(name)
        if self.output_ref is not None and self.output_ref not in names:
            raise ValueError(f'Pipeline output {self.output_ref!r} names no step')

        known: set[str] = set()
        uses: dict[str, int] = {}
        for index, operation in enumerate(self.operations):
            for ref in operation_refs(operation):
                if ref not in known:
                    raise ValueError(f'Step {index} refers to {ref!r}, which no earlier step produces')
                uses[ref] = uses.get(ref, 0) + 1
            known.add(step_key(operation, index))

        last = len(self.operations) - 1
        for index, operation in enumerate(self.operations):
            if operation.result_id is not None:
                continue
            key = step_key(operation, index)
            is_output = self.output_ref is None and index == last
            if is_output:
                continue
            if uses.get(key, 0) != 1:
                raise ValueError(f'Anonymous step {index} must be used exactly once; name the step')
            if operation.description is not None:
                raise ValueError(f'Anonymous step {index} cannot have a description; name the step')
        return self


def operation_refs(operation: OperationSpec) -> list[str]:
    """Return the intermediate results ``operation`` reads, conditions included, in order."""
    refs: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, IntermediateInputRef):
            refs.append(value.ref)
        elif isinstance(value, BaseModel):
            for field in type(value).model_fields:
                visit(getattr(value, field))
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for field in type(operation).model_fields:
        visit(getattr(operation, field))
    return refs


class PipelineNodeIR(PipelineSpec):
    """
    Lowered runtime representation for legacy nodes.

    The operations are already canonical pipeline operations. The extra
    information here exists only to bind those ports back to the live legacy
    node so we can execute and compare before persisting anything.
    """

    node_id: NodeIdentifier | None = None
    source_node_class: str | None = None
    port_bindings: dict[NodePortIdentifier, PipelinePortBinding] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def compile_pipeline_ir_to_spec(ir: PipelineNodeIR) -> PipelineSpec:
    """Drop runtime-only bindings and return the canonical pipeline specification."""

    return PipelineSpec(
        operations=ir.operations,
        output_ref=ir.output_ref,
        description=ir.description,
    )
