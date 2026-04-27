from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from paths.identifiers import DatasetIdentifier, Identifier, MixedCaseIdentifier, NodeIdentifier, NodePortIdentifier

from .ops.arithmetic import AnyOperationSpec


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
    """Canonical pipeline specification independent of legacy runtime wiring."""

    operations: list[AnyOperationSpec] = Field(default_factory=list, min_length=1)
    output_ref: Identifier | None = None
    description: str | None = None


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
