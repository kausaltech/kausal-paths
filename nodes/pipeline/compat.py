from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from nodes.node import Node

if TYPE_CHECKING:
    import common.polars as ppl
    from nodes.pipeline import PipelineSpec
    from nodes.pipeline.ir import PipelineNodeIR


class PipelineCompatibleNode(Node, ABC):
    @abstractmethod
    def lower_to_pipeline_ir(self) -> PipelineNodeIR:
        """
        Lower this runtime node into pipeline IR.

        Legacy node subclasses can override this incrementally. The default
        implementation marks the node as not yet lowerable.
        """

    def lower_to_pipeline_spec(self) -> PipelineSpec | None:
        """Compile lowered pipeline IR into the canonical pipeline spec."""

        from nodes.pipeline.ir import compile_pipeline_ir_to_spec

        ir = self.lower_to_pipeline_ir()
        return compile_pipeline_ir_to_spec(ir)

    def compute_with_lowered_pipeline(self) -> ppl.PathsDataFrame:
        """Execute this node through its lowered pipeline IR."""

        from nodes.pipeline.executor import execute_pipeline_ir

        ir = self.lower_to_pipeline_ir()
        return execute_pipeline_ir(self, ir)
