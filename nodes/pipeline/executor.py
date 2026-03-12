from __future__ import annotations

import operator
import re
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from common import polars as ppl
from nodes.constants import VALUE_COLUMN
from nodes.exceptions import NodeError
from nodes.units import Quantity

from .ir import (
    InputDatasetBinding,
    InputNodeBinding,
    compile_pipeline_ir_to_spec,
)
from .ops.arithmetic import (
    AddOperationSpec,
    ClipOperationSpec,
    DivideOperationSpec,
    IdentityOperationSpec,
    MultiplyOperationSpec,
    SubtractOperationSpec,
)
from .ops.base import (
    ComparisonCondition,
    DatasetInputRef,
    IntermediateInputRef,
    ParameterInputRef,
    PortInputRef,
    ScalarValue,
    TruthyCondition,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nodes.node import Node

    from .ir import (
        PipelineNodeIR,
        PipelinePortBinding,
        PipelineSpec,
    )
    from .ops.base import (
        OperationCondition,
        OperationInput,
    )

PDF = ppl.PathsDataFrame
type RuntimeValue = PDF | Quantity | bool | str | None


OPERATORS = {
    'eq': operator.eq,
    'ne': operator.ne,
    'gt': operator.gt,
    'gte': operator.ge,
    'lt': operator.lt,
    'lte': operator.le,
}


class PipelineExecutor:
    """Execute pipeline operations against a live runtime node."""

    def __init__(self, node: Node, *, port_bindings: Mapping[str, PipelinePortBinding] | None = None):
        self.node = node
        self.port_bindings = dict(port_bindings or {})
        self._results: dict[str, RuntimeValue] = {}

    def execute(self, spec: PipelineSpec) -> PDF:
        last_result_key: str | None = None
        for idx, operation in enumerate(spec.operations):
            if operation.only_if is not None and not self._evaluate_condition(operation.only_if):
                continue
            if operation.skip_if is not None and self._evaluate_condition(operation.skip_if):
                continue

            result = self._execute_operation(operation)
            result_key = operation.result_id or f'_step_{idx}'
            self._results[result_key] = result
            last_result_key = result_key

        if spec.output_ref is not None:
            output = self._results.get(spec.output_ref)
            if output is None:
                raise NodeError(self.node, f"Pipeline output ref '{spec.output_ref}' was not produced")
        elif last_result_key is not None:
            output = self._results[last_result_key]
        else:
            raise NodeError(self.node, 'Pipeline produced no output')

        if not isinstance(output, PDF):
            raise NodeError(self.node, f'Pipeline output must be a PathsDataFrame, got {type(output).__name__}')
        return self._normalize_output(output)

    def _normalize_output(self, output: PDF) -> PDF:
        if VALUE_COLUMN not in output.columns:
            return output
        if len(self.node.output_metrics) != 1:
            return output
        metric = self.node.get_default_output_metric()
        return output.ensure_unit(VALUE_COLUMN, metric.unit)

    def _execute_operation(self, operation: Any) -> RuntimeValue:
        if isinstance(operation, IdentityOperationSpec):
            return self._resolve_input(operation.input)

        if isinstance(operation, AddOperationSpec):
            result = self._resolve_input(operation.input)
            for operand in operation.values:
                result = self._apply_add(result, self._resolve_input(operand))
            return result

        if isinstance(operation, SubtractOperationSpec):
            result = self._resolve_input(operation.input)
            for operand in operation.values:
                result = self._apply_sub(result, self._resolve_input(operand))
            return result

        if isinstance(operation, MultiplyOperationSpec):
            result = self._resolve_input(operation.input)
            for operand in operation.values:
                result = self._apply_mul(result, self._resolve_input(operand))
            return result

        if isinstance(operation, DivideOperationSpec):
            return self._apply_div(self._resolve_input(operation.input), self._resolve_input(operation.other))

        if isinstance(operation, ClipOperationSpec):
            return self._apply_clip(
                self._resolve_input(operation.input),
                self._resolve_input(operation.min_value) if operation.min_value is not None else None,
                self._resolve_input(operation.max_value) if operation.max_value is not None else None,
            )

        raise NodeError(self.node, f'Unsupported pipeline operation kind: {operation.kind}')

    def _evaluate_condition(self, condition: OperationCondition) -> bool:
        if isinstance(condition, TruthyCondition):
            return self._is_truthy(self._resolve_input(condition.input))

        if isinstance(condition, ComparisonCondition):
            left = self._resolve_input(condition.left)
            right = self._resolve_input(condition.right)
            return self._compare(left, right, condition.op)

        raise NodeError(self.node, f'Unsupported pipeline condition kind: {condition.kind}')

    def _is_truthy(self, value: RuntimeValue) -> bool:
        if isinstance(value, Quantity):
            return bool(value.m)
        if isinstance(value, PDF):
            raise NodeError(self.node, 'Truthy conditions are not defined for PathsDataFrame values')
        return bool(value)

    def _compare(self, left: RuntimeValue, right: RuntimeValue, op: str) -> bool:
        left_val: float
        right_val: float
        if isinstance(left, Quantity) and isinstance(right, Quantity):
            right = right.to(left.units)
            left_val = left.m
            right_val = right.m
        elif isinstance(left, PDF) or isinstance(right, PDF):
            raise NodeError(self.node, 'Comparison conditions do not yet support PathsDataFrame values')
        else:
            if type(left) is not type(right):
                raise NodeError(
                    self.node, 'Comparison conditions do not support mixed types (%s and %s)', type(left), type(right)
                )
            # FIXME: Check if op is allowed for type.
            left_val = cast('float', left)
            right_val = cast('float', right)

        op_func = OPERATORS[op]
        return bool(op_func(left_val, right_val))

    def _resolve_input(self, value: OperationInput | None) -> RuntimeValue:
        if value is None:
            return None
        if isinstance(value, IntermediateInputRef):
            result = self._results.get(value.ref)
            if result is None:
                raise NodeError(self.node, f"Intermediate ref '{value.ref}' was not produced")
            return result
        if isinstance(value, PortInputRef):
            return self._resolve_port(value.port)
        if isinstance(value, DatasetInputRef):
            return self._resolve_dataset_ref(value.dataset)
        if isinstance(value, ParameterInputRef):
            return self._resolve_parameter(value.parameter)
        if isinstance(value, ScalarValue):
            return Quantity(value.value, value.unit)
        raise NodeError(self.node, f'Unsupported pipeline input type: {type(value).__name__}')

    def _resolve_port(self, port_id: str) -> RuntimeValue:
        binding = self.port_bindings.get(port_id)
        if binding is not None:
            return self._resolve_port_binding(binding)

        tagged_node = self.node.get_input_node(tag=port_id, required=False)
        if tagged_node is not None:
            return tagged_node.get_output_pl(target_node=self.node)

        tagged_dataset = self.node.get_input_dataset_pl(tag=port_id, required=False)
        if tagged_dataset is not None:
            return tagged_dataset

        match = re.fullmatch(r'input_(\d+)', port_id)
        if match:
            index = int(match.group(1))
            if 0 <= index < len(self.node.input_nodes):
                return self.node.input_nodes[index].get_output_pl(target_node=self.node)
            raise NodeError(self.node, f"Input port '{port_id}' refers to missing input node index {index}")

        for input_node in self.node.input_nodes:
            if input_node.id == port_id:
                return input_node.get_output_pl(target_node=self.node)

        raise NodeError(self.node, f"Input port '{port_id}' is not bound for node {self.node.id}")

    def _resolve_port_binding(self, binding: PipelinePortBinding) -> RuntimeValue:
        if isinstance(binding, InputNodeBinding):
            source = self._resolve_input_node(binding)
            return source.get_output_pl(target_node=self.node, metric=binding.metric)
        if isinstance(binding, InputDatasetBinding):
            return self._resolve_input_dataset(binding)
        raise NodeError(self.node, f'Unsupported pipeline port binding: {type(binding).__name__}')

    def _resolve_input_node(self, binding: InputNodeBinding) -> Node:
        if binding.node is not None:
            for source in self.node.input_nodes:
                if source.id == binding.node:
                    return source
            raise NodeError(self.node, f"Input node '{binding.node}' not found for node {self.node.id}")
        if binding.tag is not None:
            node = self.node.get_input_node(tag=binding.tag, required=False)
            if node is None:
                raise NodeError(self.node, f"Input node tag '{binding.tag}' not found for node {self.node.id}")
            return node
        if binding.index is not None:
            try:
                return self.node.input_nodes[binding.index]
            except IndexError as exc:
                raise NodeError(self.node, f'Input node index {binding.index} out of range for node {self.node.id}') from exc
        return self.node.get_input_node()

    def _resolve_input_dataset(self, binding: InputDatasetBinding) -> PDF:
        if binding.dataset is not None:
            for dataset in self.node.input_dataset_instances:
                if dataset.id == binding.dataset:
                    return dataset.get_copy()
            raise NodeError(self.node, f"Input dataset '{binding.dataset}' not found for node {self.node.id}")
        if binding.tag is not None:
            ds = self.node.get_input_dataset_pl(tag=binding.tag, required=False)
            if ds is None:
                raise NodeError(self.node, f"Input dataset tag '{binding.tag}' not found for node {self.node.id}")
            return ds
        if binding.index is not None:
            try:
                dataset = self.node.input_dataset_instances[binding.index]
            except IndexError as exc:
                raise NodeError(self.node, f'Input dataset index {binding.index} out of range for node {self.node.id}') from exc
            return dataset.get_copy()
        ds = self.node.get_input_dataset_pl(required=False)
        if ds is None:
            raise NodeError(self.node, f'Node {self.node.id} does not have a default input dataset')
        return ds

    def _resolve_dataset_ref(self, dataset_id: str) -> PDF:
        for dataset in self.node.input_dataset_instances:
            if dataset.id == dataset_id:
                return dataset.get_copy()
        raise NodeError(self.node, f"Input dataset '{dataset_id}' not found for node {self.node.id}")

    def _resolve_parameter(self, parameter_id: str) -> RuntimeValue:
        param = self.node.get_parameter(parameter_id, required=False)
        if param is None:
            param = self.node.context.get_parameter(parameter_id, required=False)
        if param is None:
            raise NodeError(self.node, f"Parameter '{parameter_id}' not found for node {self.node.id}")
        value = param.get()
        if value is None:
            return None
        if isinstance(value, bool | str):
            return value
        return Quantity(value, param.get_unit() if param.has_unit() else None)

    def _apply_add(self, left: RuntimeValue, right: RuntimeValue) -> RuntimeValue:
        def both_df(df1: PDF, df2: PDF) -> PDF:
            return df1.paths.add_with_dims(df2, how='outer')

        def one_df(df: PDF, val: Quantity) -> PDF:
            val = val.to(df.get_unit(VALUE_COLUMN))
            return df.with_columns((pl.col(VALUE_COLUMN) + val.m).alias(VALUE_COLUMN))

        def both_quantity(val1: Quantity, val2: Quantity) -> Quantity:
            out = val1 + val2
            assert isinstance(out, Quantity)
            return out

        return self._apply_binom_commutative(left, right, both_df, one_df, both_quantity)

    def _apply_sub(self, left: RuntimeValue, right: RuntimeValue) -> RuntimeValue:
        if isinstance(right, PDF):
            right = right.multiply_quantity(VALUE_COLUMN, Quantity(-1))
        elif isinstance(right, Quantity):
            right = cast('Quantity', right * Quantity(-1))
        return self._apply_add(left, right)

    def _apply_mul(self, left: RuntimeValue, right: RuntimeValue) -> RuntimeValue:
        def both_df(df1: PDF, df2: PDF) -> PDF:
            return df1.paths.multiply_with_dims(df2, how='inner')

        def one_df(df: PDF, val: Quantity) -> PDF:
            return df.multiply_quantity(VALUE_COLUMN, val)

        def both_quantity(val1: Quantity, val2: Quantity) -> Quantity:
            out = val1 * val2
            assert isinstance(out, Quantity)
            return out

        return self._apply_binom_commutative(left, right, both_df, one_df, both_quantity)

    def _apply_div(self, left: RuntimeValue, right: RuntimeValue) -> RuntimeValue:
        def both_df(df1: PDF, df2: PDF) -> PDF:
            return df1.paths.divide_with_dims(df2, how='inner')

        def left_df(df: PDF, val: Quantity) -> PDF:
            return df.divide_by_quantity(VALUE_COLUMN, val)

        def right_df(val: Quantity, df: PDF) -> PDF:
            return df.divide_quantity(VALUE_COLUMN, val)

        def both_quantity(val1: Quantity, val2: Quantity) -> Quantity:
            out = val1 / val2
            assert isinstance(out, Quantity)
            return out

        return self._apply_binom(left, right, both_df, left_df, right_df, both_quantity)

    def _apply_clip(self, value: RuntimeValue, min_value: RuntimeValue, max_value: RuntimeValue) -> RuntimeValue:
        if isinstance(value, PDF):
            lower = self._coerce_clip_bound(min_value, unit=value.get_unit(VALUE_COLUMN), label='min')
            upper = self._coerce_clip_bound(max_value, unit=value.get_unit(VALUE_COLUMN), label='max')
            expr = pl.col(VALUE_COLUMN)
            if lower is not None:
                expr = pl.when(expr < lower).then(pl.lit(lower)).otherwise(expr)
            if upper is not None:
                expr = pl.when(expr > upper).then(pl.lit(upper)).otherwise(expr)
            return value.with_columns(expr.alias(VALUE_COLUMN))

        if isinstance(value, Quantity):
            lower_q = self._coerce_quantity_bound(min_value, template=value, label='min')
            upper_q = self._coerce_quantity_bound(max_value, template=value, label='max')
            result = value
            if lower_q is not None and result < lower_q:
                result = lower_q
            if upper_q is not None and result > upper_q:
                result = upper_q
            return result

        raise NodeError(self.node, f'Clip is not supported for {type(value).__name__}')

    def _coerce_clip_bound(self, value: RuntimeValue, *, unit: Any, label: str) -> float | None:
        if value is None:
            return None
        if not isinstance(value, Quantity):
            raise NodeError(self.node, f'Clip {label} bound must resolve to a scalar quantity')
        return float(value.to(unit).m)

    def _coerce_quantity_bound(self, value: RuntimeValue, *, template: Quantity, label: str) -> Quantity | None:
        if value is None:
            return None
        if not isinstance(value, Quantity):
            raise NodeError(self.node, f'Clip {label} bound must resolve to a scalar quantity')
        return value.to(template.units)

    def _apply_binom(
        self,
        left: RuntimeValue,
        right: RuntimeValue,
        both_df: Any,
        left_df: Any,
        right_df: Any,
        both_quantity: Any,
    ) -> RuntimeValue:
        if isinstance(left, PDF) and isinstance(right, PDF):
            return both_df(left, right)

        if isinstance(left, PDF):
            if not isinstance(right, Quantity):
                raise NodeError(self.node, f'Expected scalar quantity, got {type(right).__name__}')
            return left_df(left, right)

        if isinstance(right, PDF):
            if not isinstance(left, Quantity):
                raise NodeError(self.node, f'Expected scalar quantity, got {type(left).__name__}')
            return right_df(left, right)

        if not isinstance(left, Quantity) or not isinstance(right, Quantity):
            raise NodeError(
                self.node,
                f'Binary arithmetic requires dataframe or quantity values, got {type(left).__name__} and {type(right).__name__}',
            )

        return both_quantity(left, right)

    def _apply_binom_commutative(
        self, left: RuntimeValue, right: RuntimeValue, both_df: Any, one_df: Any, both_quantity: Any
    ) -> RuntimeValue:
        def right_df(val: Quantity, df: PDF) -> PDF:
            return one_df(df, val)

        return self._apply_binom(left, right, both_df, one_df, right_df, both_quantity)


def execute_pipeline_spec(
    node: Node,
    spec: PipelineSpec,
    *,
    port_bindings: Mapping[str, PipelinePortBinding] | None = None,
) -> PDF:
    executor = PipelineExecutor(node, port_bindings=port_bindings)
    return executor.execute(spec)


def execute_pipeline_ir(node: Node, ir: PipelineNodeIR) -> PDF:
    spec = compile_pipeline_ir_to_spec(ir)
    return execute_pipeline_spec(node, spec, port_bindings=ir.port_bindings)
