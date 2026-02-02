from __future__ import annotations

import ast
from typing import Any, Callable, Literal, NamedTuple, TypeVar

from django.utils.translation import gettext_lazy as _

import polars as pl

from common import polars as ppl
from nodes.calc import convert_to_co2e, extend_last_historical_value_pl
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN
from nodes.exceptions import NodeError
from nodes.units import Quantity, QuantityType
from params.param import BoolParameter, NumberParameter, StringParameter

from .node import Node

PDF = ppl.PathsDataFrame
type EvalConst = float
type EvalOutput = PDF | QuantityType


class EvalVars(NamedTuple):
    nodes: dict[str, Node]
    datasets: dict[str, PDF]
    parameters: dict[str, Quantity]


ASTType = TypeVar('ASTType', bound=ast.expr)

type BinomBothDF = Callable[[PDF, PDF], PDF]
type BinomLeftDF = Callable[[PDF, Quantity], PDF]
type BinomBothQuantity = Callable[[Quantity, Quantity], Quantity]
type BinomRightDF = Callable[[Quantity, PDF], PDF]


class FormulaNode(Node):
    explanation = _('This is a Formula Node. It uses a specified formula to calculate the output.')
    allowed_parameters = [
        StringParameter(local_id='formula'),
        BoolParameter(local_id='extend_last_historical_value'),
        NumberParameter(local_id='constant', label='Constant value to add to the formula', is_customizable=True)
    ]

    # Use varss instead of vars for variables to avoid shadowing.
    def eval_expression(self, expr: ast.Expression, varss: EvalVars) -> EvalOutput:
        return self.eval_tree(expr.body, varss)

    def eval_constant(self, node: ast.Constant, _varss: EvalVars) -> Quantity:
        q = Quantity(node.value)
        return q  # pyright: ignore[reportReturnType]

    def eval_name(self, name: ast.Name, varss: EvalVars) -> EvalOutput:
        if name.id in varss.nodes:
            node = varss.nodes[name.id]
            return node.get_output_pl(target_node=self)
        if name.id in varss.datasets:
            df = varss.datasets[name.id]
            if FORECAST_COLUMN not in df.columns:
                is_forecast = False
                df = df.with_columns(pl.lit(is_forecast).alias(FORECAST_COLUMN))
            assert len(df.metric_cols) == 1
            return df.rename({df.metric_cols[0]: VALUE_COLUMN})
        return varss.parameters[name.id]

    def apply_binom(
        self, left: EvalOutput, right: EvalOutput, both_df: BinomBothDF, left_df: BinomLeftDF,
        right_df: BinomRightDF, both_quantity: BinomBothQuantity
    ) -> EvalOutput:
        if isinstance(left, PDF) and isinstance(right, PDF):
            return both_df(left, right)

        if isinstance(left, PDF):
            assert isinstance(right, Quantity)
            return left_df(left, right)

        if isinstance(right, PDF):
            assert isinstance(left, Quantity)
            return right_df(left, right)

        assert isinstance(left, Quantity)
        assert isinstance(right, Quantity)
        return both_quantity(right, left)

    def apply_binom_commutative(
            self, left: EvalOutput, right: EvalOutput, both_df: BinomBothDF, one_df: BinomLeftDF,
            both_quantity: BinomBothQuantity
    ) -> EvalOutput:
        def right_df(val: Quantity, df: PDF) -> PDF:
            return one_df(df, val)
        return self.apply_binom(left, right, both_df, one_df, right_df, both_quantity)

    def apply_add(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        """Add two values using add_with_dims for DataFrames, which handles dimensions properly."""
        def both_df(df1: PDF, df2: PDF) -> PDF:
            # add_with_dims requires matching dimensions and handles unit conversion
            return df1.paths.add_with_dims(df2, how='outer')
        def one_df(df: PDF, val: QuantityType) -> PDF:
            val = val.to(df.get_unit(VALUE_COLUMN))
            df = df.with_columns(pl.col(VALUE_COLUMN) + val)
            return df
        def both_quantity(val1: Quantity, val2: Quantity) -> Quantity:
            out = val1 + val2
            assert isinstance(out, Quantity)
            return out
        return self.apply_binom_commutative(left, right, both_df, one_df, both_quantity)

    def apply_mul(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        def both_df(df1: PDF, df2: PDF) -> PDF:
            return df1.paths.multiply_with_dims(df2, how='inner')
        def one_df(df: PDF, val: Quantity) -> PDF:
            return df.multiply_quantity(VALUE_COLUMN, val)
        def both_quantity(val1: Quantity, val2: Quantity) -> Quantity:
            out = val1 * val2
            assert isinstance(out, Quantity)
            return out
        return self.apply_binom_commutative(left, right, both_df, one_df, both_quantity)

    def apply_div(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        def both_df(df1: PDF, df2: PDF) -> PDF:
            r = df2.copy().rename({VALUE_COLUMN: '_Right'})
            df = df1.paths.join_over_index(r)
            df = df.divide_cols([VALUE_COLUMN, '_Right'], VALUE_COLUMN).drop('_Right')
            return df
        def left_df(df: PDF, val: QuantityType) -> PDF:
            df_unit = df.get_unit(VALUE_COLUMN)
            df = df.with_columns(pl.col(VALUE_COLUMN) / pl.lit(val.m))
            df = df.set_unit(VALUE_COLUMN, df_unit / val.units, force=True)  # pyright: ignore[reportArgumentType]
            return df
        def right_df(val: QuantityType, df: PDF) -> PDF:
            df_unit = df.get_unit(VALUE_COLUMN)
            df = df.with_columns(val / pl.col(VALUE_COLUMN))
            df = df.set_unit(VALUE_COLUMN, val.units / df_unit)  # pyright: ignore[reportArgumentType]
            return df
        def both_quantity(val1: Quantity, val2: Quantity) -> Quantity:
            out = val1 / val2
            assert isinstance(out, Quantity)
            return out
        return self.apply_binom(left, right, both_df, left_df, right_df, both_quantity)

    def _compare_pdf_quantity(self, df: PDF, val: Quantity, op: str) -> PDF:
        """Compare a PDF with a Quantity using the specified operation."""
        converted_val = val.to(df.get_unit(VALUE_COLUMN))
        val_m: float = converted_val.m
        op_map = {
            'eq': pl.col(VALUE_COLUMN).eq(pl.lit(val_m)),
            'ne': pl.col(VALUE_COLUMN).ne(pl.lit(val_m)),
            'gt': pl.col(VALUE_COLUMN).gt(pl.lit(val_m)),
            'ge': pl.col(VALUE_COLUMN).ge(pl.lit(val_m)),
            'lt': pl.col(VALUE_COLUMN).lt(pl.lit(val_m)),
            'le': pl.col(VALUE_COLUMN).le(pl.lit(val_m)),
        }
        df = df.with_columns(op_map[op].cast(pl.Float64).alias(VALUE_COLUMN))
        return df.set_unit(VALUE_COLUMN, 'dimensionless', force=True)

    def _apply_compare(
        self, left: EvalOutput, right: EvalOutput, op: Literal['eq', 'ne', 'gt', 'ge', 'lt', 'le']
    ) -> EvalOutput:
        """Apply comparison operation, handling PDF vs PDF, PDF vs Quantity, and Quantity vs PDF."""
        # Map for reversing operations when Quantity is on the left
        reverse_op_map: dict[Literal['eq', 'ne', 'gt', 'ge', 'lt', 'le'], Literal['eq', 'ne', 'gt', 'ge', 'lt', 'le']] = {
            'eq': 'eq',
            'ne': 'ne',
            'gt': 'lt',
            'ge': 'le',
            'lt': 'gt',
            'le': 'ge',
        }

        if isinstance(left, PDF) and isinstance(right, PDF):
            return left.paths.compare_df(right, op=op)
        if isinstance(left, PDF) and isinstance(right, Quantity):
            return self._compare_pdf_quantity(left, right, op)
        if isinstance(left, Quantity) and isinstance(right, PDF):
            return self._compare_pdf_quantity(right, left, reverse_op_map[op])
        raise NotImplementedError("Comparisons require at least one PathsDataFrame")

    def apply_eq(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        """Equal comparison (==)."""
        return self._apply_compare(left, right, 'eq')

    def apply_ne(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        """Not equal comparison (!=)."""
        return self._apply_compare(left, right, 'ne')

    def apply_lt(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        """Less than comparison (<)."""
        return self._apply_compare(left, right, 'lt')

    def apply_le(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        """Less than or equal comparison (<=)."""
        return self._apply_compare(left, right, 'le')

    def apply_gt(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        """Greater than comparison (>)."""
        return self._apply_compare(left, right, 'gt')

    def apply_ge(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        """Greater than or equal comparison (>=)."""
        return self._apply_compare(left, right, 'ge')

    def eval_binop(self, node: ast.BinOp, varss: EvalVars) -> EvalOutput:
        OPERATIONS: dict[type, Callable[[EvalOutput, EvalOutput], EvalOutput]] = {
            ast.Add: self.apply_add,
            #ast.Sub: operator.sub,
            ast.Mult: self.apply_mul,
            ast.Div: self.apply_div,
        }

        left_value = self.eval_tree(node.left, varss)
        right_value = self.eval_tree(node.right, varss)
        apply = OPERATIONS[type(node.op)]
        return apply(left_value, right_value)

    def eval_compare(self, node: ast.Compare, varss: EvalVars) -> EvalOutput:
        OPERATIONS: dict[type, Callable[[EvalOutput, EvalOutput], EvalOutput]] = {
            ast.Eq: self.apply_eq,
            ast.NotEq: self.apply_ne,
            ast.Lt: self.apply_lt,
            ast.LtE: self.apply_le,
            ast.Gt: self.apply_gt,
            ast.GtE: self.apply_ge,
        }

        # For now, handle single comparisons (not chained like a < b < c)
        if len(node.ops) != 1:
            raise NotImplementedError("Chained comparisons are not yet supported")
        if len(node.comparators) != 1:
            raise NotImplementedError("Multiple comparators are not yet supported")

        left_value = self.eval_tree(node.left, varss)
        right_value = self.eval_tree(node.comparators[0], varss)
        apply = OPERATIONS[type(node.ops[0])]
        return apply(left_value, right_value)

    def eval_call(self, node: ast.Call, varss: EvalVars) -> EvalOutput:
        func_name = node.func
        assert isinstance(func_name, ast.Name)
        func = func_name.id

        # Evaluate first argument
        assert len(node.args) >= 1, f"Function {func} requires at least one argument"
        df = self.eval_tree(node.args[0], varss)  # Get the first result

        # Try PathsExt operations first
        if isinstance(df, PDF) and func in df.paths.OPERATIONS:
            operation = df.paths.OPERATIONS[func]
            return operation(df, self.context)

        # Handle non-PathsExt functions
        return self._handle_custom_function(func, node, varss, df)

    def _handle_custom_function(
        self, func: str, node: ast.Call, _varss: EvalVars, df: EvalOutput
    ) -> EvalOutput:
        """Handle custom functions not in PathsExt.OPERATIONS."""

        if func == 'convert_gwp':
            assert isinstance(df, PDF)
            return convert_to_co2e(df, 'greenhouse_gases')

        if func == 'sum_dim':
            assert len(node.args) == 2
            assert isinstance(df, PDF)
            dim_arg = node.args[1]
            assert isinstance(dim_arg, ast.Name)
            assert isinstance(dim_arg.id, str)
            return df.paths.sum_over_dims(dim_arg.id)

        if func == 'zero_fill':
            assert isinstance(df, PDF)
            df = df.paths.to_wide()
            meta = df.get_meta()
            zdf = df.fill_null(0)
            return ppl.to_ppdf(zdf, meta=meta).paths.to_narrow()

        raise NotImplementedError(f"Unknown function: {func}")

    def eval_tree(self, tree: ast.AST, varss: EvalVars) -> EvalOutput:
        EVALUATORS: dict[type, Callable[[Any, EvalVars], EvalOutput]] = {
            ast.Expression: self.eval_expression,
            ast.Constant: self.eval_constant,
            ast.Name: self.eval_name,
            ast.BinOp: self.eval_binop,
            ast.Compare: self.eval_compare,
            ast.Call: self.eval_call,
            #ast.UnaryOp: self.eval_unaryop,
        }

        for ast_type, evaluator in EVALUATORS.items():
            if isinstance(tree, ast_type):
                return evaluator(tree, varss)

        raise KeyError(tree)

    def _collect_used_node_names(self, tree: ast.AST, varss: EvalVars) -> set[str]:
        """Collect node names that are actually used in the formula AST."""
        used_names: set[str] = set()

        def visit(node: ast.AST) -> None:
            if isinstance(node, ast.Name) and node.id in varss.nodes:
                used_names.add(node.id)
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)
        return used_names

    def evaluate_formula(self, formula: str, varss: EvalVars) -> PDF:
        tree = ast.parse(formula, "<string>", mode="eval")
        ret = self.eval_tree(tree, varss)
        assert isinstance(ret, PDF)
        return ret

    def _collect_eval_vars(self) -> EvalVars:  # noqa: C901
        nodes = {}
        for edge in self.edges:
            if edge.output_node != self:
                continue
            n = edge.input_node
            nodes[n.id] = n
            for tag in edge.tags:
                nodes[tag] = n

        datasets = {}
        for ds in self.input_dataset_instances:
            tags = [tag for tag in ds.tags if tag != 'cleaned']
            has_cleaned = 'cleaned' in ds.tags

            for tag in tags:
                if has_cleaned:
                    df = self.get_cleaned_dataset(tag=tag)
                else:
                    df = self.get_input_dataset_pl(tag=tag)
                if df is not None:
                    datasets[tag] = df

        # Collect parameters that have units (for use in formulas)
        from params.param import ParameterWithUnit
        parameters: dict[str, Quantity] = {}
        for param_id, param in self.parameters.items():
            if isinstance(param, ParameterWithUnit) and param.unit is not None:
                val = self.get_parameter_value(param_id, required=False, units=True)
                if val is not None:
                    assert isinstance(val, Quantity)
                    parameters[param_id] = val

        return EvalVars(nodes, datasets, parameters)

    def compute(self) -> PDF:
        varss = self._collect_eval_vars()
        formula = self.get_parameter_value_str('formula')
        tree = ast.parse(formula, "<string>", mode="eval")
        used_node_names = self._collect_used_node_names(tree, varss)

        df = self.eval_tree(tree, varss)
        assert isinstance(df, PDF)
        df = df.ensure_unit(VALUE_COLUMN, self.get_default_output_metric().unit)
        extend = self.get_parameter_value('extend_last_historical_value', required=False)
        if extend:
            df = extend_last_historical_value_pl(df, self.get_end_year())

        # Find unused nodes: computational nodes in varss.nodes that weren't referenced in the formula
        all_nodes = set(varss.nodes.values())
        used_nodes = {varss.nodes[name] for name in used_node_names if name in varss.nodes}
        unused_nodes = list(all_nodes - used_nodes)
        for edge in self.edges:
            node = edge.input_node
            if 'ignore_content' in edge.tags or 'ignore_content' in node.tags or node.quantity == 'argument':
                unused_nodes.remove(node)

        if unused_nodes:
            try:
                df = self.add_nodes_pl(df, unused_nodes)
            except NodeError as e:
                err = _('Input nodes that are not used in the formula are used for addition in the end. Error:')
                raise NodeError(self, f"{err} {e}") from e
        return df
