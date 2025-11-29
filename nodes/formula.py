from __future__ import annotations

import ast
from typing import Any, Callable, NamedTuple, TypeAlias, TypeVar

from django.utils.translation import gettext_lazy as _

import polars as pl

from common import polars as ppl
from nodes.calc import convert_to_co2e, extend_last_historical_value_pl
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN
from nodes.units import Quantity, QuantityType
from params.param import BoolParameter, StringParameter

from .node import Node

PDF: TypeAlias = ppl.PathsDataFrame
EvalConst: TypeAlias = float
EvalOutput: TypeAlias = PDF | QuantityType  # noqa: UP040


class EvalVars(NamedTuple):
    nodes: dict[str, Node]
    datasets: dict[str, PDF]


ASTType = TypeVar('ASTType', bound=ast.expr)

BinomBothDF: TypeAlias = Callable[[PDF, PDF], PDF]
BinomLeftDF: TypeAlias = Callable[[PDF, Quantity], PDF]
BinomBothQuantity: TypeAlias = Callable[[Quantity, Quantity], Quantity]
BinomRightDF: TypeAlias = Callable[[Quantity, PDF], PDF]


class FormulaNode(Node):  # FIXME The formula is not commutative, i.e. a * b != b * a with some dimensions
    explanation = _('This is a Formula Node. It uses a specified formula to calculate the output.')
    allowed_parameters = [
        StringParameter(local_id='formula'),
        BoolParameter(local_id='extend_last_historical_value')
    ]

    # Use varss instead of vars for variables to avoid shadowing.
    def eval_expression(self, expr: ast.Expression, varss: EvalVars) -> EvalOutput:
        return self.eval_tree(expr.body, varss)

    def eval_constant(self, node: ast.Constant, _varss: EvalVars) -> Quantity:
        q = Quantity(node.value)
        return q  # pyright: ignore[reportReturnType]

    def eval_name(self, name: ast.Name, varss: EvalVars) -> PDF:
        if name.id in varss.nodes:
            node = varss.nodes[name.id]
            df = node.get_output_pl(target_node=self)
        else:
            df = varss.datasets[name.id]
            if FORECAST_COLUMN not in df.columns:
                df = df.with_columns(pl.lit(False).alias(FORECAST_COLUMN))
            assert len(df.metric_cols) == 1
            df = df.rename({df.metric_cols[0]: VALUE_COLUMN})
        return df

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

        return both_quantity(right, left)

    def apply_binom_commutative(
            self, left: EvalOutput, right: EvalOutput, both_df: BinomBothDF, one_df: BinomLeftDF,
            both_quantity: BinomBothQuantity
    ) -> EvalOutput:
        def right_df(val: Quantity, df: PDF):
            return one_df(df, val)
        return self.apply_binom(left, right, both_df, one_df, right_df, both_quantity)

    def apply_add(self, left: EvalOutput, right: EvalOutput) -> EvalOutput: # FIXME Refactor to use add_over_dims() etc.
        def both_df(df1: PDF, df2: PDF):
            assert set(df1.dim_ids) == set(df2.dim_ids)
            r = df2.copy().rename({VALUE_COLUMN: '_Right'}).ensure_unit('_Right', df1.get_unit(VALUE_COLUMN))
            df = df1.paths.join_over_index(r, how='outer')
            df = df.with_columns(pl.col(VALUE_COLUMN).fill_null(0) + pl.col('_Right').fill_null(0)).drop('_Right')
            return df
        def one_df(df: PDF, val: QuantityType) -> PDF:
            val = val.to(df.get_unit(VALUE_COLUMN))
            df = df.with_columns(pl.col(VALUE_COLUMN) + val)
            return df
        def both_quantity(val1: Quantity, val2: Quantity):
            out = left + right
            assert isinstance(out, Quantity)
            return out
        return self.apply_binom_commutative(left, right, both_df, one_df, both_quantity)

    def apply_mul(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        def both_df(df1: PDF, df2: PDF):
            r = df2.copy().rename({VALUE_COLUMN: '_Right'})
            df = df1.paths.join_over_index(r, index_from='union')
            df = df.multiply_cols([VALUE_COLUMN, '_Right'], VALUE_COLUMN).drop('_Right')
            return df
        def one_df(df: PDF, val: Quantity):
            df_unit = df.get_unit(VALUE_COLUMN)
            df = df.with_columns(pl.col(VALUE_COLUMN) * val)
            df = df.set_unit(VALUE_COLUMN, df_unit * val.units)
            return df
        def both_quantity(val1: Quantity, val2: Quantity):
            out = val1 + val2
            assert isinstance(out, Quantity)
            return out
        return self.apply_binom_commutative(left, right, both_df, one_df, both_quantity)

    def apply_div(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        def both_df(df1: PDF, df2: PDF):
            r = df2.copy().rename({VALUE_COLUMN: '_Right'})
            df = df1.paths.join_over_index(r)
            df = df.divide_cols([VALUE_COLUMN, '_Right'], VALUE_COLUMN).drop('_Right')
            return df
        def left_df(df: PDF, val: QuantityType) -> PDF:
            df_unit = df.get_unit(VALUE_COLUMN)
            df = df.with_columns(pl.col(VALUE_COLUMN) / val.m)
            df = df.set_unit(VALUE_COLUMN, df_unit / val.units, force=True)
            return df
        def right_df(val: QuantityType, df: PDF) -> PDF:
            df_unit = df.get_unit(VALUE_COLUMN)
            df = df.with_columns(val / pl.col(VALUE_COLUMN))
            df = df.set_unit(VALUE_COLUMN, val.units / df_unit)
            return df
        def both_quantity(val1: Quantity, val2: Quantity):
            out = val1 / val2
            assert isinstance(out, Quantity)
            return out
        return self.apply_binom(left, right, both_df, left_df, right_df, both_quantity)

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

    def eval_call(self, node: ast.Call, varss: EvalVars) -> EvalOutput:
        func_name = node.func
        assert isinstance(func_name, ast.Name)
        func = func_name.id
        if func == 'convert_gwp':
            assert len(node.args) == 1
            df = self.eval_tree(node.args[0], varss)
            assert isinstance(df, PDF)
            df = convert_to_co2e(df, 'greenhouse_gases')
        elif func == 'sum_dim':
            assert len(node.args) == 2
            df = self.eval_tree(node.args[0], varss)
            assert isinstance(df, PDF)
            dim_arg = node.args[1]
            assert isinstance(dim_arg, ast.Constant)
            assert isinstance(dim_arg.value, str)
            df = df.paths.sum_over_dims(dim_arg.value)
        elif func == 'zero_fill':
            assert len(node.args) == 1
            df = self.eval_tree(node.args[0], varss)
            assert isinstance(df, PDF)
            df = df.paths.to_wide()
            meta = df.get_meta()
            zdf = df.fill_null(0)
            df = ppl.to_ppdf(zdf, meta=meta).paths.to_narrow()
        elif func == 'complement': # TODO Example function. Generalize.
            assert len(node.args) == 1
            arg = node.args[0]
            assert isinstance(arg, ast.Name), "complement() requires a direct node reference"
            assert arg.id in varss.nodes, f"complement() argument must be a node, got {arg.id}"
            source_node = varss.nodes[arg.id]

            # Evaluate to get the dataframe
            # This calls eval_name() which does: source_node.get_output_pl(target_node=self)
            df = self.eval_tree(arg, varss)
            assert isinstance(df, PDF)

            # Call complement with both the df and the source node
            df = df.paths._complement(df, source_node)
        else:
            raise NotImplementedError("Unknown function: %s" % func)
        return df

    def eval_tree(self, tree: ast.AST, varss: EvalVars) -> EvalOutput:
        EVALUATORS: dict[type, Callable[[Any, EvalVars], EvalOutput]] = {
            ast.Expression: self.eval_expression,
            ast.Constant: self.eval_constant,
            ast.Name: self.eval_name,
            ast.BinOp: self.eval_binop,
            ast.Call: self.eval_call,
            #ast.UnaryOp: self.eval_unaryop,
        }

        for ast_type, evaluator in EVALUATORS.items():
            if isinstance(tree, ast_type):
                return evaluator(tree, varss)

        raise KeyError(tree)

    def evaluate_formula(self, formula: str, varss: EvalVars) -> PDF:
        tree = ast.parse(formula, "<string>", mode="eval")
        ret = self.eval_tree(tree, varss)
        assert isinstance(ret, PDF)
        return ret

    def compute(self) -> PDF:
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
            for tag in ds.tags:
                datasets[tag] = self.get_input_dataset_pl(tag=tag)

        formula = self.get_parameter_value_str('formula')
        df = self.evaluate_formula(formula, EvalVars(nodes, datasets))
        df = df.ensure_unit(VALUE_COLUMN, self.get_default_output_metric().unit)
        extend = self.get_parameter_value('extend_last_historical_value', required=False)
        if extend:
            df = extend_last_historical_value_pl(df, self.get_end_year())
        return df
