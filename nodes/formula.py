from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple, TypeVar, cast

from django.utils.translation import gettext_lazy as _

import polars as pl

from common import polars as ppl
from nodes.calc import convert_to_co2e, extend_last_historical_value_pl
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN
from nodes.exceptions import NodeError
from nodes.units import Quantity, QuantityType, Unit, unit_registry
from params.param import BoolParameter, NumberParameter, StringParameter

from .node import Node

if TYPE_CHECKING:
    from collections.abc import Iterable

PDF = ppl.PathsDataFrame
type EvalConst = float
type EvalOutput = PDF | QuantityType | bool

# Tolerance for treating values as logical 0/1 in and()/or() runtime warnings
LOGICAL_TOLERANCE = 1e-6


class EvalVars(NamedTuple):
    nodes: dict[str, Node]
    datasets: dict[str, PDF]
    parameters: dict[str, QuantityType | bool]


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
        BoolParameter(local_id='condition'),
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
            return df1.paths.divide_with_dims(df2, how='inner')
        def left_df(df: PDF, val: QuantityType) -> PDF:
            assert isinstance(val, Quantity)
            return df.divide_by_quantity(VALUE_COLUMN, val)
        def right_df(val: QuantityType, df: PDF) -> PDF:
            assert isinstance(val, Quantity)
            return df.divide_quantity(VALUE_COLUMN, val)
        def both_quantity(val1: Quantity, val2: Quantity) -> Quantity:
            out = val1 / val2
            assert isinstance(out, Quantity)
            return out
        return self.apply_binom(left, right, both_df, left_df, right_df, both_quantity)

    def apply_sub(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        neg_one = cast("Quantity", Quantity(-1))
        if isinstance(right, PDF):
            right = right.multiply_quantity(VALUE_COLUMN, neg_one)
        elif isinstance(right, Quantity):
            right = right * neg_one
        return self.apply_add(left, right)

    def apply_pow(self, left: EvalOutput, right: EvalOutput) -> EvalOutput:
        if isinstance(right, PDF):
            raise NotImplementedError("Power with dataframe exponent is not supported.")
        assert isinstance(right, Quantity)
        try:
            right = right.to('dimensionless')
        except Exception as exc:
            raise NodeError(self, "Power exponent must be dimensionless.") from exc
        exponent = right.m
        if isinstance(left, PDF):
            val_col = VALUE_COLUMN
            df = left.with_columns((pl.col(val_col) ** pl.lit(exponent)).alias(val_col))
            df = df.set_unit(
                val_col,
                cast("Unit", left.get_unit(val_col) ** exponent),
                force=True,
            )
            return df
        assert isinstance(left, Quantity)
        out = left ** exponent
        assert isinstance(out, Quantity)
        return out

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
            ast.Sub: self.apply_sub,
            ast.Mult: self.apply_mul,
            ast.Div: self.apply_div,
            ast.Pow: self.apply_pow,
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
        self, func: str, node: ast.Call, varss: EvalVars, df: EvalOutput
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

        if func == 'prod_dim':
            assert len(node.args) == 2
            assert isinstance(df, PDF)
            dim_arg = node.args[1]
            assert isinstance(dim_arg, ast.Name)
            assert isinstance(dim_arg.id, str)
            return df.paths.prod_over_dims(dim_arg.id)

        if func == 'zero_fill':
            assert isinstance(df, PDF)
            df = df.paths.to_wide()
            meta = df.get_meta()
            zdf = df.fill_null(0)
            return ppl.to_ppdf(zdf, meta=meta).paths.to_narrow()

        if func == 'select_port':
            assert len(node.args) == 3
            assert isinstance(df, bool)
            if df:
                return self.eval_tree(node.args[1], varss)
            return self.eval_tree(node.args[2], varss)

        if func == 'float': # FIXME Does this actually make sense?
            assert isinstance(df, bool)
            assert len(node.args) == 1
            if df:
                return Quantity(1.0, 'dimensionless')
            return Quantity(0.0, 'dimensionless')

        if func == 'coalesce_df':
            assert len(node.args) == 2
            df1 = self.eval_tree(node.args[0], varss)
            df2 = self.eval_tree(node.args[1], varss)
            assert isinstance(df1, PDF)
            assert isinstance(df2, PDF)
            return df1.paths.coalesce_df(df2, how='outer')

        if func in ('max', 'min'):
            assert len(node.args) == 2, f"{func}(a, b) requires two arguments"
            left = self.eval_tree(node.args[0], varss)
            right = self.eval_tree(node.args[1], varss)
            assert func in ('max', 'min')
            return self._apply_max_min(left, right, func)

        if func in ('and', 'or'):
            assert len(node.args) == 2, f"{func}(a, b) requires two arguments"
            left = self.eval_tree(node.args[0], varss)
            right = self.eval_tree(node.args[1], varss)
            op = 'min' if func == 'and' else 'max'
            result = self._apply_max_min(left, right, op)
            self._append_non_binary_warning_if_needed(left, right, result, func)
            return result

        raise NotImplementedError(f"Unknown function: {func}")

    def _apply_max_min(
        self, left: EvalOutput, right: EvalOutput, op: Literal['max', 'min']
    ) -> EvalOutput:
        """Element-wise max or min. For 0/1 values, max(a,b) is logical OR, min(a,b) is logical AND."""
        if isinstance(left, Quantity) and isinstance(right, Quantity):
            unit = left.u
            r_m = right.to(unit).m
            l_m = left.m
            val = max(l_m, r_m) if op == 'max' else min(l_m, r_m)
            return Quantity(val, unit)
        if isinstance(left, PDF) and isinstance(right, Quantity):
            unit = left.get_unit(VALUE_COLUMN)
            scalar = float(right.to(unit).m)
            return left.paths.max_with_scalar(scalar) if op == 'max' else left.paths.min_with_scalar(scalar)
        if isinstance(left, Quantity) and isinstance(right, PDF):
            unit = right.get_unit(VALUE_COLUMN)
            scalar = float(left.to(unit).m)
            return right.paths.max_with_scalar(scalar) if op == 'max' else right.paths.min_with_scalar(scalar)
        if isinstance(left, PDF) and isinstance(right, PDF):
            return left.paths.max_with(right) if op == 'max' else left.paths.min_with(right)
        raise NotImplementedError(f"{op} requires two parameters that are quantities or dataframes.")

    def _has_non_binary_values(self, x: EvalOutput) -> bool:
        """Return True if any value deviates from 0 and from 1 by more than LOGICAL_TOLERANCE."""
        tol = LOGICAL_TOLERANCE
        if isinstance(x, Quantity):
            m = float(x.to(unit_registry('dimensionless')).m)
            return abs(m) > tol and abs(m - 1.0) > tol
        if isinstance(x, PDF):
            col = x.get_column(VALUE_COLUMN)
            near_zero = col.abs() <= tol
            near_one = (col - 1.0).abs() <= tol
            return bool((~(near_zero | near_one)).any())
        return False

    def _append_non_binary_warning_if_needed(
        self,
        left: EvalOutput,
        right: EvalOutput,
        result: EvalOutput,
        func_name: Literal['and', 'or'],
    ) -> None:
        """If and()/or() received non-binary values, append a warning to result._explanation (when result is PDF)."""
        if (self._has_non_binary_values(left) or self._has_non_binary_values(right)) and isinstance(result, PDF):
            msg = _(
                "Logical %(func)s() received values outside {0, 1}; interpreted as fuzzy logic."
            ) % {'func': func_name}
            result._explanation.append(msg)

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

    def _collect_eval_vars(self) -> EvalVars:  # noqa: C901, PLR0912
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
        parameters: dict[str, QuantityType | bool] = {}
        for param_id, param in self.parameters.items():
            if isinstance(param, ParameterWithUnit) and param.unit is not None:
                val = self.get_parameter_value(param_id, required=False, units=True)
                if val is not None:
                    assert isinstance(val, Quantity)
                    parameters[param_id] = val
            else:
                val = self.get_parameter_value(param_id, required=False)
                if isinstance(val, bool):
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


@dataclass
class DimensionAnalysis:
    dims: set[str] | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class UnitAnalysis:
    unit: Unit | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FormulaSpec:
    expression: str
    display_expression: str
    terms: list[dict[str, Any]]


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def make_identifier(name: str) -> str:
    if _IDENTIFIER_RE.match(name):
        return name
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not normalized:
        return "_term"
    if normalized[0].isdigit():
        return f"_{normalized}"
    return normalized


def collect_term_names(terms: Iterable[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for term in terms:
        for field_name in ('label', 'key'):
            val = term.get(field_name)
            if isinstance(val, str) and val:
                names.add(val)
        for var_name in term.get('var_names', []) or []:
            if isinstance(var_name, str) and var_name:
                names.add(var_name)
    return names


def normalize_formula_identifiers(formula: str, term_names: Iterable[str]) -> str:
    updated = formula
    for name in sorted(set(term_names), key=len, reverse=True):
        safe = make_identifier(name)
        if safe == name:
            continue
        updated = updated.replace(name, safe)
    return updated


def build_name_unit_map(
    terms: Iterable[dict[str, Any]],
) -> dict[str, Unit | None]:
    name_units: dict[str, Unit | None] = {}
    for term in terms:
        unit_val = term.get('unit')
        unit: Unit | None
        if isinstance(unit_val, Unit):
            unit = unit_val
        elif isinstance(unit_val, str) and unit_val:
            unit = unit_registry.parse_units(unit_val)
        elif term.get('kind') == 'constant':
            unit = unit_registry.parse_units('dimensionless')
        else:
            unit = None
        names: list[str] = []
        for field_name in ('label', 'key'):
            val = term.get(field_name)
            if isinstance(val, str) and val:
                names.append(val)
        names.extend([
            var_name
            for var_name in term.get('var_names', []) or []
            if isinstance(var_name, str) and var_name
        ])
        for name in dict.fromkeys(names):
            for candidate in {name, make_identifier(name)}:
                if candidate not in name_units:
                    name_units[candidate] = unit
    return name_units


UnitOverride = Unit | str | Callable[[Unit | None], Unit | None]


def analyze_formula_units(  # noqa: C901, PLR0915
    formula: str,
    name_units: dict[str, Unit | None],
    passthrough_functions: set[str] | None = None,
    unit_overrides: dict[str, UnitOverride] | None = None,
) -> UnitAnalysis:
    try:
        tree = ast.parse(formula, "<string>", mode="eval")
    except SyntaxError as exc:
        return UnitAnalysis(
            unit=None,
            errors=[f"Formula parse error: {exc}"],
        )

    analysis = UnitAnalysis(unit=None)
    passthrough = passthrough_functions or set()
    overrides = unit_overrides or {}
    dimensionless = unit_registry.parse_units('dimensionless')

    def _merge_compatible(op: str, left: Unit | None, right: Unit | None) -> Unit | None:
        if left is None or right is None:
            analysis.warnings.append(f"Unknown unit for '{op}'.")
            return left or right
        if left.dimensionality != right.dimensionality:
            analysis.errors.append(
                f"Unit mismatch for '{op}': {left} vs {right}"
            )
            return left
        return left

    def _eval(node: ast.AST) -> Unit | None:  # noqa: C901, PLR0911, PLR0912, PLR0915
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return dimensionless
        if isinstance(node, ast.Name):
            name = node.id
            if name not in name_units:
                analysis.errors.append(f"Unknown formula term '{name}'.")
                return None
            unit = name_units[name]
            if unit is None:
                analysis.warnings.append(f"Missing unit for term '{name}'.")
            return unit
        if isinstance(node, ast.UnaryOp):
            return _eval(node.operand)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return _merge_compatible('+', left, right)
            if isinstance(node.op, ast.Mult):
                if left is None or right is None:
                    analysis.warnings.append("Unknown unit for '*'.")
                    return left or right
                return cast("Unit", left * right)
            if isinstance(node.op, ast.Div):
                if left is None or right is None:
                    analysis.warnings.append("Unknown unit for '/'.")
                    return left or right
                return cast("Unit", left / right)
            if isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant):
                    analysis.warnings.append("Power exponent is not a constant.")
                    return left
                if left is None:
                    analysis.warnings.append("Unknown unit for '**'.")
                    return None
                exponent = node.right.value
                if not isinstance(exponent, (int, float)):
                    analysis.warnings.append("Power exponent is not numeric.")
                    return left
                return cast("Unit", left ** exponent)
            analysis.warnings.append(f"Unsupported binary operator: {type(node.op).__name__}")
            return left or right
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for comp in node.comparators:
                right = _eval(comp)
                _merge_compatible('compare', left, right)
                left = right
            return dimensionless
        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Name):
                analysis.warnings.append("Unsupported callable in formula.")
                return None
            func_name = func.id
            if not node.args:
                analysis.warnings.append(f"Function '{func_name}' has no arguments.")
                return None
            first = _eval(node.args[0])
            if func_name in overrides:
                override_unit = overrides[func_name]
                if callable(override_unit):
                    return override_unit(first)
                if isinstance(override_unit, str):
                    return unit_registry.parse_units(override_unit)
                return override_unit
            if func_name == 'sum_dim':
                return first
            if func_name == 'coalesce':
                unit = first
                for arg in node.args[1:]:
                    unit = _merge_compatible('coalesce', unit, _eval(arg))
                return unit
            if func_name in ('max', 'min', 'and', 'or'):
                if len(node.args) != 2:
                    analysis.warnings.append(f"{func_name}(a, b) requires two arguments.")
                else:
                    second = _eval(node.args[1])
                    return _merge_compatible(func_name, first, second)
                return first
            if func_name == 'geometric_inverse':
                if first is None:
                    analysis.warnings.append("Unknown unit for 'geometric_inverse'.")
                    return None
                return cast("Unit", dimensionless / first)
            if func_name == 'complement':
                if (
                    first is not None
                    and first.dimensionality != dimensionless.dimensionality
                ):
                    analysis.errors.append("Function 'complement' requires dimensionless units.")
                return dimensionless
            if func_name in {'convert_gwp', 'zero_fill'} or func_name in passthrough:
                return first
            analysis.warnings.append(f"Unknown function '{func_name}' in formula.")
            return first
        analysis.warnings.append(f"Unsupported AST node: {type(node).__name__}")
        return None

    analysis.unit = _eval(tree)
    return analysis


def analyze_formula_dimensions(  # noqa: C901, PLR0915
    formula: str,
    name_dimensions: dict[str, set[str]],
    passthrough_functions: set[str] | None = None,
) -> DimensionAnalysis:
    try:
        tree = ast.parse(formula, "<string>", mode="eval")
    except SyntaxError as exc:
        return DimensionAnalysis(
            dims=None,
            errors=[f"Formula parse error: {exc}"],
        )

    analysis = DimensionAnalysis(dims=None)
    passthrough = passthrough_functions or set()

    def _merge_union(left: set[str] | None, right: set[str] | None) -> set[str] | None:
        if left is None and right is None:
            return None
        if left is None:
            return set(right or set())
        if right is None:
            return set(left)
        return set(left) | set(right)

    def _require_same(op: str, left: set[str] | None, right: set[str] | None) -> set[str] | None:
        if left is None or right is None:
            return left if left is not None else right
        if left != right:
            analysis.errors.append(
                f"Dimension mismatch for '{op}': {sorted(left)} vs {sorted(right)}"
            )
            return _merge_union(left, right)
        return left

    def _eval(node: ast.AST) -> set[str] | None:  # noqa: C901, PLR0911, PLR0912
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return set()
        if isinstance(node, ast.Name):
            name = node.id
            if name not in name_dimensions:
                analysis.errors.append(f"Unknown formula term '{name}'.")
                return None
            return set(name_dimensions[name])
        if isinstance(node, ast.UnaryOp):
            return _eval(node.operand)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return _require_same('+', left, right)
            if isinstance(node.op, (ast.Mult, ast.Div)):
                return _merge_union(left, right)
            if isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant):
                    analysis.warnings.append("Power exponent is not a constant.")
                return left
            analysis.warnings.append(f"Unsupported binary operator: {type(node.op).__name__}")
            return _merge_union(left, right)
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for comp in node.comparators:
                right = _eval(comp)
                left = _require_same('compare', left, right)
            return left
        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Name):
                analysis.warnings.append("Unsupported callable in formula.")
                return None
            func_name = func.id
            if not node.args:
                analysis.warnings.append(f"Function '{func_name}' has no arguments.")
                return None
            first = _eval(node.args[0])
            if func_name == 'sum_dim':
                if len(node.args) != 2:
                    analysis.errors.append("sum_dim requires exactly two arguments.")
                    return first
                dim_arg = node.args[1]
                if isinstance(dim_arg, ast.Name):
                    dim = dim_arg.id
                    if first is None:
                        return None
                    return set(d for d in first if d != dim)
                analysis.errors.append("sum_dim expects a dimension name as second argument.")
                return first
            if func_name == 'coalesce':
                dims = first
                for arg in node.args[1:]:
                    dims = _require_same('coalesce', dims, _eval(arg))
                return dims
            if func_name in ('max', 'min', 'and', 'or'):
                if len(node.args) != 2:
                    analysis.warnings.append(f"{func_name}(a, b) requires two arguments.")
                else:
                    return _require_same(func_name, first, _eval(node.args[1]))
                return first
            if func_name in {'convert_gwp', 'zero_fill'} or func_name in passthrough:
                return first
            analysis.warnings.append(f"Unknown function '{func_name}' in formula.")
            return first
        analysis.warnings.append(f"Unsupported AST node: {type(node).__name__}")
        return None

    analysis.dims = _eval(tree)
    return analysis


def build_name_dimension_map(
    terms: Iterable[dict[str, Any]],
) -> tuple[dict[str, set[str]], list[str]]:
    name_dimensions: dict[str, set[str]] = {}
    conflicts: list[str] = []
    for term in terms:
        kind = term.get('kind')
        dims = set(term.get('output_dimensions') or []) if kind != 'constant' else set()
        names: list[str] = []
        for field_name in ('label', 'key'):
            val = term.get(field_name)
            if isinstance(val, str) and val:
                names.append(val)
        names.extend([
            var_name
            for var_name in term.get('var_names', []) or []
            if isinstance(var_name, str) and var_name
        ])
        for name in dict.fromkeys(names):
            for candidate in {name, make_identifier(name)}:
                if candidate in name_dimensions and name_dimensions[candidate] != dims:
                    conflicts.append(candidate)
                    continue
                name_dimensions[candidate] = dims
    return name_dimensions, conflicts
