"""
Formulas as a text view of pipelines.

A pipeline (``PipelineSpec``) is the stored form of a node's computation; a
formula is a lossless, canonically formatted text view of it. Experts can edit
either, and both edit the same pipeline::

    # Heat-plan path, stated per carrier as a factor of today's use.
    path = interpolate(factor)
    # The hook adds the change against the node's own value.
    final_energy_use * (path - 1)

- ``name = expression`` is a step named ``name``; a nested sub-expression is an
  anonymous step, written inline.
- Comments above a statement, at its end or inside it are that statement's
  step description. Comments after the last statement describe the pipeline.
- The output is the last statement. A last line that is only the name of an
  earlier step makes that step the output (``output_ref``).
- ``+ - * /`` are ``add``, ``subtract``, ``multiply`` and ``divide``; every
  operation also has a call form (``interpolate(x)``, ``clip(x, min=0)``,
  ``add(a, b, only_if=flag)``), which carries conditions.
- ``select_category(x, dimension='category')`` keeps one category and drops
  the dimension; ``dimension=variant`` takes the category from a parameter.
- A name refers to what the scope says: an input port, a dataset, a
  parameter, or the un-hooked value of a node an action acts on.

``render_pipeline(compile_formula(text))`` is the canonical form of ``text``,
and ``compile_formula(render_pipeline(pipeline)) == pipeline``. Normalised:
whitespace, redundant parentheses, and the position of comments.

See docs/plans/formula-pipeline-round-trip.md.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from pydantic import ValidationError

from pint import PintError

from .ir import PipelineSpec
from .ops.arithmetic import (
    AddOperationSpec,
    ClipOperationSpec,
    DivideOperationSpec,
    IdentityOperationSpec,
    MultiplyOperationSpec,
    SubtractOperationSpec,
)
from .ops.base import (
    ANONYMOUS_STEP_PREFIX,
    ComparisonCondition,
    ComparisonOperator,
    HookBaseInputRef,
    InputOperationSpec,
    IntermediateInputRef,
    OperationSpec,
    ParameterInputRef,
    PortInputRef,
    ScalarValue,
    TruthyCondition,
    VariadicOperationSpec,
    step_key,
)
from .ops.dimensional import SelectCategoryOperationSpec
from .ops.temporal import BackfillOperationSpec, ExtendOperationSpec, InterpolateOperationSpec

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from nodes.defs.node_defs import NodeSpec

    from .ops.base import OperationCondition, OperationInput
    from .ops.union import AnyOperationSpec


IDENTIFIER_RE = re.compile(r'^[a-z_][a-z0-9_]*$')

type _SingleInputType = type[IdentityOperationSpec | InterpolateOperationSpec | ExtendOperationSpec | BackfillOperationSpec]
type _VariadicType = type[AddOperationSpec | SubtractOperationSpec | MultiplyOperationSpec]

VARIADIC_OPERATORS: dict[type[ast.operator], _VariadicType] = {
    ast.Add: AddOperationSpec,
    ast.Sub: SubtractOperationSpec,
    ast.Mult: MultiplyOperationSpec,
}
OPERATOR_SYMBOLS: dict[type[OperationSpec], str] = {
    AddOperationSpec: '+',
    SubtractOperationSpec: '-',
    MultiplyOperationSpec: '*',
    DivideOperationSpec: '/',
}
PRECEDENCE: dict[type[OperationSpec], int] = {
    AddOperationSpec: 1,
    SubtractOperationSpec: 1,
    MultiplyOperationSpec: 2,
    DivideOperationSpec: 2,
}
ATOM = 3

SINGLE_INPUT_OPERATIONS: dict[str, _SingleInputType] = {
    'identity': IdentityOperationSpec,
    'interpolate': InterpolateOperationSpec,
    'extend': ExtendOperationSpec,
    'backfill': BackfillOperationSpec,
}
VARIADIC_OPERATIONS: dict[str, _VariadicType] = {
    'add': AddOperationSpec,
    'subtract': SubtractOperationSpec,
    'multiply': MultiplyOperationSpec,
}
CONDITION_KEYWORDS = ('only_if', 'skip_if')
SELECT_CATEGORY_FUNCTION = 'select_category'
QUANTITY_FUNCTION = 'quantity'
"""Call form of a scalar with a unit: ``quantity(2.5, 'kg/a')``. Plain numbers are dimensionless."""

COMPARISONS: dict[type[ast.cmpop], ComparisonOperator] = {
    ast.Eq: ComparisonOperator.EQ,
    ast.NotEq: ComparisonOperator.NE,
    ast.Gt: ComparisonOperator.GT,
    ast.GtE: ComparisonOperator.GTE,
    ast.Lt: ComparisonOperator.LT,
    ast.LtE: ComparisonOperator.LTE,
}
COMPARISON_SYMBOLS = {
    ComparisonOperator.EQ: '==',
    ComparisonOperator.NE: '!=',
    ComparisonOperator.GT: '>',
    ComparisonOperator.GTE: '>=',
    ComparisonOperator.LT: '<',
    ComparisonOperator.LTE: '<=',
}


class FormulaError(ValueError):
    """A formula that does not compile to a pipeline, with the position of the cause."""

    def __init__(self, message: str, line: int | None = None, column: int | None = None) -> None:
        self.message = message
        self.line = line
        self.column = column
        where = f'line {line}: ' if line is not None else ''
        super().__init__(f'{where}{message}')

    @classmethod
    def at(cls, node: ast.AST, message: str) -> FormulaError:
        return cls(message, getattr(node, 'lineno', None), getattr(node, 'col_offset', None))


def _input_key(value: OperationInput) -> str:
    return value.model_dump_json(exclude_defaults=True)


@dataclass(frozen=True)
class FormulaScope:
    """What the names in a formula refer to, and the name each input is written as."""

    inputs: Mapping[str, OperationInput] = field(default_factory=dict)
    _names: dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        seen: dict[str, str] = {}
        for name, value in self.inputs.items():
            if not IDENTIFIER_RE.match(name):
                raise ValueError(f'{name!r} cannot be written as a formula name')
            key = _input_key(value)
            if key in seen:
                raise ValueError(f'{seen[key]!r} and {name!r} name the same input')
            seen[key] = name
        object.__setattr__(self, '_names', dict(seen))

    @classmethod
    def for_node_spec(
        cls, spec: NodeSpec, *, hook_targets: Iterable[str] = (), global_parameters: Iterable[str] = ()
    ) -> FormulaScope:
        """
        Build the scope of a node's pipeline.

        Input ports are written as their identifiers and parameters as their
        local ids; a node parameter shadows a global one of the same id.
        ``hook_targets`` are the identifiers of the nodes an action acts on; in
        the action's own formula they stand for those nodes' un-hooked values.
        """
        inputs: dict[str, OperationInput] = {}
        for port in spec.input_ports:
            if port.identifier is not None:
                inputs[port.identifier] = PortInputRef(port=port.id)
        for param in spec.params:
            inputs.setdefault(param.local_id, ParameterInputRef(parameter=param.local_id))
        for param_id in global_parameters:
            inputs.setdefault(param_id, ParameterInputRef(parameter=param_id))
        for node_id in hook_targets:
            inputs[node_id] = HookBaseInputRef(hook_base=node_id)
        return cls(inputs)

    def resolve(self, name: str) -> OperationInput | None:
        return self.inputs.get(name)

    def name_for(self, value: OperationInput) -> str:
        name = self._names.get(_input_key(value))
        if name is None:
            raise FormulaError(f'The formula scope has no name for {_input_key(value)}')
        return name


# --- Compiling ---------------------------------------------------------------


@dataclass
class _Comment:
    line: int
    text: str


def _collect_comments(text: str) -> list[_Comment]:
    comments: list[_Comment] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                body = token.string[1:]
                body = body.removeprefix(' ')
                comments.append(_Comment(token.start[0], body.rstrip()))
    except tokenize.TokenError as exc:
        raise FormulaError(str(exc.args[0]), *exc.args[1]) from exc
    return comments


def _describe(comments: list[_Comment]) -> str | None:
    if not comments:
        return None
    return '\n'.join(comment.text for comment in comments)


class _Compiler:
    def __init__(self, scope: FormulaScope) -> None:
        self.scope = scope
        self.operations: list[AnyOperationSpec] = []
        self.named: set[str] = set()

    def append(self, operation: AnyOperationSpec) -> IntermediateInputRef:
        index = len(self.operations)
        self.operations.append(operation)
        return IntermediateInputRef(ref=step_key(operation, index))

    def compile(self, text: str) -> PipelineSpec:
        try:
            module = ast.parse(text, '<formula>', mode='exec')
        except SyntaxError as exc:
            raise FormulaError(exc.msg, exc.lineno, (exc.offset or 1) - 1) from exc
        if not module.body:
            raise FormulaError('The formula is empty')
        comments = _collect_comments(text)

        output_ref: str | None = None
        previous_end = 0
        last = len(module.body) - 1
        for index, statement in enumerate(module.body):
            if index == last and isinstance(statement, ast.Expr) and self._is_step_name(statement.value):
                # `name` alone on the last line: that step is the output. Comments around it
                # have no step of their own, so they go to the pipeline's description.
                output_ref = cast('ast.Name', statement.value).id
                break
            end = statement.end_lineno or statement.lineno
            mine = [comment for comment in comments if previous_end < comment.line <= end]
            previous_end = end
            self._statement(statement, _describe(mine), is_last=index == last)

        description = _describe([comment for comment in comments if comment.line > previous_end])
        try:
            return PipelineSpec(operations=self.operations, output_ref=output_ref, description=description)
        except ValidationError as exc:
            raise FormulaError(exc.errors()[0]['msg']) from exc

    def _is_step_name(self, expr: ast.expr) -> bool:
        return isinstance(expr, ast.Name) and expr.id in self.named

    def _statement(self, statement: ast.stmt, description: str | None, *, is_last: bool) -> None:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise FormulaError.at(statement, 'Assign to exactly one name')
            name = statement.targets[0].id
            if not IDENTIFIER_RE.match(name) or name.startswith(ANONYMOUS_STEP_PREFIX):
                raise FormulaError.at(statement, f'{name!r} cannot name a step; use lowercase letters, digits and _')
            if name in self.named or self.scope.resolve(name) is not None:
                raise FormulaError.at(statement, f'{name!r} is already defined')
            self._step(statement.value, name, description)
            self.named.add(name)
            return
        if isinstance(statement, ast.Expr):
            if not is_last:
                raise FormulaError.at(statement, 'Only the last line can be an expression without a name')
            self._step(statement.value, None, description)
            return
        raise FormulaError.at(statement, 'A formula line is either `name = expression` or, last, an expression')

    def _step(self, expr: ast.expr, name: str | None, description: str | None) -> None:
        """Compile a statement's expression into a step of its own, named ``name``."""
        count = len(self.operations)
        value = self.expr(expr)
        if len(self.operations) > count:
            # The expression's outermost operation is the statement's step.
            operation = self.operations[-1]
            operation.result_id = name
            operation.description = description
            return
        # A bare input or step name: `x = y`.
        self.operations.append(IdentityOperationSpec(input=value, result_id=name, description=description))

    def expr(self, expr: ast.expr) -> OperationInput:
        match expr:
            case ast.Constant(value=bool()):
                raise FormulaError.at(expr, 'Use a parameter for a flag')
            case ast.Constant(value=int() | float() as number):
                return ScalarValue(value=float(number), dimensionless=True)
            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=int() | float() as number)) if not isinstance(
                number, bool
            ):
                return ScalarValue(value=-float(number), dimensionless=True)
            case ast.UnaryOp():
                raise FormulaError.at(expr, 'Write a negation as a subtraction or a multiplication by -1')
            case ast.Name(id=name):
                if name in self.named:
                    return IntermediateInputRef(ref=name)
                value = self.scope.resolve(name)
                if value is None:
                    raise FormulaError.at(expr, f'Unknown name {name!r}')
                return value
            case ast.BinOp(op=op):
                return self._binop(expr, op)
            case ast.Call():
                return self._call(expr)
        raise FormulaError.at(expr, f'{type(expr).__name__} is not representable as a pipeline yet')

    def _binop(self, expr: ast.BinOp, op: ast.operator) -> OperationInput:
        if isinstance(op, ast.Div):
            left = self.expr(expr.left)
            right = self.expr(expr.right)
            return self.append(DivideOperationSpec(input=left, other=right))
        spec_type = VARIADIC_OPERATORS.get(type(op))
        if spec_type is None:
            raise FormulaError.at(expr, f'The operator {type(op).__name__} is not representable as a pipeline yet')
        # A left-leaning chain of the same operator is one step: `a + b + c`.
        operands: list[ast.expr] = [expr.right]
        head: ast.expr = expr.left
        while isinstance(head, ast.BinOp) and type(head.op) is type(op):
            operands.append(head.right)
            head = head.left
        first = self.expr(head)
        values = [self.expr(operand) for operand in reversed(operands)]
        return self.append(spec_type(input=first, values=values))

    def _keyword(self, call: ast.Call, name: str) -> ast.expr | None:
        return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)

    def _add_conditions(self, call: ast.Call, operation: AnyOperationSpec) -> AnyOperationSpec:
        # Compiled in the order the renderer writes them, whatever order they were written in,
        # so that the anonymous steps they contain are numbered canonically.
        for keyword in CONDITION_KEYWORDS:
            value = self._keyword(call, keyword)
            if value is None:
                continue
            condition: OperationCondition
            if isinstance(value, ast.Compare):
                if len(value.ops) != 1:
                    raise FormulaError.at(value, 'A condition compares exactly two values')
                comparison = COMPARISONS.get(type(value.ops[0]))
                if comparison is None:
                    raise FormulaError.at(value, f'{type(value.ops[0]).__name__} is not a supported comparison')
                left = self.expr(value.left)
                right = self.expr(value.comparators[0])
                condition = ComparisonCondition(left=left, op=comparison, right=right)
            else:
                condition = TruthyCondition(input=self.expr(value))
            setattr(operation, keyword, condition)
        return operation

    def _call(self, call: ast.Call) -> OperationInput:  # noqa: C901, PLR0912
        if not isinstance(call.func, ast.Name):
            raise FormulaError.at(call, 'Call an operation by its name')
        func = call.func.id
        if func == QUANTITY_FUNCTION:
            return self._quantity(call)
        if func == SELECT_CATEGORY_FUNCTION:
            return self._select_category(call)
        keywords = {keyword.arg for keyword in call.keywords}
        if None in keywords:
            raise FormulaError.at(call, 'Keyword unpacking is not supported')
        allowed = set(CONDITION_KEYWORDS) | ({'min', 'max'} if func == 'clip' else set())
        unknown = sorted(cast('set[str]', keywords) - allowed)
        if unknown:
            raise FormulaError.at(call, f'{func}() has no argument {unknown[0]!r}')
        if any(isinstance(arg, ast.Starred) for arg in call.args):
            raise FormulaError.at(call, 'Argument unpacking is not supported')

        operation: AnyOperationSpec
        if func in SINGLE_INPUT_OPERATIONS:
            if len(call.args) != 1:
                raise FormulaError.at(call, f'{func}() takes one input')
            value = self.expr(call.args[0])
            operation = SINGLE_INPUT_OPERATIONS[func](input=value)
        elif func in VARIADIC_OPERATIONS:
            if len(call.args) < 2:
                raise FormulaError.at(call, f'{func}() takes two or more inputs')
            values = [self.expr(arg) for arg in call.args]
            operation = VARIADIC_OPERATIONS[func](input=values[0], values=values[1:])
        elif func == 'divide':
            if len(call.args) != 2:
                raise FormulaError.at(call, 'divide() takes two inputs')
            left, right = (self.expr(arg) for arg in call.args)
            operation = DivideOperationSpec(input=left, other=right)
        elif func == 'clip':
            if len(call.args) != 1:
                raise FormulaError.at(call, 'clip() takes one input')
            value = self.expr(call.args[0])
            min_expr, max_expr = self._keyword(call, 'min'), self._keyword(call, 'max')
            min_value = self.expr(min_expr) if min_expr is not None else None
            max_value = self.expr(max_expr) if max_expr is not None else None
            try:
                operation = ClipOperationSpec(input=value, min_value=min_value, max_value=max_value)
            except ValidationError as exc:
                raise FormulaError.at(call, exc.errors()[0]['msg']) from exc
        else:
            raise FormulaError.at(call, f'{func}() is not representable as a pipeline yet')
        return self.append(self._add_conditions(call, operation))

    def _select_category(self, call: ast.Call) -> OperationInput:
        """`select_category(x, dimension='category')`, or with a parameter naming the category: `dimension=variant`."""
        selections = [keyword for keyword in call.keywords if keyword.arg not in CONDITION_KEYWORDS]
        if len(call.args) != 1 or len(selections) != 1 or selections[0].arg is None:
            raise FormulaError.at(call, "select_category() takes one input and one dimension=category, e.g. sector='heating'")
        selection = selections[0]
        dimension = cast('str', selection.arg)
        value = self.expr(call.args[0])
        category: str | ParameterInputRef
        match selection.value:
            case ast.Constant(value=str() as literal):
                category = literal
            case ast.Name(id=name):
                resolved = self.scope.resolve(name)
                if not isinstance(resolved, ParameterInputRef):
                    raise FormulaError.at(selection.value, f'{name!r} is not a parameter')
                category = resolved
            case _:
                raise FormulaError.at(selection.value, 'A category is a quoted id or a parameter')
        try:
            operation = SelectCategoryOperationSpec(input=value, dimension=dimension, category=category)
        except ValidationError as exc:
            raise FormulaError.at(call, exc.errors()[0]['msg']) from exc
        return self.append(self._add_conditions(call, operation))

    def _quantity(self, call: ast.Call) -> ScalarValue:
        match call.args:
            case [value_node, ast.Constant(value=str() as unit)] if not call.keywords:
                value = self.expr(value_node)
                if not isinstance(value, ScalarValue):
                    raise FormulaError.at(call, 'quantity() takes a number and a unit')
                try:
                    return ScalarValue(value=value.value, unit=unit)  # type: ignore[arg-type]
                except (ValidationError, PintError) as exc:
                    raise FormulaError.at(call, f'Invalid unit {unit!r}') from exc
        raise FormulaError.at(call, "quantity() takes a number and a unit, e.g. quantity(2.5, 'kg/a')")


def compile_formula(text: str, scope: FormulaScope) -> PipelineSpec:
    """Compile formula text into a pipeline. Raises ``FormulaError`` with the line of the cause."""
    return _Compiler(scope).compile(text)


# --- Rendering ---------------------------------------------------------------


def _number(value: float) -> str:
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


class _Renderer:
    def __init__(self, pipeline: PipelineSpec, scope: FormulaScope) -> None:
        self.pipeline = pipeline
        self.scope = scope
        self.steps: dict[str, AnyOperationSpec] = {}
        output_index = len(pipeline.operations) - 1 if pipeline.output_ref is None else None
        self.inline: set[str] = set()
        for index, operation in enumerate(pipeline.operations):
            key = step_key(operation, index)
            self.steps[key] = operation
            if operation.result_id is None and index != output_index:
                self.inline.add(key)

    def render(self) -> str:
        lines: list[str] = []
        output_index = len(self.pipeline.operations) - 1 if self.pipeline.output_ref is None else None
        for index, operation in enumerate(self.pipeline.operations):
            if step_key(operation, index) in self.inline:
                continue
            if operation.description is not None:
                if lines:
                    lines.append('')
                lines.extend(self._comment(operation.description))
            expr = self.statement(operation, is_output=index == output_index)
            lines.append(expr if operation.result_id is None else f'{operation.result_id} = {expr}')
        if self.pipeline.output_ref is not None:
            lines.append(self.pipeline.output_ref)
        if self.pipeline.description is not None:
            lines.append('')
            lines.extend(self._comment(self.pipeline.description))
        return '\n'.join(lines) + '\n'

    @staticmethod
    def _comment(text: str) -> list[str]:
        return [f'# {line}' if line else '#' for line in text.split('\n')]

    def statement(self, operation: AnyOperationSpec, *, is_output: bool) -> str:
        # `x = y` is an identity step; so is a bare last line naming an input, but a bare
        # last line naming a step would make that step the output instead.
        if isinstance(operation, IdentityOperationSpec) and not self._has_conditions(operation):
            value = operation.input
            if not isinstance(value, IntermediateInputRef):
                return self.value(value, 0)
            if value.ref not in self.inline and not is_output:
                return value.ref
        return self.operation(operation, 0)

    @staticmethod
    def _has_conditions(operation: OperationSpec) -> bool:
        return operation.only_if is not None or operation.skip_if is not None

    def value(self, value: OperationInput, context: int, *, right: bool = False) -> str:
        """Render an input where the surrounding operator has precedence ``context``."""
        match value:
            case IntermediateInputRef(ref=ref) if ref in self.inline:
                return self.operation(self.steps[ref], context, right=right)
            case IntermediateInputRef(ref=ref):
                return ref
            case ScalarValue(value=number, unit=None):
                return _number(number)
            case ScalarValue(value=number, unit=unit):
                return f'{QUANTITY_FUNCTION}({_number(number)}, {str(unit)!r})'
        return self.scope.name_for(value)

    def operation(self, operation: AnyOperationSpec, context: int, *, right: bool = False) -> str:
        symbol = OPERATOR_SYMBOLS.get(type(operation))
        if symbol is None or self._has_conditions(operation) or self._needs_call(operation):
            return self.call(operation)
        precedence = PRECEDENCE[type(operation)]
        if isinstance(operation, DivideOperationSpec):
            operands = [operation.input, operation.other]
        else:
            assert isinstance(operation, VariadicOperationSpec)
            operands = [operation.input, *operation.values]
        parts = [self.value(operands[0], precedence)]
        parts.extend(self.value(operand, precedence, right=True) for operand in operands[1:])
        text = f' {symbol} '.join(parts)
        if precedence < context or (right and precedence == context):
            return f'({text})'
        return text

    def _needs_call(self, operation: AnyOperationSpec) -> bool:
        """
        Whether operators would merge this step with its first input.

        `a + b + c` is one step. Two steps, `(a + b) + c`, are written
        `add(a + b, c)`.
        """
        if not isinstance(operation, VariadicOperationSpec):
            return False
        first = operation.input
        if not isinstance(first, IntermediateInputRef) or first.ref not in self.inline:
            return False
        inner = self.steps[first.ref]
        return type(inner) is type(operation) and not self._has_conditions(inner)

    def call(self, operation: AnyOperationSpec) -> str:
        args: list[str]
        match operation:
            case VariadicOperationSpec():
                args = [self.value(value, 0) for value in (operation.input, *operation.values)]
            case DivideOperationSpec():
                args = [self.value(operation.input, 0), self.value(operation.other, 0)]
            case ClipOperationSpec():
                args = [self.value(operation.input, 0)]
                if operation.min_value is not None:
                    args.append(f'min={self.value(operation.min_value, 0)}')
                if operation.max_value is not None:
                    args.append(f'max={self.value(operation.max_value, 0)}')
            case SelectCategoryOperationSpec():
                category = operation.category
                chosen = self.scope.name_for(category) if isinstance(category, ParameterInputRef) else repr(category)
                args = [self.value(operation.input, 0), f'{operation.dimension}={chosen}']
            case InputOperationSpec():
                args = [self.value(operation.input, 0)]
            case _:
                raise FormulaError(f'Operation {operation.kind!r} has no formula form')
        for keyword in CONDITION_KEYWORDS:
            condition = getattr(operation, keyword)
            if condition is not None:
                args.append(f'{keyword}={self.condition(condition)}')
        return f'{operation.kind}({", ".join(args)})'

    def condition(self, condition: OperationCondition) -> str:
        if isinstance(condition, TruthyCondition):
            return self.value(condition.input, 0)
        symbol = COMPARISON_SYMBOLS[condition.op]
        return f'{self.value(condition.left, 0)} {symbol} {self.value(condition.right, 0)}'


def render_pipeline(pipeline: PipelineSpec, scope: FormulaScope) -> str:
    """Render a pipeline as canonical formula text."""
    return _Renderer(pipeline, scope).render()
