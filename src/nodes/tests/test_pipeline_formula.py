"""Formulas as a lossless text view of pipelines (`nodes.pipeline.formula`)."""

from __future__ import annotations

from uuid import uuid4

from pydantic import ValidationError

import pytest

from nodes.pipeline import PipelineSpec
from nodes.pipeline.formula import FormulaError, FormulaScope, compile_formula, render_pipeline
from nodes.pipeline.ops import (
    AddOperationSpec,
    DatasetInputRef,
    HookBaseInputRef,
    IntermediateInputRef,
    InterpolateOperationSpec,
    MultiplyOperationSpec,
    ParameterInputRef,
    PortInputRef,
    ScalarValue,
    SubtractOperationSpec,
)

pytestmark = pytest.mark.django_db

A, B, C = PortInputRef(port=uuid4()), PortInputRef(port=uuid4()), PortInputRef(port=uuid4())
SCOPE = FormulaScope({
    'a': A,
    'b': B,
    'c': C,
    'factor': DatasetInputRef(dataset='mainz/zielpfad_fernwaerme'),
    'flag': ParameterInputRef(parameter='flag'),
    'limit': ParameterInputRef(parameter='limit'),
    'final_energy_use': HookBaseInputRef(hook_base='final_energy_use'),
})


def _round_trip(text: str) -> PipelineSpec:
    pipeline = compile_formula(text, SCOPE)
    assert render_pipeline(pipeline, SCOPE) == text
    assert compile_formula(render_pipeline(pipeline, SCOPE), SCOPE) == pipeline
    return pipeline


CANONICAL = [
    'a + b\n',
    'a + b + c\n',
    'a - b - c\n',
    'a + b - c\n',
    'a - (b + c)\n',
    'a + (b + c)\n',
    '(a + b) * c\n',
    'a * b / c\n',
    'a / b / c\n',
    'a / (b / c)\n',
    'a * (b / c)\n',
    'a * -1\n',
    'a * 0.25\n',
    "a * quantity(2.5, 'kg/a')\n",
    'a\n',
    'x = a + b\nx * c\n',
    'x = a + b\ny = x * c\nx\n',
    'x = a\nidentity(x)\n',
    'interpolate(factor)\n',
    'clip(a - b, min=0, max=limit)\n',
    'add(a, b, only_if=flag)\n',
    'add(a, b, skip_if=limit > 3)\n',
    'add(a + b, c)\n',
    'multiply(a * b, c)\n',
]


@pytest.mark.parametrize('text', CANONICAL)
def test_canonical_formulas_round_trip(text: str) -> None:
    _round_trip(text)


def test_comments_become_step_and_pipeline_descriptions() -> None:
    text = (
        "# Heat-plan path, stated per carrier as a factor of today's use.\n"
        'path = interpolate(factor)\n'
        '\n'
        "# The hook adds the change against the node's own value.\n"
        '# Two lines.\n'
        'final_energy_use * (path - 1)\n'
        '\n'
        '# About the whole.\n'
    )
    pipeline = _round_trip(text)
    interpolate, subtract, multiply = pipeline.operations
    assert isinstance(interpolate, InterpolateOperationSpec)
    assert interpolate.result_id == 'path'
    assert interpolate.description == "Heat-plan path, stated per carrier as a factor of today's use."
    assert isinstance(subtract, SubtractOperationSpec)
    assert subtract.result_id is None
    assert subtract.input == IntermediateInputRef(ref='path')
    assert subtract.values == [ScalarValue(value=1, dimensionless=True)]
    assert isinstance(multiply, MultiplyOperationSpec)
    assert multiply.input == HookBaseInputRef(hook_base='final_energy_use')
    assert multiply.values == [IntermediateInputRef(ref='_step_1')]
    assert multiply.description == "The hook adds the change against the node's own value.\nTwo lines."
    assert pipeline.description == 'About the whole.'


@pytest.mark.parametrize(
    ('text', 'canonical'),
    [
        ('a+b', 'a + b\n'),
        ('((a + b)) + c', 'a + b + c\n'),
        ('(a * b)', 'a * b\n'),
        ('x = a + b  # the sum\nx', '# the sum\nx = a + b\nx\n'),
        ('x = (a +  # inside\n     b)\nx * 2', '# inside\nx = a + b\nx * 2\n'),
        ('x = a\n# loose\n\ny = x + b\ny', 'x = a\n\n# loose\ny = x + b\ny\n'),
        ('x = a + b\n# about the output\nx', 'x = a + b\nx\n\n# about the output\n'),
        ('add(a, b + c)', 'a + (b + c)\n'),
        ('clip(a, max=limit, min=0)', 'clip(a, min=0, max=limit)\n'),
        ('a * 1.0', 'a * 1\n'),
    ],
)
def test_formulas_are_normalised(text: str, canonical: str) -> None:
    assert render_pipeline(compile_formula(text, SCOPE), SCOPE) == canonical
    _round_trip(canonical)


def test_a_structured_chain_renders_in_call_form_so_it_does_not_merge() -> None:
    """`(a + b) + c` as two steps: operators alone would make it the one step `a + b + c`."""
    pipeline = PipelineSpec(
        operations=[
            AddOperationSpec(input=A, values=[B]),
            AddOperationSpec(input=IntermediateInputRef(ref='_step_0'), values=[C]),
        ],
    )
    assert render_pipeline(pipeline, SCOPE) == 'add(a + b, c)\n'
    assert compile_formula('add(a + b, c)', SCOPE) == pipeline
    assert compile_formula('add(a, b) + c', SCOPE) == pipeline


def test_the_step_identity_follows_the_name() -> None:
    before = compile_formula('x = a + b\nx * c', SCOPE)
    after = compile_formula('# new comment\nx = a + b + c\nx * c', SCOPE)
    assert before.operations[0].result_id == after.operations[0].result_id == 'x'


@pytest.mark.parametrize(
    ('text', 'message', 'line'),
    [
        ('a +', 'invalid syntax', 1),
        ('a + unknown', "Unknown name 'unknown'", 1),
        ('x = a\nx = b\nx', "'x' is already defined", 2),
        ('a = b\na', "'a' is already defined", 1),
        ('a + b\nc', 'Only the last line', 1),
        ('_step_0 = a\n_step_0', 'cannot name a step', 1),
        ('a ** 2', 'not representable', 1),
        ('x = a\nsum_dim(x, sector)', r'sum_dim\(\) is not representable', 2),
        ('-a', 'negation', 1),
        ('add(a)', 'two or more', 1),
        ('clip(a)', 'at least one bound', 1),
        ('interpolate(a, 2)', 'takes one input', 1),
        ('add(a, b, when=flag)', "no argument 'when'", 1),
        ("a * quantity(2, 'no_such_unit')", 'Invalid unit', 1),
        ('', 'empty', None),
    ],
)
def test_errors_point_at_the_line(text: str, message: str, line: int | None) -> None:
    with pytest.raises(FormulaError, match=message) as exc_info:
        compile_formula(text, SCOPE)
    assert exc_info.value.line == line


def test_anonymous_steps_are_inline_sub_expressions() -> None:
    shared = AddOperationSpec(input=A, values=[B])
    with pytest.raises(ValidationError, match='used exactly once'):
        PipelineSpec(
            operations=[
                shared,
                MultiplyOperationSpec(input=IntermediateInputRef(ref='_step_0'), values=[IntermediateInputRef(ref='_step_0')]),
            ],
        )
    with pytest.raises(ValidationError, match='cannot have a description'):
        PipelineSpec(
            operations=[
                AddOperationSpec(input=A, values=[B], description='why'),
                MultiplyOperationSpec(input=IntermediateInputRef(ref='_step_0'), values=[C]),
            ],
        )
    with pytest.raises(ValidationError, match='no earlier step'):
        PipelineSpec(operations=[AddOperationSpec(input=A, values=[IntermediateInputRef(ref='later')])])


def test_scope_names_each_input_once() -> None:
    with pytest.raises(ValueError, match='name the same input'):
        FormulaScope({'a': A, 'also_a': A})
