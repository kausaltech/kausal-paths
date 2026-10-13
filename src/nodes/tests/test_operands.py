"""
The rule that decides what each of a node's inputs is for.

``nodes.operands`` is the one copy of a rule that used to be written out in four places.
These tests pin the rule itself; ``test_add_multiply_semantics.py`` pins what the node
classes then do with the answer.
"""

import pytest

from nodes.operands import (
    NodeOperands,
    claimed_by_other_operation,
    declared_unit_of,
    output_unit_of,
    resolve_input_nodes,
    role_from_tags,
)
from nodes.tests.test_add_multiply_semantics import _connect, _generic, _make_context, _source

pytestmark = pytest.mark.django_db


# --- the tag half of the rule, which needs no node at all ------------------------


@pytest.mark.parametrize(
    ('tags', 'expected'),
    [
        (['additive'], 'additive'),
        (['non_additive'], 'factor'),
        (['impute'], 'impute'),
        ([], None),
        (['goal'], None),
        (['use_as_totals'], None),
        # The arithmetic tags win over impute — the precedence GenericNode has always had.
        (['additive', 'impute'], 'additive'),
        (['non_additive', 'impute'], 'factor'),
    ],
)
def test_role_from_tags(tags, expected):
    assert role_from_tags(tags) == expected


@pytest.mark.parametrize(
    ('tags', 'expected'),
    [
        (['use_as_totals'], True),
        (['split_by_existing_shares'], True),
        (['skip_dim_test'], True),
        (['primary'], True),
        # Roles this module assigns itself are not "some other operation".
        (['additive'], False),
        (['non_additive'], False),
        (['impute'], False),
        # A tag outside the vocabulary entirely says nothing.
        ([], False),
        (['goal'], False),
        (['some_project_specific_tag'], False),
    ],
)
def test_claimed_by_other_operation(tags, expected):
    assert claimed_by_other_operation(tags) is expected


# --- resolving a real node's inputs ----------------------------------------------


def _roles(operands: NodeOperands) -> dict[str, list[str]]:
    return {
        'additive': [n.id for n in operands.additive],
        'factors': [n.id for n in operands.factors],
        'impute': [n.id for n in operands.impute],
        'claimed_elsewhere': [n.id for n in operands.claimed_elsewhere],
    }


def test_untagged_inputs_are_sorted_by_unit_compatibility():
    """The default: a unit that matches the node is added, one that does not is a factor."""
    ctx = _make_context('operands-units')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    _connect(_source(ctx, 'same_unit', [(2020, 1.0)], unit='kWh'), node)
    _connect(_source(ctx, 'convertible', [(2020, 1.0)], unit='MWh'), node)
    _connect(_source(ctx, 'incompatible', [(2020, 1.0)], unit='h', quantity='duration'), node)

    assert _roles(resolve_input_nodes(node)) == {
        'additive': ['same_unit', 'convertible'],
        'factors': ['incompatible'],
        'impute': [],
        'claimed_elsewhere': [],
    }


def test_tags_override_the_unit_test():
    """A tag is a statement of intent and beats whatever the units happen to say."""
    ctx = _make_context('operands-tags')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    _connect(_source(ctx, 'same_unit_as_factor', [(2020, 1.0)], unit='kWh'), node, tags=['non_additive'])
    _connect(_source(ctx, 'other_unit_as_additive', [(2020, 1.0)], unit='h', quantity='duration'), node, tags=['additive'])

    assert _roles(resolve_input_nodes(node)) == {
        'additive': ['other_unit_as_additive'],
        'factors': ['same_unit_as_factor'],
        'impute': [],
        'claimed_elsewhere': [],
    }


def test_impute_is_its_own_role():
    ctx = _make_context('operands-impute')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    _connect(_source(ctx, 'summed', [(2020, 1.0)], unit='kWh'), node)
    _connect(_source(ctx, 'overlaid', [(2020, 1.0)], unit='kWh'), node, tags=['impute'])

    assert _roles(resolve_input_nodes(node))['additive'] == ['summed']
    assert _roles(resolve_input_nodes(node))['impute'] == ['overlaid']


def test_inputs_claimed_by_another_operation_are_reported_not_dropped():
    """
    An input tagged for another operation is neither added nor multiplied.

    The old code dropped it on the floor. Listing it lets a caller that cannot honour the
    tag say so instead of quietly ignoring an input the modeller wired up.
    """
    ctx = _make_context('operands-claimed')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    _connect(_source(ctx, 'summed', [(2020, 1.0)], unit='kWh'), node)
    _connect(_source(ctx, 'splitter', [(2020, 1.0)], unit='kWh'), node, tags=['use_as_totals'])

    roles = _roles(resolve_input_nodes(node))
    assert roles['additive'] == ['summed']
    assert roles['factors'] == []
    assert roles['claimed_elsewhere'] == ['splitter']


def test_ignore_content_inputs_are_skipped_entirely():
    ctx = _make_context('operands-ignore')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    _connect(_source(ctx, 'summed', [(2020, 1.0)], unit='kWh'), node)
    _connect(_source(ctx, 'ignored', [(2020, 1.0)], unit='kWh'), node, tags=['ignore_content'])

    roles = _roles(resolve_input_nodes(node))
    assert roles['additive'] == ['summed']
    assert roles['claimed_elsewhere'] == []


def test_excluded_ids_are_skipped():
    """What WeightedSumNode relies on: an input already consumed must not be added twice."""
    ctx = _make_context('operands-excluded')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    _connect(_source(ctx, 'summed', [(2020, 1.0)], unit='kWh'), node)
    _connect(_source(ctx, 'already_weighted', [(2020, 1.0)], unit='kWh'), node)

    roles = _roles(resolve_input_nodes(node, exclude_ids={'already_weighted'}))
    assert roles['additive'] == ['summed']


def test_input_order_is_preserved():
    ctx = _make_context('operands-order')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    for i in range(4):
        _connect(_source(ctx, f'input_{i}', [(2020, 1.0)], unit='kWh'), node)

    assert _roles(resolve_input_nodes(node))['additive'] == ['input_0', 'input_1', 'input_2', 'input_3']


def test_declared_and_computed_unit_sources_agree_here_but_are_different_questions():
    """
    ``unit_of`` picks whether the unit test reads the declared unit or the computed output.

    ``GenericNode`` reads the output (authoritative, costs a compute); ``MultiplicativeNode``
    reads the declaration (free, can be a lie). They agree for a well-formed node, and the
    parameter exists so the choice stays visible rather than baked in.
    """
    ctx = _make_context('operands-unitsource')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    _connect(_source(ctx, 'same_unit', [(2020, 1.0)], unit='kWh'), node)
    _connect(_source(ctx, 'incompatible', [(2020, 1.0)], unit='h', quantity='duration'), node)

    from_output = _roles(resolve_input_nodes(node, unit_of=output_unit_of))
    from_declaration = _roles(resolve_input_nodes(node, unit_of=declared_unit_of))
    assert from_output == from_declaration


def test_unreadable_unit_raises_unless_a_default_is_given():
    ctx = _make_context('operands-nounit')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    source = _source(ctx, 'unitless', [(2020, 1.0)], unit='kWh')
    _connect(source, node)

    def no_unit(_target, _source):
        return None

    with pytest.raises(Exception, match=r'(?i)additive or a factor'):
        resolve_input_nodes(node, unit_of=no_unit)

    roles = _roles(resolve_input_nodes(node, unit_of=no_unit, default_role='factor'))
    assert roles['factors'] == ['unitless']
