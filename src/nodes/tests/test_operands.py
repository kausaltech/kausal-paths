"""
The rule that decides what each of a node's inputs is for.

``nodes.operands`` is the one copy of a rule that used to be written out in four places.
These tests pin the rule itself; ``test_add_multiply_semantics.py`` pins what the node
classes then do with the answer.
"""

import pytest

from nodes.constants import VALUE_COLUMN
from nodes.operands import (
    NodeOperands,
    claimed_by_other_operation,
    declared_unit_of,
    output_unit_of,
    resolve_input_nodes,
    role_from_tags,
)
from nodes.tests.test_add_multiply_semantics import (
    _additive,
    _connect,
    _generic,
    _make_context,
    _source,
)

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


def test_reference_edges_are_skipped_entirely():
    ctx = _make_context('operands-reference')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    _connect(_source(ctx, 'summed', [(2020, 1.0)], unit='kWh'), node)
    _connect(_source(ctx, 'referenced', [(2020, 1.0)], unit='kWh'), node, tags=['reference'])

    roles = _roles(resolve_input_nodes(node))
    assert roles['additive'] == ['summed']
    assert roles['claimed_elsewhere'] == []


def test_reference_edge_is_dropped_from_the_sum_not_zeroed():
    """
    The difference from ``ignore_content``: the input is absent, not present as a zero.

    A zero is a no-op for addition only. The old tag rewrote the frame to the target's
    no-effect value and let it be summed, which made addition load-bearing for something
    unrelated to it — and left a *factor* multiplied by zero.
    """
    ctx = _make_context('operands-reference-sum')
    plain = _generic(ctx, 'plain', 'add', unit='kWh')
    _connect(_source(ctx, 'data_a', [(2020, 3.0)], unit='kWh'), plain)

    annotated = _generic(ctx, 'annotated', 'add', unit='kWh')
    _connect(_source(ctx, 'data_b', [(2020, 3.0)], unit='kWh'), annotated)
    _connect(_source(ctx, 'not_yet_known', [(2020, 7.0)], unit='kWh'), annotated, tags=['reference'])

    assert annotated.get_output_pl()[VALUE_COLUMN].to_list() == plain.get_output_pl()[VALUE_COLUMN].to_list()


def test_reference_edge_need_not_match_unit_or_dimensions():
    """
    No shape rule names the reference role, so the link asserts nothing about shape.

    A unit that would be sorted into the *factors* bucket, and a missing dimension that
    would fail the target's dimension test, both pass without comment.
    """
    ctx = _make_context('operands-reference-dims')
    plain = _generic(ctx, 'plain', 'add', unit='kWh', dims=['sector'])
    _connect(_source(ctx, 'data_a', [(2020, 'x', 1.0)], unit='kWh', dim='sector'), plain)

    annotated = _generic(ctx, 'annotated', 'add', unit='kWh', dims=['sector'])
    _connect(_source(ctx, 'data_b', [(2020, 'x', 1.0)], unit='kWh', dim='sector'), annotated)
    _connect(_source(ctx, 'unquantified', [(2020, 5.0)], unit='pcs', quantity='number'), annotated, tags=['reference'])

    expected = plain.get_output_pl()
    got = annotated.get_output_pl()
    assert got.dim_ids == expected.dim_ids
    assert got[VALUE_COLUMN].to_list() == expected[VALUE_COLUMN].to_list()


def test_retired_ignore_content_tag_no_longer_marks_a_reference():
    """
    ``ignore_content`` is retired, and this pins that on purpose.

    It was accepted as an alias only while database-sourced mirrors still held it in their
    stored specs; those have been re-synced, so the name now means nothing and such an edge
    is an ordinary input, sorted by unit like any other. That is a real hazard for a mirror
    nobody re-synced — which is exactly why it is asserted here rather than left implicit:
    re-introducing the alias should be a deliberate act with a failing test to change.
    """
    ctx = _make_context('operands-retired-tag')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    _connect(_source(ctx, 'summed', [(2020, 1.0)], unit='kWh'), node)
    _connect(_source(ctx, 'stale_tag', [(2020, 1.0)], unit='kWh'), node, tags=['ignore_content'])

    roles = _roles(resolve_input_nodes(node))
    assert roles['additive'] == ['summed', 'stale_tag']


def test_reference_role_binding_is_ignored_by_the_additive_port():
    """A migrated class never sees the binding: no operation resolves the reference role."""
    ctx = _make_context('operands-reference-port')
    plain = _additive(ctx, 'plain', unit='kWh')
    _connect(_source(ctx, 'data_a', [(2020, 3.0)], unit='kWh'), plain)

    annotated = _additive(ctx, 'annotated', unit='kWh')
    _connect(_source(ctx, 'data_b', [(2020, 3.0)], unit='kWh'), annotated)
    _connect(_source(ctx, 'not_yet_known', [(2020, 7.0)], unit='pcs', quantity='number'), annotated, tags=['reference'])

    assert [b.port_role for b in annotated.runtime_input_bindings] == ['additive', 'reference']
    assert annotated.get_output_pl()[VALUE_COLUMN].to_list() == plain.get_output_pl()[VALUE_COLUMN].to_list()


def test_reference_port_is_declared_on_every_node_class():
    """The role is universal, so any class can hold a half-constructed edge."""
    from nodes.constants import REFERENCE_ROLE
    from nodes.formula import FormulaNode
    from nodes.generic import GenericNode
    from nodes.node import Node
    from nodes.simple import AdditiveNode, MultiplicativeNode

    for cls in (Node, GenericNode, FormulaNode, AdditiveNode, MultiplicativeNode):
        roles = [declaration.role for declaration in cls.input_port_declarations]
        assert REFERENCE_ROLE in roles, cls.__name__
        assert roles.count(REFERENCE_ROLE) == 1, cls.__name__

    # It must not enter the computational set a migrated class declares, and an empty
    # `declared_input_ports` is still what marks a class as unmigrated.
    assert REFERENCE_ROLE not in [d.role for d in AdditiveNode.declared_input_ports]
    assert GenericNode.declared_input_ports == ()

    # Never auto-selected by a plain connect, and never instantiated at node creation.
    assert Node.reference_port.min_count == 0
    assert Node.reference_port.effective_default_count == 0
    assert Node.reference_port.required is False
    assert Node.reference_port.multi is True


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


# --- argument nodes: in the graph, out of the arithmetic -------------------------


def test_argument_node_is_never_an_operand():
    """``quantity: argument`` excludes an input from every bucket, tagged or not."""
    ctx = _make_context('operands-argument')
    node = _generic(ctx, 'target', 'add', unit='kWh')
    _connect(_source(ctx, 'real', [(2020, 1.0)], unit='kWh'), node)
    _connect(_source(ctx, 'bare_arg', [(2020, 5.0)], unit='dimensionless', quantity='argument'), node)
    # Tags must not rescue it into the arithmetic either.
    _connect(
        _source(ctx, 'tagged_arg', [(2020, 5.0)], unit='kWh', quantity='argument'),
        node,
        tags=['additive'],
    )

    assert _roles(resolve_input_nodes(node)) == {
        'additive': ['real'],
        'factors': [],
        'impute': [],
        'claimed_elsewhere': [],
    }


def test_argument_node_does_not_have_to_match_dimensions():
    """
    A dimensionless argument node attaches to a *dimensioned* additive node.

    This is the case the old ``ignore_content`` tag could not express: it rewrote the
    frame's values but left it dimensionless, so the target's dimension check rejected it
    and every node downstream of the target failed with it.
    """
    ctx = _make_context('operands-argument-dims')
    plain = _additive(ctx, 'plain', unit='kWh', dims=['sector'])
    _connect(_source(ctx, 'data_a', [(2020, 'x', 1.0)], unit='kWh', dim='sector'), plain)

    annotated = _additive(ctx, 'annotated', unit='kWh', dims=['sector'])
    _connect(_source(ctx, 'data_b', [(2020, 'x', 1.0)], unit='kWh', dim='sector'), annotated)
    _connect(
        _source(ctx, 'objection', [(2020, 0.0)], unit='dimensionless', quantity='argument'),
        annotated,
    )

    expected = plain.get_output_pl()
    got = annotated.get_output_pl()
    assert got.dim_ids == expected.dim_ids
    assert got[VALUE_COLUMN].to_list() == expected[VALUE_COLUMN].to_list()
