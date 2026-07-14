import pytest

from nodes.explanations import BasketRule

pytestmark = pytest.mark.django_db


def test_apply_term_functions_nests_in_execution_order():
    """Tags applied in list order must nest with the last-applied function outermost."""
    rule = BasketRule()
    terms = [{'var_names': ['observed'], 'functions': ['inventory_only', 'extend_values']}]

    result = rule._apply_term_functions('observed / modelled', terms)

    assert result == 'extend_values(inventory_only(observed)) / modelled'


def test_apply_term_functions_single_function():
    rule = BasketRule()
    terms = [{'var_names': ['observed'], 'functions': ['geometric_inverse']}]

    result = rule._apply_term_functions('observed * factor', terms)

    assert result == 'geometric_inverse(observed) * factor'


def test_apply_term_functions_no_functions_is_noop():
    rule = BasketRule()
    terms = [{'var_names': ['observed'], 'functions': []}]

    result = rule._apply_term_functions('observed / modelled', terms)

    assert result == 'observed / modelled'
