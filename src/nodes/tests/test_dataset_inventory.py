"""
The inventory's verdict columns, which are pure functions of the two sides' counts.

`where` and `drift` decide what an operator looks at first, and both are read as facts
about the data rather than about how it was counted -- so the counting rule they rest on
is pinned here.
"""

from __future__ import annotations

import pytest

from nodes.management.commands.dataset_inventory import Row

# The counts below are literals, but this package's fixtures reach for the DB on setup.
pytestmark = pytest.mark.django_db


def test_equal_cell_counts_are_no_drift_however_empty_the_data_is():
    """
    Drift must mean a difference in shape, not a difference in fullness.

    Both sides count every (row, metric) cell, because an import creates a null-valued
    DataPoint for an empty one. Counting values on the DVC side and cells on the DB side
    reported drift on every dataset with a gap in it.
    """
    row = Row(identifier='de/x', db_points=8262, db_valued=2822, dvc_points=8262, dvc_valued=2822)

    assert row.drift == 0
    assert row.where == 'both'


def test_a_template_against_a_filled_in_row_is_called_out():
    """Same shape, same counts, opposite content: the one case the numbers alone hide."""
    row = Row(identifier='kommune/x', db_points=476, db_valued=476, dvc_points=476, dvc_valued=0)

    assert row.drift == 0, 'the grids match, which is exactly why this needs saying'
    assert row.is_template


def test_an_empty_row_under_an_empty_dvc_copy_is_not_a_template():
    """Nothing has been filled in, so there is nothing an import could blank."""
    row = Row(identifier='kommune/x', db_points=476, db_valued=0, dvc_points=476, dvc_valued=0)

    assert not row.is_template


def test_real_data_on_both_sides_is_not_a_template():
    row = Row(identifier='de/x', db_points=10, db_valued=10, dvc_points=10, dvc_valued=10)

    assert not row.is_template
