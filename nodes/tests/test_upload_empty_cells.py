"""
Uploading a template whose value cells are deliberately blank.

BISKO Prüfschritt 1.4 turns on being able to tell a municipality-confirmed zero from a cell
nobody has touched, which means a template has to reach DVC with its cells empty rather than
pre-filled with zeros. Two steps of the wide-format path dropped them: `clean_dataframe`
removed a year column that was entirely null, and `convert_to_standard_format` then filtered
out every null value. The long-format path (a file that already has a `Year` column) skips
both, which is why this was thought to work.
"""

import polars as pl
import pytest

from notebooks.upload_new_dataset import clean_dataframe, convert_to_standard_format

# The functions under test are pure, but importing the uploader pulls in Django models.
pytestmark = pytest.mark.django_db

TEMPLATE = pl.DataFrame(
    {
        'Dataset': ['t', 't', 't'],
        'vehicle_type': ['passenger_cars', 'passenger_cars', 'trucks'],
        'road_type': ['highway', 'urban_roads', 'highway'],
        '2023': [None, None, None],
    },
    schema_overrides={'2023': pl.Float64},
)

SPARSE = pl.DataFrame({
    'Dataset': ['t', 't'],
    'vehicle_type': ['passenger_cars', 'trucks'],
    '2022': [1.0, None],
    '2023': [2.0, 3.0],
})


def test_an_all_blank_template_uploads_as_rows_with_no_value():
    cleaned = clean_dataframe(TEMPLATE, keep_empty_cells=True)
    assert '2023' in cleaned.columns, 'the empty year column is the template; it must survive'

    out = convert_to_standard_format(cleaned, keep_empty_cells=True)

    assert len(out) == 3, 'every prompted combination must reach DVC'
    assert out['Value'].null_count() == 3, 'and each must arrive with no value, not a zero'


def test_without_the_flag_an_all_blank_template_is_not_uploadable():
    """The old behaviour, kept as the default: it is what a sparse data file needs."""
    cleaned = clean_dataframe(TEMPLATE, keep_empty_cells=False)
    assert '2023' not in cleaned.columns

    with pytest.raises(ValueError, match='No year columns found'):
        convert_to_standard_format(cleaned, keep_empty_cells=False)


def test_a_sparse_data_file_is_unchanged_by_default():
    """
    The flag must stay opt-in.

    Dropping blank cells is what keeps a sparse wide file from becoming a dense one; turning
    that off for every upload would materialise every absent combination in every existing
    dataset.
    """
    out = convert_to_standard_format(clean_dataframe(SPARSE), keep_empty_cells=False)

    assert len(out) == 3, 'the 2022 blank for trucks should be dropped'
    assert out['Value'].null_count() == 0


def test_the_same_sparse_file_keeps_its_holes_when_asked():
    out = convert_to_standard_format(clean_dataframe(SPARSE, keep_empty_cells=True), keep_empty_cells=True)

    assert len(out) == 4
    assert out['Value'].null_count() == 1


def test_placeholder_markers_are_still_dropped_either_way():
    """`.` and `-` mean "no data" in a source file; they are not values and never were."""
    df = pl.DataFrame({'Dataset': ['t', 't'], 'sector': ['a', 'b'], '2023': ['.', '5']})

    for keep in (False, True):
        out = convert_to_standard_format(clean_dataframe(df, keep_empty_cells=keep), keep_empty_cells=keep)
        assert out['Value'].to_list() == ['5'], f'keep_empty_cells={keep}'
