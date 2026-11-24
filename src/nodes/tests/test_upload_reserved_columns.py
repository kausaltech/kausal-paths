"""
Provenance columns are never read as dimensions by the uploader.

With `--instance`, the uploader converts category names to ids in every column named after one
of the instance's dimensions. An instance with a dimension called `source` (an emission factor's
publisher, say) then took every dataset's `Source` provenance column for that dimension and
failed on the first source name, so no dataset of that instance could be uploaded with sources.
"""

from typing import TYPE_CHECKING, cast

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from nodes.dimensions import Dimension, DimensionCategory
from tools.upload_new_dataset import convert_names_to_cats

if TYPE_CHECKING:
    from nodes.context import Context

# The function under test is pure, but importing the uploader pulls in Django models.
pytestmark = pytest.mark.django_db


class _Context:
    def __init__(self, *dims: Dimension):
        self.dimensions = {d.id: d for d in dims}


def _dimension(dim_id: str, *labels: str) -> Dimension:
    return Dimension(
        id=dim_id,
        label=TranslatedString(dim_id, default_language='en'),
        categories=[
            DimensionCategory(id=label.lower().replace(' ', '_'), label=TranslatedString(label, default_language='en'))
            for label in labels
        ],
    )


CONTEXT = cast('Context', _Context(_dimension('source', 'EPA', 'ICLEI'), _dimension('vehicle_type', 'Passenger cars')))


def test_a_source_column_is_not_converted_as_the_source_dimension():
    df = pl.DataFrame({
        'Vehicle_type': ['Passenger cars'],
        'Source': ['Longmont 2024 GHG Inventory; Kausal model constants'],
        'Comment': ['A comment'],
        'Value': [1.0],
    })

    out = convert_names_to_cats(df, CONTEXT)

    assert out['vehicle_type'].cast(pl.Utf8).to_list() == ['passenger_cars'], 'real dimensions are still converted'
    assert out['Source'].to_list() == ['Longmont 2024 GHG Inventory; Kausal model constants']
    assert out['Comment'].to_list() == ['A comment']


def test_without_a_reserved_column_other_dimensions_convert_as_before():
    df = pl.DataFrame({'vehicle_type': ['Passenger cars'], 'Value': [1.0]})

    out = convert_names_to_cats(df, CONTEXT)

    assert out['vehicle_type'].cast(pl.Utf8).to_list() == ['passenger_cars']
