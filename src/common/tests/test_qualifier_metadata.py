"""Qualifier structs are sidecars, never numeric metrics."""

import polars as pl
import pytest

from common import polars as ppl, qualifiers
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def test_qualifier_struct_can_pass_through_wide_interpolation_input() -> None:
    frame = ppl.to_ppdf(
        pl.DataFrame({
            'Year': [2023, 2023],
            'sector': ['private_households', 'industry'],
            'Value': [10.0, 20.0],
        }),
        meta=ppl.DataFrameMeta(
            units={'Value': unit_registry.parse_units('MWh/a')},
            primary_keys=['Year', 'sector'],
        ),
    )
    qualified = frame.with_columns(
        qualifiers.make(supplied=pl.col('Value').is_not_null()).alias(qualifiers.qualifier_column('Value'))
    )

    assert qualified.metric_cols == ['Value']
    wide = qualified.paths.to_wide()
    assert wide['Value@sector:industry'][0] == 20.0
    assert wide['Value@sector:private_households'][0] == 10.0
