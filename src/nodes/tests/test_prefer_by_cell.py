"""
Tests for ``prefer_by_cell`` and ``split_categories``.

``prefer_by_cell`` models an input form whose fields come pre-filled with a default: a value the
city enters overwrites that one field, and every field it leaves alone keeps its default. The
properties that matter are about which cells move, not about arithmetic.

``split_categories`` turns a measured fuel figure that includes a blend (diesel with its
biodiesel share) into the fuels the balance reports, in the proportions of a reference.
"""

from typing import TYPE_CHECKING

import polars as pl
import pytest

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.units import unit_registry

if TYPE_CHECKING:
    from common import polars as ppl

pytestmark = pytest.mark.django_db

Row = tuple[int, str, str, float | None]


def _frame(rows: list[Row], unit: str = 'MWh/a', forecast: bool = False) -> ppl.PathsDataFrame:
    df = pl.DataFrame(
        {
            YEAR_COLUMN: [r[0] for r in rows],
            'vehicle_type': [r[1] for r in rows],
            'energy_carrier': [r[2] for r in rows],
            VALUE_COLUMN: [r[3] for r in rows],
            FORECAST_COLUMN: [forecast] * len(rows),
        },
        schema={
            YEAR_COLUMN: pl.Int64,
            'vehicle_type': pl.Utf8,
            'energy_carrier': pl.Utf8,
            VALUE_COLUMN: pl.Float64,
            FORECAST_COLUMN: pl.Boolean,
        },
    )
    meta = DataFrameMeta(
        units={VALUE_COLUMN: unit_registry.parse_units(unit)},
        primary_keys=[YEAR_COLUMN, 'vehicle_type', 'energy_carrier'],
    )
    return to_ppdf(df, meta)


def _coverage(rows: list[tuple[int, str, float]]) -> ppl.PathsDataFrame:
    df = pl.DataFrame(
        {YEAR_COLUMN: [r[0] for r in rows], 'vehicle_type': [r[1] for r in rows], VALUE_COLUMN: [r[2] for r in rows]},
        schema={YEAR_COLUMN: pl.Int64, 'vehicle_type': pl.Utf8, VALUE_COLUMN: pl.Float64},
    )
    meta = DataFrameMeta(
        units={VALUE_COLUMN: unit_registry.parse_units('dimensionless')}, primary_keys=[YEAR_COLUMN, 'vehicle_type']
    )
    return to_ppdf(df, meta)


def _as_map(df: ppl.PathsDataFrame) -> dict[tuple[int, str, str], float]:
    return {
        (row[YEAR_COLUMN], str(row['vehicle_type']), str(row['energy_carrier'])): row[VALUE_COLUMN]
        for row in df.iter_rows(named=True)
    }


DEFAULT = _frame([
    (2020, 'cars', 'diesel', 100.0),
    (2020, 'cars', 'petrol', 200.0),
    (2020, 'buses', 'diesel', 50.0),
    (2021, 'cars', 'diesel', 110.0),
    (2021, 'buses', 'diesel', 55.0),
])


def test_one_entered_cell_replaces_only_that_cell():
    """The Prüfstadt case: one own value must not switch the rest of the year off."""
    own = _frame([(2020, 'buses', 'diesel', 7.0)])
    coverage = _frame([(2020, 'buses', 'diesel', 1.0)], unit='dimensionless')

    out = own.paths.prefer_by_cell(DEFAULT, coverage)

    assert _as_map(out) == {
        (2020, 'cars', 'diesel'): 100.0,
        (2020, 'cars', 'petrol'): 200.0,
        (2020, 'buses', 'diesel'): 7.0,
        (2021, 'cars', 'diesel'): 110.0,
        (2021, 'buses', 'diesel'): 55.0,
    }


def test_an_entered_zero_wins_over_the_default():
    """Zero is an entry: the protocol says a mandatory field may be zero."""
    own = _frame([(2021, 'cars', 'diesel', 0.0)])
    coverage = _frame([(2021, 'cars', 'diesel', 1.0)], unit='dimensionless')

    out = own.paths.prefer_by_cell(DEFAULT, coverage)

    assert _as_map(out)[2021, 'cars', 'diesel'] == 0.0


def test_a_filled_zero_without_coverage_does_not_win():
    """An own frame zero-filled to compute at all must not silence the default."""
    own = _frame([(2020, 'cars', 'diesel', 0.0), (2020, 'cars', 'petrol', 0.0)])
    coverage = _frame([(2020, 'cars', 'diesel', 0.0), (2020, 'cars', 'petrol', 0.0)], unit='dimensionless')

    out = own.paths.prefer_by_cell(DEFAULT, coverage)

    assert _as_map(out) == _as_map(DEFAULT)


def test_coarser_coverage_marks_a_whole_block():
    own = _frame([(2020, 'cars', 'diesel', 1.0), (2020, 'cars', 'petrol', 2.0)])
    coverage = _coverage([(2020, 'cars', 1.0)])

    out = own.paths.prefer_by_cell(DEFAULT, coverage)

    assert _as_map(out)[2020, 'cars', 'diesel'] == 1.0
    assert _as_map(out)[2020, 'cars', 'petrol'] == 2.0
    assert _as_map(out)[2020, 'buses', 'diesel'] == 50.0


def test_forecast_flag_follows_the_value():
    own = _frame([(2020, 'buses', 'diesel', 7.0)], forecast=False)
    default = _frame([(2020, 'buses', 'diesel', 50.0), (2020, 'cars', 'diesel', 100.0)], forecast=True)
    coverage = _frame([(2020, 'buses', 'diesel', 1.0)], unit='dimensionless')

    out = own.paths.prefer_by_cell(default, coverage)
    flags = {(r['vehicle_type'], r['energy_carrier']): r[FORECAST_COLUMN] for r in out.iter_rows(named=True)}

    assert flags == {('buses', 'diesel'): False, ('cars', 'diesel'): True}


def test_coverage_with_a_dimension_the_data_lacks_is_refused():
    own = _frame([(2020, 'cars', 'diesel', 1.0)])
    df = pl.DataFrame(
        {YEAR_COLUMN: [2020], 'road_type': ['urban'], VALUE_COLUMN: [1.0]},
        schema={YEAR_COLUMN: pl.Int64, 'road_type': pl.Utf8, VALUE_COLUMN: pl.Float64},
    )
    coverage = to_ppdf(
        df,
        DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('dimensionless')}, primary_keys=[YEAR_COLUMN, 'road_type']),
    )

    with pytest.raises(ValueError, match='road_type'):
        own.paths.prefer_by_cell(DEFAULT, coverage)


BLENDS = {'diesel': ['diesel', 'biodiesel'], 'petrol': ['petrol', 'bio_petrol']}


def test_measured_diesel_is_split_in_the_reference_blend():
    measured = _frame([(2020, 'cars', 'diesel', 1000.0), (2020, 'cars', 'electricity', 30.0)])
    reference = _frame([
        (2020, 'cars', 'diesel', 93.0),
        (2020, 'cars', 'biodiesel', 7.0),
        (2020, 'cars', 'electricity', 5.0),
    ])

    out = measured.paths.split_categories(reference, 'energy_carrier', BLENDS)

    assert _as_map(out) == pytest.approx({
        (2020, 'cars', 'diesel'): 930.0,
        (2020, 'cars', 'biodiesel'): 70.0,
        (2020, 'cars', 'electricity'): 30.0,
    })


def test_the_split_keeps_the_total():
    measured = _frame([(2020, 'cars', 'diesel', 1000.0), (2020, 'cars', 'petrol', 500.0)])
    reference = _frame([
        (2020, 'cars', 'diesel', 80.0),
        (2020, 'cars', 'biodiesel', 20.0),
        (2020, 'cars', 'petrol', 90.0),
        (2020, 'cars', 'bio_petrol', 10.0),
    ])

    out = measured.paths.split_categories(reference, 'energy_carrier', BLENDS)

    assert sum(_as_map(out).values()) == pytest.approx(1500.0)


def test_without_reference_proportions_the_value_stays_on_the_fossil_member():
    """A vehicle class the reference never fuels with diesel must not lose the measured diesel."""
    measured = _frame([(2020, 'buses', 'diesel', 40.0), (2021, 'buses', 'diesel', 41.0)])
    reference = _frame([(2020, 'buses', 'diesel', 0.0), (2020, 'buses', 'biodiesel', 0.0)])

    out = measured.paths.split_categories(reference, 'energy_carrier', BLENDS)

    values = _as_map(out)
    assert values[2020, 'buses', 'diesel'] == 40.0
    assert values[2021, 'buses', 'diesel'] == 41.0
    assert values.get((2020, 'buses', 'biodiesel'), 0.0) == 0.0


def test_a_reference_without_the_fuel_dimension_splits_every_fuel_alike():
    """Energy by vehicle class and fuel, split over road categories by mileage that has no fuel."""
    energy = pl.DataFrame(
        {
            YEAR_COLUMN: [2020, 2020, 2020, 2020],
            'vehicle_type': ['cars', 'cars', 'buses', 'vans'],
            'road_type': ['all_road_types'] * 4,
            'energy_carrier': ['diesel', 'petrol', 'diesel', 'diesel'],
            VALUE_COLUMN: [100.0, 50.0, 30.0, 20.0],
        },
    )
    energy_ppdf = to_ppdf(
        energy,
        DataFrameMeta(
            units={VALUE_COLUMN: unit_registry.parse_units('MWh/a')},
            primary_keys=[YEAR_COLUMN, 'vehicle_type', 'road_type', 'energy_carrier'],
        ),
    )
    mileage = pl.DataFrame(
        {
            YEAR_COLUMN: [2020] * 5,
            'vehicle_type': ['cars', 'cars', 'buses', 'vans', 'vans'],
            'road_type': ['highway', 'urban_roads', 'all_road_types', 'highway', 'urban_roads'],
            VALUE_COLUMN: [3.0, 1.0, 0.0, 0.0, 0.0],
        },
    )
    mileage_ppdf = to_ppdf(
        mileage,
        DataFrameMeta(
            units={VALUE_COLUMN: unit_registry.parse_units('Mvkm/a')}, primary_keys=[YEAR_COLUMN, 'vehicle_type', 'road_type']
        ),
    )
    groups = {'all_road_types': ['all_road_types', 'highway', 'rural_roads', 'urban_roads']}

    out = energy_ppdf.paths.split_categories(mileage_ppdf, 'road_type', groups)
    values = {
        (r['vehicle_type'], r['road_type'], r['energy_carrier']): r[VALUE_COLUMN]
        for r in out.iter_rows(named=True)
        if r[VALUE_COLUMN]
    }

    # Buses and vans have no mileage this year: their measured energy stays, on all road types,
    # even for vans, whose mileage rows name only the specific road categories.
    assert values == pytest.approx({
        ('cars', 'highway', 'diesel'): 75.0,
        ('cars', 'urban_roads', 'diesel'): 25.0,
        ('cars', 'highway', 'petrol'): 37.5,
        ('cars', 'urban_roads', 'petrol'): 12.5,
        ('buses', 'all_road_types', 'diesel'): 30.0,
        ('vans', 'all_road_types', 'diesel'): 20.0,
    })


def test_an_aggregate_whose_members_are_entered_is_not_split_again():
    """A city entering diesel and biodiesel separately has entered fossil diesel."""
    measured = _frame([
        (2020, 'cars', 'diesel', 930.0),
        (2020, 'cars', 'biodiesel', 70.0),
        (2020, 'buses', 'diesel', 100.0),
    ])
    reference = _frame([
        (2020, 'cars', 'diesel', 90.0),
        (2020, 'cars', 'biodiesel', 10.0),
        (2020, 'buses', 'diesel', 90.0),
        (2020, 'buses', 'biodiesel', 10.0),
    ])

    out = measured.paths.split_categories(reference, 'energy_carrier', BLENDS)

    assert _as_map(out) == pytest.approx({
        (2020, 'cars', 'diesel'): 930.0,
        (2020, 'cars', 'biodiesel'): 70.0,
        (2020, 'buses', 'diesel'): 90.0,
        (2020, 'buses', 'biodiesel'): 10.0,
    })


# --- prod_dim carries units ----------------------------------------------------------------------


def _by_part(values: list[float], unit: str, parts: list[str] | None = None) -> ppl.PathsDataFrame:
    parts = parts or [f'p{i}' for i in range(len(values))]
    df = pl.DataFrame(
        {YEAR_COLUMN: [2020] * len(values), 'part': parts, VALUE_COLUMN: values},
        schema={YEAR_COLUMN: pl.Int64, 'part': pl.Utf8, VALUE_COLUMN: pl.Float64},
    )
    return to_ppdf(df, DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=[YEAR_COLUMN, 'part']))


def _quantity(df: ppl.PathsDataFrame, unit: str) -> float:
    return (df[VALUE_COLUMN][0] * df.get_unit(VALUE_COLUMN)).to(unit).m


def test_prod_dim_of_percentages_is_a_percentage_squared():
    out = _by_part([10.0, 10.0], '%').paths.prod_over_dims('part')

    assert _quantity(out, 'dimensionless') == pytest.approx(0.01)


def test_prod_dim_of_lengths_is_a_volume():
    out = _by_part([1.0, 2.0, 3.0], 'm').paths.prod_over_dims('part')

    assert _quantity(out, 'm**3') == pytest.approx(6.0)


def test_prod_dim_of_flags_stays_dimensionless():
    out = _by_part([1.0, 0.0, 1.0], 'dimensionless').paths.prod_over_dims('part')

    assert out.get_unit(VALUE_COLUMN).dimensionless
    assert out[VALUE_COLUMN][0] == 0.0


def test_prod_dim_with_uneven_groups_converts_percentages_and_refuses_lengths():
    """Groups of different sizes have no common unit power; a fraction can still be converted."""
    df = pl.DataFrame(
        {
            YEAR_COLUMN: [2020, 2020, 2020],
            'group': ['a', 'a', 'b'],
            'part': ['x', 'y', 'x'],
            VALUE_COLUMN: [50.0, 50.0, 50.0],
        },
    )

    def frame(unit: str) -> ppl.PathsDataFrame:
        return to_ppdf(
            df,
            DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=[YEAR_COLUMN, 'group', 'part']),
        )

    out = frame('%').paths.prod_over_dims('part')
    assert out.get_unit(VALUE_COLUMN).dimensionless
    assert sorted(out[VALUE_COLUMN].to_list()) == pytest.approx([0.25, 0.5])

    with pytest.raises(ValueError, match='no single unit'):
        frame('m').paths.prod_over_dims('part')
