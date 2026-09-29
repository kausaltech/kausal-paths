"""
Qualifier columns: what the frame operations do to the grade and provenance of a value.

The properties that matter are the ones a Datengüte and a source choice depend on: a sum's grade
is its components' grades weighted by magnitude, an ungraded part lowers the graded share
rather than disappearing, a factor with no grade leaves the activity's grade alone, and a value
an operation made up is marked as such.
"""

from typing import TYPE_CHECKING

import polars as pl
import pytest

from common import qualifiers
from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.transforms import QualifierSource
from nodes.units import unit_registry

if TYPE_CHECKING:
    from common import polars as ppl

pytestmark = pytest.mark.django_db

QUAL = qualifiers.qualifier_column(VALUE_COLUMN)

type Row = tuple[int, str, float | None, float | None, bool | None]
"""Year, energy carrier, value, quality, supplied; a quality of None is ungraded."""


def _frame(rows: list[Row], unit: str = 'MWh/a', *, qualified: bool = True) -> ppl.PathsDataFrame:
    df = pl.DataFrame(
        {
            YEAR_COLUMN: [r[0] for r in rows],
            'energy_carrier': [r[1] for r in rows],
            VALUE_COLUMN: [r[2] for r in rows],
            FORECAST_COLUMN: [False] * len(rows),
            '_q': [r[3] for r in rows],
            '_s': [r[4] for r in rows],
        },
        schema={
            YEAR_COLUMN: pl.Int64,
            'energy_carrier': pl.Utf8,
            VALUE_COLUMN: pl.Float64,
            FORECAST_COLUMN: pl.Boolean,
            '_q': pl.Float64,
            '_s': pl.Boolean,
        },
    )
    if qualified:
        df = df.with_columns(qualifiers.make(quality=pl.col('_q'), supplied=pl.col('_s')).alias(QUAL))
    df = df.drop('_q', '_s')
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=[YEAR_COLUMN, 'energy_carrier'])
    return to_ppdf(df, meta)


def _qual(df: ppl.PathsDataFrame) -> dict[tuple[int, ...], dict[str, object]]:
    keys = [col for col in df.primary_keys if col != YEAR_COLUMN]
    return {
        (row[YEAR_COLUMN], *(row[k] for k in keys)): row[QUAL]
        for row in df.select([YEAR_COLUMN, *keys, QUAL]).sort([YEAR_COLUMN, *keys]).to_dicts()
    }


class TestPairing:
    def test_rename_select_and_drop_carry_the_qualifier(self) -> None:
        df = _frame([(2020, 'gas', 1.0, 1.0, True)])
        assert df.qualifier_cols == {VALUE_COLUMN: QUAL}
        assert qualifiers.qualifier_column('Energy') in df.rename({VALUE_COLUMN: 'Energy'}).columns
        assert QUAL not in df.drop(VALUE_COLUMN).columns

    def test_a_projection_keeps_a_qualifier_only_when_asked(self) -> None:
        """Frames built to be stacked must come out of a projection with the columns it names."""
        df = _frame([(2020, 'gas', 1.0, 1.0, True)])
        cols = [YEAR_COLUMN, 'energy_carrier', VALUE_COLUMN]
        assert df.select(cols).columns == cols
        assert df.select(df.qualified(cols)).columns == [*cols, QUAL]


class TestSums:
    def test_the_grade_of_a_sum_is_weighted_by_magnitude(self) -> None:
        """Methodenpapier §3.4: each component's grade weighted by its share."""
        df = _frame([(2020, 'gas', 3.0, 1.0, True), (2020, 'oil', 1.0, 0.0, True)])
        out = df.paths.sum_over_dims('energy_carrier')
        assert out[QUAL].to_list() == [{'quality': 0.75, 'graded': 1.0, 'supplied': True}]

    def test_an_ungraded_part_lowers_the_graded_share_not_the_grade(self) -> None:
        df = _frame([(2020, 'gas', 3.0, 1.0, True), (2020, 'oil', 1.0, 0.0, True), (2020, 'coal', 4.0, None, True)])
        out = df.paths.sum_over_dims('energy_carrier')
        assert out[QUAL].to_list() == [{'quality': 0.75, 'graded': 0.5, 'supplied': True}]

    def test_zeros_have_nothing_to_weight_by_and_fall_back_to_a_plain_mean(self) -> None:
        df = _frame([(2020, 'gas', 0.0, 1.0, False), (2020, 'oil', 0.0, 0.5, False)])
        out = df.paths.sum_over_dims('energy_carrier')
        assert out[QUAL].to_list() == [{'quality': 0.75, 'graded': 1.0, 'supplied': False}]

    def test_adding_frames_weights_the_same_way(self) -> None:
        left = _frame([(2020, 'gas', 3.0, 1.0, True)])
        right = _frame([(2020, 'gas', 1.0, 0.0, False), (2021, 'gas', 2.0, 0.5, True)])
        out = left.paths.add_with_dims(right)
        assert _qual(out) == {
            (2020, 'gas'): {'quality': 0.75, 'graded': 1.0, 'supplied': True},
            (2021, 'gas'): {'quality': 0.5, 'graded': 1.0, 'supplied': True},
        }

    @pytest.mark.parametrize('qualified_side', ['left', 'right'])
    def test_an_unqualified_addend_counts_as_ungraded(self, qualified_side: str) -> None:
        """Either way round: a join leaves the right-hand qualifier unsuffixed when the left has none."""
        graded = _frame([(2020, 'gas', 1.0, 1.0, True)])
        ungraded = _frame([(2020, 'gas', 3.0, None, None)], qualified=False)
        left, right = (graded, ungraded) if qualified_side == 'left' else (ungraded, graded)
        out = left.paths.add_with_dims(right)
        assert _qual(out) == {(2020, 'gas'): {'quality': 1.0, 'graded': 0.25, 'supplied': True}}


class TestProducts:
    def test_a_factor_without_a_grade_leaves_the_activity_grade_alone(self) -> None:
        activity = _frame([(2020, 'gas', 10.0, 0.5, True)], unit='Mvkm/a')
        factor = _frame([(2020, 'gas', 0.3, None, None)], unit='MWh/vkm', qualified=False)
        out = activity.paths.multiply_with_dims(factor)
        assert _qual(out) == {(2020, 'gas'): {'quality': 0.5, 'graded': 1.0, 'supplied': True}}

    def test_a_graded_factor_on_the_right_keeps_its_grade(self) -> None:
        share = _frame([(2020, 'gas', 0.5, None, None)], unit='dimensionless', qualified=False)
        activity = _frame([(2020, 'gas', 10.0, 0.5, True)], unit='Mvkm/a')
        out = share.paths.multiply_with_dims(activity)
        assert _qual(out) == {(2020, 'gas'): {'quality': 0.5, 'graded': 1.0, 'supplied': True}}

    def test_two_graded_factors_take_the_lower_grade(self) -> None:
        left = _frame([(2020, 'gas', 10.0, 1.0, True)])
        right = _frame([(2020, 'gas', 2.0, 0.25, False)], unit='dimensionless')
        out = left.paths.multiply_with_dims(right)
        assert _qual(out) == {(2020, 'gas'): {'quality': 0.25, 'graded': 1.0, 'supplied': False}}


class TestFills:
    def test_empty_to_zero_marks_what_it_filled(self) -> None:
        df = _frame([(2020, 'gas', 5.0, None, None), (2021, 'oil', None, None, None)], qualified=False)
        out = df.paths.get_operation('empty_to_zero')(df, None)
        supplied = {key: qual['supplied'] for key, qual in _qual(out).items()}
        assert supplied == {(2020, 'gas'): True, (2020, 'oil'): False, (2021, 'gas'): False, (2021, 'oil'): False}

    def test_other_fills_keep_a_record_but_do_not_start_one(self) -> None:
        df = _frame([(2020, 'gas', 5.0, None, None)], qualified=False)
        assert qualifiers.carry_over(df, df) is df


class TestChoosingASource:
    def test_a_zero_filled_template_covers_no_year(self) -> None:
        """The zero a fill wrote is not a zero the city reported, so the default stands."""
        template = _frame([(2021, 'gas', None, None, None)], qualified=False)
        own = template.paths.get_operation('empty_to_zero')(template, None)
        default = _frame([(2021, 'gas', 7.0, 0.5, True), (2022, 'gas', 8.0, 0.5, True)])
        out = own.paths.prefer_by_year(default)
        assert dict(zip(out[YEAR_COLUMN], out[VALUE_COLUMN], strict=True)) == {2021: 7.0, 2022: 8.0}

    def test_the_qualifier_follows_the_chosen_source(self) -> None:
        own = _frame([(2022, 'gas', 10.0, 1.0, True), (2021, 'gas', 0.0, None, False)])
        default = _frame([(2021, 'gas', 7.0, 0.5, True), (2022, 'gas', 8.0, 0.5, True)])
        out = own.paths.prefer_by_year(default)
        assert dict(zip(out[YEAR_COLUMN], out[VALUE_COLUMN], strict=True)) == {2021: 7.0, 2022: 10.0}
        assert _qual(out) == {
            (2021, 'gas'): {'quality': 0.5, 'graded': 1.0, 'supplied': True},
            (2022, 'gas'): {'quality': 1.0, 'graded': 1.0, 'supplied': True},
        }


class TestSources:
    """How a binding's values get their qualifiers from the dataset they come from."""

    @staticmethod
    def _dataset_frame() -> ppl.PathsDataFrame:
        df = pl.DataFrame(
            {
                YEAR_COLUMN: [2020, 2020, 2020],
                'carrier': ['gas', 'oil', 'coal'],
                'mileage': [1.0, 2.0, None],
                'quality': [1.0, None, None],
            },
            schema={YEAR_COLUMN: pl.Int64, 'carrier': pl.Utf8, 'mileage': pl.Float64, 'quality': pl.Float64},
        )
        meta = DataFrameMeta(
            units={'mileage': unit_registry.parse_units('Mvkm/a'), 'quality': unit_registry.parse_units('dimensionless')},
            primary_keys=[YEAR_COLUMN, 'carrier'],
        )
        return to_ppdf(df, meta)

    def _qualifiers(self, source: QualifierSource) -> list[dict[str, object]]:
        df = self._dataset_frame()
        return df.select(source.qualifier_expr(df, 'mileage').alias('q'))['q'].to_list()

    def test_a_cell_grade_comes_from_its_evidence_projection(self) -> None:
        assert self._qualifiers(QualifierSource(quality_columns={'mileage': 'quality'})) == [
            {'quality': 1.0, 'graded': 1.0, 'supplied': True},
            {'quality': None, 'graded': 0.0, 'supplied': True},
            {'quality': None, 'graded': 0.0, 'supplied': False},
        ]

    def test_the_dataset_default_grades_what_evidence_does_not_and_never_an_empty_cell(self) -> None:
        source = QualifierSource(quality_columns={'mileage': 'quality'}, default_quality=0.5)
        assert [q['quality'] for q in self._qualifiers(source)] == [1.0, 0.5, None]

    def test_a_missing_projection_column_reads_as_ungraded(self) -> None:
        """An ungraded dataset has no projected column at all, which must not break the binding."""
        source = QualifierSource(quality_columns={'mileage': 'grades_nobody_entered'})
        assert [q['graded'] for q in self._qualifiers(source)] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize('metric', ['Value', 'Energy'])
def test_qualifier_column_names_round_trip(metric: str) -> None:
    assert qualifiers.qualified_metric(qualifiers.qualifier_column(metric)) == metric
    assert qualifiers.qualified_metric(metric) is None
