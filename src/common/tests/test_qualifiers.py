"""
Qualifier columns: what the frame operations do to the grade and provenance of a value.

The properties that matter are the ones a Datengüte and a source choice depend on: a sum's grade
is its components' grades weighted by magnitude, an ungraded part lowers the graded share
rather than disappearing, a factor with no grade leaves the activity's grade alone, and a value
an operation made up is marked as such.
"""

from typing import TYPE_CHECKING, TypedDict

import polars as pl
import pytest

from common import qualifiers
from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.simple import SimpleNode
from nodes.tests.factories import SimpleNodeFactory
from nodes.transforms import QualifierSource
from nodes.units import unit_registry

if TYPE_CHECKING:
    from common import polars as ppl

pytestmark = pytest.mark.django_db


class FixtureQualifier(TypedDict):
    quality: qualifiers.CoveredScore | None
    reported: qualifiers.ReportingStatus | None


TEST_CATALOG = qualifiers.QualifierCatalog((
    *qualifiers.BUILTIN_QUALIFIERS.definitions,
    qualifiers.QualifierDefinition('quality', qualifiers.Propagation.COVERED_SCORE),
))


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
        df = df.with_columns(
            qualifiers.make(
                catalog=TEST_CATALOG,
                assessments={'quality': qualifiers.covered_score(pl.col('_q'), pl.col('_q').is_not_null().cast(pl.Float64))},
                reported=pl.col('_s'),
            ).alias(QUAL)
        )
    df = df.drop('_q', '_s')
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=[YEAR_COLUMN, 'energy_carrier'])
    return to_ppdf(df, meta)


def _reported(qual: FixtureQualifier) -> qualifiers.ReportingStatus:
    reported = qual['reported']
    assert reported is not None
    return reported


def _qual(df: ppl.PathsDataFrame) -> dict[tuple[int | str, ...], FixtureQualifier]:
    keys = [col for col in df.primary_keys if col != YEAR_COLUMN]
    return {
        (row[YEAR_COLUMN], *(row[k] for k in keys)): row[QUAL]
        for row in df.select([YEAR_COLUMN, *keys, QUAL]).sort([YEAR_COLUMN, *keys]).to_dicts()
    }


class TestPairing:
    @pytest.mark.parametrize('qualified_side', ['left', 'right', 'both', 'neither'])
    def test_stacking_aligns_columns_without_reassigning_values_or_qualifiers(self, qualified_side: str) -> None:
        left = _frame([(2020, 'gas', 3.0, 1.0, True)], qualified=qualified_side in ('left', 'both'))
        right = _frame([(2021, 'gas', 5.0, 0.5, False)], qualified=qualified_side in ('right', 'both'))
        right = right.with_columns(pl.lit(value=True).alias(FORECAST_COLUMN)).select(list(reversed(right.columns)))

        stacked = left.paths.concat_vertical(right)

        assert stacked[YEAR_COLUMN].to_list() == [2020, 2021]
        assert stacked[VALUE_COLUMN].to_list() == [3.0, 5.0]
        assert stacked[FORECAST_COLUMN].to_list() == [False, True]
        assert stacked.get_meta() == left.get_meta()
        if qualified_side == 'neither':
            assert not stacked.qualifier_cols
        else:
            rows = _qual(stacked)
            for source in (left, right):
                if source.qualifier_cols:
                    assert all(rows[key] == value for key, value in _qual(source).items())
                else:
                    assert rows[source[YEAR_COLUMN][0], 'gas']['quality'] is None

    def test_reference_year_scaling_drops_the_reference_qualifier_with_its_value(self) -> None:
        df = _frame([
            (2020, 'gas', 2.0, 1.0, True),
            (2021, 'gas', 6.0, 0.5, False),
            (2020, 'electricity', 4.0, 0.8, True),
            (2021, 'electricity', 8.0, 0.4, False),
        ]).with_columns((pl.col(YEAR_COLUMN) == 2021).alias(FORECAST_COLUMN))

        scaled = df.paths._scale_by_reference_year(df, 2020).sort(YEAR_COLUMN, 'energy_carrier')

        assert set(scaled.columns) == set(df.columns)
        assert scaled.qualifier_cols == {VALUE_COLUMN: QUAL}
        assert scaled.metric_cols == [VALUE_COLUMN]
        assert scaled[VALUE_COLUMN].to_list() == [1.0, 1.0, 2.0, 3.0]
        assert scaled[FORECAST_COLUMN].to_list() == [False, False, True, True]
        assert scaled.get_unit(VALUE_COLUMN) == unit_registry.dimensionless
        assert _qual(scaled) == _qual(df)

    @pytest.mark.parametrize('qualified_side', ['left', 'right', 'both'])
    def test_joined_qualifiers_follow_their_metric_names(self, qualified_side: str) -> None:
        left = _frame([(2020, 'gas', 3.0, 1.0, True)], qualified=qualified_side != 'right')
        right = _frame([(2020, 'gas', 1.0, 0.5, False)], qualified=qualified_side != 'left')
        joined = left.paths.join_over_index(right)
        expected = {}
        if qualified_side != 'right':
            expected[VALUE_COLUMN] = QUAL
        if qualified_side != 'left':
            expected['Value_right'] = 'Value_right__qual'
        assert joined.qualifier_cols == expected
        assert joined.metric_cols == [VALUE_COLUMN, 'Value_right']
        assert 'Value__qual_right' not in joined.columns
        remaining = joined.drop('Value_right')
        assert 'Value_right__qual' not in remaining.columns
        assert remaining.rename({VALUE_COLUMN: 'Energy'}).qualifier_cols == (
            {'Energy': 'Energy__qual'} if qualified_side != 'right' else {}
        )

    @pytest.mark.parametrize('qualified_side', ['left', 'right'])
    def test_stacking_with_one_qualified_side_leaves_the_other_unassessed(self, qualified_side: str) -> None:
        left = _frame([(2020, 'gas', 3.0, 1.0, True)], qualified=qualified_side == 'left')
        right = _frame([(2021, 'gas', 1.0, 1.0, True)], qualified=qualified_side == 'right')
        stacked = left.paths.concat_vertical(right)
        assert stacked.qualifier_cols == {VALUE_COLUMN: QUAL}
        assessed, unassessed = (2020, 2021) if qualified_side == 'left' else (2021, 2020)
        rows = _qual(stacked)
        assert rows[assessed, 'gas']['quality'] == {'score': 1.0, 'coverage': 1.0}
        assert rows[unassessed, 'gas']['quality'] is None
        assert rows[unassessed, 'gas']['reported'] == {'any': None, 'all': None}

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
    @pytest.mark.parametrize(
        ('left', 'right', 'expected'),
        [(True, True, True), (True, False, False), (True, None, None), (False, None, False), (None, None, None)],
    )
    def test_reporting_requires_all_contributions(self, left: bool | None, right: bool | None, expected: bool | None) -> None:
        # Zero has no grading weight, but remains a contributing reported cell.
        frame = _frame([(2020, 'gas', 10.0, 1.0, left), (2020, 'oil', 0.0, None, right)])
        reduced = frame.paths.sum_over_dims('energy_carrier')
        assert reduced[QUAL][0]['reported']['all'] is expected
        pair = _frame([(2020, 'gas', 10.0, 1.0, left)]).paths.add_with_dims(_frame([(2020, 'gas', 0.0, None, right)]))
        assert pair[QUAL][0]['reported']['all'] is expected

    def test_positive_sums_preserve_assessment_through_multiple_reductions(self) -> None:
        frame = _frame([(2020, 'gas', 6.0, 1.0, True), (2020, 'oil', 4.0, None, False)])
        first = frame.paths.sum_over_dims('energy_carrier')
        more = _frame([(2020, 'gas', 10.0, 0.5, True)]).paths.sum_over_dims('energy_carrier')
        result = first.paths.add_with_dims(more)
        assert result[QUAL][0]['quality'] == {'score': 11 / 16, 'coverage': 0.8}
        assert result[QUAL][0]['reported']['all'] is False

    def test_the_grade_of_a_sum_is_weighted_by_magnitude(self) -> None:
        """Methodenpapier §3.4: each component's grade weighted by its share."""
        df = _frame([(2020, 'gas', 3.0, 1.0, True), (2020, 'oil', 1.0, 0.0, True)])
        out = df.paths.sum_over_dims('energy_carrier')
        assert out[QUAL].to_list() == [{'quality': {'score': 0.75, 'coverage': 1.0}, 'reported': {'any': True, 'all': True}}]

    def test_an_ungraded_part_lowers_the_graded_share_not_the_grade(self) -> None:
        df = _frame([(2020, 'gas', 3.0, 1.0, True), (2020, 'oil', 1.0, 0.0, True), (2020, 'coal', 4.0, None, True)])
        out = df.paths.sum_over_dims('energy_carrier')
        assert out[QUAL].to_list() == [{'quality': {'score': 0.75, 'coverage': 0.5}, 'reported': {'any': True, 'all': True}}]

    def test_zero_group_has_no_energy_weighted_assessment(self) -> None:
        df = _frame([(2020, 'gas', 0.0, 1.0, False), (2020, 'oil', 0.0, 0.5, False)])
        out = df.paths.sum_over_dims('energy_carrier')
        assert out[QUAL].to_list() == [{'quality': {'score': None, 'coverage': None}, 'reported': {'any': False, 'all': False}}]

    def test_adding_frames_weights_the_same_way(self) -> None:
        left = _frame([(2020, 'gas', 3.0, 1.0, True)])
        right = _frame([(2020, 'gas', 1.0, 0.0, False), (2021, 'gas', 2.0, 0.5, True)])
        out = left.paths.add_with_dims(right)
        assert _qual(out) == {
            (2020, 'gas'): {'quality': {'score': 0.75, 'coverage': 1.0}, 'reported': {'any': True, 'all': False}},
            (2021, 'gas'): {'quality': {'score': 0.5, 'coverage': 1.0}, 'reported': {'any': True, 'all': True}},
        }

    @pytest.mark.parametrize('qualified_side', ['left', 'right'])
    def test_an_unqualified_addend_counts_as_ungraded(self, qualified_side: str) -> None:
        """Either way round: a join leaves the right-hand qualifier unsuffixed when the left has none."""
        graded = _frame([(2020, 'gas', 1.0, 1.0, True)])
        ungraded = _frame([(2020, 'gas', 3.0, None, None)], qualified=False)
        left, right = (graded, ungraded) if qualified_side == 'left' else (ungraded, graded)
        out = left.paths.add_with_dims(right)
        assert _qual(out) == {
            (2020, 'gas'): {'quality': {'score': 1.0, 'coverage': 0.25}, 'reported': {'any': True, 'all': None}}
        }


class TestProducts:
    def test_a_factor_without_a_grade_leaves_the_activity_grade_alone(self) -> None:
        activity = _frame([(2020, 'gas', 10.0, 0.5, True)], unit='Mvkm/a')
        factor = _frame([(2020, 'gas', 0.3, None, None)], unit='MWh/vkm', qualified=False)
        out = activity.paths.multiply_with_dims(factor)
        assert _qual(out) == {(2020, 'gas'): {'quality': {'score': 0.5, 'coverage': 1.0}, 'reported': {'any': True, 'all': None}}}

    def test_a_graded_factor_on_the_right_keeps_its_grade(self) -> None:
        share = _frame([(2020, 'gas', 0.5, None, None)], unit='dimensionless', qualified=False)
        activity = _frame([(2020, 'gas', 10.0, 0.5, True)], unit='Mvkm/a')
        out = share.paths.multiply_with_dims(activity)
        assert _qual(out) == {(2020, 'gas'): {'quality': {'score': 0.5, 'coverage': 1.0}, 'reported': {'any': True, 'all': None}}}

    def test_two_graded_factors_take_the_lower_grade(self) -> None:
        left = _frame([(2020, 'gas', 10.0, 1.0, True)])
        right = _frame([(2020, 'gas', 2.0, 0.25, False)], unit='dimensionless')
        out = left.paths.multiply_with_dims(right)
        assert _qual(out) == {
            (2020, 'gas'): {'quality': {'score': 0.25, 'coverage': 1.0}, 'reported': {'any': True, 'all': False}}
        }


class TestFills:
    @pytest.mark.parametrize('qualified_side', ['left', 'right', 'both'])
    def test_gap_filling_carries_the_chosen_source_and_can_be_added_again(self, qualified_side: str) -> None:
        node = SimpleNodeFactory.create()
        assert isinstance(node, SimpleNode)
        calculated = _frame(
            [(2020, 'gas', 3.0, 1.0, False), (2021, 'gas', None, None, None)],
            qualified=qualified_side != 'right',
        )
        dataset = _frame(
            [(2020, 'gas', 10.0, 0.5, True), (2021, 'gas', 2.0, 0.5, True)],
            qualified=qualified_side != 'left',
        )
        filled = node._fill_gaps_from_input(calculated, dataset).sort(YEAR_COLUMN)
        assert filled[VALUE_COLUMN].to_list() == [3.0, 2.0]
        assert _qual(filled) == {
            (2020, 'gas'): (
                {'quality': {'score': 1.0, 'coverage': 1.0}, 'reported': {'any': False, 'all': False}}
                if qualified_side != 'right'
                else None
            ),
            (2021, 'gas'): (
                {'quality': {'score': 0.5, 'coverage': 1.0}, 'reported': {'any': True, 'all': True}}
                if qualified_side != 'left'
                else None
            ),
        }
        assert set(filled.columns) == {YEAR_COLUMN, FORECAST_COLUMN, VALUE_COLUMN, 'energy_carrier', QUAL}
        summed = filled.paths.add_with_dims(dataset).paths.add_with_dims(dataset).sort(YEAR_COLUMN)
        assert summed[VALUE_COLUMN].to_list() == [23.0, 6.0]

    @pytest.mark.parametrize(('end_grade', 'score', 'coverage'), [(1.0, 1.0, 1.0), (0.5, 0.875, 1.0), (None, 1.0, 0.75)])
    def test_interpolation_carries_assessment_mass_using_actual_years(
        self, end_grade: float | None, score: float, coverage: float
    ) -> None:
        before = _frame([(2020, 'gas', 10.0, 1.0, True), (2024, 'gas', 30.0, end_grade, True)])
        after = _frame(
            [(2020, 'gas', 10.0, None, None), (2021, 'gas', 15.0, None, None), (2024, 'gas', 30.0, None, None)], qualified=False
        )
        result = qualifiers.carry_over(before, after, fill='interpolate')
        assert _qual(result)[(2021, 'gas')] == {
            'quality': {'score': score, 'coverage': coverage},
            'reported': {'any': False, 'all': False},
        }
        assert _reported(_qual(result)[(2020, 'gas')])['all'] is True

    @pytest.mark.parametrize(('fill', 'year'), [('backfill', 2019), ('extend', 2021)])
    def test_constant_fill_preserves_assessment_and_marks_it_not_reported(self, fill: str, year: int) -> None:
        before = _frame([(2020, 'gas', 10.0, 0.5, True)])
        after = _frame([(year, 'gas', 10.0, None, None), (2020, 'gas', 10.0, None, None)], qualified=False)
        result = qualifiers.carry_over(before, after, fill=fill)
        assert _qual(result)[(year, 'gas')] == {
            'quality': {'score': 0.5, 'coverage': 1.0},
            'reported': {'any': False, 'all': False},
        }

    def test_interpolation_does_not_invent_an_assessment_or_cross_categories(self) -> None:
        before = _frame([
            (2020, 'gas', 10.0, None, True),
            (2022, 'gas', 20.0, None, True),
            (2020, 'oil', 10.0, 1.0, True),
            (2022, 'oil', 20.0, 1.0, True),
        ])
        after = _frame(
            [(year, carrier, 15.0, None, None) for year in (2020, 2021, 2022) for carrier in ('gas', 'oil')], qualified=False
        )
        result = qualifiers.carry_over(before, after, fill='interpolate')
        assert _qual(result)[(2021, 'gas')]['quality'] == {'score': None, 'coverage': 0.0}
        assert _qual(result)[(2021, 'oil')]['quality'] == {'score': 1.0, 'coverage': 1.0}

    def test_empty_to_zero_marks_what_it_filled(self) -> None:
        df = _frame([(2020, 'gas', 5.0, None, None), (2021, 'oil', None, None, None)], qualified=False)
        out = df.paths.get_operation('empty_to_zero')(df, None)
        supplied = {key: _reported(qual)['all'] for key, qual in _qual(out).items()}
        assert supplied == {(2020, 'gas'): True, (2020, 'oil'): False, (2021, 'gas'): False, (2021, 'oil'): False}

    @pytest.mark.parametrize('qualified', [False, True])
    def test_empty_to_zero_keeps_the_columns_of_a_frame_without_rows(self, *, qualified: bool) -> None:
        # A dataset nobody has filled in yet: the wide round trip would leave no metric column.
        df = _frame([], qualified=qualified)
        out = df.paths.get_operation('empty_to_zero')(df, None)
        assert out.is_empty()
        assert out.metric_cols == [VALUE_COLUMN]
        assert out.primary_keys == df.primary_keys

    def test_other_fills_keep_a_record_but_do_not_start_one(self) -> None:
        df = _frame([(2020, 'gas', 5.0, None, None)], qualified=False)
        assert qualifiers.carry_over(df, df) is df


class TestChoosingASource:
    def test_partial_reported_year_uses_own_data_without_filling_from_default(self) -> None:
        own = _frame([(2020, 'gas', 0.0, 1.0, True), (2020, 'oil', 0.0, None, False)])
        default = _frame([(2020, 'gas', 9.0, 0.5, True), (2020, 'oil', 8.0, 0.5, True)])
        result = own.paths.prefer_by_year(default)
        assert result[VALUE_COLUMN].to_list() == [0.0, 0.0]
        assert _reported(_qual(result)[(2020, 'oil')])['all'] is False

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
            (2021, 'gas'): {'quality': {'score': 0.5, 'coverage': 1.0}, 'reported': {'any': True, 'all': True}},
            (2022, 'gas'): {'quality': {'score': 1.0, 'coverage': 1.0}, 'reported': {'any': True, 'all': True}},
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

    def _qualifiers(self, source: QualifierSource) -> list[FixtureQualifier]:
        df = self._dataset_frame()
        return df.select(source.qualifier_expr(df, 'mileage').alias('q'))['q'].to_list()

    def test_a_cell_grade_comes_from_its_evidence_projection(self) -> None:
        assert self._qualifiers(
            QualifierSource(catalog=TEST_CATALOG, quality_identifier='quality', quality_columns={'mileage': 'quality'})
        ) == [
            {'quality': {'score': 1.0, 'coverage': 1.0}, 'reported': {'any': True, 'all': True}},
            {'quality': {'score': None, 'coverage': 0.0}, 'reported': {'any': True, 'all': True}},
            {'quality': {'score': None, 'coverage': 0.0}, 'reported': {'any': False, 'all': False}},
        ]

    def test_the_dataset_default_grades_what_evidence_does_not_and_never_an_empty_cell(self) -> None:
        source = QualifierSource(
            catalog=TEST_CATALOG, quality_identifier='quality', quality_columns={'mileage': 'quality'}, default_quality=0.5
        )
        assert [q['quality']['score'] for q in self._qualifiers(source) if q['quality'] is not None] == [1.0, 0.5, None]

    def test_a_missing_projection_column_reads_as_ungraded(self) -> None:
        """An ungraded dataset has no projected column at all, which must not break the binding."""
        source = QualifierSource(
            catalog=TEST_CATALOG, quality_identifier='quality', quality_columns={'mileage': 'grades_nobody_entered'}
        )
        assert [q['quality']['coverage'] for q in self._qualifiers(source) if q['quality'] is not None] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize('metric', ['Value', 'Energy'])
def test_qualifier_column_names_round_trip(metric: str) -> None:
    assert qualifiers.qualified_metric(qualifiers.qualifier_column(metric)) == metric
    assert qualifiers.qualified_metric(metric) is None


class TestRemoveSubset:
    def test_remainder_keeps_parent_grade_instead_of_mixing_subset_grade(self) -> None:
        total = _frame([(2020, 'gas', 100.0, 0.25, True), (2020, 'oil', 50.0, 0.0, True)])
        subset = _frame([(2020, 'gas', 20.0, 1.0, True)])
        out = total.paths.remove_subset(subset)
        assert out.filter(pl.col('energy_carrier') == 'gas')['Value'][0] == 80.0
        assert _qual(out) == {
            (2020, 'gas'): {'quality': {'score': 0.25, 'coverage': 1.0}, 'reported': {'any': True, 'all': True}},
            (2020, 'oil'): {'quality': {'score': 0.0, 'coverage': 1.0}, 'reported': {'any': True, 'all': True}},
        }
        general = total.paths.subtract_with_dims(subset)
        assessment = _qual(general)[(2020, 'gas')]['quality']
        assert assessment is not None
        assert assessment['score'] == pytest.approx(0.375)

    @pytest.mark.parametrize('coverage', [0.0, 0.5])
    def test_partial_subset_does_not_invent_remainder_coverage(self, coverage: float) -> None:
        total = _frame([(2020, 'gas', 100.0, 1.0, True)])
        subset = _frame([(2020, 'gas', 20.0, 0.5, True)]).with_columns(
            qualifiers.make(
                catalog=TEST_CATALOG,
                reported=pl.lit(value=True),
                assessments={'quality': qualifiers.covered_score(pl.lit(0.5), pl.lit(coverage))},
            ).alias(QUAL)
        )
        assert _qual(total.paths.remove_subset(subset))[(2020, 'gas')]['quality'] is None

    def test_zero_subset_preserves_partial_parent_and_zero_remainder_has_no_grade(self) -> None:
        total = _frame([(2020, 'gas', 100.0, 0.5, True)])
        zero = _frame([(2020, 'gas', 0.0, None, False)])
        assert _qual(total.paths.remove_subset(zero)) == _qual(total)
        assert _qual(total.paths.remove_subset(total))[(2020, 'gas')]['quality'] is None

    @pytest.mark.parametrize(('total_value', 'subset_value'), [(10.0, 20.0), (10.0, -1.0), (-1.0, 0.0)])
    def test_invalid_subset_is_rejected(self, total_value: float, subset_value: float) -> None:
        total = _frame([(2020, 'gas', total_value, 1.0, True)])
        subset = _frame([(2020, 'gas', subset_value, 1.0, True)])
        with pytest.raises(ValueError, match='nonnegative'):
            total.paths.remove_subset(subset)

    def test_subset_without_parent_is_rejected(self) -> None:
        total = _frame([(2020, 'gas', 10.0, 1.0, True)])
        subset = _frame([(2020, 'oil', 1.0, 1.0, True)])
        with pytest.raises(ValueError, match='subset no larger'):
            total.paths.remove_subset(subset)


def test_unassessed_factor_with_explicit_zero_coverage_is_neutral_for_product() -> None:
    activity = _frame([(2020, 'gas', 10.0, 1.0, True)])
    factor = _frame([(2020, 'gas', 2.0, None, False)], unit='dimensionless').with_columns(
        qualifiers.make(
            catalog=TEST_CATALOG,
            reported=pl.lit(value=False),
            assessments={'quality': qualifiers.covered_score(pl.lit(None, dtype=pl.Float64), pl.lit(0.0))},
        ).alias(QUAL)
    )
    assert _qual(activity.paths.multiply_with_dims(factor))[(2020, 'gas')]['quality'] == {
        'score': 1.0,
        'coverage': 1.0,
    }


@pytest.mark.parametrize(
    ('left', 'right', 'expected'),
    [(True, False, True), (True, None, True), (False, None, None), (False, False, False), (None, None, None)],
)
def test_reporting_any_survives_reduction(left: bool | None, right: bool | None, expected: bool | None) -> None:
    frame = _frame([(2020, 'gas', 10.0, None, left), (2020, 'oil', 0.0, None, right)])
    assert frame.paths.sum_over_dims('energy_carrier')[QUAL][0]['reported']['any'] is expected
