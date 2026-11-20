"""
Tests for how a conversion to a framework pairs an instance's dimension categories with the framework's.

An instance set up from an older model can hold a category under an identifier the framework has
since renamed, and categories its model no longer declares; see `_category_replacements`.
"""

import pytest

from kausal_common.datasets.models import DimensionCategory, DimensionScope
from kausal_common.datasets.tests.factories import DataPointFactory, DimensionCategoryFactory, DimensionFactory

from frameworks.conversion import _category_replacements

pytestmark = pytest.mark.django_db


def _scopes(local_ids: list[str]) -> tuple[DimensionScope, DimensionScope, dict[str, DimensionCategory]]:
    target_dimension = DimensionFactory.create()
    for identifier, spec in (('geothermal', {'aliases': ['geothermatl']}), ('biomass', {})):
        DimensionCategoryFactory.create(dimension=target_dimension, identifier=identifier, spec=spec)
    local_dimension = DimensionFactory.create()
    local = {
        identifier: DimensionCategoryFactory.create(dimension=local_dimension, identifier=identifier) for identifier in local_ids
    }
    # Unsaved: the pairing reads only the scopes' identifier and dimension.
    return (
        DimensionScope(identifier='energy_carrier', dimension=local_dimension),
        DimensionScope(identifier='energy_carrier', dimension=target_dimension),
        local,
    )


def _pairs(local: DimensionScope, target: DimensionScope) -> dict[str | None, str | None]:
    pairs, problems = _category_replacements(local, target)
    assert not problems
    return {category.identifier: replacement.identifier for category, replacement in pairs}


def test_a_renamed_category_moves_to_the_framework_category_that_lists_its_old_identifier():
    local, target, categories = _scopes(['geothermatl', 'biomass'])
    DataPointFactory.create(dimension_categories=[categories['geothermatl']])

    assert _pairs(local, target) == {'geothermatl': 'geothermal', 'biomass': 'biomass'}


def test_a_leftover_category_without_data_is_left_behind():
    local, target, _categories = _scopes(['biomass', 'total'])

    assert _pairs(local, target) == {'biomass': 'biomass'}


def test_a_category_with_data_and_no_counterpart_is_refused():
    local, target, categories = _scopes(['biomass', 'total'])
    DataPointFactory.create(dimension_categories=[categories['total']])

    pairs, problems = _category_replacements(local, target)

    assert [c.identifier for c, _ in pairs] == ['biomass']
    assert len(problems) == 1
    assert 'total carry data and have no counterpart' in problems[0]


def test_an_alias_whose_data_would_duplicate_existing_data_is_refused():
    local, target, categories = _scopes(['geothermatl', 'geothermal'])
    point = DataPointFactory.create(dimension_categories=[categories['geothermal']])
    DataPointFactory.create(dataset=point.dataset, metric=point.metric, dimension_categories=[categories['geothermatl']])

    _pairs_found, problems = _category_replacements(local, target)

    assert len(problems) == 1
    assert 'would duplicate data already under geothermal' in problems[0]


def test_an_alias_and_its_new_identifier_merge_when_their_data_does_not_overlap():
    local, target, categories = _scopes(['geothermatl', 'geothermal'])
    point = DataPointFactory.create(dimension_categories=[categories['geothermal']])
    DataPointFactory.create(
        dataset=point.dataset,
        metric=point.metric,
        date=point.date.replace(year=2022),
        dimension_categories=[categories['geothermatl']],
    )

    assert _pairs(local, target) == {'geothermatl': 'geothermal', 'geothermal': 'geothermal'}
