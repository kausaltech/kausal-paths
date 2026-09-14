from django.utils.translation import override

import pytest

from paths.graphql_types import format_unit

from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize(
    ('unit', 'expected_short', 'expected_html_short'),
    [
        ('1/a', '1/yr', '1\u2215yr'),
        ('1/%', '1/%', '1\u2215%'),
    ],
)
def test_format_unit_handles_denominator_only_units(
    unit: str,
    expected_short: str,
    expected_html_short: str,
) -> None:
    parsed_unit = unit_registry.parse_units(unit)

    with override('en'):
        assert format_unit(parsed_unit) == expected_short
        assert format_unit(parsed_unit, html=True) == expected_html_short


def test_format_unit_preserves_numerator_pluralization() -> None:
    unit = unit_registry.parse_units('a')

    with override('en'):
        assert format_unit(unit) == 'yrs'
