"""Quantity-kind registry semantics and the v1 product algebra."""

import pytest

from nodes.quantities import (
    QuantityKind,
    QuantityKindRegistry,
    QuantityOperand,
    derive_product_quantity,
    get_registry,
)
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def _unit(text: str):
    return unit_registry.parse_units(text)


def _operand(quantity: str | None, unit: str | None) -> QuantityOperand:
    return QuantityOperand(quantity=quantity, unit=_unit(unit) if unit is not None else None)


def test_registry_carries_the_algebra_vocabulary() -> None:
    registry = get_registry()
    assert registry.scalar_identities == {'fraction', 'ratio', 'mix'}
    assert registry['emission_factor'].numerator == 'emissions'
    assert registry['fuel_factor'].numerator == 'fuel_consumption'
    # Every numerator must resolve to a registered kind on a factor kind.
    registry.validate_cross_references()


def test_registry_rejects_inconsistent_cross_references() -> None:
    from kausal_common.i18n.pydantic import TranslatedString

    registry = QuantityKindRegistry()
    registry.register(QuantityKind(id='bogus', label=TranslatedString(en='Bogus'), is_factor=True, numerator='nowhere'))
    with pytest.raises(ValueError, match='unknown numerator'):
        registry.validate_cross_references()

    registry2 = QuantityKindRegistry()
    registry2.register(QuantityKind(id='emissions', label=TranslatedString(en='E')))
    registry2.register(QuantityKind(id='notafactor', label=TranslatedString(en='N'), numerator='emissions'))
    with pytest.raises(ValueError, match='not a factor'):
        registry2.validate_cross_references()


def test_scalar_identity_preserves_the_other_operand() -> None:
    result = derive_product_quantity((_operand('fraction', 'dimensionless'), _operand('energy', 'GWh/a')), ())
    assert result == 'energy'
    # A dimensionless number is a scalar; a dimensioned number is not.
    assert derive_product_quantity((_operand('number', 'dimensionless'), _operand('energy', 'GWh/a')), ()) == 'energy'
    assert derive_product_quantity((_operand('number', 'cap'), _operand('energy', 'GWh/a')), ()) is None


def test_factor_cancellation_is_guarded_by_units() -> None:
    factor = _operand('emission_factor', 'kg/vkm')
    assert derive_product_quantity((factor, _operand('vehicle_mileage', 'vkm/a')), ()) == 'emissions'
    # No substance cancellation, no claim — even though the quantity classes match.
    assert derive_product_quantity((factor, _operand('energy', 'GWh/a')), ()) is None


def test_price_and_division_rules() -> None:
    assert derive_product_quantity((_operand('unit_price', 'EUR/MWh'), _operand('energy', 'MWh/a')), ()) == 'currency'
    # A non-scalar divisor makes the result unknown in v1.
    assert derive_product_quantity((_operand('emissions', 't/a'),), (_operand('population', 'cap'),)) is None
    # A scalar divisor does not.
    assert derive_product_quantity((_operand('emissions', 't/a'),), (_operand('ratio', 'dimensionless'),)) == 'emissions'
