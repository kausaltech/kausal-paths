from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from kausal_common.i18n.pydantic import I18nBaseModel, TranslatedString

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nodes.units import Unit

QUDT_NS = 'http://qudt.org/vocab/quantitykind/'
QUANTITY_KINDS_PATH = Path(__file__).resolve().parent.parent / 'configs' / 'quantities' / 'quantity_kinds.yaml'


class QuantityKind(I18nBaseModel):
    """A semantic classification of what a numeric value measures."""

    id: str
    label: TranslatedString
    icon: str | None = None
    qudt_iri: str | None = None
    is_stackable: bool = False
    is_activity: bool = False
    is_factor: bool = False
    is_unit_price: bool = False
    is_scalar_identity: bool = False
    """In a product, values of this kind preserve the other operand's quantity."""
    numerator: str | None = None
    """
    For a factor kind: the quantity its product with an activity yields
    (``emission_factor`` times activity gives ``emissions``). The concrete activity is
    recovered from units, never enumerated here.
    """


def _parse_qudt(raw: str | None) -> str | None:
    """Expand a ``quantitykind:Foo`` shorthand to the full QUDT IRI."""
    if raw is None:
        return None
    if raw.startswith('quantitykind:'):
        return QUDT_NS + raw.removeprefix('quantitykind:')
    return raw


def _load_kind(id: str, entry: dict[str, Any]) -> QuantityKind:
    label_raw = entry['label']
    if isinstance(label_raw, str):
        label = TranslatedString(en=label_raw)
    elif isinstance(label_raw, dict):
        label = TranslatedString(**label_raw)
    else:
        raise TypeError(f'Unexpected label type for {id!r}: {type(label_raw)}')
    icon: str | None = entry.get('icon')
    qudt: str | None = entry.get('qudt')
    return QuantityKind(
        id=id,
        label=label,
        icon=icon,
        qudt_iri=_parse_qudt(qudt),
        is_stackable=bool(entry.get('is_stackable', False)),
        is_activity=bool(entry.get('is_activity', False)),
        is_factor=bool(entry.get('is_factor', False)),
        is_unit_price=bool(entry.get('is_unit_price', False)),
        is_scalar_identity=bool(entry.get('is_scalar_identity', False)),
        numerator=entry.get('numerator'),
    )


class QuantityKindRegistry:
    _kinds: dict[str, QuantityKind]

    def __init__(self) -> None:
        self._kinds = {}

    def register(self, kind: QuantityKind) -> QuantityKind:
        if kind.id in self._kinds:
            raise ValueError(f'Quantity kind {kind.id!r} is already registered')
        self._kinds[kind.id] = kind
        return kind

    def get(self, id: str) -> QuantityKind | None:
        return self._kinds.get(id)

    def __getitem__(self, id: str) -> QuantityKind:
        return self._kinds[id]

    def __contains__(self, id: str) -> bool:
        return id in self._kinds

    def __iter__(self) -> Iterator[QuantityKind]:
        return iter(self._kinds.values())

    def __len__(self) -> int:
        return len(self._kinds)

    @property
    def stackable(self) -> frozenset[str]:
        return frozenset(k.id for k in self._kinds.values() if k.is_stackable)

    @property
    def activities(self) -> frozenset[str]:
        return frozenset(k.id for k in self._kinds.values() if k.is_activity)

    @property
    def factors(self) -> frozenset[str]:
        return frozenset(k.id for k in self._kinds.values() if k.is_factor)

    @property
    def unit_prices(self) -> frozenset[str]:
        return frozenset(k.id for k in self._kinds.values() if k.is_unit_price)

    @property
    def scalar_identities(self) -> frozenset[str]:
        return frozenset(k.id for k in self._kinds.values() if k.is_scalar_identity)

    def validate_cross_references(self) -> None:
        for kind in self._kinds.values():
            if kind.numerator is None:
                continue
            if not kind.is_factor:
                raise ValueError(f'Quantity kind {kind.id!r} has a numerator but is not a factor')
            if kind.numerator not in self._kinds:
                raise ValueError(f'Quantity kind {kind.id!r} has unknown numerator {kind.numerator!r}')

    @classmethod
    def from_yaml(cls, path: Path) -> QuantityKindRegistry:
        reg = cls()
        with path.open() as f:
            data = yaml.safe_load(f)
        for id, entry in data.items():
            reg.register(_load_kind(id, entry))
        reg.validate_cross_references()
        return reg


@cache
def get_registry() -> QuantityKindRegistry:
    return QuantityKindRegistry.from_yaml(QUANTITY_KINDS_PATH)


# --- Product algebra (validation-only v1) -------------------------------------
#
# Quantities are semantic templates layered over pint, not a parallel exponent
# system — pint remains the hard gate for dimensional arithmetic. These rules
# derive a product's quantity only where one of three patterns applies
# (measured to cover 82% of multiplicative nodes, 2026-08-07):
#
# * scalar identity — ``is_scalar_identity`` kinds (and a dimensionless
#   ``number``) preserve the other operand's quantity;
# * factor cancellation — a factor kind with a declared ``numerator`` times an
#   activity yields that numerator, guarded by actual pint unit cancellation
#   of the activity's substance dimensions;
# * price — a unit-price kind times anything yields ``currency``.
#
# Everything else returns None (unknown), and unknown never conflicts:
# incompleteness must not block a computation whose units check out. Factor
# kinds without a ``numerator`` (the generic ``factor``, ``occupancy_factor``,
# ``time_factor``…) stay advisory.

_TIME_DIMENSION = '[time]'


@dataclass(frozen=True, slots=True)
class QuantityOperand:
    """One product operand: its authored quantity and unit, either possibly unknown."""

    quantity: str | None
    unit: Unit | None


def _is_scalar_identity(operand: QuantityOperand, registry: QuantityKindRegistry) -> bool:
    if operand.quantity is None:
        return False
    if operand.quantity in registry.scalar_identities:
        return True
    if operand.quantity == 'number':
        return operand.unit is not None and operand.unit.dimensionless
    return False


def _substance_cancels(factor: QuantityOperand, activity: QuantityOperand) -> bool:
    """
    Check that the factor's denominator exactly consumes the activity's substance.

    The activity's substance is its unit dimensionality without ``[time]`` —
    activities are typically flows (vkm/a, MWh/a) and the per-time part is
    not what a factor cancels. The guard demands an exact opposite exponent
    in the factor for every substance dimension, so a partially-cancelling
    garbage product never gets a confident quantity claim.
    """
    if factor.unit is None or activity.unit is None:
        return False
    factor_dims = dict(factor.unit.dimensionality)
    return all(
        factor_dims.get(dimension, 0) == -exponent
        for dimension, exponent in activity.unit.dimensionality.items()
        if dimension != _TIME_DIMENSION
    )


def derive_product_quantity(  # noqa: C901
    operands: tuple[QuantityOperand, ...],
    inverse_operands: tuple[QuantityOperand, ...],
    registry: QuantityKindRegistry | None = None,
) -> str | None:
    """
    Derive the quantity of a product value, or ``None`` when no rule applies.

    Division is outside the v1 rule set: any inverse operand that is not a
    scalar identity makes the result unknown.
    """
    reg = registry if registry is not None else get_registry()
    if any(operand.quantity is None for operand in (*operands, *inverse_operands)):
        return None
    if any(not _is_scalar_identity(operand, reg) for operand in inverse_operands):
        return None

    significant = tuple(operand for operand in operands if not _is_scalar_identity(operand, reg))
    if not significant:
        return None
    if len(significant) == 1:
        return significant[0].quantity

    if any(operand.quantity in reg.unit_prices for operand in significant):
        return 'currency'

    if len(significant) == 2:
        for factor, activity in (significant, tuple(reversed(significant))):
            assert factor.quantity is not None
            assert activity.quantity is not None
            factor_kind = reg.get(factor.quantity)
            activity_kind = reg.get(activity.quantity)
            if factor_kind is None or factor_kind.numerator is None or activity_kind is None:
                continue
            # The quantity class is only a plausibility gate; the load-bearing
            # guard is the unit cancellation below.
            if not (activity_kind.is_activity or activity_kind.is_stackable):
                continue
            if _substance_cancels(factor, activity):
                return factor_kind.numerator
    return None
