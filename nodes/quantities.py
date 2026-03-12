from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from kausal_common.i18n.pydantic import I18nBaseModel, TranslatedString

if TYPE_CHECKING:
    from collections.abc import Iterator

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

    @classmethod
    def from_yaml(cls, path: Path) -> QuantityKindRegistry:
        reg = cls()
        with path.open() as f:
            data = yaml.safe_load(f)
        for id, entry in data.items():
            reg.register(_load_kind(id, entry))
        return reg


@cache
def get_registry() -> QuantityKindRegistry:
    return QuantityKindRegistry.from_yaml(QUANTITY_KINDS_PATH)
