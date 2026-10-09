"""
Give a document new uuids for what it defines, as a copy does.

`rekeyed` walks a Pydantic document by the uuid kinds of its fields (`paths.uuid_kinds`)
in two passes. The first collects the identities the document owns and mints a new uuid
for each. The second rewrites those identities and every reference to them, keeps the
references to anything else, sets provenance to the source and drops tokens.

An identity the document also carries as someone else's is not its own, wherever else it
appears: an instance that closes a template's shape redeclares it under the template's
uuid, and a copy must override the same shape.

What a document owns is mostly where it sits, but a model can say otherwise. These
optional hooks on a model class customize the walk:

- ``__rekey_foreign__``: names of fields holding what the document refers to but does not
  own, such as a bundled template. Their identities are read as foreign, and they are
  passed through untouched.
- ``rekey_owns(field, owned) -> bool``: whether the identities under ``field`` belong to
  the document, given whether the model itself does. A framework's dimension inside an
  instance's catalog answers no.
- ``rekey_finish(original, rekeying) -> Self``: called on the rewritten model to restamp
  what is derived from the uuids it holds, such as content hashes.

Provenance needs no hook when the model holds its own identity: a model whose identity of
some entity was rewritten has its provenance field of that entity set to the old uuid.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID, uuid4

from pydantic import BaseModel

from .uuid_kinds import (
    AnyOf,
    DictOf,
    Identity,
    Provenance,
    Ref,
    SeqOf,
    Shape,
    Token,
    UuidEntity,
    UuidLeaf,
    uuid_slots,
)


@dataclass
class Rekeying:
    """The outcome of `rekeyed`, also handed to ``rekey_finish`` hooks."""

    mapping: dict[UUID, UUID] = field(default_factory=dict)
    """Old uuid to new, for every identity the document owns."""
    owned: dict[UuidEntity, set[UUID]] = field(default_factory=lambda: defaultdict(set))
    """The identities the document owns, by entity, as they were before rekeying."""
    foreign: dict[UuidEntity, set[UUID]] = field(default_factory=lambda: defaultdict(set))
    """Identities the document carries but does not own, such as a framework's dimensions; kept."""
    bundled: set[UUID] = field(default_factory=set)
    """The foreign identities that travel with the document, in its ``__rekey_foreign__`` fields."""
    kept: dict[UuidEntity, set[UUID]] = field(default_factory=lambda: defaultdict(set))
    """References to something the document does not own; kept as they are."""

    def outside(self) -> dict[UuidEntity, set[UUID]]:
        """Return what the document refers to and neither owns nor bundles: what must exist where it goes."""
        result: dict[UuidEntity, set[UUID]] = defaultdict(set)
        for source in (self.foreign, self.kept):
            for entity, ids in source.items():
                result[entity] |= ids - self.bundled
        return {entity: ids for entity, ids in result.items() if ids}


def rekeyed[M: BaseModel](
    document: M,
    *,
    seed: Mapping[UUID, UUID] | None = None,
    set_provenance: bool = True,
) -> tuple[M, Rekeying]:
    """
    Return ``document`` with new uuids for everything it owns, and the record of the change.

    ``seed`` fixes the new uuid of some identities in advance, such as an instance's,
    when its row exists before the document is imported into it. With
    ``set_provenance``, provenance fields point at the source they were copied from.
    """
    rekeying = _collect(document)
    seeds = dict(seed or {})
    for ids in rekeying.owned.values():
        for identity in ids:
            rekeying.mapping[identity] = seeds.get(identity) or uuid4()
    rewritten = _Rewriter(rekeying, set_provenance).model(document)
    return cast('M', rewritten), rekeying


def survey(document: BaseModel) -> Rekeying:
    """
    Return what ``document`` owns and what it refers to outside itself, changing nothing.

    The record of a rekeying that keeps every uuid: ``mapping`` maps each owned identity
    to itself, and ``outside()`` is what the document needs to find where it goes.
    """
    rekeying = _collect(document)
    rekeying.mapping = {identity: identity for ids in rekeying.owned.values() for identity in ids}
    _Rewriter(rekeying, set_provenance=False).model(document)
    return rekeying


def _collect(document: BaseModel) -> Rekeying:
    rekeying = Rekeying()
    collector = _Collector(rekeying)
    collector.model(document, owned=True)
    foreign = set().union(*rekeying.foreign.values()) if rekeying.foreign else set()
    for identity, entity in collector.owned.items():
        if identity not in foreign:
            rekeying.owned[entity].add(identity)
    return rekeying


class _Collector:
    def __init__(self, rekeying: Rekeying):
        self.rekeying = rekeying
        self.owned: dict[UUID, UuidEntity] = {}
        """The owned identities, in the order met, with their entities."""

    def model(self, value: BaseModel, *, owned: bool, bundled: bool = False) -> None:
        foreign = _foreign_fields(value)
        for name, shape in uuid_slots(type(value)):
            if name in foreign:
                self.value(getattr(value, name), shape, owned=False, bundled=True)
                continue
            field_owned = owns(name, owned) if (owns := getattr(value, 'rekey_owns', None)) else owned
            self.value(getattr(value, name), shape, owned=field_owned, bundled=bundled)

    def value(self, value: Any, shape: Shape, *, owned: bool, bundled: bool) -> None:
        if isinstance(value, BaseModel):
            self.model(value, owned=owned, bundled=bundled)
        elif isinstance(value, UUID):
            leaf = _uuid_leaf(shape)
            if leaf is None or not isinstance(leaf.kind, Identity):
                return
            if owned:
                self.owned[value] = leaf.kind.entity
            else:
                self.rekeying.foreign[leaf.kind.entity].add(value)
                if bundled:
                    self.rekeying.bundled.add(value)
        else:
            for item, item_shape in _parts(value, shape):
                self.value(item, item_shape, owned=owned, bundled=bundled)


class _Rewriter:
    def __init__(self, rekeying: Rekeying, set_provenance: bool):
        self.rekeying = rekeying
        self.set_provenance = set_provenance

    def model(self, value: BaseModel) -> BaseModel:
        skip = _foreign_fields(value)
        changes: dict[str, Any] = {}
        replaced: dict[UuidEntity, UUID] = {}
        provenance: list[tuple[str, UuidEntity]] = []
        for name, shape in uuid_slots(type(value)):
            if name in skip:
                continue
            current = getattr(value, name)
            leaf = _uuid_leaf(shape)
            if leaf is not None and isinstance(leaf.kind, Provenance):
                provenance.append((name, leaf.kind.entity))
                continue
            new = self.value(current, shape)
            if new is not current:
                changes[name] = new
                if leaf is not None and isinstance(leaf.kind, Identity) and isinstance(current, UUID):
                    replaced[leaf.kind.entity] = current
        if self.set_provenance:
            for name, entity in provenance:
                if entity in replaced:
                    changes[name] = replaced[entity]
        result = value.model_copy(update=changes) if changes else value
        finish = getattr(result, 'rekey_finish', None)
        if finish is not None:
            result = finish(value, self.rekeying)
        return result

    def value(self, value: Any, shape: Shape) -> Any:
        if isinstance(value, BaseModel):
            return self.model(value)
        if isinstance(value, UUID):
            return self.uuid(value, _uuid_leaf(shape))
        if isinstance(value, Mapping):
            return self.mapping(value, shape)
        if isinstance(value, (list, tuple, set, frozenset)):
            return self.sequence(value, shape)
        return value

    def mapping(self, value: Mapping[Any, Any], shape: Shape) -> Mapping[Any, Any]:
        dict_shape = _find(shape, DictOf)
        if dict_shape is None:
            return value
        key_shape, item_shape = dict_shape.key, dict_shape.value
        items = [
            (
                self.value(key, key_shape) if key_shape is not None else key,
                self.value(item, item_shape) if item_shape is not None else item,
            )
            for key, item in value.items()
        ]
        unchanged = all(new[0] is old[0] and new[1] is old[1] for new, old in zip(items, value.items(), strict=True))
        return value if unchanged else dict(items)

    def sequence[C: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any]](self, value: C, shape: Shape) -> C:
        seq_shape = _find(shape, SeqOf)
        if seq_shape is None:
            return value
        items = [self.value(item, seq_shape.item) for item in value]
        if all(new is old for new, old in zip(items, value, strict=True)):
            return value
        return cast('C', type(value)(items))

    def uuid(self, value: UUID, leaf: UuidLeaf | None) -> UUID | None:
        if leaf is None or leaf.kind is None:
            return value
        kind = leaf.kind
        if isinstance(kind, Token):
            return None
        new = self.rekeying.mapping.get(value)
        if new is not None and isinstance(kind, (Identity, Ref)):
            return new
        if isinstance(kind, Ref):
            self.rekeying.kept[kind.entity].add(value)
        return value


def _find[S: (SeqOf, DictOf)](shape: Shape, kind: type[S]) -> S | None:
    if isinstance(shape, kind):
        return shape
    if isinstance(shape, AnyOf):
        for option in shape.options:
            found = _find(option, kind)
            if found is not None:
                return found
    return None


def _foreign_fields(value: BaseModel) -> frozenset[str]:
    return getattr(type(value), '__rekey_foreign__', frozenset())


def _parts(value: Any, shape: Shape) -> list[tuple[Any, Shape]]:
    """Return the items of a container ``value`` that can hold uuids, each with its shape."""
    if isinstance(value, Mapping):
        dict_shape = _find(shape, DictOf)
        if dict_shape is None:
            return []
        parts: list[tuple[Any, Shape]] = []
        for key, item in value.items():
            if dict_shape.key is not None:
                parts.append((key, dict_shape.key))
            if dict_shape.value is not None:
                parts.append((item, dict_shape.value))
        return parts
    if isinstance(value, (list, tuple, set, frozenset)):
        seq_shape = _find(shape, SeqOf)
        return [(item, seq_shape.item) for item in value] if seq_shape is not None else []
    return []


def _uuid_leaf(shape: Shape) -> UuidLeaf | None:
    if isinstance(shape, UuidLeaf):
        return shape
    if isinstance(shape, AnyOf):
        for option in shape.options:
            if isinstance(option, UuidLeaf):
                return option
    return None
