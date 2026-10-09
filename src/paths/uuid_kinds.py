"""
What a uuid field means, carried on its type.

Every uuid in a model snapshot is one of four kinds, marked in its ``Annotated`` type:

- `Identity`: the field defines the entity. A copy mints a new uuid for it.
- `Ref`: the field points at an entity defined elsewhere, in the same document or not.
  A copy rewrites it when its target was copied and keeps it otherwise.
- `Provenance`: the field records where something came from (``copy_of``). A copy never
  rewrites it, but sets it to the source.
- `Token`: the value belongs to one database's state (an optimistic-locking token) and
  means nothing in a copy, which drops it.

The marked types are defined in `paths.identifiers` (`NodeId`, ...) and `paths.refs`
(`NodeRef`, ...). `uuid_slots` reads the marks off a model class, and `nodes.rekey`
walks a document by them.
"""

import types
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Literal, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel

type UuidEntity = Literal[
    'instance',
    'node',
    'port',
    'binding',
    'action_group',
    'dimension',
    'category',
    'dataset',
    'dataset_schema',
    'metric',
    'validation_rule',
    'shape',
    'shape_combination',
    'shape_required_group',
    'data_point',
    'comment',
    'data_source',
    'source_reference',
    'data_entry_section',
    'data_entry_table',
]


@dataclass(frozen=True, slots=True)
class Identity:
    entity: UuidEntity


@dataclass(frozen=True, slots=True)
class Ref:
    entity: UuidEntity


@dataclass(frozen=True, slots=True)
class Provenance:
    entity: UuidEntity


@dataclass(frozen=True, slots=True)
class Token:
    pass


type UuidKind = Identity | Ref | Provenance | Token
_KINDS = (Identity, Ref, Provenance, Token)


# -- annotation shapes ---------------------------------------------------------
# A field's annotation reduced to what matters for uuids: where a marked uuid sits, where
# a model sits (walked by its runtime class), and the containers in between.


@dataclass(frozen=True, slots=True)
class UuidLeaf:
    kind: UuidKind | None
    """None for a uuid with no mark, which `unmarked_uuid_fields` reports."""


@dataclass(frozen=True, slots=True)
class ModelLeaf:
    model: type[BaseModel]
    """The declared class; a walker visits a value by its runtime class, which may be a subclass."""


@dataclass(frozen=True, slots=True)
class SeqOf:
    item: Shape


@dataclass(frozen=True, slots=True)
class DictOf:
    key: Shape | None
    value: Shape | None


@dataclass(frozen=True, slots=True)
class AnyOf:
    options: tuple[Shape, ...]


type Shape = UuidLeaf | ModelLeaf | SeqOf | DictOf | AnyOf


def _kind_of(metadata: Iterable[object]) -> UuidKind | None:
    kinds = [item for item in metadata if isinstance(item, _KINDS)]
    if len(kinds) > 1:
        raise TypeError(f'A uuid carries more than one kind: {kinds}')
    return kinds[0] if kinds else None


def _non_empty(shapes: Iterable[Shape | None]) -> list[Shape]:
    return [shape for shape in shapes if shape is not None]


def _one_of(shapes: list[Shape]) -> Shape | None:
    if not shapes:
        return None
    return shapes[0] if len(shapes) == 1 else AnyOf(tuple(shapes))


def _container_shape(origin: object, args: tuple[object, ...]) -> Shape | None:
    if not isinstance(origin, type):
        return None
    if issubclass(origin, Mapping):
        key, value = (shape_of(arg) for arg in args) if args else (None, None)
        return DictOf(key, value) if key is not None or value is not None else None
    if issubclass(origin, (Sequence, set, frozenset)):
        item = _one_of(_non_empty(shape_of(arg) for arg in args if arg is not Ellipsis))
        return SeqOf(item) if item is not None else None
    return None


def shape_of(annotation: object, metadata: tuple[object, ...] = ()) -> Shape | None:
    """Reduce ``annotation`` to its uuid shape, or None when it can hold no uuid and no model."""
    if isinstance(annotation, typing.TypeAliasType):
        return shape_of(annotation.__value__, metadata)
    origin = get_origin(annotation)
    if origin is Annotated:
        inner, *extra = get_args(annotation)
        return shape_of(inner, (*metadata, *extra))
    if annotation is UUID:
        return UuidLeaf(_kind_of(metadata))
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return ModelLeaf(annotation)
    if origin is types.UnionType:
        return _one_of(_non_empty(shape_of(arg, metadata) for arg in get_args(annotation)))
    return _container_shape(origin, get_args(annotation))


_REGISTERED: dict[type[BaseModel], dict[str, Shape]] = {}


def register_uuid_kinds(model: type[BaseModel], **fields: Shape) -> None:
    """
    Give the uuid fields of ``model`` their kinds from outside its definition.

    For models defined where `paths.uuid_kinds` cannot be imported, such as the ones
    `kausal_common` shares with Watch. Must run before `uuid_slots` first reads ``model``.
    """
    _REGISTERED[model] = fields


@cache
def uuid_slots(model: type[BaseModel]) -> tuple[tuple[str, Shape], ...]:
    """Return the fields of ``model`` that can hold a uuid or a model, with their shapes."""
    registered = _REGISTERED.get(model, {})
    slots: list[tuple[str, Shape]] = []
    for name, field in model.model_fields.items():
        shape = registered.get(name) or shape_of(field.annotation, tuple(field.metadata))
        if shape is not None:
            slots.append((name, shape))
    return tuple(slots)


def _shape_parts(shape: Shape) -> Iterable[Shape]:
    match shape:
        case SeqOf(item=item):
            return (item,)
        case DictOf(key=key, value=value):
            return _non_empty((key, value))
        case AnyOf(options=options):
            return options
    return ()


def unmarked_uuid_fields(root: type[BaseModel]) -> list[str]:
    """Return every ``Model.field`` reachable from ``root`` that holds a uuid with no kind."""
    found: set[str] = set()
    seen: set[type[BaseModel]] = set()
    models = [root]
    while models:
        model = models.pop()
        if model in seen:
            continue
        seen.add(model)
        models.extend(model.__subclasses__())
        for name, slot in uuid_slots(model):
            shapes = [slot]
            while shapes:
                shape = shapes.pop()
                if isinstance(shape, UuidLeaf) and shape.kind is None:
                    found.add(f'{model.__name__}.{name}')
                elif isinstance(shape, ModelLeaf):
                    models.append(shape.model)
                shapes.extend(_shape_parts(shape))
    return sorted(found)
