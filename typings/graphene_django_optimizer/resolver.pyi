from typing import Any, Callable, Sequence, Type, TypeAlias, TypeVar, Union
from django.db.models import Prefetch


class noop: ...

F = TypeVar('F', bound=Callable)

SelectRelatedVal: TypeAlias = str
SelectRelated: TypeAlias = SelectRelatedVal | Callable[[Any], str]
PrefetchRelatedVal: TypeAlias = Union[str, Prefetch]
PrefetchRelated: TypeAlias = PrefetchRelatedVal | Callable[[Any], PrefetchRelatedVal]


def resolver_hints(
    model_field: str | Sequence[str] | None = None,
    select_related: SelectRelated | Sequence[SelectRelated] | Type[noop] = noop,
    prefetch_related: PrefetchRelated | Sequence[PrefetchRelated] | Type[noop] = noop,
    only: Sequence[str] | Type[noop] = noop,
) -> Callable[[F], F]: ...
