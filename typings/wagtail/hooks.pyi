from typing import Callable, overload, TypeVar


FN = TypeVar("FN", bound=Callable)


@overload
def register(hook_name: str, fn: Callable, order: int = ...) -> None: ...

@overload
def register(hook_name: str) -> Callable[[FN], FN]: ...


def get_hooks(hook_name: str) -> list[Callable]: ...
