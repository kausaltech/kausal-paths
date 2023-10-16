from typing import Callable, overload, TypeVar


FN = TypeVar("FN", bound=Callable)


@overload
def register(hook_name: str, fn: Callable, order: int = ...): ...


@overload
def register(hook_name: str, fn: None = None, order: int = ...) -> Callable[[FN], FN]: ...


def get_hooks(hook_name) -> list[Callable]: ...
