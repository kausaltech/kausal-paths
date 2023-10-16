from typing import TypeVar
from graphene import Field

T = TypeVar('T', bound=Field)

def field(field_type: T, *args, **kwargs) -> T: ...
