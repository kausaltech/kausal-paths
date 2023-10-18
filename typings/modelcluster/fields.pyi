from typing import TypeVar
from django.db.models.fields.related import ForeignKey, ManyToManyField, ManyToManyDescriptor

# __set__ value type
_ST = TypeVar("_ST")
# __get__ return type
_GT = TypeVar("_GT")


class ParentalKey(ForeignKey[_GT, _ST]):  # pyright: ignore
    pass


class ParentalManyToManyField(ManyToManyField[_ST, _GT]):  # pyright: ignore
    pass


class ParentalManyToManyDescriptor(ManyToManyDescriptor):
    pass
