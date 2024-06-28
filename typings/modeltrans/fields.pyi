from typing import Literal, Sequence
from collections.abc import Generator
from django.db.models import JSONField, Field


class TranslatedVirtualField:
    original_field: Field
    language: str | None
    default_language_field: str | None
    blank: bool
    null: bool
    concrete: Literal[False]




class TranslationField(JSONField):
    description: str
    fields: Sequence[str]
    default_language_field: str | None

    def __new__(
        cls,
        fields: Sequence[str] | None = ...,
        default_language_field: str | None = ...,
        required_languages: list[str] | dict | None = ...,
        virtual_fields: bool = ...,
        fallback_language_field: str | None = ...,
        *args, **kwargs
    ) -> TranslationField: ...

    def __init__(
        self,
        fields: Sequence[str] | None = ...,
        default_language_field: str | None = ...,
        required_languages: list[str] | dict | None = ...,
        virtual_fields: bool = ...,
        fallback_language_field: str | None = ...,
        *args, **kwargs
    ) -> None: ...
    def get_translated_fields(self) -> Generator[TranslatedVirtualField, None, None]: ...
