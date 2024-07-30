from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, Set, cast

from pydantic import BaseModel, ValidationInfo, field_validator

from common.i18n import TranslatedString

if TYPE_CHECKING:
    from common.i18n import I18nString as CommonI18nString


_i18n_context: ContextVar['I18nContext'] = ContextVar('_i18n_context')

@dataclass
class I18nContext:
    default_language: str
    supported_languages: Set[str]

    @contextmanager
    def activate(self):
        token = _i18n_context.set(self)
        try:
            yield
        finally:
            _i18n_context.reset(token)


class I18nString(BaseModel):
    i18n: dict[str, str]
    # default_language: str

    @field_validator('i18n')
    @classmethod
    def validate_i18n(cls, value: dict[str, str], info: ValidationInfo):
        ctx = cast(I18nContext | None, info.context)
        if not ctx:
            raise ValueError("No i18n context")
        if ctx.default_language not in value:
            raise ValueError("No value for the default language (%s)" % ctx.default_language)
        for lang in value.keys():
            if lang not in ctx.supported_languages:
                raise ValueError("Value given for an unsupported language (%s)" % lang)
        return value

    @classmethod
    def from_common_i18n(cls, s: CommonI18nString) -> Self:
        ctx = _i18n_context.get()
        if isinstance(s, str):
            return cls(i18n={ctx.default_language: s})
        elif isinstance(s, TranslatedString):
            return cls(i18n=s.i18n)
        else:
            # StrPromise not okay
            raise ValueError("No promises!")

    def get_value(self) -> str:
        vals = list(self.i18n.values())
        return vals[0]

    def __init__(self, /, **data: Any) -> None:
        self.__pydantic_validator__.validate_python(
            data,
            self_instance=self,
            context=_i18n_context.get(),
        )
