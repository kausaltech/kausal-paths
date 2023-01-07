from __future__ import annotations

from typing import TypeAlias, Union
from pydantic import ConstrainedStr, errors


class MixedCaseIdentifier(ConstrainedStr):
    regex = r'^[A-Za-z_]+$'

    @classmethod
    def validate(cls, value: str) -> MixedCaseIdentifier:
        try:
            return super().validate(value)  # type: ignore
        except Exception as e:
            raise ValueError("string is not a valid identifier (only ASCII letters and '_' allowed)")


class Identifier(ConstrainedStr):
    regex = r'^[a-z_]+$'

    @classmethod
    def validate(cls, value: str) -> Identifier:
        try:
            return super().validate(value)  # type: ignore
        except Exception as e:
            raise ValueError("string is not a valid identifier (only lower case and '_' allowed)")
