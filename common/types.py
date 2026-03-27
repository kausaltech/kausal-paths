from __future__ import annotations

from typing import Annotated, Literal, overload

from pydantic import Field, TypeAdapter

MixedCaseIdentifier = Annotated[str, Field(pattern=r'^[A-Za-z0-9_]+$')]
Identifier = Annotated[str, Field(pattern=r'^[a-z0-9_]+$')]

# Semantic identifier types for cross-referencing within defs.
# The pattern is the same as Identifier; the distinct types exist so that
# validators can check referential integrity (e.g. "does this node exist?").
NodeIdentifier = Annotated[str, Field(pattern=r'^[a-z0-9_]+$')]
ActionGroupIdentifier = Annotated[str, Field(pattern=r'^[a-z0-9_]+$')]
ParameterLocalId = Annotated[str, Field(pattern=r'^[a-z0-9_]+$')]
ParameterGlobalId = Annotated[str, Field(pattern=r'^[a-z0-9_]+(\.[a-z0-9_]+)?$')]
ScenarioIdentifier = Annotated[str, Field(pattern=r'^[a-z0-9_]+$')]
MetricIdentifier = Annotated[str, Field(pattern=r'^[A-Za-z0-9_]+$')]
NodePortIdentifier = Annotated[str, Field(pattern=r'^[A-Za-z0-9_-]+$')]

MixedCaseIdentifierAdapter = TypeAdapter(MixedCaseIdentifier)
IdentifierAdapter = TypeAdapter(Identifier)


@overload
def validate_identifier(s: str, mixed: Literal[True]) -> MixedCaseIdentifier: ...


@overload
def validate_identifier(s: str, mixed: Literal[False] = ...) -> Identifier: ...


def validate_identifier(s: str, mixed: bool = False):
    if mixed:
        return MixedCaseIdentifierAdapter.validate_python(s)
    return IdentifierAdapter.validate_python(s)
