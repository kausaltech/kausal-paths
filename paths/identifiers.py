from __future__ import annotations

from typing import Annotated, Literal, overload
from uuid import UUID

from pydantic import Field, TypeAdapter

"""
Canonical identifier shapes for Paths domain objects.

This module is intended to hold structural identifier definitions only:
what a valid identifier looks like syntactically. It should not depend on a
runtime Context or perform referential integrity checks.

Migration note:
- Existing code still primarily imports these definitions from `common.types`.
- This module is introduced first as scaffolding so identifier and reference
  concepts have explicit homes before we migrate call sites.
"""


LOWER_IDENTIFIER_PATTERN = r'^[a-z0-9_]+$'
MIXED_IDENTIFIER_PATTERN = r'^[A-Za-z0-9_]+$'
GLOBAL_PARAMETER_ID_PATTERN = r'^[a-z0-9_]+(\.[a-z0-9_]+)?$'
NODE_PORT_IDENTIFIER_PATTERN = r'^[A-Za-z0-9_:-]+$'
DATASET_IDENTIFIER_PATTERN = r'^[A-Za-z0-9_/-]+$'

MixedCaseIdentifier = Annotated[str, Field(pattern=MIXED_IDENTIFIER_PATTERN)]
Identifier = Annotated[str, Field(pattern=LOWER_IDENTIFIER_PATTERN)]

NodeIdentifier = Annotated[str, Field(pattern=LOWER_IDENTIFIER_PATTERN)]
ActionGroupIdentifier = Annotated[str, Field(pattern=LOWER_IDENTIFIER_PATTERN)]
DimensionIdentifier = Annotated[str, Field(pattern=LOWER_IDENTIFIER_PATTERN)]
DimensionCategoryIdentifier = Annotated[str, Field(pattern=LOWER_IDENTIFIER_PATTERN)]
ParameterLocalId = Annotated[str, Field(pattern=LOWER_IDENTIFIER_PATTERN)]
ParameterGlobalId = Annotated[str, Field(pattern=GLOBAL_PARAMETER_ID_PATTERN)]
ScenarioIdentifier = Annotated[str, Field(pattern=LOWER_IDENTIFIER_PATTERN)]
MetricIdentifier = Annotated[str, Field(pattern=MIXED_IDENTIFIER_PATTERN)]
NodeOutputMetricIdentifier = MetricIdentifier
NodeOutputDimensionIdentifier = DimensionIdentifier
NodePortIdentifier = UUID
DatasetIdentifier = Annotated[str, Field(pattern=DATASET_IDENTIFIER_PATTERN)]
QuantityKindIdentifier = Annotated[str, Field(pattern=LOWER_IDENTIFIER_PATTERN)]


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
