from __future__ import annotations

from typing import Annotated, Literal, overload
from uuid import UUID

from pydantic import Field, TypeAdapter, ValidationError

from .uuid_kinds import Identity

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
# Shapes share the dataset pattern so a framework can prefix its own (`bisko/end_energy`); nothing parses it.
ShapeIdentifier = Annotated[str, Field(pattern=DATASET_IDENTIFIER_PATTERN)]
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


def identifier_or_none(value: str | None) -> MixedCaseIdentifier | None:
    """
    Return ``value`` if it is usable as an identifier, else None.

    For deriving optional identifiers from strings that may or may not be
    identifier-shaped (labels, dataset column headings). Callers that need a
    valid identifier should use ``validate_identifier`` and let it raise.
    """
    if value is None:
        return None
    try:
        return MixedCaseIdentifierAdapter.validate_python(value)
    except ValidationError:
        return None


@overload
def validate_identifier(s: str, mixed: Literal[True]) -> MixedCaseIdentifier: ...


@overload
def validate_identifier(s: str, mixed: Literal[False] = ...) -> Identifier: ...


def validate_identifier(s: str, mixed: bool = False):
    if mixed:
        return MixedCaseIdentifierAdapter.validate_python(s)
    return IdentifierAdapter.validate_python(s)


# -- uuids that define an entity (see `paths.uuid_kinds`) ----------------------
# A copy mints a new uuid for each. The `*Ref` types in `paths.refs` point at them.

InstanceId = Annotated[UUID, Identity('instance')]
NodeId = Annotated[UUID, Identity('node')]
"""A node's uuid. The runtime `Node.id` is its human-readable `NodeIdentifier`, not this."""
PortId = Annotated[UUID, Identity('port')]
BindingId = Annotated[UUID, Identity('binding')]
ActionGroupId = Annotated[UUID, Identity('action_group')]
DimensionId = Annotated[UUID, Identity('dimension')]
DimensionCategoryId = Annotated[UUID, Identity('category')]
DatasetId = Annotated[UUID, Identity('dataset')]
DatasetSchemaId = Annotated[UUID, Identity('dataset_schema')]
DatasetMetricId = Annotated[UUID, Identity('metric')]
ValidationRuleId = Annotated[UUID, Identity('validation_rule')]
ShapeId = Annotated[UUID, Identity('shape')]
ShapeCombinationId = Annotated[UUID, Identity('shape_combination')]
ShapeRequiredGroupId = Annotated[UUID, Identity('shape_required_group')]
DataPointId = Annotated[UUID, Identity('data_point')]
DataPointCommentId = Annotated[UUID, Identity('comment')]
DataSourceId = Annotated[UUID, Identity('data_source')]
SourceReferenceId = Annotated[UUID, Identity('source_reference')]
DataEntrySectionId = Annotated[UUID, Identity('data_entry_section')]
DataEntryTableId = Annotated[UUID, Identity('data_entry_table')]
