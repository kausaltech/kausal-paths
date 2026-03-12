from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from kausal_common.i18n.pydantic import TranslatedString

from common.types import MetricIdentifier
from params import AnyParameter

from .port_def import InputPortDef, OutputPortDef


class OutputMetricDef(BaseModel):
    """A single output metric produced by a node."""

    id: MetricIdentifier
    label: TranslatedString | None = None
    unit: str
    quantity: str = ''


class FormulaConfig(BaseModel):
    """Type-specific config for formula nodes."""

    kind: Literal['formula'] = 'formula'
    formula: str


class ActionConfig(BaseModel):
    """Type-specific config for action nodes."""

    kind: Literal['action'] = 'action'
    decision_level: str | None = None


class SimpleConfig(BaseModel):
    """Type-specific config for nodes that are fully defined by their Python class."""

    kind: Literal['simple'] = 'simple'


TypeConfig = Annotated[
    FormulaConfig | ActionConfig | SimpleConfig,
    Field(discriminator='kind'),
]


class NodeSpec(BaseModel):
    """Computation schema for a node, stored as a SchemaField on NodeConfig."""

    type_config: TypeConfig = Field(default_factory=SimpleConfig)

    # Inputs
    input_ports: list[InputPortDef] = Field(default_factory=list)

    # Outputs
    output_ports: list[OutputPortDef] = Field(default_factory=list)
    output_metrics: list[OutputMetricDef] = Field(default_factory=list)

    # Computation
    pipeline: list[dict[str, object]] | None = None
    params: list[AnyParameter] = Field(default_factory=list)

    # Node behaviour flags
    is_outcome: bool = False
