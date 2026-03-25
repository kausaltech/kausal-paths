from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from kausal_common.i18n.pydantic import TranslatedString

from common.types import MetricIdentifier
from params.discover import AnyParameter

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


class NodeSpecExtra(BaseModel):
    """
    Attic for legacy node config fields.

    These fields are passed through to the InstanceLoader config dict
    but are not part of the long-term NodeSpec schema. Each field here
    is a candidate for removal once we stop relying on the corresponding
    YAML-era feature.
    """

    historical_values: list[tuple[int, float]] | None = None
    forecast_values: list[tuple[int, float]] | None = None
    input_dataset_processors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # Catch-all for anything else the node config had
    other: dict[str, Any] = Field(default_factory=dict)


class NodeSpec(BaseModel):
    """Computation schema for a node, stored as a SchemaField on NodeConfig."""

    # Python class path, e.g. "nodes.simple.AdditiveNode"
    node_class: str = ''

    type_config: TypeConfig = Field(default_factory=SimpleConfig)

    # Inputs
    input_ports: list[InputPortDef] = Field(default_factory=list)

    # Outputs
    output_ports: list[OutputPortDef] = Field(default_factory=list)
    output_metrics: list[OutputMetricDef] = Field(default_factory=list)

    # Datasets and dimensions — raw configs, will be properly modeled later
    input_datasets: list[dict[str, Any]] = Field(default_factory=list)
    input_dimensions: list[str] = Field(default_factory=list)
    output_dimensions: list[str] = Field(default_factory=list)

    # Computation
    pipeline: list[dict[str, object]] | None = None
    params: list[AnyParameter] = Field(default_factory=list)

    # Node behaviour flags
    is_outcome: bool = False

    # Legacy fields — see NodeSpecExtra docstring
    extra: NodeSpecExtra = NodeSpecExtra()
