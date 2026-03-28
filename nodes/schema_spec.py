"""
Strawberry GraphQL types derived from InstanceSpec and NodeSpec Pydantic models.

These types mirror the Pydantic spec models stored in InstanceConfig.spec and
NodeConfig.spec, providing a structured query API for the model editor.

TranslatedString fields are flattened to str | None (serialized via str()).
AnyParameter and unmodeled blobs (input_datasets, dimensions) are exposed as JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import strawberry as sb
from strawberry.scalars import JSON

from kausal_common.strawberry.pydantic import StrawberryPydanticType

from nodes.defs.instance_defs import (
    DatasetRepoSpec,
    YearsSpec,
)
from nodes.defs.node_defs import OutputMetricDef
from nodes.defs.port_def import InputPortDef, OutputPortDef

if TYPE_CHECKING:
    from nodes.defs.instance_defs import ActionGroup, InstanceSpec
    from nodes.defs.node_defs import NodeSpec
    from nodes.scenario import Scenario

# ---------------------------------------------------------------------------
# Simple models — fully auto-derived via pydantic bridge
# ---------------------------------------------------------------------------


@sb.experimental.pydantic.type(model=YearsSpec, all_fields=True, name='ModelYears')
class YearsDefType(StrawberryPydanticType[YearsSpec]):
    pass


@sb.experimental.pydantic.type(model=DatasetRepoSpec, all_fields=True, name='ModelDatasetRepo')
class DatasetRepoType(StrawberryPydanticType[DatasetRepoSpec]):
    pass


# ---------------------------------------------------------------------------
# Models with TranslatedString — sb.auto for primitive fields, str | None for label
# ---------------------------------------------------------------------------


@sb.experimental.pydantic.type(model=OutputMetricDef, name='ModelOutputMetric')
class OutputMetricType(StrawberryPydanticType[OutputMetricDef]):
    id: str
    unit: str
    quantity: str
    label: str | None  # TranslatedString → str via from_pydantic extra=


@sb.experimental.pydantic.type(model=InputPortDef, name='ModelInputPort')
class InputPortType(StrawberryPydanticType[InputPortDef]):
    id: str
    quantity: str
    unit: str
    required_dimensions: list[str]
    supported_dimensions: list[str]
    label: str | None


@sb.experimental.pydantic.type(model=OutputPortDef, name='ModelOutputPort')
class OutputPortType(StrawberryPydanticType[OutputPortDef]):
    id: str
    quantity: str
    unit: str
    dimensions: list[str]
    label: str | None


# ---------------------------------------------------------------------------
# Scenario types — hand-written (TranslatedString + float|bool|str value union)
# ---------------------------------------------------------------------------


@sb.type(name='ModelScenarioParamOverride')
class ScenarioParamOverrideType:
    parameter_id: str
    node_id: str | None
    value: JSON  # float | bool | str — no clean scalar union in GQL


@sb.type(name='ModelScenarioSpec')
class ScenarioSpecType:
    id: str
    name: str | None
    description: str | None
    kind: str | None
    all_actions_enabled: bool
    params: list[ScenarioParamOverrideType]


# ---------------------------------------------------------------------------
# ActionGroup — hand-written (TranslatedString name)
# ---------------------------------------------------------------------------


@sb.type(name='ModelActionGroupSpec')
class ActionGroupSpecType:
    id: str
    name: str | None
    color: str | None
    order: int


# ---------------------------------------------------------------------------
# NodeSpec — TypeConfig flattened to kind + formula + decision_level
# ---------------------------------------------------------------------------


@sb.type(name='ModelNodeSpec')
class NodeSpecType:
    node_class: str
    # TypeConfig flattened — avoids a discriminated union in GQL
    kind: str  # 'simple' | 'formula' | 'action'
    formula: str | None
    decision_level: str | None
    action_group: str | None
    action_parent: str | None
    no_effect_value: float | None
    # Ports and metrics
    input_ports: list[InputPortType]
    output_ports: list[OutputPortType]
    output_metrics: list[OutputMetricType]
    # Dimensions
    input_dimensions: list[str]
    output_dimensions: list[str]
    # Flags
    is_outcome: bool
    minimum_year: int | None
    allow_nulls: bool
    node_group: str | None
    # Not yet modeled — exposed as opaque blobs
    input_datasets: JSON
    params: JSON
    goals: JSON
    visualizations: JSON
    pipeline: JSON | None
    extra: JSON


# ---------------------------------------------------------------------------
# InstanceSpec
# ---------------------------------------------------------------------------


@sb.type(name='ModelInstanceSpec')
class InstanceSpecType:
    years: YearsDefType
    dataset_repo: DatasetRepoType | None
    action_groups: list[ActionGroupSpecType]
    scenarios: list[ScenarioSpecType]
    # Not yet modeled — exposed as opaque blobs
    features: JSON
    terms: JSON
    result_excels: JSON
    pages: JSON
    impact_overviews: JSON
    normalizations: JSON
    params: JSON
    dimensions: JSON


# ---------------------------------------------------------------------------
# Converter functions: Pydantic → GraphQL
# ---------------------------------------------------------------------------


def _ts(val: Any) -> str | None:
    """Serialize a TranslatedString (or None) to str."""
    return str(val) if val is not None else None


def action_group_to_gql(ag: ActionGroup) -> ActionGroupSpecType:
    return ActionGroupSpecType(
        id=ag.id,
        name=_ts(ag.name),
        color=ag.color,
        order=ag.order,
    )


def scenario_to_gql(s: Scenario) -> ScenarioSpecType:
    return ScenarioSpecType(
        id=s.id,
        name=_ts(s.name),
        description=_ts(s.description),
        kind=s.kind.value if s.kind else None,
        all_actions_enabled=s.all_actions_enabled,
        params=[
            ScenarioParamOverrideType(
                parameter_id=param_id,
                node_id=None,
                value=cast('JSON', value),
            )
            for param_id, value in s.param_values.items()
        ],
    )


def output_metric_to_gql(m: OutputMetricDef) -> OutputMetricType:
    return OutputMetricType.from_pydantic(m, extra={'label': _ts(m.label)})


def input_port_to_gql(p: InputPortDef) -> InputPortType:
    return InputPortType.from_pydantic(p, extra={'label': _ts(p.label)})


def output_port_to_gql(p: OutputPortDef) -> OutputPortType:
    return OutputPortType.from_pydantic(p, extra={'label': _ts(p.label)})


def node_spec_to_gql(spec: NodeSpec) -> NodeSpecType:
    tc = spec.type_config
    return NodeSpecType(
        node_class=spec.node_class,
        kind=tc.kind,
        formula=getattr(tc, 'formula', None),
        decision_level=getattr(tc, 'decision_level', None),
        action_group=getattr(tc, 'group', None),
        action_parent=getattr(tc, 'parent', None),
        no_effect_value=getattr(tc, 'no_effect_value', None),
        input_ports=[input_port_to_gql(p) for p in spec.input_ports],
        output_ports=[output_port_to_gql(p) for p in spec.output_ports],
        output_metrics=[output_metric_to_gql(m) for m in spec.output_metrics],
        input_dimensions=spec.input_dimensions,
        output_dimensions=spec.output_dimensions,
        is_outcome=spec.is_outcome,
        minimum_year=spec.minimum_year,
        allow_nulls=spec.allow_nulls,
        node_group=spec.node_group,
        input_datasets=cast('JSON', spec.input_datasets),
        params=cast('JSON', [p.model_dump() for p in spec.params]),
        goals=cast('JSON', spec.goals.model_dump()),
        visualizations=cast('JSON', spec.visualizations.model_dump()),
        pipeline=cast('JSON', spec.pipeline),
        extra=cast('JSON', spec.extra.model_dump()),
    )


def instance_spec_to_gql(spec: InstanceSpec) -> InstanceSpecType:
    return InstanceSpecType(
        years=YearsDefType.from_pydantic(spec.years),
        dataset_repo=DatasetRepoType.from_pydantic(spec.dataset_repo) if spec.dataset_repo else None,
        action_groups=[action_group_to_gql(ag) for ag in spec.action_groups],
        scenarios=[scenario_to_gql(s) for s in spec.scenarios],
        features=cast('JSON', spec.features.model_dump()),
        terms=cast('JSON', spec.terms.model_dump(exclude_none=True)),
        result_excels=cast('JSON', [result.model_dump(exclude_none=True) for result in spec.result_excels]),
        pages=cast('JSON', [page.model_dump(exclude_none=True) for page in spec.pages]),
        impact_overviews=cast('JSON', [overview.model_dump(exclude_none=True) for overview in spec.impact_overviews]),
        normalizations=cast('JSON', [normalization.model_dump(exclude_none=True) for normalization in spec.normalizations]),
        params=cast('JSON', [p.model_dump() for p in spec.params]),
        dimensions=cast('JSON', spec.dimensions),
    )
