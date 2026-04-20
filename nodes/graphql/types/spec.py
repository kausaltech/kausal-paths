"""
Strawberry GraphQL types derived from InstanceSpec and NodeSpec Pydantic models.

These types mirror the Pydantic spec models stored in InstanceConfig.spec and
NodeConfig.spec, providing a structured query API for the model editor.

TranslatedString fields are flattened to str | None (serialized via str()).
AnyParameter and unmodeled blobs (input_datasets, dimensions) are exposed as JSON.
"""

from typing import TYPE_CHECKING, Annotated

import strawberry as sb
from strawberry import auto
from strawberry.scalars import JSON

from kausal_common.strawberry.pydantic import StrawberryPydanticType, pydantic_type

from paths.refs import DimensionRef

from nodes.defs.instance_defs import DatasetRepoSpec, InstanceSpec, YearsSpec
from nodes.defs.node_defs import OutputMetricDef
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.graphql.types.metric import DimensionalMetricType
from nodes.metric import DimensionalMetric

if TYPE_CHECKING:
    from nodes.node import Node
    from nodes.schema import InputPortBinding, NodeEdgeType
    from params.schema import ParameterInterface


@pydantic_type(model=YearsSpec, all_fields=True)
class YearsDefType(StrawberryPydanticType[YearsSpec]):
    pass


@pydantic_type(model=DatasetRepoSpec, all_fields=True)
class DatasetRepoType(StrawberryPydanticType[DatasetRepoSpec]):
    pass


@pydantic_type(model=OutputMetricDef)
class OutputMetricType(StrawberryPydanticType[OutputMetricDef]):
    id: auto
    label: auto
    unit: auto
    quantity: auto


@pydantic_type(model=InputPortDef)
class InputPortType(StrawberryPydanticType[InputPortDef]):
    id: auto
    label: auto
    quantity: auto
    unit: auto
    multi: auto
    required_dimensions: list[DimensionRef]
    supported_dimensions: list[DimensionRef]
    bindings: list[Annotated['InputPortBinding', sb.lazy('nodes.schema')]] = sb.field(default_factory=list)

    @classmethod
    def from_def(cls, spec: InputPortDef, bindings: list[InputPortBinding]) -> InputPortType:
        return InputPortType(
            id=spec.id,
            label=spec.label,
            quantity=spec.quantity,
            unit=spec.unit,
            multi=spec.multi,
            required_dimensions=spec.required_dimensions,
            supported_dimensions=spec.supported_dimensions,
            bindings=bindings,
        )


@pydantic_type(model=OutputPortDef)
class OutputPortType(StrawberryPydanticType[OutputPortDef]):
    id: auto
    label: auto
    quantity: auto
    unit: auto
    column_id: auto
    dimensions: list[DimensionRef]
    edges: list[Annotated['NodeEdgeType', sb.lazy('nodes.schema')]] = sb.field(default_factory=list)

    _node: sb.Private['Node | None'] = None
    _spec: sb.Private['OutputPortDef | None'] = None

    @sb.field(graphql_type=DimensionalMetricType | None)
    @staticmethod
    def output(root: 'OutputPortType') -> DimensionalMetric | None:
        from nodes.actions.action import ActionNode
        from nodes.metric import DimensionalMetric

        if root._node is None or root._spec is None:
            return None
        if isinstance(root._node, ActionNode):
            # FIXME: Doesn't work yet for action nodes
            return None
        return DimensionalMetric.from_output_port(root._node, root._spec)

    @classmethod
    def from_def(cls, spec: OutputPortDef, edges: list[NodeEdgeType], node: Node | None) -> OutputPortType:
        port = OutputPortType(
            id=spec.id,
            label=spec.label,
            quantity=spec.quantity,
            unit=spec.unit,
            column_id=spec.column_id,
            dimensions=spec.dimensions,
            edges=edges,
        )
        port._node = node
        port._spec = spec
        return port


@sb.type(name='ModelScenarioParamOverride')
class ScenarioParamOverride:
    parameter: Annotated['ParameterInterface', sb.lazy('params.schema')]
    value: JSON


@sb.type(name='ModelScenarioSpec')
class ScenarioSpecType:
    id: str
    name: str | None
    description: str | None
    kind: str | None
    all_actions_enabled: bool
    params: list[ScenarioParamOverride]


@pydantic_type(InstanceSpec, name='InstanceSpec')
class InstanceSpecType:
    config_source: str
    dataset_repo: DatasetRepoType | None
    years: YearsDefType
