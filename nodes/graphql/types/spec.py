"""
Strawberry GraphQL types derived from InstanceSpec and NodeSpec Pydantic models.

These types mirror the Pydantic spec models stored in InstanceConfig.spec and
NodeConfig.spec, providing a structured query API for the model editor.

TranslatedString fields are flattened to str | None (serialized via str()).
AnyParameter and unmodeled blobs (input_datasets, dimensions) are exposed as JSON.
"""

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import strawberry as sb
from strawberry import auto
from strawberry.scalars import JSON

from kausal_common.strawberry.pydantic import StrawberryPydanticType, pydantic_type

from paths import gql
from paths.refs import DimensionRef

from nodes.defs.instance_defs import DatasetRepoSpec, InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import OutputMetricDef
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.graphql.types.constraints import EffectiveShapeType, effective_port_shape
from nodes.graphql.types.metric import DimensionalMetricType
from nodes.metric import DimensionalMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nodes.defs.port_def import InputPortDeclaration
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


@sb.type(
    name='InputPortDeclaration',
    description='One semantic input role a node class declares, with the ports currently instantiating it.',
)
class InputPortDeclarationType:
    role: str
    label: str | None = sb.field(description='Presentation fallback for the role before a port carries its own label.')
    multi: bool = sb.field(description='One port instance accepting many bindings as a homogeneous aggregate.')
    repeatable: bool = sb.field(description='Many heterogeneous port instances of this role (e.g. each factor of a product).')
    required: bool = sb.field(description='Whether computation requires at least one binding for this role.')
    aggregation: str | None = sb.field(description='Operation combining a multi port into one delivered value.')
    min_count: int
    default_count: int = sb.field(description='Port instances created by default at node creation.')
    instantiated_port_ids: list[UUID]

    @classmethod
    def from_declaration(
        cls,
        declaration: InputPortDeclaration,
        ports: Sequence[InputPortDef],
    ) -> InputPortDeclarationType:
        instantiated = [
            port.id
            for port in ports
            if (port.role is not None and str(port.role) == str(declaration.role))
            or (
                port.role is None and port.identifier is not None and str(port.identifier) == str(declaration.instance_identifier)
            )
        ]
        return cls(
            role=str(declaration.role),
            label=str(declaration.label) if declaration.label else None,
            multi=declaration.multi,
            repeatable=declaration.repeatable,
            required=declaration.required,
            aggregation=declaration.aggregation,
            min_count=declaration.min_count,
            default_count=declaration.effective_default_count,
            instantiated_port_ids=instantiated,
        )


@pydantic_type(model=InputPortDef)
class InputPortType(StrawberryPydanticType[InputPortDef]):
    id: auto
    identifier: auto
    label: auto
    role: auto
    quantity: auto
    unit: auto
    multi: auto
    required_dimensions: list[DimensionRef]
    supported_dimensions: list[DimensionRef] = sb.field(
        deprecation_reason='Never had solver semantics; effectiveShape carries the derived shape.',
    )
    bindings: list[Annotated['InputPortBinding', sb.lazy('nodes.schema')]] = sb.field(default_factory=list)

    _node_uuid: sb.Private[UUID | None] = None

    @sb.field(
        graphql_type=EffectiveShapeType | None,
        description='Solver-derived shape of the aggregate value delivered to this port.',
    )
    @staticmethod
    def effective_shape(root: 'InputPortType', info: gql.Info) -> EffectiveShapeType | None:
        if root._node_uuid is None:
            return None
        return effective_port_shape(info.context.require_constraint_solve(), root._node_uuid, root.id, 'input')

    @classmethod
    def from_def(cls, spec: InputPortDef, bindings: list[InputPortBinding], node_uuid: UUID | None = None) -> InputPortType:
        port = InputPortType(
            id=spec.id,
            identifier=spec.identifier,
            label=spec.label,
            role=spec.role,
            quantity=spec.quantity,
            unit=spec.unit,
            multi=spec.multi,
            required_dimensions=spec.required_dimensions,
            supported_dimensions=spec.supported_dimensions,
            bindings=bindings,
        )
        port._node_uuid = node_uuid
        return port


@pydantic_type(model=OutputPortDef)
class OutputPortType(StrawberryPydanticType[OutputPortDef]):
    id: auto
    identifier: auto
    label: auto
    role: auto
    quantity: auto
    unit: auto
    column_id: auto
    dimensions: list[DimensionRef]
    edges: list[Annotated['NodeEdgeType', sb.lazy('nodes.schema')]] = sb.field(default_factory=list)

    _node: sb.Private['Node | None'] = None
    _spec: sb.Private['OutputPortDef | None'] = None
    _node_uuid: sb.Private[UUID | None] = None

    @sb.field(
        graphql_type=EffectiveShapeType | None,
        description='Solver-derived shape of the value this port produces.',
    )
    @staticmethod
    def effective_shape(root: 'OutputPortType', info: gql.Info) -> EffectiveShapeType | None:
        if root._node_uuid is None:
            return None
        return effective_port_shape(info.context.require_constraint_solve(), root._node_uuid, root.id, 'output')

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
    def from_def(
        cls,
        spec: OutputPortDef,
        edges: list[NodeEdgeType],
        node: Node | None,
        node_uuid: UUID | None = None,
    ) -> OutputPortType:
        port = OutputPortType(
            id=spec.id,
            identifier=spec.identifier,
            label=spec.label,
            role=spec.role,
            quantity=spec.quantity,
            unit=spec.unit,
            column_id=spec.column_id,
            dimensions=spec.dimensions,
            edges=edges,
        )
        port._node = node
        port._spec = spec
        port._node_uuid = node_uuid
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


@pydantic_type(InstanceModelSpec, name='InstanceSpec')
class InstanceSpecType:
    config_source: str
    dataset_repo: DatasetRepoType | None
    years: YearsDefType
