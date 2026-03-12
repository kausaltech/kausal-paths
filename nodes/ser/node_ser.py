from __future__ import annotations

import abc
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr

from nodes.ser.i18n_ser import I18nContext, I18nString

# from nodes.dimensions import Dimension, DimensionCategory, DimensionCategoryGroup

if TYPE_CHECKING:
    from ..edges import Edge
    from ..instance import Instance
    from ..node import Node, NodeMetric


IDENTIFIER_RE = r'^[a-z0-9_-]+$'

Identifier = Annotated[str, Field(pattern=IDENTIFIER_RE)]


class UnitSer(BaseModel):
    unit: str
    "Unit (compatible with SI)"

    specifier: I18nString | None = None
    "Optional specifier for the unit (human readable)"


class IdentifiedModel(BaseModel):
    id: Identifier = Field(pattern=IDENTIFIER_RE, default_factory=lambda: str(uuid4()))
    "Identifier of the object."


class NodePort(IdentifiedModel):
    label: I18nString | None = None
    "Human-readable label for the port."

    unit: UnitSer
    quantity: str

    _node: Node = PrivateAttr()


class NodeInputPort(NodePort):
    required_dimensions: list[str] = Field(default_factory=list)
    supported_dimensions: list[str] = Field(default_factory=list)


class NodeOutputPort(NodePort):
    dimensions: list[str] = Field(default_factory=list)


class NodeType(IdentifiedModel):
    python_class: str
    "Python class name for the node type"


class NodeSer(IdentifiedModel):
    name: I18nString
    "Human-readable label for the Node instance."

    type: str
    "Type of the node"

    short_name: I18nString | None = None
    "Shortened label for the node"

    description: I18nString | None = None
    "Long description for the node"

    # if the node has an established visualisation color
    color: str | None = None
    "Color to use when visualising this node, if possible"

    node_group: str | None = None
    "Set if node is part of a node group"

    # if this node should have its own outcome page
    is_outcome: bool = False

    input_ports: list[NodeInputPort] = Field(default_factory=list)
    output_ports: list[NodeOutputPort] = Field(default_factory=list)

    _model: PathsModelConfig = PrivateAttr()

    @classmethod
    def from_node(cls, model: PathsModelConfig, node: Node) -> NodeSer:
        node_cls = type(node)

        output_ports: list[NodeOutputPort] = []
        outputs_by_metric: dict[str, NodeOutputPort] = {}
        for idx, (metric_id, metric) in enumerate(node.output_metrics.items()):
            label = ('Output %d' % idx) if metric.label is None else metric.label
            output_port = NodeOutputPort(
                id='%s__output%d' % (node.id, idx),
                label=I18nString.from_common_i18n(label),
                unit=UnitSer(unit=str(metric.unit)),
                quantity=metric.quantity
            )
            output_ports.append(output_port)
            outputs_by_metric[metric_id] = output_port

        out = cls(
            id=node.id,
            name=I18nString.from_common_i18n(node.name),
            type='%s.%s' % (node_cls.__module__, node_cls.__qualname__),
            output_ports=output_ports,
            color=node.color,
            node_group=node.node_group,
        )
        out._model = model
        return out


class ActionNodeSer(NodeSer):
    pass


class EdgeTransformation(BaseModel, abc.ABC):
    type: str

    label: str | None
    "Label for the processing step"


class SelectCategoriesEdgeTransformation(EdgeTransformation):
    type: Literal['select_categories']

    class SelectionMode(str, Enum):
        include = 'include'
        exclude = 'exclude'

    dimension_id: str
    "Which dimension to operate on."

    category_ids: list[str]
    "List of categories to select."

    flatten: bool = False
    "Whether to remove the dimension after selection. Will calculate a sum over the remaining categories."

    exclude: str = SelectionMode.include
    "Whether to include or exclude the selected categories"


class AssignCategoryEdgeTransformation(EdgeTransformation):
    type: Literal['assign_category']
    # TYPE: ClassVar = 'assign_category'

    dimension_id: str
    "Which dimension to add."

    category_id: str
    "Category to assign."


class DatasetPort(BaseModel):
    dataset_id: Identifier


class EdgeNodePort(BaseModel):
    node_id: Identifier
    port_id: Identifier

    _node: NodeSer = PrivateAttr()
    _port: NodePort = PrivateAttr()


class EdgeSer(IdentifiedModel):
    source: EdgeNodePort
    sink: EdgeNodePort
    transformations: list[EdgeTransformation] = Field(default_factory=list)

    @staticmethod
    def _make_input_ports(edge: Edge, nodes: dict[str, NodeSer]) -> EdgeNodePort:
        output_node = edge.output_node
        output_node_ser = nodes[output_node.id]
        input_node = edge.input_node
        input_node_ser = nodes[input_node.id]

        metrics: list[tuple[str, NodeMetric]] = []
        if len(edge.metrics):
            for m_id in edge.metrics:
                m = input_node.output_metrics[m_id]
                metrics.append((m_id, m))
        else:
            m = output_node.get_default_output_metric()
            metrics.append((m.quantity, m))

        sink_port: NodeInputPort | None = None

        for _, (m_id, m) in enumerate(metrics):
            port_nr = len(output_node_ser.input_ports)
            port = NodeInputPort(
                id='%s__input%d' % (output_node.id, port_nr),
                label=I18nString.from_common_i18n(m_id),
                unit=UnitSer(
                    unit=str(m.unit),
                ),
                quantity=m.quantity,
            )
            output_node_ser.input_ports.append(port)
            sink_port = port
        assert sink_port is not None
        enp = EdgeNodePort(node_id=output_node_ser.id, port_id=sink_port.id)
        enp._node = output_node_ser
        return enp

    @classmethod
    def from_edge(cls, edge: Edge, nodes: dict[str, NodeSer]) -> Self:
        src_node = nodes[edge.input_node.id]
        src = EdgeNodePort(node_id=src_node.id, port_id=src_node.output_ports[0].id)
        src._node = src_node
        sink = cls._make_input_ports(edge, nodes)
        edge_id = '%s__%s' % (src_node.id, edge.output_node.id)
        return cls(id=edge_id, source=src, sink=sink)


class PathsModelConfig(BaseModel):
    id: Identifier
    nodes: list[NodeSer] = Field(default_factory=list)
    edges: list[EdgeSer] = Field(default_factory=list)
    #dimensions: list[Dimension] = Field(default_factory=list)

    @classmethod
    def from_instance(cls, instance: Instance) -> PathsModelConfig:
        ctx = instance.context
        i18n_ctx = I18nContext(default_language=instance.default_language, supported_languages=set(instance.supported_languages))
        nodes: dict[str, NodeSer] = {}
        edges: dict[str, EdgeSer] = {}
        mc = cls(id=instance.id)

        included_nodes = set(ctx.nodes.values())
        with i18n_ctx.activate():
            for node in included_nodes:
                nodes[node.id] = NodeSer.from_node(mc, node)
            for node in included_nodes:
                for edge in node.edges:
                    if edge.input_node != node:
                        continue
                    if edge.output_node not in included_nodes:
                        continue
                    edge_ser = EdgeSer.from_edge(edge, nodes)
                    assert edge_ser.id not in edges
                    edges[edge_ser.id] = edge_ser

        mc.nodes = list(nodes.values())
        mc.edges = list(edges.values())
        return mc
