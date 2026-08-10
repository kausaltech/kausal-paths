from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING, Annotated
from uuid import NAMESPACE_URL, UUID, uuid5

from django.utils.module_loading import import_string
from pydantic import Field, model_validator

from kausal_common.i18n.pydantic import I18nString  # noqa: TC002

from nodes.defs.binding_def import AnyPortBindingDef, DatasetBindingDef, EdgeBindingDef, NodePortRef
from nodes.defs.graph import (
    DatasetMeta,
    DatasetMetricMeta,
    DimensionCategoryMeta,
    DimensionMeta,
    FrozenGraphModel,
    InstanceGraphBoundModel,
)
from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec  # noqa: TC001
from nodes.defs.node_defs import ActionConfig, FormulaConfig, NodeSpec, PipelineConfig, SimpleConfig

if TYPE_CHECKING:
    import networkx as nx

    from nodes.defs.port_def import InputPortDef, OutputPortDef
    from nodes.instance_serialization import InstanceSnapshot
    from nodes.node import Node


INSTANCE_GRAPH_FORMAT_VERSION = 1


class InstanceGraphDiagnostic(FrozenGraphModel):
    code: str
    message: str
    node_id: UUID | None = None
    port_id: UUID | None = None
    binding_id: UUID | None = None


class NodeMeta(InstanceGraphBoundModel):
    id: UUID
    identifier: str | None = None
    node_class_path: str
    spec: NodeSpec
    name: I18nString | None = None
    short_name: I18nString | None = None
    short_description: I18nString | None = None
    description: I18nString | None = None
    goal: I18nString | None = None
    color: str = ''
    order: int | None = None
    is_visible: bool = True
    indicator_node_id: UUID | None = None
    copy_of_id: UUID | None = None

    @cached_property
    def node_class(self) -> type[Node]:
        return import_string(self.node_class_path)

    @property
    def input_bindings(self) -> tuple[AnyPortBindingDef, ...]:
        return self.graph.bindings_for_node(self.id)

    def bindings_for_port(self, port_id: UUID) -> tuple[AnyPortBindingDef, ...]:
        return self.graph.bindings_by_input.get((self.id, port_id), ())

    def require_input_port(self, role: str) -> InputPortDef:
        declaration = next((item for item in self.node_class.input_port_declarations if item.role == role), None)
        identifier = declaration.instance_identifier if declaration is not None else role
        try:
            return self.spec.input_port_by_identifier[identifier]
        except KeyError:
            raise ValueError(f'Node {self.id} has no input port for role {role!r}') from None

    def require_output_port(self, role: str) -> OutputPortDef:
        declaration = next((item for item in self.node_class.output_port_declarations if item.role == role), None)
        identifier = declaration.identifier if declaration is not None else role
        try:
            return self.spec.output_port_by_identifier[identifier]
        except KeyError:
            raise ValueError(f'Node {self.id} has no output port for role {role!r}') from None


GraphBinding = Annotated[AnyPortBindingDef, Field(discriminator='kind')]


class InstanceGraph(FrozenGraphModel):
    format_version: int = INSTANCE_GRAPH_FORMAT_VERSION
    instance_id: UUID
    copy_of_id: UUID | None = None
    metadata: InstanceMetadata
    spec: InstanceModelSpec
    nodes: tuple[NodeMeta, ...] = ()
    bindings: tuple[GraphBinding, ...] = ()
    dimensions: tuple[DimensionMeta, ...] = ()
    datasets: tuple[DatasetMeta, ...] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InstanceGraph):
            return NotImplemented
        # Pydantic's default also compares PrivateAttr/cached_property state.
        # Those are deliberately outside this value object's identity.
        return self.model_dump(mode='python') == other.model_dump(mode='python')

    def __hash__(self) -> int:
        raise TypeError(f'unhashable type: {type(self).__name__!r}')

    @model_validator(mode='after')
    def validate_and_bind(self) -> InstanceGraph:
        if self.format_version != INSTANCE_GRAPH_FORMAT_VERSION:
            raise ValueError(
                f'Unsupported InstanceGraph format version {self.format_version}; expected {INSTANCE_GRAPH_FORMAT_VERSION}'
            )
        self._validate_unique_ids()
        for child in (*self.nodes, *self.bindings):
            child._bind_graph(self)
        return self

    def _validate_unique_ids(self) -> None:
        for name, values in (
            ('node', self.nodes),
            ('binding', self.bindings),
            ('dimension', self.dimensions),
            ('dataset', self.datasets),
        ):
            ids = [value.id for value in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f'Duplicate {name} UUID in InstanceGraph')
        category_ids = [category.id for dimension in self.dimensions for category in dimension.categories]
        if len(category_ids) != len(set(category_ids)):
            raise ValueError('Duplicate dimension category UUID in InstanceGraph')
        metric_ids = [metric.id for dataset in self.datasets for metric in dataset.metrics]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError('Duplicate dataset metric UUID in InstanceGraph')

    @cached_property
    def diagnostics(self) -> tuple[InstanceGraphDiagnostic, ...]:  # noqa: C901, PLR0912
        diagnostics: list[InstanceGraphDiagnostic] = []
        for binding in self.bindings:
            target = binding.port_ref
            if target.node_uuid is None:
                diagnostics.append(
                    InstanceGraphDiagnostic(
                        code='missing_target_node_uuid',
                        message=f'Binding {binding.id} has no target node UUID',
                        binding_id=binding.id,
                        port_id=target.port_id,
                    )
                )
                continue
            node = self.node_by_id.get(target.node_uuid)
            if node is None:
                diagnostics.append(
                    InstanceGraphDiagnostic(
                        code='unknown_target_node',
                        message=f'Binding {binding.id} targets unknown node {target.node_uuid}',
                        node_id=target.node_uuid,
                        port_id=target.port_id,
                        binding_id=binding.id,
                    )
                )
                continue
            if target.port_id not in node.spec.input_port_by_id:
                diagnostics.append(
                    InstanceGraphDiagnostic(
                        code='unknown_input_port',
                        message=f'Binding {binding.id} targets unknown input port {target.port_id}',
                        node_id=target.node_uuid,
                        port_id=target.port_id,
                        binding_id=binding.id,
                    )
                )
            if isinstance(binding, EdgeBindingDef):
                source = binding.from_ref
                if source.node_uuid is None:
                    diagnostics.append(
                        InstanceGraphDiagnostic(
                            code='missing_source_node_uuid',
                            message=f'Binding {binding.id} has no source node UUID',
                            binding_id=binding.id,
                            port_id=source.port_id,
                        )
                    )
                    continue
                source_node = self.node_by_id.get(source.node_uuid)
                if source_node is None:
                    diagnostics.append(
                        InstanceGraphDiagnostic(
                            code='unknown_source_node',
                            message=f'Binding {binding.id} references unknown source node {source.node_uuid}',
                            node_id=source.node_uuid,
                            port_id=source.port_id,
                            binding_id=binding.id,
                        )
                    )
                    continue
                if source.port_id not in source_node.spec.output_port_by_id:
                    diagnostics.append(
                        InstanceGraphDiagnostic(
                            code='unknown_output_port',
                            message=f'Binding {binding.id} references unknown output port {source.port_id}',
                            node_id=source.node_uuid,
                            port_id=source.port_id,
                            binding_id=binding.id,
                        )
                    )
            elif isinstance(binding, DatasetBindingDef):
                if binding.dataset_uuid is None or binding.metric_uuid is None:
                    diagnostics.append(
                        InstanceGraphDiagnostic(
                            code='missing_dataset_or_metric_uuid',
                            message=f'Dataset binding {binding.id} lacks canonical dataset or metric UUID',
                            binding_id=binding.id,
                        )
                    )
                    continue
                dataset = self.dataset_by_id.get(binding.dataset_uuid)
                if dataset is None:
                    diagnostics.append(
                        InstanceGraphDiagnostic(
                            code='unknown_dataset',
                            message=f'Binding {binding.id} references unknown dataset {binding.dataset_uuid}',
                            binding_id=binding.id,
                        )
                    )
                    continue
                if binding.metric_uuid not in dataset.metric_by_id:
                    diagnostics.append(
                        InstanceGraphDiagnostic(
                            code='unknown_metric',
                            message=f'Binding {binding.id} references unknown metric {binding.metric_uuid}',
                            binding_id=binding.id,
                        )
                    )

        import networkx as nx

        if not nx.is_directed_acyclic_graph(self.nx_graph):
            diagnostics.append(
                InstanceGraphDiagnostic(
                    code='directed_cycle',
                    message='InstanceGraph contains a directed cycle',
                )
            )
        return tuple(diagnostics)

    @cached_property
    def node_by_id(self) -> dict[UUID, NodeMeta]:
        return {node.id: node for node in self.nodes}

    @cached_property
    def node_by_identifier(self) -> dict[str, NodeMeta]:
        return {node.identifier: node for node in self.nodes if node.identifier is not None}

    @cached_property
    def input_port_by_id(self) -> dict[tuple[UUID, UUID], InputPortDef]:
        return {(node.id, port.id): port for node in self.nodes for port in node.spec.input_ports}

    @cached_property
    def output_port_by_id(self) -> dict[tuple[UUID, UUID], OutputPortDef]:
        return {(node.id, port.id): port for node in self.nodes for port in node.spec.output_ports}

    @cached_property
    def binding_by_id(self) -> dict[UUID, AnyPortBindingDef]:
        return {binding.id: binding for binding in self.bindings}

    @cached_property
    def bindings_by_input(self) -> dict[tuple[UUID, UUID], tuple[AnyPortBindingDef, ...]]:
        grouped: dict[tuple[UUID, UUID], list[AnyPortBindingDef]] = {}
        for binding in self.bindings:
            node_id = binding.port_ref.node_uuid
            if node_id is None:
                continue
            grouped.setdefault((node_id, binding.port_ref.port_id), []).append(binding)
        return {key: tuple(sorted(values, key=lambda binding: binding.position)) for key, values in grouped.items()}

    def bindings_for_node(self, node_id: UUID) -> tuple[AnyPortBindingDef, ...]:
        values = [binding for binding in self.bindings if binding.port_ref.node_uuid == node_id]
        return tuple(sorted(values, key=lambda binding: (binding.position, str(binding.id))))

    @cached_property
    def dimension_by_id(self) -> dict[UUID, DimensionMeta]:
        return {dimension.id: dimension for dimension in self.dimensions}

    @cached_property
    def category_by_id(self) -> dict[UUID, DimensionCategoryMeta]:
        return {category.id: category for dimension in self.dimensions for category in dimension.categories}

    @cached_property
    def dataset_by_id(self) -> dict[UUID, DatasetMeta]:
        return {dataset.id: dataset for dataset in self.datasets}

    @cached_property
    def metric_by_id(self) -> dict[UUID, DatasetMetricMeta]:
        return {metric.id: metric for dataset in self.datasets for metric in dataset.metrics}

    @cached_property
    def nx_graph(self) -> nx.DiGraph[UUID]:
        import networkx as nx

        graph: nx.DiGraph[UUID] = nx.DiGraph()
        graph.add_nodes_from(self.node_by_id)
        graph.add_edges_from(
            (binding.from_ref.node_uuid, binding.port_ref.node_uuid)
            for binding in self.bindings
            if isinstance(binding, EdgeBindingDef)
            and binding.from_ref.node_uuid in self.node_by_id
            and binding.port_ref.node_uuid in self.node_by_id
        )
        return graph

    @cached_property
    def topological_order(self) -> tuple[UUID, ...]:
        import networkx as nx

        if not nx.is_directed_acyclic_graph(self.nx_graph):
            raise ValueError('InstanceGraph contains a directed cycle')
        return tuple(nx.topological_sort(self.nx_graph))


def node_class_path(spec: NodeSpec) -> str:
    config = spec.type_config
    if isinstance(config, (SimpleConfig, ActionConfig)):
        if config.node_class.startswith('nodes.'):
            return config.node_class
        prefix = 'nodes.actions' if isinstance(config, ActionConfig) else 'nodes'
        return f'{prefix}.{config.node_class}'
    if isinstance(config, FormulaConfig):
        return 'nodes.formula.FormulaNode'
    assert isinstance(config, PipelineConfig)
    return 'nodes.pipeline.compat.PipelineCompatibleNode'


def build_instance_graph(
    snapshot: InstanceSnapshot,
    *,
    legacy_dimensions: tuple[DimensionMeta, ...] = (),
    legacy_datasets: tuple[DatasetMeta, ...] = (),
) -> InstanceGraph:
    """Normalize one structural snapshot into the canonical UUID graph."""

    dimensions = tuple(snapshot.dimensions) or legacy_dimensions
    datasets = tuple(snapshot.datasets) or legacy_datasets
    datasets_by_id = {dataset.id: dataset for dataset in datasets}
    datasets_by_identifier = {dataset.identifier: dataset for dataset in datasets if dataset.identifier is not None}

    nodes: list[NodeMeta] = []
    node_identifiers: dict[UUID, str | None] = {}
    for node in snapshot.nodes:
        if node.spec is None:
            raise ValueError(f'Node {node.uuid} has no computation spec')
        nodes.append(
            NodeMeta(
                id=node.uuid,
                identifier=node.identifier,
                node_class_path=node_class_path(node.spec),
                spec=node.spec,
                name=node.name,
                short_name=node.short_name,
                short_description=node.short_description,
                description=node.description,
                goal=node.goal,
                color=node.color,
                order=node.order,
                is_visible=node.is_visible,
                indicator_node_id=node.indicator_node,
                copy_of_id=node.copy_of,
            )
        )
        node_identifiers[node.uuid] = node.identifier

    bindings: list[AnyPortBindingDef] = []
    positions: defaultdict[tuple[UUID, UUID], int] = defaultdict(int)

    for edge_index, edge in enumerate(snapshot.edges):
        key = (edge.to_node, edge.to_port)
        binding_id = edge.uuid or uuid5(
            NAMESPACE_URL,
            f'kausal-paths:legacy-edge:{snapshot.metadata.uuid}:{edge.from_node}:{edge.from_port}:{edge.to_node}:{edge.to_port}:{edge_index}',
        )
        bindings.append(
            EdgeBindingDef(
                id=binding_id,
                port_ref=NodePortRef(
                    node_uuid=edge.to_node,
                    node_id=node_identifiers.get(edge.to_node) or str(edge.to_node),
                    port_id=edge.to_port,
                ),
                from_ref=NodePortRef(
                    node_uuid=edge.from_node,
                    node_id=node_identifiers.get(edge.from_node) or str(edge.from_node),
                    port_id=edge.from_port,
                ),
                position=positions[key],
                tags=list(edge.tags),
                transformations=list(edge.transformations),
            )
        )
        positions[key] += 1

    sorted_ports = sorted(
        snapshot.dataset_ports,
        key=lambda item: (item.node, item.dataset_index, str(item.port_id), item.metric),
    )
    for port_index, port in enumerate(sorted_ports):
        dataset = None
        if port.dataset_uuid is not None:
            dataset = datasets_by_id.get(port.dataset_uuid)
        if dataset is None:
            dataset = datasets_by_identifier.get(port.dataset)
        if dataset is None:
            raise ValueError(f'Dataset port {port.uuid} references unresolved dataset {port.dataset!r}')

        metric = dataset.metric_by_id.get(port.metric_uuid) if port.metric_uuid is not None else None
        if metric is None:
            metric = next((item for item in dataset.metrics if item.identifier == port.metric), None)
        if metric is None:
            raise ValueError(f'Dataset port {port.uuid} references unresolved metric {port.metric!r}')

        key = (port.node, port.port_id)
        binding_id = port.uuid or uuid5(
            NAMESPACE_URL,
            f'kausal-paths:legacy-dataset-binding:{snapshot.metadata.uuid}:{port.node}:{port.port_id}:{port.dataset}:{port.metric}:{port.dataset_index}:{port_index}',
        )
        bindings.append(
            DatasetBindingDef(
                id=binding_id,
                port_ref=NodePortRef(
                    node_uuid=port.node,
                    node_id=node_identifiers.get(port.node) or str(port.node),
                    port_id=port.port_id,
                ),
                position=positions[key],
                tags=list(port.spec.tags),
                transformations=list(port.spec.transformations),
                dataset_uuid=dataset.id,
                metric_uuid=metric.id,
                dataset_is_external_placeholder=dataset.is_external_placeholder,
                dataset_external_ref=dataset.external_ref,
                external_dataset_id=dataset.identifier,
                external_metric_id=metric.identifier,
            )
        )
        positions[key] += 1

    return InstanceGraph(
        instance_id=snapshot.metadata.uuid,
        copy_of_id=UUID(snapshot.copy_of) if snapshot.copy_of is not None else None,
        metadata=snapshot.metadata,
        spec=snapshot.spec,
        nodes=tuple(nodes),
        bindings=tuple(bindings),
        dimensions=dimensions,
        datasets=datasets,
    )
