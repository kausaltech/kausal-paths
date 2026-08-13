from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any
from uuid import NAMESPACE_URL, UUID, uuid5

from django.utils.module_loading import import_string
from pydantic import Field, PrivateAttr, model_validator

from kausal_common.i18n.pydantic import I18nString, set_i18n_context

from nodes.constraints.rules import MissingPortRoleError
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
from nodes.defs.node_defs import ActionConfig, FormulaConfig, NodeSpec, PipelineConfig, SimpleConfig, TypeConfig

if TYPE_CHECKING:
    from collections.abc import Mapping

    import networkx as nx

    from nodes.constraints.compile import ShapeRuleCompilation
    from nodes.constraints.solver import ConstraintProgram, ConstraintSolveResult, GraphOverlay
    from nodes.dataset_shape import DatasetMetricPair, DatasetShapeProfile
    from nodes.defs.port_def import InputPortDef, OutputPortDef
    from nodes.instance_serialization import InstanceSnapshot
    from nodes.node import Node


# v5: canonical edge order is creation (pk) order, not NodeEdge.Meta ordering;
#     binding positions built from a snapshot change accordingly.
INSTANCE_GRAPH_FORMAT_VERSION = 5


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

    def _has_declaration_identifier_match(self, port: InputPortDef) -> bool:
        if port.identifier is None:
            return False
        return any(declaration.instance_identifier == port.identifier for declaration in self.node_class.input_port_declarations)

    @cached_property
    def _port_role_inference(self) -> tuple[dict[UUID, str], tuple[InstanceGraphDiagnostic, ...]]:
        """
        Legacy role classification, delegated to the node class.

        Derived state, recomputed per hydrated graph: the class hook
        (``Node.infer_legacy_port_roles()``) sees only candidate ports —
        authored roles and declaration-identifier matches are filtered out
        here — and every inferred role must exist in the class declarations.
        Goes away once persisted ports carry explicit roles.
        """
        from nodes.constraints.rules import ShapeRuleError

        try:
            node_class = self.node_class
        except ImportError:
            return {}, ()  # a broken class path gets its diagnostic from rule compilation
        candidates = tuple(
            port for port in self.spec.input_ports if port.role is None and not self._has_declaration_identifier_match(port)
        )
        if not candidates:
            return {}, ()
        metadata = self.graph.metadata
        # The hook may import runtime modules that construct i18n values.
        with set_i18n_context(metadata.primary_language, metadata.other_languages):
            result = node_class.infer_legacy_port_roles(self, candidates)

        candidate_ids = {port.id for port in candidates}
        declared_roles = {declaration.role for declaration in node_class.input_port_declarations}
        roles: dict[UUID, str] = {}
        diagnostics: list[InstanceGraphDiagnostic] = []
        for item in result.inferred:
            if item.port_id not in candidate_ids:
                raise ShapeRuleError(
                    f'{self.node_class_path} (node {self.identifier or self.id}): '
                    f'inferred a role for non-candidate port {item.port_id}'
                )
            if item.role not in declared_roles:
                raise ShapeRuleError(
                    f'{self.node_class_path} (node {self.identifier or self.id}): '
                    f'inferred undeclared role {item.role!r} for port {item.port_id}'
                )
            roles[item.port_id] = item.role
            diagnostics.append(
                InstanceGraphDiagnostic(
                    code='inferred_port_role',
                    message=f'Input port {item.port_id} classified as {item.role!r} from {item.basis}',
                    node_id=self.id,
                    port_id=item.port_id,
                )
            )
        diagnostics.extend(
            InstanceGraphDiagnostic(
                code='unclassified_port_role',
                message=f'Input port {refusal.port_id}: {refusal.reason}',
                node_id=self.id,
                port_id=refusal.port_id,
            )
            for refusal in result.unclassified
        )
        return roles, tuple(diagnostics)

    @property
    def inferred_port_roles(self) -> dict[UUID, str]:
        return self._port_role_inference[0]

    @property
    def port_role_diagnostics(self) -> tuple[InstanceGraphDiagnostic, ...]:
        return self._port_role_inference[1]

    def role_for_input_port(self, port: InputPortDef) -> str | None:
        """
        Resolve one input port's semantic role.

        Precedence: the authored ``InputPortDef.role``, then the derived
        legacy classification, then matching the port identifier against the
        class declaration's instance identifier (covers snapshots synced
        before roles were persisted).
        """
        if port.role is not None:
            return port.role
        inferred = self.inferred_port_roles.get(port.id)
        if inferred is not None:
            return inferred
        if port.identifier is not None:
            for declaration in self.node_class.input_port_declarations:
                if declaration.instance_identifier == port.identifier:
                    return declaration.role
        return None

    def role_for_output_port(self, port: OutputPortDef) -> str | None:
        if port.role is not None:
            return port.role
        if port.identifier is not None:
            for declaration in self.node_class.output_port_declarations:
                if declaration.identifier == port.identifier:
                    return declaration.role
        return None

    def input_ports_for_role(self, role: str) -> tuple[InputPortDef, ...]:
        return tuple(port for port in self.spec.input_ports if self.role_for_input_port(port) == role)

    def input_port_ids_for_roles(self, *roles: str) -> tuple[UUID, ...]:
        """Port UUIDs whose role is any of the given roles, in spec port order."""
        wanted = set(roles)
        return tuple(port.id for port in self.spec.input_ports if self.role_for_input_port(port) in wanted)

    def require_input_port(self, role: str) -> InputPortDef:
        ports = self.input_ports_for_role(role)
        if len(ports) == 1:
            return ports[0]
        if not ports:
            raise MissingPortRoleError(self.id, 'input', role)
        raise ValueError(f'Node {self.id} has {len(ports)} input ports for role {role!r}; use input_ports_for_role()')

    def require_output_port(self, role: str) -> OutputPortDef:
        ports = tuple(port for port in self.spec.output_ports if self.role_for_output_port(port) == role)
        if len(ports) == 1:
            return ports[0]
        if len(ports) > 1:
            raise ValueError(f'Node {self.id} has {len(ports)} output ports for role {role!r}')
        declarations = self.node_class.output_port_declarations
        if len(declarations) == 1 and declarations[0].role == role and len(self.spec.output_ports) == 1:
            return self.spec.output_ports[0]
        raise MissingPortRoleError(self.id, 'output', role)


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

    _solve_cache: dict[Any, ConstraintSolveResult] = PrivateAttr(default_factory=dict)

    @cached_property
    def constraint_program(self) -> ConstraintProgram:
        """The compiled constraint program for the graph's own bindings. Derived, not serialized."""
        from nodes.constraints.solver import compile_constraint_program

        with set_i18n_context(self.metadata.primary_language, self.metadata.other_languages):
            return compile_constraint_program(self)

    def describe_uuid(self, value: UUID) -> str:
        """Best-effort human-readable label for a graph UUID, for diagnostics only."""
        dimension = self.dimension_by_id.get(value)
        if dimension is not None:
            return dimension.identifier
        category = self.category_by_id.get(value)
        if category is not None and category.identifier is not None:
            return category.identifier
        node = self.node_by_id.get(value)
        if node is not None and node.identifier is not None:
            return node.identifier
        dataset = self.dataset_by_id.get(value)
        if dataset is not None and dataset.identifier is not None:
            return dataset.identifier
        metric = self.metric_by_id.get(value)
        if metric is not None and metric.identifier is not None:
            return metric.identifier
        return str(value)

    def solve_constraints(
        self,
        *,
        profiles: Mapping[DatasetMetricPair, DatasetShapeProfile] | None = None,
        overlay: GraphOverlay | None = None,
    ) -> ConstraintSolveResult:
        """
        Solve the constraint program, optionally against a hypothetical binding overlay.

        Results are memoized in-process only, keyed by profile versions and the
        overlay content: the compiled program and solver are code, so a cache
        entry must never outlive this hydrated graph object.
        """
        from nodes.constraints.solver import compile_constraint_program, solve_constraint_program

        profiles_key = (
            tuple(sorted((str(pair[0]), str(pair[1]), profile.source_version) for pair, profile in profiles.items()))
            if profiles
            else ()
        )
        cache_key = (profiles_key, overlay.cache_key() if overlay is not None else None)
        cached = self._solve_cache.get(cache_key)
        if cached is not None:
            return cached
        if overlay is None:
            program = self.constraint_program
        else:
            with set_i18n_context(self.metadata.primary_language, self.metadata.other_languages):
                program = compile_constraint_program(self, bindings=overlay.apply(self.bindings))
        result = solve_constraint_program(program, describe=self.describe_uuid, profiles=profiles)
        self._solve_cache[cache_key] = result
        return result

    @cached_property
    def shape_rule_compilation(self) -> ShapeRuleCompilation:
        """
        Compiled shape rules for every node, with per-node diagnostics.

        Derived state: imports node classes lazily and validates each declared
        rule against this graph. Structurally invalid rules raise
        ``ShapeRuleError``; nodes whose legacy ports lack roles compile to no
        rules and a diagnostic.
        """
        from nodes.constraints.compile import compile_shape_rules

        # Rule compilation imports runtime node modules lazily; some of them
        # construct i18n Pydantic values at import time and need a language
        # context, exactly like the loaders that import them today.
        with set_i18n_context(self.metadata.primary_language, self.metadata.other_languages):
            return compile_shape_rules(self)

    @cached_property
    def diagnostics(self) -> tuple[InstanceGraphDiagnostic, ...]:  # noqa: C901, PLR0912
        diagnostics: list[InstanceGraphDiagnostic] = [
            diagnostic for node in self.nodes for diagnostic in node.port_role_diagnostics
        ]
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
    def dimension_by_identifier(self) -> dict[str, DimensionMeta]:
        return {dimension.identifier: dimension for dimension in self.dimensions}

    def require_dimension(self, identifier: str) -> DimensionMeta:
        """Resolve a dimension role selector to graph identity, for use in ``shape_rules()``."""
        try:
            return self.dimension_by_identifier[identifier]
        except KeyError:
            raise ValueError(f'Instance {self.instance_id} has no dimension {identifier!r}') from None

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
    return type_config_class_path(spec.type_config)


def type_config_class_path(config: TypeConfig) -> str:
    if isinstance(config, (SimpleConfig, ActionConfig)):
        if config.node_class.startswith('nodes.'):
            return config.node_class
        prefix = 'nodes.actions' if isinstance(config, ActionConfig) else 'nodes'
        return f'{prefix}.{config.node_class}'
    if isinstance(config, FormulaConfig):
        return 'nodes.formula.FormulaNode'
    assert isinstance(config, PipelineConfig)
    return 'nodes.pipeline.compat.PipelineCompatibleNode'


def node_class_for_type_config(config: TypeConfig) -> type[Node]:
    """Import the runtime class a type config names, lazily like ``NodeMeta.node_class``."""
    return import_string(type_config_class_path(config))


def node_class_for_spec(spec: NodeSpec) -> type[Node]:
    """Import the runtime class a spec names, lazily like ``NodeMeta.node_class``."""
    return node_class_for_type_config(spec.type_config)


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

    from nodes.defs.transform_def import FlattenTransformation
    from nodes.instance_serialization import DatasetPortSnapshot

    # v9 snapshots store positions assigned by ``ordered_binding_snapshots``
    # (the shared ordering authority the ``NodeInputPortBinding`` mirror also
    # uses); older snapshots get the same assignment computed on upgrade.
    edge_index = 0
    port_index = 0
    for item, position in snapshot.bindings_with_positions():
        if isinstance(item, DatasetPortSnapshot):
            port = item
            dataset = None
            if port.dataset_uuid is not None:
                dataset = datasets_by_id.get(port.dataset_uuid)
            if dataset is None:
                dataset = datasets_by_identifier.get(port.dataset)
            if dataset is None:
                raise ValueError(f'Dataset port {port.uuid} references unresolved dataset {port.dataset!r}')

            metric = dataset.metric_by_id.get(port.metric_uuid) if port.metric_uuid is not None else None
            if metric is None:
                metric = next((m for m in dataset.metrics if m.identifier == port.metric), None)
            if metric is None:
                raise ValueError(f'Dataset port {port.uuid} references unresolved metric {port.metric!r}')

            binding_id = port.uuid or uuid5(
                NAMESPACE_URL,
                f'kausal-paths:legacy-dataset-binding:{snapshot.metadata.uuid}:{port.node}:{port.port_id}:{port.dataset}:{port.metric}:{port.dataset_index}:{port_index}',
            )
            port_index += 1
            bindings.append(
                DatasetBindingDef(
                    id=binding_id,
                    port_ref=NodePortRef(
                        node_uuid=port.node,
                        node_id=node_identifiers.get(port.node) or str(port.node),
                        port_id=port.port_id,
                    ),
                    position=position,
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
            continue

        edge = item
        binding_id = edge.uuid or uuid5(
            NAMESPACE_URL,
            f'kausal-paths:legacy-edge:{snapshot.metadata.uuid}:{edge.from_node}:{edge.from_port}:{edge.to_node}:{edge.to_port}:{edge_index}',
        )
        edge_index += 1
        # Legacy bare `to_dimensions` declarations must be recovered here,
        # before the binding validator's modernization drops them.
        declared_dimensions = [t.dimension for t in edge.transformations if isinstance(t, FlattenTransformation)]
        bindings.append(
            EdgeBindingDef(
                id=binding_id,
                declared_dimensions=declared_dimensions,
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
                position=position,
                tags=list(edge.tags),
                transformations=list(edge.transformations),
            )
        )

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
