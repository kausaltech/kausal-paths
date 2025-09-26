from __future__ import annotations

import inspect
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, overload

import rich
from rich.tree import Tree
from sentry_sdk import start_span

from kausal_common.debugging.perf import PerfCounter
from kausal_common.deployment import env_bool
from kausal_common.perf.perf_context import PerfContext

from paths.const import MODEL_CACHE_OP, MODEL_CALC_OP

from common import (
    base32_crockford,
    polars as pl,  # noqa: F401
    polars_ext,  # noqa: F401
)
from common.cache import Cache
from params.discover import discover_parameter_types

from .datasets import Dataset, DVCDataset, FixedDataset
from .units import Unit, unit_registry

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime
    from types import FrameType

    import dvc_pandas
    import networkx  # noqa: ICN001
    from rich.repr import RichReprResult

    from nodes.explanations import NodeExplanationSystem
    from params import Parameter
    from params.storage import SettingStorage

    from .actions.action import ActionNode, ImpactOverview
    from .dimensions import Dimension
    from .instance import Instance
    from .node import Node
    from .normalization import Normalization
    from .scenario import CustomScenario, Scenario
    from .units import CachingUnitRegistry


@dataclass
class FrameworkConfigData:
    last_modified_at: datetime
    id: int


class Context:
    """
    The main calculation execution context managing nodes, parameters, and scenarios.

    Context manages the entire calculation graph, including nodes, their connections,
    global parameters, scenarios, and datasets. It provides methods for graph
    manipulation, calculation execution, and result management.

    It is initialized by a computation model `Instance`.
    """

    nodes: dict[str, Node]
    """All nodes in the context keyed by the node identifier."""

    datasets: dict[str, Dataset]
    """All datasets in the context keyed by the dataset identifier."""

    dvc_datasets: dict[str, dvc_pandas.Dataset]
    """All the loaded dvc-pandas datasets keyed by the dataset identifier."""

    global_parameters: dict[str, Parameter]
    """Global parameters not specific to any individual node."""

    node_explanation_system: NodeExplanationSystem
    """Explanations and validations for nodes and node graph."""

    scenarios: dict[str, Scenario]
    """All scenarios in the context keyed by the scenario identifier."""

    custom_scenario: CustomScenario
    """The custom scenario for the context."""

    options: dict[str, Any]
    """Options for the context."""

    normalizations: dict[str, Normalization]
    """Available normalizations keyed by the normalization identifier.

    A common normalization would be "per capita", where the normalizer
    node is the population node.
    """

    dimensions: dict[str, Dimension]
    """Global dimensions available for nodes."""

    target_year: int
    """The target year for the most important goals in the computation model."""

    model_end_year: int
    """The end year for the model. This is the last year for which data is computed."""

    dataset_repo: dvc_pandas.Repository
    """The dvc-pandas dataset repository for the computation model."""

    dataset_repo_default_path: str | None
    sample_size: int

    unit_registry: CachingUnitRegistry
    """The pint unit registry used for unit parsing, conversion, and formatting."""

    active_scenario: Scenario
    """The active scenario for the context."""

    active_normalization: Normalization | None
    """The currently active normalization."""

    default_normalization: Normalization | None = None
    """The default normalization for the context.

    Will be `None` if no normalization is the default.
    """

    supported_parameter_types: dict[str, type]
    """All supported parameter types.

    Dictionary values will be subclasses of `Parameter`.
    """

    cache: Cache
    """Cache for computation results and datasets."""

    skip_cache: bool = False
    """Can be set to disable caching for the context. Commonly used for debugging."""

    check_mode: bool = False
    """If set, extra checks will be performed during computation runs."""

    instance: Instance
    """The computation model instance."""

    impact_overviews: list[ImpactOverview]
    """List of action efficiency pairs available for the computation model."""

    setting_storage: SettingStorage | None
    """
    The setting storage for the context.

    This is used to store customized parameter values, the active scenario,
    the active normalization, and other settings.

    Will commonly be SessionStorage when in HTTP request context.
    """

    perf_context: PerfContext[Node]
    """Performance context for the context."""

    node_graph: networkx.DiGraph[str]
    """Directed NetworkX graph for the nodes and edges in the model."""

    baseline_values_generated: bool = False
    """If the baseline values have been generated."""

    obj_id: str
    """
    The identifier for the context.

    It is generated randomly and used mostly for logging.
    """

    def __init__(
        self,
        instance: Instance,
        dataset_repo: dvc_pandas.Repository,
        target_year: int,
        model_end_year: int | None = None,
        dataset_repo_default_path: str | None = None,
        sample_size: int = 0,
    ):
        from nodes.actions.action import ActionNode

        self.obj_id = base32_crockford.gen_obj_id(id(self))
        # Avoid circular import
        self.Action = ActionNode
        self.perf_context = PerfContext(supports_cache=True)
        self.nodes = {}
        self.datasets = {}
        self.dvc_datasets = {}
        self.global_parameters = {}
        self.scenarios = {}
        self.target_year = target_year
        self.model_end_year = model_end_year or target_year
        self.unit_registry = unit_registry
        self.dataset_repo = dataset_repo
        self.dataset_repo_default_path = dataset_repo_default_path
        self.sample_size = sample_size
        self.supported_parameter_types = discover_parameter_types()
        # will be set later
        self.instance = None  # type: ignore
        self.active_scenario = None  # type: ignore
        self.custom_scenario = None  # type: ignore
        self.active_normalization = None
        self.impact_overviews = []
        self.dimensions = {}
        self.options = {}
        self.normalizations = {}
        self.instance = instance
        self.log = self.instance.log.bind(context=self.obj_id, markup=True)
        self.log.debug('Context initialized')
        self.cache = Cache(
            ureg=self.unit_registry,
            redis_url=os.getenv('REDIS_URL'),
            base_logger=self.log,
        )
        self.node_explanation_system: NodeExplanationSystem | None = None
        if env_bool('DISABLE_PATHS_MODEL_CACHE', default=False):
            self.skip_cache = True
        super().__init__()

    def __rich_repr__(self) -> RichReprResult:
        yield 'instance', self.instance.id
        yield 'obj_id', self.obj_id

    @contextmanager
    def start_span(self, name: str, op: str | None = None, attributes: dict[str, Any] | None = None):
        _rich_traceback_omit = True
        with start_span(name=name, op=op) as span:
            if attributes is not None:
                for key, val in attributes.items():
                    span.set_data(key, val)
            yield span

    def finalize_nodes(self):
        """
        Finalize the node graph.

        Called when nodes and their connections have been configured.
        """
        import networkx as nx

        g: nx.DiGraph[str] = nx.DiGraph()
        g.add_nodes_from([n.id for n in self.nodes.values()])
        for node in self.nodes.values():
            for output in node.output_nodes:
                g.add_edge(node.id, output.id)

        if not nx.is_directed_acyclic_graph(g):
            raise Exception('Node graph is not directed (there are loops between nodes)')

        for node in self.nodes.values():
            node.finalize_init()

        self.node_graph = g

        if self.instance.result_excels:
            for excel_res in self.instance.result_excels:
                excel_res.validate_for_instance(self.instance)

    def get_parameter_type(self, parameter_id: str) -> type:
        param_type = self.supported_parameter_types.get(parameter_id)
        if param_type is None:
            raise Exception('Unknown parameter: %s' % parameter_id)
        return param_type

    def load_dvc_dataset(self, ds_id: str) -> dvc_pandas.Dataset:
        """
        Load a DVC dataset into the context.

        If the dataset hasn't yet been loaded, it will be read from the DVC
        repository.
        """

        ds = self.dvc_datasets.get(ds_id)
        if ds is not None:
            return ds

        if not self.dataset_repo.has_dataset(ds_id):
            raise Exception('Dataset %s not found in DVC repo' % ds_id)
        if not self.dataset_repo.is_dataset_cached(ds_id):
            self.log.info("Dataset '%s' not found in DVC cache; loading all datasets" % ds_id)
            self.load_all_dvc_datasets()
        with self.start_span('load dataset: %s' % ds_id, op='model.load'):
            ds = self.dataset_repo.load_dataset(ds_id)
        self.dvc_datasets[ds_id] = ds
        return ds

    def get_all_dvc_dataset_ids(self) -> set[str]:
        all_datasets = set()
        for node in self.nodes.values():
            for ds in node.input_dataset_instances:
                if not isinstance(ds, DVCDataset):
                    continue
                all_datasets.add(ds.id)
        return all_datasets

    def load_all_dvc_datasets(self):
        """
        Load all the DVC datasets that are needed by the nodes.

        Individual DVC operations can be slow, so loading all the datasets at once
        is generally faster.
        """

        all_datasets = self.get_all_dvc_dataset_ids()
        with self.start_span('load all datasets', op='model.load'):
            try:
                self.dataset_repo.load_datasets(list(all_datasets))
            except Exception:
                self.log.error('Unable to load DVC datasets: %s' % ', '.join(all_datasets))
                raise
        self.log.debug('All DVC datasets loaded')

    def add_node(self, node: Node):
        """
        Add a node to the context.

        Called only during the initialization phase.
        """

        if node.id in self.nodes:
            raise Exception('Node %s already defined' % (node.id))
        self.nodes[node.id] = node
        for param_id in node.global_parameters:
            if param_id not in self.global_parameters:
                continue
            param = self.global_parameters[param_id]
            assert node not in param.subscription_nodes
            param.subscription_nodes.append(node)

    def get_node(self, id: str) -> Node:
        """
        Get a node from the context.

        Raises:
            KeyError: If the node is not found.

        """
        if id not in self.nodes:
            raise KeyError("Node '%s' not found" % id)
        return self.nodes[id]

    def get_action(self, id: str) -> ActionNode:
        """
        Get an action node from the context.

        Raises:
            TypeError: If the node is not an action node.
            KeyError: If the node is not found.

        """

        node = self.nodes[id]
        if not isinstance(node, self.Action):
            raise TypeError('Node %s is not an action node' % id)
        return node

    def add_global_parameter(self, parameter: Parameter):
        """
        Add a global parameter to the context.

        Called only during the initialization phase.
        """
        if parameter.local_id in self.global_parameters:
            msg = f'Global parameter {parameter.local_id} already defined'
            raise Exception(msg)
        self.global_parameters[parameter.local_id] = parameter

    def add_normalization(self, id: str, norm: Normalization):
        assert id not in self.normalizations
        if norm.default:
            assert not self.default_normalization
            self.default_normalization = norm
        self.normalizations[id] = norm

    def _get_caller_node(self, frame: FrameType) -> Node | None:
        from nodes.node import Node

        caller_frame: FrameType | None = inspect.getouterframes(frame, 0)[1].frame
        while caller_frame is not None:
            cl = caller_frame.f_locals
            cs = cl.get('self')
            if cs is None:
                return None
            if isinstance(cs, Context):
                caller_frame = caller_frame.f_back
                continue
            if isinstance(cs, Node):
                return cs
            break
        return None

    @overload
    def get_parameter(self, param_id: str, *, required: Literal[True] = True) -> Parameter: ...

    @overload
    def get_parameter(self, param_id: str, *, required: Literal[False]) -> Parameter | None: ...

    @overload
    def get_parameter(self, param_id: str, *, required: bool) -> Parameter | None: ...

    def get_parameter(self, param_id: str, *, required: bool = True) -> Parameter | None:
        if self.check_mode:
            from nodes.exceptions import NodeError
            frame = inspect.currentframe()
            if frame is not None:
                node = self._get_caller_node(frame)
                if node is not None and param_id not in node.global_parameters:
                    raise NodeError(
                        node,
                        "Attempting to access global parameter '%s', but it's not listed in global_parameters" % param_id,
                    )

        param = None
        try:
            node_id, param_name = param_id.split('.', 1)
        except ValueError:
            param = self.global_parameters.get(param_id)
        else:
            if node_id in self.nodes:
                param = self.nodes[node_id].get_parameter(param_name, required=required)
        if param is None and required:
            msg = f'Parameter {param_id} not found'
            raise Exception(msg)
        return param

    def get_parameter_value(self, param_id: str, *, required: bool = True) -> Any:
        param = self.get_parameter(param_id, required=required)
        if param is None:
            return None
        return param.value

    def set_parameter_value(self, param_id: str, value: Any):
        param = self.global_parameters.get(param_id)
        if param is None:
            msg = f'Parameter {param_id} not found'
            raise Exception(msg)
        param.set(value)

    def add_scenario(self, scenario: Scenario):
        assert scenario.id not in self.scenarios
        self.scenarios[scenario.id] = scenario
        for node in self.nodes.values():
            node.on_scenario_created(scenario)

    def set_custom_scenario(self, scenario: CustomScenario):
        assert self.custom_scenario is None
        self.add_scenario(scenario)
        self.custom_scenario = scenario

    def get_scenario(self, id: str) -> Scenario:
        return self.scenarios[id]

    def set_option(self, id: Literal['normalizer'], val: Any):
        if id == 'normalizer':
            if val is None:
                self.active_normalization = None
            else:
                if not isinstance(val, str):
                    raise TypeError('Expecting str')
                self.active_normalization = self.normalizations[val]
        else:
            raise KeyError('Unknown option: %s' % id)

    def get_root_nodes(self) -> list[Node]:
        """
        Get a list of all the root nodes.

        Root nodes are the ones that are not acting as inputs to any other nodes.
        """

        all_nodes = self.nodes.values()
        root_nodes = list(filter(lambda node: not node.output_nodes, all_nodes))
        return root_nodes

    def get_outcome_nodes(self) -> list[Node]:
        """
        Get a list of all the outcome nodes.

        Outcome nodes are chosen manually in the model configuration.
        They are meant to be the nodes that are the most important outcomes
        in the model.
        """
        return [node for node in self.nodes.values() if node.is_outcome]

    def activate_scenario(self, scenario: Scenario):
        """
        Activate a scenario.

        Sets the parameter values included in the scenario and marks the scenario as active.
        """
        scenario.activate()
        self.active_scenario = scenario

    def get_default_scenario(self) -> Scenario:
        """
        Get the default scenario.

        The default scenario is the "main" one in the model configuration.

        Raises:
            Exception: If no default scenario is found.

        """
        for scenario in self.scenarios.values():
            if scenario.default:
                return scenario
        raise Exception('No default scenario found')

    def prefetch_node_cache(self, nodes: list[Node]):
        """
        Prefetch the node cache for a list of nodes.

        This is used to speed up the cache access for a model run by
        attempting to prefetch the cached outputs from the external cache
        (Redis) with one request.
        """
        from nodes.node_cache import NodeHasher
        NodeHasher.prefetch_nodes(context=self, nodes=nodes)

    def generate_baseline_values(self):
        """
        Generate the baseline values for the model.

        Baseline values are the values that are used for the model run.
        """
        if self.baseline_values_generated:
            return
        if 'baseline' not in self.scenarios:
            return
        assert self.active_scenario

        baseline = self.scenarios['baseline']
        pc = PerfCounter('generate baseline values')
        with self.start_span('Generate baseline values', op=MODEL_CALC_OP), baseline.override(set_active=True):
            self.log.info('Generating baseline values')
            nodes = list(self.nodes.values())
            with self.start_span('Prefetch node cache', op=MODEL_CACHE_OP):
                self.prefetch_node_cache(nodes)
            with self.start_span('Compute baseline values', op=MODEL_CALC_OP):
                for node in nodes:
                    _ = node.get_baseline_values()
        self.log.info('Baseline values generated in %.1f ms' % pc.measure())
        self.baseline_values_generated = True

    def get_all_parameters(self) -> Generator[Parameter]:
        """Return all the parameters (global and node-specific)."""

        for param in self.global_parameters.values():
            yield param
        for node in self.nodes.values():
            for param in node.get_parameters():
                yield param

    def print_all_parameters(self):
        """Print all the parameters (global and node-specific) together with their values."""

        for param in self.get_all_parameters():
            print('%s: %s' % (param.global_id, param.value))

    def pull_datasets(self):
        """
        Pull the datasets from the DVC repository.

        This is used to update the datasets to the latest version.

        Returns:
            str: The commit ID of the updated datasets.

        """
        self.dataset_repo.set_target_commit(None)
        self.dataset_repo.pull_datasets()
        commit_id = self.dataset_repo.commit_id
        self.dataset_repo.set_target_commit(commit_id)
        self.instance.update_dataset_repo_commit(commit_id)

    def print_graph(self, include_datasets: bool = False) -> None:  # noqa: C901, PLR0915
        import inspect

        visited_nodes: set[Node] = set()
        node_class_cache: dict[type, str] = {}

        def make_node_tree(node: Node, tree: Tree | None = None) -> Tree:  # noqa: C901, PLR0912
            node_color = 'yellow'
            if isinstance(node, self.Action):
                node_color = 'green'
            elif node.quantity == 'emissions':
                node_color = 'magenta'

            node_icon = node.get_icon() or ''
            if node_icon:
                node_icon += ' '

            node_class = type(node)
            node_class_str = node_class_cache.get(node_class)
            if node_class_str is None:
                node_module = node_class.__module__
                module_file = inspect.getabsfile(node_class)
                line_nr = inspect.getsourcelines(node_class)[1]
                link = 'file://%s#%d' % (module_file, line_nr)
                node_class_str = f'[link={link}][grey50]{node_module}.[grey70]{node_class.__name__}[/link]'
                node_class_cache[node_class] = node_class_str

            metrics = ', '.join([f'{m.quantity} [#a63fa4]{m.unit}[orchid]' for m in node.output_metrics.values()])
            unit_quantity = f'({metrics})'
            node_id = node.id
            if node.config_location is not None:
                url = 'file://%s#%d' % (node.config_location['file_path'], node.config_location['line'])
                node_name = f'[link={url}]{node.name}[/link]'
                node_id = f'[link={url}]{node_id}[/link]'
            else:
                node_name = str(node.name)
            node_str = f'{node_icon}[{node_color}]{node_id}[/] [light_sea_green]{node_name}[/]'
            node_str += f' [orchid]{unit_quantity}[/] {node_class_str}'
            if include_datasets:
                for ds in node.input_dataset_instances:
                    if isinstance(ds, FixedDataset):
                        ds_icon = '⌨'
                    elif isinstance(ds, DVCDataset):
                        ds_icon = '🐼'
                    else:
                        ds_icon = '❓'
                    node_str += '\n  %s %s (%s)' % (ds_icon, ds.id, ', '.join(ds.get_copy(self).columns))
            if tree is None:
                branch = Tree(node_str)
            else:
                branch = tree.add(node_str)
            if node not in visited_nodes:
                for in_node in node.input_nodes:
                    make_node_tree(in_node, branch)
                visited_nodes.add(node)
            return branch

        tree = Tree('Nodes')
        all_nodes = self.nodes.values()
        root_nodes = list(filter(lambda node: not node.output_nodes, all_nodes))
        for node in root_nodes:
            tree = make_node_tree(node, tree)
            rich.print(tree)

    def summarize_graph(self):
        """Summarize the graph by printing the nodes with their properties."""

        for node in self.nodes.values():
            if node.unit is not None:
                unit_str = 'unit=%s  quantity=%s' % (str(node.unit), str(node.quantity))
            else:
                unit_str = ''
            print(
                'id="%s"  nr_inputs=%s  nr_outputs=%s  type=%s%s'
                % (
                    node.id,
                    len(node.input_nodes),
                    len(node.output_nodes),
                    type(node).__name__,
                    unit_str,
                ),
            )

    def describe_unit(self, unit: Unit) -> dict[str, str]:
        formats = dict(
            short='~P',
            long='P',
            html_short='~H',
            html_long='H',
        )
        return {k: self.unit_registry.formatter.format_unit_babel(unit) for k, v in formats.items()}

    def get_actions(self) -> list[ActionNode]:
        """Get a list of all the action nodes in the context."""
        from nodes.actions.action import ActionNode

        return [n for n in self.nodes.values() if isinstance(n, ActionNode)]

    def warning(self, msg: Any, *args, depth: int = 0, **kwargs) -> None:
        self.instance.warning(msg, *args, depth=depth + 1, **kwargs)

    @cached_property
    def framework_config_data(self) -> FrameworkConfigData | None:
        from frameworks.models import FrameworkConfig

        fwc = (
            FrameworkConfig.objects.filter(instance_config__identifier=self.instance.id)
            .values_list('last_modified_at', 'id')
            .first()
        )
        if fwc is None:
            return None
        return FrameworkConfigData(last_modified_at=fwc[0], id=fwc[1])

    @contextmanager
    def run(self):
        with ExitStack() as stack:
            span_ctx = self.start_span(
                'context run',
                op=MODEL_CALC_OP,
                attributes=dict(
                    instance_id=self.instance.id,
                    context_id=self.obj_id,
                ),
            )
            stack.enter_context(span_ctx)
            stack.enter_context(self.cache)
            stack.enter_context(self.perf_context)
            yield

    def clean(self):
        for param in self.get_all_parameters():
            param.context = None
            param.node = None
        for node in self.nodes.values():
            node.context = None  # type: ignore
            node.edges = []
            node.output_metrics = {}
            if node.db_obj is not None:
                node.db_obj._node = None
            node.db_obj = None
            node.parameters = {}
        self.nodes = {}
        self.global_parameters = {}
        for scenario in self.scenarios.values():
            scenario.context = None  # type: ignore
        self.custom_scenario = None  # type: ignore
        self.setting_storage = None
        self.active_scenario = None  # type: ignore
        self.normalizations = {}
        self.active_normalization = None
        self.default_normalization = None
        self.scenarios = {}
        self.datasets = {}
        self.dvc_datasets = {}
        self.instance = None  # type: ignore
