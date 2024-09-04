from __future__ import annotations

import inspect
import os
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Literal, overload

import networkx as nx
import rich
from opentelemetry import trace
from rich.tree import Tree

from kausal_common.debugging.perf import PerfCounter

from common import (
    base32_crockford,
    polars as pl,  # noqa: F401
    polars_ext,  # noqa: F401
)
from common.cache import Cache
from params.discover import discover_parameter_types

from .datasets import Dataset, DVCDataset, FixedDataset
from .perf import PerfContext
from .units import Unit, unit_registry

if TYPE_CHECKING:
    from types import FrameType

    import dvc_pandas

    from params import Parameter
    from params.storage import SettingStorage

    from .actions.action import ActionEfficiencyPair, ActionNode
    from .instance import Instance
    from .node import Dimension, Node
    from .normalization import Normalization
    from .scenario import CustomScenario, Scenario
    from .units import CachingUnitRegistry


class Context:
    nodes: dict[str, Node]
    datasets: dict[str, Dataset]
    dvc_datasets: dict[str, dvc_pandas.Dataset]

    # global_parameters contains parameters that are not specific to a node.
    # Node-specific parameters are managed by the node.
    global_parameters: dict[str, Parameter]
    scenarios: dict[str, Scenario]
    custom_scenario: CustomScenario
    options: dict[str, Any]
    normalizations: dict[str, Normalization]

    dimensions: dict[str, Dimension]
    """Global dimensions available for nodes."""

    target_year: int
    model_end_year: int
    dataset_repo: dvc_pandas.Repository
    dataset_repo_default_path: str | None

    unit_registry: CachingUnitRegistry
    active_scenario: Scenario
    active_normalization: Normalization | None
    default_normalization: Normalization | None = None
    supported_parameter_types: dict[str, type]
    cache: Cache
    skip_cache: bool = False
    check_mode: bool = False
    instance: Instance
    action_efficiency_pairs: list[ActionEfficiencyPair]
    setting_storage: SettingStorage | None
    perf_context: PerfContext[Node]
    node_graph: nx.DiGraph
    baseline_values_generated: bool = False
    obj_id: str
    tracer: trace.Tracer

    def __init__(
        self, instance: Instance, dataset_repo: dvc_pandas.Repository, target_year: int,
        model_end_year: int | None = None, dataset_repo_default_path: str | None = None,
    ):
        from nodes.actions import ActionNode

        self.obj_id = base32_crockford.gen_obj_id(id(self))
        self.tracer = trace.get_tracer(
            'nodes.context', attributes=dict(
                context_id=self.obj_id,
                instance_id=instance.id,
            ),
        )

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
        self.supported_parameter_types = discover_parameter_types()
        # will be set later
        self.instance = None  # type: ignore
        self.active_scenario = None  # type: ignore
        self.custom_scenario = None  # type: ignore
        self.active_normalization = None
        self.action_efficiency_pairs = []
        self.dimensions = {}
        self.options = {}
        self.normalizations = {}
        self.instance = instance
        self.log = self.instance.log.bind(context=self.obj_id, markup=True)
        self.log.debug('Context initialized')
        self.cache = Cache(
            ureg=self.unit_registry, redis_url=os.getenv('REDIS_URL'),
            base_logger=self.log,
        )

    def finalize_nodes(self):
        """
        Finalize the node graph.

        Called when nodes and their connections have been configured.
        """

        g = nx.DiGraph()
        g.add_nodes_from([n.id for n in self.nodes.values()])
        for node in self.nodes.values():
            for output in node.output_nodes:
                g.add_edge(node.id, output.id)

        if not nx.is_directed_acyclic_graph(g):
            raise Exception("Node graph is not directed (there are loops between nodes)")

        for node in self.nodes.values():
            node.finalize_init()

        self.node_graph = g

    def get_parameter_type(self, parameter_id: str) -> type:
        param_type = self.supported_parameter_types.get(parameter_id)
        if param_type is None:
            raise Exception("Unknown parameter: %s" % parameter_id)
        return param_type

    def load_dvc_dataset(self, ds_id: str) -> dvc_pandas.Dataset:
        ds = self.dvc_datasets.get(ds_id)
        if ds is None:
            if not self.dataset_repo.has_dataset(ds_id):
                raise Exception('Dataset %s not found in DVC repo' % ds_id)
            with self.tracer.start_as_current_span('load dataset: %s' % ds_id):
                ds = self.dataset_repo.load_dataset(ds_id)
            self.dvc_datasets[ds_id] = ds
        return ds

    def load_all_dvc_datasets(self):
        all_datasets = set()
        for node in self.nodes.values():
            for ds in node.input_dataset_instances:
                if not isinstance(ds, DVCDataset):
                    continue
                all_datasets.add(ds.id)

        with self.tracer.start_as_current_span('load all datasets'):
            try:
                self.dataset_repo.load_datasets(list(all_datasets))
            except Exception:
                self.log.error("Unable to load DVC datasets: %s" % ', '.join(all_datasets))
                raise
        self.log.debug("All DVC datasets loaded")

    def add_node(self, node: Node):
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
        if id not in self.nodes:
            raise KeyError("Node '%s' not found" % id)
        return self.nodes[id]

    def get_action(self, id: str) -> 'ActionNode':
        node = self.nodes[id]
        if not isinstance(node, self.Action):
            raise TypeError("Node %s is not an action node" % id)
        return node

    def add_global_parameter(self, parameter: Parameter):
        if parameter.local_id in self.global_parameters:
            raise Exception(f"Global parameter {parameter.local_id} already defined")
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
            else:
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
            frame = inspect.currentframe()
            if frame is not None:
                node = self._get_caller_node(frame)
                if node is not None and param_id not in node.global_parameters:
                    raise Exception(
                        "Attempting to access global parameter '%s', but it's "
                        "not listed in global_parameters of node %s" % (param_id, node.id),
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
            msg = f"Parameter {param_id} not found"
            raise Exception(msg)
        return param

    def get_parameter_value(self, param_id: str, *, required: bool = True) -> Any:  # noqa: ANN401
        param = self.get_parameter(param_id, required=required)
        if param is None:
            return None
        return param.value

    def set_parameter_value(self, param_id: str, value: Any):  # noqa: ANN401
        param = self.global_parameters.get(param_id)
        if param is None:
            msg = f"Parameter {param_id} not found"
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
            raise KeyError("Unknown option: %s" % id)

    def get_option(self, id: Literal['normalizer']) -> Any:
        pass

    def get_root_nodes(self) -> list[Node]:
        all_nodes = self.nodes.values()
        root_nodes = list(filter(lambda node: not node.output_nodes, all_nodes))
        return root_nodes

    def get_outcome_nodes(self) -> list[Node]:
        return [node for node in self.nodes.values() if node.is_outcome]

    def activate_scenario(self, scenario: Scenario):
        # Set the new parameters
        scenario.activate()
        self.active_scenario = scenario

    def get_default_scenario(self) -> Scenario:
        for scenario in self.scenarios.values():
            if scenario.default:
                return scenario
        raise Exception("No default scenario found")

    def generate_baseline_values(self):
        if self.baseline_values_generated:
            return
        if 'baseline' not in self.scenarios:
            return
        assert self.active_scenario
        old_scenario = self.active_scenario

        scenario = self.scenarios['baseline']
        with self.tracer.start_as_current_span('baseline'):
            self.activate_scenario(scenario)
            self.log.info('Generating baseline values')
            pc = PerfCounter('generate baseline values')
            for node in self.nodes.values():
                node.generate_baseline_values()
            self.log.info('Baseline values generated in %.1f ms' % pc.measure())
            self.activate_scenario(old_scenario)
        self.baseline_values_generated = True

    def get_all_parameters(self):
        """Return global and node-specific parameters."""
        for param in self.global_parameters.values():
            yield param
        for node in self.nodes.values():
            for param in node.get_parameters():
                yield param

    def print_all_parameters(self):
        """Print global and node-specific parameters."""
        for param in self.get_all_parameters():
            print('%s: %s' % (param.global_id, param.value))

    def pull_datasets(self):
        self.dataset_repo.set_target_commit(None)
        self.dataset_repo.pull_datasets()
        commit_id = self.dataset_repo.commit_id
        self.dataset_repo.set_target_commit(commit_id)
        self.instance.update_dataset_repo_commit(commit_id)

    def print_graph(self, include_datasets: bool = False):
        import inspect

        visited_nodes: set[Node] = set()
        node_class_cache: dict[type, str] = {}

        def make_node_tree(node: Node, tree: Tree | None = None) -> Tree:
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

            metrics = ', '.join([f"{m.quantity} [#a63fa4]{m.unit}[orchid]" for m in node.output_metrics.values()])
            unit_quantity = f'({metrics})'
            if node.yaml_lc is not None:
                url = 'file://%s#%d' % (node.yaml_fn, node.yaml_lc[0])
                node_name = f'[link={url}]{node.name}[/link]'
            else:
                node_name = str(node.name)
            node_str = f'{node_icon}[{node_color}]{node.id} [light_sea_green]{node_name} [orchid]{unit_quantity} {node_class_str}'
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
        for node in self.nodes.values():
            if node.unit is not None:
                unit_str = 'unit=%s  quantity=%s' % (str(node.unit), str(node.quantity))
            else:
                unit_str = ''
            print('id="%s"  nr_inputs=%s  nr_outputs=%s  type=%s%s' % (
                node.id, len(node.input_nodes), len(node.output_nodes), type(node).__name__, unit_str,
            ))

    def describe_unit(self, unit: Unit):
        formats = dict(
            short='~P',
            long='P',
            html_short='~H',
            html_long='H',
        )
        return {k: unit.format_babel(v) for k, v in formats.items()}  # type: ignore

    def get_actions(self) -> list['ActionNode']:
        from nodes.actions.action import ActionNode
        return [n for n in self.nodes.values() if isinstance(n, ActionNode)]

    def warning(self, msg: Any, *args):
        self.instance.warning(msg, *args)

    @contextmanager
    def run(self):
        with ExitStack() as stack:
            span_ctx = self.tracer.start_as_current_span("context run", attributes=dict(
                instance_id=self.instance.id,
                context_id=self.obj_id,
            ))
            stack.enter_context(span_ctx)
            stack.enter_context(self.cache)
            stack.enter_context(self.perf_context)
            if self.dataset_repo is not None:
                stack.enter_context(self.dataset_repo.lock.lock)
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

