from __future__ import annotations
from dataclasses import dataclass, field

import os
import inspect
from re import S
from types import FrameType
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, overload, Literal
from datetime import datetime

import networkx as nx
import dvc_pandas
import pint
import pint_pandas
import rich
from common import polars as pl
from common import polars_ext  # noqa
from rich.tree import Tree

from common.cache import Cache
from params import Parameter
from params.discover import discover_parameter_types
from params.storage import SettingStorage

from .datasets import Dataset, DVCDataset, FixedDataset
from .units import unit_registry, Unit
from .perf import PerfContext

if TYPE_CHECKING:
    from .node import Node, Dimension
    from .instance import Instance
    from .scenario import CustomScenario, Scenario
    from nodes.actions.action import ActionEfficiencyPair, ActionNode


class Context:
    nodes: dict[str, Node]
    datasets: Dict[str, Dataset]
    dvc_datasets: Dict[str, dvc_pandas.Dataset]
    # global_parameters contains parameters that are not specific to a node.
    # Node-specific parameters are managed by the node.
    global_parameters: dict[str, Parameter]
    scenarios: dict[str, Scenario]
    custom_scenario: CustomScenario

    dimensions: dict[str, Dimension]
    """Global dimensions available for nodes."""

    target_year: int
    model_end_year: int
    unit_registry: CachingUnitRegistry
    dataset_repo: dvc_pandas.Repository
    active_scenario: Scenario
    supported_parameter_types: dict[str, type]
    cache: Cache
    skip_cache: bool = False
    check_mode: bool = False
    instance: Instance
    action_efficiency_pairs: list[ActionEfficiencyPair]
    setting_storage: Optional[SettingStorage]
    perf_context: PerfContext
    node_graph: nx.DiGraph
    baseline_values_generated: bool = False

    def __init__(
        self, dataset_repo: dvc_pandas.Repository, target_year: int,
        model_end_year: int | None = None,
    ):
        from nodes.actions import ActionNode

        # Avoid circular import
        self.Action = ActionNode

        self.perf_context = PerfContext()
        self.nodes = {}
        self.datasets = {}
        self.dvc_datasets = {}
        self.global_parameters = {}
        self.scenarios = {}
        self.target_year = target_year
        self.model_end_year = model_end_year or target_year
        self.unit_registry = unit_registry
        self.dataset_repo = dataset_repo
        self.supported_parameter_types = discover_parameter_types()
        self.cache = Cache(ureg=self.unit_registry, redis_url=os.getenv('REDIS_URL'))
        # will be set later
        self.instance = None  # type: ignore
        self.active_scenario = None  # type: ignore
        self.custom_scenario = None  # type: ignore
        self.action_efficiency_pairs = []
        self.dimensions = {}

    def finalize_nodes(self):
        """Finalize the node graph.

        Called when nodes and their connections have been configured.
        """

        g = nx.DiGraph()
        g.add_nodes_from([n.id for n in self.nodes.values()])
        for node in self.nodes.values():
            for output in node.output_nodes:
                g.add_edge(node.id, output.id)

        if not nx.is_directed_acyclic_graph(g):
            raise Exception("Node graph is not directed (there are loops between nodes)")

        self.node_graph = g

    def get_parameter_type(self, parameter_id: str) -> type:
        param_type = self.supported_parameter_types.get(parameter_id)
        if param_type is None:
            raise Exception("Unknown parameter: {param_id}")
        return param_type

    def load_dvc_dataset(self, id: str) -> dvc_pandas.Dataset:
        ds = self.dvc_datasets.get(id)
        if ds is None:
            if not self.dataset_repo.has_dataset(id):
                raise Exception('Dataset %s not found in DVC repo' % id)
            ds = self.dataset_repo.load_dataset(id, skip_pull_if_exists=True)
            self.dvc_datasets[id] = ds
        return ds

    def add_dataset(self, config: dict):
        assert config['id'] not in self.datasets
        ds = DVCDataset(**config)
        self.datasets[ds.id] = ds

    def add_node(self, node: Node):
        if node.id in self.nodes:
            raise Exception('Node %s already defined' % (node.id))
        self.nodes[node.id] = node
        for param_id in node.global_parameters:
            param = self.global_parameters[param_id]
            assert node not in param.subscription_nodes
            param.subscription_nodes.append(node)

    def get_node(self, id: str) -> Node:
        return self.nodes[id]

    def get_action(self, id: str) -> 'ActionNode':
        node = self.nodes[id]
        if not isinstance(node, self.Action):
            raise Exception("Node %s is not an action node" % id)
        return node

    def add_global_parameter(self, parameter: Parameter):
        if parameter.local_id in self.global_parameters:
            raise Exception(f"Global parameter {parameter.local_id} already defined")
        self.global_parameters[parameter.local_id] = parameter

    def _get_caller_node(self, frame: FrameType) -> Node | None:
        from nodes import Node

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
    def get_parameter(self, id: str, *, required: Literal[True] = True) -> Parameter: ...

    @overload
    def get_parameter(self, id: str, *, required: Literal[False]) -> Optional[Parameter]: ...

    @overload
    def get_parameter(self, id: str, *, required: bool) -> Optional[Parameter]: ...

    def get_parameter(self, id: str, *, required: bool = True) -> Optional[Parameter]:
        if self.check_mode:
            frame = inspect.currentframe()
            if frame is not None:
                node = self._get_caller_node(frame)
                if node is not None and id not in node.global_parameters:
                    raise Exception(
                        "Attempting to access global parameter '%s', but it's "
                        "not listed in global_parameters of node %s" % (id, node.id)
                    )

        param = None
        try:
            node_id, param_name = id.split('.', 1)
        except ValueError:
            param = self.global_parameters.get(id)
        else:
            if node_id in self.nodes:
                param = self.nodes[node_id].get_parameter(param_name, required=required)
        if param is None and required:
            raise Exception(f"Parameter {id} not found")
        return param

    def get_parameter_value(self, id: str, *, required: bool = True) -> Any:
        param = self.get_parameter(id, required=required)
        if param is None:
            return None
        return param.value

    def set_parameter_value(self, id: str, value: Any):
        param = self.global_parameters.get(id)
        if param is None:
            raise Exception(f"Parameter {id} not found")
        param.set(value)

    def add_scenario(self, scenario: Scenario):
        assert scenario.id not in self.scenarios
        self.scenarios[scenario.id] = scenario

    def set_custom_scenario(self, scenario: CustomScenario):
        assert self.custom_scenario is None
        self.add_scenario(scenario)
        self.custom_scenario = scenario

    def get_scenario(self, id) -> Scenario:
        return self.scenarios[id]

    def get_root_nodes(self) -> list[Node]:
        all_nodes = self.nodes.values()
        root_nodes = list(filter(lambda node: not node.output_nodes, all_nodes))
        return root_nodes

    def activate_scenario(self, scenario: Scenario):
        # Set the new parameters
        scenario.activate(self)
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
        self.activate_scenario(scenario)
        for node in self.nodes.values():
            node.generate_baseline_values()
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

    def print_graph(self, include_datasets=False):
        import inspect

        def make_node_tree(node: Node, tree: Tree | None = None) -> Tree:
            node_color = 'yellow'
            if isinstance(node, self.Action):
                node_color = 'green'
            elif node.quantity == 'emissions':
                node_color = 'magenta'

            node_icon = node.get_icon() or ''

            node_class = type(node)
            node_module = node_class.__module__
            module_file = inspect.getabsfile(node_class)
            line_nr = inspect.getsourcelines(node_class)[1]
            link = 'file://%s#%d' % (module_file, line_nr)
            node_class_str = f'[link={link}][grey50]{node_module}.[grey70]{node_class.__name__}[/link]'
            metrics = ', '.join([f"{m.quantity} [#a63fa4]{m.unit}[orchid]" for m in node.output_metrics.values()])
            unit_quantity = f'({metrics})'
            node_str = f'{node_icon}[{node_color}]{node.id} [light_sea_green]{node.name} [orchid]{unit_quantity} {node_class_str}'
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
            for in_node in node.input_nodes:
                make_node_tree(in_node, branch)
            return branch

        tree = Tree('Nodes')
        all_nodes = self.nodes.values()
        root_nodes = list(filter(lambda node: not node.output_nodes, all_nodes))
        for node in root_nodes:
            tree = make_node_tree(node, tree)
            rich.print(tree)

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
