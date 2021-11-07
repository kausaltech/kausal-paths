from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import dvc_pandas
import pint
import pint_pandas
import rich
from rich.tree import Tree

from common.cache import Cache
from params import Parameter
from params.discover import discover_parameter_types

from .datasets import Dataset, DVCDataset, FixedDataset

if TYPE_CHECKING:
    from .node import Node
    from .scenario import CustomScenario, Scenario


unit_registry = pint.UnitRegistry(
    preprocessors=[
        lambda s: s.replace('%', ' percent '),
    ],
    on_redefinition='raise'
)

# By default, kt is knots, but here kilotonne is the most common
# usage.
del unit_registry._units['kt']
unit_registry.define('kt = kilotonne')
# Mega-kilometers is often used for mileage
unit_registry.define('Mkm = gigameters')
# We also need population
unit_registry.define('person = [population] = cap = case = py')
unit_registry.define('DALY = [disease_burden] = _ = YLL = YLD = QALY')
unit_registry.define('[mass_concentration] = [mass] / [mass]')
unit_registry.define('ppm = 1e-6 * kg / kg')
unit_registry.define('ppb = 1e-9 * kg / kg')
unit_registry.define('ppt = 1e-12 * kg / kg')
unit_registry.define('[ingestion] = [mass] / [time] / [population]')
# unit_registry.define('case = count')
unit_registry.define('[incidence] = 1 / [population]')
#unit_registry.define('py = person * a')
unit_registry.define('_100000py = 100000 * py')
# unit_registry.define('[case_burden] = [disease_burden] / [population]')
unit_registry.define(pint.unit.UnitDefinition(
    'percent', '%', (), pint.converters.ScaleConverter(0.01)
))
unit_registry.default_format = '~P'
pint.set_application_registry(unit_registry)
pint_pandas.PintType.ureg = unit_registry  # type: ignore


class Context:
    nodes: dict[str, Node]
    datasets: Dict[str, Dataset]
    dvc_datasets: Dict[str, dvc_pandas.Dataset]
    # global_parameters contains parameters that are not specific to a node.
    # Node-specific parameters are managed by the node.
    global_parameters: dict[str, Parameter]
    scenarios: dict[str, Scenario]
    custom_scenario: CustomScenario
    target_year: int
    unit_registry: pint.UnitRegistry
    dataset_repo: dvc_pandas.Repository
    active_scenario: Scenario
    supported_parameter_types: dict[str, type]
    cache: Cache
    skip_cache: bool = False

    def __init__(self, dataset_repo, target_year):
        from nodes.actions import ActionNode

        # Avoid circular import
        self.Action = ActionNode

        self.nodes = {}
        self.datasets = {}
        self.dvc_datasets = {}
        self.global_parameters = {}
        self.scenarios = {}
        self.custom_scenario = None
        self.target_year = target_year
        self.unit_registry = unit_registry
        self.dataset_repo = dataset_repo
        self.active_scenario = None
        self.supported_parameter_types = discover_parameter_types()
        self.cache = Cache(ureg=self.unit_registry, redis_url=os.getenv('REDIS_URL'))

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

    def get_node(self, id: str) -> Node:
        return self.nodes[id]

    def add_global_parameter(self, parameter: Parameter):
        if parameter.local_id in self.global_parameters:
            raise Exception(f"Global parameter {parameter.local_id} already defined")
        self.global_parameters[parameter.local_id] = parameter

    def get_parameter(self, id: str, required: bool = True) -> Optional[Parameter]:
        try:
            node_id, param_name = id.split('.', 1)
        except ValueError:
            param = self.global_parameters.get(id)
        else:
            param = self.nodes[node_id].get_parameter(param_name, required=required)
        if param is None and required:
            raise Exception(f"Parameter {id} not found")
        return param

    def get_parameter_value(self, id: str, required: bool = True) -> Any:
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
        assert self.active_scenario
        old_scenario = self.active_scenario

        scenario = self.scenarios['baseline']
        self.activate_scenario(scenario)
        for node in self.nodes.values():
            node.generate_baseline_values()
        self.activate_scenario(old_scenario)

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
        self.dataset_repo.pull_datasets()

    def print_graph(self, include_datasets=False):
        import inspect

        def make_node_tree(node: Node, tree: Tree = None) -> Tree:
            node_icon = ''
            node_color = 'yellow'
            if isinstance(node, self.Action):
                node_color = 'green'
            elif node.quantity == 'emissions':
                node_color = 'magenta'
                node_icon = '💨'
            elif node.quantity == 'population':
                node_icon = '👪'
            if node_icon:
                node_icon += ' '

            node_class = type(node)
            node_module = node_class.__module__
            module_file = inspect.getabsfile(node_class)
            line_nr = inspect.getsourcelines(node_class)[1]
            link = 'file://%s#%d' % (module_file, line_nr)
            node_class_str = f'[link={link}][grey50]{node_module}.[grey70]{node_class.__name__}[/link]'
            unit_quantity = f'[orchid]({node.quantity}: {node.unit})'
            node_str = f'{node_icon}[{node_color}]{node.id} [light_sea_green]{node.name} {unit_quantity} {node_class_str}'
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

    def describe_unit(self, unit: pint.Unit):
        formats = dict(
            short='~P',
            long='P',
            html_short='~H',
            html_long='H',
        )
        return {k: unit.format_babel(v) for k, v in formats.items()}
