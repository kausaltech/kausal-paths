from __future__ import annotations
import os

from typing import Any, Dict, Optional, TYPE_CHECKING
import pint
import pint_pandas

import dvc_pandas
from .datasets import Dataset
from common.cache import Cache
from params import Parameter
from params.discover import discover_parameter_types

if TYPE_CHECKING:
    from .node import Node
    from .scenario import CustomScenario, Scenario


unit_registry = pint.UnitRegistry(preprocessors=[
    lambda s: s.replace('%', ' percent '),
])

# By default, kt is knots, but here kilotonne is the most common
# usage.
unit_registry.define('kt = kilotonne')
# Mega-kilometers is often used for mileage
unit_registry.define('Mkm = gigameters')
# We also need population
unit_registry.define('person = [population] = cap')
unit_registry.define(pint.unit.UnitDefinition(
    'percent', '%', (), pint.converters.ScaleConverter(0.01)
))
unit_registry.default_format = '~P'
pint.set_application_registry(unit_registry)
pint_pandas.PintType.ureg = unit_registry


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
        ds = self.datasets.get(id)
        if ds is None:
            if not self.dataset_repo.has_dataset(id):
                raise Exception('Dataset %s not found in DVC repo' % id)
            ds = self.dataset_repo.load_dataset(id)
            self.dvc_datasets[id] = ds
        return ds

    def add_dataset(self, config: dict):
        assert config['id'] not in self.datasets
        ds = Dataset(**config)
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

    def compute(self):
        all_nodes = self.nodes.values()
        root_nodes = list(filter(lambda node: not node.output_nodes, all_nodes))
        assert len(root_nodes) == 1
        return root_nodes[0].compute(self)

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
            node.generate_baseline_values(self)
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

    def print_graph(self, node=None, indent=0):
        from colored import fg, attr

        if node is None:
            all_nodes = self.nodes.values()
            root_nodes = list(filter(lambda node: not node.output_nodes, all_nodes))
            if len(root_nodes) != 1:
                raise Exception('Too many root nodes: %s' % (', '.join([x.id for x in root_nodes])))
            node = root_nodes[0]

        if isinstance(node, self.Action):
            node_color = 'green'
        else:
            node_color = 'yellow'
        node_str = f"{fg(node_color)}{node.id} "
        node_str += f"{fg('grey_50')}{str(type(node))} "
        node_str += attr('reset')
        print('  ' * indent + node_str)
        for in_node in node.input_nodes:
            self.print_graph(in_node, indent + 1)
