from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
import pint
import pint_pandas

import dvc_pandas
from params.discover import discover_parameters
from params import Parameter

from .datasets import Dataset
if TYPE_CHECKING:
    from .node import Node
    from .scenario import Scenario


unit_registry = pint.UnitRegistry(preprocessors=[
    lambda s: s.replace('%', ' percent '),
])

# By default, kt is knots, but here kilotonne is the most common
# usage.
unit_registry.define('kt = kilotonne')
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
    datasets: dict[str, Dataset]
    params: dict[str, Parameter]
    scenarios: dict[str, Scenario]
    target_year: int
    unit_registry: pint.UnitRegistry
    dataset_repo: dvc_pandas.Repository
    active_scenario: Scenario
    supported_params: dict[str, type]

    def __init__(self):
        from nodes.actions import ActionNode
        # Avoid circular import
        self.Action = ActionNode

        self.nodes = {}
        self.datasets = {}
        self.scenarios = {}
        self.params = {}
        self.unit_registry = unit_registry
        self.active_scenario = None
        self.supported_params = discover_parameters()

    def load_dataset(self, identifier: str):
        if identifier in self.datasets:
            return self.datasets[identifier].copy()
        dataset = self.dataset_repo.load_dataset(identifier)
        self.datasets[identifier] = dataset.df
        return dataset.df.copy()

    def add_dataset(self, config: dict):
        assert config['id'] not in self.datasets
        ds = Dataset(**config)
        df = ds.load(self)
        self.datasets[config['id']] = df

    def add_node(self, node: Node):
        if node.id in self.nodes:
            raise Exception('Node %s already defined' % (node.id))
        self.nodes[node.id] = node

        assert node.context == self
        node.register_params()

    def get_node(self, id: str) -> Node:
        return self.nodes[id]

    def get_param(self, id: str, required: bool = True) -> Optional[Parameter]:
        if id not in self.params:
            if not required:
                return None
            raise Exception('Param %s not found' % id)
        return self.params[id]

    def get_param_value(self, id: str, required: bool = True) -> Any:
        param = self.get_param(id, required=required)
        if param is None:
            return None
        return param.value

    def set_param_value(self, id: str, value: Any):
        param = self.params.get(id)
        if param is None:
            raise Exception('Param %s not found' % id)
        param.set(value)

    def add_scenario(self, scenario: Scenario):
        assert scenario.id not in self.scenarios
        self.scenarios[scenario.id] = scenario

    def get_scenario(self, id) -> Scenario:
        return self.scenarios[id]

    def compute(self):
        all_nodes = self.nodes.values()
        root_nodes = list(filter(lambda node: not node.output_nodes, all_nodes))
        assert len(root_nodes) == 1
        return root_nodes[0].compute()

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
        old_scenario = self.active_scenario

        scenario = self.scenarios['baseline']
        self.activate_scenario(scenario)
        for node in self.nodes.values():
            node.baseline_values = node.get_output()

        self.activate_scenario(old_scenario)

    def print_params(self):
        for param_id, param in self.params.items():
            print('%s: %s' % (param_id, param.value))

    def pull_datasets(self):
        self.dataset_repo.pull_datasets()
