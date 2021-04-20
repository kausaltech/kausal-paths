from __future__ import annotations

from typing import TYPE_CHECKING
import pint
import pint_pandas

import dvc_pandas
from .datasets import Dataset
if TYPE_CHECKING:
    from .node import Node
    from .params import Parameter
    from .scenario import Scenario


unit_registry = pint.UnitRegistry()
# By default, kt is knots, but here kilotonne is the most common
# usage.
unit_registry.define('kt = kilotonne')
pint.set_application_registry(unit_registry)
pint_pandas.PintType.ureg = unit_registry


class Context:
    nodes: dict[str, Node]
    datasets: dict[str, Dataset]
    params: dict[str, Parameter]
    scenarios: dict[str, Scenario]
    target_year: int
    unit_registry: pint.UnitRegistry
    # The URL for the default dataset repo for dvc-pandas
    dataset_repo_url: str
    active_scenario: Scenario

    def __init__(self):
        from nodes.actions import Action
        # Avoid circular import
        self.Action = Action

        self.nodes = {}
        self.datasets = {}
        self.scenarios = {}
        self.unit_registry = unit_registry
        self.active_scenario = None

    def load_dataset(self, identifier: str):
        if identifier in self.datasets:
            return self.datasets[identifier]
        df = dvc_pandas.load_dataset(identifier, repo_url=self.dataset_repo_url)
        self.datasets[identifier] = df
        return df

    def add_dataset(self, config: dict):
        assert config['id'] not in self.datasets
        ds = Dataset(**config)
        df = ds.load(self)
        self.datasets[config['id']] = df

    def add_node(self, node: Node):
        assert node.id not in self.nodes
        self.nodes[node.id] = node

    def get_node(self, id) -> Node:
        return self.nodes[id]

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
        # Reset every action node to its default params
        for node in self.nodes.values():
            if not isinstance(node, self.Action):
                continue
            node.set_params(node.param_defaults)

        # Set the new parameters
        scenario.activate()
        self.active_scenario = scenario

    def generate_baseline_values(self):
        old_scenario = self.active_scenario

        scenario = self.scenarios['baseline']
        self.activate_scenario(scenario)
        for node in self.nodes.values():
            node.baseline_values = node.get_output()

        self.activate_scenario(old_scenario)
