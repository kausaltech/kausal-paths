from __future__ import annotations

from typing import TYPE_CHECKING
import dvc_pandas
from .datasets import Dataset
if TYPE_CHECKING:
    from .base import Node


class Context:
    nodes: dict[str, Node]
    datasets: dict[str, Dataset]

    def __init__(self):
        self.nodes = {}
        self.datasets = {}

    def load_dataset(self, identifier: str):
        if identifier in self.datasets:
            return self.datasets[identifier]
        df = dvc_pandas.load_dataset(identifier)
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
