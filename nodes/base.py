from dataclasses import dataclass
from typing import Iterable, Dict
from django.utils.translation import gettext_lazy as _
from dvc_pandas import load_dataset
import pandas as pd


class YearlyValue:
    year: int
    value: float


class Metric:
    identifier: str = None
    name: str = None
    unit: str = None


class Variable:
    identifier: str = None
    name: str = None
    unit: str = None

    def __init__(self, identifier, unit):
        self.identifier = identifier
        self.unit = unit

    def serialize(self):
        raise Exception('Implement in subclass')

    def deserialize(self):
        raise Exception('Implement in subclass')


class YearlyVariable(Variable):
    values: Iterable[YearlyValue]


@dataclass
class Dataset:
    identifier: str
    column: str = None


class Node:
    # identifier of the Node instance
    identifier: str = None
    output_metrics: Iterable[Metric]

    # name is the human-readable description for the Node class
    name: str = None

    input_datasets: Iterable[Dataset] = []
    input_nodes: Iterable  # of Nodes
    output_nodes: Iterable  # of Nodes

    def __init__(self, identifier: str, input_datasets: Iterable[Dataset] = None):
        self.identifier = identifier
        self.input_datasets = input_datasets
        self.input_nodes = []
        self.output_nodes = []

    def get_input_dataset(self):
        pass

    def compute(self) -> pd.DataFrame:
        pass

    def __str__(self):
        return '%s [%s]' % (self.identifier, str(type(self)))
