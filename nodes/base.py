from typing import Iterable, Dict
from django.utils.translation import gettext_lazy as _
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


class Node:
    # identifier of the Node instance
    identifier: str = None
    output_metrics: Iterable[Metric]

    # name is the human-readable description for the Node class
    name: str = None

    input_datasets: Dict[str, str]
    input_nodes: Iterable  # of Nodes

    def __init__(self, identifier: str, datasets: Iterable[str]):
        self.identifier = identifier

    def get_dataset(self):
        pass

    def compute(self) -> pd.DataFrame:
        pass
