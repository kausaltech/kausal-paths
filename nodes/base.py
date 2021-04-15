from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict
from django.utils.translation import gettext_lazy as _
import dvc_pandas
import pandas as pd

from .datasets import Dataset
from .context import Context


class Variable:
    id: str = None
    name: str = None
    unit: str = None

    def __init__(self, id, unit):
        self.id = id
        self.unit = unit

    def serialize(self):
        raise Exception('Implement in subclass')

    def deserialize(self):
        raise Exception('Implement in subclass')


class Node:
    # identifier of the Node instance
    id: str = None
    # output_metrics: Iterable[Metric]

    # name is the human-readable description for the Node class
    name: str = None

    input_datasets: Iterable[Dataset] = []
    input_nodes: Iterable[Node]
    output_nodes: Iterable[Node]
    context: Context

    def __init__(self, context: Context, id: str, input_datasets: Iterable[Dataset] = None):
        self.context = context
        self.id = id
        self.input_datasets = input_datasets or []
        self.input_nodes = []
        self.output_nodes = []

    def get_input_datasets(self):
        dfs = []
        for ds in self.input_datasets:
            df = ds.load(self.context)
            dfs.append(df)
        return dfs

    def get_input_dataset(self):
        """Gets the first (and only) dataset if it exists."""
        if not self.input_datasets:
            return None

        datasets = self.get_input_datasets()
        assert len(datasets) == 1
        return datasets[0]

    def get_output(self) -> pd.DataFrame:
        # FIXME: Implement caching
        return self.compute()

    def compute(self) -> pd.DataFrame:
        raise Exception('Implement in subclass')

    def __str__(self):
        return '%s [%s]' % (self.id, str(type(self)))
