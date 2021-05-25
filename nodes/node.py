from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Type, Union
import pint
import pandas as pd
import pint_pandas

from common.i18n import TranslatedString

from .datasets import Dataset
from .context import Context
from .exceptions import NodeError
from params import Parameter


class Node:
    # identifier of the Node instance
    id: str = None
    # output_metrics: Iterable[Metric]

    # name is the human-readable description for the Node class
    name: TranslatedString = None

    # if the node has an established visualisation color
    color: str = None

    # output unit (from pint)
    unit: pint.Unit
    # output quantity (like 'energy' or 'emissions')
    quantity: str = None

    input_datasets: Iterable[Dataset] = []
    input_nodes: Iterable[Node]
    output_nodes: Iterable[Node]
    params: Dict[str, Parameter]

    # Output for the node in the baseline scenario
    baseline_values: Optional[pd.DataFrame]

    context: Context

    def __init__(self, context: Context, id: str, input_datasets: Iterable[Dataset] = None):
        self.context = context
        self.id = id
        self.input_datasets = input_datasets or []
        self.input_nodes = []
        self.output_nodes = []
        self.params = {}

        # Call the subclass post-init method if it is defined
        if hasattr(self, '__post_init__'):
            self.__post_init__()

    def register_param(self, id: str, param_class: Type):
        global_id = '%s.%s' % (self.id, id)
        param = param_class(id=global_id, node=self)
        assert global_id not in self.context.params
        self.context.params[global_id] = param
        self.params[id] = param

    def register_params(self):
        pass

    def get_param(self, id: str):
        if id in self.params:
            return self.params[id]
        return self.context.get_param(id)

    def get_param_value(self, id: str) -> Any:
        return self.get_param(id).value

    def set_param_value(self, id: str, value: Any):
        if id not in self.params:
            raise NodeError(self, 'Node param %s not found' % id)
        self.params[id].set_value(value)

    def get_input_datasets(self):
        dfs = []
        for ds in self.input_datasets:
            df = ds.load(self.context).copy()
            if df.index.duplicated().any():
                raise NodeError(self, "Input dataset has duplicate index rows")
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
        out = self.compute()
        if out is None:
            return None
        if out.index.duplicated().any():
            raise NodeError(self, "Node output has duplicate index rows")
        return out.copy()

    def get_target_year(self) -> int:
        return self.context.target_year

    def compute(self) -> pd.DataFrame:
        raise Exception('Implement in subclass')

    @property
    def ureg(self) -> pint.UnitRegistry:
        return self.context.unit_registry

    def is_compatible_unit(self, unit_a: Union[str, pint.Unit], unit_b: Union[str, pint.Unit]):
        if isinstance(unit_a, str):
            unit_a = self.ureg(unit_a).units
        if isinstance(unit_b, str):
            unit_b = self.ureg(unit_b).units
        if unit_a.dimensionality != unit_b.dimensionality:
            return False
        return True

    def ensure_output_unit(self, s: pd.Series):
        pt = pint_pandas.PintType(self.unit)
        if hasattr(s, 'pint'):
            if not self.unit.is_compatible_with(s.pint.units):
                raise NodeError(self, 'Series with type %s is not compatible with %s' % (
                    s.pint.units, self.unit
                ))
        return s.astype(pt)

    def __str__(self):
        return '%s [%s]' % (self.id, str(type(self)))
