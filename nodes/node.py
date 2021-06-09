from __future__ import annotations
from types import FunctionType
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN

from typing import Any, Dict, Iterable, List, Optional, Type, Union
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

    # name is the human-readable label for the Node instance
    name: TranslatedString = None

    # description for the Node instance
    description: TranslatedString = None

    # if the node has an established visualisation color
    color: str = None

    # output unit (from pint)
    unit: pint.Unit
    # output quantity (like 'energy' or 'emissions')
    quantity: str = None

    input_datasets: Iterable[Dataset] = []
    input_nodes: Iterable[Node]
    output_nodes: Iterable[Node]

    # Parameters with their values
    params: Dict[str, Parameter]

    # All allowed parameters for this class or object
    allowed_params: Iterable[Parameter]

    # Output for the node in the baseline scenario
    baseline_values: Optional[pd.DataFrame]

    context: Context

    def __init__(self, context: Context, id: str, input_datasets: Iterable[Dataset] = None):
        self.context = context
        self.id = id
        self.input_datasets = input_datasets or []
        self.input_nodes = []
        self.output_nodes = []
        self.allowed_params = []
        self.baseline_values = None
        self.params = {}

        # Call the subclass post-init method if it is defined
        if hasattr(self, '__post_init__'):
            self.__post_init__()

    def register_param(self, param: Parameter):
        local_id = param.id
        global_id = '%s.%s' % (self.id, param.id)
        param.id = global_id
        param.node = self
        assert global_id not in self.context.params
        self.context.params[global_id] = param
        assert local_id not in self.params
        self.params[local_id] = param

    def register_params(self):
        for param in self.allowed_params:
            self.register_param(param)

    def get_param(self, id: str, local: bool = False, required: bool = True):
        # First attempt to find the parameter in the node-local parameter
        # set, then fall back to global parameters (unless 'local' specifically
        # requested).
        if id in self.params:
            return self.params[id]
        if local:
            if required:
                raise NodeError(self, 'Local parameter %s not found' % id)
            else:
                return None
        return self.context.get_param(id, required=required)

    def get_param_value(self, id: str, local: bool = False, required: bool = True) -> Any:
        param = self.get_param(id, local=local, required=required)
        if param is None:
            return None
        return param.value

    def set_param_value(self, id: str, value: Any):
        if id not in self.params:
            raise NodeError(self, 'Node param %s not found' % id)
        self.params[id].set(value)

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

    def get_output(self, target_node: Node = None) -> pd.DataFrame:
        # FIXME: Implement caching
        out = self.compute()
        if out is None:
            return None
        if out.index.duplicated().any():
            raise NodeError(self, "Node output has duplicate index rows")

        # If a node has multiple outputs, we can specify only one series
        # to include.
        if target_node is not None:
            if target_node.id in out.columns:
                cols = [target_node.id]
                if FORECAST_COLUMN in out.columns:
                    cols.append(FORECAST_COLUMN)
                out = out[cols]
                out = out.rename(columns={target_node.id: VALUE_COLUMN})

        return out.copy()

    def print_output(self):
        df = self.get_output()

        # Strip pint units before displaying to prevent a warning
        for col in list(df.columns):
            if hasattr(df[col], 'pint'):
                df[col] = df[col].pint.m

        if self.baseline_values is not None and VALUE_COLUMN in df.columns:
            df['Baseline'] = self.baseline_values[VALUE_COLUMN]
            if hasattr(df.Baseline, 'pint'):
                df.Baseline = df.Baseline.pint.m

        print(df)

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

    def get_descendant_nodes(self, proper=False) -> List[Node]:
        # Depth-first traversal
        result = []
        closed = set()
        if proper:
            open = self.output_nodes.copy()
        else:
            open = [self]
        while open:
            current = open.pop()
            if current not in closed:
                closed.add(current)
                result.append(current)
                open += current.output_nodes
        return result

    def get_upstream_nodes(self, filter: FunctionType = None) -> List[Node]:
        result = []
        closed = set()
        open = [self]
        while open:
            current = open.pop()
            if current not in closed:
                closed.add(current)
                if filter is None or filter(current):
                    result.append(current)
                open += current.input_nodes
        return result


    def __str__(self):
        return '%s [%s]' % (self.id, str(type(self)))
