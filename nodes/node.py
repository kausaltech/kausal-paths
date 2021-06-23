from __future__ import annotations

import hashlib
import os
import inspect
from types import FunctionType
from typing import Any, Callable, Dict, Iterable, List, Optional, TYPE_CHECKING, Union

import pandas as pd
import pint
import pint_pandas

from common.i18n import TranslatedString
from nodes.constants import FORECAST_COLUMN, KNOWN_QUANTITIES, VALUE_COLUMN
from params import Parameter

from .context import Context
from .datasets import Dataset
from .exceptions import NodeError

if TYPE_CHECKING:
    from pages.models import NodeContent


class Node:
    # identifier of the Node instance
    id: str
    # output_metrics: Iterable[Metric]

    # name is the human-readable label for the Node instance
    name: TranslatedString = None

    # description for the Node instance
    description: TranslatedString = None

    # if the node has an established visualisation color
    color: Optional[str] = None

    # output unit (from pint)
    unit: pint.Unit
    # output quantity (like 'energy' or 'emissions')
    quantity: Optional[str] = None

    # set if this node has a specific goal for the simulation target year
    target_year_goal: Optional[float] = None

    input_datasets: List[str]
    # List of global parameters that this node requires
    input_params: List[str]

    input_dataset_instances: List[Dataset]
    input_nodes: List[Node]
    output_nodes: List[Node]

    # Parameters with their values
    params: Dict[str, Parameter]
    # Maps a scenario to scenario-specific parameters and their values
    scenario_params: Dict[str, Dict[str, Parameter]]

    # All allowed parameters for this class or object
    allowed_params: Iterable[Parameter]

    # Output for the node in the baseline scenario
    baseline_values: Optional[pd.DataFrame]

    context: Context
    debug: bool = False
    content: Optional[NodeContent]
    __post_init__: Callable[[Node], None]

    def __init__(self, context: Context, id: str, input_datasets: List[Dataset] = None):
        self.context = context
        self.id = id
        if input_datasets:
            self.input_dataset_instances = input_datasets
        else:
            self.input_dataset_instances = []
        self.input_params = getattr(self, 'input_params', [])

        self.input_nodes = []
        self.output_nodes = []
        self.allowed_params = []
        self.baseline_values = None
        self.params = {}
        self.content = None
        self.scenario_params = {}

        # Call the subclass post-init method if it is defined
        if hasattr(self, '__post_init__'):
            self.__post_init__()

    def register_param(self, param: Parameter):
        param.set_node(self)
        assert param.id not in self.context.params
        self.context.params[param.id] = param
        assert param.node_relative_id not in self.params
        self.params[param.node_relative_id] = param

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

        # if id not in self.input_params:
        #     raise NodeError(self, 'Node is trying to access parameter %s but it is not listed in Node.input_params' % id)

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

    def get_input_datasets(self) -> List[Union[pd.DataFrame, pd.Series]]:
        dfs = []
        for ds in self.input_dataset_instances:
            df = ds.get_copy(self.context)
            if df.index.duplicated().any():
                raise NodeError(self, "Input dataset has duplicate index rows")
            dfs.append(df)
        return dfs

    def get_input_dataset(self) -> Optional[Union[pd.DataFrame, pd.Series]]:
        """Gets the first (and only) dataset if it exists."""
        datasets = self.get_input_datasets()
        if not datasets:
            return None
        if len(datasets) != 1:
            raise NodeError(self, 'Expected only 1 input dataset, got %d' % len(datasets))
        return datasets[0]

    def calculate_hash(self) -> bytes:
        h = hashlib.md5()
        for node in self.input_nodes:
            h.update(node.calculate_hash())
        for param in self.params.values():
            h.update(param.calculate_hash())
        for ds in self.input_dataset_instances:
            h.update(ds.calculate_hash(self.context))
        for klass in type(self).mro():
            try:
                mod_mtime = os.path.getmtime(inspect.getfile(klass))
            except TypeError:
                continue
            h.update(str(mod_mtime).encode('utf8'))
        return h.digest()

    def get_output(self, target_node: Node = None) -> pd.DataFrame:
        node_hash = self.calculate_hash().hex()
        out = self.context.cache.get(node_hash)
        if out is None or self.debug or self.context.skip_cache:
            try:
                out = self.compute()
            except Exception as e:
                print('Exception when computing node %s' % self.id)
                raise e
            if out is None:
                raise NodeError(self, "Node returned no output")
            cache_hit = False
        else:
            cache_hit = True

        if out is None:
            return None
        if out.index.duplicated().any():
            raise NodeError(self, "Node output has duplicate index rows")
        if FORECAST_COLUMN in out.columns:
            if out.dtypes[FORECAST_COLUMN] != bool:
                raise NodeError(self, "Forecast column is not a boolean")

        if not cache_hit:
            self.context.cache.set(node_hash, out)

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
        if self.baseline_values is not None and VALUE_COLUMN in df.columns:
            df['Baseline'] = self.baseline_values[VALUE_COLUMN]
        self.print_pint_df(df)

    def print_pint_df(self, df: pd.DataFrame):
        pint_cols = [col for col in df.columns if hasattr(df[col], 'pint')]
        if not pint_cols:
            print(df)
            return

        out = df[pint_cols].pint.dequantify()
        for col in df.columns:
            if col in pint_cols:
                continue
            out[col] = df[col]
        print(out)

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

    def ensure_output_unit(self, s: pd.Series, input_node: Node = None):
        pt = pint_pandas.PintType(self.unit)
        if hasattr(s, 'pint'):
            if not self.unit.is_compatible_with(s.pint.units):
                if input_node is not None:
                    node_str = ' from node %s' % input_node.id
                else:
                    node_str = ''
                raise NodeError(self, 'Series with type %s%s is not compatible with %s' % (
                    s.pint.units, node_str, self.unit
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
                if current is not self:
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

    def on_scenario_created(self, scenario):
        """Called when a scenario is created with this node among the nodes to be notified."""
        scenario_params = self.scenario_params.get(scenario.id, {})
        for param_id, val in scenario_params.items():
            param = self.get_param(param_id, local=True)
            scenario.params[param.id] = val

    def __str__(self):
        return '%s [%s]' % (self.id, str(type(self)))

    @property
    def short_description(self):
        if self.content is not None:
            # FIXME: Format RichTextField?
            return self.content.short_description or self.description

    @property
    def body(self):
        if self.content is None:
            return None
        # FIXME: Format RichTextField?
        return self.content.body

    def check(self):
        if self.quantity is None:
            raise NodeError(self, 'No quantity set')
        if self.quantity not in KNOWN_QUANTITIES:
            raise NodeError(self, 'Quantity %s is unknown' % self.quantity)
