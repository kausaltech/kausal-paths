from __future__ import annotations

import hashlib
import os
import inspect
from types import FunctionType
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, TYPE_CHECKING, Union

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
    name: Optional[TranslatedString]

    # description for the Node instance
    description: Optional[TranslatedString]

    # if the node has an established visualisation color
    color: Optional[str]

    # output unit (from pint)
    unit: pint.Unit
    # output quantity (like 'energy' or 'emissions')
    quantity: Optional[str]

    # set if this node has a specific goal for the simulation target year
    target_year_goal: Optional[float]

    input_datasets: List[str]

    input_dataset_instances: List[Dataset]
    input_nodes: List[Node]
    output_nodes: List[Node]

    # Parameters with their values
    parameters: Dict[str, Parameter]

    # All allowed parameters for this class
    allowed_parameters: ClassVar[Iterable[Parameter]]

    # Output for the node in the baseline scenario
    baseline_values: Optional[pd.DataFrame]

    debug: bool = False
    content: Optional[NodeContent]
    __post_init__: Callable[[Node], None]

    def __init__(
        self, id, name, description=None, color=None, unit=None, quantity=None, target_year_goal=None,
        input_datasets: List[Dataset] = None,
    ):
        if input_datasets is None:
            input_datasets = []

        self.id = id
        self.name = name
        self.description = description
        self.color = color
        self.unit = unit
        self.quantity = quantity
        self.target_year_goal = target_year_goal
        self.input_dataset_instances = input_datasets
        self.input_nodes = []
        self.output_nodes = []
        self.baseline_values = None
        self.parameters = {}
        self.content = None

        # Call the subclass post-init method if it is defined
        if hasattr(self, '__post_init__'):
            self.__post_init__()

    def add_parameter(self, param: Parameter):
        if param.local_id in self.parameters:
            raise Exception(f"Local parameter {param.local_id} already defined for node {self.id}")
        self.parameters[param.local_id] = param
        param.set_node(self)

    def get_parameters(self):
        for param in self.parameters.values():
            yield param

    def get_parameter(self, local_id: str, required: bool = True):
        """Get the parameter with the given local id from this node's parameters."""
        if local_id in self.parameters:
            return self.parameters[local_id]
        if required:
            raise NodeError(self, f"Local parameter {local_id} not found for node {self.id}")
        return None

    def get_parameter_value(self, id: str, required: bool = True) -> Any:
        param = self.get_parameter(id, required=required)
        if param is None:
            return None
        return param.value

    def set_parameter_value(self, local_id: str, value: Any):
        if local_id not in self.parameters:
            raise NodeError(self, f"Local parameter {local_id} not found for node {self.id}")
        self.parameters[local_id].set(value)

    def get_input_datasets(self, context: Context) -> List[Union[pd.DataFrame, pd.Series]]:
        dfs = []
        for ds in self.input_dataset_instances:
            df = ds.get_copy(context)
            if df.index.duplicated().any():
                raise NodeError(self, "Input dataset has duplicate index rows")
            dfs.append(df)
        return dfs

    def get_input_dataset(self, context: Context) -> Optional[Union[pd.DataFrame, pd.Series]]:
        """Gets the first (and only) dataset if it exists."""
        datasets = self.get_input_datasets(context)
        if not datasets:
            return None
        if len(datasets) != 1:
            raise NodeError(self, 'Expected only 1 input dataset, got %d' % len(datasets))
        return datasets[0]

    def calculate_hash(self, context: Context) -> bytes:
        h = hashlib.md5()
        for node in self.input_nodes:
            h.update(node.calculate_hash(context))
        for param in self.parameters.values():
            h.update(param.calculate_hash())
        for ds in self.input_dataset_instances:
            h.update(ds.calculate_hash(context))
        for klass in type(self).mro():
            try:
                mod_mtime = os.path.getmtime(inspect.getfile(klass))
            except TypeError:
                continue
            h.update(str(mod_mtime).encode('utf8'))
        return h.digest()

    def get_output(self, context: Context, target_node: Node = None) -> pd.DataFrame:
        node_hash = self.calculate_hash(context).hex()
        out = context.cache.get(node_hash)
        if out is None or self.debug or context.skip_cache:
            try:
                out = self.compute(context)
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
            context.cache.set(node_hash, out)

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

    def print_output(self, context: Context):
        df = self.get_output(context)
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

    def get_target_year(self, context: Context) -> int:
        return context.target_year

    def compute(self, context: Context) -> pd.DataFrame:
        raise Exception('Implement in subclass')

    def is_compatible_unit(self, context: Context, unit_a: Union[str, pint.Unit], unit_b: Union[str, pint.Unit]):
        if isinstance(unit_a, str):
            unit_a = context.unit_registry(unit_a).units
        if isinstance(unit_b, str):
            unit_b = context.unit_registry(unit_b).units
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
        pass

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

    def add_input_node(self, node):
        if node in self.input_nodes:
            raise Exception(f"Node {node} already added to input nodes for {self.id}")
        self.input_nodes.append(node)

    def add_output_node(self, node):
        if node in self.output_nodes:
            raise Exception(f"Node {node} already added to output nodes for {self.id}")
        self.output_nodes.append(node)

    def generate_baseline_values(self, context):
        assert self.baseline_values is None
        assert context.active_scenario.id == 'baseline'
        self.baseline_values = self.get_output(context)
