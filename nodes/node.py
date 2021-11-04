from __future__ import annotations

import hashlib
import os
import io
import inspect
import json
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Literal, Optional, Union, overload

import numpy as np
import pandas as pd
import pint
import pint_pandas

from common.i18n import TranslatedString
from nodes.constants import EMISSION_FACTOR_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN, KNOWN_QUANTITIES, MILEAGE_QUANTITY, VALUE_COLUMN
from params import Parameter

from .context import Context
from .datasets import Dataset, JSONDataset
from .exceptions import NodeError


class Node:
    # identifier of the Node instance
    id: str
    # output_metrics: Iterable[Metric]

    # the id of the database row corresponding to this node
    database_id: Optional[int]

    # name is the human-readable label for the Node instance
    name: Optional[Union[TranslatedString, str]]

    # Description for the Node instance
    #
    # This gets mapped to NodeType.short_description in the GraphQL schema and
    # wrapped in a <p> tag.
    description: Optional[Union[TranslatedString, str]]

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

    # Cache last historical year
    _last_historical_year: Optional[int]
    context: Context

    debug: bool = False
    __post_init__: Callable[[Node], None]

    def __init__(
        self, id: str, context: Context, name, quantity: str, description,
        color: str = None, unit=None, target_year_goal=None,
        input_datasets: List[Dataset] = None,
    ):
        if quantity not in KNOWN_QUANTITIES:
            raise Exception(f"Quantity {quantity} is unknown")

        if input_datasets is None:
            input_datasets = []

        self.id = id
        self.context = context
        self.database_id = None
        self.name = name
        self.description = description
        self.color = color
        self.unit = unit
        if unit is None:
            raise NodeError(self, "Attempting to initialize node without a unit")
        self.quantity = quantity
        self.target_year_goal = target_year_goal
        self.input_dataset_instances = input_datasets
        self.input_nodes = []
        self.output_nodes = []
        self.baseline_values = None
        self.parameters = {}

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

    def get_input_datasets(self) -> List[Union[pd.DataFrame, pd.Series]]:
        dfs = []
        for ds in self.input_dataset_instances:
            df = ds.get_copy(self.context)
            if df.index.duplicated().any():
                raise NodeError(self, "Input dataset has duplicate index rows")
            dfs.append(df)
        return dfs

    @overload
    def get_input_dataset(self, required: Literal[True]) -> Union[pd.DataFrame, pd.Series]: ...

    @overload
    def get_input_dataset(self) -> Union[pd.DataFrame, pd.Series]: ...

    @overload
    def get_input_dataset(self, required: Literal[False]) -> Optional[Union[pd.DataFrame, pd.Series]]: ...

    def get_input_dataset(self, required: bool = True):
        """Gets the first (and only) dataset if it exists."""
        datasets = self.get_input_datasets()
        if not datasets:
            if required:
                raise NodeError(self, 'No input datasets, but node requires one')
            return None
        if len(datasets) != 1:
            raise NodeError(self, 'Expected only 1 input dataset, got %d' % len(datasets))
        df = datasets[0]
        assert isinstance(df, pd.DataFrame)
        return df

    def get_last_historical_year(self) -> Optional[int]:
        year = getattr(self, '_last_historical_year', None)
        if year is not None:
            return year

        if len(self.input_dataset_instances) != 1:
            return None
        ds = self.input_dataset_instances[0]
        df = ds.get_copy(self.context)
        if FORECAST_COLUMN not in df:
            return None

        year = df[~df[FORECAST_COLUMN]].index.max()
        if np.isnan(year):
            year = None
        else:
            year = int(year)

        self._last_historical_year = year
        return year

    def calculate_hash(self) -> bytes:
        h = hashlib.md5()
        for node in self.input_nodes:
            h.update(node.calculate_hash())
        for param in self.parameters.values():
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

        assert out is not None
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

    def is_compatible_unit(self, unit_a: Union[str, pint.Unit], unit_b: Union[str, pint.Unit]):
        if isinstance(unit_a, str):
            unit_a = self.context.unit_registry(unit_a).units
        if isinstance(unit_b, str):
            unit_b = self.context.unit_registry(unit_b).units
        if unit_a.dimensionality != unit_b.dimensionality:
            return False
        return True

    def ensure_output_unit(self, s: pd.Series, input_node: Node = None):
        node_pt = pint_pandas.PintType(self.unit)
        if hasattr(s, 'pint'):
            s_pt = pint_pandas.PintType(s.pint.units)
            if not self.unit.is_compatible_with(s.pint.units):
                if input_node is not None:
                    node_str = ' from node %s' % input_node.id
                else:
                    node_str = ''
                raise NodeError(self, 'Series with type %s%s is not compatible with %s' % (
                    s.pint.units, node_str, self.unit
                ))
        else:
            s_pt = None
        s = s.astype(float)
        if s_pt is not None:
            s = s.astype(s_pt)
        s = s.astype(node_pt)
        return s

    def get_downstream_nodes(self) -> List[Node]:
        # Depth-first traversal
        result = []
        closed = set()
        open = self.output_nodes.copy()
        while open:
            current = open.pop()
            if current not in closed:
                closed.add(current)
                result.append(current)
                open += current.output_nodes
        return result

    def get_upstream_nodes(self, filter: Callable[[Node], bool] = None) -> List[Node]:
        result = []
        closed = set()
        open = self.input_nodes.copy()
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

    def get_icon(self) -> Optional[str]:
        from nodes.actions import ActionNode
        if isinstance(self, ActionNode):
            return '⚒'
        elif self.quantity == EMISSION_QUANTITY:
            return '💨'
        elif self.quantity == ENERGY_QUANTITY:
            return '⚡'
        elif self.quantity == MILEAGE_QUANTITY:
            return '🚗'
        elif self.quantity == EMISSION_FACTOR_QUANTITY:
            return '✖'
        else:
            return None

    def __str__(self):
        return '%s [%s]' % (self.id, str(type(self)))

    def add_input_node(self, node):
        if node in self.input_nodes:
            raise Exception(f"Node {node} already added to input nodes for {self.id}")
        self.input_nodes.append(node)

    def add_output_node(self, node):
        if node in self.output_nodes:
            raise Exception(f"Node {node} already added to output nodes for {self.id}")
        self.output_nodes.append(node)

    def generate_baseline_values(self):
        assert self.baseline_values is None
        assert self.context.active_scenario.id == 'baseline'
        self.baseline_values = self.get_output()

    def serialize_input_data(self):
        if len(self.input_dataset_instances) != 1:
            datasets = [x.id for x in self.input_dataset_instances]
            raise NodeError(self, 'Too many input datasets: %s' % ', '.join(datasets))
        ds = self.input_dataset_instances[0]
        df = ds.get_copy(self.context)
        if not isinstance(df, pd.DataFrame) or FORECAST_COLUMN not in df.columns:
            raise Exception('Dataset %s is not suitable for serialization')

        try:
            unit = ds.get_unit(self.context)
        except Exception:
            # FIXME: Make this more robust
            unit = self.unit
        units = {}
        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            if hasattr(df[col], 'pint'):
                assert df[col].pint.units == unit
                df[col] = df[col].pint.m
            units[col] = str(unit)

        d = json.loads(df.to_json(orient='table'))
        fields = d['schema']['fields']
        for field in fields:
            if field['name'] in units:
                field['unit'] = units[field['name']]
        return d

    def validate_input_data(self, data: dict):
        return self._make_input_dataset(data)

    def _make_input_dataset(self, data: dict):
        sio = io.StringIO(json.dumps(data))
        old = self.serialize_input_data()
        old['data'] = data['data']
        df = pd.read_json(sio, orient='table')
        return old

    def replace_input_data(self, data: dict):
        if len(self.input_dataset_instances) != 1:
            raise NodeError(self, "Can't replace data for node with %d input datasets" % len(self.input_dataset_instances))

        d = self._make_input_dataset(data)
        old_ds = self.input_dataset_instances[0]
        try:
            unit = old_ds.get_unit(self.context)
        except Exception:
            # FIXME: Make this more robust
            unit = self.unit
        self.input_dataset_instances[0] = JSONDataset(
            id=old_ds.id, data=d, unit=unit
        )

        self._last_historical_year = None
