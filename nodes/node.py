from __future__ import annotations
import base64

import logging
import hashlib
import os
import io
import inspect
import json
import typing
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Union, Set, overload

import numpy as np
import pandas as pd
import pint
import pint_pandas
from rich import print as pprint

from common.i18n import TranslatedString, get_modeltrans_attrs_from_str
from common.utils import hash_unit
from nodes.constants import (
    EMISSION_FACTOR_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN,
    MILEAGE_QUANTITY, VALUE_COLUMN, YEAR_COLUMN,
    ensure_known_quantity
)
from params import Parameter

from .context import Context
from .datasets import Dataset, JSONDataset
from .exceptions import NodeError

if typing.TYPE_CHECKING:
    from .processors import Processor


class NodeDimension:
    id: str
    unit: pint.Unit
    quantity: str

    def __init__(self, unit: Union[str, pint.Unit], quantity: str):
        self._unit = unit
        ensure_known_quantity(quantity)
        self.quantity = quantity

    def populate_unit(self, context: Context):
        unit = self._unit
        if isinstance(unit, pint.Unit):
            self.unit = unit
        else:
            self.unit = context.unit_registry(unit).units

    def calculate_hash(self, id: str) -> bytes:
        s = '%s:%s' % (id, self.quantity)
        return s.encode('utf-8') + hash_unit(self.unit)


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
    # order comes from NodeConfig
    order: Optional[int] = None
    # if this node should have its own outcome page
    is_outcome: bool = False

    # output unit (from pint)
    unit: pint.Unit
    # default unit for a node class (defined as a class variable)
    default_unit: ClassVar[str]
    # output quantity (like 'energy' or 'emissions')
    quantity: str

    # optional tags to differentiate between multiple input/output nodes
    tags: Set[str]

    # output units and quantities (for multi-dimensional nodes)
    dimensions: Optional[dict[str, NodeDimension]] = None

    # set if this node has a specific goal for the simulation target year
    target_year_goal: Optional[float]

    input_datasets: List[str]

    input_dataset_instances: List[Dataset]
    input_dataset_processors: List[Processor]

    input_nodes: List[Node]
    output_nodes: List[Node]

    # Global input parameters the node needs
    global_parameters: List[str]

    # Parameters with their values
    parameters: Dict[str, Parameter]

    # All allowed parameters for this class
    allowed_parameters: ClassVar[list[Parameter]]

    # Output for the node in the baseline scenario
    baseline_values: Optional[pd.DataFrame]

    # Cache last historical year
    _last_historical_year: Optional[int]
    context: Context

    logger: logging.Logger
    debug: bool = False

    def __post_init__(self): ...

    def __init__(
        self, id: str, context: Context, name, quantity: str, description,
        color: str | None = None, is_outcome: bool = False, unit=None, target_year_goal=None,
        input_datasets: List[Dataset] | None = None,
    ):
        if self.dimensions:
            for dim in self.dimensions.values():
                dim.populate_unit(context)
        else:
            ensure_known_quantity(quantity)

        if input_datasets is None:
            input_datasets = []

        self.id = id
        self.context = context
        self.database_id = None
        self.name = name
        self.description = description
        self.color = color
        self.is_outcome = is_outcome
        self.unit = unit
        if unit is None and not self.dimensions:
            raise NodeError(self, "Attempting to initialize node without a unit")
        self.quantity = quantity
        self.target_year_goal = target_year_goal
        self.input_dataset_instances = input_datasets
        self.input_nodes = []
        self.output_nodes = []
        self.baseline_values = None
        self.parameters = {}
        self.tags = set()
        self.input_dataset_processors = []

        kls = type(self)
        self.logger = logging.getLogger('%s.%s' % (kls.__module__, kls.__name__))

        if not hasattr(self, 'global_parameters'):
            self.global_parameters = []

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

    @overload
    def get_parameter_value(self, id: str, units: Literal[True]) -> pint.Quantity: ...

    @overload
    def get_parameter_value(self, id: str, required: Literal[True], units: Literal[True]) -> pint.Quantity: ...

    @overload
    def get_parameter_value(self, id: str, required: Literal[False], units: Literal[True]) -> pint.Quantity | None: ...

    @overload
    def get_parameter_value(self, id: str) -> Any: ...

    def get_parameter_value(self, id: str, required: bool = True, units: bool = False) -> Any:
        param = self.get_parameter(id, required=required)
        if param is None:
            return None
        if units:
            if not hasattr(param, 'unit'):
                raise NodeError(self, f"Parameter {id} does not support units")
            return param.value * param.unit
        return param.value

    @overload
    def get_parameter_value_w_unit(self, id: str) -> pint.Quantity: ...

    @overload
    def get_parameter_value_w_unit(self, id: str, required: Literal[False]) -> Optional[pint.Quantity]: ...

    def get_parameter_value_w_unit(self, id: str, required: bool = True) -> Optional[pint.Quantity]:
        return self.get_parameter_value(id, required=required, units=True)

    def get_global_parameter_value(self, id: str, required: bool = True, units: bool = False) -> Any:
        if id not in self.global_parameters:
            raise NodeError(self, f"Attempting to access global parameter {id} which is not declared")
        if units:
            param = self.context.get_parameter(id, required=required)
            if param is None:
                return None
            if not hasattr(param, 'unit'):
                raise NodeError(self, f"Parameter {id} does not support units")
            return param.value * param.unit
        return self.context.get_parameter_value(id, required=required)

    def set_parameter_value(self, local_id: str, value: Any, force: bool = False):
        # if force:  # FIXME if force, create this parameter
        #     self.parameters[local_id] = Parameter()
        if local_id not in self.parameters:
            raise NodeError(self, f"Local parameter {local_id} not found for node {self.id}")
        self.parameters[local_id].set(value)

    def get_input_datasets(self) -> List[Union[pd.DataFrame, pd.Series]]:
        dfs = []
        for ds in self.input_dataset_instances:
            df = ds.get_copy(self.context)
            if df.index.duplicated().any():
                raise NodeError(self, "Input dataset has duplicate index rows")
            assert isinstance(df, pd.DataFrame)
            dfs.append(df)

        if self.input_dataset_processors:
            for proc in self.input_dataset_processors:
                for idx, df in enumerate(list(dfs)):
                    df = proc.process_input_dataset(df)
                    dfs[idx] = df
        return dfs

    @overload
    def get_input_dataset(self, required: Literal[True]) -> pd.DataFrame: ...

    @overload
    def get_input_dataset(self) -> pd.DataFrame: ...

    @overload
    def get_input_dataset(self, required: Literal[False]) -> Optional[pd.DataFrame]: ...

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

    def get_input_node(self, tag: Optional[str] = None) -> Node:
        matching_nodes = []
        for node in self.input_nodes:
            if tag is not None:
                if tag in node.tags:
                    matching_nodes.append(node)
        if len(matching_nodes) != 1:
            tag_str = (' with tag %s' % tag) if tag is not None else ''
            raise NodeError(self, 'Found %d input nodes %s' % (len(matching_nodes), tag_str))
        return matching_nodes[0]

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

    def calculate_hash(self, cache: dict = None) -> bytes:
        h = hashlib.md5()
        debug = self.debug
        if cache is None:
            cache = {}
        if debug:
            print('hashing %s' % self.id)

        def hash_part(part: str, val: str | bytes):
            if isinstance(val, str):
                if debug:
                    print('\t%10s: "%s"' % (part, val))
                val = val.encode('utf8')
            else:
                if debug:
                    print('\t%10s: %s' % (part, base64.b64encode(val)))
            h.update(val)

        if self.unit:
            # __str__() of Unit is very slow, try another way
            hash_part('unit', hash_unit(self.unit))

        if self.dimensions:
            for dim_id, dim in self.dimensions.items():
                hash_part('dim %s' % dim_id, dim.calculate_hash(dim_id))

        for node in self.input_nodes:
            hash_part('input %s' % node.id, node.calculate_hash(cache=cache))
        for param in self.parameters.values():
            hash_part('param %s' % param.local_id, param.calculate_hash())
        for param_id in self.global_parameters:
            param = self.context.get_parameter(param_id, required=False)
            if param is not None:
                hash_part('global param %s' % param.global_id, param.calculate_hash())

        for ds in self.input_dataset_instances:
            hash_part('dataset %s' % ds.id, ds.calculate_hash(self.context))

        for klass in type(self).mro():
            if klass == object:
                continue
            if klass in cache:
                mod_mtime = cache[klass]
            else:
                fn = getattr(klass, '_paths_fname', None)
                if fn is None:
                    fn = inspect.getfile(klass)
                    setattr(klass, '_paths_fname', fn)

                try:
                    mod_mtime = os.path.getmtime(fn)
                except TypeError:
                    continue
                cache[klass] = mod_mtime
            hash_part('mtime %s' % klass.__name__, str(mod_mtime))

        return h.digest()

    def get_output(self, target_node: Node = None, dimension=None) -> pd.DataFrame:
        node_hash = self.calculate_hash().hex()
        out = self.context.cache.get(node_hash)
        if self.debug:
            print('%s: cache %s' % (self.id, 'hit' if out is not None else 'miss'))
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
        if dimension is not None:
            assert dimension in out.columns
            cols = [dimension]  # FIXME This assumes that the column name and dimension are identical
            if FORECAST_COLUMN in out.columns:
                cols.append(FORECAST_COLUMN)
            out = out[cols]
            out = out.rename(columns={dimension: VALUE_COLUMN})
            return out.copy()

        if target_node is not None:
            col_name = None

            if target_node.id in out.columns:
                col_name = target_node.id
            elif self.dimensions:
                assert isinstance(self.dimensions, dict)
                if target_node.quantity in self.dimensions:
                    col_name = target_node.quantity
                else:
                    raise NodeError(self, "Quantity '%s' for node %s not found dimensions" % (target_node.quantity, target_node))

            if col_name:
                cols = [col_name]
                if FORECAST_COLUMN in out.columns:
                    cols.append(FORECAST_COLUMN)
                out = out[cols]
                out = out.rename(columns={col_name: VALUE_COLUMN})

        return out.copy()

    def print_output(self):
        df = self.get_output()
        if self.baseline_values is not None and VALUE_COLUMN in df.columns:
            df['Baseline'] = self.baseline_values[VALUE_COLUMN]
        self.print_pint_df(df)

    def print_pint_df(self, df: Union[pd.DataFrame, pd.Series]):
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)

        pint_cols = [col for col in df.columns if hasattr(df[col], 'pint')]
        if not pint_cols:
            pprint(df)
            return

        out = df[pint_cols].pint.dequantify()
        for col in df.columns:
            if col in pint_cols:
                continue
            out[col] = df[col]
        pprint(out)

    def print_outline(self, df):
        print(type(self))
        print(self.id)
        if YEAR_COLUMN in df.index.names:
            pick_rows = df.index.get_level_values(YEAR_COLUMN).isin([2017, 2021, 2022])
            self.print_pint_df(df.iloc[pick_rows])
        elif len(df) > 6:
            self.print_pint_df(df.iloc[[0, 1, 2, -3, -2, -1]])
        else:
            self.print_pint_df(df)

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

    def strip_units(self, s: pd.Series) -> pd.Series:
        if not hasattr(s, 'pint'):
            return s
        return s.pint.m

    def convert_to_unit(self, s: pd.Series, unit: pint.Unit) -> pd.Series:
        if not s.pint.units.is_compatible_with(unit):
            raise NodeError(self, 'Series with type %s is not compatible with %s' % (
                s.pint.units, unit
            ))
        return s.astype(pint_pandas.PintType(unit))

    def ensure_output_unit(self, s: pd.Series, input_node: Node = None):
        if hasattr(s, 'pint'):
            s_u: pint.Unit = s.pint.u
            if self.unit.dimensionality != s_u.dimensionality:
                if input_node is not None:
                    node_str = ' from node %s' % input_node.id
                else:
                    node_str = ''
                raise NodeError(self, 'Series with type %s%s is not compatible with %s' % (
                    s.pint.units, node_str, self.unit
                ))
            # Units match exactly
            if s_u == self.unit:
                return s
            s_pt = pint_pandas.PintType(s.pint.units)
        else:
            s_pt = None

        s = s.astype(float)
        if s_pt is not None:
            s = s.astype(s_pt)
        node_pt = pint_pandas.PintType(self.unit)
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
        _ = pd.read_json(sio, orient='table')
        return old

    def replace_input_data(self, data: dict):
        if len(self.input_dataset_instances) != 1:
            raise NodeError(
                self, "Can't replace data for node with %d input datasets" % len(self.input_dataset_instances))

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

    def as_node_config_attributes(self):
        """Return a dict that can be used to set corresponding fields of NodeConfig."""
        attributes = {
            'identifier': self.id,
        }

        i18n = {}
        default_lang = self.context.instance.default_language

        def set_from_translated_str(s: str | TranslatedString | None, field_name: str):
            if s is None:
                return

            val, tr = get_modeltrans_attrs_from_str(s, field_name, default_lang)
            i18n.update(tr)

        set_from_translated_str(self.name, 'name')
        # Node's description is called `short_description` in NodeConfig and there is no equivalent for
        # NodeConfig's `long_description` in Node
        set_from_translated_str(self.description, 'short_description')

        if self.color:
            attributes['color'] = self.color

        return attributes
